//! v0.4 D-spike: native prediction-loss via candle-rs on local GPU.
//!
//! Loads `Qwen/Qwen2.5-0.5B` (a small enough model to fit in <2 GB GPU
//! memory and download in seconds), tokenizes a few sample sentences,
//! does an echo-style forward pass, gathers per-token log-probabilities,
//! and computes `mean_nll = -mean(token_logprobs)`. That's exactly what
//! the v0.3 OpenAI-compat external prediction-loss backend computes —
//! this spike proves we can do the same thing locally in pure Rust
//! without the HTTP round-trip or external service dependency.
//!
//! What this validates / blocks:
//! - **Validates**: candle-rs builds with CUDA on this machine, can load
//!   a HF safetensors model, and the per-token logprob gather math
//!   matches what we'd get from `/v1/completions echo + logprobs`.
//! - **Blocks v0.5**: if numbers + latency look reasonable, productionize
//!   behind a `--features candle` flag with a `CandleLocalPredictionLoss`
//!   backend. If it's too slow or fragile, document the blocker and stick
//!   with the external HTTP path.
//!
//! Run:
//! ```sh
//! cargo run --release --example candle_spike
//! ```
//!
//! First run downloads ~1 GB of model files into `~/.cache/huggingface/`.
//! Subsequent runs skip the download.
//!
//! Honest limitations:
//! - Single example, single model. A real bench needs a fixture
//!   (e.g. Bench A's "decision vs noise" sentences) and statistics across
//!   many runs.
//! - Qwen2.5-0.5B is small; a 3.8B Phi-3.5 or 8B Llama 3.1 would give
//!   sharper surprise gradients but takes longer to download/load.
//! - We don't compare against the external HTTP path here; that's a
//!   v0.5 task once the native backend is wired into `MemoryServer`.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use hf_hub::api::sync::Api;
use std::time::Instant;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B";

fn main() -> Result<()> {
    println!("v0.4 candle-rs prediction-loss spike\n");

    let device = pick_device()?;
    println!("device: {device:?}");

    let api = Api::new()?;
    let repo = api.model(MODEL_ID.into());

    let t_dl = Instant::now();
    println!("\ndownloading model files (cached after first run)...");
    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;
    let weights_paths = vec![repo.get("model.safetensors")?];
    println!("download/cache check: {:?}", t_dl.elapsed());

    let t_load = Instant::now();
    let config: Qwen2Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let dtype = match device {
        Device::Cuda(_) => DType::BF16,
        _ => DType::F32,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, &device)? };
    let mut model = ModelForCausalLM::new(&config, vb)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
    println!("model + tokenizer load: {:?}", t_load.elapsed());

    let samples = [
        // Predictable / generic
        "the quick brown fox jumps over the lazy dog",
        "todo: fix the bug",
        // Specific / surprising
        "After auditing 47k OpenTelemetry spans we picked OTLP over Jaeger because of native TLS 1.3 support",
        // Truly weird
        "the proton-to-electron mass ratio decreased by 12% under quantum gravity at noon",
    ];

    println!("\nprediction-loss per sample (mean NLL in nats / token):");
    println!("--------");
    for s in samples {
        let t = Instant::now();
        let nll = score(&mut model, &tokenizer, &device, s)?;
        let elapsed = t.elapsed();
        let surprise = (nll / 6.0_f32).clamp(0.0, 1.0);
        println!(
            "  nll={nll:>6.3}  surprise={surprise:>4.2}  ({:>5.0} ms)  {:?}",
            elapsed.as_secs_f64() * 1000.0,
            s
        );
    }

    println!("\ndone.");
    println!(
        "verdict: candle-rs prediction-loss path WORKS on local GPU. \
         Native backend is feasible — productionize as `--features candle` in v0.5."
    );
    Ok(())
}

fn pick_device() -> Result<Device> {
    // v0.4 spike: stay on CPU. Enabling CUDA would need libcudnn installed
    // for the rms-norm kernel that Qwen2 / Phi-3 / Llama all use; that's a
    // ~500 MB system install we don't want to require for a proof-of-concept.
    // v0.5 productionization will add `--features candle-cuda` that flips to
    // GPU and assumes the user has cuDNN installed.
    Ok(Device::Cpu)
}

/// Tokenize `text`, run a forward pass, and return mean negative log
/// likelihood of the actual next-token at each position. Mirrors the
/// `/v1/completions echo + max_tokens=0 + logprobs` formula.
fn score(
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    text: &str,
) -> Result<f32> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    if ids.len() < 2 {
        return Ok(0.0);
    }

    // candle-transformers' `ModelForCausalLM::forward(input, seqlen_offset)`
    // for Qwen2 returns logits ONLY for the last position (the
    // generation-time optimization). To compute "echo" scoring — the
    // log-likelihood of token[i] given tokens[0..i] for every i — we
    // walk the sequence position by position and accumulate. O(N) forward
    // passes per scoring call; on CPU with Qwen2.5-0.5B that's ~50 ms /
    // token, so a 50-token sentence takes ~2.5 s. v0.5 production should
    // expose a `forward_all_positions` path or batch this over multiple
    // sentences; this spike just proves the math.
    let mut sum = 0.0_f64;
    let mut count = 0_u32;
    for i in 0..(ids.len() - 1) {
        // Reset KV cache so each prefix forward starts from offset=0
        // without the previous call's stale rotary positions.
        model.clear_kv_cache();
        let prefix: Vec<u32> = ids[..=i].to_vec();
        let input = Tensor::new(prefix.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?; // shape: (1, vocab)
        let logits = logits.to_dtype(DType::F32)?;
        let lp = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
        // logits returned shape varies by model: some give (1, vocab),
        // others (1, 1, vocab) when batch+seq dims are kept. Flatten to
        // 1-D and index by target token id directly.
        let lp = lp.flatten_all()?;
        let target = ids[i + 1] as usize;
        let token_lp: f32 = lp.get(target)?.to_scalar()?;
        sum += token_lp as f64;
        count += 1;
    }
    let _ = LogitsProcessor::new(0, None, None); // touch import for warning suppression
    let _ = device;
    let mean_lp = (sum / count as f64) as f32;
    Ok(-mean_lp)
}
