//! CLI — clap derive。serve / verify / embed / bench。

use crate::embeddings::{Embedder, EmbeddingModelKind, FastEmbedder};
use crate::server::{
    RankingConfig, DEFAULT_DECAY_FLOOR, DEFAULT_HALF_LIFE_DAYS, DEFAULT_OVERSAMPLE_FACTOR,
};
use crate::surprise::SurpriseWeights;
use crate::{server, storage, VERSION};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "hippo", version = VERSION,
          about = "Claude Code に海馬を足す MCP server (claude-hippo)",
          long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the MCP server over stdio (default when no subcommand is given).
    Serve {
        /// SQLite DB path. Defaults to $HIPPO_DB_PATH or
        /// ~/.local/share/claude-hippo/memory.db.
        #[arg(long, env = "HIPPO_DB_PATH")]
        db: Option<PathBuf>,
        /// Embedding model cache directory. Defaults to $HIPPO_MODEL_CACHE
        /// or ~/.cache/claude-hippo/models/.
        #[arg(long, env = "HIPPO_MODEL_CACHE")]
        model_cache: Option<PathBuf>,
        /// Surprise score weights as `w_outlier,w_engagement,w_explicit,w_prediction`.
        /// All in 0.0..=1.0, sum must be 1.0 (±1e-3). Default: `0.4,0.2,0.1,0.3`.
        #[arg(long, env = "HIPPO_SURPRISE_WEIGHTS")]
        surprise_weights: Option<String>,
        /// Embedding model. `minilm-l6-v2` (default, mcp-memory-service-rs と
        /// 同 vector space) or `bge-small-en-v15-q` (量子化、~33 MB)。
        /// Both are 384 dim.
        #[arg(long, env = "HIPPO_EMBEDDING_MODEL")]
        embedding_model: Option<String>,
        /// Forgetting-curve half-life in days. Default 30. Lower = faster
        /// decay of old surprise. Set to 0 to disable decay.
        #[arg(long, env = "HIPPO_HALF_LIFE_DAYS")]
        half_life_days: Option<f32>,
        /// Decay floor in 0.0..=1.0. Caps how much the forgetting curve can
        /// shrink an old item's surprise contribution. Default 0.5 keeps
        /// high-importance items competitive at any age. Set to 0 to
        /// reproduce v0.2 behavior (old high-surprise items can be demoted
        /// by fresh low-surprise items past ~12 half-lives).
        #[arg(long, env = "HIPPO_DECAY_FLOOR")]
        decay_floor: Option<f32>,
        /// Server-wide default for KNN over-fetch multiplier before surprise
        /// rerank. Default 6 (was 3 in v0.2). Larger values widen the
        /// rerank candidate pool at the cost of more SQL work. Per-call
        /// override is also exposed via `RecallParams.oversample_factor`.
        #[arg(long, env = "HIPPO_OVERSAMPLE_FACTOR")]
        oversample_factor: Option<usize>,
    },
    /// Open the database, apply schema, verify sqlite-vec, print stats.
    /// Does not read/write any memories. Safe against a live DB.
    Verify {
        #[arg(long, env = "HIPPO_DB_PATH")]
        db: Option<PathBuf>,
    },
    /// Load the embedding model and embed a single string.
    /// Smoke-tests the full pipeline (download + tokenize + inference + pool).
    Embed {
        text: String,
        #[arg(long, env = "HIPPO_MODEL_CACHE")]
        model_cache: Option<PathBuf>,
        #[arg(long, env = "HIPPO_EMBEDDING_MODEL")]
        embedding_model: Option<String>,
    },
    /// Run a quick self-bench: cold start + N store + N retrieve + RSS.
    Bench {
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, env = "HIPPO_MODEL_CACHE")]
        model_cache: Option<PathBuf>,
        #[arg(long, env = "HIPPO_SURPRISE_WEIGHTS")]
        surprise_weights: Option<String>,
        #[arg(long, env = "HIPPO_EMBEDDING_MODEL")]
        embedding_model: Option<String>,
        #[arg(long, env = "HIPPO_HALF_LIFE_DAYS")]
        half_life_days: Option<f32>,
        #[arg(long, env = "HIPPO_DECAY_FLOOR")]
        decay_floor: Option<f32>,
        #[arg(long, env = "HIPPO_OVERSAMPLE_FACTOR")]
        oversample_factor: Option<usize>,
    },
}

fn default_db_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("claude-hippo")
        .join("memory.db")
}

fn ensure_parent_dir(p: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn parse_model_kind(opt: Option<&str>) -> anyhow::Result<EmbeddingModelKind> {
    match opt {
        None => Ok(EmbeddingModelKind::default()),
        Some(s) => EmbeddingModelKind::parse(s).map_err(|e| anyhow::anyhow!(e)),
    }
}

fn parse_weights(opt: Option<&str>) -> anyhow::Result<SurpriseWeights> {
    match opt {
        None => Ok(SurpriseWeights::default()),
        Some(s) => SurpriseWeights::parse_csv(s).map_err(|e| anyhow::anyhow!(e)),
    }
}

fn build_ranking_config(
    half_life_days: Option<f32>,
    decay_floor: Option<f32>,
    oversample_factor: Option<usize>,
) -> anyhow::Result<RankingConfig> {
    let hl = half_life_days.unwrap_or(DEFAULT_HALF_LIFE_DAYS);
    if hl < 0.0 {
        anyhow::bail!("--half-life-days must be ≥ 0 (0 disables decay), got {hl}");
    }
    let floor = decay_floor.unwrap_or(DEFAULT_DECAY_FLOOR);
    if !(0.0..=1.0).contains(&floor) {
        anyhow::bail!("--decay-floor must be in 0.0..=1.0, got {floor}");
    }
    let factor = oversample_factor.unwrap_or(DEFAULT_OVERSAMPLE_FACTOR);
    if factor == 0 {
        anyhow::bail!("--oversample-factor must be ≥ 1, got 0");
    }
    Ok(RankingConfig {
        half_life_days: hl,
        decay_floor: floor,
        default_oversample_factor: factor,
    })
}

fn build_embedder(
    model_cache: Option<PathBuf>,
    model: EmbeddingModelKind,
) -> anyhow::Result<Arc<dyn Embedder>> {
    let cache = model_cache.unwrap_or_else(crate::embeddings::default_cache_dir);
    let e = FastEmbedder::new_with_model(cache, model)?;
    Ok(Arc::new(e))
}

pub async fn run() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let cli = Cli::parse();
    let cmd = cli.command.unwrap_or(Cmd::Serve {
        db: None,
        model_cache: None,
        surprise_weights: None,
        embedding_model: None,
        half_life_days: None,
        decay_floor: None,
        oversample_factor: None,
    });

    storage::register_sqlite_vec();

    match cmd {
        Cmd::Serve {
            db,
            model_cache,
            surprise_weights,
            embedding_model,
            half_life_days,
            decay_floor,
            oversample_factor,
        } => {
            let path = db.unwrap_or_else(default_db_path);
            ensure_parent_dir(&path)?;
            let weights = parse_weights(surprise_weights.as_deref())?;
            let ranking = build_ranking_config(half_life_days, decay_floor, oversample_factor)?;
            let model = parse_model_kind(embedding_model.as_deref())?;
            let store = storage::Storage::open(&path)?;
            let embedder = build_embedder(model_cache, model)?;
            tracing::info!(
                ?path,
                model = model.as_str(),
                ?weights,
                ?ranking,
                "claude-hippo serve starting (rmcp stdio)"
            );
            server::run_stdio_with_config(store, embedder, weights, ranking).await
        }
        Cmd::Verify { db } => {
            let path = db.unwrap_or_else(default_db_path);
            ensure_parent_dir(&path)?;
            let store = storage::Storage::open(&path)?;
            let alive = store.count_alive()?;
            let total = store.count_total()?;
            let vec_v = store.vec_version()?;
            println!("hippo verify ✓");
            println!("  db path     : {}", path.display());
            println!("  vec_version : {vec_v}");
            println!("  alive       : {alive}");
            println!("  total       : {total} (incl. soft-deleted)");
            Ok(())
        }
        Cmd::Embed {
            text,
            model_cache,
            embedding_model,
        } => {
            let model = parse_model_kind(embedding_model.as_deref())?;
            let embedder = build_embedder(model_cache, model)?;
            let t0 = std::time::Instant::now();
            let v = embedder.embed_one(&text)?;
            let dt = t0.elapsed();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("hippo embed ✓");
            println!("  text     : {text:?}");
            println!("  model    : {}", model.as_str());
            println!("  total    : {dt:?}");
            println!("  dim      : {}", v.len());
            println!("  L2 norm  : {norm:.6}");
            println!("  first 5  : {:?}", &v[..5.min(v.len())]);
            Ok(())
        }
        Cmd::Bench {
            n,
            db,
            model_cache,
            surprise_weights,
            embedding_model,
            half_life_days,
            decay_floor,
            oversample_factor,
        } => {
            let weights = parse_weights(surprise_weights.as_deref())?;
            let ranking = build_ranking_config(half_life_days, decay_floor, oversample_factor)?;
            let model = parse_model_kind(embedding_model.as_deref())?;
            run_self_bench(n, db, model_cache, weights, ranking, model).await
        }
    }
}

async fn run_self_bench(
    n: usize,
    db: Option<PathBuf>,
    model_cache: Option<PathBuf>,
    weights: SurpriseWeights,
    ranking: RankingConfig,
    model: EmbeddingModelKind,
) -> anyhow::Result<()> {
    use std::time::Instant;
    let db_path = db.unwrap_or_else(|| {
        let mut p = std::env::temp_dir();
        p.push(format!("claude-hippo-bench-{}.db", std::process::id()));
        p
    });
    ensure_parent_dir(&db_path)?;
    // クリーンスタート
    let _ = std::fs::remove_file(&db_path);

    let cold0 = Instant::now();
    let store = storage::Storage::open(&db_path)?;
    let embedder = build_embedder(model_cache, model)?;
    // first embed = model load
    let _ = embedder.embed_one("warmup")?;
    let cold = cold0.elapsed();

    let server = server::MemoryServer::new_with_config(store, embedder, weights, ranking);

    // store N
    let t1 = Instant::now();
    let mut store_lats = Vec::with_capacity(n);
    for i in 0..n {
        let st = Instant::now();
        let _ = server
            .do_remember(server::RememberParams {
                content: format!("bench memory {i}: timing harness"),
                tags: vec!["bench".into(), format!("i{}", i % 10)],
                memory_type: Some("Observation".into()),
                importance: Some(0.5),
                metadata: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("store err: {:?}", e))?;
        store_lats.push(st.elapsed().as_secs_f64() * 1000.0);
    }
    let store_total = t1.elapsed();

    // retrieve N
    let t2 = Instant::now();
    let mut retrieve_lats = Vec::with_capacity(n);
    for _ in 0..n {
        let st = Instant::now();
        let _ = server
            .do_recall(server::RecallParams {
                query: "timing harness memory".into(),
                limit: 5,
                no_surprise_boost: false,
                oversample_factor: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("retrieve err: {:?}", e))?;
        retrieve_lats.push(st.elapsed().as_secs_f64() * 1000.0);
    }
    let retrieve_total = t2.elapsed();

    fn pct(xs: &mut [f64], p: f64) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ((xs.len() - 1) as f64) * p;
        let f = k.floor() as usize;
        let c = (f + 1).min(xs.len() - 1);
        if f == c {
            xs[f]
        } else {
            xs[f] + (xs[c] - xs[f]) * (k - f as f64)
        }
    }

    let rss_kb = read_self_rss_kb().unwrap_or(0);

    println!("claude-hippo self-bench ✓");
    println!("  model    : {}", model.as_str());
    println!(
        "  weights  : outlier={:.2} engagement={:.2} explicit={:.2} prediction={:.2}",
        server.weights().w_outlier,
        server.weights().w_engagement,
        server.weights().w_explicit,
        server.weights().w_prediction,
    );
    let rc = server.ranking_config();
    println!(
        "  ranking  : half_life_days={:.1} decay_floor={:.2} oversample_factor={}",
        rc.half_life_days, rc.decay_floor, rc.default_oversample_factor,
    );
    println!("  cold-start (db open + embed warmup) : {cold:?}");
    println!(
        "  store    x{n}: total={store_total:?}  p50={:.1}ms p95={:.1}ms",
        pct(&mut store_lats.clone(), 0.5),
        pct(&mut store_lats.clone(), 0.95),
    );
    println!(
        "  retrieve x{n}: total={retrieve_total:?}  p50={:.1}ms p95={:.1}ms",
        pct(&mut retrieve_lats.clone(), 0.5),
        pct(&mut retrieve_lats.clone(), 0.95),
    );
    println!("  peak RSS (self) : {:.1} MB", rss_kb as f64 / 1024.0);
    Ok(())
}

fn read_self_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            // "VmHWM:    12345 kB"
            return rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok());
        }
    }
    None
}
