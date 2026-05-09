//! CLI — clap derive。serve / verify / embed / bench。

use crate::embeddings::{Embedder, FastEmbedder};
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
    },
    /// Run a quick self-bench: cold start + N store + N retrieve + RSS.
    Bench {
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, env = "HIPPO_MODEL_CACHE")]
        model_cache: Option<PathBuf>,
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

fn build_embedder(model_cache: Option<PathBuf>) -> anyhow::Result<Arc<dyn Embedder>> {
    let cache = model_cache.unwrap_or_else(crate::embeddings::default_cache_dir);
    let e = FastEmbedder::new(cache)?;
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
    });

    storage::register_sqlite_vec();

    match cmd {
        Cmd::Serve { db, model_cache } => {
            let path = db.unwrap_or_else(default_db_path);
            ensure_parent_dir(&path)?;
            let store = storage::Storage::open(&path)?;
            let embedder = build_embedder(model_cache)?;
            tracing::info!(?path, "claude-hippo serve starting (rmcp stdio)");
            server::run_stdio(store, embedder).await
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
        Cmd::Embed { text, model_cache } => {
            let embedder = build_embedder(model_cache)?;
            let t0 = std::time::Instant::now();
            let v = embedder.embed_one(&text)?;
            let dt = t0.elapsed();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("hippo embed ✓");
            println!("  text     : {text:?}");
            println!("  total    : {dt:?}");
            println!("  dim      : {}", v.len());
            println!("  L2 norm  : {norm:.6}");
            println!("  first 5  : {:?}", &v[..5.min(v.len())]);
            Ok(())
        }
        Cmd::Bench { n, db, model_cache } => run_self_bench(n, db, model_cache).await,
    }
}

async fn run_self_bench(
    n: usize,
    db: Option<PathBuf>,
    model_cache: Option<PathBuf>,
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
    let embedder = build_embedder(model_cache)?;
    // first embed = model load
    let _ = embedder.embed_one("warmup")?;
    let cold = cold0.elapsed();

    let server = server::MemoryServer::new(store, embedder);

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
