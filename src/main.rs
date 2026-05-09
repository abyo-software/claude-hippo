use claude_hippo::cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::run().await
}
