#!/usr/bin/env python3
"""Benchmark mcp-memory-service-rs and claude-hippo over the same stdio MCP
protocol, sequential workload (cold-start + N store + N retrieve + peak RSS).

Both servers expose SHODH-compatible tool names: store_memory, retrieve_memory.
We use those names so the same harness works for both.

Usage:
    python3 scripts/bench_competitor.py [--n 100]
        [--mms ~/git/mcp-memory-service-rs/target/release/mcp-memory-service-rs]
        [--hippo ~/git/hippo/target/release/hippo]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

HOME = Path.home()
DEFAULT_MMS = HOME / "git" / "mcp-memory-service-rs" / "target" / "release" / "mcp-memory-service-rs"
DEFAULT_HIPPO = HOME / "git" / "hippo" / "target" / "release" / "hippo"


def _frame(msg): return (json.dumps(msg) + "\n").encode()


def _init_msgs():
    return [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "bench", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]


class Server:
    def __init__(self, label, argv, env):
        self.label = label
        self.argv = argv
        self.env = env
        self.proc = None

    def start(self):
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=self.env)
        for m in _init_msgs():
            self.proc.stdin.write(_frame(m))
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} died during handshake")
            r = json.loads(line)
            if r.get("id") == 1:
                break
        return time.perf_counter() - t0

    def call(self, name, args, id_):
        t0 = time.perf_counter()
        req = {"jsonrpc": "2.0", "id": id_, "method": "tools/call",
               "params": {"name": name, "arguments": args}}
        self.proc.stdin.write(_frame(req))
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} died during {name}")
            r = json.loads(line)
            if r.get("id") == id_:
                if "error" in r:
                    raise RuntimeError(f"{self.label} {name} error: {r['error']}")
                return r["result"], time.perf_counter() - t0

    def peak_rss_mb(self):
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(self.proc.pid)], text=True).strip()
        return int(out) / 1024.0

    def close(self):
        if self.proc and self.proc.poll() is None:
            try: self.proc.stdin.close()
            except Exception: pass
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.proc.kill()


def percentile(xs, p):
    s = sorted(xs); k = (len(s) - 1) * p
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def bench(label, argv, env, n_stores, n_retrieves):
    print(f"\n=== {label} ===")
    print(f"  binary: {argv[0]}")
    s = Server(label, argv, env)
    cold = s.start() * 1000
    print(f"  cold-start      : {cold:7.1f} ms")

    store_lats = []
    for i in range(n_stores):
        _, dt = s.call(
            "store_memory",
            {"content": f"bench memory {i}: timing harness",
             "tags": ["bench", f"i{i % 10}"]},
            100 + i,
        )
        store_lats.append(dt * 1000)

    retr_lats = []
    for i in range(n_retrieves):
        _, dt = s.call(
            "retrieve_memory",
            {"query": "timing harness memory", "n_results": 5},
            10000 + i,
        )
        retr_lats.append(dt * 1000)

    rss = s.peak_rss_mb()
    s.close()

    return {
        "cold_ms": cold,
        "store_p50": percentile(store_lats, 0.5),
        "store_p95": percentile(store_lats, 0.95),
        "retrieve_p50": percentile(retr_lats, 0.5),
        "retrieve_p95": percentile(retr_lats, 0.95),
        "rss_mb": rss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mms", type=Path, default=DEFAULT_MMS)
    ap.add_argument("--hippo", type=Path, default=DEFAULT_HIPPO)
    ap.add_argument("--skip-mms", action="store_true")
    ap.add_argument("--skip-hippo", action="store_true")
    args = ap.parse_args()

    results = {}

    with tempfile.TemporaryDirectory(prefix="hippo-bench-") as d:
        if not args.skip_mms:
            if not args.mms.exists():
                print(f"warning: mms binary not found: {args.mms}", file=sys.stderr)
            else:
                env = os.environ.copy()
                env["MCP_MEMORY_DB_PATH"] = str(Path(d) / "mms.db")
                results["mcp-memory-service-rs"] = bench(
                    "mcp-memory-service-rs (Rust)",
                    [str(args.mms), "serve"], env, args.n, args.n,
                )

        if not args.skip_hippo:
            if not args.hippo.exists():
                print(f"warning: hippo binary not found: {args.hippo}", file=sys.stderr)
            else:
                env = os.environ.copy()
                env["HIPPO_DB_PATH"] = str(Path(d) / "hippo.db")
                results["claude-hippo"] = bench(
                    "claude-hippo (Rust + surprise)",
                    [str(args.hippo), "serve"], env, args.n, args.n,
                )

    if not results:
        print("no results")
        return 1

    print()
    print("| Metric            | " + " | ".join(f"{k:30}" for k in results) + " |")
    print("|-------------------|" + "|".join("-" * 32 for _ in results) + "|")
    rows = [
        ("cold-start (ms)",   "cold_ms"),
        ("store p50 (ms)",    "store_p50"),
        ("store p95 (ms)",    "store_p95"),
        ("retrieve p50 (ms)", "retrieve_p50"),
        ("retrieve p95 (ms)", "retrieve_p95"),
        ("RSS (MB)",          "rss_mb"),
    ]
    for label, key in rows:
        cells = " | ".join(f"{results[k][key]:30.1f}" for k in results)
        print(f"| {label:<17} | {cells} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
