#!/usr/bin/env python3
"""Reproduce mcp-memory-service-rs's cold-start + warm latency + RSS without
needing the Python upstream. Adapted from scripts/bench.py."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path("/home/y1/git/mcp-memory-service-rs")
RUST_BIN = REPO / "target" / "release" / "mcp-memory-service-rs"
NUM_STORES = 100
NUM_RETRIEVES = 100


def _frame(msg): return (json.dumps(msg) + "\n").encode()


def _init_msgs():
    return [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "bench", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]


class Server:
    def __init__(self, argv, env):
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
                raise RuntimeError("died during handshake")
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
                raise RuntimeError(f"died during call {name}")
            r = json.loads(line)
            if r.get("id") == id_:
                dt = time.perf_counter() - t0
                if "error" in r:
                    raise RuntimeError(f"{name} error: {r['error']}")
                return r["result"], dt

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
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main():
    with tempfile.TemporaryDirectory(prefix="rust-bench-") as d:
        db = Path(d) / "rust.db"
        env = os.environ.copy()
        env["MCP_MEMORY_DB_PATH"] = str(db)

        s = Server([str(RUST_BIN), "serve"], env)
        cold = s.start() * 1000
        print(f"cold-start      : {cold:.1f} ms")

        store_lats = []
        for i in range(NUM_STORES):
            _, dt = s.call("store_memory",
                           {"content": f"bench memory {i}: timing harness",
                            "tags": ["bench", f"i{i % 10}"]},
                           100 + i)
            store_lats.append(dt * 1000)

        retr_lats = []
        for i in range(NUM_RETRIEVES):
            _, dt = s.call("retrieve_memory",
                           {"query": "timing harness memory", "n_results": 5},
                           10000 + i)
            retr_lats.append(dt * 1000)

        rss = s.peak_rss_mb()
        s.close()

        def fmt(xs):
            return (f"p50={percentile(xs, 0.5):.1f}ms "
                    f"p95={percentile(xs, 0.95):.1f}ms "
                    f"min={min(xs):.1f}ms max={max(xs):.1f}ms")

        print(f"store    x{NUM_STORES}  : {fmt(store_lats)}")
        print(f"retrieve x{NUM_RETRIEVES}  : {fmt(retr_lats)}")
        print(f"peak RSS        : {rss:.1f} MB")


if __name__ == "__main__":
    main()
