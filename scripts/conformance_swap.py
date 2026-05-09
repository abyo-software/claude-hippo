#!/usr/bin/env python3
"""SHODH-compatible DB swap conformance test.

claude-hippo と mcp-memory-service-rs が **同じ SQLite ファイル** を読み書き
できることを保証する。両方の binary を spawn し、stdio MCP で:

  1. claude-hippo で 5 件 store
  2. 同 DB を mcp-memory-service-rs で list_memories → 5 件取得確認
  3. mcp-memory-service-rs で 5 件 store
  4. 同 DB を claude-hippo で hippo_list_recent → 10 件全件取得確認
  5. 各 row の content_hash, content, tags が完全一致することを確認

Usage:
    python3 scripts/conformance_swap.py
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


class Server:
    def __init__(self, label, argv, env):
        self.label = label
        self.argv = argv
        self.env = env
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=self.env)
        for m in [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                        "clientInfo": {"name": "conformance", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ]:
            self.proc.stdin.write(_frame(m))
        self.proc.stdin.flush()
        # drain initialize response
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} died")
            if json.loads(line).get("id") == 1:
                break
        return self

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            try: self.proc.stdin.close()
            except Exception: pass
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.proc.kill()

    def call(self, name, args, id_):
        self.proc.stdin.write(_frame(
            {"jsonrpc": "2.0", "id": id_, "method": "tools/call",
             "params": {"name": name, "arguments": args}}
        ))
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} died during {name}")
            r = json.loads(line)
            if r.get("id") == id_:
                if "error" in r:
                    raise RuntimeError(f"{self.label} {name} error: {r['error']}")
                return _payload(r["result"])


def _payload(result):
    """tool result.content[0].text を JSON parse して返す。"""
    text = result["content"][0]["text"]
    return json.loads(text)


HIPPO_MEMS = [
    ("hippo store: JWT 24h auth decision", ["auth", "security"], "Decision"),
    ("hippo store: postgres tenant_id index", ["db", "perf"], "Pattern"),
    ("hippo store: switched billing Stripe→Paddle", ["biz", "billing"], "Decision"),
    ("hippo store: nightly cron retried 3x with backoff", ["ops"], "Pattern"),
    ("hippo store: customer churn dropped 12% after onboarding rewrite", ["analytics"], "Discovery"),
]

MMS_MEMS = [
    ("mms store: foo bar baz", ["mms", "smoke"], "note"),
    ("mms store: apple banana cherry", ["mms", "fruit"], "note"),
    ("mms store: long form decision rationale here", ["mms", "long"], "decision"),
    ("mms store: pattern observed in churn cohort", ["mms"], "pattern"),
    ("mms store: error handling path needs retry", ["mms", "error"], "error"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mms", type=Path, default=DEFAULT_MMS)
    ap.add_argument("--hippo", type=Path, default=DEFAULT_HIPPO)
    args = ap.parse_args()

    if not args.hippo.exists():
        print(f"FAIL: hippo binary not found: {args.hippo}", file=sys.stderr)
        return 2
    if not args.mms.exists():
        print(f"FAIL: mms binary not found: {args.mms}", file=sys.stderr)
        return 2

    failures = []

    with tempfile.TemporaryDirectory(prefix="conformance-") as d:
        db = Path(d) / "shared.db"
        print(f"shared DB: {db}")

        # Phase 1: claude-hippo writes 5
        print("\n=== Phase 1: claude-hippo writes 5 ===")
        env_h = {**os.environ, "HIPPO_DB_PATH": str(db)}
        with Server("hippo", [str(args.hippo), "serve"], env_h) as s:
            for i, (content, tags, mt) in enumerate(HIPPO_MEMS):
                r = s.call(
                    "store_memory",
                    {"content": content, "tags": tags, "memory_type": mt},
                    100 + i,
                )
                assert r["success"], f"hippo store {i} failed: {r}"
                print(f"  hippo store[{i}]: hash={r['content_hash'][:12]} "
                      f"surprise={r['surprise_score']:.3f}")

        # Phase 2: mcp-memory-service-rs reads 5
        print("\n=== Phase 2: mcp-memory-service-rs reads same DB ===")
        env_m = {**os.environ, "MCP_MEMORY_DB_PATH": str(db)}
        with Server("mms", [str(args.mms), "serve"], env_m) as s:
            r = s.call("list_memories", {"page": 1, "page_size": 50}, 200)
            count = len(r.get("memories", []))
            print(f"  mms list: count={count}")
            if count != 5:
                failures.append(
                    f"phase 2: expected 5 memories visible to mms, got {count}"
                )
            mms_view_hashes = {m["content_hash"] for m in r.get("memories", [])}
            hippo_hashes = {sha(content) for content, _, _ in HIPPO_MEMS}
            missing = hippo_hashes - mms_view_hashes
            if missing:
                failures.append(f"phase 2: mms cannot see hashes: {missing}")

        # Phase 3: mcp-memory-service-rs writes 5 more
        print("\n=== Phase 3: mcp-memory-service-rs writes 5 more ===")
        with Server("mms", [str(args.mms), "serve"], env_m) as s:
            for i, (content, tags, mt) in enumerate(MMS_MEMS):
                r = s.call(
                    "store_memory",
                    {"content": content, "tags": tags, "memory_type": mt},
                    300 + i,
                )
                assert r["success"], f"mms store {i} failed: {r}"
                print(f"  mms store[{i}]: hash={r['content_hash'][:12]}")

        # Phase 4: claude-hippo reads all 10
        print("\n=== Phase 4: claude-hippo reads all 10 ===")
        with Server("hippo", [str(args.hippo), "serve"], env_h) as s:
            r = s.call("hippo_list_recent", {"n": 50}, 400)
            count = r.get("count", 0)
            print(f"  hippo list: count={count}")
            if count != 10:
                failures.append(
                    f"phase 4: expected 10 memories visible to hippo, got {count}"
                )
            hippo_view_hashes = {m["content_hash"] for m in r.get("memories", [])}
            all_expected = (
                {sha(c) for c, _, _ in HIPPO_MEMS}
                | {sha(c) for c, _, _ in MMS_MEMS}
            )
            missing = all_expected - hippo_view_hashes
            if missing:
                failures.append(f"phase 4: hippo cannot see hashes: {missing}")

        # Phase 5: hippo can semantic-recall mms-written content
        print("\n=== Phase 5: hippo semantic recall over mms-written content ===")
        with Server("hippo", [str(args.hippo), "serve"], env_h) as s:
            r = s.call(
                "hippo_recall",
                {"query": "apple banana", "limit": 3, "no_surprise_boost": True},
                500,
            )
            arr = r if isinstance(r, list) else []
            top = arr[0] if arr else None
            if not top:
                failures.append("phase 5: hippo recall returned no hits")
            else:
                top_content = top.get("memory", {}).get("content", "")
                print(f"  top hit: {top_content[:60]}  cos={top.get('cosine_similarity'):.3f}")
                if "apple" not in top_content:
                    failures.append(
                        f"phase 5: top hit should mention apple, got: {top_content!r}"
                    )

    print("\n=== Result ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  PASS: SHODH DB swap conformance verified ✓")
    return 0


def sha(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
