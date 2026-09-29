"""Generate the synthetic `shopfront` CI history used by the demo.

Deterministic: the same seed always produces the same runs, so results are reproducible.
Run: python scripts/build_dataset.py
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "shopfront"
START = datetime(2026, 9, 7, tzinfo=timezone.utc)
PYTHON = "/opt/hostedtoolcache/Python/3.11.9/x64/bin/python"

# (day offset from 2026-09-07, "HH:MM", pattern)
SCHEDULE = [
    (0, "09:14", "flaky_checkout"), (0, "15:40", "stripe"), (1, "10:05", "disk"), (1, "16:22", "flaky_checkout"),
    (2, "11:30", "flaky_inventory"), (2, "17:10", "npm"), (3, "09:45", "flaky_checkout"), (3, "14:00", "refund"),
    (4, "10:20", "pydantic"), (4, "13:35", "pydantic"), (4, "16:50", "pydantic"), (5, "09:05", "disk"),
    (5, "12:15", "flaky_inventory"), (5, "18:40", "flaky_checkout"), (6, "11:00", "npm"), (6, "15:25", "stripe"),
    (7, "09:30", "discount"), (7, "11:45", "discount"), (7, "16:05", "flaky_checkout"), (8, "10:10", "discount"),
    (8, "14:30", "disk"), (9, "09:50", "flaky_inventory"), (9, "13:15", "timeout"), (10, "10:40", "flaky_checkout"),
    (10, "17:20", "fuzzy"), (11, "09:25", "fuzzy"), (11, "15:55", "stripe"), (12, "11:05", "flaky_checkout"),
    (12, "16:45", "flaky_inventory"), (13, "10:30", "tax"), (13, "14:10", "disk"), (13, "18:00", "npm"),
    (14, "09:15", "flaky_checkout"), (14, "12:40", "keyerror"), (14, "15:30", "keyerror"), (15, "10:00", "pydantic"),
    (15, "13:20", "pydantic"), (15, "16:35", "pydantic"), (16, "09:40", "flaky_inventory"), (16, "14:05", "timeout"),
    (17, "11:25", "flaky_checkout"), (17, "17:15", "stripe"), (18, "10:45", "disk"), (18, "15:00", "email"),
    (19, "09:10", "flaky_checkout"), (19, "13:50", "flaky_inventory"), (20, "10:35", "flaky_checkout"), (20, "16:20", "disk"),
]

# pattern -> (truth label, rerun passed, branch choices, fix notes cycled per occurrence)
OUTCOMES: dict[str, tuple[str, bool, list[str], list[str]]] = {
    "flaky_checkout": ("flaky", True, ["main", "feat/wishlist", "fix/cart-badge", "chore/logging"], [
        "Reran the job, passed.",
        "Passed on rerun, no code change.",
        "Failed again, passed on rerun. Only fails under pytest-xdist when test_inventory runs on another worker; both use the same temp sqlite file.",
        "Rerun passed. Same shared tmp_db race between test_checkout_total and test_reserve_stock.",
    ]),
    "flaky_inventory": ("flaky", True, ["main", "feat/wishlist", "chore/logging"], [
        "Passed on rerun.",
        "Rerun green. Same shared tmp_db race as test_checkout_total.",
    ]),
    "pydantic": ("dependency", False, ["renovate/pydantic-2.x"], [
        "Renovate bumped pydantic to 2.9.2 and BaseSettings moved to pydantic-settings. Pinned pydantic<2 in requirements.txt.",
        "Same break on the same Renovate PR; pin applied.",
        "Closed the Renovate PR; pin holds.",
        "Same pydantic 2.x BaseSettings break from a new Renovate PR (2.10). Re-pinned pydantic<2 and added a Renovate ignore rule.",
        "Same break, same fix: pydantic<2 pin.",
        "Same break; ignore rule now merged.",
    ]),
    "disk": ("infra", False, ["main", "feat/wishlist"], [
        "runner-arm64-02 disk full from pip and docker caches. Ran docker system prune -af and cleared ~/.cache/pip, then the rerun passed.",
        "Disk full again on the arm64 runner. Cleared caches again.",
        "Same arm64 disk problem. Added a cache cleanup step to the workflow.",
    ]),
    "stripe": ("infra", True, ["main", "feat/checkout-v2"], [
        "DNS blip reaching api.stripe.com from the runner; rerun passed.",
        "Network failure to api.stripe.com again. This test should use stripe-mock instead of the real API.",
    ]),
    "npm": ("dependency", False, ["renovate/react-19"], [
        "react 19 breaks the react-beautiful-dnd peer range. Pinned react@18.3.1 until we move to @hello-pangea/dnd.",
    ]),
    "discount": ("regression", False, ["feat/pricing-floats"], [
        "Commit on feat/pricing-floats removed the quantize() call. Restored Decimal quantize to cents with ROUND_HALF_UP.",
    ]),
    "fuzzy": ("regression", False, ["refactor/search-tokenizer"], [
        "Tokenizer refactor dropped synonym expansion (navy -> blue). Restored the synonyms lookup.",
    ]),
    "keyerror": ("regression", False, ["feat/eu-shipping"], [
        "Real bug, not the usual shared-DB failure: feat/eu-shipping added region eu-west without a SHIPPING_ZONES entry. Added the zone.",
    ]),
    "timeout": ("infra", True, ["main"], [
        "Job hit the 30 minute limit while the runner queue was saturated; rerun finished in 9 minutes.",
    ]),
    "refund": ("regression", False, ["fix/refund-rounding"], ["Refund amount sign flipped in refunds.py; fixed."]),
    "tax": ("regression", False, ["main"], ["Stale California tax table (7.25 vs 7.75); updated rates."]),
    "email": ("regression", False, ["feat/order-emails"], ["Template variable renamed order -> purchase; updated the template."]),
}


class Gen:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def addr(self) -> str:
        return f"0x7f{self.rng.randrange(16**10):010x}"

    def sha(self) -> str:
        return f"{self.rng.getrandbits(160):040x}"

    def gha(self, ts: datetime, lines: list[str]) -> str:
        out, t = [], ts
        for line in lines:
            t += timedelta(milliseconds=self.rng.randint(3, 900))
            out.append(f"{t.strftime('%Y-%m-%dT%H:%M:%S')}.{self.rng.randrange(10**7):07d}Z {line}")
        return "\n".join(out) + "\n"

    def pytest_header(self) -> list[str]:
        return [
            "##[group]Run pytest -n 4 --maxfail=5 -q",
            "============================= test session starts ==============================",
            "platform linux -- Python 3.11.9, pytest-8.3.2, pluggy-1.5.0",
            "rootdir: /home/runner/work/shopfront/shopfront",
            "plugins: xdist-3.6.1, cov-5.0.0, anyio-4.4.0",
            "4 workers [212 items]",
            "",
        ]

    def pytest_footer(self, summary: str, failed: int = 1) -> list[str]:
        return [
            "=========================== short test summary info ============================",
            summary,
            f"=================== {failed} failed, {212 - failed} passed in {self.rng.uniform(38, 46):.2f}s ===================",
            "##[error]Process completed with exit code 1.",
        ]

    def assertion_failure(self, header: str, test_file: str, line: int, code: list[str], errors: list[str], summary: str) -> list[str]:
        return self.pytest_header() + [
            "=================================== FAILURES ===================================",
            header,
            f"[gw{self.rng.randint(0, 3)}] linux -- Python 3.11.9 {PYTHON}",
            "",
            *code,
            *errors,
            "",
            f"{test_file}:{line}: {summary.split(' - ')[1].split(':')[0]}",
        ] + self.pytest_footer(summary)

    # ---- patterns -------------------------------------------------------------------------
    def flaky_checkout(self, ts: datetime) -> str:
        a = self.addr()
        summary = "FAILED tests/test_checkout.py::test_checkout_total - AssertionError: assert Decimal('0.00') == Decimal('54.97')"
        return self.gha(ts, self.assertion_failure(
            "_____________________________ test_checkout_total ______________________________",
            "tests/test_checkout.py", 47,
            [f"tmp_db = <sqlite3.Connection object at {a}>", "",
             "    def test_checkout_total(tmp_db):",
             '        cart = seed_cart(tmp_db, items=[("SKU-1042", 2), ("SKU-2210", 1)])',
             '>       assert checkout_total(tmp_db, cart.id) == Decimal("54.97")'],
            ["E       AssertionError: assert Decimal('0.00') == Decimal('54.97')",
             f"E        +  where Decimal('0.00') = checkout_total(<sqlite3.Connection object at {a}>, {self.rng.randint(100, 999)})"],
            summary,
        ))

    def flaky_inventory(self, ts: datetime) -> str:
        a = self.addr()
        summary = "FAILED tests/test_inventory.py::test_reserve_stock - AssertionError: assert 1 == 3"
        return self.gha(ts, self.assertion_failure(
            "______________________________ test_reserve_stock ______________________________",
            "tests/test_inventory.py", 31,
            [f"tmp_db = <sqlite3.Connection object at {a}>", "",
             "    def test_reserve_stock(tmp_db):",
             '        add_stock(tmp_db, "SKU-1042", 5)',
             '        reserve(tmp_db, "SKU-1042", 2)',
             '>       assert get_stock(tmp_db, "SKU-1042") == 3'],
            ["E       AssertionError: assert 1 == 3",
             f"E        +  where 1 = get_stock(<sqlite3.Connection object at {a}>, 'SKU-1042')"],
            summary,
        ))

    def pydantic(self, ts: datetime, version: str = "2.9") -> str:
        full = {"2.9": "2.9.2", "2.10": "2.10.1", "2.11": "2.11.3"}[version]
        err = (f"pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package. "
               f"See https://errors.pydantic.dev/{version}/u/import-error for more details.")
        lines = [
            "##[group]Run pip install -r requirements.txt",
            "Collecting pydantic>=1.10 (from -r requirements.txt (line 7))",
            f"  Downloading pydantic-{full}-py3-none-any.whl.metadata (149 kB)",
            f"Successfully installed annotated-types-0.7.0 pydantic-{full} pydantic-core-2.23.4 typing-extensions-4.12.2",
        ] + self.pytest_header()[:-2] + ["collected 0 items / 3 errors", "",
            "==================================== ERRORS ====================================",
        ]
        for module in ["test_api_cart", "test_api_orders", "test_settings"]:
            lines += [
                f"_________________ ERROR collecting tests/{module}.py _________________",
                f"ImportError while importing test module '/home/runner/work/shopfront/shopfront/tests/{module}.py'.",
                "Traceback:",
                f"tests/{module}.py:4: in <module>",
                "    from shopfront.api import create_app",
                "shopfront/config.py:3: in <module>",
                "    from pydantic import BaseSettings",
                f".venv/lib/python3.11/site-packages/pydantic/__init__.py:{self.rng.choice([380, 402, 411])}: in __getattr__",
                "    return _getattr_migration(attr_name)",
                f"E   {err}",
            ]
        lines += ["=========================== short test summary info ============================"]
        lines += [f"ERROR tests/{m}.py - {err}" for m in ["test_api_cart", "test_api_orders", "test_settings"]]
        lines += ["!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!",
                  f"============================== 3 errors in {self.rng.uniform(1.5, 3.5):.2f}s ===============================",
                  "##[error]Process completed with exit code 2."]
        return self.gha(ts, lines)

    def disk(self, ts: datetime) -> str:
        h = self.sha()
        return self.gha(ts, [
            "Runner name: 'runner-arm64-02'",
            "Machine name: 'runner-arm64-02'",
            "##[group]Run pip install -r requirements.txt",
            "Collecting torch==2.4.1 (from -r requirements.txt (line 12))",
            "  Downloading torch-2.4.1-cp311-cp311-manylinux2014_aarch64.whl (89.7 MB)",
            f"     {self.rng.uniform(40, 89):.1f}/89.7 MB {self.rng.uniform(20, 60):.1f} MB/s eta 0:00:0{self.rng.randint(1, 9)}",
            f"ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device: "
            f"'/home/runner/.cache/pip/http-v2/{h[0]}/{h[1]}/{h[2]}/{h[3]}/{h[4]}/{h}.body'",
            "",
            "##[error]Process completed with exit code 1.",
        ])

    def stripe(self, ts: datetime) -> str:
        err = ("requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.stripe.com', port=443): "
               "Max retries exceeded with url: /v1/webhook_endpoints (Caused by NameResolutionError: "
               "Failed to resolve 'api.stripe.com' ([Errno -3] Temporary failure in name resolution))")
        return self.gha(ts, self.pytest_header() + [
            "=================================== FAILURES ===================================",
            "____________________________ test_stripe_signature _____________________________",
            "    def test_stripe_signature(stripe_client):",
            ">       endpoint = stripe_client.WebhookEndpoint.create(url=WEBHOOK_URL, enabled_events=['charge.succeeded'])",
            f"E       {err}",
            "",
            "tests/test_webhooks.py:19: ConnectionError",
        ] + self.pytest_footer(f"FAILED tests/test_webhooks.py::test_stripe_signature - {err}"))

    def npm(self, ts: datetime) -> str:
        return self.gha(ts, [
            "##[group]Run npm ci",
            "npm ERR! code ERESOLVE",
            "npm ERR! ERESOLVE could not resolve",
            "npm ERR!",
            "npm ERR! While resolving: react-beautiful-dnd@13.1.1",
            "npm ERR! Found: react@19.0.0",
            "npm ERR! node_modules/react",
            'npm ERR!   react@"^19.0.0" from the root project',
            "npm ERR!",
            "npm ERR! Could not resolve dependency:",
            'npm ERR! peer react@"^16.8.5 || ^17.0.0 || ^18.0.0" from react-beautiful-dnd@13.1.1',
            f"npm ERR! A complete log of this run can be found in: /home/runner/.npm/_logs/{ts.strftime('%Y-%m-%dT%H_%M_%S')}_{self.rng.randrange(1000):03d}Z-debug-0.log",
            "##[error]Process completed with exit code 1.",
        ])

    def discount(self, ts: datetime) -> str:
        summary = "FAILED tests/test_pricing.py::test_apply_discount_rounding - AssertionError: assert Decimal('16.9915') == Decimal('16.99')"
        return self.gha(ts, self.assertion_failure(
            "________________________ test_apply_discount_rounding _________________________",
            "tests/test_pricing.py", 22,
            ["    def test_apply_discount_rounding():",
             '>       assert apply_discount(Decimal("19.99"), pct=15) == Decimal("16.99")'],
            ["E       AssertionError: assert Decimal('16.9915') == Decimal('16.99')",
             "E        +  where Decimal('16.9915') = apply_discount(Decimal('19.99'), pct=15)"],
            summary,
        ))

    def fuzzy(self, ts: datetime) -> str:
        summary = "FAILED tests/test_search.py::test_fuzzy_match_colours - AssertionError: assert ['blue shirt'] == ['blue shirt', 'navy shirt']"
        return self.gha(ts, self.assertion_failure(
            "___________________________ test_fuzzy_match_colours ___________________________",
            "tests/test_search.py", 88,
            ["    def test_fuzzy_match_colours():",
             '>       assert search("blue shirt", fuzzy=True) == ["blue shirt", "navy shirt"]'],
            ["E       AssertionError: assert ['blue shirt'] == ['blue shirt', 'navy shirt']",
             "E         Right contains one more item: 'navy shirt'"],
            summary,
        ))

    def keyerror(self, ts: datetime) -> str:
        a = self.addr()
        return self.gha(ts, self.pytest_header() + [
            "=================================== FAILURES ===================================",
            "_____________________________ test_checkout_total ______________________________",
            f"tmp_db = <sqlite3.Connection object at {a}>",
            "    def checkout_total(conn, cart_id):",
            "        cart = load_cart(conn, cart_id)",
            ">       zone = SHIPPING_ZONES[cart.region]",
            "E       KeyError: 'eu-west'",
            "",
            "shopfront/checkout.py:58: KeyError",
        ] + self.pytest_footer("FAILED tests/test_checkout.py::test_checkout_total - KeyError: 'eu-west'"))

    def timeout(self, ts: datetime) -> str:
        return self.gha(ts, self.pytest_header() + [
            "........................................................................ [ 33%]",
            "..................................................",
            "##[error]The job running on runner GitHub Actions 7 has exceeded the maximum execution time of 30 minutes.",
            "##[error]The operation was canceled.",
        ])

    def refund(self, ts: datetime) -> str:
        summary = "FAILED tests/test_refunds.py::test_refund_partial - AssertionError: assert Decimal('-5.00') == Decimal('5.00')"
        return self.gha(ts, self.assertion_failure(
            "_____________________________ test_refund_partial ______________________________",
            "tests/test_refunds.py", 40,
            ["    def test_refund_partial(order):", '>       assert refund(order, amount=Decimal("5.00")).amount == Decimal("5.00")'],
            ["E       AssertionError: assert Decimal('-5.00') == Decimal('5.00')"],
            summary,
        ))

    def tax(self, ts: datetime) -> str:
        summary = "FAILED tests/test_tax.py::test_tax_california - AssertionError: assert Decimal('7.25') == Decimal('7.75')"
        return self.gha(ts, self.assertion_failure(
            "_____________________________ test_tax_california ______________________________",
            "tests/test_tax.py", 15,
            ["    def test_tax_california():", '>       assert tax_rate("CA", "94103") == Decimal("7.75")'],
            ["E       AssertionError: assert Decimal('7.25') == Decimal('7.75')"],
            summary,
        ))

    def email(self, ts: datetime) -> str:
        err = "jinja2.exceptions.UndefinedError: 'order' is undefined"
        return self.gha(ts, self.pytest_header() + [
            "=================================== FAILURES ===================================",
            "_______________________ test_order_confirmation_template _______________________",
            "    def test_order_confirmation_template(sample_purchase):",
            ">       html = render('emails/order_confirmation.html', purchase=sample_purchase)",
            f"E       {err}",
            "",
            "tests/test_emails.py:27: UndefinedError",
        ] + self.pytest_footer(f"FAILED tests/test_emails.py::test_order_confirmation_template - {err}"))

    def redis(self, ts: datetime) -> str:
        err = "redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused."
        return self.gha(ts, self.pytest_header() + [
            "=================================== FAILURES ===================================",
            "____________________________ test_session_roundtrip ____________________________",
            "    def test_session_roundtrip(redis_client):",
            '>       redis_client.set("session:42", "cart-1042")',
            f"E       {err}",
            "",
            "tests/test_sessions.py:12: ConnectionError",
        ] + self.pytest_footer(f"FAILED tests/test_sessions.py::test_session_roundtrip - {err}"))


def build() -> tuple[list[dict], dict[str, str]]:
    gen = Gen(seed=7)
    counters: dict[str, int] = {}
    runs: list[dict] = []
    for index, (day, hhmm, pattern) in enumerate(SCHEDULE, start=1):
        hour, minute = map(int, hhmm.split(":"))
        ts = START + timedelta(days=day, hours=hour, minutes=minute)
        occurrence = counters.get(pattern, 0)
        counters[pattern] = occurrence + 1
        label, rerun_passed, branches, notes = OUTCOMES[pattern]
        if pattern == "pydantic":
            log = gen.pydantic(ts, version="2.9" if day < 10 else "2.10")
        else:
            log = getattr(gen, pattern)(ts)
        runner = "runner-arm64-02" if pattern == "disk" else gen.rng.choice(["gh-ubuntu-x64", "gh-ubuntu-x64", "runner-arm64-02"])
        runs.append({
            "run_id": f"run-{200 + index}",
            "started_at": ts.isoformat(timespec="seconds"),
            "branch": gen.rng.choice(branches),
            "runner": runner,
            "commit": gen.sha(),
            "log": log,
            "truth_label": label,
            "outcome": {"rerun_passed": rerun_passed, "label": label, "fix_note": notes[occurrence % len(notes)]},
            "pattern": pattern,
        })
    demo_day = datetime(2026, 9, 29, 10, 12, tzinfo=timezone.utc)
    demos = {
        "01_checkout_flaky_again.log": gen.flaky_checkout(demo_day),
        "02_pydantic_again.log": gen.pydantic(demo_day + timedelta(minutes=30), version="2.11"),
        "03_new_redis_outage.log": gen.redis(demo_day + timedelta(hours=1)),
    }
    return runs, demos


def main() -> None:
    runs, demos = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for run in runs:
            fh.write(json.dumps(run, ensure_ascii=False) + "\n")
    demo_dir = OUT_DIR / "demo_logs"
    demo_dir.mkdir(exist_ok=True)
    for name, text in demos.items():
        (demo_dir / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(runs)} runs and {len(demos)} demo logs to {OUT_DIR}")


if __name__ == "__main__":
    main()
