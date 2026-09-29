from dejafail.signature import MAX_EXCERPT_CHARS, extract_signature, normalize

FLAKY_A = """2026-09-08T09:14:02.1182911Z ============================= test session starts ==============================
2026-09-08T09:14:02.1190000Z rootdir: /home/runner/work/shopfront/shopfront
2026-09-08T09:14:40.0000000Z tmp_db = <sqlite3.Connection object at 0x7f3a2c1e4b40>
2026-09-08T09:14:40.0000001Z >       assert checkout_total(tmp_db, cart.id) == Decimal("54.97")
2026-09-08T09:14:40.0000002Z E       AssertionError: assert Decimal('0.00') == Decimal('54.97')
2026-09-08T09:14:40.0000003Z E        +  where Decimal('0.00') = checkout_total(<sqlite3.Connection object at 0x7f3a2c1e4b40>, 318)
2026-09-08T09:14:40.0000004Z tests/test_checkout.py:47: AssertionError
2026-09-08T09:14:41.0000000Z =========================== short test summary info ============================
2026-09-08T09:14:41.0000001Z FAILED tests/test_checkout.py::test_checkout_total - AssertionError: assert Decimal('0.00') == Decimal('54.97')
2026-09-08T09:14:41.0000002Z ========================= 1 failed, 211 passed in 41.22s =========================
"""

FLAKY_B = """2026-09-19T16:02:11.5550000Z rootdir: /home/runner/work/shopfront/shopfront
2026-09-19T16:02:50.0000000Z tmp_db = <sqlite3.Connection object at 0x7f99aa01c2d0>
2026-09-19T16:02:50.0000002Z E       AssertionError: assert Decimal('0.00') == Decimal('54.97')
2026-09-19T16:02:50.0000003Z E        +  where Decimal('0.00') = checkout_total(<sqlite3.Connection object at 0x7f99aa01c2d0>, 512)
2026-09-19T16:02:51.0000001Z FAILED tests/test_checkout.py::test_checkout_total - AssertionError: assert Decimal('0.00') == Decimal('54.97')
2026-09-19T16:02:51.0000002Z ========================= 1 failed, 211 passed in 39.80s =========================
"""

KEYERROR = """E       KeyError: 'eu-west'
FAILED tests/test_checkout.py::test_checkout_total - KeyError: 'eu-west'
"""

DISK_1 = """Collecting torch==2.4.1 (from -r requirements.txt (line 12))
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device: '/home/runner/.cache/pip/http-v2/a/1/f/3/9/a1f39c0de4b7a1f39c0de4b7a1f39c0de4b7aa12.body'
##[error]Process completed with exit code 1.
"""

DISK_2 = """ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device: '/home/runner/.cache/pip/http-v2/7/c/0/2/e/7c02e55b19d07c02e55b19d07c02e55b19d0ff31.body'
##[error]Process completed with exit code 1.
"""

PYDANTIC = """ERROR tests/test_api_cart.py - pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package. See https://errors.pydantic.dev/2.9/u/import-error for more details.
ERROR tests/test_api_orders.py - pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package. See https://errors.pydantic.dev/2.9/u/import-error for more details.
"""

PYDANTIC_NEWER = PYDANTIC.replace("2.9", "2.10")


def test_pytest_failure_extracts_test_and_error():
    sig = extract_signature(FLAKY_A)
    assert sig.test_id == "tests/test_checkout.py::test_checkout_total"
    assert sig.error_type == "AssertionError"
    assert sig.message.startswith("assert Decimal('0.00')")
    assert len(sig.sig_hash) == 12


def test_same_failure_with_different_noise_has_same_hash():
    assert extract_signature(FLAKY_A).sig_hash == extract_signature(FLAKY_B).sig_hash


def test_different_error_in_same_test_has_different_hash():
    a = extract_signature(FLAKY_A)
    b = extract_signature(KEYERROR)
    assert b.test_id == a.test_id
    assert b.error_type == "KeyError"
    assert b.sig_hash != a.sig_hash


def test_infra_failure_has_no_test_id_and_paths_normalize():
    one, two = extract_signature(DISK_1), extract_signature(DISK_2)
    assert one.test_id is None
    assert one.error_type == "OSError"
    assert "No space left on device" in one.message
    assert one.sig_hash == two.sig_hash


def test_collection_error_uses_first_module_and_dotted_error_type():
    sig = extract_signature(PYDANTIC)
    assert sig.test_id == "tests/test_api_cart.py"
    assert sig.error_type == "pydantic.errors.PydanticImportError"


def test_library_version_in_docs_url_does_not_change_hash():
    assert extract_signature(PYDANTIC).sig_hash == extract_signature(PYDANTIC_NEWER).sig_hash


def test_ansi_codes_and_gha_timestamps_are_stripped():
    plain = "FAILED tests/x.py::test_y - AssertionError: boom\n"
    noisy = "2026-09-08T09:14:02.1182911Z \x1b[31mFAILED\x1b[0m tests/x.py::test_y - AssertionError: boom\n"
    assert extract_signature(noisy).sig_hash == extract_signature(plain).sig_hash
    assert extract_signature(noisy).test_id == "tests/x.py::test_y"


def test_windows_line_endings_parse_the_same():
    crlf = FLAKY_A.replace("\n", "\r\n")
    assert extract_signature(crlf).sig_hash == extract_signature(FLAKY_A).sig_hash
    assert extract_signature("FAILED tests/x.py::test_y\r\n").test_id == "tests/x.py::test_y"


def test_npm_error_code():
    sig = extract_signature("npm ERR! code ERESOLVE\nnpm ERR! ERESOLVE could not resolve\n")
    assert sig.test_id is None
    assert sig.error_type == "npm ERR"
    assert sig.message == "code ERESOLVE"


def test_github_actions_error_line_fallback():
    log = "##[error]The job running on runner GitHub Actions 7 has exceeded the maximum execution time of 30 minutes.\n"
    sig = extract_signature(log)
    assert sig.error_type == "CIError"
    assert "maximum execution time" in sig.message


def test_jest_failure_test_id():
    log = "  ● CartTotals › applies discount\n\n    expect(received).toBe(expected)\n"
    assert extract_signature(log).test_id == "CartTotals › applies discount"


def test_junit_failure_test_id():
    log = "[ERROR] testTotal(com.shop.CartTest)  Time elapsed: 0.01 s  <<< FAILURE!\njava.lang.AssertionError: expected:<42> but was:<41>\n"
    sig = extract_signature(log)
    assert sig.test_id == "com.shop.CartTest.testTotal"
    assert sig.error_type == "java.lang.AssertionError"


def test_empty_log_does_not_crash():
    sig = extract_signature("")
    assert sig.test_id is None
    assert sig.error_type == "Unknown"
    assert len(sig.sig_hash) == 12


def test_excerpt_is_capped():
    log = "\n".join(f"line {i} ok" for i in range(20000)) + "\nE   AssertionError: late\n"
    assert len(extract_signature(log).excerpt) <= MAX_EXCERPT_CHARS


def test_normalize_strips_volatile_tokens():
    text = "at 0x7f3a2c1e4b40 in /home/runner/work/app/file.py:42 took 41.22s id 3f2a9c1d"
    out = normalize(text)
    assert "0x7f" not in out and "/home/runner" not in out and ":42" not in out
    assert "41.22s" not in out and "3f2a9c1d" not in out
