## Description (Required)
Fix tests that broke after `Query()` in `cyborgdb-core` / `cyborgdb-service` was changed to return only `id` by default (previously returned `id`, `distance`, and `metadata` when no `include` argument was provided). Updated affected tests in `test_api_contract.py`, `test_client.py`, and `test_concurrency.py` to pass `include=["distance"]` (or `["distance", "metadata"]`) wherever they assert on those keys. Also added a new contract test that pins the new default behavior — `query()` with no `include` returns only `{id}` — mirroring the test added in `cyborgdb-js`.

## Related Issue (Required)
Mirrors the fix already landed in `cyborgdb-js` (commits `f161735` "distance fixed in tests" and `38b917b` "added test to make sure we should only return ids from query"). Brings the Python SDK test suite in line with the new server-side default `include` behavior.

## Scope of This PR (Required)

- [ ] Feature Implementation
- [ ] Refactoring
- [ ] Performance Improvement
- [ ] Security Fix
- [x] Bug Fix
- [ ] Other (describe below)

**If "Other" was selected, describe the scope here:**

## Test Changes (Required)

- **Added/Removed Tests:**

  Updated:
  - `tests/test_api_contract.py::TestAPIContract::test_15_encrypted_index_query`
  - `tests/test_api_contract.py::TestAPIContract::test_16_encrypted_index_query_patterns`
  - `tests/test_api_contract.py::TestAPIContract::test_18_encrypted_index_query_binary`
  - `tests/test_client.py::ClientIntegrationTest::test_upsert_and_query`
  - `tests/test_client.py::ClientIntegrationTest::test_upsert_with_numpy_array`
  - `tests/test_client.py::ClientIntegrationTest::test_query_with_2d_numpy_array`
  - `tests/test_concurrency.py::TestConcurrentReadsAndWrites::test_queries_during_upserts`
  - `tests/test_concurrency.py::TestConcurrentReadsAndWrites::test_deletes_during_queries`
  - `tests/test_concurrency.py::TestStressHighConcurrency::test_20_threads_200_vectors_each`

  Added:
  - New default-include assertion inside `test_15_encrypted_index_query` verifying `query()` with no `include` arg returns only `{id}` (Python equivalent of the new `cyborgdb-js` "should not return distance by default" test).

  - [ ] No test changes

- **Reason:**

  `Query()` no longer returns `distance` and `metadata` by default. Tests that asserted on those keys without passing them in `include` were failing. Updated each affected call site to pass the appropriate `include` argument and pinned the new default behavior with an explicit contract assertion.

  - [ ] No test changes

## Breaking Changes

- [ ] This PR introduces breaking changes

  **If checked, please describe:**

  - **Impact:**

  - **Migration Steps:**

(No SDK code changes in this PR — the breaking change in `query()`'s default `include` was made in `cyborgdb-core` / `cyborgdb-service`. This PR only updates the Python SDK's tests to match the new contract.)

## Performance & Security Considerations

- [x] No known performance impact
- [x] No security concerns
- [ ] Requires additional security review

## Checklist

- [x] Code follows project style guidelines
- [x] Tests have been updated if needed
- [ ] Documentation has been updated if applicable

## Additional Context
Verified the full `test_api_contract.py`, `test_client.py`, and `test_concurrency.py` suites pass against a local `cyborgdb-service` instance. Audited the updated tests against the corresponding `cyborgdb-js` files (`api_contract.test.ts`, `basic_test.test.ts`, `concurrency_test.test.ts`) to confirm parity — every Python query assertion that checks `distance` or `metadata` now passes those keys via `include`, matching the JS suite's contract.

---
_Information regarding test changes will be automatically stored in the test log._
