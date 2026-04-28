## Description (Required)
Two related fixes bundled into one PR:

1. **Test fixes for the `query()` default-include change.** `Query()` in `cyborgdb-core` / `cyborgdb-service` was changed to return only `id` by default (previously returned `id`, `distance`, and `metadata` when no `include` argument was provided). Updated affected tests in `test_api_contract.py`, `test_client.py`, and `test_concurrency.py` to pass `include=["distance"]` (or `["distance", "metadata"]`) wherever they assert on those keys. Also added a new contract test that pins the new default behavior — `query()` with no `include` returns only `{id}` — mirroring the test added in `cyborgdb-js`.

2. **CI workflow migration from 1Password `PIP_EXTRA_INDEX_URL` to AWS CodeArtifact.** The `PIP_EXTRA_INDEX_URL` field was removed from the `CyborgDB Dev` 1Password item, breaking `test.yml` and `build_and_package_wheels.yml`. Migrated both workflows to the same CodeArtifact-via-OIDC pattern already used in `cyborgdb-js`: load `CA_AWS_ACCOUNT_ID` from 1Password, assume an IAM role via OIDC, fetch a CodeArtifact authorization token, and export the resulting URL as `PIP_INDEX_URL`. Copied the `codeartifact-pip-url` composite action from `cyborgdb-js` into `.github/workflows/actions/`.

## Related Issue (Required)
Mirrors fixes already landed in `cyborgdb-js`:
- Test side: commits `f161735` "distance fixed in tests" and `38b917b` "added test to make sure we should only return ids from query".
- CI side: the CodeArtifact migration pattern in `cyborgdb-js/.github/workflows/test.yml` and the shared `codeartifact-pip-url` composite action.

Brings the Python SDK test suite and CI workflows in line with the new server-side default `include` behavior and the new package-index auth pattern.

## Scope of This PR (Required)

- [ ] Feature Implementation
- [ ] Refactoring
- [ ] Performance Improvement
- [ ] Security Fix
- [x] Bug Fix
- [x] Other (describe below)

**If "Other" was selected, describe the scope here:**
CI/CD: migrating the GitHub Actions workflows from a static 1Password pip-index secret to AWS CodeArtifact via OIDC role assumption.

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

(No SDK code changes in this PR — the breaking change in `query()`'s default `include` was made in `cyborgdb-core` / `cyborgdb-service`. This PR only updates the Python SDK's tests to match the new contract and migrates the CI workflows to AWS CodeArtifact.)

## Performance & Security Considerations

- [x] No known performance impact
- [x] No security concerns
- [ ] Requires additional security review

CI auth change: the `CyborgGitHubActionsS3Role` IAM trust policy must include `cyborginc/cyborgdb-py` as a permitted OIDC subject. The same role is already used by `cyborgdb-js`; if it is currently scoped to that repo only, its trust policy needs to be widened (one-time IAM change, not a code change in this PR).

## Checklist

- [x] Code follows project style guidelines
- [x] Tests have been updated if needed
- [ ] Documentation has been updated if applicable

## Additional Context
- Verified the full `test_api_contract.py`, `test_client.py`, and `test_concurrency.py` suites pass against a local `cyborgdb-service` instance.
- Audited the updated tests against the corresponding `cyborgdb-js` files (`api_contract.test.ts`, `basic_test.test.ts`, `concurrency_test.test.ts`) to confirm parity — every Python query assertion that checks `distance` or `metadata` now passes those keys via `include`, matching the JS suite's contract.
- Workflow changes: added `permissions: id-token: write` to the `test` job in both workflow files (required for OIDC), copied `codeartifact-pip-url/action.yml` from `cyborgdb-js`, and dropped the stale `--extra-index-url` flags from the `pip install` steps. The `build_and_package_wheels.yml` `env != 'prod'` conditional is preserved so prod releases still resolve `cyborgdb-service` from public PyPI.

---
_Information regarding test changes will be automatically stored in the test log._
