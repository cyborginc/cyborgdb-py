## Background
We updated the behavior of the query function in cyborgdb-core and cyborgdb-service so that if a user calls the Query() function without an include argument, it returns only ids by default. The previous behavior was that if the user didn't provide an include argument, Query would return ids, distances, and metadata. This is causing several tests to fail.

We already made a fix for this on the cyborgdb-js repo and added an additional test on the contract tests to specifically check for this.

## Goal
- Fix all failing tests
- Add the same contract test that we added to cyborgdb-js

## Failing Tests
====================================================== short test summary info ======================================================
FAILED tests/test_api_contract.py::TestAPIContract::test_15_encrypted_index_query - AssertionError: query() result with default include: Missing required keys: {'distance', 'metadata'}
FAILED tests/test_api_contract.py::TestAPIContract::test_16_encrypted_index_query_patterns - AssertionError: 'distance' not found in {'id': '0'}
FAILED tests/test_api_contract.py::TestAPIContract::test_18_encrypted_index_query_binary - AssertionError: 'distance' not found in {'id': '104'}
FAILED tests/test_client.py::ClientIntegrationTest::test_query_with_2d_numpy_array - KeyError: 'distance'
FAILED tests/test_client.py::ClientIntegrationTest::test_upsert_and_query - AssertionError: False is not true
FAILED tests/test_client.py::ClientIntegrationTest::test_upsert_with_numpy_array - KeyError: 'distance'
FAILED tests/test_concurrency.py::TestConcurrentReadsAndWrites::test_deletes_during_queries - AssertionError: 4 != 0 : Delete-during-query errors: [('querier', 1, AssertionError("'distance' not found in {'id': 'seed_74'}")...
FAILED tests/test_concurrency.py::TestConcurrentReadsAndWrites::test_queries_during_upserts - AssertionError: 5 != 0 : Concurrent read/write errors: [('reader', 0, AssertionError("'distance' not found in {'id': 'seed_42'}"...
FAILED tests/test_concurrency.py::TestStressHighConcurrency::test_20_threads_200_vectors_each - AssertionError: 20 != 0 : Stress test errors: [(1, AssertionError("'distance' not found in {'id': 'stress_5_109'}")), (3, Assert...
======================================= 9 failed, 106 passed, 3 warnings in 151.27s (0:02:31) =======================================

## Related repos in my repos folder
- cyborgdb-core
- cyborgdb-service
- cyborgdb-js