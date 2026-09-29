"""Shared test helpers.

Mirrors js src/__tests__/test-helpers.ts and go test/helpers_test.go.

The waits here replace the fixed ``time.sleep(2)`` the suites used to do after
every upsert. A fixed delay is the worst of both worlds: flaky on a loaded
machine, and pure waste on an idle one. These poll for the condition that
actually matters and fail with a useful message if it never arrives.
"""

import time

DEFAULT_TIMEOUT = 30.0
POLL_INTERVAL = 0.2


def wait_for_ids(index, expected_ids, timeout=DEFAULT_TIMEOUT):
    """Block until every id in ``expected_ids`` is visible to query_metadata."""
    expected = set(expected_ids)
    deadline = time.monotonic() + timeout
    seen = set()
    while time.monotonic() < deadline:
        try:
            seen = {row["id"] for row in index.query_metadata()}
            if expected <= seen:
                return
        except Exception:
            # The index may not be queryable for a moment after creation.
            pass
        time.sleep(POLL_INTERVAL)
    raise AssertionError(
        f"upserted ids still not visible after {timeout}s; "
        f"missing {sorted(expected - seen)}"
    )


def wait_until_gone(index, gone_ids, timeout=DEFAULT_TIMEOUT):
    """Block until none of ``gone_ids`` are visible — the delete-side counterpart."""
    gone = set(gone_ids)
    deadline = time.monotonic() + timeout
    still_there = gone
    while time.monotonic() < deadline:
        still_there = {row["id"] for row in index.query_metadata()} & gone
        if not still_there:
            return
        time.sleep(POLL_INTERVAL)
    raise AssertionError(
        f"deleted ids still visible after {timeout}s: {sorted(still_there)}"
    )


def wait_for(predicate, description, timeout=DEFAULT_TIMEOUT):
    """Block until ``predicate()`` returns truthy.

    For conditions that are not simply "these ids exist" — an updated text field
    becoming searchable, for instance.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"condition never held within {timeout}s: {description}")
