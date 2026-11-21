"""Terminal-state lift for the ``active_pr`` respawn guard.

``active_pr`` exists to stop a ready card from being re-spawned into a duplicate
PR while the implementer's PR is still in flight. It held for the whole 24-hour
window even after every linked PR had been MERGED or CLOSED — the work was
finished, but the card stayed un-spawnable.

These tests pin the boundary of the lift, all of it through the guard's real
code path with only the ``gh`` transport stubbed:

* MERGED and CLOSED lift (the guard returns None, and one
  ``respawn_guard_lifted`` event records the PR numbers and states);
* OPEN holds, and so does every unreadable answer — a non-zero ``gh`` exit, a
  timeout, a missing ``gh``, a malformed or unexpected payload — fail closed;
* a card linking several PRs (across several comments, duplicates included) is
  clear only when EVERY one of them is terminal, and each identity costs one
  lookup;
* the handoff waiver and the 24-hour window are untouched;
* the lift event is written once per (task, comment generation, PR set), so a
  replayed tick or a dry run does not re-emit it.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_active_pr as kapr

# The shared hermes_cli conftest stubs the resolver so no test reaches the
# network; the tests below drive the real one (in ``kanban_active_pr``) through
# the ``gh`` stub.
_REAL_QUERY_PR_STATE = getattr(kapr, "_query_pr_state", None)

PR_OPEN = "https://github.com/example/repo/pull/44"
PR_MERGED = "https://github.com/example/repo/pull/45"
PR_CLOSED = "https://github.com/example/repo/pull/46"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _reset_pr_cache() -> None:
    """Empty the resolver's bounded cache; absent on a checkout without the lift."""
    for name in ("_pr_state_cache", "_pr_state_inflight"):
        getattr(kapr, name, {}).clear()


class _gh:
    """A ``gh`` transport stub: ``answers`` maps URL -> ``(returncode, stdout)``
    or an exception to raise; anything unlisted is an OPEN PR.

    A non-zero returncode is raised as ``CalledProcessError`` exactly as
    ``subprocess.run(..., check=True)`` reports it, so the guard sees the same
    shapes it sees in production.
    """

    def __init__(self, answers: dict | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, dict]] = []
        self._saved_run = None

    def __enter__(self) -> "_gh":
        _reset_pr_cache()
        if _REAL_QUERY_PR_STATE is not None:
            # Undo the conftest's hermetic stub: these tests exercise the real
            # resolver, with only its subprocess transport replaced.
            kapr._query_pr_state = _REAL_QUERY_PR_STATE
        self._saved_run = kapr.subprocess.run
        kapr.subprocess.run = self._run
        return self

    def __exit__(self, *_exc) -> None:
        kapr.subprocess.run = self._saved_run
        _reset_pr_cache()

    def _run(self, command, **kwargs):
        self.calls.append((command[3], kwargs))
        answer = self.answers.get(command[3], (0, '{"state":"OPEN"}'))
        if isinstance(answer, BaseException):
            raise answer
        returncode, stdout = answer
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command, output=stdout, stderr=stdout)
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]

    def count(self, url: str) -> int:
        return sum(1 for called, _ in self.calls if called == url)


def _events(conn, tid: str, kind: str | None = None) -> list[tuple[str, Any]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    return out if kind is None else [e for e in out if e[0] == kind]


def _backdate_comments(conn, tid: str, seconds: int) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, tid),
        )


def _pr_comment(conn, tid: str, body: str = f"Opened {PR_OPEN} for review.") -> None:
    kb.add_comment(conn, tid, author="worker", body=body)


# ---------------------------------------------------------------------------
# The resolver: one bounded gh read, fail-closed on everything else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer,expected",
    [
        ((0, '{"state":"MERGED"}'), "MERGED"),
        ((0, '{"state":"CLOSED"}'), "CLOSED"),
        ((0, '{"state":"OPEN"}'), "OPEN"),
        ((0, '{"state":"merged"}'), "UNKNOWN"),  # exact enum only: lowercase holds
        ((0, '{"state":"Closed"}'), "UNKNOWN"),  # ...and mixed case holds
        ((0, '{"state":"WEIRD"}'), "UNKNOWN"),
        ((0, '{"number":44}'), "UNKNOWN"),
        ((0, "[]"), "UNKNOWN"),
        ((0, "not json"), "UNKNOWN"),
        ((0, ""), "UNKNOWN"),
        ((1, "no pull requests found"), "UNKNOWN"),
        ((4, "gh: not logged in"), "UNKNOWN"),
        (FileNotFoundError("gh"), "UNKNOWN"),
        (subprocess.TimeoutExpired(["gh"], 5), "UNKNOWN"),
        (OSError("connection reset"), "UNKNOWN"),
    ],
)
def test_pr_state_reads_only_exact_known_states(answer, expected) -> None:
    """Every non-terminal answer is UNKNOWN, and the guard holds on UNKNOWN."""
    with _gh({PR_OPEN: answer}):
        assert kapr._query_pr_state(PR_OPEN) == expected


def test_pr_state_query_is_a_bounded_gh_pr_view() -> None:
    """The lookup is ``gh pr view <url> --json state`` with a 5-second timeout."""
    with _gh({PR_OPEN: (0, '{"state":"MERGED"}')}) as gh:
        assert kapr._query_pr_state(PR_OPEN) == "MERGED"
    url, kwargs = gh.calls[0]
    assert url == PR_OPEN
    assert kwargs["timeout"] == 5
    assert kwargs["check"] is True


def test_pr_links_dedupes_by_identity_across_comments() -> None:
    """All URLs, all comments, one entry per canonical PR identity."""
    links = kapr.pr_links([
        f"Opened {PR_OPEN} and {PR_MERGED}/files",
        f"see https://github.com/Example/Repo/pull/44 and {PR_CLOSED}",
        "no PR here",
    ])
    assert [key for key, _, _, _ in links] == ["example/repo#44", "example/repo#45", "example/repo#46"]
    assert links[1][3] == PR_MERGED  # a /files suffix still queries the PR
    assert links[0][3] == PR_OPEN  # first spelling of a duplicated identity wins


# ---------------------------------------------------------------------------
# The cache: bounded, at most one lookup per identity per interval
# ---------------------------------------------------------------------------


def test_cached_pr_state_serves_a_repeat_lookup_from_the_cache(monkeypatch) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    with _gh({PR_OPEN: (0, '{"state":"MERGED"}')}) as gh:
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "MERGED"
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "MERGED"
        assert gh.count(PR_OPEN) == 1
        clock.now += kapr._PR_STATE_CACHE_TTL_SECONDS - 1
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "MERGED"
        assert gh.count(PR_OPEN) == 1


def test_cached_pr_state_re_queries_after_the_ttl(monkeypatch) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    with _gh() as gh:
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "OPEN"
        clock.now += kapr._PR_STATE_CACHE_TTL_SECONDS + 1
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "OPEN"
    assert gh.count(PR_OPEN) == 2


def test_cached_pr_state_caches_failures() -> None:
    """A gh outage costs one lookup per identity per interval, not one per tick."""
    with _gh({PR_OPEN: subprocess.TimeoutExpired(["gh"], 5)}) as gh:
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "UNKNOWN"
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "UNKNOWN"
    assert gh.count(PR_OPEN) == 1


def test_cached_pr_state_saturates_without_evicting_a_live_entry(monkeypatch) -> None:
    """F2: at capacity, a new identity is held WITHOUT a lookup — a live entry is
    never displaced, so an identity is never re-queried inside its interval."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    calls: list[str] = []

    def query(url: str) -> str:
        calls.append(url)
        return "OPEN"

    monkeypatch.setattr(kapr, "_query_pr_state", query)
    for number in range(kapr._PR_STATE_CACHE_MAX):
        kapr._cached_pr_state(f"example/repo#{number}",
                              f"https://github.com/example/repo/pull/{number}")
    # Assert WHILE the cache is populated (the old test inspected it after
    # _gh.__exit__ had emptied it, so its bounds were vacuous).
    assert len(kapr._pr_state_cache) == kapr._PR_STATE_CACHE_MAX
    assert len(calls) == kapr._PR_STATE_CACHE_MAX
    retained = "example/repo#0"
    overflow = "example/repo#9999"
    overflow_url = "https://github.com/example/repo/pull/9999"
    assert kapr._cached_pr_state(overflow, overflow_url) == "UNKNOWN"
    assert overflow_url not in calls  # held fail-closed, no lookup
    assert len(kapr._pr_state_cache) == kapr._PR_STATE_CACHE_MAX
    assert retained in kapr._pr_state_cache  # the 129th identity did not displace it
    # Repeated scans stay bounded and never re-query a retained identity.
    for _ in range(3):
        assert kapr._cached_pr_state(retained, "https://github.com/example/repo/pull/0") == "OPEN"
        assert kapr._cached_pr_state(overflow, overflow_url) == "UNKNOWN"
    assert calls.count("https://github.com/example/repo/pull/0") == 1
    assert calls.count(overflow_url) == 0


def test_saturated_cache_reclaims_only_expired_entries(monkeypatch) -> None:
    """F2: once an entry expires its slot is reusable; the cap still holds."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    calls: list[str] = []
    monkeypatch.setattr(kapr, "_query_pr_state", lambda url: (calls.append(url), "OPEN")[1])
    for number in range(kapr._PR_STATE_CACHE_MAX):
        kapr._cached_pr_state(f"example/repo#{number}",
                              f"https://github.com/example/repo/pull/{number}")
    assert kapr._cached_pr_state("example/repo#9999",
                                 "https://github.com/example/repo/pull/9999") == "UNKNOWN"
    clock.now += kapr._PR_STATE_CACHE_TTL_SECONDS + 1
    assert kapr._cached_pr_state("example/repo#9999",
                                 "https://github.com/example/repo/pull/9999") == "OPEN"
    assert len(kapr._pr_state_cache) == 1  # every expired entry was reclaimed
    assert len(calls) == kapr._PR_STATE_CACHE_MAX + 1


def test_failed_lookups_count_against_capacity(monkeypatch) -> None:
    """F2: cached failures bound retries — and they occupy slots like any answer."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    calls: list[str] = []
    monkeypatch.setattr(kapr, "_query_pr_state", lambda url: (calls.append(url), "UNKNOWN")[1])
    for number in range(kapr._PR_STATE_CACHE_MAX):
        assert kapr._cached_pr_state(f"example/repo#{number}",
                                     f"https://github.com/example/repo/pull/{number}") == "UNKNOWN"
    overflow_url = "https://github.com/example/repo/pull/9999"
    assert kapr._cached_pr_state("example/repo#9999", overflow_url) == "UNKNOWN"
    assert overflow_url not in calls
    assert len(kapr._pr_state_cache) == kapr._PR_STATE_CACHE_MAX


def test_concurrent_admissions_never_exceed_the_cap(monkeypatch) -> None:
    """F2: in-flight lookups count toward capacity, so concurrent admissions of
    distinct identities cannot push the cache past ``_PR_STATE_CACHE_MAX``."""
    full = threading.Event()
    release = threading.Event()
    queries: list[str] = []
    lock = threading.Lock()

    def query(url: str) -> str:
        with lock:
            queries.append(url)
            if len(queries) >= kapr._PR_STATE_CACHE_MAX:
                full.set()
        release.wait(5)
        return "OPEN"

    monkeypatch.setattr(kapr, "_query_pr_state", query)
    extra = 16
    results: dict[str, str] = {}

    def run(number: int) -> None:
        results[str(number)] = kapr._cached_pr_state(
            f"example/repo#{number}", f"https://github.com/example/repo/pull/{number}")

    threads = [threading.Thread(target=run, args=(n,))
               for n in range(kapr._PR_STATE_CACHE_MAX + extra)]
    for thread in threads:
        thread.start()
    assert full.wait(5)
    # MAX owners are all in flight; the surplus callers must already have held
    # without a lookup. No thread beyond the cap has issued a query.
    assert len(queries) == kapr._PR_STATE_CACHE_MAX
    assert len(kapr._pr_state_inflight) == kapr._PR_STATE_CACHE_MAX
    release.set()
    for thread in threads:
        thread.join(5)
    assert len(queries) == kapr._PR_STATE_CACHE_MAX
    assert len(kapr._pr_state_cache) <= kapr._PR_STATE_CACHE_MAX
    assert list(results.values()).count("UNKNOWN") >= extra


def test_stale_waiter_never_returns_an_expired_terminal_state(monkeypatch) -> None:
    """F1: an expired MERGED entry plus a held refresh must yield UNKNOWN, not
    the stale MERGED — the timeout path is fail-closed."""
    monkeypatch.setattr(kapr, "_PR_STATE_TIMEOUT_SECONDS", 0.2)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    kapr._pr_state_cache["example/repo#44"] = (clock.now - 1, "MERGED")  # expired
    started = threading.Event()
    release = threading.Event()

    def blocking_query(url: str) -> str:
        started.set()
        release.wait(5)
        return "UNKNOWN"

    monkeypatch.setattr(kapr, "_query_pr_state", blocking_query)
    owner = threading.Thread(
        target=lambda: kapr._cached_pr_state("example/repo#44", PR_OPEN))
    owner.start()
    assert started.wait(5)
    try:
        assert kapr._cached_pr_state("example/repo#44", PR_OPEN) == "UNKNOWN"
    finally:
        release.set()
        owner.join(5)


def test_guard_holds_when_the_only_terminal_answer_is_stale(monkeypatch, kanban_home) -> None:
    """F1 through the guard: a stale MERGED that a waiter cannot refresh holds."""
    monkeypatch.setattr(kapr, "_PR_STATE_TIMEOUT_SECONDS", 0.2)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(kapr.time, "time", lambda: clock.now)
    started = threading.Event()
    release = threading.Event()

    def blocking_query(url: str) -> str:
        started.set()
        release.wait(5)
        return "UNKNOWN"

    monkeypatch.setattr(kapr, "_query_pr_state", blocking_query)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale waiter", assignee="worker")
        _pr_comment(conn, tid)
        kapr._pr_state_cache["example/repo#44"] = (clock.now - 1, "MERGED")
        owner = threading.Thread(target=lambda: kapr.terminal_states([f"Opened {PR_OPEN}"]))
        owner.start()
        assert started.wait(5)
        try:
            assert kapr.terminal_states([f"Opened {PR_OPEN}"]) is None  # hold, not stale lift
            guard = kbd.check_respawn_guard(conn, tid)
        finally:
            release.set()
            owner.join(5)
    assert guard == "active_pr"


def test_concurrent_lookup_of_one_identity_calls_gh_once(monkeypatch) -> None:
    """Two callers, one identity, one ``gh`` call: the second takes the first's result."""
    started = threading.Event()
    release = threading.Event()
    queries: list[str] = []

    def blocking_query(url: str) -> str:
        queries.append(url)
        started.set()
        release.wait(5)
        return "MERGED"

    monkeypatch.setattr(kapr, "_query_pr_state", blocking_query)
    results: list[str] = []
    owner = threading.Thread(target=lambda: results.append(kapr._cached_pr_state("example/repo#44", PR_OPEN)))
    owner.start()
    assert started.wait(5)
    waiter = threading.Thread(target=lambda: results.append(kapr._cached_pr_state("example/repo#44", PR_OPEN)))
    waiter.start()
    release.set()
    owner.join(5)
    waiter.join(5)
    assert queries == [PR_OPEN]
    assert results == ["MERGED", "MERGED"]


# ---------------------------------------------------------------------------
# The guard: terminal states lift, everything else holds
# ---------------------------------------------------------------------------


def test_merged_pr_lifts_the_active_pr_guard(kanban_home) -> None:
    """A merged PR is finished work: the card may be spawned again."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already merged", assignee="worker")
        _pr_comment(conn, tid)
        generation = conn.execute(
            "SELECT MAX(created_at) AS ts FROM task_comments WHERE task_id = ?", (tid,)
        ).fetchone()["ts"]
        with _gh({PR_OPEN: (0, '{"state":"MERGED"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard is None
    assert gh.urls() == [PR_OPEN]
    assert lifted == [("respawn_guard_lifted",
                       {"generation": generation,
                        "prs": [{"repo": "example/repo", "number": 44, "state": "MERGED"}]})]


def test_closed_pr_lifts_the_active_pr_guard(kanban_home) -> None:
    """CLOSED (abandoned, superseded) is terminal too: no duplicate can follow it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="closed pr", assignee="worker")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: (0, '{"state":"CLOSED"}')}):
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard is None
    assert lifted[0][1]["prs"] == [{"repo": "example/repo", "number": 44, "state": "CLOSED"}]


def test_open_pr_holds_the_active_pr_guard(kanban_home) -> None:
    """The PR is still in flight: re-spawning would duplicate it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="open pr", assignee="worker")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: (0, '{"state":"OPEN"}')}):
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard == "active_pr"
    assert lifted == []


@pytest.mark.parametrize(
    "answer",
    [
        (1, "no pull requests found"),
        (4, "gh: not logged in"),
        (0, "not json"),
        (0, '{"state":"WEIRD"}'),
        (0, ""),
        FileNotFoundError("gh"),
        subprocess.TimeoutExpired(["gh"], 5),
    ],
)
def test_unreadable_pr_state_holds_fail_closed(kanban_home, answer) -> None:
    """An unreadable state is not evidence of a merge — hold, and say why."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unknown pr", assignee="worker")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: answer}):
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard == "active_pr"
    assert lifted == []


@pytest.mark.parametrize("answer", [(0, '{"state":"merged"}'), (0, '{"state":"Closed"}')])
def test_unexpected_terminal_spelling_holds_the_guard(kanban_home, answer) -> None:
    """F4: only the exact MERGED/CLOSED enum lifts; a case variant holds."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="lowercase terminal", assignee="worker")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: answer}):
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard == "active_pr"
    assert lifted == []


def test_one_open_pr_among_several_holds_the_guard(kanban_home) -> None:
    """Every linked PR must be terminal; one open PR keeps the card held."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="three prs", assignee="worker")
        _pr_comment(conn, tid, body=f"Opened {PR_MERGED} and {PR_CLOSED}")
        kb.add_comment(conn, tid, author="worker", body=f"Also opened {PR_OPEN}")
        with _gh({PR_MERGED: (0, '{"state":"MERGED"}'), PR_CLOSED: (0, '{"state":"CLOSED"}'),
                  PR_OPEN: (0, '{"state":"OPEN"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard == "active_pr"
    assert sorted(gh.urls()) == sorted([PR_MERGED, PR_CLOSED, PR_OPEN])
    assert lifted == []


def test_prs_across_several_comments_lift_when_all_are_terminal(kanban_home) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="two comments", assignee="worker")
        _pr_comment(conn, tid, body=f"Opened {PR_MERGED}")
        kb.add_comment(conn, tid, author="worker", body=f"Also {PR_CLOSED}")
        with _gh({PR_MERGED: (0, '{"state":"MERGED"}'), PR_CLOSED: (0, '{"state":"CLOSED"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard is None
    assert sorted(gh.urls()) == sorted([PR_MERGED, PR_CLOSED])
    assert len(lifted) == 1
    assert lifted[0][1]["prs"] == [
        {"repo": "example/repo", "number": 45, "state": "MERGED"},
        {"repo": "example/repo", "number": 46, "state": "CLOSED"},
    ]


def test_duplicate_pr_urls_cost_one_lookup(kanban_home) -> None:
    """The same PR linked from three comments (and re-cased) is one identity."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dupes", assignee="worker")
        _pr_comment(conn, tid, body=f"Opened {PR_OPEN}")
        kb.add_comment(conn, tid, author="worker", body=f"still {PR_OPEN}")
        kb.add_comment(conn, tid, author="other",
                       body="and https://github.com/Example/Repo/pull/44/files")
        with _gh({PR_OPEN: (0, '{"state":"MERGED"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
    assert guard is None
    assert gh.count(PR_OPEN) == 1
    assert gh.count("https://github.com/Example/Repo/pull/44") == 0


def test_handoff_after_the_newest_pr_comment_lifts_without_a_lookup(kanban_home) -> None:
    """The handoff waiver keeps priority: no gh call, no lift event needed."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="handed over", assignee="worker")
        _pr_comment(conn, tid)
        _backdate_comments(conn, tid, 60)
        assert kb.assign_task(conn, tid, "closer") is True
        with _gh({PR_OPEN: (0, '{"state":"OPEN"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    assert guard is None
    assert gh.calls == []
    assert lifted == []


def test_a_newer_pr_comment_after_the_handoff_guards_again(kanban_home) -> None:
    """Newest-comment semantics survive: the closer's own PR comment re-holds it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="handed over then commented", assignee="worker")
        _pr_comment(conn, tid)
        _backdate_comments(conn, tid, 60)
        assert kb.assign_task(conn, tid, "closer") is True
        with _gh({PR_OPEN: (0, '{"state":"OPEN"}')}):
            assert kbd.check_respawn_guard(conn, tid) is None
        kb.add_comment(conn, tid, author="closer", body=f"Pushed to {PR_OPEN}")
        with _gh({PR_OPEN: (0, '{"state":"OPEN"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
    assert guard == "active_pr"
    assert gh.count(PR_OPEN) == 1


def test_pr_comment_older_than_the_guard_window_is_ignored(kanban_home) -> None:
    """The 24-hour upper bound is unchanged: an old PR comment neither holds nor lifts."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="old pr comment", assignee="worker")
        _pr_comment(conn, tid)
        _backdate_comments(conn, tid, kbd._RESPAWN_GUARD_PR_WINDOW + 1)
        with _gh({PR_OPEN: (0, '{"state":"OPEN"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid)
    assert guard is None
    assert gh.calls == []


def test_review_lane_never_looks_up_pr_state(kanban_home) -> None:
    """The review lane's PR comment is the input to a review, not a duplicate signal."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: (0, '{"state":"OPEN"}')}) as gh:
            guard = kbd.check_respawn_guard(conn, tid, lane="review")
    assert guard is None
    assert gh.calls == []


# ---------------------------------------------------------------------------
# The lift event: written once per (task, generation, terminal PR set)
# ---------------------------------------------------------------------------


def test_lift_event_is_not_re_emitted_for_the_same_generation(kanban_home) -> None:
    """A replayed dispatch tick (or a later tick over unchanged comments) is silent."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="replayed tick", assignee="worker")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: (0, '{"state":"MERGED"}')}):
            assert kbd.check_respawn_guard(conn, tid) is None
            assert kbd.check_respawn_guard(conn, tid) is None
            # A dry run holds the event back entirely, and writes nothing.
            assert kbd.check_respawn_guard(conn, tid, dry_run=True) is None
        assert len(_events(conn, tid, "respawn_guard_lifted")) == 1

        # A NEW PR comment is a new generation: the lift is recorded again.
        kb.add_comment(conn, tid, author="worker", body=f"Opened {PR_MERGED} too")
        with _gh({PR_OPEN: (0, '{"state":"MERGED"}'), PR_MERGED: (0, '{"state":"MERGED"}')}):
            assert kbd.check_respawn_guard(conn, tid) is None
        assert len(_events(conn, tid, "respawn_guard_lifted")) == 2


def test_dry_run_records_no_lift_event(kanban_home) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dry run only", assignee="worker")
        _pr_comment(conn, tid)
        with _gh({PR_OPEN: (0, '{"state":"MERGED"}')}):
            assert kbd.check_respawn_guard(conn, tid, dry_run=True) is None
        assert _events(conn, tid, "respawn_guard_lifted") == []


def _canonical_payload(generation: int) -> dict:
    return {"generation": generation,
            "prs": [{"repo": "example/repo", "number": 44, "state": "MERGED"}]}


def test_record_lift_canonicalises_identity_case_and_order(kanban_home) -> None:
    """F3: the same PR set spelled or ordered differently is ONE event."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="canonical", assignee="worker")
        kapr.record_lift(conn, tid, [("Example/Repo", 44, "MERGED")], generation=1000, dry_run=False)
        kapr.record_lift(conn, tid, [("example/repo", 44, "MERGED")], generation=1000, dry_run=False)
        kapr.record_lift(
            conn, tid, [("example/repo", 45, "CLOSED"), ("example/repo", 44, "MERGED")],
            generation=1000, dry_run=False)
        kapr.record_lift(
            conn, tid, [("example/repo", 44, "MERGED"), ("example/repo", 45, "CLOSED")],
            generation=1000, dry_run=False)
        lifted = _events(conn, tid, "respawn_guard_lifted")
    # One event for the re-cased identity, one for the two-PR set; each spelling
    # and ordering of the same identity set collapses to its canonical form.
    assert [event[1] for event in lifted] == [
        {"generation": 1000,
         "prs": [{"repo": "example/repo", "number": 44, "state": "MERGED"}]},
        {"generation": 1000,
         "prs": [{"repo": "example/repo", "number": 44, "state": "MERGED"},
                 {"repo": "example/repo", "number": 45, "state": "CLOSED"}]},
    ]


def test_record_lift_persists_across_independent_connections(kanban_home) -> None:
    """F3: a second connection sees the first's event and does not re-append."""
    with kbc.connect() as conn1:
        tid = kb.create_task(conn1, title="persist", assignee="worker")
        kapr.record_lift(conn1, tid, [("example/repo", 44, "MERGED")], generation=1000, dry_run=False)
    with kbc.connect() as conn2:
        kapr.record_lift(conn2, tid, [("example/repo", 44, "MERGED")], generation=1000, dry_run=False)
        assert len(_events(conn2, tid, "respawn_guard_lifted")) == 1


def test_record_lift_check_and_append_are_atomic_across_connections(kanban_home) -> None:
    """F3: a check BEFORE the write txn lets two connections both read "absent"
    and append twice. The check must run inside the same ``BEGIN IMMEDIATE``."""
    with kbc.connect() as setup:
        tid = kb.create_task(setup, title="atomic", assignee="worker")
    started = threading.Event()
    attempted = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def holder() -> None:
        conn = kbc.connect()
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "respawn_guard_lifted", _canonical_payload(1000))
                started.set()
                release.wait(5)
        except BaseException as exc:  # noqa: BLE001 - surfaced via errors
            errors.append(exc)
        finally:
            conn.close()

    def writer() -> None:
        conn = kbc.connect()
        try:
            attempted.set()
            kapr.record_lift(conn, tid, [("example/repo", 44, "MERGED")], generation=1000, dry_run=False)
        except BaseException as exc:  # noqa: BLE001 - surfaced via errors
            errors.append(exc)
        finally:
            conn.close()

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert started.wait(5)
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert attempted.wait(5)
    release.set()
    holder_thread.join(10)
    writer_thread.join(10)
    assert errors == []
    with kbc.connect() as check:
        assert len(_events(check, tid, "respawn_guard_lifted")) == 1


def test_dispatch_once_spawns_a_card_whose_linked_pr_is_terminal(
    kanban_home, all_assignees_spawnable
) -> None:
    """End to end through the dispatcher: merged lifts, open still holds."""
    with kbc.connect() as conn:
        merged = kb.create_task(conn, title="merged pr", assignee="worker")
        _pr_comment(conn, merged, body=f"Opened {PR_MERGED}")
        opened = kb.create_task(conn, title="open pr", assignee="worker")
        _pr_comment(conn, opened, body=f"Opened {PR_OPEN}")
        with _gh({PR_MERGED: (0, '{"state":"MERGED"}'), PR_OPEN: (0, '{"state":"OPEN"}')}):
            res = kbd.dispatch_once(conn, dry_run=True)
        lifted = _events(conn, merged, "respawn_guard_lifted")
    assert merged in [s[0] for s in res.spawned]
    assert dict(res.respawn_guarded).get(opened) == "active_pr"
    assert opened not in [s[0] for s in res.spawned]
    # dry_run: no lift event, even though the guard lifted.
    assert lifted == []
