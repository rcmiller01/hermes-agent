"""The ``active_pr`` respawn hold's evidence: the PRs a card links, and their GitHub state.

A ready card that a recent comment links to a GitHub PR is not re-spawned — the
implementer already opened the PR, and a second run would duplicate it. That
hold lifts only when the evidence says the PR work is finished: EVERY PR the
card links, across every comment in the guard window, is MERGED or CLOSED.
Anything else holds, fail-closed — a PR that is OPEN, missing, or unreadable (a
non-zero ``gh`` exit, the 5-second timeout, a malformed payload) is not evidence
that the work landed.

The reads are bounded by a small process-local cache that caches failures too,
so an outage costs one ``gh`` call per PR identity per interval rather than one
per dispatch tick. The lift itself is recorded in the DB (``respawn_guard_lifted``
on the task), so it survives a process replay.

Split out of ``hermes_cli.kanban_db_dispatch``, whose :func:`check_respawn_guard`
owns the priority order across all guard reasons and calls in here for this step
(including :func:`is_handoff_event`, the waiver that lifts the hold before any
lookup).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from typing import Iterable
from typing import Optional

_log = logging.getLogger(__name__)

# Grouped so one scan yields every PR identity a comment links, not just the
# first: a card that links three PRs is only clear once all three are terminal.
PR_URL_RE = re.compile(
    r"https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)",
    re.IGNORECASE,
)

# A linked PR lifts ``active_pr`` only when GitHub reports a terminal state.
_TERMINAL_PR_STATES = frozenset({"MERGED", "CLOSED"})
_PR_STATE_KNOWN = _TERMINAL_PR_STATES | {"OPEN"}
_PR_STATE_UNKNOWN = "UNKNOWN"
_PR_STATE_TIMEOUT_SECONDS = 5
# Bounded, process-local, and it caches failures too: a NOT-FOUND or a gh
# outage costs one lookup per identity per interval, not one per dispatch tick.
_PR_STATE_CACHE_TTL_SECONDS = 300
_PR_STATE_CACHE_MAX = 128

_pr_state_cache: dict[str, tuple[float, str]] = {}
_pr_state_inflight: dict[str, threading.Event] = {}
_pr_state_lock = threading.Lock()


def pr_links(bodies: Iterable[str]) -> list[tuple[str, str, int, str]]:
    """``(cache key, repo, number, url)`` for every PR URL in ``bodies``.

    ALL URLs across ALL comments in the guard window, deduplicated by canonical
    identity (``owner/repo#N``, case-folded) — the same PR linked from three
    comments is one identity and one lookup. URLs are normalised to
    ``https://github.com/<repo>/pull/<N>`` so a trailing ``/files`` suffix still
    queries the PR itself.
    """
    seen: dict[str, tuple[str, str, int, str]] = {}
    for body in bodies:
        for match in PR_URL_RE.finditer(body or ""):
            repo, number = f"{match[1]}/{match[2]}", int(match[3])
            key = f"{repo.lower()}#{number}"
            seen.setdefault(key, (key, repo, number, f"https://github.com/{repo}/pull/{number}"))
    return list(seen.values())


def _query_pr_state(url: str) -> str:
    """One ``gh pr view <url> --json state``, fail-closed.

    Returns ``OPEN`` / ``MERGED`` / ``CLOSED`` for an exact known state, and
    ``UNKNOWN`` for every other outcome: ``gh`` missing, a non-zero exit
    (not-found, auth, network), the timeout, or a payload that is not the
    expected ``{"state": ...}`` object. The caller holds the guard on anything
    that is not terminal, so the log line here is the operator's explanation.
    """
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", url, "--json", "state"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_PR_STATE_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning(
            "kanban: gh pr view %s failed (%s) — holding the respawn guard",
            url, type(exc).__name__,
        )
        return _PR_STATE_UNKNOWN
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        payload = None
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, str) or state.upper() not in _PR_STATE_KNOWN:
        _log.warning(
            "kanban: gh pr view %s returned an unexpected payload — holding the respawn guard", url,
        )
        return _PR_STATE_UNKNOWN
    state = state.upper()
    if state not in _TERMINAL_PR_STATES:
        _log.info("kanban: %s is %s — the respawn guard holds until it is MERGED/CLOSED", url, state)
    return state


def _cached_pr_state(key: str, url: str, *, now: Optional[float] = None) -> str:
    """``_query_pr_state`` behind a bounded TTL cache, at most one lookup per
    identity per interval.

    Failures are cached like any other answer, so a gh outage or a 404 costs one
    lookup per identity per TTL rather than one per dispatch tick. A concurrent
    caller for an identity already being looked up waits for that lookup's
    result instead of issuing a second ``gh`` call.
    """
    now = time.time() if now is None else now
    with _pr_state_lock:
        entry = _pr_state_cache.get(key)
        if entry is not None and entry[0] > now:
            return entry[1]
        waiter = _pr_state_inflight.get(key)
        owner = waiter is None
        if owner:
            waiter = threading.Event()
            _pr_state_inflight[key] = waiter
    if not owner:
        # Another caller owns the lookup: take its result once it lands.
        waiter.wait(_PR_STATE_TIMEOUT_SECONDS + 1)
        with _pr_state_lock:
            entry = _pr_state_cache.get(key)
        return entry[1] if entry is not None else _PR_STATE_UNKNOWN
    state = _PR_STATE_UNKNOWN
    try:
        state = _query_pr_state(url)
    except Exception:  # health: allow BLE001 -- a lookup failure must hold the guard, never spawn a duplicate PR; logs exc_info
        _log.warning("kanban: PR state lookup for %s raised — holding the respawn guard",
                     url, exc_info=True)
    finally:
        with _pr_state_lock:
            _pr_state_store(key, state, now)
            _pr_state_inflight.pop(key, None)
            waiter.set()
    return state


def _pr_state_store(key: str, state: str, now: float) -> None:
    """Insert one cache entry, evicting expired entries first and then the
    soonest-expiring one — the cache stays within ``_PR_STATE_CACHE_MAX``."""
    if key not in _pr_state_cache and len(_pr_state_cache) >= _PR_STATE_CACHE_MAX:
        for expired in [k for k, (expires, _) in _pr_state_cache.items() if expires <= now]:
            del _pr_state_cache[expired]
        while len(_pr_state_cache) >= _PR_STATE_CACHE_MAX:
            del _pr_state_cache[min(_pr_state_cache, key=lambda k: _pr_state_cache[k][0])]
    _pr_state_cache[key] = (now + _PR_STATE_CACHE_TTL_SECONDS, state)


def terminal_states(bodies: Iterable[str]) -> Optional[list[tuple[str, int, str]]]:
    """``[(repo, number, state)]`` when EVERY linked PR is terminal, else None.

    None is the hold: one OPEN PR among three linked ones still means the work
    is in flight, and an unknown answer is not evidence of anything.
    """
    states = [(repo, number, _cached_pr_state(key, url)) for key, repo, number, url in pr_links(bodies)]
    if not states or any(state not in _TERMINAL_PR_STATES for _, _, state in states):
        return None
    return states


def record_lift(
    conn, task_id: str, states: list[tuple[str, int, str]], *, generation: int, dry_run: bool,
) -> None:
    """Persist one ``respawn_guard_lifted`` event per (task, comment generation,
    terminal PR set) — a replayed dispatch tick, or the next tick while the PR
    comments are unchanged, must not re-emit it.

    ``generation`` is the newest in-window PR comment's timestamp: a NEW PR
    comment (the implementer linking a second PR) is a new generation and emits
    again. The event names the PR numbers and their terminal states only.
    """
    if dry_run:
        return
    payload = {
        "generation": int(generation),
        "prs": [{"repo": repo, "number": number, "state": state} for repo, number, state in states],
    }
    if conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'respawn_guard_lifted' "
        "AND payload = ? LIMIT 1",
        (task_id, _kb._json_or_null(payload)),
    ).fetchone() is not None:
        return
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, "respawn_guard_lifted", payload)


def is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff — and a handoff recorded after the newest PR comment lifts the
    hold for the profile now named on the card.

    A no-op re-assign (dev→dev via CLI/dashboard/``reassign --reclaim``), an
    unassign, or the dispatcher's own ``kanban.default_assignee`` write would
    otherwise lift ``active_pr`` for the very implementer that opened the PR.
    Events without ``from`` (written before it was recorded) are not trusted as
    handoffs — fail closed.
    """
    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


# Import-cycle breaking, exactly as in ``kanban_db_dispatch``: the origin-resident
# helpers are reached late-bound so monkeypatching ``kanban_db.<name>`` keeps working.
from hermes_cli import kanban_db as _kb
