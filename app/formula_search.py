"""Molecular-formula reverse search by branch-and-bound enumeration.

Given a target monoisotopic mass, observed isotope peaks and per-element
count ranges (2-6 distinct elements, each count between a ``min_count`` and
a ``max_count``), every composition whose monoisotopic mass falls within the
mass tolerance is enumerated depth-first. Elements are visited from the
heaviest monoisotopic mass down; at every node the minimum and maximum mass
still obtainable from the not-yet-assigned elements restricts the current
element's integer count window, so counts that cannot possibly land inside
the tolerance window are pruned without recursion.

Enumeration is fused with scoring: each within-tolerance composition is
scored the moment it is found — natural-abundance isotope convolution,
base-peak normalisation and one-to-one peak matching (unmatched peaks
counted at zero intensity) — and offered to a bounded structure holding at
most ``max_results`` candidates, so memory stays proportional to
``max_results`` instead of growing with the hit count. Nothing of a
discarded candidate (composition, envelope, pairing) is retained beyond
its own scoring step. Candidates rank by MSE, then absolute mass error,
then the element-symbol-lexicographic formula string; a fully identical
key keeps the earlier-traversed candidate. Searches visiting more than
:data:`MAX_VISITED_NODES` nodes abort with :class:`SearchLimitExceeded` so
an overly wide request cannot exhaust the worker; because enumeration is
cheap next to scoring, the production search first streams a counting-only
pass that lets the cap abort before any envelope is built, then streams a
second (deterministic, hence identical) pass that scores each hit.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from .enrichment import normalise_base_peak, pair_and_score
from .envelope import Peak, isotope_envelope
from .isotopes import most_abundant_isotope_mass
from .parser import ParsedFormula

# Hard cap on enumeration nodes (the root counts as one). Exceeding it
# aborts the search and the endpoint answers HTTP 422.
MAX_VISITED_NODES = 1_000_000

# Absolute guard (Da) against ordinary floating-point round-off when
# deriving integer count windows and deciding window membership. It only
# ever keeps boundary-neighbour counts (which are re-checked at the leaf);
# at 1e-10 Da it is far smaller than any physically meaningful mass
# tolerance and cannot include an off-by-one-atom neighbour (the lightest
# element, H, shifts the mass by ~1 Da per count).
_ROUND_EPS = 1e-10


class SearchLimitExceeded(Exception):
    """Enumeration exceeded :data:`MAX_VISITED_NODES` visited nodes."""

    def __init__(self, visited_nodes: int, pruned_branches: int) -> None:
        super().__init__(
            f"Formula search visited {visited_nodes} nodes; the limit is "
            f"{MAX_VISITED_NODES}"
        )
        self.visited_nodes = visited_nodes
        self.pruned_branches = pruned_branches


@dataclass(frozen=True)
class ElementSpec:
    """One search element with its inclusive atom-count range."""

    symbol: str
    min_count: int
    max_count: int


@dataclass
class FormulaHit:
    """One within-tolerance composition scored against the observed peaks."""

    formula: str
    composition: dict[str, int]
    monoisotopic_mass: float
    mass_error: float  # predicted monoisotopic mass - target mass (Da)
    mse: float
    pairs: list[tuple[int | None, int | None]]
    predicted: list[Peak]  # base-peak normalised envelope


@dataclass
class FormulaSearchResult:
    """Ranked hits (capped at the requested ``max_results``) plus search stats."""

    hits: list[FormulaHit]
    total_hits: int
    visited_nodes: int
    pruned_branches: int


def formula_string(composition: dict[str, int]) -> str:
    """Canonical formula: element symbols in lexicographic order, counts of
    1 omitted and zero-count elements skipped."""
    return "".join(
        symbol + (str(count) if count > 1 else "")
        for symbol, count in sorted(composition.items())
        if count > 0
    )


def _visit_order(elements: list[ElementSpec]) -> list[ElementSpec]:
    """Element visit order: heaviest monoisotopic mass first (ties broken
    by symbol) so the suffix mass bounds tighten as early as possible."""
    return sorted(
        elements,
        key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol),
    )


def stream_compositions(
    target_mass: float,
    mass_tolerance: float,
    elements: list[ElementSpec],
    emit: Callable[[tuple[int, ...], float], None],
) -> tuple[int, int]:
    """Streaming branch-and-bound enumeration of compositions in the window.

    Elements are visited heaviest-mononisotopic-mass first (ties by symbol)
    so the suffix mass bounds tighten as early as possible. Every
    within-tolerance composition is handed to ``emit`` as ``(counts in
    visit order, monoisotopic mass)`` the moment it is found; nothing is
    accumulated here, so callers can score hits one at a time without
    holding them all. Returns the visited-node and pruned-branch counters.
    Raises :class:`SearchLimitExceeded` past :data:`MAX_VISITED_NODES`
    visits.
    """
    specs = _visit_order(elements)
    masses = [most_abundant_isotope_mass(s.symbol) for s in specs]
    n = len(specs)

    # suffix_min[i]/suffix_max[i]: smallest/largest mass the elements
    # i..n-1 can still contribute given their count ranges.
    suffix_min = [0.0] * (n + 1)
    suffix_max = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + specs[i].min_count * masses[i]
        suffix_max[i] = suffix_max[i + 1] + specs[i].max_count * masses[i]

    visited_nodes = 0
    pruned_branches = 0

    def visit(index: int, counts: list[int], partial_mass: float) -> None:
        nonlocal visited_nodes, pruned_branches
        visited_nodes += 1
        if visited_nodes > MAX_VISITED_NODES:
            raise SearchLimitExceeded(visited_nodes, pruned_branches)

        if index == n:
            # Re-check membership with a pure round-off guard: count-window
            # widening at the parent may have reached boundary neighbours.
            if (
                partial_mass > 0.0
                and abs(partial_mass - target_mass) <= mass_tolerance + _ROUND_EPS
            ):
                emit(tuple(counts), partial_mass)
            return

        mass_i = masses[index]
        # For count c of this element to leave a feasible completion:
        #   partial + c*m_i + suffix_max[i+1] >= target - tolerance
        #   partial + c*m_i + suffix_min[i+1] <= target + tolerance
        low = target_mass - mass_tolerance - partial_mass - suffix_max[index + 1]
        high = target_mass + mass_tolerance - partial_mass - suffix_min[index + 1]
        window_lo = math.ceil((low - _ROUND_EPS) / mass_i)
        window_hi = math.floor((high + _ROUND_EPS) / mass_i)

        lo = max(specs[index].min_count, window_lo)
        hi = min(specs[index].max_count, window_hi)
        candidate_count = specs[index].max_count - specs[index].min_count + 1
        visited_count = max(0, hi - lo + 1)
        # Every count value outside the feasible window is one pruned
        # branch (the empty-window case included, via this subtraction).
        pruned_branches += candidate_count - visited_count

        for count in range(lo, hi + 1):
            counts.append(count)
            visit(index + 1, counts, partial_mass + count * mass_i)
            counts.pop()

    visit(0, [], 0.0)
    return visited_nodes, pruned_branches


def enumerate_compositions(
    target_mass: float,
    mass_tolerance: float,
    elements: list[ElementSpec],
) -> tuple[list[tuple[tuple[int, ...], float]], int, int]:
    """List-returning compatibility entry over :func:`stream_compositions`.

    Collects every emitted hit as ``((counts in visit order, monoisotopic
    mass), ...)`` plus the visited-node and pruned-branch counters; the
    production search uses the streaming entry instead so its memory stays
    bounded. Counters and :class:`SearchLimitExceeded` behaviour are
    identical.
    """
    raw_hits: list[tuple[tuple[int, ...], float]] = []
    visited_nodes, pruned_branches = stream_compositions(
        target_mass,
        mass_tolerance,
        elements,
        lambda counts, mass: raw_hits.append((counts, mass)),
    )
    return raw_hits, visited_nodes, pruned_branches


class _HitEntry:
    """Heap entry ordering kept candidates worst-first.

    Python's min-heap can then evict the worst kept candidate in
    O(log capacity): a larger (worse) ranking key compares *smaller* here,
    and a fully identical key is lost by the later traversal (larger
    sequence number), which is what keeps ties on the first-traversed
    candidate.
    """

    __slots__ = ("key", "seq", "hit")

    def __init__(
        self, key: tuple[float, float, str], seq: int, hit: FormulaHit
    ) -> None:
        self.key = key
        self.seq = seq
        self.hit = hit

    def __lt__(self, other: _HitEntry) -> bool:
        if self.key != other.key:
            return self.key > other.key
        return self.seq > other.seq


class _TopHitBuffer:
    """Bounded keeper of the best ``capacity`` candidates offered so far.

    At most ``capacity`` hits (with their compositions, envelopes and
    pairings) are referenced at any moment; a candidate that does not make
    the cut is dropped by the caller right after :meth:`offer`, so memory
    never grows with the number of hits.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._heap: list[_HitEntry] = []
        self._seq = 0

    def offer(self, hit: FormulaHit) -> None:
        """Consider ``hit`` for the kept set in traversal order."""
        key = (hit.mse, abs(hit.mass_error), hit.formula)
        entry = _HitEntry(key, self._seq, hit)
        self._seq += 1
        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, entry)
        elif self._heap[0] < entry:
            # Strictly better than the worst kept candidate: evict it. A
            # fully tied key fails this comparison — the earlier traversal
            # wins the tie — so only strict improvements ever evict.
            heapq.heapreplace(self._heap, entry)

    def ranked(self) -> list[FormulaHit]:
        """Kept hits in final rank order: ascending key, ties by traversal."""
        return [
            entry.hit
            for entry in sorted(self._heap, key=lambda e: (e.key, e.seq))
        ]


def _score_hit(
    counts: tuple[int, ...],
    mono_mass: float,
    target_mass: float,
    specs: list[ElementSpec],
    observed: list[Peak],
    mass_tolerance: float,
    merge_tolerance: float,
    prune: float,
) -> FormulaHit:
    composition = {
        specs[i].symbol: count for i, count in enumerate(counts) if count > 0
    }
    parsed = ParsedFormula(composition=composition, labels={})
    envelope = isotope_envelope(parsed, merge_tolerance, prune)
    predicted = normalise_base_peak(envelope)
    pairs, mse = pair_and_score(observed, predicted, mass_tolerance)
    return FormulaHit(
        formula=formula_string(composition),
        composition=composition,
        monoisotopic_mass=mono_mass,
        mass_error=mono_mass - target_mass,
        mse=mse,
        pairs=pairs,
        predicted=predicted,
    )


def search_formulas(
    target_mass: float,
    observed: list[Peak],
    elements: list[ElementSpec],
    mass_tolerance: float,
    max_results: int,
    merge_tolerance: float,
    prune: float,
) -> FormulaSearchResult:
    """Enumerate within-tolerance compositions and score them against the
    base-peak-normalised observed peaks, one hit at a time.

    A first streaming pass over :func:`stream_compositions` only counts
    hits: enumeration is cheap next to scoring, and the
    :data:`MAX_VISITED_NODES` cap aborts there — before any envelope is
    built — so an overly wide request fails fast. A second (deterministic,
    hence identical) streaming pass scores every composition as it is
    emitted and offers it to a bounded top-``max_results`` buffer ranked by
    ascending MSE, then absolute mass error, then lexicographic formula (a
    fully identical key keeps the earlier traversal); only the kept
    candidates survive, so memory does not grow with the hit count.
    """
    obs = normalise_base_peak(observed)
    specs = _visit_order(elements)

    total_hits = 0

    def count_hit(counts: tuple[int, ...], mono_mass: float) -> None:
        nonlocal total_hits
        total_hits += 1

    visited_nodes, pruned_branches = stream_compositions(
        target_mass, mass_tolerance, elements, count_hit
    )

    buffer = _TopHitBuffer(max_results)

    def score_and_offer(counts: tuple[int, ...], mono_mass: float) -> None:
        buffer.offer(
            _score_hit(
                counts,
                mono_mass,
                target_mass,
                specs,
                obs,
                mass_tolerance,
                merge_tolerance,
                prune,
            )
        )

    stream_compositions(target_mass, mass_tolerance, elements, score_and_offer)

    return FormulaSearchResult(
        hits=buffer.ranked(),
        total_hits=total_hits,
        visited_nodes=visited_nodes,
        pruned_branches=pruned_branches,
    )


__all__ = [
    "MAX_VISITED_NODES",
    "ElementSpec",
    "FormulaHit",
    "FormulaSearchResult",
    "SearchLimitExceeded",
    "enumerate_compositions",
    "formula_string",
    "search_formulas",
    "stream_compositions",
]
