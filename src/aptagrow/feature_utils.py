"""Shared helpers for sequence and secondary-structure descriptors."""

from __future__ import annotations


def longest_run(text: str, accepted: set[str]) -> int:
    """Return the longest consecutive run composed only of accepted symbols."""
    best = current = 0
    for character in text:
        if character in accepted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
