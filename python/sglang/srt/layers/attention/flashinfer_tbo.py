"""Conservative FA2 eager-prefill split geometry for native two-batch overlap.

This only adjusts a cut inside the already selected sequence. It never drops
tokens, changes KV positions, changes the selected sequence, or disables TBO.
Unsupported shapes and batches without a compatible cut retain the original
split; this helper alone is not a general numerical-invariance guarantee.
"""

from math import gcd
from typing import Sequence


def fa2_split_is_compatible(
    lengths: Sequence[int],
    split_sequence: int,
    split_token: int,
    group_size: int,
    *,
    internal: bool,
) -> bool:
    """Check eager CTA choice and phase before the native DP TBO vote.

    Empty sides at a sequence boundary are allowed: the nonempty side is the
    original batch, and the empty side does not invoke attention. Empty or
    padded request entries are outside this policy's supported input domain.
    """
    if group_size <= 0 or not lengths or any(n <= 0 for n in lengths):
        return False
    if not 0 <= split_sequence <= len(lengths):
        return False
    total = sum(lengths)
    if not 0 <= split_token <= total:
        return False

    def tile(tokens, sequences):
        avg = tokens * group_size // sequences
        return 128 if avg > 64 else 64 if avg > 16 else 16

    parent_tile = tile(total, len(lengths))
    if internal:
        if split_sequence == len(lengths):
            return False
        offset = split_token - sum(lengths[:split_sequence])
        if not 0 < offset < lengths[split_sequence]:
            return False
        if offset * group_size % parent_tile:
            return False
        left_count = split_sequence + 1
    else:
        if split_token != sum(lengths[:split_sequence]):
            return False
        left_count = split_sequence
    right_count = len(lengths) - split_sequence
    return (left_count == 0 or tile(split_token, left_count) == parent_tile) and (
        right_count == 0 or tile(total - split_token, right_count) == parent_tile
    )


def aligned_fa2_split(
    lengths: Sequence[int], split_sequence: int, split_token: int, group_size: int
) -> int:
    """Match FA2's BF16 head-dim128 eager CTA_Q choice and packed-Q phase.

    The caller restricts this to ordinary deterministic FA2 prefill. The
    installed FA2 planner uses floor(total_queries * GQA / sequence_count),
    selecting CTA_Q 16/64/128 at thresholds 16/64. A packed tile boundary
    corresponds to CTA_Q / gcd(CTA_Q, GQA) query tokens.
    """
    if group_size <= 0 or not lengths or not 0 <= split_sequence < len(lengths):
        return split_token
    if any(n <= 0 for n in lengths):
        return split_token
    start = sum(lengths[:split_sequence])
    length = lengths[split_sequence]
    relative = split_token - start
    if not 0 < relative < length:
        return split_token

    total = sum(lengths)
    packed_average = total * group_size // len(lengths)
    tile, minimum, maximum = (
        (128, 65, None)
        if packed_average > 64
        else (64, 17, 64)
        if packed_average > 16
        else (16, 0, 16)
    )
    period = tile // gcd(tile, group_size)
    left_sequences = split_sequence + 1
    right_sequences = len(lengths) - split_sequence

    def ceil_div(a, b):
        return -(-a // b)

    # Both child planners must choose the parent's CTA_Q. Translate their
    # average-length limits into an interval for the absolute token cut.
    lower = max(start + 1, ceil_div(minimum * left_sequences, group_size))
    upper = min(
        start + length - 1,
        total - ceil_div(minimum * right_sequences, group_size),
    )
    if maximum is not None:
        upper = min(upper, ceil_div((maximum + 1) * left_sequences, group_size) - 1)
        lower = max(
            lower,
            total - (ceil_div((maximum + 1) * right_sequences, group_size) - 1),
        )
    first = max(1, ceil_div(lower - start, period))
    last = (upper - start) // period
    if first > last:
        return split_token
    # No scan proportional to sequence length. A tie chooses the earlier cut.
    before = min(last, max(first, relative // period))
    after = min(last, max(first, ceil_div(relative, period)))
    best = min(
        (before, after), key=lambda n: (abs(start + n * period - split_token), n)
    )
    return start + best * period
