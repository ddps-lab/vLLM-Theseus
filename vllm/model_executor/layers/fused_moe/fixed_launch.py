# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-size launch of the Triton experts kernel.

The point-to-point backends (``p2p_nccl``, and ``prerouted`` behind the
one-hop AFD connector) receive a different number of rows on every rank and
layer. The Triton experts kernel is compiled again for every new shape it
meets, so each rank would stop at a different time to compile while the
others wait for it. These backends therefore hand the experts kernel their
whole receive buffer, whose row count never changes, and record here how many
leading rows are real. The Triton GEMMs keep the full shape (one compile
key); the steps whose cost grows with the row count and that need no JIT
(activation, sum over top-k, output copy) stop at the real rows.

The producer (the prepare step) sets or clears the value before every
experts call; consumers only read it, and only when the call has the row
count it was recorded for. Thread-local because dual batch overlap runs each
micro-batch on its own thread.
"""

import threading

_state = threading.local()


def set_valid_rows(valid_rows: int, total_rows: int) -> None:
    """The next experts call has ``total_rows`` rows; the first
    ``valid_rows`` are real, the rest have expert id -1."""
    _state.rows = (valid_rows, total_rows)


def clear_valid_rows() -> None:
    """The next experts call is not a fixed-size launch."""
    _state.rows = None


def valid_rows(total_rows: int) -> int | None:
    """Real rows of a fixed-size launch with ``total_rows`` rows, else None."""
    rows = getattr(_state, "rows", None)
    if rows is None or rows[1] != total_rows:
        return None
    return rows[0]
