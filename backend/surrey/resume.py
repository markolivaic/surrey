"""What a checkpoint must hold for a resumed run to draw the same batches as an unbroken one.

M1 seed A lost power at step 5,750 and resumed from a checkpoint that held the model, the
optimiser, the scaler and the step, and nothing random: the block sampler was reseeded with the
seed plus the step, and the generator behind dropout and the network's random centroids started
fresh. The run is still a valid draw of the recipe, but its data order after 5,750 is not the one
an unbroken run would have seen.

So a checkpoint now carries every random state the loop touches: Python's, the NumPy generator
that picks tiles and crops, torch's CPU generator and every CUDA generator. Training is sampled
by step, with no epochs, so the position in the data is the step itself. Each resume is appended
to the checkpoint's interruption record, with whether it was exact.
"""

from __future__ import annotations

import datetime as dt
import random

import numpy as np
import torch


def capture(rng: np.random.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore(state: dict, rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    rng.bit_generator.state = state["numpy_generator"]
    # torch.load with map_location="cuda" moves these byte tensors to the card, and both setters
    # take CPU byte tensors only (scripts/investigations/check_resume_gpu.py).
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def resume_event(step: int, exact: bool, note: str) -> dict:
    return {
        "resumed_from_checkpoint_step": step,
        "exact": exact,
        "note": note,
        "resumed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def interruption_record(events: list[dict]) -> dict:
    """The artefact field: how many times the run was resumed, where, and whether exactly."""
    return {
        "count": len(events),
        "events": events,
        "all_exact": all(e["exact"] for e in events) if events else None,
    }
