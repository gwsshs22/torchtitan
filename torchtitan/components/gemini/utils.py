from enum import Enum, auto
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

def stateful_to_state_dict(
    states: dict[str, Any],
) -> dict[str, Any]:
    state_dict = {}
    for key, elem in states.items():
        if isinstance(elem, Stateful):
            state_dict[key] = elem.state_dict()
        else:
            state_dict[key] = elem

    return state_dict


def state_dict_to_stateful(
    states: dict[str, Any],
    state_dict: dict[str, Any],
) -> None:
    for key, elem in states.items():
        if key not in state_dict:
            continue
        if isinstance(elem, Stateful):
            elem.load_state_dict(state_dict[key])
        else:
            states[key] = state_dict[key]

class InMemStateType(Enum):
    LOCAL = auto()
    REMOTE = auto()