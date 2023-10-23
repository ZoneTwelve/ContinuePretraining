from functools import update_wrapper
from typing import Callable

import fire


def Fire(wrapped: Callable):
    return update_wrapper(lambda: fire.Fire(wrapped), wrapped)

