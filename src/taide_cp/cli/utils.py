from functools import wraps
from typing import Callable

import fire

def Fire(wrapped: Callable):
    return wraps(wrapped)(lambda: fire.Fire(wrapped))

