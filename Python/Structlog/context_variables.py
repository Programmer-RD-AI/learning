from structlog.contextvars import (
    bind_contextvars,
    bound_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
    reset_contextvars,
)
from structlog import configure
import structlog

configure(
    processors=[
        merge_contextvars,
        structlog.processors.KeyValueRenderer(key_order=["event", "a"]),
    ]
)
log = structlog.get_logger()
clear_contextvars()
bind_contextvars(a=1, b=2)
log.info("hello")
unbind_contextvars("b")
log.info("world")
with bound_contextvars(b=2):
    log.info("hi")
log.info("hi")
clear_contextvars()
log.info("hi there")


# Temporary Override
def foo():
    bind_contextvars(a=1)
    _helper()
    log.info("a is restored!")  # a=1


def _helper():
    tokens = bind_contextvars(a=2)
    log.info("a is overridden")  # a=2
    reset_contextvars(**tokens)
