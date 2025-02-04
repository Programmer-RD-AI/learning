import structlog
import logging
import datetime


def timestamper(_, __, event_dict):
    event_dict["time"] = datetime.datetime.now().isoformat()
    return event_dict


structlog.configure(
    processors=[
        timestamper,
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
structlog.contextvars.bind_contextvars(peer_ip="1.2.3.4")  # Globally Used Parameters
log = structlog.get_logger()
log.info("hello, %s!", "world", key="value!", more_than_strings=[1, 2, 3])
print(structlog.get_context(log))

