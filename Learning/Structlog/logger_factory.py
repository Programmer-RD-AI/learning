from structlog import get_logger, configure
from structlog.stdlib import LoggerFactory

configure(logger_factory=LoggerFactory())
log = get_logger("a name")
log.info("User logged in", ip="192.168.1.1")
