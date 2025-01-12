from structlog import get_logger, configure
from structlog.stdlib import LoggerFactory

# Configure structlog with Python's logging
# configure(logger_factory=LoggerFactory())

# Create a logger
log = get_logger("example_logger")

# Bind some context
log = log.bind(user="Alice", action="login")
log.info("User action started")  # Includes bound context

# Add more context dynamically
log = log.bind(ip="192.168.1.1")
log.info("User IP logged")  # Includes all bound context

# Unbind a specific key
log = log.unbind("action")
log.info("Action removed")  # Excludes 'action' but keeps others

# Unbind all context
log = log.unbind("user", "ip")
log.info("Context cleared")  # No bound context included
