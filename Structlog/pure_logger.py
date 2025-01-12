import logging

# Create or get a logger named 'my_logger'
logger = logging.getLogger("my_logger")

# Set the logger to handle DEBUG and above messages
logger.setLevel(logging.DEBUG)

# Add a StreamHandler to print logs to the console
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Log messages
logger.debug("This is a DEBUG message.")
logger.info("This is an INFO message.")
logger.warning("This is a WARNING message.")
