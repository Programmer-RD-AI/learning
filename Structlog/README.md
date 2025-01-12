# StructLog

## Bound Loggers

The Bound Logger is returned from the `structlog.get_logger` method.
It contains 3 parts of: Context, Processors and Wrapped Logger.

### Context

A context dictionary that contains key value pairs. Merged into each log entry that is logged from this logger specifically.

Can inspect through `structlog.get_context()`

### Processors

List of processors that are called in every log entry. Set using the configurations.

### Wrapped Logger

Wrapped Logger is responsible for the output of the log entry, it is usually the standard `logging.Logger`

## Examples

### Setting the Logger Factory

`configure(logger_factory=LoggerFactory())` sets up structlog's base logging functionality, integrating it with Python's built-in logging module. It ensures that any calls to `structlog.get_logger()` produce compatible loggers, which manages log handling to Python's logging system. This setup allows structlog to leverage the backend logging configuration (e.g., handlers, formatters, and levels) defined in the logging module.

Without configure(), structlog functions with minimal features and defaults, which may not be suitable for production environments. Calling configure() ensures integration with Python’s logging and allows customization of log handling and formatting, making it essential for real-world applications.

```python
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG) # Set the logger to handle DEBUG and above messages
```

### `structlog.configure`

`processors`: A list of functions that the logged information goes through to finally be displayed.

`wrapper_class`: Optional class to use instead of the default BoundLogger

`context_class`: Class to be used for internal context storage (Default: dict)

`logger_factory`: [Already Discussed in the Previous Section](###settingtheloggerfactory)

`cache_logger_on_first_use`: The passed `wrapper_class` would be cached if this parameter is `True`.

### Bind and Unbind

#### Bind

```python

log = structlog.get_logger()
log.bind(x=42)
# 2024-12-05 20:11:45 [info     ] testing                        x=42
```

Adds context (extra fields) to the logger.
These fields are automatically included in subsequent log messages using that logger.
Multiple calls to .bind() accumulate context.

#### Unbind

Removes specific keys from the logger’s context.
If a key doesn't exist, it’s ignored.

### Custom Loggers

Utilize the `structlog.wrap_logger`, and create a class that contains the methods required for the types of logging such as: `msg`, `info`, etc...

```python
class CustomPrintLogger:
    def msg(self, message):
        print(message)
```

### Context Vars

```python

# merge_contextvars: A processor that merges in a global (context-local) context.

configure(
    processors=[
        merge_contextvars,
        structlog.processors.KeyValueRenderer(key_order=["event", "a"]),
    ]
)

# clear_contextvars: Clear the context-local context.

clear_contextvars()

# bind_contextvars: Put keys and values into the context-local context.

bind_contextvars(a=1, b=2)

# unbind_contextvars: Remove *keys* from the context-local context if they are present.

unbind_contextvars("b")

# bound_contextvars: Bind *kw* to the current context-local context. Unbind or restore *kw* afterwards. Do **not** affect other keys.

with bound_contextvars(b=2):
    log.info("hi")

# reset_contextvars: Reset contextvars corresponding to the given Tokens.

def foo():
    bind_contextvars(a=1)
    _helper()
    log.info("a is restored!")  # a=1


def _helper():
    tokens = bind_contextvars(a=2)
    log.info("a is overridden")  # a=2
    reset_contextvars(**tokens)
```
