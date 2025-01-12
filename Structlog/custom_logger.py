import structlog


class CustomPrintLogger:
    def msg(self, message):
        print(message)


def proc(logger, method_name, event_dict):
    print("I got called with", event_dict)
    return repr(event_dict)


log = structlog.wrap_logger(
    CustomPrintLogger(),
    wrapper_class=structlog.BoundLogger,
    processors=[proc],
)
log2 = log.bind(x=42)
log == log2
log.msg("hello world")
log2.msg("hello world")
log3 = log2.unbind("x")  # remove the default value
print(log3)

new_log = structlog.get_logger(structlog.stdlib.LoggerFactory()).bind(x=42)
new_log.info("testing", x=42)
