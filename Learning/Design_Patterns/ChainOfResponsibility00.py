from abc import ABC, abstractmethod
from typing import Optional

# Step 1: Define the Handler Interface
class SupportHandler(ABC):
    @abstractmethod
    def set_next(self, handler: 'SupportHandler') -> 'SupportHandler':
        """Sets the next handler in the chain"""
        pass

    @abstractmethod
    def handle_request(self, request_type: str) -> str:
        """Handles the request or passes it to the next handler"""
        pass


# Step 2: Concrete Handlers for each type of support
class TechnicalSupport(SupportHandler):
    def __init__(self):
        self._next_handler: Optional[SupportHandler] = None

    def set_next(self, handler: SupportHandler) -> SupportHandler:
        self._next_handler = handler
        return handler

    def handle_request(self, request_type: str) -> str:
        if request_type == "technical":
            return "Technical support handled the request."
        elif self._next_handler:
            return self._next_handler.handle_request(request_type)
        return "No handler could handle the request."


class BillingSupport(SupportHandler):
    def __init__(self):
        self._next_handler: Optional[SupportHandler] = None

    def set_next(self, handler: SupportHandler) -> SupportHandler:
        self._next_handler = handler
        return handler

    def handle_request(self, request_type: str) -> str:
        if request_type == "billing":
            return "Billing support handled the request."
        elif self._next_handler:
            return self._next_handler.handle_request(request_type)
        return "No handler could handle the request."


class GeneralSupport(SupportHandler):
    def __init__(self):
        self._next_handler: Optional[SupportHandler] = None

    def set_next(self, handler: SupportHandler) -> SupportHandler:
        self._next_handler = handler
        return handler

    def handle_request(self, request_type: str) -> str:
        if request_type == "general":
            return "General support handled the request."
        elif self._next_handler:
            return self._next_handler.handle_request(request_type)
        return "No handler could handle the request."


# Step 3: Client Code
def handle_support_request(request_type: str, handler: SupportHandler) -> str:
    return handler.handle_request(request_type)


# Example usage:
# Create the handlers
technical = TechnicalSupport()
billing = BillingSupport()
general = GeneralSupport()

# Set up the chain of responsibility
technical.set_next(billing).set_next(general)

# Handle various requests
requests = ["technical", "billing", "general", "other"]
for request in requests:
    result = handle_support_request(request, technical)
    print(f"Request: {request} -> {result}")
