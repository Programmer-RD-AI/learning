from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Log the request
        start_time = time.time()
        print(f"Request started: {request.method} {request.url}")

        # Call the next middleware or the request handler
        response = await call_next(request)

        # Log the response and the time taken to process the request
        process_time = time.time() - start_time
        print(
            f"Request finished: {request.method} {request.url} - Time taken: {process_time:.4f} seconds"
        )

        # Return the response
        return response
