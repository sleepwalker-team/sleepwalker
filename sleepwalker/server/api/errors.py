"""The API's error format, and helpers to raise its common errors."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(status_code=status_code, detail={"code": code, "message": message, "details": details})


def not_found(code: str, message: str, details: dict[str, Any] | None = None) -> APIError:
    return APIError(status.HTTP_404_NOT_FOUND, code, message, details)


def bad_request(message: str, details: dict[str, Any] | None = None, code: str = "INVALID_REQUEST") -> APIError:
    return APIError(status.HTTP_400_BAD_REQUEST, code, message, details)


def conflict(code: str, message: str, details: dict[str, Any] | None = None) -> APIError:
    return APIError(status.HTTP_409_CONFLICT, code, message, details)


def unprocessable(code: str, message: str, details: dict[str, Any] | None = None) -> APIError:
    return APIError(status.HTTP_422_UNPROCESSABLE_CONTENT, code, message, details)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _internal_error_handler)


async def _request_validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = {"errors": jsonable_encoder(exc.errors(), custom_encoder={Exception: str})}
    return _error_response(status.HTTP_400_BAD_REQUEST, "INVALID_REQUEST", "Request validation failed.", details)


async def _http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Plain HTTP errors (e.g. an unknown route) get the same envelope as APIError.
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return _error_response(exc.status_code, detail["code"], detail["message"], detail.get("details"), exc.headers)


async def _internal_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled server exception", exc_info=exc)
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "An unexpected server error occurred.")


def _error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
) -> JSONResponse:
    content = {"error": {"code": code, "message": message, "details": details}}
    return JSONResponse(status_code=status_code, content=jsonable_encoder(content), headers=headers)
