# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Union

import requests
from requests.exceptions import HTTPError

from modelscope.hub.constants import MODELSCOPE_REQUEST_ID
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


class NotSupportError(Exception):
    """Requested operation is not supported in the current context (E3023)."""
    error_code = 'E3023'
    retryable = False
    suggestion = 'This operation is not supported. Please check the documentation.'


class NoValidRevisionError(Exception):
    """Requested revision does not exist (E2021)."""
    error_code = 'E2021'
    retryable = False
    suggestion = 'The requested revision does not exist. Please verify the version.'


class NotExistError(Exception):
    """Resource does not exist (E3020)."""
    error_code = 'E3020'
    retryable = False
    suggestion = 'The requested resource does not exist or has been deleted.'


class RequestError(Exception):
    """HTTP request error with structured fields for programmatic handling."""
    error_code = 'E9001'
    retryable = False
    suggestion = 'Unexpected request error. Please retry or report the issue.'

    def __init__(self, message, status_code=None, request_id=None, url=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.url = url
        if status_code is not None:
            if status_code >= 500:
                self.error_code = 'E1002'
                self.retryable = True
                self.suggestion = 'Server is currently unavailable. Please retry later.'
            elif status_code == 429:
                self.error_code = 'E1021'
                self.retryable = True
                self.suggestion = 'Rate limit exceeded. Please reduce request frequency.'
            elif status_code == 404:
                self.error_code = 'E3020'
                self.retryable = False
            elif status_code == 403:
                self.error_code = 'E3002'
                self.retryable = False
            elif status_code == 401:
                self.error_code = 'E3001'
                self.retryable = False


class GitError(Exception):
    """Git CLI operation failed (E1024)."""
    error_code = 'E1024'
    retryable = False
    suggestion = 'Git operation failed. Please check network and repository permissions.'


class InvalidParameter(Exception):
    """Invalid request parameter (E3021)."""
    error_code = 'E3021'
    retryable = False
    suggestion = 'Invalid request parameters. Please check and retry.'


class NotLoginException(Exception):
    """User is not logged in (E3022)."""
    error_code = 'E3022'
    retryable = False
    suggestion = 'Please login first: `modelscope login`.'


class FileIntegrityError(Exception):
    """File integrity check failed (E2020)."""
    error_code = 'E2020'
    retryable = True
    suggestion = 'File SHA256 checksum mismatch. Will retry automatically.'


class FileDownloadError(Exception):
    """File download failed (E1023)."""
    error_code = 'E1023'
    retryable = True
    suggestion = 'File download failed. Please check network connection and retry.'


class NetworkError(ConnectionError):
    """Network-level failure: connection refused, DNS resolution failed, etc. (E1020)."""
    error_code = 'E1020'
    retryable = True
    suggestion = 'Unable to connect to the server. Please check your network.'


class RateLimitError(Exception):
    """HTTP 429 -- request rate exceeded (E1021)."""
    error_code = 'E1021'
    retryable = True
    suggestion = 'Rate limit exceeded. Please reduce request frequency and retry.'

    def __init__(self, message=None, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class AccessDeniedError(PermissionError):
    """Base for access control errors (covers both 401 and 403).

    Downstream classifiers can catch this for a unified "access denied" semantic
    (e.g., E2003 in dataset context), or catch the specific subclass for finer
    granularity (E3001 / E3002).
    """
    error_code = 'E3002'
    retryable = False
    suggestion = 'Access denied. Please verify your credentials and permissions.'

    def __init__(self, message=None, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(AccessDeniedError):
    """HTTP 401 -- authentication token missing, expired, or invalid (E3001)."""
    error_code = 'E3001'
    retryable = False
    suggestion = 'Authentication failed. Please verify your token is valid.'

    def __init__(self, message=None):
        super().__init__(message, status_code=401)


class PermissionDeniedError(AccessDeniedError):
    """HTTP 403 -- authenticated but insufficient permissions (E3002)."""
    error_code = 'E3002'
    retryable = False
    suggestion = 'Permission denied. Please verify your access rights.'

    def __init__(self, message=None):
        super().__init__(message, status_code=403)


class SplitNotFoundError(ValueError):
    """Requested split does not exist in the dataset (E2004)."""
    error_code = 'E2004'
    retryable = False
    suggestion = 'The requested split does not exist.'

    def __init__(self, split, available_splits=None):
        self.split = split if isinstance(split, list) else [split]
        self.available_splits = available_splits or []
        names = ', '.join(f'"{s}"' for s in self.split)
        msg = f'Split {names} not found.'
        if self.available_splits:
            msg += f' Available splits: {self.available_splits}'
        super().__init__(msg)


class UnsupportedFormatError(ValueError):
    """Data format is not supported (E2001)."""
    error_code = 'E2001'
    retryable = False
    suggestion = 'This format is not supported. Please check the supported formats list.'

    def __init__(self, format_name=None, reason=None):
        self.format_name = format_name
        msg = reason or f'Format "{format_name}" is not supported'
        super().__init__(msg)


class CacheNotFound(Exception):
    """Exception thrown when the ModelScope cache is not found (E1022)."""
    error_code = 'E1022'
    retryable = False
    suggestion = 'Local cache directory error. Please check disk space and permissions.'

    cache_dir: Union[str, Path]

    def __init__(self, msg: str, cache_dir: Union[str, Path], *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.cache_dir = cache_dir


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the ModelScope cache-system (E1022)."""
    error_code = 'E1022'
    retryable = False
    suggestion = 'Local cache directory is corrupted. Please clear cache and retry.'


def get_request_id(response: requests.Response):
    if MODELSCOPE_REQUEST_ID in response.request.headers:
        return response.request.headers[MODELSCOPE_REQUEST_ID]
    else:
        return ''


def is_ok(rsp):
    """ Check the request is ok

    Args:
        rsp (Response): The request response body

    Returns:
       bool: `True` if success otherwise `False`.
    """
    return rsp['Code'] == HTTPStatus.OK and rsp['Success']


def _decode_response_error(response: requests.Response):
    if 'application/json' in response.headers.get('content-type', ''):
        message = response.json()
    else:
        message = response.content.decode('utf-8')
    return message


def handle_http_post_error(response, url, request_body):
    try:
        response.raise_for_status()
    except HTTPError as error:
        message = _decode_response_error(response)
        raise HTTPError(
            'Request %s with body: %s exception, '
            'Response details: %s, request id: %s' %
            (url, request_body, message, get_request_id(response))) from error


def handle_http_response(response: requests.Response,
                         logger,
                         cookies,
                         model_id,
                         raise_on_error: Optional[bool] = True) -> int:
    http_error_msg = ''
    if isinstance(response.reason, bytes):
        try:
            reason = response.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = response.reason.decode('iso-8859-1')
    else:
        reason = response.reason
    request_id = get_request_id(response)
    if 404 == response.status_code:
        http_error_msg = 'The request model: %s does not exist!' % (model_id)
    elif 403 == response.status_code:
        if cookies is None:
            http_error_msg = (
                f'Authentication token does not exist, failed to access model {model_id} '
                'which may not exist or may be private. Please login first.')

        else:
            http_error_msg = f'The authentication token is invalid, failed to access model {model_id}.'
    elif 400 <= response.status_code < 500:
        http_error_msg = u'%s Client Error: %s, Request id: %s for url: %s' % (
            response.status_code, reason, request_id, response.url)

    elif 500 <= response.status_code < 600:
        http_error_msg = u'%s Server Error: %s, Request id: %s, for url: %s' % (
            response.status_code, reason, request_id, response.url)
    if http_error_msg and raise_on_error:  # there is error.
        logger.error(http_error_msg)
        raise HTTPError(http_error_msg, response=response)
    else:
        return response.status_code


def raise_on_error(rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response

    Raises:
        RequestError: the response error message.

    Returns:
        bool: True if request is OK, otherwise raise `RequestError` exception.
    """
    if rsp['Code'] == HTTPStatus.OK:
        return True
    else:
        raise RequestError(
            rsp['Message'],
            status_code=rsp.get('Code'),
            request_id=rsp.get('RequestId'),
        )


def datahub_raise_on_error(url, rsp, http_response: requests.Response):
    """If response error, raise exception

    Args:
        url (str): The request url
        rsp (HTTPResponse): The server response.
        http_response: the origin http response.

    Raises:
        RequestError: the http request error.

    Returns:
        bool: `True` if request is OK, otherwise raise `RequestError` exception.
    """
    if rsp.get('Code') == HTTPStatus.OK:
        return True
    else:
        request_id = rsp.get('RequestId', '')
        http_status = getattr(http_response, 'status_code',
                              None) if http_response is not None else None
        status_code = http_status if http_status and http_status != 200 else rsp.get(
            'Code')
        raise RequestError(
            rsp.get('Message', 'Unknown error'),
            status_code=status_code,
            request_id=request_id,
            url=url,
        )


def raise_for_http_status(rsp):
    """Attempt to decode utf-8 first since some servers
    localize reason strings, for invalid utf-8, fall back
    to decoding with iso-8859-1.

    Args:
        rsp: The http response.

    Raises:
        HTTPError: The http error info.
    """
    http_error_msg = ''
    if isinstance(rsp.reason, bytes):
        try:
            reason = rsp.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = rsp.reason.decode('iso-8859-1')
    else:
        reason = rsp.reason
    request_id = get_request_id(rsp)
    if 404 == rsp.status_code:
        http_error_msg = 'The request resource(model or dataset) does not exist!,'
        'url: %s, reason: %s' % (rsp.url, reason)
    elif 403 == rsp.status_code:
        http_error_msg = 'Authentication token does not exist or invalid.'
    elif 400 <= rsp.status_code < 500:
        http_error_msg = u'%s Client Error: %s, Request id: %s for url: %s' % (
            rsp.status_code, reason, request_id, rsp.url)

    elif 500 <= rsp.status_code < 600:
        http_error_msg = u'%s Server Error: %s, Request id: %s, for url: %s' % (
            rsp.status_code, reason, request_id, rsp.url)

    if http_error_msg:
        req = rsp.request
        if req.method == 'POST':
            http_error_msg = u'%s, body: %s' % (http_error_msg, req.body)
        raise HTTPError(http_error_msg, response=rsp)
