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
    pass


class NoValidRevisionError(Exception):
    pass


class NotExistError(Exception):
    pass


class RequestError(Exception):
    pass


class GitError(Exception):
    pass


class InvalidParameter(Exception):
    pass


class NotLoginException(Exception):
    pass


class FileIntegrityError(Exception):
    pass


class FileDownloadError(Exception):
    pass


class CacheNotFound(Exception):
    """Exception thrown when the ModelScope cache is not found."""

    cache_dir: Union[str, Path]

    def __init__(self, msg: str, cache_dir: Union[str, Path], *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.cache_dir = cache_dir


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the ModelScope cache-system."""


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
        raise RequestError(rsp['Message'])


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
        request_id = rsp['RequestId']
        raise RequestError(
            f"Url = {url}, Request id={request_id} Code = {rsp['Code']} Message = {rsp['Message']},\
                Please specify correct dataset_name and namespace.")


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


class CommitError(ValueError):
    """Raised when a commit operation fails with an HTTP error.

    Inherits ValueError for backward compatibility with existing callers
    that catch ValueError from create_commit.

    Attributes:
        http_status_code: HTTP status code from server response (e.g. 400, 500).
        url: Request URL that produced the error.
        error_detail: Parsed response body (dict or str).
        is_retryable: Whether the error is transient and can be retried.
    """

    # 4xx error messages considered transient (git ref conflicts, etc.)
    _RETRYABLE_PATTERNS = ('Could not update refs', 'try again')

    def __init__(
        self,
        message: str,
        *,
        http_status_code: int = 0,
        url: str = '',
        error_detail=None,
    ):
        super().__init__(message)
        self.http_status_code = http_status_code
        self.url = url
        self.error_detail = error_detail
        self.is_retryable = self._determine_retryability()

    def _determine_retryability(self) -> bool:
        """Determine if error is transient and retryable."""
        code = self.http_status_code
        if code >= 500 or code == 429:
            return True
        if 400 <= code < 500:
            detail_str = str(self.error_detail) if self.error_detail else ''
            return any(p in detail_str for p in self._RETRYABLE_PATTERNS)
        return False
