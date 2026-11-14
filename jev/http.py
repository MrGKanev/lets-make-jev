"""Thread-local pooled HTTP connections and bounded retries for inference requests."""
import threading
import time

import requests

_LOCAL = threading.local()
TRANSIENT = {429, 500, 502, 503, 504}


class ResponseError(ValueError):
    """The server returned a successful HTTP response with an invalid payload."""


def session():
    if not hasattr(_LOCAL, 'session'):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def post(url, *, retries=2, **kwargs):
    """Retry inference only: up to 3 attempts. A timed-out call may compute twice."""
    for attempt in range(retries + 1):
        try:
            response = session().post(url, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
        else:
            if response.status_code not in TRANSIENT or attempt == retries:
                return response
            response.close()
        time.sleep(min(.25 * 2 ** attempt, 2))


def chat_content(response):
    try:
        content = response.json()['choices'][0]['message']['content']
        if not isinstance(content, str):
            raise TypeError('content must be text')
        return content
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ResponseError('Invalid chat completion response') from exc
