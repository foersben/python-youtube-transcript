from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

# Optional dependency – google-api-core
try:
    from google.api_core.exceptions import ResourceExhausted
except ModuleNotFoundError:  # pragma: no cover
    class ResourceExhausted(Exception):  # pylint: disable=too-few-public-methods
        """Stub replacement raised when the genuine package is unavailable."""


# Mandatory dependency – google-genai
try:
    import google.generativeai as genai
except ModuleNotFoundError as exc:
    raise ImportError(
        "Package 'google-genai' is required. "
        "Add it to your dependencies or install it manually."
    ) from exc

# Configuration                                                               #
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_API_KEY_ENV = "GOOGLE_API_KEY"
_api_key = os.getenv(_API_KEY_ENV)
if not _api_key:
    raise ValueError(
        f"Environment variable {_API_KEY_ENV} is missing. "
        "Create a '.env' file or export the key before launching the app."
    )

genai.configure(api_key=_api_key)


# Public helper                                                               #
def request_gemini(
    prompt: str,
    filepaths: Iterable[str] | None = None,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
    retry_delay: int = 20,
) -> str:
    """ Call Google Gemini with optional file uploads.

    Args:
        prompt:         The text prompt to send.
        filepaths:      An iterable of local paths to attach to the request.
        model:          Gemini model name.
        max_retries:    Maximum number of *additional* attempts if the call fails.
        retry_delay:    Seconds to wait between retries.

    Returns:
        The generated text content.

    Raises:
        RuntimeError: When all retries have been exhausted.
        ValueError:   On invalid input (e.g. non-existing files).
    """

    if filepaths:
        _validate_paths(filepaths)

    client = genai.GenerativeModel(model)
    attempt = 0
    while True:
        try:
            uploaded_files = _upload_files(filepaths)
            response = client.generate_content([prompt, *uploaded_files])

            _delete_files(uploaded_files)

            return _extract_text(response)

        except ResourceExhausted:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError("Gemini API quota exhausted and all retries failed") from None
            logger.warning("Quota exhausted – retrying in %s s (%s/%s)…",
                           retry_delay, attempt, max_retries)
            time.sleep(retry_delay)

        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"Gemini API failed after {max_retries} retries: {exc}"
                ) from exc
            logger.error("Gemini API error %s – retrying in %s s (%s/%s)",
                         exc, retry_delay, attempt, max_retries, exc_info=False)
            time.sleep(retry_delay)


def _validate_paths(paths: Iterable[str]) -> None:
    for path in paths:
        if not Path(path).is_file():
            raise ValueError(f"File not found: {path}")


def _upload_files(paths: Iterable[str] | None):
    uploaded = []
    if not paths:
        return uploaded

    for path in paths:
        # The Google GenAI SDK returns a File object.
        uploaded.append(genai.upload_file(path=path))
    return uploaded


def _delete_files(files) -> None:  # pragma: no cover
    for file in files:
        try:
            genai.delete_file(file.name)
        except Exception:  # noqa: BLE001
            # Best-effort cleanup; ignore failures.
            pass


def _extract_text(response) -> str:
    """
    The SDK sometimes exposes `response.text`, sometimes only a candidates list.
    This helper works for both.
    """
    if hasattr(response, "text") and response.text:
        return response.text.strip()

    # Fallback: take the first candidate/part
    if (getattr(response, "candidates", None)
            and response.candidates[0].content.parts):
        txt = response.candidates[0].content.parts[0].text
        return txt.strip()

    raise RuntimeError("Gemini response did not contain any text")