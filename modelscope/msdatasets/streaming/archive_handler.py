# Copyright (c) Alibaba, Inc. and its affiliates.
"""Streaming archive abstraction layer.

Provides a registry-based mechanism to detect and wrap archive URLs
for fsspec-compatible streaming access without full download.
"""
from __future__ import annotations
import os
from typing import Dict, Optional, Tuple, Type

from typing_extensions import Protocol, runtime_checkable

from modelscope.utils.logger import get_logger

logger = get_logger()


@runtime_checkable
class ArchiveStreamHandler(Protocol):
    """Protocol for archive streaming handlers.

    Each handler declares which file extensions it supports and how to
    transform a plain URL into an fsspec-compatible streaming URL.
    """

    @classmethod
    def supported_extensions(cls) -> Tuple[str, ...]:
        """Return file extensions this handler supports (e.g. ('.zip',))."""
        ...

    @classmethod
    def wrap_url(cls, remote_url: str) -> str:
        """Wrap a remote URL into an fsspec streaming URL."""
        ...


class ZipStreamHandler:
    """Handler for ZIP archives using fsspec ZipFileSystem."""

    @classmethod
    def supported_extensions(cls) -> Tuple[str, ...]:
        return ('.zip', )

    @classmethod
    def wrap_url(cls, remote_url: str) -> str:
        """Wrap URL for zip:// fsspec access."""
        return f'zip://::{remote_url}'


class ArchiveStreamRegistry:
    """Registry that maps file extensions to their stream handlers."""

    _handlers: Dict[str, Type[ArchiveStreamHandler]] = {}

    @classmethod
    def register(cls, handler_cls: Type[ArchiveStreamHandler]) -> None:
        """Register a handler, mapping each of its supported extensions."""
        for ext in handler_cls.supported_extensions():
            ext_lower = ext.lower()
            cls._handlers[ext_lower] = handler_cls
            logger.debug(f'Registered archive stream handler for {ext_lower}')

    @classmethod
    def get_handler(cls, path: str) -> Optional[Type[ArchiveStreamHandler]]:
        """Look up a handler by file path extension."""
        _, ext = os.path.splitext(path)
        return cls._handlers.get(ext.lower())

    @classmethod
    def get_streaming_url(cls, url: str) -> Optional[str]:
        """Return wrapped streaming URL if the archive type is supported."""
        handler = cls.get_handler(url)
        if handler is None:
            return None
        return handler.wrap_url(url)

    @classmethod
    def is_streamable_archive(cls, path: str) -> bool:
        """Check whether the given path points to a streamable archive."""
        return cls.get_handler(path) is not None


# Auto-register built-in handlers on module load
ArchiveStreamRegistry.register(ZipStreamHandler)
