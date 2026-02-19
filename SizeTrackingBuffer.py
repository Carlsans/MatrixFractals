import io


class SizeTrackingBuffer:
    """Wraps a file-like object and tracks total bytes written."""

    def __init__(self, inner=None):
        self._inner = inner or io.BytesIO()
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        n = self._inner.write(data)
        self.bytes_written += n
        return n

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._inner.seek(offset, whence)

    def tell(self) -> int:
        return self._inner.tell()