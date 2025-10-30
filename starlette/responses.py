import http.cookies
import json
import os
import stat
import sys
import typing
from datetime import datetime
from email.utils import format_datetime, formatdate
from functools import partial
from mimetypes import guess_type as mimetypes_guess_type
from urllib.parse import quote

import anyio

from starlette._compat import md5_hexdigest
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, Headers, MutableHeaders
from starlette.types import Receive, Scope, Send

if sys.version_info >= (3, 8):  # pragma: no cover
    from typing import Literal
else:  # pragma: no cover
    from typing_extensions import Literal

# Workaround for adding samesite support to pre 3.8 python
http.cookies.Morsel._reserved["samesite"] = "SameSite"  # type: ignore[attr-defined]


# Compatibility wrapper for `mimetypes.guess_type` to support `os.PathLike` on <py3.8
def guess_type(
    url: typing.Union[str, "os.PathLike[str]"], strict: bool = True
) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    if sys.version_info < (3, 8):  # pragma: no cover
        url = os.fspath(url)
    return mimetypes_guess_type(url, strict)


class Response:
    media_type = None
    charset = "utf-8"

    def __init__(
        self,
        content: typing.Any = None,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> None:
        self.status_code = status_code
        if media_type is not None:
            self.media_type = media_type
        self.background = background
        self.body = self.render(content)
        self.init_headers(headers)

    def render(self, content: typing.Any) -> bytes:
        if content is None:
            return b""
        if isinstance(content, bytes):
            return content
        return content.encode(self.charset)

    def init_headers(
        self, headers: typing.Optional[typing.Mapping[str, str]] = None
    ) -> None:
        if headers is None:
            raw_headers: typing.List[typing.Tuple[bytes, bytes]] = []
            populate_content_length = True
            populate_content_type = True
        else:
            raw_headers = [
                (k.lower().encode("latin-1"), v.encode("latin-1"))
                for k, v in headers.items()
            ]
            keys = [h[0] for h in raw_headers]
            populate_content_length = b"content-length" not in keys
            populate_content_type = b"content-type" not in keys

        body = getattr(self, "body", None)
        if (
            body is not None
            and populate_content_length
            and not (self.status_code < 200 or self.status_code in (204, 304))
        ):
            content_length = str(len(body))
            raw_headers.append((b"content-length", content_length.encode("latin-1")))

        content_type = self.media_type
        if content_type is not None and populate_content_type:
            if content_type.startswith("text/"):
                content_type += "; charset=" + self.charset
            raw_headers.append((b"content-type", content_type.encode("latin-1")))

        self.raw_headers = raw_headers

    @property
    def headers(self) -> MutableHeaders:
        if not hasattr(self, "_headers"):
            self._headers = MutableHeaders(raw=self.raw_headers)
        return self._headers

    def set_cookie(
        self,
        key: str,
        value: str = "",
        max_age: typing.Optional[int] = None,
        expires: typing.Optional[typing.Union[datetime, str, int]] = None,
        path: str = "/",
        domain: typing.Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: typing.Optional[Literal["lax", "strict", "none"]] = "lax",
    ) -> None:
        cookie: "http.cookies.BaseCookie[str]" = http.cookies.SimpleCookie()
        cookie[key] = value
        if max_age is not None:
            cookie[key]["max-age"] = max_age
        if expires is not None:
            if isinstance(expires, datetime):
                cookie[key]["expires"] = format_datetime(expires, usegmt=True)
            else:
                cookie[key]["expires"] = expires
        if path is not None:
            cookie[key]["path"] = path
        if domain is not None:
            cookie[key]["domain"] = domain
        if secure:
            cookie[key]["secure"] = True
        if httponly:
            cookie[key]["httponly"] = True
        if samesite is not None:
            assert samesite.lower() in [
                "strict",
                "lax",
                "none",
            ], "samesite must be either 'strict', 'lax' or 'none'"
            cookie[key]["samesite"] = samesite
        cookie_val = cookie.output(header="").strip()
        self.raw_headers.append((b"set-cookie", cookie_val.encode("latin-1")))

    def delete_cookie(
        self,
        key: str,
        path: str = "/",
        domain: typing.Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: typing.Optional[Literal["lax", "strict", "none"]] = "lax",
    ) -> None:
        self.set_cookie(
            key,
            max_age=0,
            expires=0,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        await send({"type": "http.response.body", "body": self.body})

        if self.background is not None:
            await self.background()


class HTMLResponse(Response):
    media_type = "text/html"


class PlainTextResponse(Response):
    media_type = "text/plain"


class JSONResponse(Response):
    media_type = "application/json"

    def __init__(
        self,
        content: typing.Any,
        status_code: int = 200,
        headers: typing.Optional[typing.Dict[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> None:
        super().__init__(content, status_code, headers, media_type, background)

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


class RedirectResponse(Response):
    def __init__(
        self,
        url: typing.Union[str, URL],
        status_code: int = 307,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> None:
        super().__init__(
            content=b"", status_code=status_code, headers=headers, background=background
        )
        self.headers["location"] = quote(str(url), safe=":/%#?=@[]!$&'()*+,;")


Content = typing.Union[str, bytes]
SyncContentStream = typing.Iterator[Content]
AsyncContentStream = typing.AsyncIterable[Content]
ContentStream = typing.Union[AsyncContentStream, SyncContentStream]


class StreamingResponse(Response):
    body_iterator: AsyncContentStream

    def __init__(
        self,
        content: ContentStream,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
    ) -> None:
        if isinstance(content, typing.AsyncIterable):
            self.body_iterator = content
        else:
            self.body_iterator = iterate_in_threadpool(content)
        self.status_code = status_code
        self.media_type = self.media_type if media_type is None else media_type
        self.background = background
        self.init_headers(headers)

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        async for chunk in self.body_iterator:
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as task_group:

            async def wrap(func: "typing.Callable[[], typing.Awaitable[None]]") -> None:
                await func()
                task_group.cancel_scope.cancel()

            task_group.start_soon(wrap, partial(self.stream_response, send))
            await wrap(partial(self.listen_for_disconnect, receive))

        if self.background is not None:
            await self.background()


class MalformedRangeHeader(Exception):
    def __init__(self, content: str = "Malformed range header.") -> None:
        self.content = content


class RangeNotSatisfiable(Exception):
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size


class FileResponse(Response):
    chunk_size = 64 * 1024

    def __init__(
        self,
        path: typing.Union[str, "os.PathLike[str]"],
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
        filename: typing.Optional[str] = None,
        stat_result: typing.Optional[os.stat_result] = None,
        method: typing.Optional[str] = None,
        content_disposition_type: str = "attachment",
    ) -> None:
        self.path = path
        self.status_code = status_code
        self.filename = filename
        self.send_header_only = method is not None and method.upper() == "HEAD"
        if media_type is None:
            media_type = guess_type(filename or path)[0] or "text/plain"
        self.media_type = media_type
        self.background = background
        self.init_headers(headers)
        self.headers.setdefault("accept-ranges", "bytes")
        if self.filename is not None:
            content_disposition_filename = quote(self.filename)
            if content_disposition_filename != self.filename:
                content_disposition = "{}; filename*=utf-8''{}".format(
                    content_disposition_type, content_disposition_filename
                )
            else:
                content_disposition = '{}; filename="{}"'.format(
                    content_disposition_type, self.filename
                )
            self.headers.setdefault("content-disposition", content_disposition)
        self.stat_result = stat_result
        if stat_result is not None:
            self.set_stat_headers(stat_result)

    def set_stat_headers(self, stat_result: os.stat_result) -> None:
        content_length = str(stat_result.st_size)
        last_modified = formatdate(stat_result.st_mtime, usegmt=True)
        etag_base = str(stat_result.st_mtime) + "-" + str(stat_result.st_size)
        etag = '"{}"'.format(md5_hexdigest(etag_base.encode(), usedforsecurity=False))

        self.headers.setdefault("content-length", content_length)
        self.headers.setdefault("last-modified", last_modified)
        self.headers.setdefault("etag", etag)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.stat_result is None:
            try:
                stat_result = await anyio.to_thread.run_sync(os.stat, self.path)
                self.set_stat_headers(stat_result)
            except FileNotFoundError:
                raise RuntimeError(f"File at path {self.path} does not exist.")
            else:
                mode = stat_result.st_mode
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f"File at path {self.path} is not a file.")
        else:
            stat_result = self.stat_result

        headers = Headers(scope=scope)
        http_range = headers.get("range")
        http_if_range = headers.get("if-range")

        if http_range is None or (http_if_range is not None and not self._should_use_range(http_if_range)):
            await self._handle_simple(send, self.send_header_only)
        else:
            try:
                ranges = self._parse_range_header(http_range, stat_result.st_size)
            except MalformedRangeHeader as exc:
                return await PlainTextResponse(exc.content, status_code=400)(scope, receive, send)
            except RangeNotSatisfiable as exc:
                response = PlainTextResponse(
                    status_code=416,
                    headers={"Content-Range": "*/{0}".format(exc.max_size)}
                )
                return await response(scope, receive, send)

            if len(ranges) == 1:
                start, end = ranges[0]
                await self._handle_single_range(send, start, end, stat_result.st_size, self.send_header_only)
            else:
                await self._handle_multiple_ranges(send, ranges, stat_result.st_size, self.send_header_only)

        if self.background is not None:
            await self.background()

    async def _handle_simple(self, send: Send, send_header_only: bool) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            async with await anyio.open_file(self.path, mode="rb") as file:
                more_body = True
                while more_body:
                    chunk = await file.read(self.chunk_size)
                    more_body = len(chunk) == self.chunk_size
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": more_body,
                        }
                    )

    async def _handle_single_range(
        self, send: Send, start: int, end: int, file_size: int, send_header_only: bool
    ) -> None:
        self.headers["content-range"] = "bytes {0}-{1}/{2}".format(start, end - 1, file_size)
        self.headers["content-length"] = str(end - start)
        await send({"type": "http.response.start", "status": 206, "headers": self.raw_headers})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            async with await anyio.open_file(self.path, mode="rb") as file:
                await file.seek(start)
                more_body = True
                while more_body:
                    chunk_size = min(self.chunk_size, end - start)
                    chunk = await file.read(chunk_size)
                    start += len(chunk)
                    more_body = len(chunk) == self.chunk_size and start < end
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": more_body,
                        }
                    )

    async def _handle_multiple_ranges(
        self, send: Send, ranges: typing.List[typing.Tuple[int, int]], file_size: int, send_header_only: bool
    ) -> None:
        import random
        import string

        boundary = "".join(random.choices(string.ascii_letters + string.digits, k=13))
        content_type = self.media_type
        content_length, generate_header = self.generate_multipart(
            ranges, boundary, file_size, content_type
        )

        self.headers["content-type"] = "multipart/byteranges; boundary={0}".format(boundary)
        self.headers["content-length"] = str(content_length)
        await send({"type": "http.response.start", "status": 206, "headers": self.raw_headers})

        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            async with await anyio.open_file(self.path, mode="rb") as file:
                for start, end in ranges:
                    await file.seek(start)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": generate_header(start, end),
                            "more_body": True,
                        }
                    )
                    remaining = end - start
                    while remaining > 0:
                        chunk_size = min(self.chunk_size, remaining)
                        chunk = await file.read(chunk_size)
                        remaining -= len(chunk)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": True,
                            }
                        )

                await send(
                    {
                        "type": "http.response.body",
                        "body": "--{0}--\n".format(boundary).encode("latin-1"),
                        "more_body": False,
                    }
                )

    def _should_use_range(self, http_if_range: str) -> bool:
        return http_if_range == self.headers["last-modified"] or http_if_range == self.headers["etag"]

    @classmethod
    def _parse_range_header(cls, http_range: str, file_size: int) -> typing.List[typing.Tuple[int, int]]:
        ranges = []  # type: typing.List[typing.Tuple[int, int]]
        try:
            units, range_ = http_range.split("=", 1)
        except ValueError:
            raise MalformedRangeHeader()

        units = units.strip().lower()

        if units != "bytes":
            raise MalformedRangeHeader("Only support bytes range")

        ranges = cls._parse_ranges(range_, file_size)

        if len(ranges) == 0:
            raise MalformedRangeHeader("Range header: range must be requested")

        if any(not (0 <= start < file_size) for start, _ in ranges):
            raise RangeNotSatisfiable(file_size)

        if any(start > end for start, end in ranges):
            raise MalformedRangeHeader("Range header: start must be less than end")

        if len(ranges) == 1:
            return ranges

        # Merge ranges
        result = []  # type: typing.List[typing.Tuple[int, int]]
        for start, end in ranges:
            for p in range(len(result)):
                p_start, p_end = result[p]
                if start > p_end:
                    continue
                elif end < p_start:
                    result.insert(p, (start, end))
                    break
                else:
                    result[p] = (min(start, p_start), max(end, p_end))
                    break
            else:
                result.append((start, end))

        return result

    @classmethod
    def _parse_ranges(cls, range_: str, file_size: int) -> typing.List[typing.Tuple[int, int]]:
        ranges = []  # type: typing.List[typing.Tuple[int, int]]

        for part in range_.split(","):
            part = part.strip()

            # If the range is empty or a single dash, we ignore it.
            if not part or part == "-":
                continue

            # If the range is not in the format "start-end", we ignore it.
            if "-" not in part:
                continue

            start_str, end_str = part.split("-", 1)
            start_str = start_str.strip()
            end_str = end_str.strip()

            try:
                start = int(start_str) if start_str else file_size - int(end_str)
                end = int(end_str) + 1 if start_str and end_str and int(end_str) < file_size else file_size
                ranges.append((start, end))
            except ValueError:
                # If the range is not numeric, we ignore it.
                continue

        return ranges

    def generate_multipart(
        self,
        ranges: typing.List[typing.Tuple[int, int]],
        boundary: str,
        max_size: int,
        content_type: str,
    ) -> typing.Tuple[int, typing.Callable[[int, int], bytes]]:
        """
        Multipart response headers generator.

        --{boundary}\\n
        Content-Type: {content_type}\\n
        Content-Range: bytes {start}-{end-1}/{max_size}\\n
        \\n
        ..........content...........\\n
        --{boundary}\\n
        Content-Type: {content_type}\\n
        Content-Range: bytes {start}-{end-1}/{max_size}\\n
        \\n
        ..........content...........\\n
        --{boundary}--\\n
        """
        boundary_len = len(boundary)
        static_header_part_len = 44 + boundary_len + len(content_type) + len(str(max_size))
        content_length = sum(
            (len(str(start)) + len(str(end - 1)) + static_header_part_len)  # Headers
            + (end - start)  # Content
            for start, end in ranges
        ) + (
            5 + boundary_len  # --boundary--\n
        )
        return (
            content_length,
            lambda start, end: (
                "--{0}\n"
                "Content-Type: {1}\n"
                "Content-Range: bytes {2}-{3}/{4}\n"
                "\n"
            ).format(boundary, content_type, start, end - 1, max_size).encode("latin-1"),
        )
