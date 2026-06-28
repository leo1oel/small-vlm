"""Streaming multimodal (image/audio) training data from Azure Blob via
Megatron-Energon — dataset type "energon" in the Hydra config.

Ported from exp/azure_streaming/energon_data.py (verified end-to-end there:
streaming, multi-worker sharding, exact resume, MSC caching, weighted blends,
FLAC decode, mixed image+audio batches). The task encoder here additionally
runs the FULL per-sample preprocessing using the shared pure functions from
dataset.py, so this path and the local-json path converge on the exact same
collator output / model contract.

Usage (wired by the trainer when dataset.type == "energon"):

    loader = build_energon_train_loader(
        dataset_config, processor, data_args, batch_size=8
    )
    for batch in loader:          # dict: input_ids/labels/attention_mask/
        ...                       #       images/image_position_ids/audios
    state = loader.save_state_rank()      # exact-resume support
    loader.restore_state_rank(state)

Dataset layout on blob (per folder inside the container):
    <folder>/train.jsonl     one JSON object per line:
        {"id": ..., "source": ..., "messages": [{"role": ..., "content":
            "text" | [{"type": "text", "text": ...},
                      {"type": "image", "path": "<relative path>"},
                      {"type": "audio", "path": "<relative path>", ...}]}]}
    media files at those relative paths, rooted at the jsonl's directory.

Credentials: AZURE_SAS_TOKEN (a full SAS URL) in the environment or in a .env
file (cwd, repo root, or the repo's parent). On a cluster, export the derived
AZURE_BLOB_ENDPOINT / AZURE_STORAGE_CONNECTION_STRING directly instead — the
bootstrap never overwrites existing variables.

Local state:
    jsonl copies + indexes   $VLM_ENERGON_DATA_DIR, else $VLM_DATA_ROOT/energon-jsonl,
                             else ~/.cache/vlm/energon-jsonl
    MSC range cache          $MSC_CACHE_DIR, else ~/.cache/vlm/msc-cache
"""

import bisect
import concurrent.futures
import fcntl
import hashlib
import io
import logging
import os
import re
import threading
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

from ..config import validate_dataset_config
from ..models.gen_image import make_position_ids, pixels_to_patches
from ..models.image_processing_raw import RawImageProcessor
from .data_arguments import DataArguments
from .dataset import (
    MEDIA_PLACEHOLDER_MARK_L,
    MEDIA_PLACEHOLDER_MARK_R,
    DataCollatorForSupervisedDataset,
    apply_image_position,
    check_audio_template_supported,
    inject_missing_media_tokens,
    inject_query_placeholders,
    load_audio_frames,
    make_dummy_audio_frames,
    make_dummy_image_entry,
    preprocess,
    process_classic_image,
    process_raw_image,
)

log: logging.Logger = logging.getLogger(name=__name__)

# ---------------------------------------------------------------------------
# 0. Environment bootstrap — must run before any msc:// access
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _MODULE_DIR.parents[2]  # src/vlm/data -> small-vlm

PROFILE = os.environ.get("VLM_MSC_PROFILE", "azure")
CONTAINER = os.environ.get("VLM_MSC_CONTAINER", "data")
CRUDE_TYPE = "mm_chat_jsonl"


def _load_dotenv(path: Path) -> dict:
    env = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _find_sas_token() -> str:
    if os.environ.get("AZURE_SAS_TOKEN"):
        return os.environ["AZURE_SAS_TOKEN"]
    for candidate in (Path.cwd() / ".env", _REPO_DIR / ".env", _REPO_DIR.parent / ".env"):
        token = _load_dotenv(candidate).get("AZURE_SAS_TOKEN", "")
        if token:
            return token
    return ""


def _patch_msc_poisoned_file_hang() -> None:
    """multi-storage-client (<= 0.49.0, latest as of 2026-06): ObjectFile's
    __init__ can raise (e.g. an Azure HEAD timeout on the fat-tail) AFTER
    creating the _download_complete event but BEFORE any code path sets it.
    Every later touch of the broken object — including GC: __del__ -> close()
    -> closed — then waits on the event forever, freezing the dataloader
    worker and, through the DataLoader's in-order result queue, the whole
    training step (observed live: all 4 workers parked in file.py `closed`).
    Bound the wait and report the poisoned object as closed instead."""
    try:
        from multistorageclient.file import ObjectFile
    except ImportError:  # streaming extras not installed; nothing to patch
        return

    def closed(self: Any) -> bool:  # mirrors upstream, with a bounded wait
        if self.readable():
            if not self._download_complete.wait(timeout=300):
                log.warning(
                    "MSC ObjectFile: open never completed for %s — treating as closed",
                    getattr(self, "_remote_path", "?"),
                )
                return True
        return self._file.closed

    ObjectFile.closed = property(closed)


def _patch_msc_cold_lock_path() -> None:
    """multi-storage-client (<= 0.49.0): CacheManager.acquire_lock builds the
    download lock path as ``os.path.dirname(os.path.join(cache_dir, key))`` where
    ``key`` is the ABSOLUTE remote object path (e.g. ``/data/yiming/.../.nv-meta/
    .info.json``). ``os.path.join`` DROPS the cache-dir prefix when the second
    arg is absolute, so the lock lands at ``/<container>/...`` — a root path — and
    a COLD cache-miss download fails with PermissionError (``/data`` is a
    root-owned mount). Cache HITS use a slash-stripped key, so only the cold
    download breaks; this bites every fresh run that streams a prepared
    CrudeWebdataset (``dataset.wds_path``) whose ``.nv-meta`` is read in text mode
    by energon's ``get_dataset_info`` (data/ds-config-2stage). Strip the leading
    slash so the lock lands under the cache dir, mirroring the cached-file path."""
    try:
        import os.path as osp

        from filelock import FileLock
        from multistorageclient.cache import CacheManager
    except ImportError:  # streaming extras not installed; nothing to patch
        return

    def acquire_lock(self: Any, key: str) -> Any:
        file_dir = osp.dirname(osp.join(self._get_cache_dir(), key.lstrip("/")))
        lock_file = osp.join(file_dir, f".{osp.basename(key)}.lock")
        return FileLock(lock_file, timeout=self.DEFAULT_FILE_LOCK_TIMEOUT)

    CacheManager.acquire_lock = acquire_lock


def _bootstrap_env() -> None:
    """Derive MSC/Azure env vars from AZURE_SAS_TOKEN (a full SAS URL). Never
    overwrites existing variables, so a cluster can provide them externally.
    Missing credentials are tolerated here (config paths are still set);
    build_energon_train_loader fails loudly when streaming actually starts."""
    # The azure SDK's http_logging_policy dumps full request/response headers
    # for EVERY blob GET — at training rates that is tens of thousands of
    # multi-line log records through the rich formatter onto a shared FS,
    # measurable overhead on the media-fetch hot path. Errors still raise
    # (and log at WARNING+), so nothing actionable is lost.
    logging.getLogger("azure").setLevel(logging.WARNING)
    _patch_msc_poisoned_file_hang()
    _patch_msc_cold_lock_path()
    cache_dir = os.environ.setdefault(
        "MSC_CACHE_DIR", str(Path.home() / ".cache" / "vlm" / "msc-cache")
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MSC_CONFIG", str(_MODULE_DIR / "msc_config.yaml"))

    sas_url = _find_sas_token().strip().strip('"').strip("'")
    if "?" not in sas_url:
        return  # no credentials found; _require_credentials reports later
    endpoint, query = sas_url.split("?", 1)
    endpoint = endpoint.rstrip("/")
    os.environ.setdefault("AZURE_BLOB_ENDPOINT", endpoint)
    os.environ.setdefault(
        "AZURE_STORAGE_CONNECTION_STRING",
        f"BlobEndpoint={endpoint};SharedAccessSignature={query}",
    )


def _require_credentials() -> None:
    if not (
        os.environ.get("AZURE_BLOB_ENDPOINT") and os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    ):
        raise RuntimeError(
            "Azure credentials not configured: provide AZURE_SAS_TOKEN (a full SAS "
            "URL) in the environment or a .env file (cwd, repo root, or repo "
            "parent), or set AZURE_BLOB_ENDPOINT + AZURE_STORAGE_CONNECTION_STRING."
        )


_bootstrap_env()

# Streaming-only optional deps (megatron-energon + multi-storage-client). The
# runtime import is guarded so environments without them — json-only training,
# CI test collection (pytest imports every src module) — can still import this
# module; building the loader without the deps raises a helpful error instead.
# The type checker always follows the real-import branch.
if TYPE_CHECKING:
    import multistorageclient as msc  # noqa: E402  # pyright: ignore[reportMissingImports]  (needs MSC_CONFIG set above)
    from megatron.energon import (  # noqa: E402  # pyright: ignore[reportMissingImports]
        Cooker,
        FileStore,
        Sample,
        TaskEncoder,
        WorkerConfig,
        basic_sample_keys,
        get_savable_loader,
        get_train_dataset,
        stateless,
    )
    from megatron.energon.epathlib import (  # noqa: E402  # pyright: ignore[reportMissingImports]
        EPath,
    )
    from megatron.energon.flavors.jsonl.ijsonl import (  # noqa: E402  # pyright: ignore[reportMissingImports]
        IJsonlIndexWriter,
    )
    from megatron.energon.savable_loader import (  # noqa: E402  # pyright: ignore[reportMissingImports]
        SavableDataLoader,
    )

    _streaming_import_error: ImportError | None = None
else:
    try:
        import multistorageclient as msc  # noqa: E402
        from megatron.energon import (  # noqa: E402
            Cooker,
            FileStore,
            Sample,
            TaskEncoder,
            WorkerConfig,
            basic_sample_keys,
            get_savable_loader,
            get_train_dataset,
            stateless,
        )
        from megatron.energon.epathlib import EPath  # noqa: E402
        from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexWriter  # noqa: E402
        from megatron.energon.savable_loader import SavableDataLoader  # noqa: E402

        _streaming_import_error = None
    except ImportError as _import_error:  # pragma: no cover - only without the extras
        _streaming_import_error = _import_error

        # Minimal runtime stand-ins so the module-level classes/annotations
        # below still define; nothing streaming ever *executes* without the
        # real deps, because build_energon_train_loader raises first.
        msc = None
        FileStore = WorkerConfig = EPath = IJsonlIndexWriter = SavableDataLoader = object
        basic_sample_keys = get_savable_loader = get_train_dataset = None

        class Sample:
            pass

        class TaskEncoder:
            pass

        def Cooker(*args, **kwargs):  # noqa: N802  (mirrors energon's name)
            return None

        def stateless(fn):
            return fn

# ---------------------------------------------------------------------------
# 1. Remote layout & local copies
# ---------------------------------------------------------------------------


def remote_url(*parts: str) -> str:
    """Join path parts into an msc:// URL inside the data container."""
    clean = [p.strip("/") for p in parts if p and p.strip("/")]
    return f"msc://{PROFILE}/{CONTAINER}/" + "/".join(clean)


def find_jsonl(folder: str, jsonl_name: str = "train.jsonl") -> EPath:
    jsonl = EPath(remote_url(folder, jsonl_name))
    if not jsonl.is_file():
        raise FileNotFoundError(f"No {jsonl_name} found at {jsonl}")
    return jsonl


def is_prepared(jsonl: EPath) -> bool:
    """True if the `.jsonl.idx` side-car exists and matches the jsonl's size."""
    idx = jsonl.with_suffix(".jsonl.idx")
    if not idx.is_file():
        return False
    with idx.open("rb") as f:
        f.seek(idx.size() - 8)
        last_offset = int.from_bytes(f.read(8), "little")
    return last_offset == jsonl.size()


def local_data_dir() -> Path:
    explicit = os.environ.get("VLM_ENERGON_DATA_DIR")
    if explicit:
        return Path(explicit)
    data_root = os.environ.get("VLM_DATA_ROOT")
    if data_root:
        return Path(data_root) / "energon-jsonl"
    return Path.home() / ".cache" / "vlm" / "energon-jsonl"


def local_jsonl_path(folder: str, jsonl_name: str = "train.jsonl") -> Path:
    return local_data_dir() / folder.strip("/") / jsonl_name


def download_and_index(folder: str, jsonl_name: str = "train.jsonl", max_threads: int = 12) -> Path:
    """Download `<folder>/<jsonl_name>` to the local data dir and build its
    index in the same pass (resumable). Media stays remote — only the jsonl
    (metadata) is localized, eliminating per-sample WAN reads for text."""
    dest = local_jsonl_path(folder, jsonl_name)
    if dest.is_file() and is_prepared(EPath(str(dest))):
        log.info(f"{dest} already downloaded and indexed.")
        return dest
    src = find_jsonl(folder, jsonl_name)
    log.info(f"downloading {src} -> {dest} (+ index in the same pass)...")
    count = download_jsonl(str(src), dest, max_threads=max_threads)
    log.info(f"done: {count} samples")
    return dest


def prepare_jsonl(jsonl: EPath, max_threads: int = 12) -> int:
    """Build the byte-offset index next to a REMOTE jsonl (needs container
    write access). Prefer download_and_index for big files."""
    return build_jsonl_index(jsonl, max_threads=max_threads)


# ---------------------------------------------------------------------------
# 2. Index building / downloading (parallel ranged reads)
#
# Why not energon's own `prepare`? Its readline() scan is incompatible with
# MSC's cache (RemoteFileReader.readline raises; cache off = whole file into
# RAM). This scanner produces byte-identical .idx output (verified against
# energon's, incl. multi-chunk, empty-line and no-trailing-newline cases).
# ---------------------------------------------------------------------------

CHUNK_SIZE = 64 * 1024 * 1024

#: Abort if no chunk completes for this long (Azure throttling can hang
#: requests indefinitely). A rerun resumes from the .chunks side-car.
STALL_TIMEOUT = 240


class DownloadStalledError(RuntimeError):
    """No chunk completed within STALL_TIMEOUT — rerun to resume."""


def _iter_completed(futures: Any, stall_timeout: float = STALL_TIMEOUT):
    """as_completed() with a gap watchdog instead of hanging forever."""
    remaining = set(futures)
    while remaining:
        done, remaining = concurrent.futures.wait(
            remaining, timeout=stall_timeout, return_when=FIRST_COMPLETED
        )
        if not done:
            raise DownloadStalledError(
                f"no chunk completed in {stall_timeout}s ({len(remaining)} left) — "
                "network stalled; rerun the same command to resume"
            )
        yield from done


def _bulk_client(url: str) -> Any:
    """(client, path) for bulk scans, preferring the cache-free `<profile>_nocache`
    twin: ObjectFile's disable_read_cache only detaches the file-level cache;
    storage_client.read still writes every range into the cache — churning the
    training cache at ~10x slowdown for one-pass bulk reads."""
    if url.startswith("msc://"):
        profile, _, rest = url[len("msc://") :].partition("/")
        try:
            return msc.resolve_storage_client(f"msc://{profile}_nocache/{rest}")
        except Exception:
            pass
    return msc.resolve_storage_client(url)


def _read_chunk_with_retry(
    client: Any, path: str, offset: int, size: int, attempts: int = 3
) -> bytes:
    for attempt in range(attempts):
        try:
            with client.open(path, "rb", disable_read_cache=True, prefetch_file=False) as f:
                f.seek(offset)
                data = f.read(size)
            # A silent short read would drop newlines and corrupt the index.
            assert len(data) == size, f"short read {len(data)} != {size}"
            return data
        except (AssertionError, OSError) as e:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
            log.warning(f"retrying chunk at offset {offset} after: {e}")
    raise AssertionError("unreachable")


def _scan_chunk(client: Any, path: str, chunk_idx: int, chunk_size: int, total: int) -> tuple:
    """(chunk_idx, absolute offsets of newline bytes within this chunk)."""
    offset = chunk_idx * chunk_size
    size = min(chunk_size, total - offset)
    data = _read_chunk_with_retry(client, path, offset, size)
    newlines = []
    pos = data.find(b"\n")
    while pos != -1:
        newlines.append(offset + pos)
        pos = data.find(b"\n", pos + 1)
    return chunk_idx, newlines


class _OrderedIndexFlusher:
    """Feeds line-start offsets to an IJsonlIndexWriter strictly in file order
    while chunks complete out of order. Skips empty lines (mirroring energon's
    preparator)."""

    def __init__(self, iw: IJsonlIndexWriter, total: int):
        self.iw: Any = iw
        self.total: int = total
        self.pending: dict[int, list[int]] = {}
        self.next_chunk: int = 0
        self.prev_line_start: int = 0
        self.count: int = 0

    def add(self, chunk_idx: int, newlines: list) -> None:
        self.pending[chunk_idx] = newlines
        while self.next_chunk in self.pending:
            for nl in self.pending.pop(self.next_chunk):
                if nl + 1 - self.prev_line_start > 1:
                    self.iw.append(self.prev_line_start)
                    self.count += 1
                self.prev_line_start = nl + 1
            self.next_chunk += 1

    def finish(self) -> int:
        if self.prev_line_start < self.total:  # file without trailing newline
            self.iw.append(self.prev_line_start)
            self.count += 1
        self.iw.append(self.total)
        return self.count


def build_jsonl_index(
    jsonl: EPath | str,
    chunk_size: int = CHUNK_SIZE,
    max_threads: int = 12,
    progress_every: int = 8,
) -> int:
    """Build `<name>.jsonl.idx` next to the (remote or local) jsonl."""
    jsonl = EPath(str(jsonl))
    client, path = _bulk_client(str(jsonl))
    total = client.info(path).content_length
    n_chunks = (total + chunk_size - 1) // chunk_size

    stale_tmp = jsonl.with_suffix(".jsonl.idx.tmp")
    if stale_tmp.is_file():
        stale_tmp.unlink()

    iw = IJsonlIndexWriter(jsonl)
    try:
        flusher = _OrderedIndexFlusher(iw, total)
        done_bytes = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_threads) as ex:
            futures = [
                ex.submit(_scan_chunk, client, path, i, chunk_size, total) for i in range(n_chunks)
            ]
            for fut in _iter_completed(futures):
                chunk_idx, newlines = fut.result()
                flusher.add(chunk_idx, newlines)
                done_bytes += min(chunk_size, total - chunk_idx * chunk_size)
                if progress_every and (chunk_idx + 1) % progress_every == 0:
                    dt = time.perf_counter() - t0
                    rate = done_bytes / dt / 1e6
                    eta = (total - done_bytes) / max(rate * 1e6, 1)
                    log.info(
                        f"scanned {done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB "
                        f"({rate:.0f} MB/s, ETA {eta / 60:.1f} min)"
                    )
        count = flusher.finish()
        iw.close(finalize=True)
    except BaseException:
        iw.close(finalize=False)
        raise
    return count


def download_jsonl(
    jsonl_url: str,
    dest: Path | str,
    chunk_size: int = CHUNK_SIZE,
    max_threads: int = 12,
    progress_every: int = 8,
) -> int:
    """Download a remote jsonl to `dest` AND build its local `.jsonl.idx` in the
    same pass (one WAN transfer yields both files).

    Chunk-resumable: completed chunks + the remote etag live in `<dest>.chunks`;
    an etag change discards the resume state (otherwise old/new chunks would be
    spliced into a silently corrupt file). Concurrent invocations on the same
    dest are rejected via `<dest>.lock`. After a hard crash, a hidden
    `.<random>` temp from the index writer may remain in the dest dir; it is
    harmless to delete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client, path = _bulk_client(str(jsonl_url))
    info = client.info(path)
    total = info.content_length
    etag = str(getattr(info, "etag", None) or "")
    n_chunks = (total + chunk_size - 1) // chunk_size

    lock_file = dest.with_name(dest.name + ".lock")
    lock_fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise RuntimeError(f"another download of {dest} is already running ({lock_file})") from None

    progress_file = dest.with_name(dest.name + ".chunks")
    done_chunks = set()
    if dest.is_file() and dest.stat().st_size == total and progress_file.is_file():
        saved_etag = None
        for line in progress_file.read_text().splitlines():
            if line.startswith("#etag="):
                saved_etag = line[len("#etag=") :]
            elif line.strip():
                done_chunks.add(int(line))
        if saved_etag is not None and etag and saved_etag != etag:
            log.warning(f"remote etag changed ({saved_etag} -> {etag}), restarting download")
            done_chunks = set()
            progress_file.unlink(missing_ok=True)
        else:
            if saved_etag is None:
                log.warning("legacy .chunks side-car without etag; trusting size match")
            log.info(f"resuming: {len(done_chunks)}/{n_chunks} chunks already downloaded")
    else:
        progress_file.unlink(missing_ok=True)

    stale_tmp = Path(str(dest.with_suffix(".jsonl.idx")) + ".tmp")
    stale_tmp.unlink(missing_ok=True)

    fd = os.open(dest, os.O_RDWR | os.O_CREAT, 0o644)
    progress_lock = threading.Lock()
    try:
        os.ftruncate(fd, total)
        if not progress_file.is_file():
            progress_file.write_text(f"#etag={etag}\n")

        def fetch(i: int) -> tuple:
            offset = i * chunk_size
            size = min(chunk_size, total - offset)
            if i in done_chunks:
                data = os.pread(fd, size, offset)
            else:
                data = _read_chunk_with_retry(client, path, offset, size)
                os.pwrite(fd, data, offset)
                with progress_lock:
                    with open(progress_file, "a") as pf:
                        pf.write(f"{i}\n")
            newlines = []
            pos = data.find(b"\n")
            while pos != -1:
                newlines.append(offset + pos)
                pos = data.find(b"\n", pos + 1)
            return i, newlines

        iw = IJsonlIndexWriter(EPath(str(dest)))
        try:
            flusher = _OrderedIndexFlusher(iw, total)
            done_bytes = 0
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                futures = [ex.submit(fetch, i) for i in range(n_chunks)]
                for fut in _iter_completed(futures):
                    chunk_idx, newlines = fut.result()
                    flusher.add(chunk_idx, newlines)
                    done_bytes += min(chunk_size, total - chunk_idx * chunk_size)
                    if progress_every and (chunk_idx + 1) % progress_every == 0:
                        dt = time.perf_counter() - t0
                        rate = done_bytes / dt / 1e6
                        eta = (total - done_bytes) / max(rate * 1e6, 1)
                        log.info(
                            f"{done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB "
                            f"({rate:.0f} MB/s, ETA {eta / 60:.1f} min)"
                        )
            count = flusher.finish()
            iw.close(finalize=True)
        except BaseException:
            iw.close(finalize=False)
            raise
    finally:
        os.close(fd)
        os.close(lock_fd)  # releases the flock
        lock_file.unlink(missing_ok=True)
    progress_file.unlink(missing_ok=True)
    return count


# ---------------------------------------------------------------------------
# 3. Sample type, cooker, task encoder
# ---------------------------------------------------------------------------


@dataclass
class MMChatRawSample(Sample):  # pyright: ignore[reportUntypedBaseClass]
    """Cooker output: media as *compressed bytes*.

    Energon's pipeline is cook -> shuffle buffer -> encode_sample -> batch, and
    the shuffle buffer holds cooked samples. Keeping bytes (~0.5MB/image)
    instead of decoded arrays keeps a large shuffle buffer affordable
    (10k buffer ~ 5GB, not ~60GB)."""

    #: Original message list; media items keep their "path" (+ metadata).
    messages: list[dict]
    #: Compressed media file contents, in order of appearance over all messages.
    image_bytes: list[bytes]
    audio_bytes: list[bytes]
    #: Dataset source tag (e.g. "ocr", "AbstractTTS_IEMOCAP"), if present.
    source: str | None = None


@stateless  # pyright: ignore[reportUntypedFunctionDecorator]
def cook_mm_chat(sample: dict, media_root: FileStore) -> MMChatRawSample:
    rec = sample["json"]  # parsed dict (DefaultCrudeJsonlDatasetFactory does json.loads)

    image_bytes, audio_bytes = [], []
    for msg in rec.get("messages", ()):
        content = msg.get("content")
        if not isinstance(content, list):
            continue  # plain-string content => text-only message
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                image_bytes.append(media_root.get(item["path"], sample))
            elif item.get("type") == "audio":
                audio_bytes.append(media_root.get(item["path"], sample))

    return MMChatRawSample(
        **basic_sample_keys(sample),
        messages=rec.get("messages", []),
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        source=rec.get("source"),
    )


_ROLE_MAP = {"user": "human", "human": "human", "assistant": "gpt", "gpt": "gpt"}

# Empty reasoning block some distilled caption sets (e.g. Bee Stage-1) prepend
# to every assistant turn. Only a whitespace-bodied block is boilerplate —
# real reasoning content must never be stripped.
_EMPTY_THINK_RE = re.compile(r"^<think>\s*</think>\s*")


def _marked_media_placeholder(token: str) -> str:
    """Wrap a GENERATED media placeholder in the private-use sentinels so
    inject_missing_media_tokens can tell it apart from a literal token quoted in
    user text (#11)."""
    return MEDIA_PLACEHOLDER_MARK_L + token + MEDIA_PLACEHOLDER_MARK_R


def messages_to_conversations(messages: list[dict], data_args: DataArguments) -> list[dict]:
    """Convert messages-style records (typed content items) to the LLaVA
    conversations format the preprocess functions consume. Media items become
    placeholders at their exact content position, so injection is reduced to a
    consistency check.

    Generated placeholders are MARKED with private-use sentinels (#11) so that a
    literal '<image>'/'<audio>' QUOTED inside user text cannot be mistaken for a
    real media position; inject_missing_media_tokens unwraps the marks back to a
    plain token (and neutralizes the quoted literals)."""
    conversations = []
    for msg in messages:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") if "content" in msg else msg.get("value")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError(f"unsupported content item: {item!r}")
                if item.get("type") == "text":
                    parts.append(item["text"])
                elif item.get("type") == "image":
                    parts.append(_marked_media_placeholder(data_args.image_token))
                elif item.get("type") == "audio":
                    parts.append(_marked_media_placeholder(data_args.audio_token))
                else:
                    raise ValueError(f"unknown content item type: {item.get('type')!r}")
            text = "\n".join(parts)
        else:
            raise ValueError(f"message has no usable content: {msg!r}")
        mapped_role = _ROLE_MAP.get(role, role)
        if data_args.strip_empty_think and mapped_role == "gpt":
            text = _EMPTY_THINK_RE.sub("", text)
        conversations.append({"from": mapped_role, "value": text})
    return conversations


def _media_token_ids(data_args: DataArguments) -> list[int]:
    """Sentinel ids whose features the model splices 1:1 — the collator's
    truncation guard (#12) refuses to drop any of these. Image always (the
    encoder-free path is multimodal); audio / BREEN query only when enabled, so
    their sentinels never appear otherwise."""
    ids = [data_args.image_token_index]
    if getattr(data_args, "audio_enabled", False):
        ids.append(data_args.audio_token_index)
    if getattr(data_args, "learnable_query_enabled", False):
        ids.append(data_args.query_token_index)
    return ids


def _media_feature_token_ids(data_args: DataArguments) -> dict[str, int]:
    """Map each collated media feature-list key to its sentinel id, for the
    modalities the model splices 1:1 from a per-sample feature list (image
    always; audio when enabled). The collator's truncation guard (#12) uses this
    to realign a truncated sample's feature lists with the sentinels that
    survive. BREEN <query> is omitted: it is a broadcast Parameter sized from
    the post-truncation ids, so it carries no per-sample feature list to trim."""
    tokens = {"image": data_args.image_token_index}
    if getattr(data_args, "audio_enabled", False):
        tokens["audio"] = data_args.audio_token_index
    return tokens


class VLMChatTaskEncoder(TaskEncoder):  # pyright: ignore[reportUntypedBaseClass]
    """Runs the full per-sample preprocessing in the DataLoader workers and
    collates with the SAME collator as the local-json path, so the loader
    yields trainer-ready batches with the frozen model contract."""

    # Raw bytes from the aux store; encode_sample decodes them itself.
    decoder: Any = None

    cookers: list = [Cooker(cook_mm_chat, has_subflavors={"crude_type": CRUDE_TYPE})]

    def __init__(self, processor: Any, data_args: DataArguments):
        super().__init__()
        self.tokenizer: Any = processor.tokenizer
        self.image_processor: Any = processor.image_processor
        self.data_args: DataArguments = data_args
        self.collator: Any = DataCollatorForSupervisedDataset(
            tokenizer=self.tokenizer,
            ignore_index=data_args.ignore_index,
            # Media-aware truncation guard (#12): a sentinel dropped past
            # model_max_length would leave its feature to mis-splice later rows.
            media_token_ids=_media_token_ids(data_args),
            media_feature_token_ids=_media_feature_token_ids(data_args),
        )

    def _process_images(self, image_bytes: list[bytes]) -> list[tuple]:
        """Decode + preprocess, dispatching on the processor family:
        RawImageProcessor -> encoder-free 4-tuples (variable resolution);
        HF processors (CLIP/SigLIP/DINO) -> classic 3-tuples. Multi-image
        samples on the classic path force square-padding, mirroring the
        local-json convention (dataset.py process_image overwrite)."""
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes]
        if isinstance(self.image_processor, RawImageProcessor):
            return [process_raw_image(im, self.image_processor) for im in pil_images]
        aspect = self.data_args.image_aspect_ratio
        if len(pil_images) > 1:
            aspect = "pad"
        return [process_classic_image(im, self.image_processor, aspect) for im in pil_images]

    @stateless  # pyright: ignore[reportUntypedFunctionDecorator]
    def encode_sample(self, sample: MMChatRawSample) -> dict:
        # Decode AFTER the shuffle buffer (the buffer holds compressed bytes).
        images = self._process_images(sample.image_bytes)
        if sample.audio_bytes and not self.data_args.audio_enabled:
            raise ValueError(
                f"sample {sample.__key__} carries audio but the model's audio "
                "pathway is off — set model.audio.enabled=true"
            )
        if sample.audio_bytes:
            check_audio_template_supported()
        audios = [load_audio_frames(io.BytesIO(b), self.data_args) for b in sample.audio_bytes]

        conversations = messages_to_conversations(sample.messages, self.data_args)
        inject_missing_media_tokens(
            conversations, n_images=len(images), n_audios=len(audios), data_args=self.data_args
        )
        apply_image_position(
            conversations,
            mode=self.data_args.image_position,
            image_token=self.data_args.image_token,
            # Stable per-sample seed: deterministic across epochs/resumes.
            seed=zlib.crc32(str(sample.__key__).encode()),
            # Protect the audio placeholder from sandwich duplication (#13); the
            # query placeholder isn't injected until after this call.
            protected_tokens=(self.data_args.audio_token,),
        )
        # BREEN port: emit one "<query>" per image at the configured placement
        # (after the image-position rewrite, so it follows the final image spot).
        inject_query_placeholders(conversations, n_images=len(images), data_args=self.data_args)

        has_media = bool(images) or bool(audios)
        out = preprocess([conversations], self.tokenizer, self.data_args, has_image=has_media)
        data_dict = {
            "input_ids": out["input_ids"][0],
            "labels": out["labels"][0],
            "id": sample.__key__,
            # Restore keys are CHAINS: each energon wrapper prepends its own
            # segment and restore unwinds them layer by layer down to the
            # loader. A dict output must carry the INNER sample's chain so
            # MapDataset can prepend to it — an empty placeholder makes
            # checkpointed bucket buffers unrestorable ("not enough values to
            # unpack" deep in the chain), and a missing key saves None
            # (add_sample_restore_key only writes dicts that have the key).
            "__restore_key__": getattr(sample, "__restore_key__", ()),
        }
        # identical dummy assembly to LazySupervisedDataset._get_item
        if images:
            data_dict["image"] = images
        elif self.data_args.is_multimodal:
            data_dict["image"] = [make_dummy_image_entry(self.image_processor)]
        if audios:
            data_dict["audio"] = audios
        elif self.data_args.audio_enabled:
            data_dict["audio"] = [make_dummy_audio_frames(self.data_args)]
        return data_dict

    def batch(self, samples: list[dict]) -> dict:
        return self.collator(samples)


def effective_sample_length(data_dict: dict, data_args: DataArguments) -> int:
    """Post-splice sequence length the GPU will see for one encoded sample:
    input_ids minus media/query sentinels, plus each real image's patch rows,
    each audio's frame rows, and each BREEN query block's learnable-query rows
    (modeling_vlm replaces every sentinel token by its feature block). Dummy
    entries (modality "text" / one zero frame) splice zero-width; the audio
    dummy's +1 here is irrelevant for bucketing.

    BREEN (#4): each "<query>" sentinel is one input_ids token that the model
    splice expands into learnable_query_num_fine + learnable_query_num_coarse
    rows (one block per image). Counting it as a single token would undercount
    real GPU tokens by ~(num_fine + num_coarse - 1) per query block, so
    token-budget microbatches overshoot the budget and land in wrong buckets.
    """
    input_ids = data_dict["input_ids"]
    n_sentinels = int(
        (input_ids == data_args.image_token_index).sum()
        + (input_ids == data_args.audio_token_index).sum()
    )
    image_rows = 0
    for entry in data_dict.get("image", []):
        if len(entry) == 4:
            # encoder-free 4-tuple: per-image patch rows are in the entry
            if entry[3] == "image":
                image_rows += int(entry[0].shape[0])
        elif entry[2] == "image":
            # classic 3-tuple (CLIP/SigLIP/DINO): fixed splice width per image
            image_rows += int(data_args.image_soft_tokens or 0)
    audio_rows = sum(int(frames.shape[0]) for frames in data_dict.get("audio", []))
    # BREEN learnable-query expansion (#4): subtract the 1-token query sentinels
    # and add the rows each expands to. Gated on learnable_query_enabled — when
    # off, no "<query>" sentinel is ever emitted, so this is a no-op.
    query_rows = 0
    n_query = 0
    if getattr(data_args, "learnable_query_enabled", False):
        n_query = int((input_ids == data_args.query_token_index).sum())
        per_query = int(getattr(data_args, "learnable_query_num_fine", 0)) + int(
            getattr(data_args, "learnable_query_num_coarse", 0)
        )
        query_rows = n_query * per_query
    return int(input_ids.shape[0]) - n_sentinels - n_query + image_rows + audio_rows + query_rows


class VLMBucketedChatTaskEncoder(VLMChatTaskEncoder):
    """VLMChatTaskEncoder with length-grouped batching: overriding
    batch_group_criterion routes batching through energon's GroupBatchDataset
    (one bucket per effective-length range, each flushing a full batch_size
    batch through the same collator). Cuts pad-to-batch-max waste (measured
    46% at bs=4 on the vision SFT mix) to roughly the bucket width.

    Buckets are worker-local and fully savable (GroupBatchDataset serializes
    every bucket's buffer into the loader state), so requeue resume is exact.
    """

    def __init__(
        self,
        processor: Any,
        data_args: DataArguments,
        length_buckets: list[int],
        batch_token_budget: int | None = None,
    ):
        super().__init__(processor, data_args)
        if not length_buckets or sorted(length_buckets) != list(length_buckets):
            raise ValueError(
                f"dataset.length_buckets must be ascending bucket edges, got {length_buckets}"
            )
        self.length_buckets: list[int] = list(length_buckets)
        # Token-budget batching: each bucket flushes batch_token_budget //
        # bucket_edge samples, so every micro-batch carries ~the same number
        # of effective tokens — uniform GPU memory across buckets, and short
        # buckets get large batches (high MFU) instead of the loader-wide
        # fixed size. None = legacy fixed batch_size per bucket.
        self.batch_token_budget: int | None = batch_token_budget

    def batch_group_criterion(self, sample: dict) -> tuple[int, int | None]:
        # Bucket key = index of the first edge >= effective length; lengths
        # beyond the last edge share the overflow bucket. Returning None for
        # the batch size selects the loader-wide fixed batch_size.
        eff = effective_sample_length(sample, self.data_args)
        key = bisect.bisect_left(self.length_buckets, eff)
        if self.batch_token_budget is None:
            return key, None
        # Overflow bucket (key == len(edges)) is bounded by max_seq_length +
        # the image budget in practice; sizing it by the last edge keeps it
        # within ~10% of the budget, well inside the memory margin.
        edge = self.length_buckets[min(key, len(self.length_buckets) - 1)]
        return key, max(1, self.batch_token_budget // edge)


# ---------------------------------------------------------------------------
# 4. Metadataset generation & loader (single dataset or weighted blend)
# ---------------------------------------------------------------------------


def normalize_folder_specs(folders: Any, jsonl_name: str = "train.jsonl") -> list:
    """Normalize the `folders` config value into a list of blend spec dicts.
    Accepts a single name, a list of names, or a {name: weight} mapping."""
    items: list[dict]
    if isinstance(folders, str):
        items = [{"folder": folders}]
    elif isinstance(folders, dict):
        items = [{"folder": f, "weight": float(w)} for f, w in folders.items()]
    else:
        items = [{"folder": f} for f in folders]
    if not items:
        raise ValueError("dataset.folders is empty — name at least one blob folder")
    for it in items:
        it.setdefault("weight", 1.0)
        it.setdefault("jsonl_name", jsonl_name)
        it.setdefault("crude_type", CRUDE_TYPE)
    return items


def _metadataset_yaml(specs: list) -> Path:
    """Generate (or reuse, content-addressed) the MetadatasetV2 YAML."""

    def entry(sp: Any, indent: int) -> str:
        jsonl_url = (
            str(sp["local_jsonl"])
            if sp.get("local_jsonl")
            else remote_url(sp["folder"], sp["jsonl_name"])
        )
        pad = " " * indent
        return (
            f"{pad}path: {jsonl_url}\n"
            f"{pad}aux:\n"
            f"{pad}  media_root: filesystem+{remote_url(sp['folder'])}\n"
            f"{pad}subflavors:\n"
            f"{pad}  crude_type: {sp['crude_type']}\n"
        )

    if len(specs) == 1:
        body = "splits:\n  train:\n" + entry(specs[0], 4)
    else:
        blocks = ["      - weight: " + str(sp["weight"]) + "\n" + entry(sp, 8) for sp in specs]
        body = "splits:\n  train:\n    blend:\n" + "".join(blocks)

    yaml_text = "__module__: megatron.energon\n__class__: MetadatasetV2\n" + body
    out_dir = local_data_dir() / ".metadatasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(yaml_text.encode()).hexdigest()[:8]
    out = out_dir / f"mds__{len(specs)}x__{digest}.yaml"
    if not out.is_file():
        out.write_text(yaml_text)
    return out


def _check_distributed_consistency(signature: tuple) -> None:
    """Fail loud on the two silent multi-process data hazards: a multi-process
    launch without torch.distributed init (every rank would read the FULL
    dataset), and ranks disagreeing on jsonl sources (overlapping/missing
    slices)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        if world > 1:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, signature)
            if len(set(gathered)) != 1:
                raise RuntimeError(
                    f"Ranks disagree on the jsonl source(s): {gathered}. Make the "
                    "local copies available on every node, or pass an explicit "
                    "use_local_jsonl=True/False."
                )
    else:
        world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1") or "1"))
        if world > 1:
            raise RuntimeError(
                f"WORLD_SIZE={world} but torch.distributed is not initialized — every "
                "rank would read the FULL dataset. Call init_process_group() before "
                "building the loader."
            )


class VLMGenTaskEncoder(VLMChatTaskEncoder):  # pyright: ignore[reportUntypedBaseClass]
    """Text->image GENERATION encoder (spec 2026-06-20). Reuses the chat cooker
    (image bytes + messages) but emits a GENERATION sample: the assistant
    caption becomes the conditioning prompt (no CE labels) and the image is
    resized to a FIXED square canvas and patchified into the connector's patch
    space as the flow-matching target. batch() stacks the fixed-N targets; the
    model's forward routes on `target_patches` to the flow-matching loss."""

    def __init__(
        self,
        processor: Any,
        data_args: DataArguments,
        resolution: int,
        caption_max_len: int = 128,
        patch_size: int | None = None,
    ):
        super().__init__(processor, data_args)
        if not isinstance(self.image_processor, RawImageProcessor):
            raise ValueError(
                "dataset.task='generation' requires the encoder-free RawImageProcessor"
            )
        # patch_size None -> reuse the connector's 48px model patch (legacy). When
        # the model runs an independent gen embedder (e.g. 16px), the dataset must
        # patchify at that SAME size, else target dim (psz^2*3) mismatches the
        # gen embedder/x-head (e.g. 768 vs 6912).
        self.model_patch_size: int = int(
            patch_size if patch_size else self.image_processor.model_patch_size
        )
        if resolution % self.model_patch_size != 0:
            raise ValueError(
                f"gen_resolution {resolution} must be a multiple of the generation "
                f"patch size {self.model_patch_size}"
            )
        self.resolution: int = int(resolution)
        self.grid: int = self.resolution // self.model_patch_size
        self.n_patch: int = self.grid * self.grid
        self.caption_max_len: int = int(caption_max_len)

    def _caption(self, messages: list[dict]) -> str:
        """The assistant turn's text (the caption used as the prompt)."""
        text = ""
        for msg in messages:
            role = msg.get("role") or msg.get("from")
            if _ROLE_MAP.get(role, role) != "gpt":
                continue
            content = msg.get("content") if "content" in msg else msg.get("value")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    it["text"]
                    for it in content
                    if isinstance(it, dict) and it.get("type") == "text"
                )
        return text

    @stateless  # pyright: ignore[reportUntypedFunctionDecorator]
    def encode_sample(self, sample: MMChatRawSample) -> dict:
        if not sample.image_bytes:
            raise ValueError(f"generation sample {sample.__key__} has no image")
        pil = Image.open(io.BytesIO(sample.image_bytes[0])).convert("RGB")
        pil = pil.resize((self.resolution, self.resolution), Image.BICUBIC)
        arr = np.asarray(pil, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
        chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous() * 2.0 - 1.0  # (3,H,W) [-1,1]
        target_patches = pixels_to_patches(chw, self.model_patch_size)  # (N, patch_dim)
        position_ids = make_position_ids(self.grid, self.grid)  # (N, 2)
        caption = self._caption(sample.messages)
        ids = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.caption_max_len,
            add_special_tokens=True,
        ).input_ids
        return {
            "input_ids": ids,  # list[int]
            "target_patches": target_patches,  # (N, patch_dim) fp32
            "image_position_ids": position_ids,  # (N, 2) long
            "id": sample.__key__,
            "__restore_key__": getattr(sample, "__restore_key__", ()),
        }

    def batch(self, samples: list[dict]) -> dict:
        # Left-pad the caption so the timestep token + image block sit
        # immediately after the real text (no RoPE position gap); positions are
        # structural arange in assemble_generation_inputs.
        enc = self.tokenizer.pad(
            {"input_ids": [s["input_ids"] for s in samples]},
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target_patches": torch.stack([s["target_patches"] for s in samples]),
            "image_position_ids": torch.stack([s["image_position_ids"] for s in samples]),
        }


def build_energon_train_loader(
    dataset_config: Any,
    processor: Any,
    data_args: DataArguments,
    batch_size: int,
    *,
    task_encoder: TaskEncoder | None = None,
    worker_config: WorkerConfig | None = None,
    auto_prepare: bool = True,
    **savable_loader_kwargs: Any,
) -> SavableDataLoader:
    """dataset.folders ({blob folder: blend weight}) -> streaming train loader
    yielding trainer-ready batch dicts.

    Two layouts, selected by config:
      * dataset.folders (this function) — loose-file jsonl-crude: one train.jsonl
        per folder + loose media; one Azure GET per image.
      * dataset.wds_path (delegated to vlm.data.energon_wds.build_wds_train_loader)
        — a prepared CrudeWebdataset (tar shards with in-tar images + .nv-meta);
        one sequential GET per ~10k samples. Mutually exclusive with folders.

    Fully automatic by default (folders path): on first use each jsonl is
    downloaded to the local data dir with its index built in the same pass
    (resumable; LOCAL_RANK==0 downloads per node, others wait at the barrier).
    Media (images/audio) always streams lazily from blob through the MSC cache.

    `dataset_config.use_local_jsonl`: None (default) = local hybrid mode,
    auto-downloading if needed; True = require the local copy; False = always
    stream the remote jsonl (remote index built if missing — needs container
    write access).

    Multi-GPU: WorkerConfig.default_worker_config() picks up rank/world_size
    from torch.distributed; each rank reads a disjoint sample-index slice.
    Resume: save_state_rank()/restore_state_rank() — valid only for the same
    world_size and num_workers.
    """
    if _streaming_import_error is not None:
        raise ImportError(
            "dataset.type='energon' requires the streaming dependencies — install "
            "megatron-energon (with the [azure-storage-blob] extra) and "
            f"multi-storage-client. Original error: {_streaming_import_error}"
        ) from _streaming_import_error
    # Dataset-only config validation (#27 etc.) — guards direct callers/tests,
    # in addition to the fail-fast validate_config(cfg) at the CLI entry.
    validate_dataset_config(dataset_config)
    # The cold Azure shuffle-buffer fill (~90 s loose-file / ~111 s WDS) exceeds
    # energon's 60 s watchdog default on EVERY startup, producing a spurious
    # all-thread stack dump that has been repeatedly misread as a PicklingError
    # (it is not — fork never pickles; data/datapipe-rootcause-m6). Raise the
    # INITIAL timeout to cover the cold fill while keeping the 60 s steady-state
    # watchdog that catches genuine per-sample hangs. Applies to both branches.
    # (Separate from the checkpoint_every_sec setdefault below, which silences
    # the rolling-checkpoint SSL-socket pickling noise.)
    savable_loader_kwargs.setdefault("watchdog_initial_timeout_seconds", 600)
    # Prepared-WebDataset branch (dataset.wds_path): image bytes are bundled IN
    # the tar shards, read by the WDS cooker (vlm.data.energon_wds) directly off
    # the .nv-meta dir — no jsonl download / index / metadataset generation. The
    # lazy import keeps the energon_dataset <-> energon_wds dependency acyclic.
    wds_path = getattr(dataset_config, "wds_path", None)
    if wds_path:
        if dataset_config.folders:
            raise ValueError(
                "dataset.wds_path and dataset.folders are mutually exclusive — "
                "wds_path streams a prepared CrudeWebdataset (in-tar images) and "
                "folders streams the loose-file jsonl layout. Set exactly one."
            )
        from .energon_wds import build_wds_train_loader

        return build_wds_train_loader(
            dataset_config,
            processor,
            data_args,
            batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            **savable_loader_kwargs,
        )
    if not dataset_config.folders:
        raise ValueError(
            "No dataset source configured for dataset.type='energon' — set "
            "exactly one of dataset.wds_path (a prepared CrudeWebdataset with "
            "in-tar images) or dataset.folders (the loose-file jsonl layout)."
        )
    _require_credentials()
    use_local_jsonl = dataset_config.use_local_jsonl
    specs = normalize_folder_specs(dataset_config.folders, dataset_config.jsonl_name)
    signature = []
    for sp in specs:
        local = local_jsonl_path(sp["folder"], sp["jsonl_name"])
        local_ok = local.is_file() and is_prepared(EPath(str(local)))
        if use_local_jsonl is None:
            use_local = local_ok or auto_prepare  # default: local hybrid mode
        else:
            use_local = use_local_jsonl
        if use_local and not local_ok:
            if not auto_prepare:
                raise RuntimeError(f"No indexed local copy at {local}")
            # One download per node; the other ranks wait at the barrier.
            if int(os.environ.get("LOCAL_RANK", "0") or "0") == 0:
                download_and_index(sp["folder"], sp["jsonl_name"])
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if not (local.is_file() and is_prepared(EPath(str(local)))):
                raise RuntimeError(f"auto-prepare did not produce an indexed copy at {local}")
        if use_local:
            sp["local_jsonl"] = local
            size = local.stat().st_size
        else:
            jsonl = find_jsonl(sp["folder"], sp["jsonl_name"])
            if not is_prepared(jsonl):
                if auto_prepare:
                    log.info(f"Index missing/stale for {jsonl}, building it (one full pass)...")
                    count = prepare_jsonl(jsonl)
                    log.info(f"Indexed {count} samples.")
                else:
                    raise RuntimeError(f"{jsonl} has no (matching) .jsonl.idx")
            size = jsonl.size()
        signature.append((sp["folder"], sp["jsonl_name"], bool(use_local), size))
    _check_distributed_consistency(tuple(signature))

    metadataset_yaml = _metadataset_yaml(specs)
    wc = worker_config or WorkerConfig.default_worker_config(num_workers=dataset_config.num_workers)
    if task_encoder is None and getattr(dataset_config, "task", "understanding") == "generation":
        _gen_psz = getattr(dataset_config, "gen_patch_size", None)
        task_encoder = VLMGenTaskEncoder(
            processor,
            data_args,
            resolution=int(getattr(dataset_config, "gen_resolution", 384)),
            caption_max_len=int(getattr(dataset_config, "gen_caption_max_len", 128)),
            patch_size=int(_gen_psz) if _gen_psz else None,
        )
    if task_encoder is None:
        length_buckets = getattr(dataset_config, "length_buckets", None)
        if length_buckets:
            task_encoder = VLMBucketedChatTaskEncoder(
                processor,
                data_args,
                list(length_buckets),
                batch_token_budget=getattr(dataset_config, "batch_token_budget", None),
            )
        else:
            task_encoder = VLMChatTaskEncoder(processor, data_args)
    # Token-budget bucketing sizes every batch itself; energon then requires
    # the loader-wide batch_size to be None ("one of the two should be None").
    if getattr(task_encoder, "batch_token_budget", None):
        batch_size = None
    dataset = get_train_dataset(
        str(metadataset_yaml),
        batch_size=batch_size,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        max_samples_per_sequence=dataset_config.max_samples_per_sequence,
        worker_config=wc,
        task_encoder=task_encoder,
    )
    # PicklingError mitigation (2026-06-24): energon's rolling in-memory checkpoint
    # (default checkpoint_every_sec=60) snapshots+pickles each worker's state about
    # every other step. A worker mid-Azure-read holds a live ssl.SSLSocket, which
    # is not picklable, so every snapshot logs a (caught, non-fatal) PicklingError —
    # harmless to data flow (samples_seen stays smooth) but it balloons the logs
    # over a multi-day stream. Durable resume does NOT use this rolling buffer: the
    # HF checkpoint saves energon_state_rank*.pt via a SEPARATE coordinated
    # get_checkpoint (verified present + correct). So lengthening the rolling
    # interval just cuts the noise without touching resume. setdefault keeps it
    # overridable.
    savable_loader_kwargs.setdefault("checkpoint_every_sec", 900)
    return get_savable_loader(dataset, **savable_loader_kwargs)
