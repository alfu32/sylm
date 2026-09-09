"""Writer/reader for the portable SYL2 float32 tensor container."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"SYL2"
VERSION = 1
FLOAT32 = 1


def _tensor_bytes(tensor):
    import numpy as np

    value = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
    if value.dtype != tensor.dtype:
        value = value.to(dtype=tensor.dtype)
    if str(value.dtype) != "torch.float32":
        raise ValueError(f"SYL2 only supports float32 tensors, got {value.dtype}")
    return value.numpy().astype(np.float32, copy=False).tobytes(order="C")


def write_syl2(path: str | Path, metadata: dict, tensors: dict[str, object]) -> None:
    """Write one self-contained artifact; tensors must be float32 PyTorch tensors."""
    entries: list[tuple[str, tuple[int, ...], bytes]] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise TypeError(f"tensor {name!r} is not a tensor")
        payload = _tensor_bytes(tensor)
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if len(shape) > 255:
            raise ValueError(f"tensor rank too large: {name}")
        entries.append((name, shape, payload))

    metadata = dict(metadata)
    metadata["tensorNames"] = [name for name, _shape, _payload in entries]
    metadata["tensorShapes"] = {name: list(shape) for name, shape, _payload in entries}
    encoded_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
    if len(encoded_metadata) > 0xFFFFFFFF:
        raise ValueError("SYL2 metadata is too large")

    body = bytearray()
    body += MAGIC
    body += struct.pack("<I", VERSION)
    body += struct.pack("<I", len(encoded_metadata))
    body += encoded_metadata
    body += struct.pack("<I", len(entries))
    for name, shape, payload in entries:
        encoded_name = name.encode("utf-8")
        if len(encoded_name) > 0xFFFFFFFF:
            raise ValueError(f"tensor name is too long: {name}")
        body += struct.pack("<I", len(encoded_name))
        body += encoded_name
        body += struct.pack("<B", FLOAT32)
        body += struct.pack("<B", len(shape))
        for dimension in shape:
            if dimension < 0 or dimension > 0xFFFFFFFF:
                raise ValueError(f"invalid tensor dimension for {name}: {dimension}")
            body += struct.pack("<I", dimension)
        body += struct.pack("<Q", len(payload))
        body += payload
    body += hashlib.sha256(body).digest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(body)


def read_syl2(path: str | Path) -> tuple[dict, dict[str, tuple[tuple[int, ...], bytes]]]:
    """Small validation reader used by exporter tests, not by Kotlin."""
    data = Path(path).read_bytes()
    if len(data) < 4 + 4 + 4 + 4 + 32 or data[:4] != MAGIC:
        raise ValueError("invalid SYL2 file")
    if data[-32:] != hashlib.sha256(data[:-32]).digest():
        raise ValueError("SYL2 checksum mismatch")
    offset = 4
    version = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if version != VERSION:
        raise ValueError(f"unsupported SYL2 version {version}")
    metadata_length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    metadata = json.loads(data[offset:offset + metadata_length].decode("utf-8"))
    offset += metadata_length
    count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    tensors = {}
    for _ in range(count):
        name_length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        name = data[offset:offset + name_length].decode("utf-8")
        offset += name_length
        dtype, rank = struct.unpack_from("<BB", data, offset)
        offset += 2
        if dtype != FLOAT32:
            raise ValueError("unsupported SYL2 dtype")
        shape = struct.unpack_from("<" + "I" * rank, data, offset) if rank else ()
        offset += 4 * rank
        payload_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        payload = data[offset:offset + payload_length]
        offset += payload_length
        if len(payload) != payload_length:
            raise ValueError("truncated SYL2 tensor")
        tensors[name] = (tuple(shape), payload)
    if offset != len(data) - 32:
        raise ValueError("trailing SYL2 bytes")
    return metadata, tensors

