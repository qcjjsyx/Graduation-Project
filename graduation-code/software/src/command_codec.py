"""Command / Descriptor fixed-record codec for schema v2.

Schema v2 is the current mainline Command/Descriptor artifact format.  It is
unrelated to the archived NodeTask ABI v2 (see
``archive/legacy-node-task-abi-v2``), which is not consumed by any mainline
code.  Relative to Command/Descriptor v1 the schema versions the artifact
explicitly (``COMMAND_SCHEMA_VERSION``, mirrored in the manifest) and defines
the pivot-policy fields of ``FRONT_DESC``; the 32B command, 64B descriptor and
64B completion record layouts are unchanged.  v1 records are rejected instead
of being reinterpreted: a v1 ``FRONT_DESC`` leaves the policy words at zero,
which v2 validation refuses.

The records in this module are device-facing and always little-endian.  Token
state vector helpers exist only for shared cross-language test fixtures; the
device ABI does not define an in-memory scoreboard layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Iterable, Optional, Sequence, Tuple


NONE = 0xFFFFFFFF
COMMAND_SCHEMA_VERSION = 2
COMMAND_RECORD_BYTES = 32
DESCRIPTOR_RECORD_BYTES = 64
COMPLETION_RECORD_BYTES = 64

# P-vector regions are initialized to this INT32 sentinel ("not written yet").
# PANEL_LU must overwrite every entry (STATIC writes the explicit identity);
# a zero-filled P-vector is never interpreted as identity.
PVECTOR_INIT_VALUE = -1
PVECTOR_ELEMENT_BYTES = 4


class Opcode(IntEnum):
    NODE_BEGIN = 0x01
    LOAD_FRONT = 0x02
    ASSEMBLE_EXTEND_ADD = 0x03
    PANEL_LU = 0x04
    TRSM_LEFT = 0x05
    TRSM_RIGHT = 0x06
    GEMM_SCHUR = 0x07
    STORE_FACTOR = 0x08
    STORE_UPDATE = 0x09
    SOLVE_FORWARD = 0x0A
    SOLVE_BACKWARD = 0x0B
    NODE_COMMIT = 0x0C
    ABORT_NODE = 0x0D


class CommandFlag(IntFlag):
    TRACE_ENABLE = 1 << 0
    ALLOW_RETRY = 1 << 1


TRACE_ENABLE = int(CommandFlag.TRACE_ENABLE)
ALLOW_RETRY = int(CommandFlag.ALLOW_RETRY)
COMMAND_ALLOWED_FLAGS = TRACE_ENABLE | ALLOW_RETRY
DESCRIPTOR_ALLOWED_FLAGS = 0


class DescriptorType(IntEnum):
    REGION_DESC = 0x01
    FRONT_DESC = 0x02
    CONTRIBUTION_DESC = 0x03
    FACTOR_DESC = 0x04
    KERNEL_DESC = 0x05
    SOLVE_DESC = 0x06
    SCALE_DESC = 0x07
    DEPENDENCY_DESC = 0x08


class PivotPolicy(IntEnum):
    """Device pivot policy declared by FRONT_DESC body[8].

    Value 0 is deliberately not a policy: it marks "unset" so that a v1 record
    (all-zero policy words) can never be mistaken for a valid v2 policy, and so
    that the policy is never guessed from the P-vector contents.
    """

    STATIC = 0x01
    DYNAMIC_PANEL = 0x02


class DataFormat(IntEnum):
    FP64 = 0x01
    FP32 = 0x02
    INT32 = 0x03
    LEGACY_INT32_EXP = 0x04


class DataLayout(IntEnum):
    ROW_MAJOR = 0x01


class KernelBackend(IntEnum):
    SYSTEMC_FP32_DEVICE_MODEL = 0x01
    SYSTEMC_INT32_GEMM_MODEL = 0x02


# ---------------------------------------------------------------------------
# SCALE_DESC (T10 frozen semantics)
# ---------------------------------------------------------------------------

#: AUTO marker for a signed power-of-two exponent word.
SCALE_AUTO = 0x80000000
#: Supported fixed exponent range (i32 two's complement words).
SCALE_EXPONENT_MIN = -256
SCALE_EXPONENT_MAX = 256


class ScaleRoundingMode(IntEnum):
    NEAREST_TIES_EVEN = 1


class ScaleSaturationPolicy(IntEnum):
    FAIL = 1
    CLAMP_AND_REPORT = 2


class ScaleOverflowPolicy(IntEnum):
    FAIL = 1


@dataclass(frozen=True)
class ScaleDesc:
    """Decoded SCALE_DESC policy (schema v2 body words 0..5).

    ``exponent_a``/``exponent_b`` are ``None`` for AUTO (the device derives a
    per-tile power-of-two scale from the local maxabs) or a fixed i32 exponent
    in ``[-256, 256]``.  The output scale is AUTO only in this version: it is
    derived from the actual ``eA + eB`` of each tile.  Words 6..9 stay zero.
    """

    exponent_a: Optional[int] = None
    exponent_b: Optional[int] = None
    rounding: ScaleRoundingMode = ScaleRoundingMode.NEAREST_TIES_EVEN
    saturation: ScaleSaturationPolicy = ScaleSaturationPolicy.FAIL
    overflow: ScaleOverflowPolicy = ScaleOverflowPolicy.FAIL

    def __post_init__(self) -> None:
        for name, value in (("exponent_a", self.exponent_a), ("exponent_b", self.exponent_b)):
            if value is None:
                continue
            if not SCALE_EXPONENT_MIN <= value <= SCALE_EXPONENT_MAX:
                raise ValueError(
                    f"{name} must be in "
                    f"[{SCALE_EXPONENT_MIN}, {SCALE_EXPONENT_MAX}], got {value}"
                )
        ScaleRoundingMode(self.rounding)
        ScaleSaturationPolicy(self.saturation)
        ScaleOverflowPolicy(self.overflow)

    def to_body_words(self) -> Tuple[int, ...]:
        def exponent_word(value: Optional[int]) -> int:
            if value is None:
                return SCALE_AUTO
            return value & 0xFFFFFFFF

        return (
            exponent_word(self.exponent_a),
            exponent_word(self.exponent_b),
            SCALE_AUTO,
            int(self.rounding),
            int(self.saturation),
            int(self.overflow),
            0,
            0,
            0,
            0,
        )

    @classmethod
    def from_body_words(cls, body_words: Sequence[int]) -> "ScaleDesc":
        """Decode and validate a SCALE_DESC body.

        Malformed encodings (undefined zero enumerations, non-zero reserved
        words) raise ``ValueError``; well-formed but unsupported values (fixed
        output scale, exponent outside the supported range) also raise, since
        the compiler never emits them and a decoder must not silently ignore
        device-visible policy.
        """

        if len(body_words) != 10:
            raise ValueError("SCALE_DESC body must have 10 words")
        if any(word != 0 for word in body_words[6:10]):
            raise ValueError("SCALE_DESC reserved words 6..9 must be zero")
        if body_words[2] != SCALE_AUTO:
            raise ValueError(
                "fixed output_scale is defined but not supported in this version"
            )

        def decode_exponent(word: int, label: str) -> Optional[int]:
            if word == SCALE_AUTO:
                return None
            signed = word - (1 << 32) if word & 0x80000000 else word
            if not SCALE_EXPONENT_MIN <= signed <= SCALE_EXPONENT_MAX:
                raise ValueError(
                    f"{label} outside the supported range "
                    f"[{SCALE_EXPONENT_MIN}, {SCALE_EXPONENT_MAX}]"
                )
            return signed

        rounding = body_words[3]
        if rounding == 0:
            raise ValueError("SCALE_DESC rounding_mode 0 is undefined")
        saturation = body_words[4]
        if saturation == 0:
            raise ValueError("SCALE_DESC saturation_policy 0 is undefined")
        overflow = body_words[5]
        if overflow == 0:
            raise ValueError("SCALE_DESC overflow_policy 0 is undefined")
        return cls(
            exponent_a=decode_exponent(body_words[0], "exponent_a"),
            exponent_b=decode_exponent(body_words[1], "exponent_b"),
            rounding=ScaleRoundingMode(rounding),
            saturation=ScaleSaturationPolicy(saturation),
            overflow=ScaleOverflowPolicy(overflow),
        )


#: Default SCALE_DESC emitted for the INT32 GEMM backend: AUTO, AUTO, AUTO,
#: nearest ties-to-even, FAIL, FAIL.
DEFAULT_GEMM_SCALE = ScaleDesc()


class SolveDirection(IntEnum):
    FORWARD = 0x01
    BACKWARD = 0x02


class StatusCode(IntEnum):
    """Completion status codes (v2).

    Failure reporting contract (see docs/interface/Command_Schema_v2.md):

    * requested pivot policy not implemented / not supported by the executor:
      ``POLICY_NOT_SUPPORTED``; the command is refused before execution and no
      READY token is published (``未实现``/``未支持``).
    * malformed policy encoding, illegal P-vector reference or mixed schema
      records: ``BAD_COMMAND`` / ``BAD_DESCRIPTOR`` at validation time.
    * numeric pivot failure during elimination (``主元失败``):

      - ``STATIC_PIVOT_FAILURE``: the fixed elimination order produced a zero
        pivot under ``STATIC``.  This is *not* a claim that the matrix is
        singular.
      - ``PIVOT_UNSTABLE``: the actual pivot was below the declared relative
        threshold (nonzero but too small).
      - ``PIVOT_NOT_FOUND``: ``DYNAMIC_PANEL`` found no admissible pivot inside
        the pivot block (search never crosses the pivot-block boundary).
      - ``NON_FINITE_VALUE``: NaN/Inf in inputs or produced by elimination.

    A failed command never publishes its READY token; the host retry must
    restart from the original inputs or from fully rebuilt state.
    """

    OK = 0x0000
    BAD_COMMAND = 0x0001
    BAD_DESCRIPTOR = 0x0002
    ADDRESS_FAULT = 0x0003
    BUFFER_FULL = 0x0004
    DEPENDENCY_FAILED = 0x0005
    POLICY_NOT_SUPPORTED = 0x0006

    PIVOT_NOT_FOUND = 0x0100
    PIVOT_UNSTABLE = 0x0101
    NUMERIC_OVERFLOW = 0x0102
    QUANTIZATION_SATURATION = 0x0103
    PRECISION_RETRY = 0x0104
    STATIC_PIVOT_FAILURE = 0x0105
    NON_FINITE_VALUE = 0x0106

    TIMEOUT = 0x0200
    ABORTED = 0x0201


class TokenState(IntEnum):
    """Stable values used by the shared fixture, not an executor scoreboard ABI."""

    UNSIGNALED = 0
    READY = 1
    FAILED = 2


_COMMAND_STRUCT = struct.Struct("<8I")
_DESCRIPTOR_STRUCT = struct.Struct("<HHIQQ10I")
_COMPLETION_STRUCT = struct.Struct("<IIHHIQQQQQII")
_TOKEN_STATE_STRUCT = struct.Struct("<I")

assert _COMMAND_STRUCT.size == COMMAND_RECORD_BYTES
assert _DESCRIPTOR_STRUCT.size == DESCRIPTOR_RECORD_BYTES
assert _COMPLETION_STRUCT.size == COMPLETION_RECORD_BYTES

_U16_MAX = 0xFFFF
_U32_MAX = 0xFFFFFFFF
_U64_MAX = 0xFFFFFFFFFFFFFFFF


def _check_uint(name: str, value: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name}={value!r} out of range [0, {maximum:#x}]")


def _check_exact_size(name: str, data: bytes, expected: int) -> None:
    if len(data) != expected:
        raise ValueError(f"{name} must be exactly {expected} bytes, got {len(data)}")


def _known_enum(name: str, value: int, enum_type: type[IntEnum]) -> None:
    try:
        enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name}: {value:#x}") from exc


@dataclass(frozen=True)
class Command:
    opcode: int
    flags: int
    command_id: int
    node_id: int = NONE
    descriptor_id: int = NONE
    wait_list_id: int = NONE
    signal_token: int = NONE
    arg0: int = 0

    def validate(self) -> None:
        for name in (
            "opcode",
            "flags",
            "command_id",
            "node_id",
            "descriptor_id",
            "wait_list_id",
            "signal_token",
            "arg0",
        ):
            _check_uint(name, getattr(self, name), _U32_MAX)
        _known_enum("opcode", self.opcode, Opcode)
        if self.flags & ~COMMAND_ALLOWED_FLAGS:
            raise ValueError(f"command flags contain reserved bits: {self.flags:#x}")
        if self.arg0 != 0:
            raise ValueError("command arg0 is reserved and must be zero")

    def to_bytes(self) -> bytes:
        self.validate()
        return _COMMAND_STRUCT.pack(
            self.opcode,
            self.flags,
            self.command_id,
            self.node_id,
            self.descriptor_id,
            self.wait_list_id,
            self.signal_token,
            self.arg0,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Command":
        _check_exact_size("command record", data, COMMAND_RECORD_BYTES)
        command = cls(*_COMMAND_STRUCT.unpack(data))
        command.validate()
        return command


_RESERVED_BODY_WORDS = {
    DescriptorType.REGION_DESC: range(9, 10),
    DescriptorType.FRONT_DESC: range(10, 10),
    DescriptorType.CONTRIBUTION_DESC: range(9, 10),
    DescriptorType.FACTOR_DESC: range(7, 10),
    DescriptorType.KERNEL_DESC: range(10, 10),
    DescriptorType.SOLVE_DESC: range(8, 10),
    DescriptorType.SCALE_DESC: range(6, 10),
    DescriptorType.DEPENDENCY_DESC: range(1, 10),
}


def normalize_pivot_threshold(ratio: float) -> float:
    """Return the single effective threshold shared by host and device.

    The device stores the threshold as IEEE-754 binary32 bits
    (``FRONT_DESC.body_words[9]``).  Every consumer — configuration, manifest,
    host reference and descriptor — must use the *same* effective value, so the
    ratio is normalized to its binary32 value here exactly once.

    Rejected cases:

    * ratio outside ``[0.0, 1.0)``;
    * positive ratio that underflows to zero in binary32 (the device would
      silently lose the threshold);
    * ratio that rounds to ``1.0`` in binary32 (would leave the legal range).
    """

    value = float(ratio)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"threshold ratio must be in [0.0, 1.0), got {ratio}")
    (normalized,) = struct.unpack("<f", struct.pack("<f", value))
    normalized = float(normalized)
    if value != 0.0 and normalized == 0.0:
        raise ValueError(
            f"threshold ratio {ratio!r} underflows to zero in binary32"
        )
    if not 0.0 <= normalized < 1.0:
        raise ValueError(
            f"threshold ratio {ratio!r} rounds to {normalized} in binary32, "
            "outside [0.0, 1.0)"
        )
    return normalized


def encode_pivot_threshold(ratio: float) -> int:
    """Encode the effective threshold as IEEE-754 binary32 bits."""

    normalized = normalize_pivot_threshold(ratio)
    bits = struct.unpack("<I", struct.pack("<f", normalized))[0]
    if decode_pivot_threshold(bits) != normalized:  # pragma: no cover - exact by construction
        raise ValueError("pivot threshold encoding is not lossless")
    return bits


def decode_pivot_threshold(bits: int) -> float:
    """Decode IEEE-754 binary32 threshold bits back to a ratio in [0, 1)."""

    _check_uint("pivot threshold bits", bits, _U32_MAX)
    (value,) = struct.unpack("<f", struct.pack("<I", bits))
    if not (0.0 <= value < 1.0):
        raise ValueError(f"pivot threshold ratio must be in [0.0, 1.0), got {value}")
    return float(value)


@dataclass(frozen=True)
class Descriptor:
    descriptor_type: int
    flags: int = 0
    reserved: int = 0
    payload_offset: int = 0
    payload_bytes: int = 0
    body_words: Tuple[int, ...] = (0,) * 10

    def validate(self, memory_image_bytes: Optional[int] = None) -> None:
        _check_uint("descriptor_type", self.descriptor_type, _U16_MAX)
        _check_uint("descriptor flags", self.flags, _U16_MAX)
        _check_uint("descriptor reserved", self.reserved, _U32_MAX)
        _check_uint("payload_offset", self.payload_offset, _U64_MAX)
        _check_uint("payload_bytes", self.payload_bytes, _U64_MAX)
        if len(self.body_words) != 10:
            raise ValueError("descriptor body must contain exactly 10 u32 words")
        for index, word in enumerate(self.body_words):
            _check_uint(f"body_words[{index}]", word, _U32_MAX)

        _known_enum("descriptor type", self.descriptor_type, DescriptorType)
        descriptor_type = DescriptorType(self.descriptor_type)
        if self.flags & ~DESCRIPTOR_ALLOWED_FLAGS:
            raise ValueError(f"descriptor flags contain reserved bits: {self.flags:#x}")
        if self.reserved != 0:
            raise ValueError("descriptor reserved field must be zero")
        for index in _RESERVED_BODY_WORDS[descriptor_type]:
            if self.body_words[index] != 0:
                raise ValueError(
                    f"descriptor body_words[{index}] is reserved and must be zero"
                )

        end = self.payload_offset + self.payload_bytes
        if end > _U64_MAX:
            raise ValueError("descriptor payload range overflows u64")
        if memory_image_bytes is not None:
            _check_uint("memory_image_bytes", memory_image_bytes, _U64_MAX)
            if end > memory_image_bytes:
                raise ValueError("descriptor payload range exceeds memory image")

        if descriptor_type == DescriptorType.REGION_DESC:
            base_addr = self.body_words[0] | (self.body_words[1] << 32)
            byte_size = self.body_words[2] | (self.body_words[3] << 32)
            row_stride, rows, cols, data_format, layout = self.body_words[4:9]
            if base_addr % 64:
                raise ValueError("REGION_DESC base_addr must be 64-byte aligned")
            if base_addr + byte_size > _U64_MAX:
                raise ValueError("REGION_DESC address range overflows u64")
            if memory_image_bytes is not None and base_addr + byte_size > memory_image_bytes:
                raise ValueError("REGION_DESC address range exceeds memory image")
            _known_enum("data format", data_format, DataFormat)
            _known_enum("data layout", layout, DataLayout)
            element_size = {
                DataFormat.FP64: 8,
                DataFormat.FP32: 4,
                DataFormat.INT32: 4,
                DataFormat.LEGACY_INT32_EXP: 4,
            }[DataFormat(data_format)]
            if rows == 0 or cols == 0:
                if byte_size != 0 or row_stride != 0:
                    raise ValueError("empty REGION_DESC must have zero size and stride")
            else:
                if row_stride < cols * element_size:
                    raise ValueError("REGION_DESC row_stride is smaller than one row")
                required = (rows - 1) * row_stride + cols * element_size
                if required > byte_size:
                    raise ValueError("REGION_DESC dimensions exceed byte_size")
        elif descriptor_type == DescriptorType.FRONT_DESC:
            if self.payload_bytes != 0:
                raise ValueError("FRONT_DESC does not define a variable payload")
            policy_word = self.body_words[8]
            threshold_word = self.body_words[9]
            try:
                PivotPolicy(policy_word)
            except ValueError as exc:
                raise ValueError(
                    f"unknown pivot policy {policy_word:#x} "
                    "(v1 FRONT_DESC records are not valid schema v2)"
                ) from exc
            try:
                decode_pivot_threshold(threshold_word)
            except ValueError as exc:
                raise ValueError(f"FRONT_DESC pivot threshold: {exc}") from exc
        elif descriptor_type == DescriptorType.CONTRIBUTION_DESC:
            expected = 4 * (self.body_words[4] + self.body_words[5])
            if self.payload_offset % 4:
                raise ValueError("CONTRIBUTION_DESC payload_offset must be u32 aligned")
            if self.payload_bytes != expected:
                raise ValueError(
                    "CONTRIBUTION_DESC payload_bytes must hold row and column maps"
                )
        elif descriptor_type == DescriptorType.DEPENDENCY_DESC:
            expected = 4 * self.body_words[0]
            if self.payload_offset % 4:
                raise ValueError("DEPENDENCY_DESC payload_offset must be u32 aligned")
            if self.payload_bytes != expected:
                raise ValueError(
                    "DEPENDENCY_DESC payload_bytes must equal token_count * 4"
                )
        elif self.payload_bytes != 0:
            raise ValueError(
                f"{descriptor_type.name} does not define a v2 variable payload"
            )

    def to_bytes(self) -> bytes:
        self.validate()
        return _DESCRIPTOR_STRUCT.pack(
            self.descriptor_type,
            self.flags,
            self.reserved,
            self.payload_offset,
            self.payload_bytes,
            *self.body_words,
        )

    @classmethod
    def from_bytes(
        cls, data: bytes, *, memory_image_bytes: Optional[int] = None
    ) -> "Descriptor":
        _check_exact_size("descriptor record", data, DESCRIPTOR_RECORD_BYTES)
        values = _DESCRIPTOR_STRUCT.unpack(data)
        descriptor = cls(
            descriptor_type=values[0],
            flags=values[1],
            reserved=values[2],
            payload_offset=values[3],
            payload_bytes=values[4],
            body_words=tuple(values[5:]),
        )
        descriptor.validate(memory_image_bytes)
        return descriptor

    def dependency_tokens(
        self,
        memory_image: bytes,
        *,
        token_count: Optional[int] = None,
        max_wait_tokens: Optional[int] = None,
    ) -> Tuple[int, ...]:
        self.validate(len(memory_image))
        if self.descriptor_type != DescriptorType.DEPENDENCY_DESC:
            raise ValueError("descriptor is not a DEPENDENCY_DESC")
        count = self.body_words[0]
        if max_wait_tokens is not None and count > max_wait_tokens:
            raise ValueError("dependency token count exceeds max_wait_tokens")
        begin = self.payload_offset
        end = begin + self.payload_bytes
        tokens = struct.unpack(f"<{count}I", memory_image[begin:end]) if count else ()
        if len(set(tokens)) != len(tokens):
            raise ValueError("dependency token list contains duplicates")
        if token_count is not None:
            _check_uint("token_count", token_count, _U32_MAX)
            if any(token >= token_count for token in tokens):
                raise ValueError("dependency token list contains an unknown token")
        return tuple(tokens)


@dataclass(frozen=True)
class Completion:
    command_id: int
    node_id: int
    status_code: int
    pivot_count: int = 0
    start_cycle: int = 0
    finish_cycle: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    stall_cycles: int = 0
    overflow_count: int = 0
    retry_count: int = 0

    def validate(self) -> None:
        for name in ("command_id", "node_id", "pivot_count", "overflow_count", "retry_count"):
            _check_uint(name, getattr(self, name), _U32_MAX)
        _check_uint("status_code", self.status_code, _U16_MAX)
        for name in (
            "start_cycle",
            "finish_cycle",
            "read_bytes",
            "write_bytes",
            "stall_cycles",
        ):
            _check_uint(name, getattr(self, name), _U64_MAX)
        _known_enum("status code", self.status_code, StatusCode)

    def to_bytes(self) -> bytes:
        self.validate()
        return _COMPLETION_STRUCT.pack(
            self.command_id,
            self.node_id,
            self.status_code,
            0,
            self.pivot_count,
            self.start_cycle,
            self.finish_cycle,
            self.read_bytes,
            self.write_bytes,
            self.stall_cycles,
            self.overflow_count,
            self.retry_count,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Completion":
        _check_exact_size("completion record", data, COMPLETION_RECORD_BYTES)
        values = _COMPLETION_STRUCT.unpack(data)
        if values[3] != 0:
            raise ValueError("completion reserved field must be zero")
        completion = cls(
            command_id=values[0],
            node_id=values[1],
            status_code=values[2],
            pivot_count=values[4],
            start_cycle=values[5],
            finish_cycle=values[6],
            read_bytes=values[7],
            write_bytes=values[8],
            stall_cycles=values[9],
            overflow_count=values[10],
            retry_count=values[11],
        )
        completion.validate()
        return completion


def encode_token_states(states: Iterable[TokenState]) -> bytes:
    """Encode the fixture-only u32 token-state vector."""

    encoded = bytearray()
    for state in states:
        try:
            value = TokenState(state)
        except ValueError as exc:
            raise ValueError(f"unknown token state: {state!r}") from exc
        encoded.extend(_TOKEN_STATE_STRUCT.pack(value))
    return bytes(encoded)


def decode_token_states(data: bytes) -> Tuple[TokenState, ...]:
    """Decode the fixture-only u32 token-state vector."""

    if len(data) % _TOKEN_STATE_STRUCT.size:
        raise ValueError("token-state fixture size must be a multiple of 4 bytes")
    result = []
    for offset in range(0, len(data), _TOKEN_STATE_STRUCT.size):
        value = _TOKEN_STATE_STRUCT.unpack_from(data, offset)[0]
        try:
            result.append(TokenState(value))
        except ValueError as exc:
            raise ValueError(f"unknown token state: {value:#x}") from exc
    return tuple(result)


def validate_command_batch(
    commands: Sequence[Command],
    descriptors: Sequence[Descriptor],
    memory_image: bytes,
    *,
    token_count: int,
    max_wait_tokens: Optional[int] = None,
) -> None:
    """Validate v2 cross-record descriptor and token references.

    This validates the static batch contract only; it does not execute commands
    or implement the GCU token state machine.
    """

    _check_uint("token_count", token_count, _U32_MAX)
    seen_command_ids = set()
    signal_producers = set()
    for index, descriptor in enumerate(descriptors):
        try:
            descriptor.validate(len(memory_image))
        except ValueError as exc:
            raise ValueError(f"descriptor {index}: {exc}") from exc

    for index, command in enumerate(commands):
        try:
            command.validate()
        except ValueError as exc:
            raise ValueError(f"command {index}: {exc}") from exc
        if command.command_id == NONE:
            raise ValueError(f"command {index}: command_id must not be NONE")
        if command.command_id in seen_command_ids:
            raise ValueError(f"command {index}: duplicate command_id")
        seen_command_ids.add(command.command_id)

        if command.descriptor_id != NONE and command.descriptor_id >= len(descriptors):
            raise ValueError(f"command {index}: unknown descriptor_id")
        if command.wait_list_id != NONE:
            if command.wait_list_id >= len(descriptors):
                raise ValueError(f"command {index}: unknown wait_list_id")
            wait_descriptor = descriptors[command.wait_list_id]
            if wait_descriptor.descriptor_type != DescriptorType.DEPENDENCY_DESC:
                raise ValueError(f"command {index}: wait_list_id is not a DEPENDENCY_DESC")
            wait_descriptor.dependency_tokens(
                memory_image,
                token_count=token_count,
                max_wait_tokens=max_wait_tokens,
            )
        if command.signal_token != NONE:
            if command.signal_token >= token_count:
                raise ValueError(f"command {index}: unknown signal_token")
            if command.signal_token in signal_producers:
                raise ValueError(f"command {index}: duplicate signal_token producer")
            signal_producers.add(command.signal_token)


def encode_commands(commands: Iterable[Command]) -> bytes:
    return b"".join(command.to_bytes() for command in commands)


def decode_commands(data: bytes) -> Tuple[Command, ...]:
    if len(data) % COMMAND_RECORD_BYTES:
        raise ValueError("command region size must be a multiple of 32 bytes")
    return tuple(
        Command.from_bytes(data[offset : offset + COMMAND_RECORD_BYTES])
        for offset in range(0, len(data), COMMAND_RECORD_BYTES)
    )


def encode_descriptors(descriptors: Iterable[Descriptor]) -> bytes:
    return b"".join(descriptor.to_bytes() for descriptor in descriptors)


def decode_descriptors(
    data: bytes, *, memory_image_bytes: Optional[int] = None
) -> Tuple[Descriptor, ...]:
    if len(data) % DESCRIPTOR_RECORD_BYTES:
        raise ValueError("descriptor region size must be a multiple of 64 bytes")
    return tuple(
        Descriptor.from_bytes(
            data[offset : offset + DESCRIPTOR_RECORD_BYTES],
            memory_image_bytes=memory_image_bytes,
        )
        for offset in range(0, len(data), DESCRIPTOR_RECORD_BYTES)
    )


def encode_completions(completions: Iterable[Completion]) -> bytes:
    return b"".join(completion.to_bytes() for completion in completions)


def decode_completions(data: bytes) -> Tuple[Completion, ...]:
    if len(data) % COMPLETION_RECORD_BYTES:
        raise ValueError("completion region size must be a multiple of 64 bytes")
    return tuple(
        Completion.from_bytes(data[offset : offset + COMPLETION_RECORD_BYTES])
        for offset in range(0, len(data), COMPLETION_RECORD_BYTES)
    )
