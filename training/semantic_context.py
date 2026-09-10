"""Normalized compiler/indexer context for the intelligence stage.

Teachers may attach this metadata to predicted records after detector decoding.
The editor still receives only ordinary Kotlin data objects and model outputs;
compiler tools are not runtime dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticContext:
    """Optional teacher/index context, using stable IDs rather than names."""
    scope_id: str | None = None
    scope_chain: tuple[str, ...] = ()
    type_name: str | None = None
    receiver_type: str | None = None
    import_path: str | None = None
    symbol_key: str | None = None
    is_write: bool = False
    is_static: bool = False

    def __post_init__(self):
        for value in (self.scope_id, self.type_name, self.receiver_type, self.import_path, self.symbol_key):
            if value is not None and not isinstance(value, str):
                raise ValueError("semantic context values must be strings or null")
        if any(not isinstance(value, str) or not value for value in self.scope_chain):
            raise ValueError("scope_chain entries must be nonempty strings")


def context_from_annotation(item: dict) -> SemanticContext:
    """Read optional indexer output without treating absent data as negative."""
    value = item.get("semanticContext", {})
    if not isinstance(value, dict):
        raise ValueError("semanticContext must be an object")
    chain = value.get("scopeChain", ())
    if isinstance(chain, list):
        chain = tuple(chain)
    if not isinstance(chain, tuple):
        raise ValueError("scopeChain must be an array")
    return SemanticContext(
        scope_id=value.get("scopeId"), scope_chain=chain,
        type_name=value.get("type"), receiver_type=value.get("receiverType"),
        import_path=value.get("importPath"), symbol_key=value.get("symbolKey"),
        is_write=bool(value.get("isWrite", False)), is_static=bool(value.get("isStatic", False)),
    )
