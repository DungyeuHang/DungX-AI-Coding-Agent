from __future__ import annotations

import re
from typing import Any

try:
    from tree_sitter import Node
except ImportError:  # Tree-sitter is optional until semantic indexing is enabled.
    Node = Any  # type: ignore[misc,assignment]

from ..models import SymbolDefinition, SymbolLocation
from .parser import TreeSitterParser


_IMPORT_RE = re.compile(r"(?:from\s*|import\s*|require\s*\()(['\"])([^'\"]+)\1")


class JavaScriptIndexer:
    """Indexes JavaScript, JSX, TypeScript, and TSX declaration symbols."""

    def __init__(self, parser: TreeSitterParser):
        self.parser = parser

    def index(self, content: bytes, language: str = "javascript") -> tuple[list[SymbolDefinition], list[str]]:
        tree = self.parser.parse(content, language)
        symbols: list[SymbolDefinition] = []
        self._walk(tree.root_node, symbols, [])
        text = content.decode("utf-8", errors="replace")
        imports = [match.group(2) for match in _IMPORT_RE.finditer(text)]
        symbols.sort(key=lambda symbol: (symbol.location.start_line, symbol.location.end_line, symbol.name, symbol.kind))
        return symbols, imports

    def _walk(self, node: Node, symbols: list[SymbolDefinition], parent_stack: list[str]) -> None:
        node_type = node.type
        if node_type in {"class_declaration", "class"}:
            name = self._named_child(node, "name")
            if name:
                class_name = self._text(name)
                symbols.append(self._definition(class_name, "class", node, parent_stack[-1] if parent_stack else None))
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._walk(child, symbols, parent_stack + [class_name])
                return

        if node_type in {"function_declaration", "generator_function_declaration"}:
            name = self._named_child(node, "name")
            if name:
                function_name = self._text(name)
                kind = "method" if parent_stack else "function"
                symbols.append(self._definition(function_name, kind, node, parent_stack[-1] if parent_stack else None))
            return

        if node_type == "method_definition":
            name = self._named_child(node, "name")
            if name:
                method_name = self._text(name)
                symbols.append(self._definition(method_name, "method", node, parent_stack[-1] if parent_stack else None))
            value = node.child_by_field_name("value")
            if value:
                for child in value.children:
                    self._walk(child, symbols, parent_stack)
            return

        if node_type == "variable_declarator":
            name = self._named_child(node, "name")
            value = node.child_by_field_name("value")
            if name and value and value.type in {"arrow_function", "function", "function_expression", "class"}:
                symbol_name = self._text(name)
                kind = "class" if value.type == "class" else "function"
                symbols.append(self._definition(symbol_name, kind, node, parent_stack[-1] if parent_stack else None))
            if value:
                for child in value.children:
                    self._walk(child, symbols, parent_stack)
            return

        for child in node.children:
            self._walk(child, symbols, parent_stack)

    @staticmethod
    def _named_child(node: Node, field: str) -> Node | None:
        child = node.child_by_field_name(field)
        if child is not None:
            return child
        for candidate in node.named_children:
            if candidate.type in {"identifier", "property_identifier", "type_identifier"}:
                return candidate
        return None

    @staticmethod
    def _text(node: Node) -> str:
        return node.text.decode("utf-8", errors="replace")

    @staticmethod
    def _definition(name: str, kind: str, node: Node, parent: str | None) -> SymbolDefinition:
        return SymbolDefinition(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            location=SymbolLocation(start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1),
            parent=parent,
        )
