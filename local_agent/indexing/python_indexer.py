from __future__ import annotations

from typing import Any

try:
    from tree_sitter import Node
except ImportError:  # Tree-sitter is optional until semantic indexing is enabled.
    Node = Any  # type: ignore[misc,assignment]

from ..models import SymbolDefinition, SymbolLocation
from .parser import TreeSitterParser


class PythonIndexer:
    def __init__(self, parser: TreeSitterParser):
        self.parser = parser

    def index(self, content: bytes) -> tuple[list[SymbolDefinition], list[str]]:
        tree = self.parser.parse(content, "python")
        symbols: list[SymbolDefinition] = []
        imports: list[str] = []
        
        self._find_symbols_and_imports(tree.root_node, symbols, imports, [])
        
        return symbols, imports

    def _find_symbols_and_imports(self, node: Node, symbols: list[SymbolDefinition], imports: list[str], parent_stack: list[str]):
        if node.type == 'class_definition':
            name_node = node.child_by_field_name('name')
            if name_node:
                class_name = name_node.text.decode('utf8')
                location = SymbolLocation(start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)
                parent_name = parent_stack[-1] if parent_stack else None
                symbols.append(SymbolDefinition(name=class_name, kind='class', location=location, parent=parent_name))
                
                new_parent_stack = parent_stack + [class_name]
                body_node = node.child_by_field_name('body')
                if body_node:
                    for child in body_node.children:
                        self._find_symbols_and_imports(child, symbols, imports, new_parent_stack)
                return # Avoid processing children again

        elif node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            if name_node:
                func_name = name_node.text.decode('utf8')
                kind = 'method' if parent_stack else 'function'
                location = SymbolLocation(start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)
                parent_name = parent_stack[-1] if parent_stack else None
                symbols.append(SymbolDefinition(name=func_name, kind=kind, location=location, parent=parent_name))
                return

        elif node.type in ('import_statement', 'import_from_statement'):
            module_name_node = node.child_by_field_name('module_name')
            if module_name_node:
                imports.append(module_name_node.text.decode('utf8'))
            return

        for child in node.children:
            self._find_symbols_and_imports(child, symbols, imports, parent_stack)
