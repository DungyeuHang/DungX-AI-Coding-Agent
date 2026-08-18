from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from tree_sitter import Language, Parser
except ImportError:  # Tree-sitter is optional until semantic indexing is enabled.
    Language = None  # type: ignore[assignment,misc]
    Parser = None  # type: ignore[assignment,misc]

# Logic to find the build/my-languages.so file
# This might need adjustment based on the actual build process
# For now, let's assume it's in a 'build' directory sibling to 'local_agent'
VENDOR_PATH = Path(__file__).parent.parent.parent / "vendor"
BUILD_PATH = Path(__file__).parent.parent.parent / "build"
LANGUAGE_SO_PATH = BUILD_PATH / "languages.so"

class TreeSitterParser:
    def __init__(self):
        if Language is None or Parser is None:
            raise ImportError("tree-sitter is not installed")
        if not LANGUAGE_SO_PATH.exists():
            raise FileNotFoundError(
                f"Language library not found at {LANGUAGE_SO_PATH}. "
                "Please run the build script to compile the tree-sitter grammars."
            )
        
        self._language_names = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "tsx": "tsx",
        }
        self._languages: dict[str, Any] = {}
        self._parsers: dict[str, Parser] = {}

    def get_parser(self, language: str) -> Parser:
        if language not in self._language_names:
            raise ValueError(f"Unsupported language for Tree-sitter parser: {language}")
        if language not in self._parsers:
            language_obj = self._languages.get(language)
            if language_obj is None:
                language_obj = Language(LANGUAGE_SO_PATH.resolve(), self._language_names[language])
                self._languages[language] = language_obj
            parser = Parser()
            parser.set_language(language_obj)
            self._parsers[language] = parser
        return self._parsers[language]

    def parse(self, content: bytes, language: str) -> Any:
        parser = self.get_parser(language)
        return parser.parse(content)
