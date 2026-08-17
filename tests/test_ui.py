from __future__ import annotations

import unittest

from local_agent.ui import is_placeholder_key


class UITests(unittest.TestCase):
    def test_placeholder_key_detection_never_needs_the_real_key(self):
        self.assertTrue(is_placeholder_key("YOUR_GEMINI_API_KEY"))
        self.assertTrue(is_placeholder_key("..."))
        self.assertFalse(is_placeholder_key("AIza-example-not-a-placeholder"))
        self.assertFalse(is_placeholder_key(""))


if __name__ == "__main__":
    unittest.main()
