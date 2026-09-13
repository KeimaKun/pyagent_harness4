import unittest

from harness.protocol import parse_action


class TestParseAction(unittest.TestCase):
    def test_parses_clean_tool_call(self):
        text = '```action\n{"thought": "look around", "tool": "list_dir", "args": {"path": "."}}\n```'
        action = parse_action(text)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.tool, "list_dir")
        self.assertEqual(action.args, {"path": "."})
        self.assertEqual(action.thought, "look around")

    def test_parses_final(self):
        text = '```action\n{"thought": "done", "final": "All finished."}\n```'
        action = parse_action(text)
        self.assertEqual(action.kind, "final")
        self.assertEqual(action.final_text, "All finished.")

    def test_tolerates_surrounding_prose(self):
        text = 'Sure, here is my action:\n```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```\nthanks'
        action = parse_action(text)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.tool, "read_file")

    def test_bare_json_without_fence(self):
        text = '{"tool": "grep_search", "args": {"pattern": "TODO"}}'
        action = parse_action(text)
        self.assertEqual(action.kind, "tool")

    def test_prefers_last_block_when_multiple(self):
        text = (
            '```action\n{"tool": "list_dir", "args": {}}\n```\n'
            'wait, actually:\n'
            '```action\n{"tool": "read_file", "args": {"path": "b.py"}}\n```'
        )
        action = parse_action(text)
        self.assertEqual(action.tool, "read_file")

    def test_invalid_json_is_reported(self):
        action = parse_action("I refuse to use the protocol.")
        self.assertEqual(action.kind, "invalid")
        self.assertIsNotNone(action.error)

    def test_missing_tool_and_final_is_invalid(self):
        action = parse_action('{"thought": "hmm"}')
        self.assertEqual(action.kind, "invalid")

    def test_non_dict_args_is_invalid(self):
        action = parse_action('{"tool": "list_dir", "args": "not a dict"}')
        self.assertEqual(action.kind, "invalid")


if __name__ == "__main__":
    unittest.main()
