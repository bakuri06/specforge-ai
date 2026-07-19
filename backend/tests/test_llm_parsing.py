import json

import pytest

from app.graph.llm import _parse_json_with_repair


def test_strips_think_block_before_extracting_json():
    """DeepSeek-R1 prefixes answers with <think>...</think> even under
    format: json, and the reasoning inside it can itself contain stray
    braces that would confuse naive brace-slicing if not stripped first."""
    content = (
        "<think>\n"
        "Let me reason about this... maybe the shape is {not: real}\n"
        "</think>\n"
        '{"ambiguous": false, "polished_spec": "Spec"}'
    )
    result = _parse_json_with_repair(content)
    assert result == {"ambiguous": False, "polished_spec": "Spec"}


def test_allows_literal_newline_inside_string_value():
    """A live run hit 'Invalid control character' because the model left a
    literal newline inside the polished_spec string instead of escaping it
    as \\n. json.loads(strict=False) must tolerate this rather than reject it."""
    content = '{"polished_spec": "## Overview\nSome text on a new line"}'
    result = _parse_json_with_repair(content)
    assert result["polished_spec"] == "## Overview\nSome text on a new line"


def test_both_quirks_combined():
    content = (
        "<think>thinking about {braces}</think>\n"
        '{"polished_spec": "## Overview\nLine two\nLine three"}'
    )
    result = _parse_json_with_repair(content)
    assert result["polished_spec"] == "## Overview\nLine two\nLine three"


def test_repairs_invalid_backslash_escape_in_string_value():
    """A live run hit 'Invalid \\escape' on both the original call AND the
    corrective retry: the model wrote a literal regex-style backslash (e.g.
    \\d for a 6-digit OTP pattern) inside a string value instead of escaping
    it as \\\\d. json.loads(strict=False) doesn't fix this - that flag only
    relaxes raw control characters, not malformed escape sequences - so
    retrying the model alone isn't reliable and the parser itself must repair it."""
    content = r'{"polished_spec": "OTP must match \d{6}"}'
    result = _parse_json_with_repair(content)
    assert result["polished_spec"] == "OTP must match \\d{6}"


def test_valid_escapes_are_left_untouched_while_invalid_ones_are_fixed():
    content = r'{"a": "line one\nline two, pattern \d{6}, quote \" done"}'
    result = _parse_json_with_repair(content)
    assert result["a"] == "line one\nline two, pattern \\d{6}, quote \" done"


def test_repairs_trailing_comma_before_closing_brace():
    content = '{"a": "one", "b": "two",}'
    result = _parse_json_with_repair(content)
    assert result == {"a": "one", "b": "two"}


def test_repairs_trailing_comma_before_closing_bracket():
    content = '{"items": ["a", "b",]}'
    result = _parse_json_with_repair(content)
    assert result == {"items": ["a", "b"]}


def test_repairs_trailing_comma_in_pretty_printed_multiline_json():
    """Matches the live shape of the bug: a genuinely multi-line, indented
    JSON response (raising 'Expecting value' several lines in) with a
    trailing comma before the closing brace - valid in a JS object literal,
    invalid in strict JSON, and a confusing error message that doesn't
    mention commas at all."""
    content = (
        "{\n"
        '  "ambiguous": false,\n'
        '  "questions": [],\n'
        '  "polished_spec": "Spec text",\n'
        "}\n"
    )
    result = _parse_json_with_repair(content)
    assert result == {"ambiguous": False, "questions": [], "polished_spec": "Spec text"}


def test_repairs_trailing_comma_and_invalid_escape_together():
    content = r'{"pattern": "\d{6}", "extra": "value",}'
    result = _parse_json_with_repair(content)
    assert result == {"pattern": "\\d{6}", "extra": "value"}


def test_raises_when_content_has_no_json_at_all():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_with_repair("just some prose, no braces here")
