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


def test_raises_when_content_has_no_json_at_all():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_with_repair("just some prose, no braces here")
