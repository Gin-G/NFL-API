#!/usr/bin/env python3
"""
Tests for /chat/ endpoint.
"""

import os
import sys
from unittest.mock import MagicMock, patch


# ── Mock SDK helpers ──────────────────────────────────────────────────────────

class _TextBlock:
    """Mimic an anthropic TextBlock."""
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _ToolUseBlock:
    """Mimic an anthropic ToolUseBlock."""
    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict):
        self.id = id
        self.name = name
        self.input = input


class _SDKMessage:
    """Mimic an anthropic Messages response object."""

    def __init__(self, stop_reason: str, content):
        self.stop_reason = stop_reason
        self.content = content


def _make_anthropic_module(messages_side_effect):
    """Return a fake 'anthropic' module whose Anthropic().messages.create returns
    the given sequence of _SDKMessage objects (via side_effect)."""
    mock_create = MagicMock(side_effect=messages_side_effect)
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_module, mock_create


# ── No API key ────────────────────────────────────────────────────────────────

class TestChatNoApiKey:
    def test_returns_503(self, client):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            response = client.post("/chat/", json={"message": "hello", "history": []})
        assert response.status_code == 503

    def test_detail_mentions_api_key(self, client):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            body = client.post("/chat/", json={"message": "hello", "history": []}).json()
        assert "ANTHROPIC_API_KEY" in body.get("detail", "")


# ── Direct text response (no tool calls) ──────────────────────────────────────

class TestChatDirectResponse:
    """Claude returns end_turn immediately without calling any tools."""

    def _post(self, client, message, history=None, response_text="Great question!"):
        mock_module, _ = _make_anthropic_module(
            [_SDKMessage("end_turn", [_TextBlock(response_text)])]
        )
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test")
        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {"anthropic": mock_module}):
                return client.post(
                    "/chat/",
                    json={"message": message, "history": history or []},
                )

    def test_returns_200(self, client):
        assert self._post(client, "hi").status_code == 200

    def test_response_field_contains_text(self, client):
        body = self._post(client, "best QB?", response_text="Patrick Mahomes!").json()
        assert body["response"] == "Patrick Mahomes!"

    def test_history_has_user_and_assistant_roles(self, client):
        body = self._post(client, "best team?", response_text="KC!").json()
        roles = [m["role"] for m in body["history"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_prior_history_is_preserved(self, client):
        prior = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        body = self._post(
            client, "next question", history=prior, response_text="42"
        ).json()
        # prior 2 + new user + new assistant = 4
        assert len(body["history"]) == 4

    def test_history_messages_are_strings(self, client):
        body = self._post(client, "test").json()
        for msg in body["history"]:
            assert isinstance(msg["content"], str)

    def test_empty_history_accepted(self, client):
        assert self._post(client, "hello", history=[]).status_code == 200


# ── Tool-use agentic loop ─────────────────────────────────────────────────────

class TestChatToolUseLoop:
    """Claude calls a tool on the first turn, then responds with end_turn."""

    _TOOL_MSG = _SDKMessage(
        "tool_use",
        [_ToolUseBlock("tu_abc", "get_player_stats", {"season": 2024, "week": 4, "sort_by": "receiving_yards"})],
    )

    def _run(self, client, final_text="Travis Kelce had 167 yards."):
        end_msg = _SDKMessage("end_turn", [_TextBlock(final_text)])
        mock_module, mock_create = _make_anthropic_module([self._TOOL_MSG, end_msg])
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test")
        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {"anthropic": mock_module}):
                with patch(
                    "api.chat._execute_tool",
                    return_value=[{"player_name": "Travis Kelce", "receiving_yards": 167}],
                ):
                    resp = client.post(
                        "/chat/",
                        json={
                            "message": "who had the most receiving yards in week 4 of 2024?",
                            "history": [],
                        },
                    )
        return resp, mock_create

    def test_returns_200(self, client):
        resp, _ = self._run(client)
        assert resp.status_code == 200

    def test_response_text(self, client):
        expected = "Travis Kelce had 167 yards."
        resp, _ = self._run(client, final_text=expected)
        assert resp.json()["response"] == expected

    def test_messages_create_called_twice(self, client):
        _, mock_create = self._run(client)
        assert mock_create.call_count == 2

    def test_history_only_contains_strings(self, client):
        resp, _ = self._run(client)
        for msg in resp.json()["history"]:
            assert isinstance(msg["content"], str), (
                f"Non-string content in history: {msg}"
            )

    def test_execute_tool_receives_correct_name(self, client):
        end_msg = _SDKMessage("end_turn", [_TextBlock("done")])
        mock_module, _ = _make_anthropic_module([self._TOOL_MSG, end_msg])
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test")
        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {"anthropic": mock_module}):
                with patch("api.chat._execute_tool", return_value=[]) as mock_exec:
                    client.post("/chat/", json={"message": "test", "history": []})
        mock_exec.assert_called_once_with(
            "get_player_stats",
            {"season": 2024, "week": 4, "sort_by": "receiving_yards"},
            db=mock_exec.call_args.kwargs["db"],
        )


# ── Request validation ────────────────────────────────────────────────────────

class TestChatRequestValidation:
    def test_missing_message_returns_422(self, client):
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test")
        with patch.dict(os.environ, env):
            resp = client.post("/chat/", json={"history": []})
        assert resp.status_code == 422

    def test_invalid_history_role_still_posts(self, client):
        """Invalid history shapes should propagate as-is; backend doesn't validate role values."""
        end_msg = _SDKMessage("end_turn", [_TextBlock("ok")])
        mock_module, _ = _make_anthropic_module([end_msg])
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test")
        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {"anthropic": mock_module}):
                resp = client.post(
                    "/chat/",
                    json={
                        "message": "hi",
                        "history": [{"role": "user", "content": "prev"}],
                    },
                )
        assert resp.status_code == 200
