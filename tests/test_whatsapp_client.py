"""
WhatsApp 客户端消息处理测试（mock neonize 和 orchestrator）。
不实际连接 WhatsApp，只测试消息解析和处理逻辑。
"""
from unittest.mock import MagicMock, patch

from app.whatsapp.client import (
    _extract_text,
    _get_sender_phone,
    _is_from_me,
    _split_response,
    _typing_delay,
)
from app.config import settings


def _make_message_ev(
    text: str = "hello",
    sender_user: str = "85298765432",
    is_from_me: bool = False,
    is_extended: bool = False,
) -> MagicMock:
    """构造一个模拟的 neonize MessageEv 对象。"""
    msg_ev = MagicMock()

    # message.Info.MessageSource
    msg_ev.Info.MessageSource.Sender.User = sender_user
    msg_ev.Info.MessageSource.IsFromMe = is_from_me

    # message.Message（普通文本 vs extendedTextMessage）
    if is_extended:
        msg_ev.Message.conversation = ""
        msg_ev.Message.extendedTextMessage.text = text
    else:
        msg_ev.Message.conversation = text
        msg_ev.Message.extendedTextMessage = None

    return msg_ev


class TestExtractText:
    def test_normal_text(self):
        msg = _make_message_ev(text="你好")
        assert _extract_text(msg) == "你好"

    def test_extended_text(self):
        msg = _make_message_ev(text="引用回复", is_extended=True)
        assert _extract_text(msg) == "引用回复"

    def test_no_text_returns_none(self):
        msg = MagicMock()
        msg.Message.conversation = ""
        msg.Message.extendedTextMessage = None
        assert _extract_text(msg) is None

    def test_empty_conversation_with_no_extended(self):
        msg = MagicMock()
        msg.Message.conversation = ""
        msg.Message.extendedTextMessage = MagicMock()
        msg.Message.extendedTextMessage.text = ""
        assert _extract_text(msg) is None


class TestGetSenderPhone:
    def test_extracts_phone(self):
        msg = _make_message_ev(sender_user="85298765432")
        assert _get_sender_phone(msg) == "85298765432"

    def test_different_country(self):
        msg = _make_message_ev(sender_user="8613800138000")
        assert _get_sender_phone(msg) == "8613800138000"


class TestIsFromMe:
    def test_from_me(self):
        msg = _make_message_ev(is_from_me=True)
        assert _is_from_me(msg) is True

    def test_from_other(self):
        msg = _make_message_ev(is_from_me=False)
        assert _is_from_me(msg) is False


class TestSplitResponse:
    def test_no_marker_returns_single(self):
        assert _split_response("一段話") == ["一段話"]

    def test_split_on_double_pipe(self):
        result = _split_response("第一段||第二段||第三段")
        assert result == ["第一段", "第二段", "第三段"]

    def test_strips_whitespace(self):
        result = _split_response("第一段 || 第二段 ")
        assert result == ["第一段", "第二段"]

    def test_drops_empty_chunks(self):
        result = _split_response("第一段|| ||第二段")
        assert result == ["第一段", "第二段"]


class TestTypingDelay:
    def test_short_text_uses_min(self):
        # 短文本不应低于最小延迟
        d = _typing_delay("hi")
        assert d == settings.min_typing_delay_seconds

    def test_long_text_capped(self):
        # 超长文本不应超过最大延迟
        d = _typing_delay("a" * 1000)
        assert d == settings.max_typing_delay_seconds

    def test_proportional_to_length(self):
        # 中等长度按字数计算
        text = "a" * 40
        expected = 40 * settings.typing_seconds_per_char
        # 必须落在 [min, max] 之间
        if settings.min_typing_delay_seconds <= expected <= settings.max_typing_delay_seconds:
            assert _typing_delay(text) == expected


class TestHandleMessage:
    """测试 _handle_message 把消息加入防抖缓冲。"""

    def test_ignores_own_messages(self):
        from app.whatsapp.client import _handle_message
        import app.whatsapp.client as client_mod

        mock_debouncer = MagicMock()
        with patch.object(client_mod, "_debouncer", mock_debouncer):
            client = MagicMock()
            msg = _make_message_ev(is_from_me=True)
            _handle_message(client, msg)
            mock_debouncer.add_message.assert_not_called()
            client.reply_message.assert_not_called()

    def test_replies_to_non_text_immediately(self):
        from app.whatsapp.client import _handle_message
        import app.whatsapp.client as client_mod

        mock_debouncer = MagicMock()
        with patch.object(client_mod, "_debouncer", mock_debouncer):
            client = MagicMock()
            msg = MagicMock()
            msg.Info.MessageSource.IsFromMe = False
            msg.Message.conversation = ""
            msg.Message.extendedTextMessage = None

            _handle_message(client, msg)
            # 非文本不进防抖，直接回复
            mock_debouncer.add_message.assert_not_called()
            client.reply_message.assert_called_once()
            assert "文字" in client.reply_message.call_args[0][0]

    def test_text_message_goes_to_debouncer(self):
        from app.whatsapp.client import _handle_message
        import app.whatsapp.client as client_mod

        mock_debouncer = MagicMock()
        with patch.object(client_mod, "_debouncer", mock_debouncer):
            client = MagicMock()
            msg = _make_message_ev(text="你好", sender_user="85298765432")
            _handle_message(client, msg)
            mock_debouncer.add_message.assert_called_once_with(
                "85298765432", "你好", msg
            )
            client.reply_message.assert_not_called()
