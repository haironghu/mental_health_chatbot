"""
基于 neonize (whatsmeow Python 绑定) 的 WhatsApp 客户端。
直连 WhatsApp Web 多设备协议，无需 Twilio / webhook / ngrok。

首次运行时终端会显示 QR 码，用手机 WhatsApp 扫码链接。
会话密钥保存在 SQLite 文件中，后续启动自动重连。

特性：
- 消息防抖：连续多条短消息合并处理，等用户讲完再回复
- 分段回复：LLM 用 `||` 标记自然分段，机器人会分多条短消息发出
- 打字模拟：发送前显示「正在输入...」，按字数延迟，更像真人
"""
import io
import logging
import signal
import sys
import time

from neonize.client import NewClient
from neonize.events import ConnectedEv, MessageEv, PairStatusEv, event
from neonize.utils import log as neonize_log
from neonize.utils.enum import ChatPresence, ChatPresenceMedia

from app.config import settings
from app.orchestrator import orchestrator
from app.whatsapp.debouncer import MessageDebouncer

logger = logging.getLogger(__name__)

# 降低 neonize 内部日志等级
neonize_log.setLevel(logging.WARNING)

# 全局客户端实例
_client: NewClient | None = None

# 全局防抖器（在 create_client 时初始化）
_debouncer: MessageDebouncer | None = None


def _extract_text(message: MessageEv) -> str | None:
    """
    从 neonize MessageEv 中提取纯文本。
    文本可能在 conversation 或 extendedTextMessage.text 中。
    非文本消息（图片、语音等）返回 None。
    """
    msg = message.Message
    # 普通文本
    if msg.conversation:
        return msg.conversation
    # 引用回复 / 链接预览消息
    if msg.extendedTextMessage and msg.extendedTextMessage.text:
        return msg.extendedTextMessage.text
    return None


def _get_sender_phone(message: MessageEv) -> str:
    """
    获取发送者的手机号（不含 @s.whatsapp.net 后缀）。
    格式示例：85298765432（无 + 前缀）
    """
    return message.Info.MessageSource.Sender.User


def _is_from_me(message: MessageEv) -> bool:
    """判断消息是否是自己发的（避免回复自己）。"""
    return message.Info.MessageSource.IsFromMe


def _show_qr(data: str) -> None:
    """
    在终端显示 QR 码。
    使用 segno 生成，输出到 UTF-8 流以绕过 Windows GBK 编码问题。
    """
    try:
        import segno
        qr = segno.make_qr(data)
        buf = io.StringIO()
        qr.terminal(compact=True, out=buf)
        # 用 UTF-8 强制写入终端
        sys.stdout.buffer.write(buf.getvalue().encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:
        # 兜底：直接打印原始数据，用户可以复制到在线 QR 生成器
        logger.warning("QR 码渲染失败，请复制以下内容到在线 QR 生成器扫码：")
        print(data)


def _split_response(text: str) -> list[str]:
    """
    将 LLM 回复按 `||` 分段标记拆分成多条短消息。
    没有标记则返回单条。
    自动清理空段和首尾空白。
    """
    parts = [p.strip() for p in text.split("||") if p.strip()]
    return parts if parts else [text.strip()]


def _typing_delay(chunk: str) -> float:
    """根据消息长度计算"打字"延迟（秒）。"""
    delay = len(chunk) * settings.typing_seconds_per_char
    return max(
        settings.min_typing_delay_seconds,
        min(delay, settings.max_typing_delay_seconds),
    )


def _send_with_typing(
    client: NewClient, message: MessageEv, chunks: list[str]
) -> None:
    """
    按段发送回复，段间显示「正在输入」状态并按字数延迟。
    第一段用 reply_message（带引用），后续用 send_message。
    """
    jid = message.Info.MessageSource.Chat
    for i, chunk in enumerate(chunks):
        # 显示「正在输入」
        try:
            client.send_chat_presence(
                jid,
                ChatPresence.CHAT_PRESENCE_COMPOSING,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )
        except Exception:
            pass  # 状态发送失败不影响主流程

        time.sleep(_typing_delay(chunk))

        # 停止「正在输入」
        try:
            client.send_chat_presence(
                jid,
                ChatPresence.CHAT_PRESENCE_PAUSED,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )
        except Exception:
            pass

        # 第一段用 reply 引用用户消息，后续段直接发送
        if i == 0:
            client.reply_message(chunk, message)
        else:
            client.send_message(jid, chunk)


def _process_merged_message(user_id: str, merged_text: str, raw_message) -> None:
    """
    防抖回调：处理用户合并后的消息。
    """
    if _client is None or raw_message is None:
        return

    phone = user_id
    logger.info("处理合并消息 [%s]: %s", phone, merged_text[:80].replace("\n", " | "))

    try:
        result = orchestrator.process(phone=phone, user_message=merged_text)
        chunks = _split_response(result.response_text)
        _send_with_typing(_client, raw_message, chunks)
        logger.info(
            "已回复 [%s] state=%s risk=%.1f level=%s K6=%d/%s%s chunks=%d",
            phone, result.state, result.risk_score, result.alert_level,
            result.k6_total, result.k6_severity,
            "(已完成)" if result.k6_complete else "(进行中)",
            len(chunks),
        )
    except Exception:
        logger.exception("处理消息时出错 [%s]", phone)
        try:
            _client.reply_message(
                "唔好意思，我暫時出咗少少問題，請稍後再試 🙏",
                raw_message,
            )
        except Exception:
            logger.exception("发送错误提示也失败了 [%s]", phone)


def create_client() -> NewClient:
    """创建并配置 neonize 客户端。"""
    global _client, _debouncer
    client = NewClient(settings.whatsapp_db)

    # 初始化防抖器
    _debouncer = MessageDebouncer(
        wait_seconds=settings.debounce_seconds,
        on_flush=_process_merged_message,
    )

    @client.qr
    def on_qr(c: NewClient, qr: bytes):
        logger.info("请用 WhatsApp 扫描以下 QR 码：")
        _show_qr(qr.decode("utf-8"))

    @client.event(PairStatusEv)
    def on_pair(c: NewClient, msg: PairStatusEv):
        logger.info("WhatsApp 配对成功: %s", msg.ID.User)

    @client.event(ConnectedEv)
    def on_connected(c: NewClient, _: ConnectedEv):
        logger.info("WhatsApp 已连接，等待消息...")

    @client.event(MessageEv)
    def on_message(c: NewClient, message: MessageEv):
        _handle_message(c, message)

    _client = client
    return client


def _handle_message(client: NewClient, message: MessageEv) -> None:
    """
    收到消息 → 加入防抖缓冲，等待用户讲完再统一处理。
    """
    # 忽略自己发的消息
    if _is_from_me(message):
        return

    text = _extract_text(message)
    if text is None:
        # 非文本消息，立即回复提示（不走防抖）
        client.reply_message(
            "唔好意思，我暫時只能睇文字消息，你可以打字同我傾計嗎？😊",
            message,
        )
        return

    phone = _get_sender_phone(message)
    logger.info("收到消息 [%s]: %s", phone, text[:50])

    # 加入防抖缓冲，等待 N 秒后统一处理
    if _debouncer is not None:
        _debouncer.add_message(phone, text, message)


def run() -> None:
    """启动 WhatsApp 客户端（阻塞运行）。"""
    # 优雅退出
    def interrupted(*_):
        event.set()
    signal.signal(signal.SIGINT, interrupted)

    client = create_client()
    logger.info("正在连接 WhatsApp... 如果是首次运行，请扫描终端中的 QR 码。")
    client.connect()
