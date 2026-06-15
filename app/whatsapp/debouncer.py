"""
消息防抖（debounce）：
用户连续发多条消息时，等待用户停顿后合并成一条再处理。
这样可以避免对每条短消息都单独回复，更接近真人聊天体验。

工作原理：
- 每个用户维护一个消息缓冲区 + 定时器
- 收到消息 → 加入缓冲 + 重置定时器
- 定时器到时 → 合并缓冲消息 → 触发回调 → 清空缓冲
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class _UserBuffer:
    """单个用户的消息缓冲。"""
    messages: list[str] = field(default_factory=list)
    timer: threading.Timer | None = None
    # 最后一条消息的原始对象（neonize MessageEv），用于回复时引用
    last_raw_message: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class MessageDebouncer:
    """
    线程安全的消息防抖器。

    add_message(user_id, text, raw_message) 添加消息；
    若 wait_seconds 内无新消息到达，则调用 on_flush(user_id, merged_text, raw_message)。
    """

    def __init__(
        self,
        wait_seconds: float,
        on_flush: Callable[[str, str, Any], None],
    ):
        self.wait_seconds = wait_seconds
        self.on_flush = on_flush
        self._buffers: dict[str, _UserBuffer] = {}
        self._global_lock = threading.Lock()

    def _get_buffer(self, user_id: str) -> _UserBuffer:
        with self._global_lock:
            if user_id not in self._buffers:
                self._buffers[user_id] = _UserBuffer()
            return self._buffers[user_id]

    def add_message(self, user_id: str, text: str, raw_message: Any) -> None:
        """添加一条消息到防抖缓冲。"""
        buf = self._get_buffer(user_id)
        with buf.lock:
            buf.messages.append(text)
            buf.last_raw_message = raw_message
            # 取消旧定时器
            if buf.timer is not None:
                buf.timer.cancel()
            # 启动新定时器
            buf.timer = threading.Timer(
                self.wait_seconds, self._flush, args=(user_id,)
            )
            buf.timer.daemon = True
            buf.timer.start()
            logger.debug(
                "用户 %s 缓冲消息（共 %d 条），%.1fs 后处理",
                user_id, len(buf.messages), self.wait_seconds,
            )

    def _flush(self, user_id: str) -> None:
        """定时器触发时调用：合并缓冲消息并交给回调。"""
        buf = self._get_buffer(user_id)
        with buf.lock:
            if not buf.messages:
                return
            merged = "\n".join(buf.messages)
            raw = buf.last_raw_message
            buf.messages = []
            buf.last_raw_message = None
            buf.timer = None
        try:
            self.on_flush(user_id, merged, raw)
        except Exception:
            logger.exception("处理用户 %s 的合并消息时出错", user_id)
