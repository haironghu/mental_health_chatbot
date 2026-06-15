"""消息防抖器测试。"""
import time
import threading

from app.whatsapp.debouncer import MessageDebouncer


class TestMessageDebouncer:
    def test_single_message_flushed_after_wait(self):
        """单条消息在等待时间后被处理。"""
        received = []
        d = MessageDebouncer(
            wait_seconds=0.2,
            on_flush=lambda uid, msg, raw: received.append((uid, msg, raw)),
        )
        d.add_message("user1", "hello", "raw1")
        time.sleep(0.4)
        assert received == [("user1", "hello", "raw1")]

    def test_consecutive_messages_merged(self):
        """连续多条消息合并为一条。"""
        received = []
        d = MessageDebouncer(
            wait_seconds=0.2,
            on_flush=lambda uid, msg, raw: received.append((uid, msg)),
        )
        d.add_message("user1", "我", "raw1")
        time.sleep(0.05)
        d.add_message("user1", "最近", "raw2")
        time.sleep(0.05)
        d.add_message("user1", "唔開心", "raw3")
        time.sleep(0.4)
        # 只应该触发一次，且消息已合并
        assert len(received) == 1
        assert received[0] == ("user1", "我\n最近\n唔開心")

    def test_different_users_independent(self):
        """不同用户的消息独立处理。"""
        received = []
        lock = threading.Lock()

        def on_flush(uid, msg, raw):
            with lock:
                received.append((uid, msg))

        d = MessageDebouncer(wait_seconds=0.2, on_flush=on_flush)
        d.add_message("user1", "hello", "raw1")
        d.add_message("user2", "你好", "raw2")
        time.sleep(0.4)
        # 两个用户都应触发，顺序不定
        assert len(received) == 2
        users = {r[0] for r in received}
        assert users == {"user1", "user2"}

    def test_last_raw_message_passed(self):
        """合并消息时，用最后一条原始消息（最新的）作为回复目标。"""
        received = []
        d = MessageDebouncer(
            wait_seconds=0.2,
            on_flush=lambda uid, msg, raw: received.append(raw),
        )
        d.add_message("user1", "a", "first")
        time.sleep(0.05)
        d.add_message("user1", "b", "second")
        time.sleep(0.05)
        d.add_message("user1", "c", "last")
        time.sleep(0.4)
        assert received == ["last"]

    def test_callback_exception_does_not_crash(self):
        """回调抛异常不应让 debouncer 崩溃。"""
        def bad_callback(uid, msg, raw):
            raise RuntimeError("boom")

        d = MessageDebouncer(wait_seconds=0.1, on_flush=bad_callback)
        # 不应抛出异常
        d.add_message("user1", "test", "raw")
        time.sleep(0.3)
        # 之后还能继续处理
        received = []
        d2 = MessageDebouncer(
            wait_seconds=0.1,
            on_flush=lambda uid, msg, raw: received.append(msg),
        )
        d2.add_message("user2", "ok", "raw")
        time.sleep(0.3)
        assert received == ["ok"]
