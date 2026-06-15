"""
心理健康 WhatsApp 聊天机器人入口。
通过 neonize 直连 WhatsApp Web 多设备协议。

启动：python main.py
首次运行时终端会显示 QR 码，用手机 WhatsApp 扫码链接。
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from app.whatsapp.client import run

if __name__ == "__main__":
    run()
