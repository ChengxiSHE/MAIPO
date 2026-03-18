# logger_utils.py （建议文件名）

import logging
import os
from datetime import datetime

# 全局 logger 实例
_logger = None

def get_logger():
    global _logger
    if _logger is not None:
        return _logger

    # 创建 logs 目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成本次运行的唯一日志文件名（精确到秒）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # 创建 logger
    logger = logging.getLogger("my_app_logger")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler（重要！）
    if logger.handlers:
        logger.handlers.clear()

    # 创建 FileHandler（不是 TimedRotating！）
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # 可选：避免日志被 root logger 重复打印

    _logger = logger
    return _logger