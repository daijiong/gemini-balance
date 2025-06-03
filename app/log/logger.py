import logging
import logging.handlers
import os
import platform
import sys
from typing import Dict, Optional

# ANSI转义序列颜色代码
COLORS = {
    "DEBUG": "\033[34m",  # 蓝色
    "INFO": "\033[32m",  # 绿色
    "WARNING": "\033[33m",  # 黄色
    "ERROR": "\033[31m",  # 红色
    "CRITICAL": "\033[1;31m",  # 红色加粗
}

# Windows系统启用ANSI支持
if platform.system() == "Windows":
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


class ColoredFormatter(logging.Formatter):
    """
    自定义的日志格式化器,添加颜色支持（仅用于控制台输出）
    """

    def format(self, record):
        # 创建记录的副本，避免修改原始记录
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # 获取对应级别的颜色代码
        color = COLORS.get(record_copy.levelname, "")
        # 添加颜色代码和重置代码
        record_copy.levelname = f"{color}{record_copy.levelname}\033[0m"
        # 创建包含文件名和行号的固定宽度字符串
        record_copy.fileloc = f"[{record_copy.filename}:{record_copy.lineno}]"
        return super().format(record_copy)


class PlainFormatter(logging.Formatter):
    """
    普通格式化器（不包含颜色代码，用于文件输出）
    """

    def format(self, record):
        # 创建包含文件名和行号的固定宽度字符串
        record.fileloc = f"[{record.filename}:{record.lineno}]"
        return super().format(record)


# 控制台日志格式（带颜色）
CONSOLE_FORMATTER = ColoredFormatter(
    "%(asctime)s | %(levelname)-17s | %(fileloc)-30s | %(message)s"
)

# 文件日志格式（无颜色）
FILE_FORMATTER = PlainFormatter(
    "%(asctime)s | %(levelname)-8s | %(fileloc)-30s | %(message)s"
)

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class Logger:
    def __init__(self):
        pass

    _loggers: Dict[str, logging.Logger] = {}

    @staticmethod
    def _create_log_directory():
        """
        创建日志目录
        """
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """
        设置并获取logger，支持控制台和文件双输出
        :param name: logger名称
        :return: logger实例
        """
        # 导入 settings 对象
        from app.config.config import settings

        # 从全局配置获取日志级别
        log_level_str = settings.LOG_LEVEL.lower()
        level = LOG_LEVELS.get(log_level_str, logging.INFO)

        if name in Logger._loggers:
            # 如果 logger 已存在，检查并更新其级别（如果需要）
            existing_logger = Logger._loggers[name]
            if existing_logger.level != level:
                existing_logger.setLevel(level)
            return existing_logger

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        # 清除现有的处理器（防止重复添加）
        logger.handlers.clear()

        # 添加控制台输出处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CONSOLE_FORMATTER)
        logger.addHandler(console_handler)

        # 创建日志目录
        log_dir = Logger._create_log_directory()

        # 添加文件输出处理器（主日志文件，所有级别）
        try:
            main_log_file = os.path.join(log_dir, "gemini-balance.log")
            # 使用 RotatingFileHandler 实现日志轮转
            # 每个文件最大10MB，保留5个备份文件
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(FILE_FORMATTER)
            logger.addHandler(file_handler)

            # 添加错误日志文件处理器（仅ERROR和CRITICAL级别）
            error_log_file = os.path.join(log_dir, "error.log")
            error_file_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.setFormatter(FILE_FORMATTER)
            logger.addHandler(error_file_handler)

            # 为每个模块创建专门的日志文件（可选）
            if name in ["openai", "gemini", "chat", "request", "security"]:
                module_log_file = os.path.join(log_dir, f"{name}.log")
                module_file_handler = logging.handlers.RotatingFileHandler(
                    module_log_file,
                    maxBytes=5*1024*1024,  # 5MB
                    backupCount=3,
                    encoding='utf-8'
                )
                module_file_handler.setFormatter(FILE_FORMATTER)
                logger.addHandler(module_file_handler)

        except Exception as e:
            # 如果文件处理器创建失败，至少保证控制台输出可用
            print(f"Warning: Failed to create file handler for logger '{name}': {e}")

        Logger._loggers[name] = logger
        return logger

    @staticmethod
    def get_logger(name: str) -> Optional[logging.Logger]:
        """
        获取已存在的logger
        :param name: logger名称
        :return: logger实例或None
        """
        return Logger._loggers.get(name)

    @staticmethod
    def update_log_levels(log_level: str):
        """
        根据当前的全局配置更新所有已创建 logger 的日志级别。
        """
        log_level_str = log_level.lower()
        new_level = LOG_LEVELS.get(log_level_str, logging.INFO)

        updated_count = 0
        for logger_name, logger_instance in Logger._loggers.items():
            if logger_instance.level != new_level:
                logger_instance.setLevel(new_level)
                # 同时更新所有处理器的级别（除了专门设置为ERROR的错误日志处理器）
                for handler in logger_instance.handlers:
                    if not (isinstance(handler, logging.handlers.RotatingFileHandler) 
                           and handler.baseFilename.endswith("error.log")):
                        handler.setLevel(new_level)
                updated_count += 1


# 预定义的loggers
def get_openai_logger():
    return Logger.setup_logger("openai")


def get_gemini_logger():
    return Logger.setup_logger("gemini")


def get_chat_logger():
    return Logger.setup_logger("chat")


def get_model_logger():
    return Logger.setup_logger("model")


def get_security_logger():
    return Logger.setup_logger("security")


def get_key_manager_logger():
    return Logger.setup_logger("key_manager")


def get_main_logger():
    return Logger.setup_logger("main")


def get_embeddings_logger():
    return Logger.setup_logger("embeddings")


def get_request_logger():
    return Logger.setup_logger("request")


def get_retry_logger():
    return Logger.setup_logger("retry")


def get_image_create_logger():
    return Logger.setup_logger("image_create")


def get_exceptions_logger():
    return Logger.setup_logger("exceptions")


def get_application_logger():
    return Logger.setup_logger("application")


def get_initialization_logger():
    return Logger.setup_logger("initialization")


def get_middleware_logger():
    return Logger.setup_logger("middleware")


def get_routes_logger():
    return Logger.setup_logger("routes")


def get_config_routes_logger():
    return Logger.setup_logger("config_routes")


def get_config_logger():
    return Logger.setup_logger("config")


def get_database_logger():
    return Logger.setup_logger("database")


def get_log_routes_logger():
    return Logger.setup_logger("log_routes")


def get_stats_logger():
    return Logger.setup_logger("stats")


def get_update_logger():
    return Logger.setup_logger("update_service")


def get_scheduler_routes():
    return Logger.setup_logger("scheduler_routes")


def get_message_converter_logger():
    return Logger.setup_logger("message_converter")


def get_api_client_logger():
    return Logger.setup_logger("api_client")


def get_openai_compatible_logger():
    return Logger.setup_logger("openai_compatible")


def get_error_log_logger():
    return Logger.setup_logger("error_log")


def get_request_log_logger():
    return Logger.setup_logger("request_log")


def get_vertex_express_logger():
    return Logger.setup_logger("vertex_express")

