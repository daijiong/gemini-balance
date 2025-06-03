#!/usr/bin/env python3
"""
日志功能测试脚本
用于演示新的文件日志功能
"""

import os
import sys

def get_project_root():
    """获取项目根目录"""
    # 获取脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果脚本在app/log目录中，项目根目录是上两级
    if script_dir.endswith(os.path.join('app', 'log')):
        return os.path.join(script_dir, '..', '..')
    
    # 如果从项目根目录运行，则当前目录就是项目根目录
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'app', 'log', 'logger.py')):
        return current_dir
    
    # 默认情况下，假设脚本在app/log目录中
    return os.path.join(script_dir, '..', '..')

# 添加项目根目录到Python路径
project_root = get_project_root()
sys.path.append(project_root)

# 模拟一个简单的配置类，避免依赖完整的应用配置
class MockSettings:
    LOG_LEVEL = "INFO"

# 创建一个模拟的配置模块
import types
config_module = types.ModuleType('config')
config_module.settings = MockSettings()
sys.modules['app.config.config'] = config_module

# 现在可以安全地导入logger
from app.log.logger import Logger

def test_logging():
    """测试日志功能"""
    print("🚀 开始测试日志功能...")
    
    # 获取不同类型的logger
    main_logger = Logger.setup_logger("main")
    chat_logger = Logger.setup_logger("chat")
    error_logger = Logger.setup_logger("error_test")
    
    # 记录不同级别的日志
    main_logger.info("这是一条普通的信息日志")
    main_logger.debug("这是一条调试日志（可能不会显示，取决于日志级别）")
    main_logger.warning("这是一条警告日志")
    
    chat_logger.info("用户开始了一次聊天会话")
    chat_logger.info("AI正在处理用户的问题...")
    chat_logger.info("AI已成功响应用户")
    
    error_logger.error("这是一个模拟的错误日志")
    error_logger.critical("这是一个严重错误日志")
    
    print("✅ 日志测试完成！")
    print("\n📁 现在您可以查看生成的日志文件：")
    
    # 检查生成的日志文件，使用自动检测的项目根目录
    log_dir = os.path.join(project_root, "logs")
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        for log_file in sorted(log_files):
            file_path = os.path.join(log_dir, log_file)
            file_size = os.path.getsize(file_path)
            print(f"  📄 {log_file} ({file_size} bytes)")
    
    print("\n🔍 尝试使用以下命令查看日志：")
    print("  python app/log/log_viewer.py                    # 查看最新的日志")
    print("  python app/log/log_viewer.py --errors           # 查看错误日志")
    print("  python app/log/log_viewer.py --search '聊天'     # 搜索聊天相关日志")
    print("  python app/log/log_viewer.py --level WARNING    # 查看警告及以上级别的日志")

if __name__ == "__main__":
    test_logging() 