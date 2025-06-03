#!/usr/bin/env python3
"""
简单的日志查看器
用于方便地查看和管理Gemini Balance项目的日志文件
"""

import os
import sys
from datetime import datetime, timedelta
import argparse
import glob
import re


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


def show_log_files():
    """显示所有可用的日志文件"""
    # 自动检测项目根目录
    project_root = get_project_root()
    log_dir = os.path.join(project_root, "logs")
    
    if not os.path.exists(log_dir):
        print("❌ 日志目录不存在。请先运行应用程序以生成日志文件。")
        return []
    
    log_files = glob.glob(os.path.join(log_dir, "*.log*"))
    if not log_files:
        print("❌ 没有找到日志文件。")
        return []
    
    print("\n📋 可用的日志文件：")
    for i, file_path in enumerate(sorted(log_files), 1):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        # 格式化文件大小
        if file_size < 1024:
            size_str = f"{file_size}B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size/1024:.1f}KB"
        else:
            size_str = f"{file_size/(1024*1024):.1f}MB"
            
        print(f"  {i}. {file_name} ({size_str}, {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return sorted(log_files)


def tail_log(file_path, lines=50):
    """显示日志文件的最后几行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content_lines = f.readlines()
            
        if len(content_lines) <= lines:
            print(f"\n📄 显示 {os.path.basename(file_path)} 的全部内容：")
            for line in content_lines:
                print(line.rstrip())
        else:
            print(f"\n📄 显示 {os.path.basename(file_path)} 的最后 {lines} 行：")
            for line in content_lines[-lines:]:
                print(line.rstrip())
                
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")


def search_logs(file_path, keyword, context_lines=2):
    """在日志文件中搜索关键词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        matches = []
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                matches.append((i + 1, start, end, lines[start:end]))
        
        if matches:
            print(f"\n🔍 在 {os.path.basename(file_path)} 中找到 {len(matches)} 个匹配：")
            for line_num, start, end, context in matches:
                print(f"\n--- 第 {line_num} 行附近 ---")
                for j, context_line in enumerate(context):
                    actual_line_num = start + j + 1
                    prefix = ">>> " if actual_line_num == line_num else "    "
                    print(f"{prefix}{actual_line_num:4d}: {context_line.rstrip()}")
        else:
            print(f"❌ 在 {os.path.basename(file_path)} 中没有找到 '{keyword}'")
            
    except Exception as e:
        print(f"❌ 搜索失败: {e}")


def filter_logs_by_level(file_path, level):
    """按日志级别过滤日志"""
    level = level.upper()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 使用正则表达式匹配日志级别，忽略ANSI颜色代码
        pattern = re.compile(rf'\|\s*(?:\x1b\[[0-9;]*m)?{re.escape(level)}(?:\x1b\[[0-9;]*m)?\s*\|')
        filtered_lines = [line for line in lines if pattern.search(line)]
        
        if filtered_lines:
            print(f"\n📊 {os.path.basename(file_path)} 中的 {level} 级别日志 ({len(filtered_lines)} 条)：")
            for line in filtered_lines:
                print(line.rstrip())
        else:
            print(f"❌ 没有找到 {level} 级别的日志")
            
    except Exception as e:
        print(f"❌ 过滤失败: {e}")


def show_recent_errors(hours=24):
    """显示最近的错误日志"""
    # 使用自动检测的项目根目录
    project_root = get_project_root()
    error_log = os.path.join(project_root, "logs", "error.log")
    
    if not os.path.exists(error_log):
        print("❌ 错误日志文件不存在")
        return
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    print(f"\n🚨 最近 {hours} 小时的错误日志：")
    
    try:
        with open(error_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        recent_errors = []
        for line in lines:
            try:
                # 尝试解析时间戳 (格式: YYYY-MM-DD HH:MM:SS,mmm)
                if len(line) > 23:
                    time_str = line[:23].replace(',', '.')
                    log_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                    if log_time > cutoff_time:
                        recent_errors.append(line)
            except ValueError:
                # 如果无法解析时间戳，跳过这行
                continue
        
        if recent_errors:
            for line in recent_errors:
                print(line.rstrip())
        else:
            print("✅ 最近没有错误日志")
            
    except Exception as e:
        print(f"❌ 读取错误日志失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Balance 日志查看器")
    parser.add_argument("--tail", "-t", type=int, default=50, help="显示最后几行 (默认: 50)")
    parser.add_argument("--search", "-s", type=str, help="搜索关键词")
    parser.add_argument("--level", "-l", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="按级别过滤")
    parser.add_argument("--errors", "-e", action="store_true", help="显示最近的错误日志")
    parser.add_argument("--hours", type=int, default=24, help="错误日志的时间范围(小时) (默认: 24)")
    
    args = parser.parse_args()
    
    print("🔍 Gemini Balance 日志查看器")
    print("=" * 50)
    
    # 如果指定了显示错误，直接显示错误
    if args.errors:
        show_recent_errors(args.hours)
        return
    
    # 显示可用的日志文件
    log_files = show_log_files()
    if not log_files:
        return
    
    # 如果只有一个日志文件，直接使用它
    if len(log_files) == 1:
        selected_file = log_files[0]
        print(f"\n✅ 自动选择唯一的日志文件: {os.path.basename(selected_file)}")
    else:
        # 让用户选择文件
        try:
            choice = input(f"\n请选择要查看的日志文件 (1-{len(log_files)}) 或输入文件名: ").strip()
            
            # 尝试解析为数字
            try:
                file_index = int(choice) - 1
                if 0 <= file_index < len(log_files):
                    selected_file = log_files[file_index]
                else:
                    print("❌ 无效的选择")
                    return
            except ValueError:
                # 如果不是数字，尝试匹配文件名
                matches = [f for f in log_files if choice in os.path.basename(f)]
                if len(matches) == 1:
                    selected_file = matches[0]
                elif len(matches) > 1:
                    print(f"❌ 找到多个匹配的文件: {[os.path.basename(f) for f in matches]}")
                    return
                else:
                    print("❌ 没有找到匹配的文件")
                    return
                    
        except KeyboardInterrupt:
            print("\n👋 退出")
            return
    
    # 根据参数执行相应操作
    if args.search:
        search_logs(selected_file, args.search)
    elif args.level:
        filter_logs_by_level(selected_file, args.level)
    else:
        tail_log(selected_file, args.tail)


if __name__ == "__main__":
    main() 