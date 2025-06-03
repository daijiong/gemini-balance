# 📝 Gemini Balance 日志功能说明

## 🎯 功能概述

现在您的Gemini Balance项目已经支持完整的文件日志功能！系统会自动将所有运行信息保存到文件中，方便您随时查看和排查问题。

## 📁 日志文件结构

当您启动应用后，系统会自动在项目根目录创建一个`logs`文件夹，包含以下日志文件：

```
logs/
├── gemini-balance.log      # 主日志文件（包含所有日志）
├── error.log              # 错误日志文件（仅包含ERROR和CRITICAL级别）
├── openai.log             # OpenAI相关日志
├── gemini.log             # Gemini API相关日志
├── chat.log               # 聊天功能相关日志
├── request.log            # 请求处理相关日志
├── security.log           # 安全相关日志
└── *.log.1, *.log.2...    # 轮转的备份文件
```

## 🔄 自动日志轮转

为了防止日志文件过大，系统自动进行日志轮转：

- **主日志文件**: 每个文件最大10MB，保留5个备份
- **错误日志文件**: 每个文件最大5MB，保留3个备份  
- **模块日志文件**: 每个文件最大5MB，保留3个备份

当文件达到大小限制时，系统会自动：
1. 将当前文件重命名为 `.log.1`
2. 创建新的日志文件
3. 删除最旧的备份文件

## 🔍 日志查看工具

我为您创建了一个简单易用的日志查看工具`log_viewer.py`，让您可以轻松查看日志：

### 基本使用

```bash
# 查看最新的50行日志
python log_viewer.py

# 查看最新的100行日志
python log_viewer.py --tail 100

# 搜索包含特定关键词的日志
python log_viewer.py --search "错误"

# 只显示错误级别的日志
python log_viewer.py --level ERROR

# 显示最近24小时的错误日志
python log_viewer.py --errors

# 显示最近3小时的错误日志
python log_viewer.py --errors --hours 3
```

### 实用示例

```bash
# 查看API调用失败的日志
python log_viewer.py --search "failed"

# 查看认证相关问题
python log_viewer.py --search "auth"

# 查看所有警告和错误
python log_viewer.py --level WARNING

# 排查最近的问题
python log_viewer.py --errors --hours 1
```

## 📊 日志级别说明

| 级别 | 说明 | 使用场景 |
|------|------|----------|
| **DEBUG** | 详细的调试信息 | 开发和深度排查时使用 |
| **INFO** | 一般信息记录 | 正常的操作流程记录 |
| **WARNING** | 警告信息 | 可能的问题，但不影响正常运行 |
| **ERROR** | 错误信息 | 功能异常，但程序可以继续运行 |
| **CRITICAL** | 严重错误 | 可能导致程序崩溃的严重问题 |

## ⚙️ 日志配置

您可以通过环境变量调整日志行为：

### 基本配置

```bash
# 设置日志级别
LOG_LEVEL=DEBUG    # 显示所有日志（开发环境推荐）
LOG_LEVEL=INFO     # 显示一般信息及以上（生产环境推荐）
LOG_LEVEL=WARNING  # 只显示警告及以上
LOG_LEVEL=ERROR    # 只显示错误及以上
```

### 自动清理配置

```bash
# 自动删除旧的错误日志（推荐开启）
AUTO_DELETE_ERROR_LOGS_ENABLED=true
AUTO_DELETE_ERROR_LOGS_DAYS=7    # 保留7天的错误日志

# 自动删除旧的请求日志（可选）
AUTO_DELETE_REQUEST_LOGS_ENABLED=false
AUTO_DELETE_REQUEST_LOGS_DAYS=30   # 保留30天的请求日志
```

## 🛠️ 常见使用场景

### 1. 日常监控
```bash
# 每天检查是否有错误
python log_viewer.py --errors --hours 24
```

### 2. 问题排查
```bash
# API调用问题
python log_viewer.py --search "api" --level ERROR

# 认证问题
python log_viewer.py --search "token"

# 性能问题
python log_viewer.py --search "timeout"
```

### 3. 开发调试
```bash
# 查看详细的调试信息（需要设置LOG_LEVEL=DEBUG）
python log_viewer.py --level DEBUG --tail 100
```

## 📋 日志内容说明

每条日志包含以下信息：
```
2024-01-15 10:30:25,123 | INFO     | [chat.py:45]              | 用户开始聊天会话
```

- **时间戳**: `2024-01-15 10:30:25,123` - 精确到毫秒
- **日志级别**: `INFO` - 日志的重要性级别
- **文件位置**: `[chat.py:45]` - 产生日志的文件和行号
- **日志消息**: `用户开始聊天会话` - 具体的日志内容

## 💡 使用技巧

### 1. 快速定位问题
- 先查看错误日志：`python log_viewer.py --errors`
- 根据时间点查看主日志：`python log_viewer.py --tail 200`

### 2. 监控特定功能
- 查看聊天功能：直接查看 `logs/chat.log` 文件
- 查看API调用：直接查看 `logs/gemini.log` 或 `logs/openai.log`

### 3. 性能分析
- 搜索慢查询：`python log_viewer.py --search "slow"`
- 查看超时问题：`python log_viewer.py --search "timeout"`

## 🔧 直接查看日志文件

您也可以使用系统工具直接查看日志：

```bash
# Linux/Mac 下查看实时日志
tail -f logs/gemini-balance.log

# 搜索特定内容
grep "错误" logs/gemini-balance.log

# 查看最新的100行
tail -n 100 logs/gemini-balance.log
```

## ⚠️ 注意事项

1. **磁盘空间**: 虽然有自动轮转，但请定期检查`logs`文件夹的大小
2. **敏感信息**: 日志可能包含API密钥等敏感信息，请妥善保管
3. **性能影响**: 大量的DEBUG级别日志可能略微影响性能
4. **备份**: 重要的错误日志建议定期备份到其他位置

## 🎉 开始使用

1. **启动应用**: 正常启动您的Gemini Balance应用
2. **生成日志**: 系统会自动开始记录日志到文件
3. **查看日志**: 使用 `python log_viewer.py` 查看日志
4. **监控错误**: 定期运行 `python log_viewer.py --errors` 检查问题

现在您就可以轻松地监控和排查应用问题了！如果遇到任何问题，首先查看错误日志，这将大大提高问题解决的效率。 