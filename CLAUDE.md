# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

Gemini Balance 是一个基于 FastAPI 的 Python 应用程序，为 Google Gemini API 提供代理和负载均衡功能。它支持多个 API 密钥、OpenAI 兼容性、图像生成和全面的日志记录。

## 核心架构

### 应用程序结构
- **FastAPI 应用**: 主应用位于 `app/main.py`，使用 `app/core/application.py` 中的应用工厂模式
- **模块化设计**: 通过专用模块明确分离关注点，实现不同功能
- **数据库抽象**: 支持 MySQL 和 SQLite，使用 SQLAlchemy 进行异步数据库操作
- **配置管理**: 具有数据库持久化和环境变量回退的动态配置
- **中间件栈**: 请求日志记录、安全中间件和 CORS 处理

### 关键组件
- **API 路由**: 为不同 API 格式（OpenAI、Gemini、管理界面）提供独立的路由模块
- **服务层**: 按领域组织的业务逻辑（聊天、嵌入、图像、密钥管理）
- **数据库层**: `app/database/` 中的模型、连接管理和初始化
- **日志系统**: 具有模块专用记录器的全面文件日志记录
- **调度器**: 密钥健康检查和维护的后台任务

### 配置系统
- **动态配置**: 设置存储在数据库中，通过管理界面实时更新
- **环境变量**: 通过 `.env` 文件进行主要配置
- **设置同步**: 环境、内存和数据库设置之间的自动同步
- **密钥管理**: 自动密钥轮换、故障检测和恢复

## 开发命令

### 运行应用程序
```bash
# 本地开发（启用自动重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 生产环境运行
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker 开发
```bash
# 构建 Docker 镜像
docker build -t gemini-balance .

# 使用 Docker 运行
docker run -d -p 8000:8000 --env-file .env gemini-balance

# 使用 docker-compose
docker-compose up -d
```

### 依赖管理
```bash
# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements.txt
```

### 日志分析
```bash
# 查看最近的日志
python app/log/log_viewer.py

# 查看过去 24 小时的错误
python app/log/log_viewer.py --errors

# 搜索特定内容的日志
python app/log/log_viewer.py --search "error_keyword"

# 按级别查看日志
python app/log/log_viewer.py --level ERROR
```

## 主要开发模式

### 配置管理
- 设置通过 `app/config/config.py` 模块管理
- 数据库设置优先于环境变量
- 通过管理界面进行的更改无需重启即可立即生效
- 始终使用 `settings` 全局实例进行配置访问

### 数据库操作
- 使用来自 `app/database/connection.py` 的 `database` 实例进行异步数据库操作
- 模型在 `app/database/models.py` 中定义
- 数据库初始化在 `app/database/initialization.py` 中进行

### 服务层模式
- 业务逻辑按领域在 `app/service/` 中组织
- 每个服务处理特定功能（聊天、密钥管理等）
- 服务采用依赖注入，可以独立测试

### 错误处理
- `app/exception/exceptions.py` 中的自定义异常
- 使用请求 ID 进行跟踪的全面错误日志记录
- 使用适当 HTTP 状态码的优雅错误处理

### API 兼容性
- `app/router/openai_*` 文件中的 OpenAI API 兼容层
- `app/router/gemini_routes.py` 中的 Gemini 原生 API 支持
- `app/handler/` 模块中的请求/响应转换

## 配置说明

### 数据库设置
- MySQL: 需要 `MYSQL_*` 环境变量
- SQLite: 使用 `SQLITE_DATABASE` 文件路径
- 数据库类型由 `DATABASE_TYPE` 环境变量控制

### API 密钥和身份验证
- `API_KEYS`: 用于负载均衡的 Gemini API 密钥列表
- `ALLOWED_TOKENS`: 有效访问令牌列表
- `AUTH_TOKEN`: 超级管理员令牌（默认为第一个 ALLOWED_TOKEN）

### 功能标志
- `IMAGE_MODELS`: 支持图像生成/编辑的模型
- `SEARCH_MODELS`: 支持网络搜索的模型
- `TOOLS_CODE_EXECUTION_ENABLED`: 代码执行能力
- `STREAM_OPTIMIZER_ENABLED`: 流响应优化

## 测试和调试

### 日志文件位置
- 主日志: `logs/gemini-balance.log`
- 错误日志: `logs/error.log`
- 模块特定日志: `logs/openai.log`、`logs/gemini.log` 等

### 健康检查
- 应用程序健康: `GET /health`
- 密钥状态监控: `GET /keys_status`（需要身份验证）
- 配置管理: `GET /config`（管理界面）

### 常见调试工作流程
1. 首先检查错误日志: `python app/log/log_viewer.py --errors`
2. 通过管理界面查看密钥状态和配置
3. 监控请求日志以了解 API 调用模式
4. 使用模块特定日志进行针对性调试

## 重要文件位置

### 核心应用文件
- `app/main.py`: 应用程序入口点
- `app/core/application.py`: 应用工厂和生命周期管理
- `app/config/config.py`: 配置管理和数据库同步

### API 端点
- `app/router/openai_routes.py`: OpenAI 兼容的 API 端点
- `app/router/gemini_routes.py`: 原生 Gemini API 端点
- `app/router/config_routes.py`: 管理配置界面

### 业务逻辑
- `app/service/chat/`: 聊天服务实现
- `app/service/key/key_manager.py`: API 密钥管理和轮换
- `app/service/client/api_client.py`: 外部 API 调用的 HTTP 客户端

### 数据库和模型
- `app/database/models.py`: SQLAlchemy 模型定义
- `app/database/connection.py`: 数据库连接管理
- `app/database/initialization.py`: 数据库架构设置