FROM python:3.10-slim

# 设置时区为中国上海
ENV TZ=Asia/Shanghai

WORKDIR /app

# 复制所需文件到容器中
COPY ./requirements.txt /app
COPY ./VERSION /app

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
COPY ./app /app/app
ENV API_KEYS='["your_api_key_1"]'
ENV ALLOWED_TOKENS='["your_token_1"]'
ENV BASE_URL=https://generativelanguage.googleapis.com/v1beta
ENV TOOLS_CODE_EXECUTION_ENABLED=false
ENV IMAGE_MODELS='["gemini-2.0-flash-exp"]'
ENV SEARCH_MODELS='["gemini-2.0-flash-exp","gemini-2.0-pro-exp"]'

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
