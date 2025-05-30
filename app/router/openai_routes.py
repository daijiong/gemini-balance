import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config.config import settings
from app.core.security import SecurityService
from app.domain.openai_models import (
    ChatRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)
from app.handler.retry_handler import RetryHandler
from app.handler.error_handler import handle_route_errors
from app.log.logger import get_openai_logger
from app.service.chat.openai_chat_service import OpenAIChatService
from app.service.embedding.embedding_service import EmbeddingService
from app.service.image.image_create_service import ImageCreateService
from app.service.key.key_manager import KeyManager, get_key_manager_instance
from app.service.model.model_service import ModelService

router = APIRouter()
logger = get_openai_logger()

security_service = SecurityService()
model_service = ModelService()
embedding_service = EmbeddingService()
image_create_service = ImageCreateService()


async def get_key_manager():
    return await get_key_manager_instance()


async def get_next_working_key_wrapper(
    key_manager: KeyManager = Depends(get_key_manager),
):
    return await key_manager.get_next_working_key()


async def get_openai_chat_service(key_manager: KeyManager = Depends(get_key_manager)):
    """获取OpenAI聊天服务实例"""
    return OpenAIChatService(settings.BASE_URL, key_manager)


@router.get("/v1/models")
@router.get("/hf/v1/models")
async def list_models(
    _=Depends(security_service.verify_authorization),
    key_manager: KeyManager = Depends(get_key_manager),
):
    """获取可用的 OpenAI 模型列表 (兼容 Gemini 和 OpenAI)。"""
    operation_name = "list_models"
    async with handle_route_errors(logger, operation_name):
        logger.info("Handling models list request")
        api_key = await key_manager.get_first_valid_key()
        logger.info(f"Using API key: {api_key}")
        return await model_service.get_gemini_openai_models(api_key)


@router.post("/v1/chat/completions")
@router.post("/hf/v1/chat/completions")
@RetryHandler(key_arg="api_key")
async def chat_completion(
    request: ChatRequest,
    _=Depends(security_service.verify_authorization),
    api_key: str = Depends(get_next_working_key_wrapper),
    key_manager: KeyManager = Depends(get_key_manager),
    chat_service: OpenAIChatService = Depends(get_openai_chat_service),
):
    """处理 OpenAI 聊天补全请求，支持流式响应和特定模型切换。"""
    # 使用更精确的时间测量
    start_time = time.perf_counter()
    operation_name = "chat_completion"
    is_image_chat = request.model == f"{settings.CREATE_IMAGE_MODEL}-chat"
    current_api_key = api_key
    if is_image_chat:
        current_api_key = await key_manager.get_paid_key()

    async with handle_route_errors(logger, operation_name):
        try:
            logger.info(f"【{request.id}】Request: \n{request.model_dump_json(indent=2)}")

            # 特殊处理：当max_tokens 不为空且 <= 100时，设置max_tokens为100
            if (request.max_tokens is not None and request.max_tokens <= 100):
                logger.info(f"【{request.id}】请求参数 max_tokens <= 100, 设置 max_tokens 为 100")
                request.max_tokens = 100

            if not await model_service.check_model_support(request.model):
                logger.error(f"【{request.id}】model: {request.model} is not supported")
                raise HTTPException(
                    status_code=400, detail=f"Model {request.model} is not supported"
                )

            if is_image_chat:
                response = await chat_service.create_image_chat_completion(request, current_api_key)
                if request.stream:
                    # 对于流式响应，我们记录开始时间，但实际完成时间将在stream处理中记录
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"【{request.id}】Stream response initiated in {elapsed_time:.3f} seconds")
                    return StreamingResponse(response, media_type="text/event-stream")
                else:
                    # 记录非流式请求的完成时间
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"【{request.id}】Image chat completion completed in {elapsed_time:.3f} seconds")
                    return response
            else:
                response = await chat_service.create_chat_completion(request, current_api_key)
                if request.stream:
                    # 对于流式响应，我们记录开始时间，但实际完成时间将在stream处理中记录
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"【{request.id}】Stream response initiated in {elapsed_time:.3f} seconds")
                    return StreamingResponse(response, media_type="text/event-stream")
                else:
                    # 记录非流式请求的完成时间
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"【{request.id}】Chat completion completed in {elapsed_time:.3f} seconds")
                    return response
        except Exception as e:
            # 记录异常情况下的耗时
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"【{request.id}】Chat completion failed after {elapsed_time:.3f} seconds: {str(e)}")
            raise


@router.post("/v1/images/generations")
@router.post("/hf/v1/images/generations")
async def generate_image(
    request: ImageGenerationRequest,
    _=Depends(security_service.verify_authorization),
):
    """处理 OpenAI 图像生成请求。"""
    operation_name = "generate_image"
    async with handle_route_errors(logger, operation_name):
        logger.info(f"Handling image generation request for prompt: {request.prompt}")
        response = image_create_service.generate_images(request)
        return response


@router.post("/v1/embeddings")
@router.post("/hf/v1/embeddings")
async def embedding(
    request: EmbeddingRequest,
    _=Depends(security_service.verify_authorization),
    key_manager: KeyManager = Depends(get_key_manager),
):
    """处理 OpenAI 文本嵌入请求。"""
    operation_name = "embedding"
    async with handle_route_errors(logger, operation_name):
        logger.info(f"Handling embedding request for model: {request.model}")
        api_key = await key_manager.get_next_working_key()
        logger.info(f"Using API key: {api_key}")
        response = await embedding_service.create_embedding(
            input_text=request.input, model=request.model, api_key=api_key
        )
        return response


@router.get("/v1/keys/list")
@router.get("/hf/v1/keys/list")
async def get_keys_list(
    _=Depends(security_service.verify_auth_token),
    key_manager: KeyManager = Depends(get_key_manager),
):
    """获取有效和无效的API key列表 (需要管理 Token 认证)。"""
    operation_name = "get_keys_list"
    async with handle_route_errors(logger, operation_name):
        logger.info("Handling keys list request")
        keys_status = await key_manager.get_keys_by_status()
        return {
            "status": "success",
            "data": {
                "valid_keys": keys_status["valid_keys"],
                "invalid_keys": keys_status["invalid_keys"],
            },
            "total": len(keys_status["valid_keys"]) + len(keys_status["invalid_keys"]),
        }
