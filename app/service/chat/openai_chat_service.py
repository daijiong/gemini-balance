# app/services/chat_service.py

import asyncio
import datetime
import json
import re
import time
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from app.config.config import settings
from app.core.constants import GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
from app.database.services import (
    add_error_log,
    add_request_log,
)
from app.domain.openai_models import ChatRequest, ImageGenerationRequest
from app.handler.message_converter import OpenAIMessageConverter
from app.handler.response_handler import OpenAIResponseHandler
from app.handler.stream_optimizer import openai_optimizer
from app.log.logger import get_openai_logger
from app.service.client.api_client import GeminiApiClient
from app.service.image.image_create_service import ImageCreateService
from app.service.key.key_manager import KeyManager

logger = get_openai_logger()


def _has_media_parts(contents: List[Dict[str, Any]]) -> bool:
    """判断消息是否包含图片、音频或视频部分 (inline_data)"""
    for content in contents:
        if content and "parts" in content and isinstance(content["parts"], list):
            for part in content["parts"]:
                if isinstance(part, dict) and "inline_data" in part:
                    return True
    return False


def _build_tools(
    request: ChatRequest, messages: List[Dict[str, Any]], request_id: str = None
) -> List[Dict[str, Any]]:
    """
    构建Gemini API请求payload中的工具

    Args:
        request: OpenAI请求对象
        messages: 消息列表
        request_id: 请求ID

    Returns:
        List[Dict[str, Any]]: Gemini API请求payload中的工具
    """
    tool = dict()
    model = request.model

    # 如果启用了代码执行工具，模型不支持搜索、思考、图像生成，并且消息中没有图片、音频或视频部分，则添加代码执行工具
    if (
        settings.TOOLS_CODE_EXECUTION_ENABLED
        and not (
            model.endswith("-search")
            or "-thinking" in model
            or model.endswith("-image")
            or model.endswith("-image-generation")
        )
        and not _has_media_parts(messages)
    ):
        tool["codeExecution"] = {}
        logger.debug(f"【{request_id}】Code execution tool enabled.")
    # 如果消息中包含图片、音频或视频部分，则不添加代码执行工具
    elif _has_media_parts(messages):
        logger.debug(f"【{request_id}】Code execution tool disabled due to media parts presence.")

    # 如果模型支持搜索，则添加搜索工具
    if model.endswith("-search"):
        tool["googleSearch"] = {}
        logger.debug(f"【{request_id}】Google search tool enabled.")

    # 将 request 中的 tools 合并到 tools 中
    if request.tools:
        function_declarations = []
        for item in request.tools:
            if not item or not isinstance(item, dict):
                continue

            if item.get("type", "") == "function" and item.get("function"):
                function = deepcopy(item.get("function"))
                parameters = function.get("parameters", {})
                if parameters.get("type") == "object" and not parameters.get(
                    "properties", {}
                ):
                    function.pop("parameters", None)

                function_declarations.append(function)

        if function_declarations:
            # 按照 function 的 name 去重
            names, functions = set(), []
            for fc in function_declarations:
                if fc.get("name") not in names:
                    if fc.get("name")=="googleSearch":
                        # cherry开启内置搜索时，添加googleSearch工具
                        tool["googleSearch"] = {}
                    else:
                        # 其他函数，添加到functionDeclarations中
                        names.add(fc.get("name"))
                        functions.append(fc)

            tool["functionDeclarations"] = functions

    # 解决 "Tool use with function calling is unsupported" 问题
    if tool.get("functionDeclarations"):
        tool.pop("googleSearch", None)
        tool.pop("codeExecution", None)

    return [tool] if tool else []


def _get_safety_settings(model: str) -> List[Dict[str, str]]:
    """获取安全设置"""
    # if (
    #     "2.0" in model
    #     and "gemini-2.0-flash-thinking-exp" not in model
    #     and "gemini-2.0-pro-exp" not in model
    # ):
    if model == "gemini-2.0-flash-exp":
        return GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
    return settings.SAFETY_SETTINGS


def _build_payload(
    request: ChatRequest,
    messages: List[Dict[str, Any]],
    instruction: Optional[Dict[str, Any]] = None,
    request_id: str = None,
) -> Dict[str, Any]:
    """
    构建Gemini API请求payload，包括消息、配置、工具、安全设置等

    Args:
        request: OpenAI请求对象
        messages: 消息列表
        instruction: 系统指令
        request_id: 请求ID

    Returns:
        Dict[str, Any]: Gemini API请求payload
    """
    payload = {
        "contents": messages,
        "generationConfig": {
            "temperature": request.temperature,
            "stopSequences": request.stop,
            "topP": request.top_p,
            "topK": request.top_k,
        },
        "tools": _build_tools(request, messages, request_id),
        "safetySettings": _get_safety_settings(request.model),
    }
    if request.max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = request.max_tokens
    if request.model.endswith("-image") or request.model.endswith("-image-generation"):
        payload["generationConfig"]["responseModalities"] = ["Text", "Image"]
    if request.model.endswith("-non-thinking"):
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    if request.model in settings.THINKING_BUDGET_MAP:
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": settings.THINKING_BUDGET_MAP.get(request.model, 1000)
        }

    if (
        instruction
        and isinstance(instruction, dict)
        and instruction.get("role") == "system"
        and instruction.get("parts")
        and not request.model.endswith("-image")
        and not request.model.endswith("-image-generation")
    ):
        payload["systemInstruction"] = instruction

    return payload


class OpenAIChatService:
    """聊天服务"""

    def __init__(self, base_url: str, key_manager: KeyManager = None):
        """
        初始化聊天服务

        Args:
            base_url: 基础URL
            key_manager: 密钥管理器

        """
        self.message_converter = OpenAIMessageConverter()
        self.response_handler = OpenAIResponseHandler(config=None)
        self.api_client = GeminiApiClient(base_url, settings.TIME_OUT)
        self.key_manager = key_manager
        self.image_create_service = ImageCreateService()

    def _extract_text_from_openai_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        从OpenAI响应块中提取文本内容

        Args:
            chunk: OpenAI响应块

        Returns:
            str: 文本内容
        """
        if not chunk.get("choices"):
            return ""

        choice = chunk["choices"][0]
        if "delta" in choice and "content" in choice["delta"]:
            return choice["delta"]["content"]
        return ""

    def _create_char_openai_chunk(
        self, original_chunk: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """
        创建包含指定文本的OpenAI响应块

        Args:
            original_chunk: 原始OpenAI响应块
            text: 指定文本

        Returns:
            Dict[str, Any]: 包含指定文本的OpenAI响应块
        """
        chunk_copy = json.loads(json.dumps(original_chunk))
        if chunk_copy.get("choices") and "delta" in chunk_copy["choices"][0]:
            chunk_copy["choices"][0]["delta"]["content"] = text
        return chunk_copy

    async def create_chat_completion(
        self,
        request: ChatRequest,
        api_key: str,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """
        创建聊天完成，包括流式和非流式

        Args:
            request: OpenAI请求对象
            api_key: API密钥

        Returns:
            Union[Dict[str, Any], AsyncGenerator[str, None]]: 返回结果
        """
        request_id = request.id
        # 消息转换
        messages, instruction = self.message_converter.convert(request.messages)
        # 构建Gemini API请求payload
        payload = _build_payload(request, messages, instruction, request_id)
        # 处理流式请求
        if request.stream:
            return self._handle_stream_completion(request.model, payload, api_key, request_id)
        # 处理非流式请求
        return await self._handle_normal_completion(request.model, payload, api_key, request_id)

    async def _handle_normal_completion(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str
    ) -> Dict[str, Any]:
        """
        处理非流式完成

        Args:
            model: 模型名称
            payload: Gemini API请求payload
            api_key: API密钥
            request_id: 请求ID

        Returns:
            Dict[str, Any]: 返回结果
        """
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = True
        status_code = 200

        try:
            # 打印非流式处理开始信息
            logger.info(f"【{request_id}】🔥 开始非流式处理 - 模型: {model}")
            logger.info(f"【{request_id}】📤 Gemini API请求payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            # 调用Gemini API生成内容
            response = await self.api_client.generate_content(payload, model, api_key)
            if not response or not response.get("candidates"):
                logger.error(f"【{request_id}】Empty response from model: {model}")
                raise Exception("Empty response from model")
                
            # logger.debug(f"【{request_id}】Successfully got response from model: {model}")
            # 打印非流式处理结束信息
            logger.info(f"【{request_id}】🏁 非流式处理完成 - 模型: {model}")
            logger.info(f"【{request_id}】📤 Gemini API返回结果: {json.dumps(response, indent=2, ensure_ascii=False)}")
            return self.response_handler.handle_response(
                response, model, stream=False, finish_reason='stop', usage_metadata=response.get("usageMetadata", {})
            )
        except Exception as e:
            is_success = False
            error_log_msg = str(e)
            logger.error(f"【{request_id}】Normal API call failed with error: {error_log_msg}")
            match = re.search(r"status code (\d+)", error_log_msg)
            if match:
                status_code = int(match.group(1))
            else:
                status_code = 500

            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="openai-chat-non-stream",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg=payload,
            )
            raise e
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.debug(f"【{request_id}】Normal completion finished in {latency_ms}ms")
            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )

    async def _fake_stream_logic_impl(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str
    ) -> AsyncGenerator[str, None]:
        """处理伪流式 (fake stream) 的核心逻辑"""
        logger.info(
            f"【{request_id}】Fake streaming enabled for model: {model}. Calling non-streaming endpoint."
        )
        keep_sending_empty_data = True

        async def send_empty_data_locally() -> AsyncGenerator[str, None]:
            """定期发送空数据以保持连接"""
            while keep_sending_empty_data:
                await asyncio.sleep(settings.FAKE_STREAM_EMPTY_DATA_INTERVAL_SECONDS)
                if keep_sending_empty_data:
                    empty_chunk = self.response_handler.handle_response({}, model, stream=True, finish_reason='stop', usage_metadata=None)
                    yield f"data: {json.dumps(empty_chunk)}\n\n"
                    logger.debug(f"【{request_id}】Sent empty data chunk for fake stream heartbeat.")

        empty_data_generator = send_empty_data_locally()
        api_response_task = asyncio.create_task(
            self.api_client.generate_content(payload, model, api_key)
        )

        try:
            while not api_response_task.done():
                try:
                    next_empty_chunk = await asyncio.wait_for(
                        empty_data_generator.__anext__(), timeout=0.1
                    )
                    yield next_empty_chunk
                except asyncio.TimeoutError:
                    pass
                except (
                    StopAsyncIteration
                ):
                    break

            response = await api_response_task
        finally:
            keep_sending_empty_data = False

        if response and response.get("candidates"):
            response = self.response_handler.handle_response(response, model, stream=True, finish_reason='stop', usage_metadata=response.get("usageMetadata", {}))
            yield f"data: {json.dumps(response)}\n\n"
            logger.info(f"【{request_id}】Sent full response content for fake stream: {model}")
        else:
            error_message = "Failed to get response from model"
            if (
                response and isinstance(response, dict) and response.get("error")
            ):
                error_message = f"{error_message}: {response['error'].get('message', 'Unknown error')}"
            logger.error(
                f"【{request_id}】{error_message}"
            )
            # 发送 [DONE] 消息结束流
            yield "data: [DONE]\n\n"

    async def _real_stream_logic_impl(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str
    ) -> AsyncGenerator[str, None]:
        """处理真实流式 (real stream) 的核心逻辑"""
        tool_call_flag = False
        usage_metadata = None
        chunk_count = 0
        complete_content = ""
        
        # 打印流式处理开始信息
        logger.info(f"【{request_id}】🔥 开始流式处理 - 模型: {model}")
        logger.info(f"【{request_id}】📤 Gemini API请求payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        async for line in self.api_client.stream_generate_content(
            payload, model, api_key
        ):
            if line.startswith("data:"):
                chunk_str = line[6:]
                if not chunk_str or chunk_str.isspace():
                    logger.debug(
                        f"【{request_id}】Received empty data line for model {model}, skipping."
                    )
                    continue
                try:
                    chunk = json.loads(chunk_str)
                    usage_metadata = chunk.get("usageMetadata", {})
                    
                except json.JSONDecodeError:
                    logger.error(
                        f"【{request_id}】Failed to decode JSON from stream for model {model}: {chunk_str}"
                    )
                    continue
                    
                openai_chunk = self.response_handler.handle_response(
                    chunk, model, stream=True, finish_reason=None, usage_metadata=usage_metadata
                )
                
                if openai_chunk:
                    chunk_count += 1
                    
                    text = self._extract_text_from_openai_chunk(openai_chunk)
                    if text:
                        complete_content += text
                    
                    if text and settings.STREAM_OPTIMIZER_ENABLED:
                        async for (
                            optimized_chunk_data
                        ) in openai_optimizer.optimize_stream_output(
                            text,
                            lambda t: self._create_char_openai_chunk(openai_chunk, t),
                            lambda c: f"data: {json.dumps(c)}\n\n",
                        ):
                            yield optimized_chunk_data
                    else:
                        if openai_chunk.get("choices") and openai_chunk["choices"][0].get("delta", {}).get("tool_calls"):
                            tool_call_flag = True

                        final_chunk_data = f"data: {json.dumps(openai_chunk)}\n\n"
                        yield final_chunk_data

        # 处理最终完成块
        final_reason = 'tool_calls' if tool_call_flag else 'stop'
        final_chunk = self.response_handler.handle_response({}, model, stream=True, finish_reason=final_reason, usage_metadata=usage_metadata)
        
        logger.info(f"【{request_id}】🏁 流式处理完成统计:")
        logger.info(f"【{request_id}】  - 返回结果长度: {len(complete_content)} 字符")
        logger.info(f"【{request_id}】  - 返回结果内容: {complete_content}")
        
        yield f"data: {json.dumps(final_chunk)}\n\n"

    async def _handle_stream_completion(
        self, model: str, payload: Dict[str, Any], api_key: str, request_id: str
    ) -> AsyncGenerator[str, None]:
        """
        处理流式聊天完成，添加重试逻辑和假流式支持

        Args:
            model: 模型名称
            payload: Gemini API请求payload
            api_key: API密钥
            request_id: 请求ID

        Returns:
            AsyncGenerator[str, None]: 流式响应生成器
        """
        retries = 0
        max_retries = settings.MAX_RETRIES
        is_success = False
        status_code = None
        final_api_key = api_key

        while retries < max_retries:
            start_time = time.perf_counter()
            request_datetime = datetime.datetime.now()
            current_attempt_key = final_api_key

            try:
                stream_generator = None
                if settings.FAKE_STREAM_ENABLED:
                    logger.info(
                        f"【{request_id}】Using fake stream logic for model: {model}, Attempt: {retries + 1}"
                    )
                    stream_generator = self._fake_stream_logic_impl(
                        model, payload, current_attempt_key, request_id
                    )
                else:
                    logger.info(
                        f"【{request_id}】Using real stream logic for model: {model}, Attempt: {retries + 1}"
                    )
                    stream_generator = self._real_stream_logic_impl(
                        model, payload, current_attempt_key, request_id
                    )

                async for chunk_data in stream_generator:
                    yield chunk_data

                yield "data: [DONE]\n\n"
                logger.info(
                    f"【{request_id}】Streaming completed successfully for model: {model}, FakeStream: {settings.FAKE_STREAM_ENABLED}, Attempt: {retries + 1}"
                )
                is_success = True
                status_code = 200
                break

            except Exception as e:
                retries += 1
                is_success = False
                error_log_msg = str(e)
                logger.warning(
                    f"【{request_id}】Streaming API call failed with error: {error_log_msg}. Attempt {retries} of {max_retries} with key {current_attempt_key}"
                )

                match = re.search(r"status code (\d+)", error_log_msg)
                if match:
                    status_code = int(match.group(1))
                else:
                    if isinstance(e, asyncio.TimeoutError):
                        status_code = 408
                    else:
                        status_code = 500

                await add_error_log(
                    gemini_key=current_attempt_key,
                    model_name=model,
                    error_type="openai-chat-stream",
                    error_log=error_log_msg,
                    error_code=status_code,
                    request_msg=payload,
                )

                if self.key_manager:
                    new_api_key = await self.key_manager.handle_api_failure(
                        current_attempt_key, retries
                    )
                    if new_api_key and new_api_key != current_attempt_key:
                        final_api_key = new_api_key
                        logger.info(
                            f"【{request_id}】Switched to new API key for next attempt: {final_api_key}"
                        )
                    elif not new_api_key:
                        logger.error(
                            f"【{request_id}】No valid API key available after {retries} retries, ceasing attempts for this request."
                        )
                        break
                else:
                    logger.error(
                        "【{request_id}】KeyManager not available, cannot switch API key. Ceasing attempts for this request."
                    )
                    break

                if retries >= max_retries:
                    logger.error(
                        f"【{request_id}】Max retries ({max_retries}) reached for streaming model {model}."
                    )
            finally:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                logger.debug(f"【{request_id}】Latency: {latency_ms}ms")
                await add_request_log(
                    model_name=model,
                    api_key=current_attempt_key,
                    is_success=is_success,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_time=request_datetime,
                )

        if not is_success:
            logger.error(
                f"【{request_id}】Streaming failed permanently for model {model} after {retries} attempts."
            )
            yield f"data: {json.dumps({'error': f'Streaming failed after {retries} retries.'})}\n\n"
            yield "data: [DONE]\n\n"

    async def create_image_chat_completion(
        self, request: ChatRequest, api_key: str
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """
        创建图像聊天完成，包括流式和非流式

        Args:
            request: OpenAI请求对象
            api_key: API密钥

        Returns:
            Union[Dict[str, Any], AsyncGenerator[str, None]]: 返回结果
        """
        request_id = request.id
        image_generate_request = ImageGenerationRequest()
        image_generate_request.prompt = request.messages[-1]["content"]
        image_res = self.image_create_service.generate_images_chat(
            image_generate_request
        )

        if request.stream:
            return self._handle_stream_image_completion(
                request.model, image_res, api_key, request_id
            )
        else:
            return await self._handle_normal_image_completion(
                request.model, image_res, api_key, request_id
            )

    async def _handle_stream_image_completion(
        self, model: str, image_data: str, api_key: str, request_id: str
    ) -> AsyncGenerator[str, None]:
        """
        处理流式图像聊天完成

        Args:
            model: 模型名称
            image_data: 图像数据
            api_key: API密钥
            request_id: 请求ID

        Returns:
            AsyncGenerator[str, None]: 流式响应生成器
        """
        logger.info(f"【{request_id}】Starting stream image completion for model: {model}")
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None

        try:
            if image_data:
                openai_chunk = self.response_handler.handle_image_chat_response(
                    image_data, model, stream=True, finish_reason=None
                )
                if openai_chunk:
                    # 提取文本内容
                    text = self._extract_text_from_openai_chunk(openai_chunk)
                    if text:
                        # 使用流式输出优化器处理文本输出
                        async for (
                            optimized_chunk
                        ) in openai_optimizer.optimize_stream_output(
                            text,
                            lambda t: self._create_char_openai_chunk(openai_chunk, t),
                            lambda c: f"data: {json.dumps(c)}\n\n",
                        ):
                            yield optimized_chunk
                    else:
                        # 如果没有文本内容（如图片URL等），整块输出
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
            yield f"data: {json.dumps(self.response_handler.handle_response({}, model, stream=True, finish_reason='stop'))}\n\n"
            logger.info(
                f"【{request_id}】Stream image completion finished successfully for model: {model}"
            )
            is_success = True
            status_code = 200
            yield "data: [DONE]\n\n"
        except Exception as e:
            is_success = False
            error_log_msg = f"Stream image completion failed for model {model}: {e}"
            logger.error(f"【{request_id}】{error_log_msg}")
            status_code = 500
            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="openai-image-stream",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg={"image_data_truncated": image_data[:1000]},
            )
            yield f"data: {json.dumps({'error': error_log_msg})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.info(
                f"【{request_id}】Stream image completion for model {model} took {latency_ms} ms. Success: {is_success}"
            )
            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )

    async def _handle_normal_image_completion(
        self, model: str, image_data: str, api_key: str, request_id: str
    ) -> Dict[str, Any]:
        logger.info(f"【{request_id}】Starting normal image completion for model: {model}")
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None
        result = None

        try:
            result = self.response_handler.handle_image_chat_response(
                image_data, model, stream=False, finish_reason="stop"
            )
            logger.info(
                f"【{request_id}】Normal image completion finished successfully for model: {model}"
            )
            is_success = True
            status_code = 200
            return result
        except Exception as e:
            is_success = False
            error_log_msg = f"Normal image completion failed for model {model}: {e}"
            logger.error(f"【{request_id}】{error_log_msg}")
            status_code = 500
            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="openai-image-non-stream",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg={"image_data_truncated": image_data[:1000]},
            )
            raise e
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.info(
                f"【{request_id}】Normal image completion for model {model} took {latency_ms} ms. Success: {is_success}"
            )
            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )
