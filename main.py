import argparse

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from model import ModelService
from openai_api_protocol import (
    UsageInfo, CompletionResponseChoice, CompletionResponse, ChatCompletionResponse, ChatCompletionResponseChoice,
    CompletionRequest, ChatCompletionRequest, ChatMessage
)

app = FastAPI()
model_service = None
model_name = ""
DEFAULT_TEMPERATURE=0.1



@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.get("/ping")
async def root():
    return "pong"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    logger.info(f"new completion request, prompt:{request.prompt}, length: {len(request.prompt)}")

    max_tokens = request.max_tokens
    model = model_name

    temperature = request.temperature
    if temperature is not None and temperature <= 0:
        temperature = DEFAULT_TEMPERATURE
    if temperature is not None and temperature > 1:
        temperature = 0.99
    stop = request.stop

    stream = request.stream
    if stream is None or not stream:
        output = model_service.generate(request.prompt, max_tokens, temperature, stop)
        usage = UsageInfo(
            prompt_tokens=output["prompt_tokens"],
            completion_tokens=output["completion_tokens"],
            total_tokens=output["total_tokens"]
        )
        choices = [
            CompletionResponseChoice(
                index=0,
                text=output["text"],
                logprobs=None,
                finish_reason="stop" if output["stop"] else "length",
            )
        ]
        return CompletionResponse(model=model, choices=choices, usage=usage)
    else:
        generator = model_service.generate_stream(model, request.prompt, max_tokens, temperature, stop, "completions")
        return StreamingResponse(generator, media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    logger.info(f"new chat request, messages:{request.messages}, stream: {request.stream}")

    max_tokens = request.max_tokens
    temperature = request.temperature
    if temperature is not None and temperature <= 0:
        temperature = DEFAULT_TEMPERATURE
    if temperature is not None and temperature > 1:
        temperature = 0.99
    stop = request.stop
    model = model_name
    messages = request.messages

    if isinstance(messages, str):
        prompt = messages
    else:
        prompt = model_service.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    stream = request.stream
    if stream is None or not stream:
        output = model_service.generate(prompt, max_tokens, temperature, stop)
        usage = UsageInfo(
            prompt_tokens=output["prompt_tokens"],
            completion_tokens=output["completion_tokens"],
            total_tokens=output["total_tokens"]
        )
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=output["text"],
                ),
                finish_reason="stop" if output["stop"] else "length",
            )
        ]
        return ChatCompletionResponse(model=model, choices=choices, usage=usage)
    else:
        generator = model_service.generate_stream(model, prompt, max_tokens, temperature, stop, "chat")
        return StreamingResponse(generator, media_type="text/event-stream")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=str, default="/data_cfs/public_weights/huggingface/deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                        help="checkpoint path")
    parser.add_argument("--model", type=str, default="codewise-7b", help="model name")
    args = parser.parse_args()

    model_name = args.model
    model_service = ModelService(args.checkpoint, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug", timeout_keep_alive=10)
