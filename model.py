import asyncio
import json
import time
from threading import Thread

import shortuuid
import torch
from loguru import logger
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList,
    TextIteratorStreamer
)

from openai_api_protocol import (
    ChatCompletionStreamResponse, ChatCompletionResponseStreamChoice,
    DeltaMessage, CompletionStreamResponse, CompletionResponseStreamChoice
)
from stop import remove_suffix

SSE_END = "data: [DONE]\n\n"

class DefaultStopWordsCriteria(StoppingCriteria):
    def __init__(self, stops=None, encounters=1):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class CancelStopCriteria(StoppingCriteria):
    def __init__(self):
        super().__init__()
        self.stop = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 判断请求是否cancel
        if self.stop:
            logger.info("cancel model generation")
            return True
        return False

    def cancel(self):
        self.stop = True


class ModelService:
    def __init__(self, checkpoint, model_name):
        start = time.time()
        self.model_name = model_name
        model, tokenizer = self.load_model(checkpoint)
        self.model = model
        self.tokenizer = tokenizer

        logger.info(f"loading model in {time.time() - start} seconds")

        # warm_up
        prompt = 'Q: What is the largest animal?\nA:'
        inputs = tokenizer(prompt, return_tensors="pt")
        logger.info(f"warm up...")
        with torch.no_grad():
            model.generate(inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), max_new_tokens=16)
            logger.info("warm up done")

    def load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto",
                                                     torch_dtype=torch.float16).eval()
        return model, tokenizer

    def get_stopping_criteria(self, stop_words):
        if stop_words is None:
            return StoppingCriteriaList([])
        stop_words_ids = [
            self.tokenizer.encode(stop_word, add_special_tokens=False, return_tensors='pt').squeeze().cuda()
            for stop_word in stop_words
        ]
        for i in range(len(stop_words_ids)):
            ids = stop_words_ids[i]
            if ids.dim() == 0:
                stop_words_ids[i] = ids = ids.unsqueeze(0)
            if "llama" in str(type(self.tokenizer)) and len(ids) > 0 and torch.all(torch.eq(ids[0], 29871)):
                stop_words_ids[i] = ids[1:]
        logger.info("original stop_words: {}, stop_ids: {}".format(stop_words, stop_words_ids)
                    .encode('unicode_escape').decode())
        stopping_criteria_stop_words = DefaultStopWordsCriteria(stops=stop_words_ids)
        return StoppingCriteriaList([stopping_criteria_stop_words])

    def generate(self, prompt, max_new_tokens=32, temperature=0.2, stop_words=None):
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            stopping_criteria = self.get_stopping_criteria(stop_words)
            generate_ids = self.model.generate(inputs.input_ids.cuda(),
                                               attention_mask=inputs.attention_mask.cuda(),
                                               max_new_tokens=max_new_tokens,
                                               temperature=temperature,
                                               stopping_criteria=stopping_criteria)
        res = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]

        prompt_tokens = len(inputs.input_ids[0])
        total_tokens = len(generate_ids[0])
        completion_tokens = total_tokens - prompt_tokens
        elapse = time.time() - start_time
        logger.info(f"Output generated in {elapse:.2f}s, {(completion_tokens / elapse):.2f} tokens/s, "
                    f"{completion_tokens} new tokens generated.\nprompt:\n{prompt}\noutput:\n{res}\n")

        logger.info("clearing cuda cache")
        torch.cuda.empty_cache()

        return {
            "text": res,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stop": False
        }

    async def _decode_generator_chat(self, generator, model, stop_words=None):
        """
        _decode_generator_chat 包装chat接口需要的generator
        :param generator:
        :param model:
        :return:
        """
        if stop_words is None:
            stop_words = []
        string_buffer = ""
        id_ = f"chatcmpl-{shortuuid.random()}"
        created = int(time.time())
        stop = False
        for data in generator:
            if stop:
                break
            if data == "":
                continue
            before = data
            data, stop, stop_word = remove_suffix(string_buffer, data, stop_words)
            if stop:
                logger.debug(f"hit stop_word: {stop_word}, before: {before}, after: {data}")
            string_buffer += data

            choices = [
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=data)
                )
            ]
            chunk = ChatCompletionStreamResponse(
                choices=choices, model=model, id=id_, created=created
            )
            chunk_dict = chunk.dict(exclude_unset=True)
            json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
            yield f"data: {json_chunk}\n\n"

        # 最后一个word，需要带上finish_reason
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=""),
            finish_reason="stop"
        )
        chunk = ChatCompletionStreamResponse(
            choices=[choice_data], model=model, id=id_, created=created
        )
        chunk_dict = chunk.dict(exclude_unset=True)
        json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
        yield f"data: {json_chunk}\n\n"

        # SSE协议stop标志
        yield SSE_END

        logger.info(f"chat request finished, output:{string_buffer}")

    async def _decode_generator_completion(self, generator, model, stop_words=None):
        """
        _decode_generator_completions 包装completions接口需要的generator
        :param generator:
        :param model:
        :return:
        """
        if stop_words is None:
            stop_words = []
        string_buffer = ""
        stop = False
        id_ = f"cmpl-{shortuuid.random()}"
        created = int(time.time())
        for data in generator:
            if stop:
                break
            if data == "":
                continue
            before = data
            data, stop, stop_word = remove_suffix(string_buffer, data, stop_words)
            if stop:
                logger.debug(f"hit stop_word: {stop_word}, before: {before}, after: {data}")
            string_buffer += data

            choices = [
                CompletionResponseStreamChoice(
                    index=0,
                    text=data
                )
            ]
            chunk = CompletionStreamResponse(
                choices=choices, model=model, id=id_, created=created
            )
            chunk_dict = chunk.dict(exclude_unset=True)
            json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
            yield f"data: {json_chunk}\n\n"

        # 最后一个word，需要带上finish_reason
        choices = [
            CompletionResponseStreamChoice(
                index=0,
                text="",
                finish_reason="stop"
            )
        ]
        chunk = CompletionStreamResponse(
            choices=choices, model=model, id=id_, created=created
        )
        chunk_dict = chunk.dict(exclude_unset=True)
        json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
        yield f"data: {json_chunk}\n\n"

        # SSE协议stop标志
        yield SSE_END

        logger.info(f"completions request finished, output: {string_buffer}")

    def get_streamer(self, prompt, max_new_tokens=32, temperature=0.2, cancel_stop_criteria=None):
        """
        generate_stream 是使用transformers库里面的TextIteratorStreamer实现的流式输出调用
        实现参考官方文档：https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextIteratorStreamer
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            stopping_criteria = StoppingCriteriaList([cancel_stop_criteria])
            # skip_prompt=True 可使最终输出中不包含输入的prompt部分
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

            def call_stream():
                try:
                    self.model.generate(inputs.input_ids.cuda(),
                                        streamer=streamer,
                                        attention_mask=inputs.attention_mask.cuda(),
                                        max_new_tokens=max_new_tokens,
                                        temperature=temperature,
                                        stopping_criteria=stopping_criteria)
                except Exception as e:
                    logger.error("streamer infer error: {}".format(e))
                    streamer.text_queue.put(streamer.stop_signal, timeout=streamer.timeout)
                    return

            thread = Thread(target=call_stream)
            thread.start()
            return streamer

    async def generate_stream(self, model, prompt, max_new_tokens=32, temperature=0.2, stop_words=None, type_="chat"):
        cancel_stop_criteria = CancelStopCriteria()
        try:
            streamer = self.get_streamer(prompt, max_new_tokens, temperature, cancel_stop_criteria)
            if type_ == "chat":
                generator = self._decode_generator_chat(streamer, model, stop_words)
            else:
                generator = self._decode_generator_completion(streamer, model, stop_words)
            async for r in generator:
                yield r
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info("catch cancel signal")
            cancel_stop_criteria.cancel()

        logger.info("clearing cuda cache")
        torch.cuda.empty_cache()
