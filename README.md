# transformers-online-inference

## 介绍
[transformers](https://github.com/huggingface/transformers) 是 python 中的一个离线推理框架，支持 [huggingface](https://huggingface.co/models) 上面的主流模型。
但是它不适用于在线推理场景，本项目基于 [transformers](https://github.com/huggingface/transformers) 和 [fastapi](https://github.com/tiangolo/fastapi) 实现了一个在线推理框架，支持 [huggingface](https://huggingface.co/models) 上面的主流模型。

### 特性
- 适配了 openai v1/completions 和 v1/chat/completions 接口
- 支持流式调用
- 支持 stop_words
- 支持 context cancel，并且在context cancel之后，模型会停止推理，节省GPU资源
- 支持多进程推理

### 不足
- 推理速度较慢，并发较低
- 只适合小规模用户量使用，可以用于早期验证场景，正式上线还是需要采用 [vllm](https://github.com/vllm-project/vllm/) 等推理框架

## 使用指南
安装依赖
```shell
pip install -r requirements.txt
```
启动服务
```shell
python main.py
```

验证服务
```shell
curl 49.235.138.227:8000/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codewise-7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "如何使用nginx进行负载均衡？"}
    ],
    "max_tokens": 128,
    "temperature": 1,
    "stream": true,
    "skip_special_tokens": false
  }'

curl 49.235.138.227:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
      "model": "codewise-7b",
      "prompt": "def quick_sort",
      "max_tokens": 64,
      "temperature": 0.2,
      "stream": true,
      "stop": []
  }'
```

openai_api_protocol.py 来源于 [openai_api_protocol.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py)