# transformers-online-inference

## 介绍
[transformers](https://github.com/huggingface/transformers) 是 python 中的一个离线推理框架，支持 [huggingface](https://huggingface.co/models) 上面的主流模型。
但是它不适用于在线推理场景，本项目基于 [transformers](https://github.com/huggingface/transformers) 和 [fastapi](https://github.com/tiangolo/fastapi) 实现了一个在线推理框架，支持 [huggingface](https://huggingface.co/models) 上面的主流模型。

1. 特性：
- 适配了 openai v1/completions 和 v1/chat/completions 接口
- 支持流式调用
- 支持 context cancel，并且在context cancel之后，模型会停止推理，节省GPU资源
- 支持多进程推理

2. 缺点：
- 推理速度较慢，并发较低
- 只适合小规模用户量使用，可以用于早期验证场景，正式上线还是需要采用 [vllm](https://github.com/vllm-project/vllm/) 等推理框架

openai_api_protocol.py 来源于 [openai_api_protocol.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py)