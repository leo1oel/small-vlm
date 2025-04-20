# small-vlm

![Architecture](assets/architecture.png)

A small vision-language model (VLM) implementation in PyTorch. The model consists of three main components:

- **Visual Encoder**: Extracts visual features from images using vision transformers
- **Language Model**: Processes text and generates responses using LLMs
- **Connector**: Connects visual and language features for multimodal understanding

You can switch different visual encoders, language models and connectors by changing the config.

If you want to use flash-attention-2 for training, run

```
uv pip install flash-attn --no-build-isolation
```

---

## Project Docs

For how to install uv and Python, see [installation.md](installation.md).

For development workflows, see [development.md](development.md).

For instructions on publishing to PyPI, see [publishing.md](publishing.md).

---

_This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv)._
