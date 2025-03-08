# Using 🤗 Transformers

## Source

### src/transformers/pipelines/__init__.py

- [src/transformers/pipelines/__init__.py](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py)

if `pipeline("sentiment-analysis")` is called, `dbmdz/bert-large-cased-finetuned-conll03-english` model is used:

```py
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    "vqa": "visual-question-answering",
    "text-to-speech": "text-to-audio",
}
SUPPORTED_TASKS = {
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "tf": (TFAutoModelForTokenClassification,) if is_tf_available() else (),
        "pt": (AutoModelForTokenClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("dbmdz/bert-large-cased-finetuned-conll03-english", "4c53496"),
                "tf": ("dbmdz/bert-large-cased-finetuned-conll03-english", "4c53496"),
            },
        },
        "type": "text",
    },
}
```

