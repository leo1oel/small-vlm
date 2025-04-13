CHAT_TEMPLATES = {
    "plain": """
        {%- for message in messages -%}
        {{ message['content'] }}{{ '\n' }}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
        {{ '\n' }}
        {%- endif -%}
    """,
    "chat": """
        {%- if messages[0]['role'] == 'system' -%}
        <|system|>{{ messages[0]['content'] }}<|end|>
        {%- endif -%}
        {%- for message in messages -%}
        {%- if message['role'] == 'user' -%}
        <|user|>{{ message['content'] }}<|end|>
        {%- elif message['role'] == 'assistant' -%}
        <|assistant|>{{ message['content'] }}<|end|>
        {%- endif -%}
        {%- endfor -%}
        <|endoftext|>
        {%- if add_generation_prompt -%}
        <|assistant|>
        {%- endif -%}
    """,
}

import logging

log = logging.getLogger(__name__)


def get_chat_template(template_name: str) -> str:
    if template_name not in CHAT_TEMPLATES:
        log.error(f"Chat template '{template_name}' not found")
        raise ValueError(f"Chat template '{template_name}' not found")
    return CHAT_TEMPLATES[template_name]
