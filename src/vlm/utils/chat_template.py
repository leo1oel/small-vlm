CHAT_TEMPLATES = {
    "plain": """
        {%- for message in messages -%}
        {%- if message['role'] == 'user' -%}
        <image>
        {%- elif message['role'] == 'assistant' -%}
        {{ message['content'] }}
        {%- endif -%}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
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
        {%- if add_generation_prompt -%}
        <|assistant|>
        {%- endif -%}
    """,
}

import logging

log = logging.getLogger(__name__)


def get_chat_template(
    template_name: str, eos_token: str, system_token: str, user_token: str, assistant_token: str
) -> str:
    if template_name not in CHAT_TEMPLATES:
        log.error(f"Chat template '{template_name}' not found")
        raise ValueError(f"Chat template '{template_name}' not found")

    template_string = CHAT_TEMPLATES[template_name]

    if template_name == "chat":
        template_string = template_string.replace("<|end|>", eos_token)
        template_string = template_string.replace("<|system|>", system_token)
        template_string = template_string.replace("<|user|>", user_token)
        template_string = template_string.replace("<|assistant|>", assistant_token)
    return template_string
