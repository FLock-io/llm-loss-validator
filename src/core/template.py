from dataclasses import dataclass
from typing import Dict


@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str


template_dict: Dict[str, Template] = dict()


def register_template(
    template_name, system_format, user_format, assistant_format, system, stop_word=None
):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )


register_template(
    template_name="default",
    system_format="System: {content}\n\n",
    user_format="User: {content}\nAssistant: ",
    assistant_format="{content} {stop_token}",
    system=None,
    stop_word=None,
)


register_template(
    template_name="qwen1.5",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    system="You are a helpful assistant.",
    stop_word="<|im_end|>",
)

register_template(
    template_name="yi",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    system=None,
    stop_word="<|im_end|>",
)


register_template(
    template_name="zephyr",
    system_format="<|system|>\n{content}</s>",
    user_format="<|user|>\n{content}</s>\n<|assistant|>\n",
    assistant_format="{content}</s>\n",
    system=None,
    stop_word="</s>",
)

register_template(
    template_name="mistral",
    system_format="<s>",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content}</s>",
    system="",
    stop_word="</s>",
)

register_template(
    template_name="mixtral",
    system_format="<s>",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content}</s>",
    system="",
    stop_word="</s>",
)

register_template(
    template_name="llama2",
    system_format="<<SYS>>\n{content}\n<</SYS>>\n\n",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content} </s>",
    system="You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information.",
    stop_word="</s>",
)

register_template(
    template_name="gemma",
    system_format="<bos>",
    user_format="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    assistant_format="{content}<eos>\n",
    system="",
    stop_word="<eos>",
)

register_template(
    template_name="llama3",
    system_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user_format="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_format="{content}<|eot_id|>",
    system=None,
    stop_word="<|eot_id|>",
)
