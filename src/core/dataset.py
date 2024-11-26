import json

from loguru import logger
from torch.utils.data import Dataset
from .tool_utils import tool_formater, function_formatter


class UnifiedSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.tool_format = template.tool_format
        self.function_format = template.function_format
        self.observation_format = template.observation_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info("Loading data: {}".format(file))
        with open(file, "r", encoding="utf8") as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data["system"].strip() if "system" in data.keys() else self.system

            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        # setting tool information
        if "tools" in data.keys() and data["tools"]:
            tools = json.loads(data["tools"])
            tool_prompt = tool_formater(tools)
            tool_text = self.tool_format.format(content=tool_prompt)
            tool_tokens = self.tokenizer.encode(tool_text, add_special_tokens=False)
            input_ids = input_ids + tool_tokens
            target_mask = target_mask + [0] * len(tool_tokens)

        conversations = data["conversations"]

        input_buffer = ""
        for conversation in conversations:
            role = conversation["role"]
            content = conversation["content"].strip()
            if role != "assistant":
                if role == "user":
                    human = self.user_format.format(
                        content=content, stop_token=self.tokenizer.eos_token
                    )
                    input_buffer += human

                elif role == "function_call":
                    tool_calls = function_formatter(json.loads(content))
                    function = self.function_format.format(content=tool_calls)
                    input_buffer += function

                elif role == "observation":
                    observation = self.observation_format.format(content=content)
                    input_buffer += observation
            else:
                assistant = self.assistant_format.format(
                    content=content, stop_token=self.tokenizer.eos_token
                )

                input_tokens = self.tokenizer.encode(
                    input_buffer, add_special_tokens=False
                )
                output_tokens = self.tokenizer.encode(
                    assistant, add_special_tokens=False
                )

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
                input_buffer = ""
        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[: self.max_seq_length]
        target_mask = target_mask[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        return inputs
