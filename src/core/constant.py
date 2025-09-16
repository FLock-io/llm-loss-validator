SUPPORTED_BASE_MODELS = [
    # qwen2.5
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    # yi 1.5
    "01-ai/Yi-1.5-6B",
    "01-ai/Yi-1.5-6B-Chat",
    "01-ai/Yi-1.5-9B",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B",
    "01-ai/Yi-1.5-34B-Chat",
    # mistral
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410",
    # gemma2
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    # llama3
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    # llama3.1
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    # phi3
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    # phi4
    "microsoft/Phi-4-mini-instruct",
    "microsoft/phi-4",
]

MODEL_TEMPLATE_MAP = {
    # Qwen
    "Qwen/Qwen2.5-0.5B": "qwen1.5",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen1.5",
    "Qwen/Qwen2.5-1.5B": "qwen1.5",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen1.5",
    "Qwen/Qwen2.5-3B": "qwen1.5",
    "Qwen/Qwen2.5-3B-Instruct": "qwen1.5",
    "Qwen/Qwen2.5-7B": "qwen1.5",
    "Qwen/Qwen2.5-7B-Instruct": "qwen1.5",
    "Qwen/Qwen2.5-14B": "qwen1.5",
    "Qwen/Qwen2.5-14B-Instruct": "qwen1.5",
    "Qwen/Qwen2.5-32B": "qwen1.5",
    "Qwen/Qwen2.5-32B-Instruct": "qwen1.5",
    "Qwen/Qwen2.5-72B": "qwen1.5",
    "Qwen/Qwen2.5-72B-Instruct": "qwen1.5",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3",
    # Yi
    "01-ai/Yi-1.5-6B": "yi",
    "01-ai/Yi-1.5-6B-Chat": "yi",
    "01-ai/Yi-1.5-9B": "yi",
    "01-ai/Yi-1.5-9B-Chat": "yi",
    "01-ai/Yi-1.5-34B": "yi",
    "01-ai/Yi-1.5-34B-Chat": "yi",
    # Mistral
    "mistralai/Mistral-7B-v0.3": "mistral",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "mistralai/Ministral-8B-Instruct-2410": "mistral",
    # Mixtral
    "mistralai/Mixtral-8x7B-v0.1": "mixtral",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral",
    # Gemma 2
    "google/gemma-2-2b": "gemma",
    "google/gemma-2-9b": "gemma",
    "google/gemma-2-27b": "gemma",
    "google/gemma-2-2b-it": "gemma",
    "google/gemma-2-9b-it": "gemma",
    "google/gemma-2-27b-it": "gemma",
    # LLaMA 3 + 3.1
    "meta-llama/Meta-Llama-3-8B": "llama3",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3",
    "meta-llama/Meta-Llama-3-70B": "llama3",
    "meta-llama/Meta-Llama-3-70B-Instruct": "llama3",
    "meta-llama/Meta-Llama-3.1-8B": "llama3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3",
    "meta-llama/Meta-Llama-3.1-70B": "llama3",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3",
    # Phi 3
    "microsoft/Phi-3.5-mini-instruct": "phi3",
    "microsoft/Phi-3-mini-4k-instruct": "phi3",
    "microsoft/Phi-3-medium-4k-instruct": "phi3",
    # Phi 4
    "microsoft/Phi-4-mini-instruct": "phi4",
    "microsoft/phi-4": "phi4",
}
