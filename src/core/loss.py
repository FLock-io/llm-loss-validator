import math
import numbers


def calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes):
    """
    Calculates BPC (Bits Per Character) and bPPL (bits Per Character Perplexity).

    Args:
        eval_loss (float): Average token-level loss in nats.
        total_target_tokens (int): Total number of target tokens.
        total_bytes (int): Total number of target bytes.

    Returns:
        dict: A dictionary containing 'bpc', 'bppl', 'nll_token_nats_total',
              'nll_token_bits_total'.
              Returns values like {'bpc': float('inf'), 'bppl': float('inf'), ...}
              if total_bytes is 0, eval_loss is invalid (non-real, NaN, or infinity).
              'bppl' will also be float('inf') if bpc is float('inf') or if
              math.pow(2, bpc) calculation overflows for a large finite bpc.
    """
    if (
        total_bytes == 0
        or not isinstance(eval_loss, numbers.Real)
        or math.isnan(eval_loss)
        or math.isinf(eval_loss)
    ):
        return {
            "bpc": float("inf"),
            "bppl": float("inf"),
            "nll_token_nats_total": float("nan"),
            "nll_token_bits_total": float("nan"),
        }

    nll_token_nats_total = eval_loss * total_target_tokens
    nll_token_bits_total = nll_token_nats_total / math.log(2)
    bpc = nll_token_bits_total / total_bytes

    if math.isinf(bpc):
        bppl = float("inf")
    else:
        try:
            bppl = math.pow(2, bpc)
        except OverflowError:
            bppl = float("inf")

    return {
        "bpc": bpc,
        "bppl": bppl,
        "nll_token_nats_total": nll_token_nats_total,
        "nll_token_bits_total": nll_token_bits_total,
    }


def get_token_byte_ratio(total_target_tokens, total_bytes):
    """
    Calculates the token to byte ratio.

    Args:
        total_target_tokens (int): Total number of target tokens.
        total_bytes (int): Total number of target bytes.

    Returns:
        float: The token to byte ratio. Returns float('inf') if total_bytes is 0.
    """
    if total_bytes == 0:
        return float("inf")
    return total_target_tokens / total_bytes


def calculate_bytes_and_tokens(eval_dataset, tokenizer, logger):
    """
    Calculates total bytes and target tokens in the evaluation dataset.

    Args:
        eval_dataset: The evaluation dataset.
        tokenizer: The tokenizer.
        logger: The logger instance.

    Returns:
        tuple: A tuple containing total_bytes and total_target_tokens.
    """
    total_bytes = 0
    total_target_tokens = 0
    logger.info(
        "Calculating total bytes and target tokens in the evaluation dataset..."
    )
    for i in range(len(eval_dataset)):
        item = eval_dataset[i]
        target_ids = [
            id for id, mask in zip(item["input_ids"], item["target_mask"]) if mask == 1
        ]
        if target_ids:
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            total_bytes += len(target_text.encode("utf-8"))
            total_target_tokens += len(target_ids)
    return total_bytes, total_target_tokens
