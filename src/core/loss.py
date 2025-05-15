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
              Returns {'bpc': float('inf'), 'bppl': float('inf'), ...} if total_bytes is 0
              or eval_loss is invalid.
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
    bppl = math.pow(2, bpc) if not math.isinf(bpc) else float("inf")

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
