import math
from loguru import logger
import numbers


def _log_summary_table(
    model_name_or_path,
    eval_loss,
    bpc_metrics,
    token_byte_ratio,
    total_target_tokens,
    total_bytes,
    vocab_size,
    model_params_m,
):
    """Helper function to log summary table in vertical format."""

    table_data = {
        "Model Name": model_name_or_path,
        "Token Loss (nats)": f"{eval_loss:.5f}"
        if isinstance(eval_loss, numbers.Real) and not math.isnan(eval_loss)
        else str(eval_loss),
        "BPC": f"{bpc_metrics['bpc']:.5f}"
        if not math.isinf(bpc_metrics["bpc"]) and not math.isnan(bpc_metrics["bpc"])
        else str(bpc_metrics["bpc"]),
        "bPPL": f"{bpc_metrics['bppl']:.5f}"
        if not math.isinf(bpc_metrics["bppl"]) and not math.isnan(bpc_metrics["bppl"])
        else str(bpc_metrics["bppl"]),
        "T/B Ratio": f"{token_byte_ratio:.4f}"
        if not math.isinf(token_byte_ratio) and not math.isnan(token_byte_ratio)
        else str(token_byte_ratio),
        "Target Tokens": str(total_target_tokens),
        "Target Bytes": str(total_bytes),
        "Vocab Size": str(vocab_size),
        "Total Params (M)": f"{model_params_m:.2f}"
        if isinstance(model_params_m, numbers.Real) and not math.isnan(model_params_m)
        else str(model_params_m),
    }

    label_width = max(len(label) for label in table_data.keys())
    value_width = max(len(str(value)) for value in table_data.values())
    total_width = label_width + value_width + 3

    header = (
        "=" * ((total_width - 20) // 2)
        + " Validation Summary "
        + "=" * ((total_width - 20) // 2)
    )
    logger.info(f"\n{header}")

    for label, value in table_data.items():
        if label == "Model Name" and len(value) > value_width:
            value = value[: value_width - 3] + "..."
        print(f"{label:<{label_width}} | {value:<{value_width}}")

    print("=" * total_width + "\n")
