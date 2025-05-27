import math
import unittest
from unittest.mock import Mock

from src.core.loss import (
    calculate_bpc_bppl_metrics,
    get_token_byte_ratio,
    calculate_bytes_and_tokens,
)


class TestCalculateBpcBpplMetrics(unittest.TestCase):
    """Test cases for calculate_bpc_bppl_metrics function."""

    def test_valid_inputs(self):
        """Test with valid inputs."""
        # Test case from TEST.md: eval_loss = 0.693147 (ln(2))
        eval_loss = 0.693147  # ln(2)
        total_target_tokens = 100
        total_bytes = 200

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        # Expected calculations:
        # nll_token_nats_total = 0.693147 * 100 = 69.3147
        # nll_token_bits_total = 69.3147 / ln(2) = 100
        # bpc = 100 / 200 = 0.5
        # bppl = 2^0.5 = sqrt(2) ≈ 1.414

        self.assertAlmostEqual(result["nll_token_nats_total"], 69.3147, places=4)
        self.assertAlmostEqual(result["nll_token_bits_total"], 100.0, places=4)
        self.assertAlmostEqual(result["bpc"], 0.5, places=4)
        self.assertAlmostEqual(result["bppl"], math.sqrt(2), places=4)

    def test_total_bytes_zero(self):
        """Test with total_bytes = 0."""
        eval_loss = 1.0
        total_target_tokens = 100
        total_bytes = 0

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        expected = {
            "bpc": float("inf"),
            "bppl": float("inf"),
            "nll_token_nats_total": float("nan"),
            "nll_token_bits_total": float("nan"),
        }

        self.assertEqual(result["bpc"], expected["bpc"])
        self.assertEqual(result["bppl"], expected["bppl"])
        self.assertTrue(math.isnan(result["nll_token_nats_total"]))
        self.assertTrue(math.isnan(result["nll_token_bits_total"]))

    def test_eval_loss_nan(self):
        """Test with eval_loss as nan."""
        eval_loss = float("nan")
        total_target_tokens = 100
        total_bytes = 200

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        self.assertEqual(result["bpc"], float("inf"))
        self.assertEqual(result["bppl"], float("inf"))
        self.assertTrue(math.isnan(result["nll_token_nats_total"]))
        self.assertTrue(math.isnan(result["nll_token_bits_total"]))

    def test_eval_loss_inf(self):
        """Test with eval_loss as inf."""
        eval_loss = float("inf")
        total_target_tokens = 100
        total_bytes = 200

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        self.assertEqual(result["bpc"], float("inf"))
        self.assertEqual(result["bppl"], float("inf"))
        self.assertTrue(math.isnan(result["nll_token_nats_total"]))
        self.assertTrue(math.isnan(result["nll_token_bits_total"]))

    def test_eval_loss_non_real(self):
        """Test with eval_loss non-real (string)."""
        eval_loss = "abc"
        total_target_tokens = 100
        total_bytes = 200

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        self.assertEqual(result["bpc"], float("inf"))
        self.assertEqual(result["bppl"], float("inf"))
        self.assertTrue(math.isnan(result["nll_token_nats_total"]))
        self.assertTrue(math.isnan(result["nll_token_bits_total"]))

    def test_large_bpc_leading_to_inf_bppl(self):
        """Test with large bpc leading to bppl = inf."""
        eval_loss = 1000  # Large nats
        total_target_tokens = 1000
        total_bytes = 1

        # This test confirms that bppl becomes inf when bpc is very large
        # due to math.pow(2, bpc) overflowing or bpc itself being inf.
        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)
        self.assertEqual(result["bppl"], float("inf"))
        # We can also check bpc if it's not excessively large to cause direct inf
        # For eval_loss=1000, total_target_tokens=1000, total_bytes=1:
        # nll_token_nats_total = 1000 * 1000 = 1,000,000
        # nll_token_bits_total = 1,000,000 / ln(2) approx 1,442,695
        # bpc = 1,442,695 / 1 = 1,442,695
        # This bpc value will cause math.pow(2, bpc) to overflow.
        self.assertTrue(
            math.isfinite(result["bpc"]) and result["bpc"] > 1000
        )  # bpc is large and finite

    def test_moderately_large_bpc(self):
        """Test with moderately large bpc that doesn't overflow."""
        eval_loss = 2.0  # Moderate nats
        total_target_tokens = 10
        total_bytes = 1

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        # bpc will be large but not overflow
        expected_nll_bits = 20 / math.log(2)  # ~28.85
        expected_bpc = expected_nll_bits / 1  # ~28.85

        self.assertAlmostEqual(result["bpc"], expected_bpc, places=1)
        self.assertTrue(result["bppl"] > 1000000)  # Very large but finite

    def test_zero_eval_loss(self):
        """Test with eval_loss = 0."""
        eval_loss = 0.0
        total_target_tokens = 100
        total_bytes = 200

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        self.assertEqual(result["nll_token_nats_total"], 0.0)
        self.assertEqual(result["nll_token_bits_total"], 0.0)
        self.assertEqual(result["bpc"], 0.0)
        self.assertEqual(result["bppl"], 1.0)  # 2^0 = 1

    def test_negative_eval_loss(self):
        """Test with negative eval_loss."""
        eval_loss = -1.0
        total_target_tokens = 100
        total_bytes = 200

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        self.assertEqual(result["nll_token_nats_total"], -100.0)
        self.assertLess(result["nll_token_bits_total"], 0)
        self.assertLess(result["bpc"], 0)
        self.assertLess(result["bppl"], 1.0)  # 2^(negative) < 1

    def test_zero_total_target_tokens(self):
        """Test with total_target_tokens = 0 but total_bytes > 0."""
        eval_loss = 0.5
        total_target_tokens = 0
        total_bytes = 100

        result = calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes)

        # nll_token_nats_total = 0.5 * 0 = 0
        # nll_token_bits_total = 0 / ln(2) = 0
        # bpc = 0 / 100 = 0
        # bppl = 2^0 = 1
        self.assertEqual(result["nll_token_nats_total"], 0.0)
        self.assertEqual(result["nll_token_bits_total"], 0.0)
        self.assertEqual(result["bpc"], 0.0)
        self.assertEqual(result["bppl"], 1.0)


class TestGetTokenByteRatio(unittest.TestCase):
    """Test cases for get_token_byte_ratio function."""

    def test_valid_inputs(self):
        """Test with valid inputs."""
        total_target_tokens = 100
        total_bytes = 200

        result = get_token_byte_ratio(total_target_tokens, total_bytes)

        self.assertEqual(result, 0.5)

    def test_total_bytes_zero(self):
        """Test with total_bytes = 0."""
        total_target_tokens = 100
        total_bytes = 0

        result = get_token_byte_ratio(total_target_tokens, total_bytes)

        self.assertEqual(result, float("inf"))

    def test_total_target_tokens_zero(self):
        """Test with total_target_tokens = 0."""
        total_target_tokens = 0
        total_bytes = 200

        result = get_token_byte_ratio(total_target_tokens, total_bytes)

        self.assertEqual(result, 0.0)

    def test_equal_tokens_and_bytes(self):
        """Test with equal tokens and bytes."""
        total_target_tokens = 100
        total_bytes = 100

        result = get_token_byte_ratio(total_target_tokens, total_bytes)

        self.assertEqual(result, 1.0)

    def test_more_tokens_than_bytes(self):
        """Test with more tokens than bytes."""
        total_target_tokens = 200
        total_bytes = 100

        result = get_token_byte_ratio(total_target_tokens, total_bytes)

        self.assertEqual(result, 2.0)


class TestCalculateBytesAndTokens(unittest.TestCase):
    """Test cases for calculate_bytes_and_tokens function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.mock_logger = Mock()
        self.mock_tokenizer = Mock()

    def test_empty_dataset(self):
        """Test with an empty dataset."""
        eval_dataset = []

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        self.assertEqual(total_bytes, 0)
        self.assertEqual(total_target_tokens, 0)
        self.mock_logger.info.assert_called_once_with(
            "Calculating total bytes and target tokens in the evaluation dataset..."
        )

    def test_dataset_with_target_tokens(self):
        """Test with a dataset containing items with target tokens."""
        # Mock dataset items
        item1 = {"input_ids": [1, 2, 3], "target_mask": [0, 1, 1]}
        item2 = {"input_ids": [4, 5, 6], "target_mask": [0, 1, 0]}

        eval_dataset = [item1, item2]

        # Mock tokenizer.decode behavior
        def mock_decode(token_ids, skip_special_tokens=True):
            if token_ids == [2, 3]:
                return "ab"  # 2 bytes
            elif token_ids == [5]:
                return "c"  # 1 byte
            else:
                return ""

        self.mock_tokenizer.decode.side_effect = mock_decode

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        # Expected: total_bytes = 3 (2 from "ab" + 1 from "c")
        # Expected: total_target_tokens = 3 (2 from item1 + 1 from item2)
        self.assertEqual(total_bytes, 3)
        self.assertEqual(total_target_tokens, 3)

        # Verify tokenizer.decode was called correctly
        self.mock_tokenizer.decode.assert_any_call([2, 3], skip_special_tokens=True)
        self.mock_tokenizer.decode.assert_any_call([5], skip_special_tokens=True)

    def test_dataset_with_no_target_tokens(self):
        """Test with a dataset where some items have no target tokens."""
        item1 = {"input_ids": [1, 2, 3], "target_mask": [0, 1, 1]}
        item2 = {
            "input_ids": [4, 5, 6],
            "target_mask": [0, 0, 0],  # No target tokens
        }

        eval_dataset = [item1, item2]

        def mock_decode(token_ids, skip_special_tokens=True):
            if token_ids == [2, 3]:
                return "ab"  # 2 bytes
            else:
                return ""

        self.mock_tokenizer.decode.side_effect = mock_decode

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        # Expected: total_bytes = 2, total_target_tokens = 2
        self.assertEqual(total_bytes, 2)
        self.assertEqual(total_target_tokens, 2)

        # Verify tokenizer.decode was only called for item1
        self.mock_tokenizer.decode.assert_called_once_with(
            [2, 3], skip_special_tokens=True
        )

    def test_dataset_with_special_tokens_skipped(self):
        """Test with dataset items having special tokens in target that are skipped."""
        item = {"input_ids": [101, 2054, 102], "target_mask": [1, 1, 1]}

        eval_dataset = [item]

        # Mock tokenizer.decode to skip special tokens
        self.mock_tokenizer.decode.return_value = (
            "word"  # 4 bytes after skipping special tokens
        )

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        # Expected: total_bytes = 4 (from "word")
        # Expected: total_target_tokens = 3 (count of target_ids)
        self.assertEqual(total_bytes, 4)
        self.assertEqual(total_target_tokens, 3)

        # Verify tokenizer.decode was called with skip_special_tokens=True
        self.mock_tokenizer.decode.assert_called_once_with(
            [101, 2054, 102], skip_special_tokens=True
        )

    def test_dataset_with_unicode_characters(self):
        """Test with dataset containing unicode characters."""
        item = {"input_ids": [1, 2], "target_mask": [1, 1]}

        eval_dataset = [item]

        # Mock tokenizer to return unicode text
        self.mock_tokenizer.decode.return_value = (
            "café"  # 5 bytes in UTF-8 (é is 2 bytes)
        )

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        self.assertEqual(total_bytes, 5)  # UTF-8 byte count
        self.assertEqual(total_target_tokens, 2)

    def test_dataset_with_empty_target_text(self):
        """Test with dataset where tokenizer.decode returns empty string."""
        item = {"input_ids": [1, 2], "target_mask": [1, 1]}

        eval_dataset = [item]

        # Mock tokenizer to return empty string
        self.mock_tokenizer.decode.return_value = ""

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        self.assertEqual(total_bytes, 0)
        self.assertEqual(total_target_tokens, 2)  # Still counts the tokens

    def test_logger_calls(self):
        """Test that logger is called with expected message."""
        eval_dataset = []

        calculate_bytes_and_tokens(eval_dataset, self.mock_tokenizer, self.mock_logger)

        self.mock_logger.info.assert_called_once_with(
            "Calculating total bytes and target tokens in the evaluation dataset..."
        )

    def test_dataset_with_mixed_scenarios(self):
        """Test with a dataset containing mixed scenarios."""
        items = [
            {
                "input_ids": [1, 2, 3],
                "target_mask": [0, 1, 1],  # 2 target tokens
            },
            {
                "input_ids": [4, 5],
                "target_mask": [0, 0],  # 0 target tokens
            },
            {
                "input_ids": [6, 7, 8, 9],
                "target_mask": [1, 0, 1, 1],  # 3 target tokens
            },
        ]

        eval_dataset = items

        def mock_decode(token_ids, skip_special_tokens=True):
            if token_ids == [2, 3]:
                return "hi"  # 2 bytes
            elif token_ids == [6, 8, 9]:
                return "test"  # 4 bytes
            else:
                return ""

        self.mock_tokenizer.decode.side_effect = mock_decode

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, self.mock_tokenizer, self.mock_logger
        )

        # Expected: total_bytes = 6 (2 + 0 + 4)
        # Expected: total_target_tokens = 5 (2 + 0 + 3)
        self.assertEqual(total_bytes, 6)
        self.assertEqual(total_target_tokens, 5)

    def test_dataset_item_missing_keys(self):
        """Test with a dataset item missing 'input_ids' or 'target_mask'."""
        eval_dataset_missing_input_ids = [
            {
                # 'input_ids': [1, 2, 3], # Missing key
                "target_mask": [0, 1, 1]
            }
        ]
        eval_dataset_missing_target_mask = [
            {
                "input_ids": [1, 2, 3],
                # 'target_mask': [0, 1, 1] # Missing key
            }
        ]

        with self.assertRaises(KeyError):
            calculate_bytes_and_tokens(
                eval_dataset_missing_input_ids, self.mock_tokenizer, self.mock_logger
            )

        with self.assertRaises(KeyError):
            calculate_bytes_and_tokens(
                eval_dataset_missing_target_mask, self.mock_tokenizer, self.mock_logger
            )

    def test_dataset_item_mismatched_lengths(self):
        """Test with input_ids and target_mask having mismatched lengths."""
        # zip will truncate to the shorter length. Here, target_mask is shorter.
        item_mask_shorter = {
            "input_ids": [1, 2, 3, 4],  # Length 4
            "target_mask": [0, 1, 1],  # Length 3
        }
        # Here, input_ids is shorter.
        item_ids_shorter = {
            "input_ids": [1, 2],  # Length 2
            "target_mask": [0, 1, 1],  # Length 3
        }

        eval_dataset_mask_shorter = [item_mask_shorter]
        eval_dataset_ids_shorter = [item_ids_shorter]

        self.mock_tokenizer.decode.side_effect = (
            lambda ids, skip_special_tokens: "a" * len(ids)
        )

        # Scenario 1: target_mask is shorter
        # target_ids will be [2, 3] (from first 3 elements of input_ids and target_mask)
        # total_target_tokens = 2
        # decoded text for [2,3] will be "aa", length 2 bytes
        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset_mask_shorter, self.mock_tokenizer, self.mock_logger
        )
        self.assertEqual(total_target_tokens, 2)
        self.assertEqual(total_bytes, 2)  # "aa" -> 2 bytes
        self.mock_tokenizer.decode.assert_called_with([2, 3], skip_special_tokens=True)

        self.mock_tokenizer.reset_mock()  # Reset mock for the next scenario
        self.mock_tokenizer.decode.side_effect = (
            lambda ids, skip_special_tokens: "b" * len(ids)
        )

        # Scenario 2: input_ids is shorter
        # target_ids will be [2] (from first 2 elements of input_ids and target_mask)
        # total_target_tokens = 1
        # decoded text for [2] will be "b", length 1 byte
        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset_ids_shorter, self.mock_tokenizer, self.mock_logger
        )
        self.assertEqual(total_target_tokens, 1)
        self.assertEqual(total_bytes, 1)  # "b" -> 1 byte
        self.mock_tokenizer.decode.assert_called_with([2], skip_special_tokens=True)


if __name__ == "__main__":
    unittest.main()
