from loguru import logger
from client.fed_ledger import FedLedger
import sys


def handle_os_error(e: OSError):
    if "No space left on device" in str(e):
        logger.error("No more disk space, exiting with code 101")
        sys.exit(101)
    else:
        logger.error("Unknown OSError detected, exiting with code 100, will restart...")
        sys.exit(100)


def handle_runtime_error(e: RuntimeError, assignment_id: str, client: FedLedger):
    if "CUDA error: device-side assert triggered" in str(e):
        logger.error(
            "CUDA device-side assert triggered error detected, exiting with code 10, will restart..."
        )
        sys.exit(100)
    if "out of memory" in str(e):
        logger.error(
            "CUDA out of memory error detected, will mark the assignment as failed"
        )
        client.mark_assignment_as_failed(assignment_id)
    else:
        logger.error(
            "Unknown RuntimeError detected, exiting with code 100, will restart..."
        )
        sys.exit(100)


def handle_value_error(e: ValueError, assignment_id: str, client: FedLedger):
    if "FP16 Mixed precision training with AMP or APEX" in str(e):
        logger.error(
            "FP16 Mixed precision training with AMP or APEX error detected, exiting with code 101"
        )
        sys.exit(101)
    else:
        logger.error(
            "Unknown ValueError detected, exiting with code 100, will restart..."
        )
        sys.exit(100)
