import logging
import os
from typing import override


class RankZeroFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return os.environ.get("LOCAL_RANK", "0") == "0"
