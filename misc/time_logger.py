from datetime import datetime


class TimeLogger:
    def __init__(self) -> None:
        super().__init__()
        self._start = datetime.now()

    def log(self):
        self._end = datetime.now()

    def __str__(self) -> str:
        return str(self._end - self._start)
