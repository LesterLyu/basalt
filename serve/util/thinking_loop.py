"""Find exact periodic suffixes in a single request's generated thinking tokens."""
from array import array


class ThinkingLoop:
    MAX_PERIOD = 4096
    PROBE_EVERY = 32
    ANCHOR = 12
    THIRD_COPY = 64

    def __init__(self, repeat_tokens: int):
        self.threshold = repeat_tokens
        self.window = self.MAX_PERIOD + max(repeat_tokens, self.MAX_PERIOD + self.THIRD_COPY)
        self.reset()

    def reset(self):
        self.ids = array("I")
        self.last = None
        self.run = 0
        self.since_probe = 0

    def push(self, token: int, clean: bool = True) -> int:
        """Return the period, or zero. A partial UTF-8 character or parser tag delays the probe."""
        self.run = self.run + 1 if token == self.last else 1
        self.last = token
        self.ids.append(token)
        if len(self.ids) > 2 * self.window:
            del self.ids[:-self.window]
        self.since_probe += 1
        if not clean:
            return 0
        if self.run >= self.threshold:
            return 1
        if self.since_probe < self.PROBE_EVERY or len(self.ids) < max(self.ANCHOR, self.threshold) + 2:
            return 0
        self.since_probe = 0

        # Find candidates by matching a short suffix in C, then verify the entire periodic suffix.
        # Two complete copies plus a prefix of a third rule out a single reread of a long passage.
        size = self.ids.itemsize
        data = self.ids.tobytes()
        end = len(data)
        anchor = data[-self.ANCHOR * size:]
        max_period = min(self.MAX_PERIOD, len(self.ids) - self.threshold, (len(self.ids) - 1) // 2)
        lo = max(0, end - (max_period + self.ANCHOR) * size)
        hi = end - 2 * size
        while (pos := data.rfind(anchor, lo, hi)) >= 0:
            hi = pos + len(anchor) - 1
            if pos % size:
                continue
            lag = end - len(anchor) - pos
            period = lag // size
            repeat = max(self.threshold, period + min(period, self.THIRD_COPY)) * size
            if end >= lag + repeat and data[end - repeat:] == data[end - lag - repeat:end - lag]:
                return period
        return 0
