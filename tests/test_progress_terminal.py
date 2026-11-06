import re

from src.fetch.progress import Progress

ANSI_UP = re.compile(r"\033\[(\d*)A")
ANSI_OTHER = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


class Terminal:
    """Minimal terminal emulator that understands the sequences we emit."""

    def __init__(self):
        self.lines = [""]
        self.row = 0
        self.col = 0

    def isatty(self):
        return True

    def flush(self):
        pass

    def write(self, text):
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\r":
                self.col = 0
                index += 1
                continue
            if char == "\n":
                self.row += 1
                if self.row >= len(self.lines):
                    self.lines.append("")
                self.col = 0
                index += 1
                continue
            if char == "\033":
                match = ANSI_UP.match(text[index:])
                if match:
                    count = int(match.group(1) or 1)
                    self.row = max(0, self.row - count)
                    index += match.end()
                    continue
                if text.startswith("\033[K", index):
                    self.lines[self.row] = self.lines[self.row][: self.col]
                    index += 3
                    continue
                match = ANSI_OTHER.match(text[index:])
                if match:
                    index += match.end()
                    continue
                index += 1
                continue
            line = self.lines[self.row]
            if self.col > len(line):
                line += " " * (self.col - len(line))
            self.lines[self.row] = line[: self.col] + char + line[self.col + 1 :]
            self.col += 1
            index += 1

    def screen(self):
        return [line for line in self.lines if line.strip()]


def test_multiple_bars_do_not_duplicate():
    terminal = Terminal()
    terminal.write("header-1\nheader-2\n")
    progress = Progress(
        ["alpha", "beta", "gamma"], stream=terminal, min_interval=0
    )

    progress.start(0)
    progress.page(0, 1, 2)
    progress.start(1)
    progress.finish(0, "done")
    progress.page(1, 1, 1)
    progress.start(2)
    progress.file_progress(2, 50, 100)
    progress.finish(1, "done")
    progress.finish(2, "done")
    progress.close()

    screen = terminal.screen()
    assert screen[:2] == ["header-1", "header-2"]
    blocks = screen[2:]
    assert len(blocks) == 3
    for name in ("alpha", "beta", "gamma"):
        matches = [line for line in blocks if re.search(rf"\b{name}\b", line)]
        assert len(matches) == 1, (name, screen)
    assert all("100%" in line for line in blocks)


def test_updates_keep_three_rows_over_many_frames():
    terminal = Terminal()
    terminal.write("header-1\nheader-2\n")
    progress = Progress(["alpha", "beta"], stream=terminal, min_interval=0)
    progress.start(0)
    progress.start(1)

    for step in range(1, 11):
        progress.page(0, step, 10)
        progress.file_progress(1, step * 10, 100)

    progress.finish(0, "done")
    progress.finish(1, "done")
    progress.close()

    screen = terminal.screen()
    assert screen[:2] == ["header-1", "header-2"]
    blocks = screen[2:]
    assert len(blocks) == 2
    assert blocks[0].count("alpha") == 1
    assert blocks[1].count("beta") == 1
