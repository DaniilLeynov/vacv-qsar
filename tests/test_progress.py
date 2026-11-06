from src.fetch.cli import _progress_enabled, build_parser
from src.fetch.progress import Progress, human_size


class FakeStream:
    def __init__(self, tty=True):
        self.parts = []
        self._tty = tty

    def isatty(self):
        return self._tty

    def write(self, text):
        self.parts.append(text)

    def flush(self):
        pass

    @property
    def text(self):
        return "".join(self.parts)


def test_renders_one_bar_per_source():
    stream = FakeStream()
    progress = Progress(["chembl_vacv", "pubchem_example"], stream=stream, min_interval=0)
    progress.start(0)

    text = stream.text
    assert "chembl_vacv" in text
    assert "pubchem_example" in text
    assert text.count("░") >= 2
    progress.close()


def test_page_and_file_progress_update_fractions():
    stream = FakeStream()
    progress = Progress(["alpha", "beta"], stream=stream, min_interval=0)
    progress.start(0)
    progress.page(0, 2, 10)
    progress.file_progress(1, 50, 100)

    assert "20%" in stream.text
    assert "50%" in stream.text
    assert "50 B/100 B" in stream.text
    progress.close()


def test_finish_sets_complete():
    stream = FakeStream()
    progress = Progress(["alpha"], stream=stream, min_interval=0)
    progress.start(0)
    progress.finish(0, "done")
    assert "100%" in stream.text
    assert "done" in stream.text
    progress.close()


def test_disabled_progress_is_silent():
    stream = FakeStream()
    progress = Progress(["alpha"], stream=stream, enabled=False, min_interval=0)
    progress.start(0)
    progress.page(0, 1, 1)
    progress.file_progress(0, 10, 10)
    progress.finish(0, "done")
    progress.close()
    assert stream.text == ""


def test_ticker_keeps_display_alive(monkeypatch):
    import time

    monkeypatch.setattr("src.fetch.progress.TICK_INTERVAL", 0.02)
    stream = FakeStream()
    progress = Progress(["slow"], stream=stream, min_interval=0)
    progress.start(0)
    before = len(stream.text)
    time.sleep(0.12)
    progress.close()

    assert len(stream.text) > before
    assert "s" in stream.text


def test_tty_enables_progress_by_default():
    assert Progress(["a"], stream=FakeStream(tty=True)).enabled is True
    assert Progress(["a"], stream=FakeStream(tty=False)).enabled is False


def test_progress_flags():
    parser = build_parser()
    assert _progress_enabled(parser.parse_args(["--progress"])) is True
    assert _progress_enabled(parser.parse_args(["--no-progress"])) is False
    assert _progress_enabled(parser.parse_args([])) is None


def test_human_size():
    assert human_size(999) == "999 B"
    assert human_size(2048) == "2.0 KiB"
    assert human_size(1536 * 1024) == "1.5 MiB"


def test_retry_is_shown():
    stream = FakeStream()
    progress = Progress(["alpha"], stream=stream, min_interval=0)
    progress.start(0)
    progress.retry(0, 2, 503)
    assert "retry 2 (503)" in stream.text
    progress.close()
