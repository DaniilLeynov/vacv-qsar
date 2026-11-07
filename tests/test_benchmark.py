from src.benchmark import run


class FakeModel:
    def __init__(self, name, mse, submission_text):
        self.name = name
        self._mse = mse
        self._text = submission_text
        self.trained = False
        self.predicted = False

    def train(self):
        self.trained = True
        return 0

    def predict(self, out):
        self.predicted = True
        out.write_text(self._text, encoding="utf-8")
        return 0

    def metrics(self):
        return {"mse_si": self._mse, "rmse_si": self._mse**0.5, "mae_si": 1.0}


def test_run_selects_lowest_mse_and_writes_submission(tmp_path):
    good = FakeModel("good", 100.0, "index,SI VACV\n0,1\n")
    bad = FakeModel("bad", 900.0, "index,SI VACV\n0,2\n")
    submission = tmp_path / "submission.csv"

    winner = run([bad, good], submission=submission, submissions_dir=tmp_path)

    assert winner == "good"
    assert good.trained and bad.trained
    assert good.predicted and not bad.predicted
    assert submission.read_text(encoding="utf-8") == "index,SI VACV\n0,1\n"
    assert (tmp_path / "submission_good.csv").exists()


def test_run_raises_on_failed_training(tmp_path):
    class Broken(FakeModel):
        def train(self):
            return 1

    broken = Broken("broken", 1.0, "x")
    try:
        run([broken], submission=tmp_path / "s.csv", submissions_dir=tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
