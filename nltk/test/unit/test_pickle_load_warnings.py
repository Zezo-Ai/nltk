# Test pickle_load warnings

import pickle
from pathlib import Path

import pytest

from nltk.parse.chart import Chart

WARN_RE = r"Security warning: loading pickles can execute arbitrary code"


def test_chartparser_app_warns_on_unpickle(tmp_path: Path):
    # Arrange: create a pickled Chart object (ChartComparer.load_chart expects this)
    pkl = tmp_path / "chart.pickle"
    chart = Chart(["a", "b"])
    with pkl.open("wb") as f:
        pickle.dump(chart, f)

    from nltk.app.chartparser_app import ChartComparer

    # Act + Assert: warning emitted when unpickling
    with pytest.warns(RuntimeWarning, match=WARN_RE):
        comparer = ChartComparer()
        comparer.load_chart(str(pkl))


def test_transitionparser_warns_on_model_unpickle(tmp_path: Path):
    # transitionparser optionally depends on numpy/scipy/sklearn; skip if missing
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")

    from nltk.parse import DependencyGraph
    from nltk.parse.transitionparser import TransitionParser

    model_path = tmp_path / "tp.model"

    # Minimal projective example (adapted from transitionparser.py doctest)
    gold_sent = DependencyGraph(
        """
Economic  JJ     2      ATT
news  NN     3       SBJ
has       VBD       0       ROOT
little      JJ      5       ATT
effect   NN     3       OBJ
on     IN      5       ATT
financial       JJ       8       ATT
markets    NNS      6       PC
.    .      3       PU
"""
    )

    # Train quickly to produce a real pickled sklearn model file
    parser = TransitionParser(TransitionParser.ARC_STANDARD)
    parser.train([gold_sent], str(model_path), verbose=False)

    # Act + Assert: warning emitted when unpickling model in parse()
    with pytest.warns(RuntimeWarning, match=WARN_RE):
        result = parser.parse([gold_sent], str(model_path))

    assert len(result) == 1
