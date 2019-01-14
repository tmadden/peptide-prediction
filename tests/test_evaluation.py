from pace.evaluation import score_by_accuracy, score_by_top_predictions


def test_ranking_score():
    t = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    p = [0.7, 0.4, 0.3, 0.6, 0.8, 0.9, 0.4, 1.0]
    assert score_by_top_predictions(t, p) == 0.5
    assert score_by_top_predictions(t, p, top_n=4) == 0.5
    assert score_by_top_predictions(t, p, top_n=3) == 1 / 3
    assert score_by_top_predictions(t, p, top_n=2) == 0.5
    assert score_by_top_predictions(t, p, top_n=1) == 1


def test_accuracy_score():
    t = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    p = [0.6, 0.3, 0.3, 0.4, 0.0, 0.0, 0.4, 0.7]
    assert score_by_accuracy(t, p) == 0.75
    assert score_by_accuracy(t, p, cutoff=0.65) == 0.625
    assert score_by_accuracy(t, p, cutoff=0.35) == 1
    assert score_by_accuracy(t, p, binder_weight=1) == 0.5
    assert score_by_accuracy(t, p, binder_weight=0) == 1
