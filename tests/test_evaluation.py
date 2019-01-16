import numpy
import pace


def test_ranking_score():
    from pace.evaluation import score_by_top_predictions
    t = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    p = [0.7, 0.4, 0.3, 0.6, 0.8, 0.9, 0.4, 1.0]
    assert score_by_top_predictions(t, p) == 0.5
    assert score_by_top_predictions(t, p, top_n=4) == 0.5
    assert score_by_top_predictions(t, p, top_n=3) == 1 / 3
    assert score_by_top_predictions(t, p, top_n=2) == 0.5
    assert score_by_top_predictions(t, p, top_n=1) == 1


def test_accuracy_score():
    from pace.evaluation import score_by_accuracy
    t = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    p = [0.6, 0.3, 0.3, 0.4, 0.0, 0.0, 0.4, 0.7]
    assert score_by_accuracy(t, p) == 0.75
    assert score_by_accuracy(t, p, cutoff=0.65) == 0.625
    assert score_by_accuracy(t, p, cutoff=0.35) == 1
    assert score_by_accuracy(t, p, binder_weight=1) == 0.5
    assert score_by_accuracy(t, p, binder_weight=0) == 1


def test_split_array():
    from pace.evaluation import split_array
    array = [1, 0, 2, 0, 3, 4, 0, 5, 0, 6]
    assert split_array(array, 4, 0) == ([1, 0], [2, 0, 3, 4, 0, 5, 0, 6])
    assert split_array(array, 4, 1) == ([2, 0, 3], [1, 0, 4, 0, 5, 0, 6])
    assert split_array(array, 4, 2) == ([4, 0], [1, 0, 2, 0, 3, 5, 0, 6])
    assert split_array(array, 4, 3) == ([5, 0, 6], [1, 0, 2, 0, 3, 4, 0])


def test_evaluation():
    from pace.evaluation import evaluate

    binders = [
        pace.Sample(allele="A", peptide="AA"),
        pace.Sample(allele="A", peptide="BB"),
        pace.Sample(allele="A", peptide="CC"),
        pace.Sample(allele="A", peptide="DD"),
        pace.Sample(allele="A", peptide="E"),
        pace.Sample(allele="A", peptide="F"),
    ]

    nonbinders = [
        pace.Sample(allele="A", peptide="GG"),
        pace.Sample(allele="A", peptide="HH"),
        pace.Sample(allele="A", peptide="I"),
        pace.Sample(allele="A", peptide="J"),
        pace.Sample(allele="A", peptide="K"),
        pace.Sample(allele="A", peptide="L"),
    ]

    # This algorithm thinks everything binds.
    # It's right half the time.
    class BlindlyOptimisticAlgorithm(pace.PredictionAlgorithm):
        def train(self, hits, misses):
            pass

        def predict(self, samples):
            return [1] * len(samples)

    boa_scores = evaluate(BlindlyOptimisticAlgorithm, binders, nonbinders)
    assert numpy.mean(boa_scores['by_accuracy']) == 0.5

    # This algorithm thinks that any peptide longer than 1 character binds.
    # It's right two thirds of the time.
    class LengthBasedAlgorithm(pace.PredictionAlgorithm):
        def train(self, hits, misses):
            pass

        def predict(self, samples):
            return [1 if len(s.peptide) > 1 else 0 for s in samples]

    lba_scores = evaluate(LengthBasedAlgorithm, binders, nonbinders)
    assert numpy.mean(lba_scores['by_accuracy']) == 2 / 3
