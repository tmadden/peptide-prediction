import random
import numpy

from pace.utilities import split_array


def score_by_top_predictions(truth, predictions, top_n=None):
    """
    Score a set of predictions by ranking them and determining what fraction of the top predictions are actually binders.

    :param truth: an array of numbers indicating the true binding scores - All entries are either 0 or 1.

    :param predictions: an array of numbers indicating the predicted binding scores - All entries are between 0 and 1.

    :param top_n: the number of 'top' predictions to consider in the score (e.g., If this is 20, then we only care that the algorithm's top 20 predictions are actually binders.) - This should be no larger than the number of true binders. If omitted, all true binders are considered.

    :returns: a score between 0 and 1
    """
    top_n = top_n or truth.count(1)
    top_predictions = numpy.argsort(predictions)[-top_n:]
    return sum([truth[i] for i in top_predictions]) / top_n


def score_by_accuracy(truth, predictions, cutoff=0.5, binder_weight=0.5):
    """
    Score a set of predictions by their accuracy.

    :param cutoff: the value separating 'binders' predictions and 'nonbinder' predictions (defaults to 0.5) - Predictions are considered accurate if they land on the same side of the cutoff value as the truth.

    :param binder_weight: the fraction that the binder score contributes to the overall score - The prediction accuracy for binders and nonbinders is considered separately and then combined according to this weight.
    """
    correctness = [(t > cutoff) == (p > cutoff) for (t, p) in zip(truth, predictions)]

    def score_for_truth(truth_value):
        filtered = [c for (c, t) in zip(correctness, truth) if t == truth_value]
        return sum(1 for c in filtered if c) / len(filtered)

    return score_for_truth(1) * binder_weight + score_for_truth(0) * (1 - binder_weight)


def score(algorithm, binders, nonbinders):
    # Combine the samples and pair them up with their truth values.
    paired_samples = list(
        zip(binders + nonbinders, [1] * len(binders) + [0] * len(nonbinders))
    )
    # This shuffle may not strictly be necessary, but otherwise the algorithm
    # would receive samples in a predictable order (with binders followed by
    # nonbinders).
    random.shuffle(paired_samples)
    # Now extract the samples and truth values in the shuffled order.
    shuffled_samples = [sample for sample, score in paired_samples]
    truth = [score for sample, score in paired_samples]
    # Ask the algorithm for predictions and score them.
    predictions = algorithm.eval(shuffled_samples)
    return score_by_accuracy(truth, predictions)


def evaluate(algorithm_class, binders, nonbinders, splits=6):
    random.shuffle(binders)
    random.shuffle(nonbinders)

    scores = []
    for i in range(splits):
        training_binders, eval_binders = split_array(binders, splits, i)
        training_nonbinders, eval_nonbinders = split_array(nonbinders, splits, i)

        algorithm = algorithm_class()
        algorithm.train(training_binders, training_nonbinders)

        scores.append(score(algorithm, eval_binders, eval_nonbinders))
    return sum(scores) / len(scores)
