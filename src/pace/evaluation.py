import random
import numpy
from collections import defaultdict
import itertools
from pace.definitions import Sample, peptide_lengths
from typing import NamedTuple, Optional, Set


def score_by_top_predictions(truth, predictions, top_n=None):
    """
    Score a set of predictions by ranking them and determining what fraction of
    the top predictions are actually binders.

    :param truth: an array of numbers indicating the true binding scores - All
    entries are either 0 or 1.

    :param predictions: an array of numbers indicating the predicted binding
    scores - All entries are between 0 and 1.

    :param top_n: the number of 'top' predictions to consider in the score
    (e.g., If this is 20, then we only care that the algorithm's top 20
    predictions are actually binders.) - This should be no larger than the
    number of true binders. If omitted, all true binders are considered.

    :returns: a score between 0 and 1
    """
    top_n = top_n or truth.count(1)
    top_predictions = numpy.argsort(predictions)[-top_n:]
    return sum([truth[i] for i in top_predictions]) / top_n


def score_by_accuracy(truth, predictions, cutoff=0.5, binder_weight=0.5):
    """
    Score a set of predictions by their accuracy.

    :param cutoff: the value separating 'binders' predictions and 'nonbinder'
    predictions (defaults to 0.5) - Predictions are considered accurate if they
    land on the same side of the cutoff value as the truth.

    :param binder_weight: the fraction that the binder score contributes to the
    overall score - The prediction accuracy for binders and nonbinders is
    considered separately and then combined according to this weight.
    """
    correctness = [(t > cutoff) == (p > cutoff)
                   for (t, p) in zip(truth, predictions)]

    def score_for_truth(truth_value):
        filtered = [
            c for (c, t) in zip(correctness, truth) if t == truth_value
        ]
        return sum(1 for c in filtered if c) / len(filtered) if filtered else 0

    return score_for_truth(1) * binder_weight + score_for_truth(0) * (
        1 - binder_weight)


def partition_samples(samples):
    d = defaultdict(list)
    for s in samples:
        d[(s.allele, len(s.peptide))].append(s)
    return d


def split_array(array, total_splits, split_index):
    start = len(array) * split_index // total_splits
    end = len(array) * (split_index + 1) // total_splits
    return (array[start:end], array[:start] + array[end:])


class SampleFilter(NamedTuple):
    alleles: Optional[Set[str]]
    lengths: Set[int]


def matches_filter(filter, allele, length):
    return (not filter.alleles
            or allele in filter.alleles) and length in filter.lengths


def stratified_split(samples, total_splits, training_filter, test_filter):
    partitioned = partition_samples(samples)
    for split_index in range(total_splits):
        training_samples = list()
        test_samples = list()
        for (allele, length), bin in partitioned.items():
            in_training = matches_filter(training_filter, allele, length)
            in_test = matches_filter(test_filter, allele, length)
            if in_training:
                if in_test:
                    a, b = split_array(bin, total_splits, split_index)
                    training_samples.extend(a)
                    test_samples.extend(b)
                else:
                    training_samples.extend(bin)
            else:
                if in_test:
                    test_samples.extend(bin)
        yield (training_samples, test_samples)


def score(algorithm, binders, nonbinders, scorers):
    # Combine the samples and pair them up with their truth values.
    paired_samples = list(
        zip(binders + nonbinders, [1] * len(binders) + [0] * len(nonbinders)))
    # This shuffle may not strictly be necessary, but otherwise the algorithm
    # would receive samples in a predictable order (with binders followed by
    # nonbinders).
    random.shuffle(paired_samples)
    # Now extract the samples and truth values in the shuffled order.
    shuffled_samples = [sample for sample, score in paired_samples]
    truth = [score for sample, score in paired_samples]
    # Ask the algorithm for predictions and score them.
    predictions = algorithm.predict(shuffled_samples)
    return {label: f(truth, predictions) for label, f in scorers.items()}


def generate_nonbinders(decoy_peptides, binders, nonbinder_ratio):
    nonbinders = []
    for (allele, length), samples in partition_samples(binders).items():
        nonbinder_count = int(len(samples) * nonbinder_ratio)
        for _ in range(0, nonbinder_count):
            nonbinders.append(
                Sample(allele=allele, peptide=next(decoy_peptides[length])))
    return nonbinders


default_scorers = {
    'by_top_predictions': score_by_top_predictions,
    'by_accuracy': score_by_accuracy
}


def evaluate(algorithm_class,
             dataset,
             folds=5,
             selected_alleles=None,
             selected_lengths=None,
             nbr_train=1,
             test_alleles=None,
             test_lengths=None,
             nbr_test=10,
             scorers=default_scorers):

    if selected_alleles:
        selected_alleles = set(selected_alleles)
    if test_alleles:
        test_alleles = set(test_alleles)
    else:
        test_alleles = selected_alleles

    if selected_lengths:
        selected_lengths = set(selected_lengths)
    else:
        selected_lengths = set(peptide_lengths)
    if test_lengths:
        test_lengths = set(test_lengths)
    else:
        test_lengths = selected_lengths
    all_lengths = selected_lengths | test_lengths

    from itertools import chain
    binders = list(
        chain.from_iterable(
            (dataset.get_binders(length) for length in all_lengths)))

    decoy_peptides = {
        length: iter(dataset.get_nonbinders(length))
        for length in all_lengths
    }

    random.shuffle(binders)
    binder_split = stratified_split(
        binders, folds,
        SampleFilter(alleles=selected_alleles, lengths=selected_lengths),
        SampleFilter(alleles=test_alleles, lengths=test_lengths))

    scores = {label: [] for label in scorers}
    for training_binders, test_binders in binder_split:
        training_nonbinders = generate_nonbinders(decoy_peptides,
                                                  training_binders, nbr_train)

        test_nonbinders = generate_nonbinders(decoy_peptides, test_binders,
                                              nbr_test)

        # Create a fresh algorithm instance and train it.
        algorithm = algorithm_class()
        algorithm.train(training_binders, training_nonbinders)

        # Do the scoring and record the scores.
        new_scores = score(algorithm, test_binders, test_nonbinders, scorers)
        for label in scorers.keys():
            scores[label].append(new_scores[label])

    return scores
