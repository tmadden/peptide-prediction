import random
import numpy
from collections import defaultdict
import itertools
from pace.definitions import *
from pace.data import load_dataset
from typing import NamedTuple, Optional, Set

def score_by_ppv(truth, predictions, top_n=None):
    """
    Score a set of predictions by ranking them and determining what fraction of
    the top predictions are actually binders.

    Parameters
    ----------
    truth
        an array of numbers indicating the true binding scores - All entries are
        either 0 or 1.

    predictions
        an array of numbers indicating the predicted binding scores - All
        entries are between 0 and 1.

    top_n
        the number of 'top' predictions to consider in the score (e.g., If this
        is 20, then we only care that the algorithm's top 20 predictions are
        actually binders.) - This should be no larger than the number of true
        binders. If omitted, all true binders are considered.

    Returns
    -------
    float
        a score between 0 and 1
    """
    top_n = top_n or truth.count(1)
    top_predictions = numpy.argsort(predictions)[-top_n:]
    return sum([truth[i] for i in top_predictions]) / top_n


class PpvScorer(Scorer):
    def score(self, results):
        return score_by_ppv([r.truth for r in results],
                            [r.prediction for r in results])


def score_by_accuracy(truth, predictions, cutoff=0.5, binder_weight=0.5):
    """
    Score a set of predictions by their accuracy.

    Parameters
    ----------
    cutoff : float
        the value separating 'binders' predictions and 'nonbinder' predictions
        (defaults to 0.5) - Predictions are considered accurate if they land on
        the same side of the cutoff value as the truth.

    binder_weight : float
        the fraction that the binder score contributes to the overall score -
        The prediction accuracy for binders and nonbinders is considered
        separately and then combined according to this weight.

    Returns
    -------
    float
        a score between 0 and 1
    """
    correctness = [(t > cutoff) == (p > cutoff)
                   for (t, p) in zip(truth, predictions)]

    def score_for_truth(truth_value):
        filtered = [
            c for (c, t) in zip(correctness, truth) if t == truth_value
        ]
        return sum(1 for c in filtered if c) / len(filtered) if filtered else 0

    # compute and log the confusion matrix:
    binaryp = numpy.where(numpy.array(predictions)>cutoff, 1, 0)
    cm = sklearn.metrics.confusion_matrix(truth, binaryp)
    print(cm)

    # compute the ROC curve and save to a file, with some useful information.
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(truth, predictions, pos_label=1)
    timestr = time.strftime('%Y%m%d-%H%M%S')
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.title('ROC. Confusion matrix displayed for cutoff = '+str(cutoff))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.text(.6,.4,numpy.array2string(cm))
    # plt.show()
    plt.savefig('/home/dcraft/tplots/'+timestr+'_auc.png')
    plt.clf()

    return score_for_truth(1) * binder_weight + score_for_truth(0) * (
        1 - binder_weight)


class AccuracyScorer(Scorer):
    def __init__(self, cutoff=0.5, binder_weight=0.5):
        self.cutoff = cutoff
        self.binder_weight = binder_weight

    def score(self, results):
        return score_by_accuracy([r.truth for r in results],
                                 [r.prediction for r in results],
                                 cutoff=self.cutoff,
                                 binder_weight=self.binder_weight)


def partition_samples(samples):
    """
    Partition samples according to allele and peptide length.
    """
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
    # Ask the algorithm for predictions and score them.
    predictions = algorithm.predict(
        [sample for sample, score in paired_samples])
    # Construct the results.
    results = [
        PredictionResult(sample=sample, truth=score, prediction=prediction)
        for (sample, score), prediction in zip(paired_samples, predictions)
    ]
    # Invoke the scorers.
    return {label: s.score(results) for label, s in scorers.items()}


def generate_nonbinders(decoy_peptides, binders, nonbinder_ratio):
    nonbinders = []
    for (allele, length), samples in partition_samples(binders).items():
        nonbinder_count = int(len(samples) * nonbinder_ratio)
        for _ in range(0, nonbinder_count):
            nonbinders.append(
                Sample(allele=allele, peptide=next(decoy_peptides[length])))
    return nonbinders


default_scorers = {'ppv': PpvScorer(), 'accuracy': AccuracyScorer()}

def evaluate(algorithm_class,
             dataset=load_dataset(),
             folds=5,
             selected_alleles=None,
             selected_lengths=None,
             nbr_train=1,
             test_alleles=None,
             test_lengths=None,
             nbr_test=10,
             scorers=default_scorers):
    """
    Evaluate an algorithm.

    Given a dataset and an algorithm, this evaluates the algorithm by repeatedly
    splitting the dataset into training and testing subsets, training a new
    algorithm instance, asking it to make predictions about the testing subset,
    and scoring those predictions.

    Parameters
    ----------
    algorithm_class : pace.PredictionAlgorithm
        a function taking no arguments that returns a new instance of the
        algorithm to test - If the algorithm class has a default constructor,
        you can simply pass in the class itself. Otherwise, pass in a lambda
        that fills in the constructor arguments appropriately. The algorithm
        must implement the interface specified by
        :class:`pace.PredictionAlgorithm`.

    dataset : pace.Dataset
        the dataset to use for testing - If omitted, the builtin dataset is
        used. The dataset must implement the interface specified by
        :class:`pace.Dataset`.

    folds : int
        the number of folds (i.e., iterations) to perform (default is 5)

    selected_alleles : List[str], optional
        a list of alleles to use for training - If a value is given here, the
        dataset is filtered so that only samples for those alleles are used for
        training. (By default, no filtering is done.) Note that this will also
        determine the filtering of the test data unless a different filter is
        explicitly specified.

    selected_lengths : List[int], optional
        a list of peptide lengths to use for training - If a value is given
        here, the dataset is filtered so that only samples for those lengths
        are used for training. (By default, no filtering is done.) Note that
        this will also determine the filtering of the test data unless a
        different filter is explicitly specified.

    nbr_train : float, optional
        the nonbinder ratio for training - This determines the ratio of
        nonbinders to binders in the set of samples used for training the
        algorithm. It defaults to 1.

    test_alleles : List[str], optional
        a list of alleles to use for testing - This is equivalent to
        ``selected_allles`` but determines the filtering for the testing phase.
        By default, the same set that was used for training is also used for
        testing.

    test_lengths : List[int], optional
        a list of peptide lengths to use for testing - This is equivalent to
        ``selected_lengths`` but determines the filtering for the testing phase.
        By default, the same set that was used for training is also used for
        testing.

    nbr_test
        the nonbinder ratio for testing - This determines the ratio of
        nonbinders to binders in the set of samples used for testing the
        algorithm. It defaults to 10. (Using a value much higher than 10 with
        the default dataset (without subselecting) will exhaust the pool of
        nonbinders.)

    scorers : Dict[str,pace.Scorer]
        a map from labels to scorers - If omitted,
        ``pace.evaluation.default_scorers`` is used.

    Returns
    -------
    Dict[str,List[Any]]

    """

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
        # tom, maybe this "create fresh instance" should go outside loop
        # things like building the architecture for a neural network only need to get done once...
        algorithm = algorithm_class()
        algorithm.train(training_binders, training_nonbinders)

        # Do the scoring and record the scores.
        new_scores = score(algorithm, test_binders, test_nonbinders, scorers)
        for label in scorers.keys():
            scores[label].append(new_scores[label])

    return scores
