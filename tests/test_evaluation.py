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


def test_partitioning():
    from pace.evaluation import partition_samples
    samples = [
        pace.Sample(allele="A", peptide="AZ"),
        pace.Sample(allele="B", peptide="UV"),
        pace.Sample(allele="C", peptide="XY"),
        pace.Sample(allele="A", peptide="OP"),
        pace.Sample(allele="A", peptide="E"),
        pace.Sample(allele="A", peptide="F")
    ]
    assert partition_samples(samples) == {
        ("A", 1): [
            pace.Sample(allele="A", peptide="E"),
            pace.Sample(allele="A", peptide="F")
        ],
        ("A", 2): [
            pace.Sample(allele="A", peptide="AZ"),
            pace.Sample(allele="A", peptide="OP")
        ],
        ("B", 2): [pace.Sample(allele="B", peptide="UV")],
        ("C", 2): [pace.Sample(allele="C", peptide="XY")]
    }


def test_sample_filtering():
    from pace.evaluation import SampleFilter, matches_filter
    a = SampleFilter(alleles=None, lengths={1, 2})
    assert not matches_filter(a, "abc", 3)
    assert matches_filter(a, "abc", 2)
    assert matches_filter(a, "abc", 1)
    b = SampleFilter(alleles={"abc"}, lengths={1, 2})
    assert not matches_filter(b, "ab", 1)
    assert matches_filter(b, "abc", 1)


def test_stratified_split():
    from pace.evaluation import stratified_split, SampleFilter
    samples = [
        pace.Sample(allele="A", peptide="AZ"),
        pace.Sample(allele="A", peptide="UV"),
        pace.Sample(allele="B", peptide="XY"),
        pace.Sample(allele="B", peptide="OP"),
        pace.Sample(allele="B", peptide="E"),
        pace.Sample(allele="B", peptide="F")
    ]
    filter = SampleFilter(alleles=None, lengths={1, 2})
    assert list(stratified_split(samples, 2, filter, filter)) == \
        [([
            pace.Sample(allele="A", peptide="AZ"),
            pace.Sample(allele="B", peptide="XY"),
            pace.Sample(allele="B", peptide="E"),
        ], [
            pace.Sample(allele="A", peptide="UV"),
            pace.Sample(allele="B", peptide="OP"),
            pace.Sample(allele="B", peptide="F")
        ]),
        ([
            pace.Sample(allele="A", peptide="UV"),
            pace.Sample(allele="B", peptide="OP"),
            pace.Sample(allele="B", peptide="F")
        ], [
            pace.Sample(allele="A", peptide="AZ"),
            pace.Sample(allele="B", peptide="XY"),
            pace.Sample(allele="B", peptide="E"),
        ])]


def test_filtered_stratified_split():
    from pace.evaluation import stratified_split, SampleFilter
    samples = [
        pace.Sample(allele="A", peptide="AZ"),
        pace.Sample(allele="A", peptide="UV"),
        pace.Sample(allele="B", peptide="XY"),
        pace.Sample(allele="B", peptide="OP"),
        pace.Sample(allele="B", peptide="E"),
        pace.Sample(allele="B", peptide="F")
    ]
    training_filter = SampleFilter(alleles=None, lengths={2})
    test_filter = SampleFilter(alleles={"B"}, lengths={1, 2})
    assert list(stratified_split(samples, 2, training_filter, test_filter)) == \
        [([
            pace.Sample(allele="A", peptide="AZ"),
            pace.Sample(allele="A", peptide="UV"),
            pace.Sample(allele="B", peptide="XY")
        ], [
            pace.Sample(allele="B", peptide="OP"),
            pace.Sample(allele="B", peptide="E"),
            pace.Sample(allele="B", peptide="F")
        ]),
        ([
            pace.Sample(allele="A", peptide="AZ"),
            pace.Sample(allele="A", peptide="UV"),
            pace.Sample(allele="B", peptide="OP")
        ], [
            pace.Sample(allele="B", peptide="XY"),
            pace.Sample(allele="B", peptide="E"),
            pace.Sample(allele="B", peptide="F")
        ])]


def test_nonbinder_generation():
    from pace.evaluation import generate_nonbinders
    samples = [
        pace.Sample(allele="A", peptide="AZ"),
        pace.Sample(allele="A", peptide="UV"),
        pace.Sample(allele="B", peptide="XY"),
        pace.Sample(allele="B", peptide="E"),
        pace.Sample(allele="B", peptide="F")
    ]
    decoy_peptides = {
        1: ["A", "B", "C", "D"],
        2: ["AB", "BC", "CD", "DE", "EF", "FG", "GH", "HI"]
    }
    decoy_iters = {
        length: iter(decoys)
        for length, decoys in decoy_peptides.items()
    }
    expected_nonbinders = [
        pace.Sample(allele="A", peptide="AB"),
        pace.Sample(allele="A", peptide="BC"),
        pace.Sample(allele="A", peptide="CD"),
        pace.Sample(allele="A", peptide="DE"),
        pace.Sample(allele="B", peptide="EF"),
        pace.Sample(allele="B", peptide="FG"),
        pace.Sample(allele="B", peptide="A"),
        pace.Sample(allele="B", peptide="B"),
        pace.Sample(allele="B", peptide="C"),
        pace.Sample(allele="B", peptide="D")
    ]
    assert generate_nonbinders(decoy_iters, samples, 2) == expected_nonbinders


def test_simple_evaluation():
    from pace.evaluation import evaluate

    nonbinders = [
        "GG", "HH", "I", "J", "K", "L", "MM", "NN", "O", "P", "Q", "R"
    ]

    class TestDataSet(pace.DataSet):
        def get_binders(self, length):
            binders = {
                2: [
                    pace.Sample(allele="A", peptide="AA"),
                    pace.Sample(allele="A", peptide="BB"),
                    pace.Sample(allele="A", peptide="CC"),
                    pace.Sample(allele="A", peptide="DD")
                ],
                1: [
                    pace.Sample(allele="A", peptide="E"),
                    pace.Sample(allele="A", peptide="F")
                ]
            }
            return binders[length]

        def get_nonbinders(self, length):
            def generator(length):
                while True:
                    yield ['Z'] * length

            return generator(length)

    # This algorithm thinks everything binds.
    class BlindlyOptimisticAlgorithm(pace.PredictionAlgorithm):
        def train(self, binders, nonbinders):
            pass

        def predict(self, samples):
            return [1] * len(samples)

    boa_scores = evaluate(
        BlindlyOptimisticAlgorithm,
        TestDataSet(),
        selected_lengths=[1, 2],
        nbr_test=1)
    assert numpy.mean(boa_scores['by_accuracy']) == 0.5

    # This algorithm thinks that any peptide starting with A-D binds.
    # Its accuracy depends on how we filter the samples.
    class LengthBasedAlgorithm(pace.PredictionAlgorithm):
        def train(self, binders, nonbinders):
            pass

        def predict(self, samples):
            return [1 if s.peptide[0] in set('ABCD') else 0 for s in samples]

    lba_scores = evaluate(
        LengthBasedAlgorithm,
        TestDataSet(),
        selected_lengths=[1, 2],
        nbr_test=1,
        folds=4)
    assert 0.8 <= numpy.mean(lba_scores['by_accuracy']) <= 0.875

    lba_scores = evaluate(
        LengthBasedAlgorithm,
        TestDataSet(),
        selected_lengths=[1, 2],
        test_lengths=[2],
        nbr_test=1,
        folds=4)
    assert numpy.mean(lba_scores['by_accuracy']) == 1

    lba_scores = evaluate(
        LengthBasedAlgorithm,
        TestDataSet(),
        selected_lengths=[1, 2],
        test_lengths=[1],
        nbr_test=1,
        folds=4)
    assert numpy.mean(lba_scores['by_accuracy']) == 0.5


def test_evaluation_splitting():
    # This tests that the evaluation algorithm isn't sending in samples that
    # were used to train.

    class TestAlgorithm(pace.PredictionAlgorithm):
        def train(self, binders, nonbinders):
            self.binders = set(binders)
            self.nonbinders = set(nonbinders)

        def predict(self, samples):
            for s in samples:
                assert s not in self.binders
                assert s not in self.nonbinders
            return [1] * len(samples)

    pace.evaluate(TestAlgorithm, pace.load_dataset(16), nbr_test=2)
