PACE
====

.. image:: https://img.shields.io/travis/tmadden/pace/master.svg?style=flat&logo=travis
    :target: https://travis-ci.org/tmadden/pace

.. image:: https://img.shields.io/codecov/c/github/tmadden/pace/master.svg?style=flat
    :target: https://codecov.io/gh/tmadden/pace

|

Welcome to PACE, a framework for comparing machine learning algorithms for the
peptide-MHC binding problem. While many potential solutions to this problem have
been published, a shared dataset and evaluation protocol have been missing,
making it impossible to compare the various approaches. PACE (peptide algorithm
comparison environment) fills that void by supplying a 16 HLA allele dataset and
a structured evaluation protocol such that algorithms can be fairly compared.

As a secondary goal, PACE aims to provide useful utilities to facilitate the
development of prediction algorithms, such as characteristics for alleles,
utilities for implementing algorithms on common machine learning frameworks,
etc.

Contributions are welcome. We will publish contribution guidelines shortly...

Installation
------------

PACE requires Python 3.6 or higher. It can be installed using pip directly from
GitHub:

::

   pip install git+https://github.com/tmadden/pace.git#egg=pace

This will get you the core evaluation mechanics and datasets. If you want
optional components for interfacing with other libraries, you'll need to specify
that explicitly. For example:

::

   pip install git+https://github.com/tmadden/pace.git#egg=pace[sklearn]

Usage
-----

In PACE, you supply an algorithm that is capable of training on sample data and
making binding predictions for other data. PACE provides an example dataset and
the mechanics for evaluating your algorithm on that dataset.

A trivial example is below. This algorithm skips the training phase and simply
predicts that peptides always bind to alleles that start with the same letter.

::

    class FairlyPoorAlgorithm(pace.PredictionAlgorithm):
        def train(self, binders, nonbinders):
            pass

        def predict(self, samples):
            return [1 if s.allele[0] == s.peptide[0] else 0 for s in samples]

    # Evaluate our algorithm using PACE.
    scores = pace.evaluate(FairlyPoorAlgorithm, pace.load_dataset(16))
    pprint.pprint(scores)

This produces the following output:

::

    {'by_accuracy': [0.399618966977138,
                     0.4155843184198713,
                     0.4877250735697541,
                     0.5181480373085939,
                     0.5184315684315685],
    'by_top_predictions': [0.028789161727349702,
                           0.0321213028081201,
                           0.0708763529353085,
                           0.1221008529103696,
                           0.12342657342657343]}

This shows the scores over multiple evaluations (holding out different subsets
of the data samples) and according to multiple scoring metrics. As you can see,
our algorithm is only barely better than random guessing.

Additional documentation will come soon...
