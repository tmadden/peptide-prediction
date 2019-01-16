pace
====

.. image:: https://img.shields.io/travis/tmadden/pace.svg?style=flat&logo=travis
    :target: https://travis-ci.org/tmadden/pace

.. image:: https://img.shields.io/codecov/c/github/tmadden/pace/master.svg?style=flat
    :target: https://codecov.io/gh/tmadden/pace

|

Welcome to PACE, a framework for comparing machine learning algorithms for the peptide-MHC binding problem. While many methods have been published, a shared dataset and evaluation protocol have been missing, making it impossible to compare the various approaches. PACE (peptide algorithm comparison environment) fills that void by supplying a 95 HLA allele dataset and a structured comparison protocol such that algorithms can be fairly compared.

Users are encouraged to implement their own methods in the PACE framework.

Installation
------------

PACE requires Python 3.6 or higher. It can be installed using pip directly from GitHub:

::

   pip install git+https://github.com/tmadden/pace.git#egg=pace

This will get you the core evaluation mechanisms and datasets. If you want optional components for interfacing with other libraries, you'll need to specify that explicitly. For example:

::

   pip install git+https://github.com/tmadden/pace.git#egg=pace[sklearn]

Usage
-----

In PACE, you supply an algorithm that is capable of training on sample data and making binding predictions for other data. PACE provides an example dataset and the mechanics for evaluating your algorithm on that dataset.

A trivial example is below. This algorithm skips the training phase and simply predicts that peptides always bind to alleles that start with the same letter.

::

    class FairlyPoorAlgorithm(pace.PredictionAlgorithm):
        def train(self, binders, nonbinders):
            pass

        def predict(self, samples):
            return [1 if s.allele[0] == s.peptide[0] else 0 for s in samples]


    scores = pace.evaluate(PureGuessingAlgorithm,
                           **pace.load_data_set(16, nonbinder_fraction=0.9))
    pprint.pprint(scores)

This produces the following output:

::

    {'by_accuracy': [0.5191357196142946,
                    0.5193158082507424,
                    0.5188450703732488,
                    0.5187279062981613,
                    0.5192999648360439,
                    0.5184246426436073],
    'by_top_predictions': [0.13100311228154177,
                            0.1320628232139437,
                            0.13215859030837004,
                            0.1350316031411607,
                            0.13340356253591265,
                            0.129716529400498]}

This shows the scores over multiple evaluations (holding out different subsets of the data samples) and according to multiple scoring metrics. As you can see, our algorithm is only barely better than random guessing.

Additional documentation will come soon...

