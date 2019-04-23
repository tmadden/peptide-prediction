PACE
====

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

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Core

   getting-started

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Extensions

    sklearn
