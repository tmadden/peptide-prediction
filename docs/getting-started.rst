---------------
Getting Started
---------------

Requirements
============

Note that PUMA does *not* have to be run on an Astroid machine. It interacts with Astroid through the cloud, so it can run on any machine with the required software installed. (i.e., Windows, Linux and OS X should all work fine, and a relatively lightweight laptop should also be fine.)

Python
------

PUMA requires Python 3.5 or higher (with pip).

Docker
------

Additionally, PUMA relies on CRADLE, which acts as a local proxy for Thinknode. The recommended way to run CRADLE is via Docker, which means that you'll need Docker installed. See `here <https://docs.docker.com/install/>`_ for installation instructions, and note that Docker by default requires root access on some OSs.

Getting PUMA
============

PUMA can be installed via pip as follows.

::

   pip install git+https://github.com/mghro/puma.git

.. note:: This requires access to `our GitHub organization <https://docs.mghro.io/github>`_.

Generating a Token
==================

Before you can interact with Astroid, you need to generate an authentication token:

::

   puma token gen

This will prompt you to log in to your Astroid account and, if successful, will store the resulting token on your computer.

Running CRADLE
==============

Some PUMA commands require CRADLE to be running in the background. If you have Docker running on your machine, CRADLE can be started with the following command.

::

   puma cradle start

PUMA also provides commands for stopping CRADLE and checking its status and logs. See ``puma cradle --help`` for details.

Trying a Useful Command
=======================

Now that PUMA is (hopefully) working, you can try pulling some actual data from Astroid:

::

   puma patient list --realm=sandbox

To see what other commands exist, see the :ref:`Command-Line Reference`, or just try ``puma --help``.

Using PUMA from Python
======================

All PUMA commands also exist as functions in the 'puma' Python package. For example, the command above could be run in a Python script as follows.

::

    import puma.patient
    patients = puma.patient.list(realm='sandbox')
    for mrn, name in patients.items():
        print(mrn + ': ' + name)

The Python API also includes some functions that aren't directly available through the command-line interface. For full details, see the :ref:`Python API Reference`.
