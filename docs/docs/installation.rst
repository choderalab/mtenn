Installation
============

``mtenn`` is on ``conda-forge``!

To install ``mtenn`` and all of its dependencies, you can simply run::

    mamba install -c conda-forge mtenn

If you want to use a version on GitHub that hasn't made it to ``conda-forge`` yet, first make sure you have all the required dependencies and then install with ``pip``::

    mamba install --only-deps -c conda-forge mtenn
    git clone git@github.com:choderalab/mtenn.git
    pip install -e ./mtenn/

Note that the ``-e`` flag can be omitted if you don't intend to do any development on the package.
