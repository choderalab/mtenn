Installation
============

``mtenn`` is on ``conda-forge``!
You can build an environment with the required dependenices using the included YAML file::

    git clone git@github.com:choderalab/mtenn.git
    mamba env create -f mtenn/devtools/conda-envs/mtenn.yaml

To install, ``mtenn``, you can then simply run::

    mamba install -c conda-forge mtenn

If you want to use a version on GitHub that hasn't made it to ``conda-forge`` yet, first build the required environment and then install with ``pip``::

    git clone git@github.com:choderalab/mtenn.git
    mamba env create -f mtenn/devtools/conda-envs/mtenn.yaml
    conda activate mtenn
    pip install ./mtenn/
