# Cibin: Tool for Final Project - Antibody Cocktail Efficacy

[Binder]

This repository contains a small Python package created for STAT159 final project. This small package will implement “Method 3” of [Li and Ding paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.6764) in Python to find 2-sided 1−𝛼 confidence bounds for the average treatment effect in a randomized experiment with binary outcomes and two treatments (active treatment and control).

It has a single source directory (`cabin`) with an `__init__.py` file and one "implementation" file (`cibin.py`) as well as a few tests in `cabin/tests`.

A top-level notebook called `cibin-demo.ipynb` that reproduces, using our implementation, column “3” of table I in the [Li and Ding paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.6764).

There are also four `analysis-*.ipynb` files and one pdf file `analysis-I-IV.pdf`, containing analysis related to data from the [Regeneron study](https://investor.regeneron.com/news-releases/news-release-details/phase-3-prevention-trial-showed-81-reduced-risk-symptomatic-sars). `analysis-I.ipynb`, `analysis-II.ipynb` , `analysis-III.ipynb` and `analysis-1V.ipynb` conducted analysis listed in the [finial project requirments](https://ucb-stat-159-s21.github.io/site/Hw/hw08-final-project.html). `analysis-I-IV.pdf` complied all four part of analysis together into a same pdf file.

And `utils.py`, it include implementation of sterne method which is used in analysis-I for underlying hypergeometric confidence intervals.

In addition to this `README.md` it includes some basic infrastructure: `LICENSE`, `requirements.txt`, and `setup.py` files.


## Installation

This project can currently only be installed from source, via

```
pip install .
```

or for a development installation via


```
pip install -e .
```

## Tests

You can run the project test suite via

```
pytest cibin
```

## License

This project is released under the terms of the BSD 3-clause License.
