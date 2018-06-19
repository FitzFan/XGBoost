# Hyperparameter Optimization With Hyperopt

## What is [Hyperopt](https://github.com/hyperopt/hyperopt)?

Hyperopt is a hyperparameter search package that implements various search algorithms for Sequetial model-based optimization (SMBO, aka Bayesian optimization) to find the best set of hyperparameters within a search space.
Citing the paper, SMBO has the following advantages:

    - can leverage smoothness w/out analytic gradient
    - handles continuous real-valued, discrete, and conditional variables/features
    - handles parallel evaluations of the scalar loss function $f(x)$,
    - copes with hundreds of variables, even with the budget of just a few hundred function evaluations.

Other advantages that can be summarized about Hyperopt are as follows:

### Persistence and Resilience
Previous hyperparameter searches can be saved and persisted in a MongoDB instance, which allows pausing/resuming hyperparameter search.
### Reproducibility:
Persisting experimental results using hyperopt allows other people to reproduce your work.

## How does Hyperopt work?
You'll need to specify the following:

    - an objective/loss function to minimize (loss function must return a scalar value)
    - the search space
    - a trials database (optional)
    - search algorithm to use (optional)

Let's break down the steps:

### 1. Define an objective function
In the simplest case, we can define an objective function that takes a single input $x$, and returns a scalar value which represents $loss(f(x))$.
For a trivial case, if we want to minimize the quadratic function $q(x,y):= x^2 + y^2$, we can define our objective as follows:

```Python
def q(args):
    x, y = args
    return x ** 2 + y ** 2
```

### 2. Define a configuration space
A configuration space specifies our search domain. For example, if we want to search for $x \in [0,1]$ and $y \in {0, 2}$, you could write the following:
```Python
from hyperopt import hp
space = [hp.uniform(’x’, 0, 1), hp.choice(’y’, 0, 2)]
```
In more detail, the configuration space is a joint probability distribution over the hyperparameters that we are trying to optimize. `hp.uniform` is a uniform distribution, while `hp.choice` makes a choice within a list of options. If you're interested in reading more about minimizing functions using hyperopt, read the following [link](https://github.com/hyperopt/hyperopt/wiki/FMin).

### 3. Choose a search algorithm

Hyperopt currently implements three search algorithms:

1. Random Search, or `hyperopt.hp.rand`, which more or less does the same thing as `sklearn`'s `RandomizedSearchCV`
2. Gaussian Processes, or `hyperopt.hp.gp`
3. Tree of Parzen Estimators, or `hyperopt.tpe`

*More information on each of the search algorithms can be found in [1].

Here's some sample code that will select the best hyperparams with `hyperopt`:

```Python
from hyperopt import hp, fmin, rand, tpe, space_eval
best = fmin(q, space, algo=rand.suggest)
print(best)
# => XXX
print(space_eval(space, best))
# => XXX
best = fmin(q, space, algo=tpe.suggest)
print(best)
# => XXX
print(space_eval(space, best))
# => XXX
```
(Bergestra 2013, Scipy 2013)

### 4. Persistence
If you wish to persist your search results or extend a search, read the following link:
https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB


## Installation:
If you use Python 3, please install Hyperopt from my fork (which fixes a fatal bug discussed in issue #234):
```bash
pip install git+git://github.com/wenyangfu/hyperopt.git
```
Otherwise, install it from the default branch:
```bash
pip install git+git://github.com/hyperopt/hyperopt.git
```

### Further examples
Source code examples for optimizing a XGBoost classifier can be found [here](hyperopt-xgb.py).

## Documentation
Documentation for Hyperopt can be found at the following locations:

1. https://github.com/hyperopt/hyperopt/wiki
2. http://hyperopt.github.io/hyperopt/

Much of the content in this post is derived from the following papers, and from the Hyperopt wiki.

[1] NIPS 2011: [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)

[2] SciPy 2013: [Hyperopt: A Python Library for Optimizing the
Hyperparameters of Machine Learning Algorithms](https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf)

[3] JMLR 2013: [Making a Science of Model Search: Hyperparameter Optimization
in Hundreds of Dimensions for Vision Architectures](http://jmlr.org/proceedings/papers/v28/bergstra13.pdf)

## Further reading and alternatives:
[Towards an Empirical Foundation for
Assessing Bayesian Optimization of Hyperparameters](http://www.cs.ubc.ca/~hutter/papers/13-BayesOpt_EmpiricalFoundation.pdf)

[Optimizing Hyperparams with Hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)

[scikit-optimize](https://scikit-optimize.github.io/): a fairly new framework, I'm not sure if this works well.

