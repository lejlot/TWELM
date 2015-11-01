# TWELM
Simple python implementation of **Weighted Tanimoto Extreme Learning Machines**

## What is TWELM?
Proposed model is a binary classifier belonging to the family of Randomized Neural Networks.
From technical perspective it is a 1-hidden layer neural network, which uses a generalized
Jaccard coefficient as an activation function

![f(x,w) = \frac{\langle x, w \rangle}{\|x\|_1 + \|w\|_1 - \langle x, w \rangle}](http://www.sciweavers.org/tex2img.php?eq=f%28x%2Cw%29%20%3D%20%5Cfrac%7B%5Clangle%20x%2C%20w%20%5Crangle%7D%7B%5C%7Cx%5C%7C_1%20%2B%20%5C%7Cw%5C%7C_1%20-%20%5Clangle%20x%2C%20w%20%5Crangle%7D&bc=White&fc=Black&im=jpg&fs=12&ff=concmath&edit=0)

where only output weights are trained
using L2 regularized least squares method. This can be seen as a variation of an old idea
of RBF networks, RVFL model or ELM approach. Whatever you call it, it is a suprisingly
simple and fast classifier which achieves a very good results in a particular types
of problems.

## When to use TWELM?
TWELM is quite specific model, so make sure that it is well suited for your problem,
by answering following questions:

* Is your data represented as sparse, binary vectors?
* It your problem a binary classification?
* Do you care about balanced accuracy (or GMean)?
* Do you need a fast, low-parametric model (possible at the cost of accuracy)?

If you answered **yes** for all the above - TWELM is for you, have fun!

## Citing
```
@article{czarnecki2015weighted,
    title={Weighted Tanimoto Extreme Learning Machine with Case Study in Drug Discovery},
    author={Czarnecki, Wojciech Marian},
    journal={Computational Intelligence Magazine, IEEE},
    volume={10},
    number={3},
    pages={19--29},
    year={2015},
    publisher={IEEE}
}
```
