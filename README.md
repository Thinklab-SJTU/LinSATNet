# LinSATNet
This is the official implementation of our ICML 2023 paper "LinSATNet: The Positive Linear Satisfiability Neural Networks".

* [[paper]](https://runzhong.wang/files/icml2023_LinSATNet.pdf)

With LinSATNet, you can enforce the satisfiability of general **positive linear constraints** to the output of neural networks.

![usecase](figures/usecase.png)

The LinSAT layer is fully differentiable, and the gradients are exactly computed. Our implementation now supports PyTorch.

You can install it by

```shell
pip install linsatnet
```

And get started by

```python
from LinSATNet import linsat_layer
```

## A Quick Example

There is a quick example if you run ``LinSATNet/linsat.py`` directly. In this
example, the doubly-stochastic constraint is enforced for 3x3 variables.

To run the example, first clone the repo:
```shell
git clone https://github.com/Thinklab-SJTU/LinSATNet.git
```

Go into the repo, and run the example code:
```shell
cd LinSATNet
python LinSATNet/linsat.py
```

In this example, we try to enforce doubly-stochastic constraint to a 3x3 matrix.
The doubly-stochastic constraint means that all rows and columns of the matrix
should sum to 1.

The 3x3 matrix is flattened into a vector, and the following positive
linear constraints are considered (for Ex=f):
```python
E = torch.tensor(
    [[1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1],
     [1, 0, 0, 1, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 1, 0, 0, 1]], dtype=torch.float32
)
f = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)
```

We randomly init ``w`` and regard it as the output of some neural networks:
```python
w = torch.rand(9) # w could be the output of neural network
w = w.requires_grad_(True)
```

We also have a "ground-truth target" for the output of ``linsat_layer``, which
is an orthogonal matrix in this example:
```python
x_gt = torch.tensor(
    [1, 0, 0,
     0, 1, 0,
     0, 0, 1], dtype=torch.float32
)
```

The forward/backward passes of LinSAT follow the standard PyTorch style and are
readily integrated into existing deep learning pipelines.

The forward pass:
```python
linsat_outp = linsat_layer(w, E=E, f=f, tau=0.1, max_iter=10, dummy_val=0)
```

The backward pass:
```python
loss = ((linsat_outp - x_gt) ** 2).sum()
loss.backward()
```

We can also do gradient-based optimization over ``w`` to make the output of
``linsat_layer`` closer to ``x_gt``. This is what's happening when you train a
neural network.
```python
niters = 10
opt = torch.optim.SGD([w], lr=0.1, momentum=0.9)
for i in range(niters):
    x = linsat_layer(w, E=E, f=f, tau=0.1, max_iter=10, dummy_val=0)
    cv = torch.matmul(E, x.t()).t() - f.unsqueeze(0)
    loss = ((x - x_gt) ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'{i}/{niters}\n'
          f'  underlying obj={torch.sum(w * x)},\n'
          f'  loss={loss},\n'
          f'  sum(constraint violation)={torch.sum(cv[cv > 0])},\n'
          f'  x={x},\n'
          f'  constraint violation={cv}')
```
And you are likely to see the loss decreasing during the gradient steps.

## API Reference

To use LinSATNet in your own project, make sure you have the package installed:
```shell
pip install linsatnet
```
and import the pacakge at the beginning of your code:
```python
from LinSATNet import linsat_layer
```

### The ``linsat_layer`` function
> linsat_layer(x, A=None, b=None, C=None, d=None, E=None, f=None, tau=0.05, max_iter=100, dummy_val=0) [[source]](https://github.com/Thinklab-SJTU/LinSATNet/blob/main/LinSATNet/linsat.py#L11)

LinSAT layer enforces positive linear constraints to the input ``x`` and
projects it with the constraints
$$\mathbf{A} \mathbf{x} <= \mathbf{b}, \mathbf{C} \mathbf{x} >= \mathbf{d}, \mathbf{E} \mathbf{x} = \mathbf{f}$$
and all elements in $\mathbf{A}, \mathbf{b}, \mathbf{C}, \mathbf{d}, \mathbf{E}, \mathbf{f}$ must be non-negative.

**Parameters:**
* ``x``: PyTorch tensor of size ($n_v$), it can optionally have a batch size ($b \times n_v$)
* ``A``, ``C``, ``E``: PyTorch tensor of size ($n_c \times n_v$), constraint matrix on the left hand side
* ``b``, ``d``, ``f``: PyTorch tensor of size ($n_c$), constraint vector on the right hand side
* ``tau``: (``default=0.05``) parameter to control the discreteness of the projection. Smaller value leads to more discrete (harder) results, larger value leads to more continuous (softer) results.
* ``max_iter``: (``default=100``) max number of iterations
* ``dummy_val``: (``default=0``) the value of dummy variables appended to the input vector

**return:** PyTorch tensor of size ($n_v$) or ($b \times n_v$), the projected variables

Notations:
* $b$ means the batch size.
* $n_c$ means the number of constraints ($\mathbf{A}$, $\mathbf{C}$, $\mathbf{E}$ may have different $`n_c`$s)
* $n_v$ means the number of variables

### Some practical notes

1. You must ensure that your input constraints have a non-empty feasible space.
Otherwise, ``linsat_layer`` will not converge.
2. You may tune the value of ``tau`` for your specific tasks. Reasonable choices
of ``tau`` may range from ``1e-4`` to ``100`` in our experience.
3. Be careful of potential numerical issues. Sometimes ``A x <= 1`` does not
work, but ``A x <= 0.999`` works.
4. The input vector ``x`` may have a batch dimension, but the constraints can
not have a batch dimension. The constraints should be consistent for all data in
one batch.

## How it works?

Here we introduce the mechanism inside LinSAT. It works by extending the
Sinkhorn algorithm to multiple sets of marginals (to our best knowledge, we are
the first to study Sinkhorn with multi-sets of marginals). The positive linear
constraints are then enforced by transforming the constraints into marginals.
For more details and formal proofs, please refer to
[our paper](https://runzhong.wang/files/icml2023_LinSATNet.pdf).

### Classic Sinkhorn with single-set marginals

Let's start with the classic Sinkhorn algorithm. Given non-negative score matrix
$`\mathbf{S}\in\mathbb{R}_{\geq 0}^{m\times n}`$ and a set of marginal
distributions on rows $`\mathbf{v}\in \mathbb{R}_{\geq 0}^m`$ and columns
$`\mathbf{u} \in \mathbb{R}_{\geq 0}^n`$ (where
$`\sum_{i=1}^m v_i = \sum_{j=1}^n u_j = h`$),
the Sinkhorn algorithm outputs a normalized matrix
$`\mathbf{\Gamma}\in[0,1]^{m\times n}`$ so that
$`\sum_{i=1}^m \Gamma_{i,j}u_{j}=u_j, \sum_{j=1}^n \Gamma_{i,j}u_{j}=v_i`$.
Conceptually, $`\Gamma_{i,j}`$ means the **proportion** of $`u_j`$ moved to $`v_i`$.

The algorithm steps are:

Initialize $`\Gamma_{i,j}=\frac{s_{i,j}}{\sum_{i=1}^m s_{i,j}}`$

$`\quad`$**repeat**:

$`\qquad{\Gamma}_{i,j}^{\prime} = \frac{{\Gamma}_{i,j}v_{i}}{\sum_{j=1}^n {\Gamma}_{i,j}u_{j}}`$; $`\triangleright`$ normalize w.r.t. $`\mathbf{v}`$

$`\qquad{\Gamma}_{i,j} = \frac{{\Gamma}_{i,j}^{\prime}u_{j}}{\sum_{i=1}^m {\Gamma}_{i,j}^{\prime}u_{j}}`$; $`\triangleright`$ normalize w.r.t. $`\mathbf{u}`$

$`\quad`$**until** convergence.

> Note that the above formulation is modified from the conventional Sinkhorn
formulation. $`\Gamma_{i,j}u_j`$ is equivalent to the elements in the "transport"
matrix in papers such as [(Cuturi 2013)](https://arxiv.org/pdf/1306.0895v1.pdf).
We prefer this new formulation as it generalize smoothly to Sinkhorn with
multi-set marginals in the following.
>
> To make a clearer comparison, the transportation matrix in [(Cuturi 2013)](https://arxiv.org/pdf/1306.0895v1.pdf)
 is $`\mathbf{P}\in\mathbb{R}_{\geq 0}^{m\times n}`$, and the constraints are
    $$\sum_{i=1}^m P_{i,j}=u_{j},\quad \sum_{j=1}^n P_{i,j}=v_{i}$$
  $`P_{i,j}`$ means the _exact mass_ moved from $`u_{j}`$ to $`v_{i}`$.
>
>  The algorithm steps are:
>
>  Initialize $`\Gamma_{i,j}=\frac{s_{i,j}}{\sum_{i=1}^m s_{i,j}}`$
>
>  $`\quad`$**repeat**:
>
>  $`\qquad{P}_{i,j}^{\prime} = \frac{P_{i,j}v_{i}}{\sum_{j=1}^n {P}_{i,j}}`$; $`\triangleright`$ normalize w.r.t. $`\mathbf{v}`$
>
>  $`\qquad{P}_{i,j} = \frac{{P}_{i,j}^{\prime}u_j}{\sum_{i=1}^m {P}_{i,j}^{\prime}}`$; $`\triangleright`$ normalize w.r.t. $`\mathbf{u}`$
>
>  $`\quad`$**until** convergence.

### Extended Sinkhorn with multi-set marginals

We discover that the Sinkhorn algorithm can generalize to multiple sets of
marginals.

Recall that $`\Gamma_{i,j}\in[0,1]`$ means the proportion of $`u_i`$ moved to
$`v_j`$. Interestingly, it yields the same formulation if we simply replace
$`\mathbf{u},\mathbf{v}`$ by another set of marginal distributions, suggesting
the potential of extending the Sinkhorn algorithm to multiple sets of marginal
distributions. Denote that there are $k$ sets of marginal distributions that are
jointly enforced to fit more complicated real-world scenarios. The sets of
marginal distributions are
$`\mathbf{u}_\eta\in \mathbb{R}_{\geq 0}^n, \mathbf{v}_\eta\in \mathbb{R}_{\geq 0}^m`$,
and we have:
$$\forall \eta\in \{1, \cdots,k\}: \sum_{i=1}^m v_{\eta,i}=\sum_{j=1}^n u_{\eta,j}=h_\eta.$$

It assumes the existence of a normalized $`\mathbf{Z} \in [0,1]^{m\times n}`$ s.t.
$$\forall \eta\in \{1,\cdots, k\}: \sum_{i=1}^m z_{i,j} u_{\eta,j}=u_{\eta,j}, \sum_{j=1}^n z_{i,j} u_{\eta,j}=v_{\eta,i}$$
i.e., the multiple sets of marginal distributions have a non-empty feasible
region (you may understand the meaning of "non-empty feasible region" after
reading the next section about how to handle positive linear constraints).
Multiple sets of marginal distributions could be jointly enforced by traversing
the Sinkhorn iterations over $k$ sets of marginal distributions.

The algorithm steps are:

Initialize $`\Gamma_{i,j}=\frac{s_{i,j}}{\sum_{i=1}^m s_{i,j}}`$

$`\quad`$**repeat**:

$`\qquad`$**for**$`\eta=1`$**to**$k$**do**

$`\quad\qquad{\Gamma}_{i,j}^{\prime} = \frac{{\Gamma}_{i,j}v_{\eta,i}}{\sum_{j=1}^n {\Gamma}_{i,j}u_{\eta,j}}`$; $`\triangleright`$ normalize w.r.t. $`\mathbf{v}_\eta`$

$`\quad\qquad{\Gamma}_{i,j} = \frac{{\Gamma}_{i,j}^{\prime}u_{\eta,j}}{\sum_{i=1}^m {\Gamma}_{i,j}^{\prime}u_{\eta,j}}`$; $`\triangleright`$ normalize w.r.t. $`\mathbf{u}_\eta`$

$`\qquad`$**end for**

$`\quad`$**until** convergence.

In [our paper](https://runzhong.wang/files/icml2023_LinSATNet.pdf), we prove
that the Sinkhorn algorithm for multi-set marginals shares the same convergence
pattern with the classic Sinkhorn, and its underlying formulation is also
similar to the classic Sinkhorn.

### Transforming positive linear constraints into marginals

Then we show how to transform the positive linear constraints into marginals,
which are handled by our proposed multi-set Sinkhorn.

#### Encoding neural network's output
For an $l$-length vector denoted as $`\mathbf{y}`$ (which can be the output of a
neural network, also it is the input to ``linsat_layer``), the following matrix
is built

$`\mathbf{W} = {y}_1 \quad {y}_2 \quad ... \quad {y}_l \quad \beta`$

$`\qquad \ \ \beta \ \quad \beta \ \quad ... \quad \ \beta \quad \ \beta`$

where $`\mathbf{W}`$ is of size $`2 \times (l+1)`$, and $`\beta`$ is the dummy
variable, the default is $`\beta=0`$. $`\mathbf{y}`$ is put at the upper-left
region of $`\mathbf{W}`$. The entropic regularizer is then enforced to control
discreteness and handle potential negative inputs:
$$\mathbf{S} = \exp \left(\frac{\mathbf{W}}{\tau}\right).$$

The score matrix $`\mathbf{S}`$ is taken as the input of Sinkhorn for multi-set
marginals.

#### From linear constraints to marginals

* **Packing constraint** $`\mathbf{A}\mathbf{x}\leq \mathbf{b}`$. Assuming that
  there is only one constraint, we rewrite the constraint as
  $`\sum_{i=1}^l a_ix_i \leq b`$. The marginal distributions are defined as

  $`\mathbf{u}_p = \underbrace{\left[a_1 \quad a_2 \quad ...\quad a_l \quad b\right]}_{l \text{ dims}+1 \text{ dummy dim}}`$

  $`\mathbf{v}_p^\top = [b \quad \sum_{i=1}^l a_i]`$

  Following the "transportation" view of Sinkhorn, the output $`\mathbf{x}`$
  _moves_ at most $`b`$ unit of mass from $`a_1, a_2, \cdots, a_l`$, and the
  dummy dimension allows the inequality by _moving_ mass from the dummy
  dimension. It is also ensured that the sum of $`\mathbf{u}_p`$ equals the sum
  of $`\mathbf{v}_p`$.

* **Covering constraint** $`\mathbf{C}\mathbf{x}\geq \mathbf{d}`$. Assuming that
  there is only one constraint, we rewrite the constraint as
  $`\sum_{i=1}^l c_ix_i\geq d`$. The marginal distributions are defined as

  $`\mathbf{u}_c = \underbrace{\left[c_1 \quad c_2 \quad ...\quad c_l \quad \gamma d\right]}_{l \text{ dims} + 1 \text{ dummy dim}}`$

  $`\mathbf{v}_c^\top = \left[ (\gamma+1) d  \quad \sum_{i=1}^l c_i - d \right]`$

  where the multiplier $`\gamma=\left\lfloor\sum_{i=1}^lc_i / d \right\rfloor`$
  is necessary because we always have $`\sum_{i=1}^l c_i \geq d`$ (else the
  constraint is infeasible), and we cannot reach the feasible solution where all
  elements in $`\mathbf{x}`$ are 1s without this multiplier. This formulation
  ensures that at least $`d`$ unit of mass is _moved_ from $`c_1, c_2, \cdots, c_l`$
  by $`\mathbf{x}`$, thus representing the covering constraint of "greater than".
  It is also ensured that the sum of $`\mathbf{u}_c`$ equals the sum of
  $`\mathbf{v}_c`$.

* **Equality constraint** $`\mathbf{E}\mathbf{x}= \mathbf{f}`$. Representing the
  equality constraint is more straightforward. Assuming that there is only one
  constraint, we rewrite the constraint as $`\sum_{i=1}^l e_ix_i= f`$. The
  marginal distributions are defined as

  $`\mathbf{u}_e = \underbrace{\left[e_1 \quad e_2 \quad ...\quad e_l \quad 0\right]}_{l \text{ dims} + \text{dummy dim}=0}`$

  $`\mathbf{v}_e^\top = \left[f \quad \sum_{i=1}^l e_i - f \right]`$

  where the output $`\mathbf{x}`$ _moves_ $`e_1, e_2, \cdots, e_l`$ to $`f`$,
  and we need no dummy element in $`\mathbf{u}_e`$ because it is an equality
  constraint. It is also ensured that the sum of $`\mathbf{u}_e`$ equals the
  sum of $`\mathbf{v}_e`$.

After encoding all constraints and stack them as multiple sets of marginals,
we can call the Sinkhorn algorithm for multi-set marginals to enforce the
constraints.

## More Complicated Use Cases (appeared in our paper)

To be updated soon.

## Citation
If you find our paper/code useful in your research, please cite
```
@inproceedings{WangICML23,
  title={LinSATNet: The Positive Linear Satisfiability Neural Networks},
  author={Wang, Runzhong and Zhang, Yunhao and Guo, Ziao and Chen, Tianyi and Yang, Xiaokang and Yan, Junchi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```
