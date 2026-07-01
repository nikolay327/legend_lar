# legend_lar

Research code for conditional neural ratio estimation on sparse LAr/HPGe-style detector observables.

The package implements a contrastive neural ratio estimation pipeline for structured detector data represented as variable-length sequences of detector hits and auxiliary event-level features. The code is written in PyTorch and is organized around multimodal encoders, k-fold ensemble training, deep-supervision variants, and empirical calibration/inference utilities.

This repository contains code only. It does not include collaboration data, detector metadata, processed datasets, trained model checkpoints, or experiment-specific configuration files.

## Scope

The main components are:

* multimodal LAr and HPGe encoders for sparse detector observables;
* geometry-aware tokenization of detector-channel coordinates;
* transformer blocks for packed variable-length sequences;
* contrastive neural ratio estimation losses;
* k-fold and bootstrap-based ensemble training;
* HPGe-prefix deep supervision;
* ensemble-based inference and calibration utilities.

The package is research code and is not a standalone reproducible analysis. External data products and configuration files are required to run the full training and inference workflow.

## Method overview

The model is designed to learn conditional neural ratio scores for structured detector observables, rather than ordinary classifier probabilities.

Let $x$ denote LAr-side observables and let $c$ denote HPGe-side conditioning information. The intended score is a conditional log-ratio of the form

$$
s_\theta(x,c)
\approx
\log r(x,c)
=
\log \frac{p_{\mathrm{TC}}(x \mid c)}{p_{\mathrm{RC}}(x)} ,
$$

where $p_{\mathrm{TC}}(x \mid c)$ denotes the conditional distribution of true-coincident LAr observables given HPGe information, and $p_{\mathrm{RC}}(x)$ denotes a reference distribution constructed from random-coincident LAr observables.

The model encodes both modalities into a shared embedding space,

$$
z_x = f_\theta(x),
\qquad
z_c = g_\theta(c),
$$

and computes a temperature-scaled bilinear score,

$$
s_\theta(x,c)
=
\frac{\langle z_x,z_c\rangle}{\tau}.
$$

Training uses a contrastive neural ratio estimation objective inspired by NRE-C [1]. The learned score is then used as a test statistic. Downstream inference uses ensemble predictions and empirical null samples to construct event-level and global p-value quantities. Ensemble spread is also used as an epistemic-uncertainty diagnostic.


## Deep supervision and prefix information gain

The deep-supervision variant trains scores at intermediate HPGe prefixes. This is useful because the HPGe context is small and structured: detector identity, energy, drift-time quantities, and pulse-shape-related features. The goal is not only to obtain a score for the full HPGe context, but also to model how the likelihood ratio changes when each additional HPGe feature is revealed.

Let

$$
c_{\leq t}=(c_1,\ldots,c_t)
$$

denote the HPGe prefix available up to feature $t$. The prefix-dependent log-ratio is

$$
\ell_t(x,c_{\leq t})
=
\log \frac{p_{\mathrm{TC}}(x \mid c_{\leq t})}{p_{\mathrm{RC}}(x)} .
$$

The incremental change from prefix $t-1$ to prefix $t$ is

$$
\ell_t(x,c_{\leq t}) - \ell_{t-1}(x,c_{<t})
=
\log
\frac{p_{\mathrm{TC}}(x \mid c_{\leq t})}
{p_{\mathrm{TC}}(x \mid c_{<t})}.
$$

Using Bayes' rule, this can also be written as

$$
\ell_t(x,c_{\leq t}) - \ell_{t-1}(x,c_{<t})
=
\log
\frac{p_{\mathrm{TC}}(c_t \mid x,c_{<t})}
{p_{\mathrm{TC}}(c_t \mid c_{<t})}.
$$

Thus, each feature contributes an incremental conditional information-gain term: it measures how much the newly observed HPGe feature $c_t$ changes the LAr-side likelihood ratio beyond what was already explained by the previous HPGe prefix.

This motivates three architectural choices.

First, the HPGe encoder used for deep supervision is causal. Its so-called pre-cumulative token output at position $t$, denoted

$$
u_t = u_\theta(c_{\leq t}),
$$

may depend on the current and previous HPGe features, but not on future features. This is necessary if $u_t$ is to represent the update associated with revealing $c_t$ in the context of $c_{<t}$.

Second, the encoder constructs prefix embeddings with a cumulative sum over the pre-cumulative token outputs,

$$
h_t
=
\sum_{k=0}^{t} u_k ,
$$

where $u_0$ is the SOS / empty-prefix contribution. The prefix score is then

$$
s_{\theta,t}(x,c_{\leq t})
=
\frac{\langle z_x,h_t\rangle}{\tau}
=
\sum_{k=0}^{t}
\frac{\langle z_x,u_k\rangle}{\tau}.
$$

Third, prefix-wise contrastive losses train each $s_{\theta,t}$ to approximate the corresponding prefix log-ratio $\ell_t$. In the ideal convergence limit,

$$
s_{\theta,t}(x,c_{\leq t})
\approx
\ell_t(x,c_{\leq t}) .
$$

Therefore the dot product between the LAr embedding and the pre-cumulative HPGe token at position $t$ has the interpretation

$$
\frac{\langle z_x,u_t\rangle}{\tau}
=
s_{\theta,t}(x,c_{\leq t})
-
s_{\theta,t-1}(x,c_{<t})
\approx
\ell_t(x,c_{\leq t})
-
\ell_{t-1}(x,c_{<t}) .
$$

Combining this with the likelihood-ratio decomposition above gives

$$
\frac{\langle z_x,u_t\rangle}{\tau}
\approx
\log
\frac{p_{\mathrm{TC}}(x \mid c_{\leq t})}
{p_{\mathrm{TC}}(x \mid c_{<t})}
=
\log
\frac{p_{\mathrm{TC}}(c_t \mid x,c_{<t})}
{p_{\mathrm{TC}}(c_t \mid c_{<t})}.
$$

This is the main inductive bias of the deep-supervision architecture: each pre-cumulative HPGe token is encouraged to learn an incremental residual contribution to the conditional log-ratio. If the correlation structure between LAr and HPGe observables is already explained by earlier HPGe features, then later HPGe tokens should have little effect on the score.

Without the cumulative-sum layer, independently parameterized prefix embeddings could still learn prefix scores, but the difference between two neighboring prefix scores would not be tied to the dot product of a single causal HPGe token. The cumulative-sum layer makes this residual information-gain interpretation explicit.


## Deep-supervision loss

The deep-supervision branch uses an NRE-C-style contrastive loss at each valid HPGe prefix. This section describes the loss as implemented in the code.

For a contrastive group of size $K$, let

$$
a_{gij}
$$

denote the score between LAr row $i$ and HPGe candidate $j$ in group $g$. The implementation prepends a null-class logit $\log K$ and shifts all candidate logits by $\log \gamma$. Therefore the normalized probabilities are

$$
q_0(a_{gi:})
=
\frac{K}
{K+\gamma\sum_{j=1}^{K}\exp(a_{gij})}
$$

for the null class, and

$$
q_j(a_{gi:})
=
\frac{\gamma \exp(a_{gij})}
{K+\gamma\sum_{\ell=1}^{K}\exp(a_{gi\ell})}
$$

for candidate $j$.

Given null / independent rows $a^{(0)}$ and matched / dependent rows $a^{(1)}$, the row-wise NRE-C loss used in the implementation is

$$
\mathcal{L}_{\mathrm{NREC}}(a^{(0)},a^{(1)})
=
\frac{1}{1+\gamma}
\left[
-\frac{1}{GK}
\sum_{g=1}^{G}
\sum_{i=1}^{K}
\log q_0(a^{(0)}_{gi:})
\right]
+
\frac{\gamma}{1+\gamma}
\left[
-\frac{1}{GK}
\sum_{g=1}^{G}
\sum_{i=1}^{K}
\log q_i(a^{(1)}_{gi:})
\right].
$$

The first term trains randomly coincident rows to select the null class. The second term trains matched rows to select the corresponding HPGe candidate.


### Main prefix loss

For prefix $t$, let $h_{tgj}$ denote the HPGe prefix embedding for candidate $j$ in group $g$. Let $z^{(0)}_{tgi}$ denote an independent / null LAr embedding and $z^{(1)}_{tgi}$ denote a matched / dependent LAr embedding. The main prefix scores are

$$
a^{(0,t)}_{gij}
=
\frac{\langle z^{(0)}_{tgi}, h_{tgj}\rangle}{\tau},
\qquad
a^{(1,t)}_{gij}
=
\frac{\langle z^{(1)}_{tgi}, h_{tgj}\rangle}{\tau}.
$$

The prefix-wise main loss is

$$
\mathcal{L}^{\mathrm{main}}_t
=
\mathcal{L}_{\mathrm{NREC}}
\left(
a^{(0,t)},
a^{(1,t)}
\right),
$$

computed only over valid contrastive groups for prefix $t$. Prefixes are then combined with configurable weights $\alpha_t$,

$$
\mathcal{L}_{\mathrm{main}}
=
\frac{
\sum_{t\in\mathcal{A}}
\alpha_t
\mathcal{L}^{\mathrm{main}}_t
}{
\sum_{t\in\mathcal{A}}
\alpha_t
},
$$

where $\mathcal{A}$ is the set of active prefixes with valid groups and nonzero weight.

In the ideal limit, this main loss trains

$$
s_{\theta,t}(x,c_{\leq t})
\approx
\log
\frac{
p_{\mathrm{TC}}(x\mid c_{\leq t})
}{
p_{\mathrm{RC}}(x)
}.
$$

For $t=0$, the HPGe context is the SOS / empty prefix, so the learned score is the marginal TC-vs-RC log-ratio,

$$
s_{\theta,0}(x)
\approx
\log
\frac{
p_{\mathrm{TC}}(x)
}{
p_{\mathrm{RC}}(x)
}.
$$


### Auxiliary interaction loss

The auxiliary interaction loss uses prefix zero as a learned marginal baseline. For $t\geq 1$, it subtracts the prefix-zero score from the prefix-$t$ score.

For matched / dependent rows, the auxiliary scores are

$$
b^{(1,t)}_{gij}
=
s_{\theta,t}(x^{(1)}_{tgi},c^{(j)}_{\leq t})
-
s_{\theta,0}(x^{(1)}_{tgi}).
$$

For null / independent rows, the implementation rolls the LAr examples across groups, giving LAr samples drawn from the TC marginal but independent of the current HPGe candidates. The corresponding auxiliary scores are

$$
b^{(0,t)}_{gij}
=
s_{\theta,t}(\tilde{x}^{(1)}_{tgi},c^{(j)}_{\leq t})
-
s_{\theta,0}(\tilde{x}^{(1)}_{tgi}).
$$

The auxiliary prefix loss is then

$$
\mathcal{L}^{\mathrm{aux}}_t
=
\mathcal{L}_{\mathrm{NREC}}
\left(
b^{(0,t)},
b^{(1,t)}
\right),
\qquad
t\geq 1.
$$

The auxiliary loss averages over active auxiliary prefixes,

$$
\mathcal{L}_{\mathrm{aux}}
=
\frac{1}{|\mathcal{A}_{\mathrm{aux}}|}
\sum_{t\in\mathcal{A}_{\mathrm{aux}}}
\mathcal{L}^{\mathrm{aux}}_t .
$$

This loss has a different ratio interpretation from the main prefix loss. If

$$
s_{\theta,t}(x,c_{\leq t})
\approx
\log
\frac{
p_{\mathrm{TC}}(x\mid c_{\leq t})
}{
p_{\mathrm{RC}}(x)
}
$$

and

$$
s_{\theta,0}(x)
\approx
\log
\frac{
p_{\mathrm{TC}}(x)
}{
p_{\mathrm{RC}}(x)
},
$$

then the auxiliary score satisfies

$$
s_{\theta,t}(x,c_{\leq t}) - s_{\theta,0}(x)
\approx
\log
\frac{
p_{\mathrm{TC}}(x\mid c_{\leq t})
}{
p_{\mathrm{TC}}(x)
}.
$$

Equivalently,

$$
s_{\theta,t}(x,c_{\leq t}) - s_{\theta,0}(x)
\approx
\log
\frac{
p_{\mathrm{TC}}(c_{\leq t}\mid x)
}{
p_{\mathrm{TC}}(c_{\leq t})
}.
$$

Thus, the auxiliary loss trains interaction ratios between LAr and HPGe under the TC marginal reference distribution $p_{\mathrm{TC}}(x)$, rather than against the RC reference $p_{\mathrm{RC}}(x)$. It encourages the model to represent the information in the HPGe prefix that is specifically relevant for explaining TC LAr structure beyond the marginal TC-vs-RC separation.

This should be distinguished from the single-step incremental contribution,

$$
s_{\theta,t}(x,c_{\leq t})
-
s_{\theta,t-1}(x,c_{<t})
=
\frac{\langle z_x,u_t\rangle}{\tau},
$$

which corresponds to the contribution of the pre-cumulative causal HPGe token at position $t$. The auxiliary loss supervises the cumulative interaction score

$$
s_{\theta,t}(x,c_{\leq t}) - s_{\theta,0}(x)
=
\sum_{k=1}^{t}
\frac{\langle z_x,u_k\rangle}{\tau}.
$$


### Total deep-supervision loss

The final deep-supervision objective combines the main prefix loss and the auxiliary interaction loss with configurable weights

$$
\lambda_{\mathrm{main}},
\qquad
\lambda_{\mathrm{aux}}.
$$

The total loss is

$$
\mathcal{L}_{\mathrm{total}}
=
\frac{
\lambda_{\mathrm{main}}\mathcal{L}_{\mathrm{main}}
+
\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}
}{
\lambda_{\mathrm{main}}+\lambda_{\mathrm{aux}}
}.
$$

If either component is disabled by setting its weight to zero, the objective reduces to the remaining active component.


## Packed segmented cumulative sums

For packed variable-length HPGe sequences, the cumulative-sum operation is applied segment-wise. For event $b$, prefix position $t$, and embedding dimension $d$,

$$
Y_{btd}
=
\sum_{k=0}^{t} X_{bkd}.
$$

The reverse-mode gradient is the corresponding reverse cumulative sum,

$$
\frac{\partial \mathcal{L}}{\partial X_{bkd}}
=
\sum_{t \geq k}
\frac{\partial \mathcal{L}}{\partial Y_{btd}}.
$$

As a result, prefix losses are coupled: a loss applied at a later prefix also updates all earlier incremental HPGe contributions. This encourages early features to learn stable shared information and later features to learn residual information beyond the previous prefix.

The implementation supports:

- prefix-wise contrastive losses;
- configurable prefix weights;
- an auxiliary interaction loss using prefix 0 as a marginal baseline;
- packed segmented cumulative sums for additive prefix representations;
- logging of prefix-level training and validation losses.

The deep-supervision training code is implemented mainly in `legend_lar.kfold_ensemble.nre_c_ds`. The packed segmented cumulative-sum operation is implemented with custom Triton kernels in `legend_lar.kernels` and wrapped by `legend_lar.model.segment_cumsum`.

## Package structure

```text
src/legend_lar/
├── model/              # Encoders, tokenizers, transformer blocks
├── data/               # Iterable datasets and collate functions
├── kfold_ensemble/     # Training code for k-fold/bootstrap ensembles
├── calibration/        # Ensemble inference and empirical calibration
├── kernels/            # Custom Triton kernels implementation
└── utils/              # Configuration, file handling, RNG utilities
```

## Installation

```bash
pip install -e .
```

The code assumes a CUDA-capable PyTorch environment for the GPU-oriented model components. Some components depend on FlashAttention and are intended for GPU/HPC execution.

## Data and configuration

The repository does not contain the data or configuration files used in the original analysis. The training and inference entry points expect externally provided directories containing processed sparse arrays, HPGe feature arrays, detector-coordinate files, model configuration JSON files, and checkpoint/output locations.

## Status

This repository is maintained as research code for method development. Interfaces and configuration formats may change.

## References

[1] Benjamin Kurt Miller, Christoph Weniger, and Patrick Forré. *Contrastive Neural Ratio Estimation for Simulation-based Inference*. arXiv:2210.06170, 2022.
