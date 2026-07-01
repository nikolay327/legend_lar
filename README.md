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

$$ s_\theta(x,c) \approx \log r(x,c) = \log \frac{p_{\mathrm{TC}}(x \mid c)}{p_{\mathrm{RC}}(x)} ,$$

where $p_{\mathrm{TC}}(x \mid c)$ denotes the conditional distribution of true-coincident LAr observables given HPGe information, and $p_{\mathrm{RC}}(x)$ denotes a reference distribution constructed from random-coincident LAr observables.

The model encodes both modalities into a shared embedding space,

$$z_x = f_\theta(x),\qquad z_c = g_\theta(c),$$

and computes a temperature-scaled bilinear score,

$$s_\theta(x,c)=\frac{\langle z_x,z_c\rangle}{\tau}.$$

Training uses a contrastive neural ratio estimation objective inspired by NRE-C [1]. The learned score is then used as a test statistic. Downstream inference uses ensemble predictions and empirical null samples to construct event-level and global p-value quantities. Ensemble spread is also used as an epistemic-uncertainty diagnostic.


## Deep supervision and prefix information gain

The deep-supervision variant trains scores at intermediate HPGe prefixes. This is useful because the HPGe context is small and structured: detector identity, energy, drift-time quantities, and pulse-shape-related features. The goal is not only to obtain a score for the full HPGe context, but also to model how the likelihood ratio changes when each additional HPGe feature is revealed.

Let

$$c_{\leq t}=(c_1,\ldots,c_t)$$

denote the HPGe prefix available up to feature $t$. The prefix-dependent log-ratio is

$$\ell_t(x,c_{\leq t})=\log \frac{p_{\mathrm{TC}}(x \mid c_{\leq t})}{p_{\mathrm{RC}}(x)} .$$

The incremental change from prefix $t-1$ to prefix $t$ is

$$\ell_t(x, c_{\leq t}) - \ell_{t-1}(x, c_{\lt t})=\log \frac{p_{\mathrm{TC}}(x \mid c_{\leq t})}{p_{\mathrm{TC}}(x \mid c_{\lt t})}.$$

Using Bayes' rule, this can also be written as

$$\ell_t(x,c_{\leq t}) - \ell_{t-1}(x,c_{\lt t})=\log\frac{p_{\mathrm{TC}}(c_t \mid x,c_{\lt t})}{p_{\mathrm{TC}}(c_t \mid c_{\lt t})}.$$

Thus, each feature contributes an incremental conditional information-gain term: it measures how much the newly observed HPGe feature $c_t$ changes the LAr-side likelihood ratio beyond what was already explained by the previous HPGe prefix.

This motivates three architectural choices.

First, the HPGe encoder used for deep supervision is causal. Its so-called pre-cumulative token output at position $t$, denoted

$$u_t = u_\theta(c_{\leq t}),$$

may depend on the current and previous HPGe features, but not on future features. This is necessary if $u_t$ is to represent the update associated with revealing $c_t$ in the context of $c_{\lt t}$.

Second, the encoder constructs prefix embeddings with a cumulative sum over the pre-cumulative token outputs,

$$h_t=\sum_{k=0}^{t} u_k ,$$

where $u_0$ is the SOS / empty-prefix contribution. The prefix score is then

$$s_{\theta,t}(x,c_{\leq t})=\frac{\langle z_x,h_t\rangle}{\tau}=\sum_{k=0}^{t}\frac{\langle z_x,u_k\rangle}{\tau}.$$

Third, prefix-wise contrastive losses train each $s_{\theta,t}$ to approximate the corresponding prefix log-ratio $\ell_t$. In the ideal convergence limit,

$$s_{\theta,t}(x,c_{\leq t})\approx\ell_t(x,c_{\leq t}) .$$

Therefore the dot product between the LAr embedding and the pre-cumulative HPGe token at position $t$ has the interpretation

$$\frac{\langle z_x,u_t\rangle}{\tau}=s_{\theta,t}(x,c_{\leq t})-s_{\theta,t-1}(x,c_{\lt t})\approx\ell_t(x,c_{\leq t})-\ell_{t-1}(x,c_{\lt t}) .$$

Combining this with the likelihood-ratio decomposition above gives

$$\frac{\langle z_x,u_t\rangle}{\tau}\approx\log\frac{p_{\mathrm{TC}}(x \mid c_{\leq t})}{p_{\mathrm{TC}}(x \mid c_{\lt t})}=\log\frac{p_{\mathrm{TC}}(c_t \mid x,c_{\lt t})}{p_{\mathrm{TC}}(c_t\mid c_{\lt t})}.$$

This is the main inductive bias of the deep-supervision architecture: each pre-cumulative HPGe token is encouraged to learn an incremental residual contribution to the conditional log-ratio. If the correlation structure between LAr and HPGe observables is already explained by earlier HPGe features, then later HPGe tokens should have little effect on the score.

Without the cumulative-sum layer, independently parameterized prefix embeddings could still learn prefix scores, but the difference between two neighboring prefix scores would not be tied to the dot product of a single causal HPGe token. The cumulative-sum layer makes this residual information-gain interpretation explicit.


## Deep-supervision loss

The deep-supervision branch uses an NRE-C-style contrastive loss at each valid HPGe prefix. This section describes the loss as implemented in the code.

For a contrastive group of size $K$, let

$$a_{gij}$$

denote the score between LAr row $i$ and HPGe candidate $j$ in group $g$. The implementation prepends a null-class logit $\log K$ and shifts all candidate logits by $\log \gamma$. Therefore the normalized probabilities are

$$q_0(a_{gi:})=\frac{K}{K+\gamma\sum_{j=1}^{K}\exp(a_{gij})}$$

for the null class, and

$$q_j(a_{gi:})=\frac{\gamma \exp(a_{gij})}{K+\gamma\sum_{\ell=1}^{K}\exp(a_{gi\ell})}$$

for candidate $j$.

Given null / independent rows $a^{(0)}$ and matched / dependent rows $a^{(1)}$, the row-wise NRE-C loss used in the implementation is

$$\mathcal{L}_{\mathrm{NREC}}(a^{(0)},a^{(1)})=\frac{1}{1+\gamma}\left[-\frac{1}{GK}\sum_{g=1}^{G}\sum_{i=1}^{K}\log q_0(a^{(0)}_{gi:})\right]+\frac{\gamma}{1+\gamma}\left[-\frac{1}{GK}\sum_{g=1}^{G}\sum_{i=1}^{K}\log q_i(a^{(1)}_{gi:})\right].$$

The first term trains randomly coincident rows to select the null class. The second term trains matched rows to select the corresponding HPGe candidate.


### Main prefix loss

For prefix $t$, let $h_{tgj}$ denote the HPGe prefix embedding for candidate $j$ in group $g$. Let $z_{tgi}^{(0)}$ denote an independent / null LAr embedding and $z^{(1)}_{tgi}$ denote a matched / dependent LAr embedding. The main prefix scores are

$$a^{(0,t)}_{gij}=\frac{\langle z^{(0)}_{tgi}, h_{tgj}\rangle}{\tau},\qquad a^{(1,t)}_{gij}=\frac{\langle z^{(1)}_{tgi}, h_{tgj}\rangle}{\tau}.$$

The prefix-wise main loss is

$$\mathcal{L}^{\mathrm{main}}_t=\mathcal{L}_{\mathrm{NREC}}\left(a^{(0,t)},a^{(1,t)}\right),$$

computed only over valid contrastive groups for prefix $t$. Prefixes are then combined with configurable weights $\alpha_t$,

$$\mathcal{L}_{\mathrm{main}}=\frac{\sum_{t\in\mathcal{A}}\alpha_t\mathcal{L}^{\mathrm{main}}_t}{\sum_{t\in\mathcal{A}}\alpha_t},$$

where $\mathcal{A}$ is the set of active prefixes with valid groups and nonzero weight.

In the ideal limit, this main loss trains

$$s_{\theta,t}(x,c_{\leq t})\approx\log\frac{p_{\mathrm{TC}}(x\mid c_{\leq t})}{p_{\mathrm{RC}}(x)}.$$

For $t=0$, the HPGe context is the SOS / empty prefix, so the learned score is the marginal TC-vs-RC log-ratio,

$$s_{\theta,0}(x)\approx\log\frac{p_{\mathrm{TC}}(x)}{p_{\mathrm{RC}}(x)}.$$


### Auxiliary interaction loss

The auxiliary interaction loss uses prefix zero as a learned marginal baseline. For $t\geq 1$, it subtracts the prefix-zero score from the score at prefix $t$.

For matched / dependent rows, the auxiliary scores are

$$b^{(1,t)}_{gij}=s_{\theta,t}(x^{(1)}_{tgi},c^{(j)}_{\leq t})-s_{\theta,0}(x^{(1)}_{tgi}).$$

For null / independent rows, the implementation rolls the LAr examples across groups, giving LAr samples drawn from the TC marginal but independent of the current HPGe candidates. The corresponding auxiliary scores are

$$b^{(0,t)}_{gij}=s_{\theta,t}(\tilde{x}^{(1)}_{tgi},c^{(j)}_{\leq t})-s_{\theta,0}(\tilde{x}^{(1)}_{tgi}).$$

The auxiliary prefix loss is then

$$\mathcal{L}^{\mathrm{aux}}_t=\mathcal{L}_{\mathrm{NREC}}\left(b^{(0,t)},b^{(1,t)}\right),\qquad t\geq 1.$$

The auxiliary loss averages over active auxiliary prefixes,

$$\mathcal{L}_{\mathrm{aux}}=\frac{1}{|\mathcal{A}_{\mathrm{aux}}|}\sum_{t\in\mathcal{A}_{\mathrm{aux}}}\mathcal{L}^{\mathrm{aux}}_t .$$

This loss has a different ratio interpretation from the main prefix loss. If

$$s_{\theta,t}(x,c_{\leq t})\approx\log\frac{p_{\mathrm{TC}}(x\mid c_{\leq t})}{p_{\mathrm{RC}}(x)}$$

and

$$s_{\theta,0}(x)\approx\log\frac{p_{\mathrm{TC}}(x)}{p_{\mathrm{RC}}(x)},$$

then the auxiliary score satisfies

$$s_{\theta,t}(x,c_{\leq t}) - s_{\theta,0}(x)\approx\log\frac{p_{\mathrm{TC}}(x\mid c_{\leq t})}{p_{\mathrm{TC}}(x)}.$$

Equivalently,

$$s_{\theta,t}(x,c_{\leq t}) - s_{\theta,0}(x)\approx\log\frac{p_{\mathrm{TC}}(c_{\leq t}\mid x)}{p_{\mathrm{TC}}(c_{\leq t})}.$$

Thus, the auxiliary loss trains interaction ratios between LAr and HPGe under the TC marginal reference distribution $p_{\mathrm{TC}}(x)$, rather than against the RC reference $p_{\mathrm{RC}}(x)$. It encourages the model to represent the information in the HPGe prefix that is specifically relevant for explaining TC LAr structure beyond the marginal TC-vs-RC separation.

This should be distinguished from the single-step incremental contribution,

$$s_{\theta,t}(x,c_{\leq t})-s_{\theta,t-1}(x,c_{\lt t})=\frac{\langle z_x,u_t\rangle}{\tau},$$

which corresponds to the contribution of the pre-cumulative causal HPGe token at position $t$. The auxiliary loss supervises the cumulative interaction score

$$s_{\theta,t}(x,c_{\leq t}) - s_{\theta,0}(x)=\sum_{k=1}^{t}\frac{\langle z_x,u_k\rangle}{\tau}.$$


### Total deep-supervision loss

The final deep-supervision objective combines the main prefix loss and the auxiliary interaction loss with configurable weights

$$\lambda_{\mathrm{main}},\qquad \lambda_{\mathrm{aux}}.$$

The total loss is

$$\mathcal{L}_{\mathrm{total}}=\frac{\lambda_{\mathrm{main}}\mathcal{L}_{\mathrm{main}}+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}}{\lambda_{\mathrm{main}}+\lambda_{\mathrm{aux}}}.$$

If either component is disabled by setting its weight to zero, the objective reduces to the remaining active component.


## Packed segmented cumulative sums

For packed variable-length HPGe sequences, the cumulative-sum operation is applied segment-wise. For event $b$, prefix position $t$, and embedding dimension $d$,

$$Y_{btd}=\sum_{k=0}^{t} X_{bkd}.$$

The reverse-mode gradient is the corresponding reverse cumulative sum,

$$\frac{\partial \mathcal{L}}{\partial X_{bkd}}=\sum_{t \geq k}\frac{\partial \mathcal{L}}{\partial Y_{btd}}.$$

As a result, prefix losses are coupled: a loss applied at a later prefix also updates all earlier incremental HPGe contributions. This encourages early features to learn stable shared information and later features to learn residual information beyond the previous prefix.

The implementation supports:

- prefix-wise contrastive losses;
- configurable prefix weights;
- an auxiliary interaction loss using prefix 0 as a marginal baseline;
- packed segmented cumulative sums for additive prefix representations;
- logging of prefix-level training and validation losses.

The deep-supervision training code is implemented mainly in `legend_lar.kfold_ensemble.nre_c_ds`. The packed segmented cumulative-sum operation is implemented with custom Triton kernels in `legend_lar.kernels` and wrapped by `legend_lar.model.segment_cumsum`.

## Inference and empirical calibration

Inference converts ensemble scores into empirical p-values by comparing the observed statistics to finite null buffers.


### Prefix selection

For deep-supervision models, inference first selects which HPGe prefix embedding is used for each event. The prefix is controlled by `max_t`:

- `max_t < 0`: use the full / rightmost available HPGe prefix;
- `max_t = 0`: use the SOS / empty-prefix embedding;
- `max_t = k > 0`: use the rightmost observed raw HPGe feature with feature index smaller than $k$.

The selected prefix index is saved as `t_used`.


### Ensemble evidence and epistemic statistics

For an event with LAr observables $x$ and selected HPGe prefix $c_{\leq t}$, each ensemble member $m$ produces a score

$$T_m(x,c_{\leq t})=\frac{\langle z^{(m)}_x,h^{(m)}_t\rangle}{\tau}.$$

The evidence statistic is the ensemble-mean score,

$$T_{\mathrm{evidence}}(x,c_{\leq t})=\frac{1}{M}\sum_{m=1}^{M}T_m(x,c_{\leq t}),
$$

where $M$ is the number of ensemble members. The epistemic statistic is the ensemble variance,

$$T_{\mathrm{epistemic}}(x,c_{\leq t})=\textbf{Var}_{m}\left[T_m(x,c_{\leq t})\right].$$

The evidence statistic measures how large the learned log-ratio score is on average across the ensemble. The epistemic statistic measures how much the ensemble members disagree on the score.


### Event-level empirical p-values

For each physical event, the selected HPGe prefix embedding is held fixed and compared against LAr embeddings from an event-level null buffer. Let

$$T^{(r)}_{\mathrm{null}}(c_{\leq t})$$

denote the evidence statistic obtained by pairing the event's HPGe prefix with null LAr sample $r$. Similarly, let

$$U^{(r)}_{\mathrm{null}}(c_{\leq t})$$

denote the corresponding epistemic statistic under the same null pairing.

The empirical evidence p-value is the upper-tail rank of the observed evidence statistic under this event-level null distribution,

$$p_{\mathrm{evidence}}=\frac{1+\sum_{r=1}^{N_{\mathrm{null}}}\mathbf{1}\left[T^{(r)}_{\mathrm{null}}\geq T_{\mathrm{evidence}}\right]}{N_{\mathrm{null}}+1}.$$

The empirical epistemic p-value is computed analogously from the ensemble-variance statistic,

$$p_{\mathrm{epistemic}}=\frac{1+\sum_{r=1}^{N_{\mathrm{null}}}\mathbf{1}\left[U^{(r)}_{\mathrm{null}}\geq T_{\mathrm{epistemic}}\right]}{N_{\mathrm{null}}+1}.$$

Here, a finite-sample empirical p-value in $(0,1]$ is used. Small $p_{\mathrm{evidence}}$ means that the observed evidence score is large compared with the event-level null distribution. Small $p_{\mathrm{epistemic}}$ means that the ensemble disagreement is unusually large compared with the null distribution.


### Global-null calibration

The inference code also evaluates a separate global-null buffer. For each event, the selected HPGe prefix is paired with many LAr samples from the global null, producing global-null evidence and epistemic statistics

$$T^{(q)}_{\mathrm{glob\ null}},\qquad U^{(q)}_{\mathrm{glob\ null}}.$$

Each global-null statistic is converted into an event-level empirical p-value using the same event-level null distribution as above,

$$p^{(q)}_{\mathrm{evidence,glob\ null}}=\frac{1+\sum_{r=1}^{N_{\mathrm{null}}}\mathbf{1}\left[T^{(r)}_{\mathrm{null}}\geq T^{(q)}_{\mathrm{glob\ null}}\right]}{N_{\mathrm{null}}+1},$$

and

$$p^{(q)}_{\mathrm{epistemic,glob\ null}}=\frac{1+\sum_{r=1}^{N_{\mathrm{null}}}\mathbf{1}\left[U^{(r)}_{\mathrm{null}}\geq U^{(q)}_{\mathrm{glob\ null}}\right]}{N_{\mathrm{null}}+1}.$$

These quantities form a global-null distribution of event-level empirical p-values.


### Global empirical p-value

If epistemic rejection is enabled, the event-level empirical p-values are combined into a global score. By default,

$$T_{\mathrm{global}}=p_{\mathrm{evidence}}.$$

For events that are both epistemically suspicious and flagged by the classical LAr classifier, the score is moved into a negative rejection region,

$$T_{\mathrm{global}}=-1+(1-2\epsilon)\frac{p_{\mathrm{epistemic}}}{\alpha_{\mathrm{epistemic}}}+\epsilon p_{\mathrm{evidence}},$$

where $\alpha_{\mathrm{epistemic}}$ is the epistemic threshold and $\epsilon$ is a small numerical constant. The same transformation is applied to the global-null empirical p-values, giving null global scores

$$T^{(q)}_{\mathrm{global,null}}.$$

The final global empirical p-value is the lower-tail rank of the observed global score under the global-null score distribution,

$$p_{\mathrm{global}}=\frac{1+\sum_{q=1}^{N_{\mathrm{glob\ null}}}\mathbf{1}\left[T^{(q)}_{\mathrm{global,null}}\leq T_{\mathrm{global}}\right]}{N_{\mathrm{glob\ null}}+1}.$$

The lower-tail convention is used because $T_{\mathrm{global}}$ is p-value-like: smaller values are more signal-like or more anomalous with respect to the calibrated global-null distribution.

If epistemic rejection is disabled, the code stores

$$p_{\mathrm{global}} = p_{\mathrm{evidence}}.$$


### Saved inference quantities

The inference table stores the selected prefix and the empirical calibration outputs, including

- `t_used`: selected HPGe prefix;
- `t_evidence`: ensemble-mean evidence statistic;
- `t_epistemic`: ensemble-variance epistemic statistic;
- `p_evidence`: event-level empirical evidence p-value;
- `p_epistemic`: event-level empirical epistemic p-value;
- `glob_null_p_evidence`: event-level empirical evidence p-values evaluated on global-null samples;
- `glob_null_p_epistemic`: event-level empirical epistemic p-values evaluated on global-null samples;
- `t_global`: combined global score, when epistemic rejection is enabled;
- `p_global`: final global empirical p-value.

The global-null empirical p-values saved by the code provide calibration and sanity-check distributions for the inference procedure. The inference code is implemented in `legend_lar.calibration.nre_c_inference`.

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
