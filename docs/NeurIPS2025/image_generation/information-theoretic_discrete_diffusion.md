---
title: >-
  [Paper Note] Information-Theoretic Discrete Diffusion
description: >-
  [NeurIPS 2025][Image Generation][discrete diffusion models] This work generalizes the classical I-MMSE identity from continuous diffusion to the discrete domain…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "discrete diffusion models"
  - "information theory"
  - "likelihood estimation"
  - "score matching"
  - "masked diffusion"
  - "I-MMSE"
date: 2026-05-08
content_hash: 4ab52044e3239597
---

# Information-Theoretic Discrete Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2510.24088](https://arxiv.org/abs/2510.24088)  
**Code**: [github.com/Dongjae0324/infodis](https://github.com/Dongjae0324/infodis)  
**Area**: Generative Model Theory / Discrete Diffusion
**Keywords**: discrete diffusion models, information theory, likelihood estimation, score matching, masked diffusion, I-MMSE

## TL;DR
This work generalizes the classical I-MMSE identity from continuous diffusion to the discrete domain, establishing the I-MDSE and I-MDCE relations. It proves that DSE/DCE losses are not merely variational upper bounds but **exact decompositions** of the log-likelihood, and derives time-free formulas, conditional likelihood estimators, and coupled likelihood-ratio estimators. The proposed methods are validated on large-scale models such as LLaDA, demonstrating low variance and out-of-distribution detection capability.

## Background & Motivation

Continuous diffusion models (e.g., Gaussian diffusion) rest on a mature information-theoretic foundation: the **I-MMSE identity** of Guo et al. (2005) relates the rate of change of mutual information to the minimum mean squared error, and Kong et al. (2023) further extended this to a pointwise decomposition of the log-likelihood. However, discrete diffusion models—which handle categorical data such as text and DNA sequences—have developed rapidly (D3PM, SEDD, LLaDA, etc.) without a corresponding information-theoretic framework.

Existing discrete diffusion training losses (DSE, DCE) are typically regarded as **variational upper bounds** on the negative log-likelihood, leaving two open questions:

1. Are these losses merely upper bounds, or can they serve as exact likelihood estimators?
2. Can an exact likelihood decomposition be obtained from first-order score functions alone, as in the continuous case?

## Core Problem

**Objective**: Establish a rigorous information-theoretic framework for discrete diffusion models, prove exact (equality rather than inequality) relations between training losses and log-likelihoods, and derive practical likelihood estimation methods from these relations.

## Method

### 1. I-MDSE Relation (General Discrete Diffusion)

Consider a discrete diffusion forward process driven by a continuous-time Markov chain (CTMC): $\frac{dp_t}{dt} = Q_t p_t$. Define the minimum denoising score entropy (MDSE) as:

$$\mathrm{mdse}(x_0, t) := \mathbb{E}_{p_{t|0}(x_t|x_0)}[\ell_{\mathrm{DSE}}(x_0, x_t, t, s_t^\star)]$$

where $s_t^\star$ is the optimal score function minimizing the DSE loss.

**Theorem 3.1 (I-MDSE Identity)**:

$$\frac{d}{dt} D_{\mathrm{KL}}(p_{t|0}(\cdot|x_0) \| p_t) = -\mathrm{mdse}(x_0, t)$$

Taking expectations yields the marginal form: $\frac{d}{dt} I(x_0; x_t) = -\mathrm{mdse}(t)$

**Core Insight**: The rate of decay of mutual information equals exactly the negative of the minimum DSE loss—the discrete-domain counterpart of the continuous I-MMSE identity.

**Theorem 3.2 (NLL Decomposition)**: Integrating over time gives:

$$-\log p_0(x_0) = \int_0^\infty \mathrm{mdse}(x_0, t) \, dt$$

→ The DSE loss is not a variational upper bound but an **exact estimator** of the log-likelihood.

### 2. I-MDCE Relation (Masked Diffusion)

For the practically prevalent absorbing (masked) diffusion model, the denoising cross-entropy (DCE) loss is defined as:

$$\ell_{\mathrm{DCE}}(\mathbf{x}_0, \mathbf{x}, c) := \sum_{i=1}^L \mathbb{1}[x^i = [\mathrm{M}]] \log \frac{1}{c(\mathbf{x})_{i, x_0^i}}$$

**Lemma 3.3 (Pointwise Equivalence)**: Via the time reparameterization $\lambda = 1 - e^{-\bar{\sigma}(t)}$, DSE and DCE are exactly equivalent at the pointwise level:

$$\ell_{\mathrm{DSE}}(\mathbf{x}_0, \mathbf{x}, t, s_t) = \frac{\bar{\sigma}(t)(1-\lambda)}{\lambda} \ell_{\mathrm{DCE}}(\mathbf{x}_0, \mathbf{x}, c)$$

**Theorem 3.4 (Training Loss Equivalence)**: $\mathcal{L}_{\mathrm{DSE}}^T(\mathbf{x}_0) = \mathcal{L}_{\mathrm{DCE}}^\Lambda(\mathbf{x}_0)$ holds for any finite $T$, not merely asymptotically as $T \to \infty$.

**Corollary 3.7 (I-MDCE NLL Decomposition)**:

$$-\log p_0(\mathbf{x}_0) = \int_0^1 \frac{1}{\lambda} \mathrm{mdce}(\mathbf{x}_0, \lambda) \, d\lambda$$

### 3. Time-Free Likelihood Estimation

The time-integral form requires sampling diffusion times in practice. This paper derives an equivalent **time-free formula**:

**Theorem 4.1**:

$$-\log p_0(\mathbf{x}_0) = H_L \, \mathbb{E}_{p(I)} \left[ \sum_{i \notin I} \log \frac{1}{p_0(x_0^i | \mathbf{x}_0^I)} \right]$$

where $I$ is a set of unmasked indices sampled from $p(I) = B(L-|I|, |I|+1)/H_L$, and $H_L$ is the $L$-th harmonic number. Monte Carlo estimation is performed by randomly masking subsets, without explicit time integration.

### 4. Conditional Likelihood Estimation

**Theorem 4.2**: For disjoint index sets $I_1$ (target) and $I_2$ (context):

$$-\log p_0(\mathbf{x}_0^{I_1} | \mathbf{x}_0^{I_2}) = \int_0^1 \frac{1}{\lambda} \mathbb{E}[\cdots] \, d\lambda$$

This admits a time-free form as well (Corollary 4.3), applicable to prompt–response modeling scenarios.

### 5. Coupled Likelihood-Ratio Estimation

By sharing masks (i.e., using the same random unmasked set $I$ for two sequences simultaneously), a coupled estimator is obtained:

$$\log \frac{p_0(\mathbf{y})}{p_0(\mathbf{x})} = H_L \, \mathbb{E}_{p(I)} \left[ \sum_{i \notin I} \log \frac{p_0(y^i | \mathbf{y}^I)}{p_0(x^i | \mathbf{x}^I)} \right]$$

Shared randomness causes positively correlated terms to cancel, substantially reducing variance—directly useful for downstream tasks such as alignment (e.g., DPO).

## Loss & Training

### Training

Models are trained under the standard masked diffusion framework: tokens in $\mathbf{x}_0$ are randomly masked at rate $\lambda$, and the network $c^\theta$ is trained to predict the conditional distribution of masked tokens. The loss is the DCE loss:

$$\mathcal{L}_{\mathrm{DCE}}(\mathbf{x}_0) = \int_0^1 \frac{1}{\lambda} \mathbb{E}_{p_{\lambda|0}} \left[ \sum_i \mathbb{1}[x_\lambda^i = [\mathrm{M}]] \log \frac{1}{c^\theta(\mathbf{x}_\lambda)_{i, x_0^i}} \right] d\lambda$$

The central theoretical contribution is proving that, at the optimum $c^\theta = c^\star$, this loss **exactly equals** $-\log p_0(\mathbf{x}_0)$ rather than merely being an upper bound. Consequently, the training objective of existing masked diffusion language models (e.g., LLaDA) already constitutes theoretically optimal likelihood maximization, requiring no additional correction terms.

In practice, $\lambda$ is sampled uniformly for Monte Carlo approximation of the integral. The network architecture is a standard Transformer (consistent with LLaDA), with time-independent parameterization.

### Inference / Likelihood Estimation

The primary inference task in this paper is **likelihood estimation** rather than generation (sampling). Three practical approaches are provided:

1. **Time-integral estimation**: Sample multiple values of $\lambda$ and numerically integrate the DCE loss, consistent with the training loss form.
2. **Time-free estimation** (recommended): Sample unmasked subsets $I$ according to a Beta distribution weighting, without explicit time integration; variance is reduced by 5–7×.
3. **Coupled likelihood ratio**: Two sequences share the same mask indices $I$; Monte Carlo estimation is applied to the $\log$-ratio, reducing variance by approximately 7×.

The computational cost of time-free estimation is approximately $K$ forward passes ($K$ = number of MC samples), each requiring a single model inference over a different mask pattern, and is straightforward to implement.

## Key Experimental Results

### Experimental Setup

- **Synthetic data**: 128 DNA sequences of length 8 over alphabet {A, T, G, C} with known exact distributions; 5M-length DNA sequences generated by a 4th-order Markov chain (for conditional likelihood validation).
- **Real data**: LLaDA 8B (Nie et al., 2025); text8 corpus with RADD model.
- **Evaluation benchmarks**: HellaSwag, ARC-hard, PIQA (conditional likelihood variance); BeaverTails (likelihood-ratio variance).

### Main Results

| Experiment | Key Result |
|---|---|
| **Synthetic DNA likelihood recovery** | Time-free estimates closely match ground-truth NLL (Fig. 1a unconditional / Fig. 1b conditional) |
| **Variance comparison (HellaSwag)** | 128 MC samples: time-integral variance 70.97 → time-free **11.57** (6× reduction) |
| **Variance comparison (ARC-hard)** | 128 MC samples: 23.18 → **5.73** |
| **Variance comparison (PIQA)** | 128 MC samples: 19.77 → **4.93** |
| **Likelihood-ratio variance (BeaverTails)** | 5 MC samples: decoupled 62469 → coupled **8897** (7× reduction) |
| **OOD detection (text8 + RADD)** | NLL histograms clearly separate in-distribution vs. GPT-4-generated text |
| **LLaDA model audit** | LLaMA 3.1-generated text receives higher likelihood than WikiText → suggests LLaDA training data may be influenced by LLaMA 3.1 |

### Result Analysis

**Variance advantage persists with more samples**: From 128 to 512 MC samples, the time-free estimator's variance on HellaSwag decreases from 11.57 to 2.92 (vs. 70.97 to 13.38 for time-integral), maintaining a relative advantage of approximately 4–6×. This demonstrates that the variance reduction is structural—arising from the elimination of time-sampling noise—rather than incidental.

**Coupled estimator advantage is more pronounced**: On BeaverTails with only 5 MC samples, the coupled estimator achieves 7× lower variance than the decoupled version, with the gap remaining stable as the sample count increases. This has direct practical value for alignment tasks requiring efficient likelihood-ratio estimation (e.g., DPO).

**Model audit finding**: The observation that LLaDA assigns higher likelihood to LLaMA 3.1-generated text suggests potential inclusion of synthetic data in the training corpus (data contamination). This application scenario demonstrates the potential of exact likelihood estimation for AI safety and model provenance analysis.

## Highlights & Insights

1. **From inequality to equality**: DSE/DCE losses are proven to be exact decompositions of the log-likelihood rather than variational upper bounds—a fundamental breakthrough at the information-theoretic level.
2. **Unified framework**: I-MDSE (general discrete diffusion) and I-MDCE (masked diffusion) together constitute a complete information-theoretic toolkit.
3. **Strong practicality**: The time-free formula eliminates Monte Carlo noise from time integration, reducing variance by 5–7×.
4. **Coupled likelihood ratio**: The shared-mask technique is simple and elegant, reducing variance by approximately 7× and directly applicable to alignment.
5. **Model auditing**: Conditional likelihood enables OOD detection and inference of training data provenance, demonstrating the practical value of the theoretical results.
6. **Far-reaching corollary**: First-order scores are proven sufficient for exact likelihood reconstruction, with no higher-order corrections required.

## Limitations & Future Work

1. **Limited experimental scale**: Real-data experiments are conducted primarily on LLaDA 8B; validation on larger models (e.g., tens of billions of parameters) is absent.
2. **Restricted to discrete domain**: The framework is designed specifically for discrete diffusion; generalization to mixed continuous–discrete diffusion models (e.g., joint image–text generation) remains unexplored.
3. **OOD detection is qualitative only**: Model audit experiments are primarily visualization-based, lacking quantitative metrics such as AUROC.
4. **Conditional likelihood estimation behavior**: The behavior of the estimator under highly imbalanced context and target lengths is not thoroughly analyzed.
5. **Relationship with ELBO**: Although exact equality is established, the deeper connection to the ELBO in the VAE framework is not fully discussed.

## Related Work & Insights

| Method | Description | Advantage of This Work |
|---|---|---|
| **SEDD** (Lou et al., 2024) | DSE loss used as variational upper bound for training | This work proves DSE is an exact likelihood estimator |
| **RADD** (Ou et al., 2025) | Asymptotic equivalence of DSE–DCE ($T \to \infty$) | This work proves exact equivalence for any finite $T$ |
| **LLaDA** (Nie et al., 2025) | Large-scale masked diffusion LM | This work provides information-theoretic justification for its training loss |
| **Kong et al. (2023/2024)** | I-MMSE / conditional likelihood for continuous diffusion | This work is the discrete-domain generalization |
| **Zhu et al. (2025)** | Masked diffusion alignment | This work provides the theoretical foundation for likelihood-ratio estimation |

## Personal Notes

### Insights & Connections

- **Theoretical guarantee for masked diffusion LM training**: This paper directly proves that the DCE training objective of models such as LLaDA minimizes the true NLL rather than a relaxed upper bound—placing masked diffusion on equal theoretical footing with autoregressive models under MLE.
- **Alignment applications**: The coupled likelihood-ratio estimator naturally fits preference learning methods such as DPO, enabling more stable estimation of $\log \frac{p_\theta(y_w)}{p_\theta(y_l)}$.
- **Model provenance**: Combining conditional likelihood estimation with OOD detection provides a tool for data contamination detection and training data source analysis (an information-theoretic variant of membership inference).
- **Bridge between continuous and discrete**: The generalization path I-MMSE → I-MDSE/I-MDCE suggests that further information-theoretic tools from continuous diffusion (e.g., capacity, coding theorems) may also be transferable to the discrete setting.

### Significance

The central contribution of this work lies not in proposing a new model or algorithm, but in providing a deeper theoretical understanding of existing methods. It elevates DSE/DCE from "useful but theoretically underspecified training objectives" to "information-theoretically exact likelihood estimators." This theoretical advancement has lasting influence on the field: it allows researchers to confidently build subsequent theoretical analyses (e.g., convergence analysis, model comparison) on DCE without concern about variational gaps undermining theoretical soundness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First complete information-theoretic framework for discrete diffusion, upgrading variational bounds to exact equalities
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic validation + variance analysis + model auditing provide broad coverage, though large-scale quantitative evaluation is limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematically rigorous, logically clear, with natural integration of theory and experiments
- Value: ⭐⭐⭐⭐⭐ — Provides fundamental theoretical foundations for discrete diffusion models with far-reaching implications for training, evaluation, and downstream applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
