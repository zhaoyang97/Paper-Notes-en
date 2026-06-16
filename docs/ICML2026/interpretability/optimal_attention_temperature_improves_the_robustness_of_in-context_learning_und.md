---
title: >-
  [Paper Note] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions
description: >-
  [ICML 2026][Interpretability][attention temperature] Under the high-dimensional linear regression ICL framework, this paper adopts an "approximate softmax attention" that preserves softmax normalization and temperature selectivity while remaining analytically solvable. **It provides a closed-form solution for ICL generalization error and an explicit expression for the op
tags:
  - ICML 2026
  - Interpretability
  - attention temperature
  - ICL
  - approximate softmax
date: 2026-05-08
content_hash: c0ba31502922ea1d
---
# Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions

**Conference**: ICML 2026  
**arXiv**: [2511.01292](https://arxiv.org/abs/2511.01292)  
**Code**: Not released  
**Area**: Interpretability / In-Context Learning / Transformer Theory  
**Keywords**: attention temperature, ICL, distribution shift, high-dimensional linear regression, approximate softmax

## TL;DR
Under the high-dimensional linear regression ICL framework, this paper adopts an "approximate softmax attention" that preserves softmax normalization and temperature selectivity while remaining analytically solvable. **It provides a closed-form solution for ICL generalization error and an explicit expression for the optimal attention temperature** $\tau_{\text{opt}}$, proving that tuning the inference-time temperature alone can recover near Bayes-optimal performance. The effectiveness of this "lightweight knob" is also validated in real-world QA tasks using GPT-2 and Llama2-7B.

## Background & Motivation

**Background**: ICL is one of the most remarkable capabilities of LLMs—solving new tasks given a few examples. The community has utilized the clean toy setup of linear attention and linear regression (Garg et al. / Zhang et al. / Raventós et al.) to prove that Transformers can approximate Bayes-optimal ridge regression.

**Limitations of Prior Work**: ICL performance degrades significantly under distribution shift (e.g., changes in input covariance, shifted task priors, or increased noise). Engineering mitigations mostly involve "retraining" or "adding data," lacking a lightweight **inference-time adjustable** knob. Attention temperature $\tau$ has been largely ignored after being set to $\sqrt{d_k}$ in the original Transformer; while some have tuned it to gain marginal improvements, a systematic theoretical analysis of its role in ICL is missing.

**Key Challenge**: Analyzing the impact of temperature on ICL requires a model that **retains key softmax properties (normalization + selective temperature dependence) while remaining analytically solvable**. Pure linear attention removes softmax, thus losing temperature dependence, while standard softmax is intractable for closed-form high-dimensional analysis.

**Goal**: 1) Derive a closed-form generalization error for ICL under distribution shift; 2) Provide an explicit expression for the optimal temperature $\tau_{\text{opt}}$; 3) Link $\tau_{\text{opt}}$ to the moments of the distribution shift; 4) Empirically demonstrate that temperature scaling can remedy ICL in LLMs.

**Key Insight**: Borrowing the **approximate softmax** from Han et al. (2024)—an analytically tractable surrogate that maintains row-wise normalization and a temperature dependence highly similar to true softmax. In the high-dimensional asymptotic limit $l, d \to \infty$, Isserlis' theorem is used to calculate high-order moments, expressing the error as a quadratic rational function of $\tau$, allowing for an explicit solution of the optimal point.

**Core Idea**: Attention temperature serves as a "training-free lever" to correct distribution shifts at inference time. By linking it to the second-order moments of pre-softmax attention scores, the optimal value can be derived from a single formula without any fine-tuning.

## Method

### Overall Architecture
The paper addresses whether tuning the inference-time attention temperature $\tau$ can restore ICL performance degraded by distribution shift. This is explored within an analytically tractable toy model of high-dimensional linear regression ICL: examples $(\mathbf x_i, y_i)$ are i.i.d. with $\mathbf x \sim \mathcal{N}(\boldsymbol\mu_x, \boldsymbol\Sigma_x)$, $y = \mathbf w^\top \mathbf x + \epsilon$, and $\mathbf w \sim \mathcal{N}(\boldsymbol\mu_w, \boldsymbol\Sigma_w)$. These are concatenated into token embeddings $\mathbf Z = [\mathbf x_1\cdots\mathbf x_l; y_1\cdots y_{l-1}\,0]\in\mathbb R^{(d+1)\times l}$ (the last column is the query with a missing label), passed through a single layer of approximate softmax attention $\mathbf E = \mathbf Z + \mathbf V \mathbf Z\cdot\widehat{\text{softmax}}\big(\frac{(\mathbf K\mathbf Z)^\top(\mathbf Q\mathbf Z)}{\tau}\big)$, and the prediction is read as $\hat y = E_{d+1,l}$. By reparameterizing $\mathbf V$ and $\mathbf M:=\mathbf K^\top\mathbf Q$ based on their roles (where only $\mathbf{v}_{21}, v_{22}, \mathbf{m}_{21}, \mathbf M_{11}$ truly affect predictions), the analysis proceeds in three steps: deriving the closed-form generalization error in the high-dimensional limit, minimizing it to find $\tau_{\text{opt}}$, and finally using a configuration that mimics Bayes-optimal ridge regression to explain why ICL becomes sub-optimal under shift without temperature adjustment.

### Key Designs

**1. Approximate softmax attention: Creating a softmax surrogate for closed-form analysis**

Pure linear attention removes the softmax entirely, losing the temperature variable, while standard softmax avoids closed-form expressions in high dimensions. This paper adopts $\widehat{\text{softmax}}$ from Han et al. (2024) as a compromise: it remains row-wise normalized ($\sum_j \widehat{\text{softmax}}_{ij}=1$), and its temperature dependence (dividing inputs by $\tau$) nearly overlaps with true softmax (as shown in the histogram comparison in Figure 1). However, its algebraic form is simple enough to allow term-by-term calculation of high-order moments under Gaussian inputs using Isserlis' theorem. This step is critical because, as noted in Remark 3.4, row-normalization naturally absorbs input mean shift—a property linear attention lacks—which justifies the model choice and foreshadows the conclusion that covariance shift, rather than mean shift, is the primary performance killer.

**2. Closed-form generalization error and optimal temperature formula: Turning temperature tuning into solvable optimal control**

Under Assumptions 3.1 (bounded and well-conditioned data), 3.2 ($l, d \to \infty$), and 4.1 (parameter norm constraints), Theorem 4.2 calculates the generalization error as $\mathcal G(\mathbf V, \mathbf M) = \frac{1}{\tau^2}\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11}) - \frac{1}{\tau}\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top)) + \text{Tr}(\mathbf{AB}) + \sigma^2$, where $\mathbf A = \boldsymbol\Sigma_x + \boldsymbol\mu_x\boldsymbol\mu_x^\top$ and $\mathbf B = \boldsymbol\Sigma_w + \boldsymbol\mu_w\boldsymbol\mu_w^\top$, and $\mathbf F_1, \mathbf F_2$ are matrices depending only on the test distribution and parameters. This is a quadratic rational function of $\tau$. Setting the derivative with respect to $\tau$ to zero yields the explicit formula in Theorem 4.3: $\tau_{\text{opt}} = \frac{2\,\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11})}{\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top))}$.

The closed-form provides two benefits. First, interpretability: the numerator corresponds to an "overfitting term when selectivity is too weak," while the denominator corresponds to a "signal alignment term"; the optimal temperature is the equilibrium. Second, applicability: under isotropic shift, the formula simplifies into a concise expression involving only $a, b, \sigma, l/d$, which can be directly read from the moments of the data shift.

**3. Bayes-optimal pre-trained parameter comparison: Explaining why $\tau_{\text{opt}}\neq 1$ is meaningful**

The paper explicitly constructs a model with pre-training temperature $\tau=1$ to simulate the Bayes-optimal ridge estimator $\hat{\mathbf w}_{\text{Bayes}} = (\frac{\bar{\mathbf X}^\top\bar{\mathbf X}}{\sigma^2} + \boldsymbol\Sigma_w^{-1})^{-1}(\frac{\bar{\mathbf X}^\top\bar{\mathbf y}}{\sigma^2} + \boldsymbol\Sigma_w^{-1}\boldsymbol\mu_w)$. By anchoring the model to this clean baseline, the authors decompose the impacts of different shifts: input mean shift is harmlessly absorbed; input covariance shift is destructive because $\mathbf M_{11}$ is fitted to the training covariance; task/noise shifts decay as $l\to\infty$. The conclusion is that only covariance-type shifts truly break ICL, and precisely this type of shift can be mitigated by temperature adjustment.

### Loss & Training
The theoretical part does not involve training loss. The empirical part targets QA tasks with distribution shift caused by noisy in-context demonstrations using GPT-2 and Llama2-7B. It applies inference-time scaling to attention temperature (without retraining), estimating $\tau_{\text{opt}}$ via Theorem 4.3 or via grid search in its vicinity.

## Key Experimental Results

### Main Results
Validated on both synthetic linear regression and LLM QA:

| Setup | Without Tuning | Tuned to $\tau_{\text{opt}}$ | Gap vs Bayes-optimal |
|------|----------|--------------------------|----------------------|
| No shift ($\mathcal D^{\text{test}}=\mathcal D^{\text{train}}$) | Already optimal | Identical | ≈ 0 |
| Input Covariance doubled ($\boldsymbol\Sigma_{\text{test}} = 2\boldsymbol\Sigma_{\text{train}}$) | Significant deviation | Nearly recovered | Greatly reduced |
| Task Covariance doubled ($\boldsymbol\Sigma_w^{\text{test}} = 3\boldsymbol\Sigma_w^{\text{train}}$ with mean shift) | Significant deviation | Near Bayes-optimal | Greatly reduced |
| Noise shift ($\sigma_{\text{train}}=0.1 \to \sigma_{\text{test}}=10$) | Severe degradation | Significant recovery, converges with $l/d$ | Significantly reduced |
| Llama2-7B / GPT-2 noisy QA | Baseline performance | Improved | — |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Linear attention vs. Approx softmax | Linear version is not robust to mean shift and lacks temperature dependence | Row-normalization is key |
| Adjusting $\sigma_{\text{test}}$ and $l/d$ | $\tau_{\text{opt}}$ varies smoothly with noise and $l/d$ | High alignment between closed-form and simulation |
| Theorem 4.3 analytical estimate vs. grid search | Nearly overlapping | The formula is reliable |

### Key Findings
- **Input mean shift is inconsequential** (absorbed by row-wise normalization), whereas **input covariance shift is the true killer of ICL**; this provides a clear priority for robustness research.
- While the impacts of task and noise shifts are gradually absorbed by a larger context as $l/d\to\infty$, the impact of covariance shift persists—it must be addressed via temperature adjustment.
- Temperature adjustment is an inference-time, training-free method with near-zero overhead, making it highly practical for real-world LLM deployment.

## Highlights & Insights
- Successfully bridged the gap between "too weak linear attention" and "intractable standard softmax" using the approximate softmax model—a model-for-analysis paradigm that warrants further promotion in Transformer theory.
- The analytical formula for $\tau_{\text{opt}}$ upgrades the empirical knowledge of "why temperature scaling works" to a computable optimal control problem estimable from data moments.
- The binary diagnosis of input mean shift vs. covariance shift is a clean and useful guideline: check if the covariance has truly changed before deciding to tune the temperature.

## Limitations & Future Work
- The theoretical analysis is built on the simplified axis of **linear regression ICL**; extensions to non-linearities, multi-layer Transformers, multi-head attention, and MLP residuals remain open.
- The assumption of Gaussian inputs and tasks is only a stylized approximation of real LLM text; while LLM QA experiments provide empirical support, theoretical guarantees are not yet present.
- Empirical validation was limited to GPT-2 and Llama2-7B; whether newer models (e.g., Llama3) benefit similarly or if optimal temperature estimation remains accurate is unverified.
- Estimating $\tau_{\text{opt}}$ requires test distribution moments; approximating these moments in a completely unseen domain remains an open question.

## Related Work & Insights
- **vs. Zhang et al. (2024) Linear Attention ICL Theory**: This work replaces linear attention with approximate softmax, captures temperature dependence, and relaxes analysis assumptions (not requiring strict $\mathcal N(0, I)$); the theory is closer to actual softmax behavior.
- **vs. Veličković et al. (2025) Adaptive Temperature**: They propose adaptive temperature during training; this paper focuses on inference-time closed-form optimal temperature, which can serve as a post-hoc correction for their method.
- **vs. Han et al. (2024) Approximate Softmax**: This paper adopts their architecture but is the first to use it for theoretical analysis of ICL under distribution shift.
- **vs. Empirical Temperature Scaling (Lin, Peng, Zou)**: This paper provides a unified theory for "why/when/to what value" to tune, connecting scattered heuristics.

## Rating
- Novelty: ⭐⭐⭐⭐ First use of approximate softmax for theoretical ICL temperature analysis.
- Experimental Thoroughness: ⭐⭐⭐ Includes both synthetic and LLM QA, but the LLM models are relatively old with narrow coverage.
- Writing Quality: ⭐⭐⭐⭐ Densely derived but logically clear; the appendix provides complete proofs.
- Value: ⭐⭐⭐⭐ Provides a simple, deployable inference-time tool for ICL robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)
- [\[ICML 2026\] GEM: Geometric Entropy Mixing for Optimal LLM Data Curation](gem_geometric_entropy_mixing_for_optimal_llm_data_curation.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](singular_vectors_of_attention_heads_align_with_features.md)

</div>

<!-- RELATED:END -->
