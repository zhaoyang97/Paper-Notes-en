---
title: >-
  [Paper Note] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions
description: >-
  [ICML 2026][Interpretability][attention temperature] Within the high-dimensional linear regression ICL framework, this paper adopts "approximate softmax attention"—a surrogate that preserves row-wise normalization and temperature selectivity while remaining analytically solvable—to **derive the closed-form solution for ICL generalization error and an explicit expression
tags:
  - ICML 2026
  - Interpretability
  - attention temperature
  - ICL
  - approximate softmax
date: 2026-05-08
content_hash: e9f794caad241857
---
# Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions

**Conference**: ICML 2026  
**arXiv**: [2511.01292](https://arxiv.org/abs/2511.01292)  
**Code**: Not released  
**Area**: Interpretability / In-Context Learning / Transformer Theory  
**Keywords**: attention temperature, ICL, distribution shift, high-dimensional linear regression, approximate softmax

## TL;DR
Within the high-dimensional linear regression ICL framework, this paper adopts "approximate softmax attention"—a surrogate that preserves row-wise normalization and temperature selectivity while remaining analytically solvable—to **derive the closed-form solution for ICL generalization error and an explicit expression for the optimal attention temperature** $\tau_{\text{opt}}$. It proves that correctly tuning the inference-time temperature can recover near-Bayes-optimal performance and validates this "lightweight knob" in real-world QA tasks using GPT-2 and Llama2-7B.

## Background & Motivation

**Background**: ICL is a hallmark capability of LLMs, enabling them to solve new tasks via a few examples. Prior theoretical work (Garg et al. / Zhang et al. / Raventós et al.) utilized the framework of linear attention and linear regression to demonstrate that Transformers can approximate Bayes-optimal ridge regression.

**Limitations of Prior Work**: ICL performance degrades significantly under distribution shift (e.g., changes in input covariance, task priors, or noise levels). Engineering solutions typically involve "retraining" or "data augmentation," lacking a lightweight, **inference-time adjustable** mechanism. While attention temperature $\tau$ is often ignored after being set to $\sqrt{d_k}$ in the original Transformer, empirical tuning suggests it can yield gains, though systematic theoretical analysis for ICL is absent.

**Key Challenge**: Analyzing the impact of temperature on ICL requires a model that **retains key softmax properties (normalization and temperature dependence) while remaining analytically solvable**. Pure linear attention loses temperature dependence by removing softmax, while standard softmax is mathematically intractable for closed-form high-dimensional analysis.

**Goal**: 1) Derive closed-form generalization errors for ICL under distribution shift; 2) Provide an explicit expression for the optimal temperature $\tau_{\text{opt}}$; 3) Link $\tau_{\text{opt}}$ to the moments of the distribution shift; 4) Empirically verify that temperature scaling improves ICL in LLMs.

**Key Insight**: The authors employ the **approximate softmax** from Han et al. (2024)—an analytically solvable surrogate that mimics row-wise normalization and temperature dependence of softmax. In the high-dimensional asymptotic limit $l, d \to \infty$, Isserlis' Theorem is used to calculate high-order moments, expressing the error as a quadratic rational function of $\tau$, which yields an explicit optimum.

**Core Idea**: Attention temperature serves as a "training-free lever" to correct distribution shifts at inference time. By linking temperature to the second-order moments of pre-softmax attention scores, the optimal value can be derived from a formula without requiring any fine-tuning.

## Method

### Overall Architecture
The paper investigates whether adjusting the attention temperature $\tau$ at inference time can mitigate ICL degradation under distribution shift. This is framed within an analytically solvable high-dimensional linear regression ICL task: examples $(\mathbf x_i, y_i)$ are i.i.d. following $\mathbf x \sim \mathcal{N}(\boldsymbol\mu_x, \boldsymbol\Sigma_x)$, $y = \mathbf w^\top \mathbf x + \epsilon$, and $\mathbf w \sim \mathcal{N}(\boldsymbol\mu_w, \boldsymbol\Sigma_w)$. These are formed into token embeddings $\mathbf Z = [\mathbf x_1\cdots\mathbf x_l; y_1\cdots y_{l-1}\,0]\in\mathbb R^{(d+1)\times l}$ (the last column is the query). After passing through a single-layer approximate softmax attention $\mathbf E = \mathbf Z + \mathbf V \mathbf Z\cdot\widehat{\text{softmax}}\big(\frac{(\mathbf K\mathbf Z)^\top(\mathbf Q\mathbf Z)}{\tau}\big)$, the prediction $\hat y = E_{d+1,l}$ is obtained. By reparameterizing $\mathbf V$ and $\mathbf M:=\mathbf K^\top\mathbf Q$, the analysis follows three steps: deriving the high-dimensional generalization error as a function of $\tau$, minimizing it to find $\tau_{\text{opt}}$, and using a Bayes-optimal ridge configuration to explain sub-optimality under shift.

### Key Designs

**1. Approximate softmax attention: A surrogate for closed-form analysis**

Linear attention lacks temperature entirely, while standard softmax prevents closed-form high-dimensional solutions. The authors use $\widehat{\text{softmax}}$ as a compromise: it maintains row-wise normalization ($\sum_j \widehat{\text{softmax}}_{ij}=1$) and exhibits temperature dependence nearly identical to true softmax, yet its algebraic structure allows for term-by-term calculation of high-order moments via Isserlis' Theorem. Remark 3.4 highlights that row-normalization naturally absorbs input mean shift (a property linear attention lacks), suggesting that mean shift is negligible while covariance shift is the primary disruptor.

**2. Closed-form generalization error and optimal temperature formula**

Under Assumptions 3.1 (well-conditioned data), 3.2 ($l, d \to \infty$), and 4.1 (parameter constraints), Theorem 4.2 expresses the generalization error as $\mathcal G(\mathbf V, \mathbf M) = \frac{1}{\tau^2}\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11}) - \frac{1}{\tau}\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top)) + \text{Tr}(\mathbf{AB}) + \sigma^2$, where $\mathbf A = \boldsymbol\Sigma_x + \boldsymbol\mu_x\boldsymbol\mu_x^\top$ and $\mathbf B = \boldsymbol\Sigma_w + \boldsymbol\mu_w\boldsymbol\mu_w^\top$. Setting the derivative with respect to $\tau$ to zero yields $\tau_{\text{opt}} = \frac{2\,\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11})}{\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top))}$. This formula provides the balance point between overfitting (insufficient selectivity) and signal alignment. For isotropic shifts, this simplifies into a concise expression involving the shift factor and the $l/d$ ratio.

**3. Bayes-optimal pre-training parameter comparison: Explaining the significance of $\tau_{\text{opt}}\neq 1$**

To explain the utility of tuning temperature in pre-trained models, Proposition 4.4 constructs a model (pre-trained at $\tau=1$) that simulates a Bayes-optimal ridge estimator $\hat{\mathbf w}_{\text{Bayes}} = (\frac{\bar{\mathbf X}^\top\bar{\mathbf X}}{\sigma^2} + \boldsymbol\Sigma_w^{-1})^{-1}(\frac{\bar{\mathbf X}^\top\bar{\mathbf y}}{\sigma^2} + \boldsymbol\Sigma_w^{-1}\boldsymbol\mu_w)$. The analysis shows that input mean shift is largely neutralized by normalization, and effects of task/noise shifts decay as $l\to\infty$. However, input covariance shifts fundamentally disrupt ICL, and it is precisely these shifts that temperature adjustment can remedy.

### Loss & Training
The theoretical derivation does not involve a training loss. For empirical validation, inference-time temperature scaling is applied to GPT-2 and Llama2-7B on QA tasks with distribution shifts (e.g., noisy demonstrations) without retraining, using Theorem 4.3 to estimate $\tau_{\text{opt}}$ or performing grid search.

## Key Experimental Results

### Main Results
Verified across synthetic linear regression and LLM QA:

| Setting | Original Temperature | Tuned to $\tau_{\text{opt}}$ | Gap with Bayes-optimal |
|------|----------|--------------------------|----------------------|
| No Shift ($\mathcal D^{\text{test}}=\mathcal D^{\text{train}}$) | Already optimal | Equivalent | ≈ 0 |
| Input Covariance Doubling ($\boldsymbol\Sigma_{\text{test}} = 2\boldsymbol\Sigma_{\text{train}}$) | Significant deviation | Near recovery | Greatly reduced |
| Task Covariance Doubling ($\boldsymbol\Sigma_w^{\text{test}} = 3\boldsymbol\Sigma_w^{\text{train}}$ with mean shift) | Significant deviation | Near Bayes-optimal | Greatly reduced |
| Noise Shift ($\sigma_{\text{train}}=0.1 \to \sigma_{\text{test}}=10$) | Severe degradation | Significant recovery | Significantly reduced |
| Llama2-7B / GPT-2 noisy QA | Baseline performance | Improved performance | — |

### Ablation Study

| Configuration | Observation | Explanation |
|------|------|------|
| Linear attention vs. Approximate softmax | Linear version fails mean shift robustness and lacks temperature dependence | Row-normalization is critical |
| Varying $\sigma_{\text{test}}$ and $l/d$ | $\tau_{\text{opt}}$ changes smoothly with noise and $l/d$ | High alignment with closed-form theory |
| Theorem 4.3 Analytical vs. Grid Search | Nearly identical results | The formula is reliable |

### Key Findings
- **Input mean shift is negligible** (absorbed by row-normalization), whereas **input covariance shift is the primary cause of ICL failure**; this provides a clear diagnostic hierarchy.
- As $l/d \to \infty$, the impact of task and noise shifts is absorbed by the context, but the impact of covariance shift persists and must be addressed via temperature adjustment.
- Temperature scaling is an inference-time, training-free method with zero compute overhead, making it highly practical for LLM deployment.

## Highlights & Insights
- Using "approximate softmax" as a surrogate fills the gap between overly simplistic linear attention and mathematically intractable standard softmax, offering a useful "model-for-analysis" paradigm.
- The analytical formula for $\tau_{\text{opt}}$ upgrades the heuristic of temperature scaling into a calculable optimal control problem based on data moments.
- The clear distinction between the harmlessness of mean shift and the danger of covariance shift serves as a clean guide for practitioners improving ICL robustness.

## Limitations & Future Work
- The theoretical scope is limited to **high-dimensional linear regression ICL**; extensions to non-linear tasks, multi-layer architectures, and MLP residuals remain open.
- The assumption of Gaussian inputs/tasks is a stylized approximation of natural language data.
- Practical experiments focused on GPT-2 and Llama2-7B; validation on newer models (e.g., Llama 3) is needed.
- Estimating $\tau_{\text{opt}}$ requires knowledge of test distribution moments, which remains a challenge in completely unseen domains.

## Related Work & Insights
- **vs. Zhang et al. (2024)**: This work replaces linear attention with approximate softmax to capture temperature dependence and relaxes data assumptions (beyond strict $\mathcal{N}(0, I)$).
- **vs. Veličković et al. (2025)**: While they propose adaptive temperature during training, this paper focuses on optimal inference-time temperature as a post-hoc correction.
- **vs. Han et al. (2024)**: This work applies their approximate softmax architecture specifically to the theoretical analysis of ICL under distribution shift.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical analysis of temperature in ICL using approximate softmax.
- Experimental Thoroughness: ⭐⭐⭐ Good mix of synthetic and LLM experiments, though model coverage is somewhat dated.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression through dense mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a simple, deployable inference-time tool for improving ICL robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)
- [\[ICML 2026\] GEM: Geometric Entropy Mixing for Optimal LLM Data Curation](gem_geometric_entropy_mixing_for_optimal_llm_data_curation.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICML 2026\] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)

</div>

<!-- RELATED:END -->
