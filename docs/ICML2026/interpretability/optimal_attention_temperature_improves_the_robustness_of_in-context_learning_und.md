---
title: >-
  [Paper Note] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions
description: >-
  [ICML 2026][Interpretability][attention temperature] Within the high-dimensional linear regression ICL framework, this work employs an "approximate softmax attention" that preserves softmax normalization and temperature…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "attention temperature"
  - "ICL"
  - "distribution shift"
  - "high-dimensional linear regression"
  - "approximate softmax"
date: 2026-05-08
content_hash: d2f4ea9b75dd1ed8
---

# Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions

**Conference**: ICML 2026  
**arXiv**: [2511.01292](https://arxiv.org/abs/2511.01292)  
**Code**: Not released  
**Area**: Interpretability / In-Context Learning / Transformer Theory  
**Keywords**: attention temperature, ICL, distribution shift, high-dimensional linear regression, approximate softmax

## TL;DR
Within the high-dimensional linear regression ICL framework, this work employs an "approximate softmax attention" that preserves softmax normalization and temperature selectivity while remaining analytically tractable, **deriving a closed-form solution for ICL generalization error and an explicit formula for the optimal attention temperature** $\tau_{\text{opt}}$. It is proven that simply tuning the temperature at inference can recover near Bayes-optimal performance. The effectiveness of this "lightweight knob" is also validated on real QA tasks with GPT-2 and Llama2-7B.

## Background & Motivation

**Background**: ICL is one of the most remarkable capabilities of LLMs—solving new tasks with just a few examples. Prior work (Garg et al. / Zhang et al. / Raventós et al.) has shown, via linear attention + linear regression, that Transformers can approximate Bayes-optimal ridge regression.

**Limitations of Prior Work**: ICL performance degrades severely under distribution shift (e.g., changes in input covariance, task prior, or increased noise). Engineering solutions typically involve "retraining" or "adding data," lacking a **lightweight, inference-time tunable knob**. The attention temperature $\tau$, originally set to $\sqrt{d_k}$ in Transformers, has been largely ignored; sporadic tuning shows minor gains, but no systematic theoretical analysis exists for ICL.

**Key Challenge**: To analyze the effect of temperature on ICL, a model is needed that **retains key softmax properties (normalization + selective temperature dependence) and is analytically solvable**. Pure linear attention removes softmax, losing temperature dependence; standard softmax is analytically intractable in high dimensions.

**Goal**: 1) Derive a closed-form expression for ICL generalization error under distribution shift; 2) Provide an explicit formula for the optimal temperature $\tau_{\text{opt}}$; 3) Link $\tau_{\text{opt}}$ to moments of the distribution shift; 4) Empirically demonstrate that temperature scaling can remedy ICL in LLMs.

**Key Insight**: The approach leverages Han et al. (2024)'s **approximate softmax**—an analytically tractable surrogate that preserves row-wise normalization and temperature dependence, closely mimicking softmax. In the high-dimensional asymptotic regime $l, d \to \infty$, Isserlis' theorem is used to compute higher-order moments, expressing the error as a quadratic rational function of $\tau$, with the optimum explicitly solvable.

**Core Idea**: Attention temperature serves as an "training-free lever" for correcting distribution shift at inference—by linking it to the second-order moments of pre-softmax attention scores, the optimal value can be computed from a single formula, eliminating the need for any fine-tuning.

## Method

### Overall Architecture
Model setup: Linear regression ICL task, $(\mathbf x_i, y_i)$ i.i.d. with $\mathbf x \sim \mathcal{N}(\boldsymbol\mu_x, \boldsymbol\Sigma_x)$, $y = \mathbf w^\top \mathbf x + \epsilon$, $\mathbf w \sim \mathcal{N}(\boldsymbol\mu_w, \boldsymbol\Sigma_w)$. Token embeddings are $\mathbf Z = [\mathbf x_1\cdots\mathbf x_l; y_1\cdots y_{l-1}\,0]\in\mathbb R^{(d+1)\times l}$ (last column is the query, label missing). One layer of approximate softmax attention: $\mathbf E = \mathbf Z + \mathbf V \mathbf Z\cdot\widehat{\text{softmax}}\big(\frac{(\mathbf K\mathbf Z)^\top(\mathbf Q\mathbf Z)}{\tau}\big)$, prediction $\hat y = E_{d+1,l}$. By reparameterizing $\mathbf V, \mathbf M:=\mathbf K^\top\mathbf Q$ (only $\mathbf v_{21}, v_{22}, \mathbf m_{21}, \mathbf M_{11}$ affect prediction), analytic derivation is possible. Three steps: (1) Under well-behaved data assumptions and high-dimensional limit, derive closed-form generalization error; (2) Minimize with respect to $\tau$ to obtain $\tau_{\text{opt}}$; (3) Use Bayes-optimal ridge parameter configuration (Proposition 4.4) to explain why unadjusted $\tau$ becomes suboptimal under shift.

### Key Designs

1. **Approximate Softmax Attention**:

    - **Function**: Preserves softmax normalization and temperature dependence, but is analytically tractable.
    - **Mechanism**: Replaces standard softmax with $\widehat{\text{softmax}}$—row-normalized ($\sum_j \widehat{\text{softmax}}_{ij}=1$), with input dependence on $\tau$ nearly identical to softmax (see Figure 1 histogram comparison), but algebraically enables computation of higher-order moments under Gaussian input via Isserlis' formula. Remark 3.4 emphasizes: row normalization makes the model inherently robust to input mean shift (unlike linear attention), which is a key reason for model selection.
    - **Design Motivation**: Linear attention is too weak (no temperature), standard softmax is too rigid (no closed-form); approximate softmax bridges both—this "analysis-oriented surrogate model" is a common paradigm in high-dimensional statistical ML theory.

2. **Closed-form Generalization Error & Optimal Temperature Formula**:

    - **Function**: Provides an analytic relationship between $\tau$ and ICL error, from which the optimal temperature is derived.
    - **Mechanism**: Under Assumptions 3.1 (bounded, well-conditioned data), 3.2 ($l, d \to \infty$), 4.1 (parameter norm constraints), Theorem 4.2 gives
      $\mathcal G(\mathbf V, \mathbf M) = \frac{1}{\tau^2}\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11}) - \frac{1}{\tau}\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top)) + \text{Tr}(\mathbf{AB}) + \sigma^2$,
      where $\mathbf A = \boldsymbol\Sigma_x + \boldsymbol\mu_x\boldsymbol\mu_x^\top$, $\mathbf B = \boldsymbol\Sigma_w + \boldsymbol\mu_w\boldsymbol\mu_w^\top$, and $\mathbf F_1, \mathbf F_2$ are matrices depending only on the test distribution and parameters. This is a quadratic rational function in $\tau$; differentiating yields Theorem 4.3:
      $\tau_{\text{opt}} = \frac{2\,\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11})}{\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top))}$.
      The closed-form has two advantages: (a) Interpretability—the numerator corresponds to "overfitting when selectivity is too weak," the denominator to "signal alignment"; (b) Under isotropic shift: if training uses $\mathcal N(0, I)$, test input variance is scaled by $a$, task variance by $b$, noise by $\sigma$, then $\tau_{\text{opt}}$ reduces to a simple formula in $a, b, \sigma, l/d$, allowing direct estimation from data shift moments.
    - **Design Motivation**: Elevates "temperature tuning for improvement" from engineering heuristic to provable optimal control; also provides criteria for "when temperature scaling truly restores Bayes optimality" (Corollary-type results), avoiding blind application.

3. **Bayes-optimal Pretraining Parameter Correspondence (Proposition 4.4)**:

    - **Function**: Explains why the "native temperature $\tau=1$" becomes suboptimal under distribution shift, making $\tau_{\text{opt}} \ne 1$ both valid and meaningful.
    - **Mechanism**: When $\tau$ is set to 1 during pretraining, the authors explicitly construct $(\mathbf M_{11}, \mathbf v_{21}, v_{22}, \mathbf m_{21})$ to simulate the Bayes-optimal ridge estimator $\hat{\mathbf w}_{\text{Bayes}} = (\frac{\bar{\mathbf X}^\top\bar{\mathbf X}}{\sigma^2} + \boldsymbol\Sigma_w^{-1})^{-1}(\frac{\bar{\mathbf X}^\top\bar{\mathbf y}}{\sigma^2} + \boldsymbol\Sigma_w^{-1}\boldsymbol\mu_w)$. This anchors the pretrained model to a clean baseline. Then, three types of shift are analyzed—input mean shift (absorbed by centering), input covariance shift (training covariance is used by $\mathbf M_{11}$, thus broken), task/noise shift (impact diminishes as $l\to\infty$)—showing that only covariance-type shift truly breaks ICL, and this is precisely the shift that temperature adjustment can mitigate.
    - **Design Motivation**: Links "theoretically optimal temperature" with "actual ICL behavior of pretrained models," making $\tau_{\text{opt}}$ not just a mathematical exercise but a deployment guideline.

### Loss & Training
The theoretical part does not involve training loss; empirically, for GPT-2 and Llama2-7B on QA tasks with distribution shift induced by noisy in-context demonstrations, attention temperature is scaled at inference (no retraining), with $\tau_{\text{opt}}$ estimated via Theorem 4.3 or nearby grid search.

## Key Experimental Results

### Main Results
Validation on both synthetic linear regression and LLM QA:

| Setting | No Temperature Tuning | Tuned to $\tau_{\text{opt}}$ | Gap to Bayes-optimal |
|---------|----------------------|------------------------------|----------------------|
| No shift ($\mathcal D^{\text{test}}=\mathcal D^{\text{train}}$) | Already optimal | Identical | ≈ 0 |
| Input covariance doubled ($\boldsymbol\Sigma_{\text{test}} = 2\boldsymbol\Sigma_{\text{train}}$) | Significant deviation | Nearly recovered | Greatly reduced |
| Task covariance tripled ($\boldsymbol\Sigma_w^{\text{test}} = 3\boldsymbol\Sigma_w^{\text{train}}$, mean shifted) | Significant deviation | Approaches Bayes-optimal | Greatly reduced |
| Noise shift ($\sigma_{\text{train}}=0.1 \to \sigma_{\text{test}}=10$) | Severe degradation | Significant recovery, further convergence as $l/d$ increases | Significantly reduced |
| Llama2-7B / GPT-2 noisy QA | Baseline performance | Improved | — |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---------------|------------|-------------|
| Linear attention vs approximate softmax | Linear version not robust to mean shift, cannot capture temperature dependence | Row normalization is key |
| Varying $\sigma_{\text{test}}$ and $l/d$ | $\tau_{\text{opt}}$ changes smoothly with noise and $l/d$ | Closed-form matches simulation closely |
| Theorem 4.3 analytic estimate vs grid search | Nearly identical | Formula is reliable |

### Key Findings
- **Input mean shift is harmless** (absorbed by row-wise normalization), **input covariance shift is the true ICL killer**; this provides a clear warning priority for the community.
- As $l/d\to\infty$, the effects of task and noise shift are gradually absorbed by the large context, but the impact of covariance shift persists—it must be addressed by temperature adjustment.
- Temperature adjustment is an inference-time, training-free method with negligible parameter and compute overhead—highly practical for real LLM deployment.

## Highlights & Insights
- The use of approximate softmax as an "analysis-oriented tool" successfully bridges the gap between "linear attention too weak, standard softmax intractable"—this model-for-analysis design paradigm is worth further promotion in transformer theory.
- The analytic formula for $\tau_{\text{opt}}$ elevates the engineering intuition of "why temperature scaling sometimes works" to a computable optimal control problem, estimable from data moments—directly applicable to LLM deployment.
- The dichotomous diagnosis of input mean shift vs covariance shift is a clean and practical guideline: check if covariance has truly changed before deciding to tune temperature.

## Limitations & Future Work
- The theoretical analysis is built on the **linear regression ICL** axis; extension to nonlinear, multi-layer Transformers, multi-head attention, MLP residuals, etc., remains open—appendix provides a rough sketch but lacks rigorous proof.
- Assumes both input and task are Gaussian, which is only a stylized approximation for real LLM text input; empirical support is provided via LLM QA experiments, but theoretical guarantees are lacking.
- Empirical validation is limited to GPT-2 / Llama2-7B; whether newer models (Llama3, Qwen3) also benefit, and whether optimal temperature estimation remains accurate, is unverified.
- Estimating $\tau_{\text{opt}}$ requires moments of the test distribution; how to approximate these moments in completely unseen domains remains an open question.

## Related Work & Insights
- **vs Zhang et al. (2024) linear attention ICL theory**: This work uses approximate softmax instead of linear attention, capturing temperature dependence and relaxing analysis assumptions (no strict $\mathcal N(0, I)$ required); theory is closer to actual softmax behavior.
- **vs Veličković et al. (2025) adaptive temperature**: They propose adaptive temperature during training; this work focuses on closed-form optimal temperature at inference, which can serve as a post-hoc correction for their method.
- **vs Han et al. (2024) approximate softmax**: This work directly adopts their architecture but is the first to apply it to ICL + distribution shift theoretical analysis.
- **vs empirical temperature scaling works (Lin, Peng, Zou)**: This work provides a unified theory for "why/when/how much to tune," connecting scattered empirical findings.

## Rating
- Novelty: ⭐⭐⭐⭐ First to use approximate softmax for ICL temperature theory analysis.
- Experimental Thoroughness: ⭐⭐⭐ Includes both synthetic and LLM QA experiments, but LLM models are somewhat outdated and coverage is narrow.
- Writing Quality: ⭐⭐⭐⭐ Dense derivations but clear logic, with full proofs in the appendix.
- Value: ⭐⭐⭐⭐ Provides a simple, deployable inference-time tool for ICL robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICML 2026\] The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)
- [\[ICML 2025\] Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts](../../ICML2025/interpretability/avoiding_leakage_poisoning_concept_interventions_under_distribution_shifts.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](../../AAAI2026/interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)

</div>

<!-- RELATED:END -->
