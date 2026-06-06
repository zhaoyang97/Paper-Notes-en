---
title: >-
  [Paper Note] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions
description: >-
  [ICML 2026][Interpretability][attention temperature] Under the high-dimensional linear regression ICL framework, this paper adopts an "approximate softmax attention"—which maintains softmax normalization and temperature…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "attention temperature"
  - "ICL"
  - "distribution shift"
  - "high-dimensional linear regression"
  - "approximate softmax"
date: 2026-05-08
content_hash: c0043ddc1c8e36bb
---

# Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions

**Conference**: ICML 2026  
**arXiv**: [2511.01292](https://arxiv.org/abs/2511.01292)  
**Code**: Not released  
**Area**: Interpretability / In-Context Learning / Transformer Theory  
**Keywords**: attention temperature, ICL, distribution shift, high-dimensional linear regression, approximate softmax

## TL;DR
Under the high-dimensional linear regression ICL framework, this paper adopts an "approximate softmax attention"—which maintains softmax normalization and temperature selectivity while being analytically tractable—to **derive the closed-form solution for ICL generalization error and the explicit expression for the optimal attention temperature** $\tau_{\text{opt}}$. It proves that correctly tuning the temperature at inference time can recover near Bayes-optimal performance; the effectiveness of this "lightweight knob" is also verified in real-world QA tasks using GPT-2 and Llama2-7B.

## Background & Motivation

**Background**: ICL is one of the most remarkable capabilities of LLMs—solving new tasks given a few examples. The community has used the clean "linear attention + linear regression" toy model (Garg et al. / Zhang et al. / Raventós et al.) to prove that Transformers can approximate Bayes-optimal ridge regression.

**Limitations of Prior Work**: ICL performance degrades severely under distribution shift (changes in input covariance, task priors, or noise levels). Current engineering solutions mostly involve "retraining" or "data augmentation," lacking a lightweight knob **adjustable at inference time**. Attention temperature $\tau$ has been largely overlooked after being set to $\sqrt{d_k}$ in the original Transformer; while some have tuned it to gain marginal improvements, a systematic theoretical analysis for ICL is absent.

**Key Challenge**: Analyzing the impact of temperature on ICL requires a model that **retains key softmax properties (normalization + selective temperature dependence) while being analytically tractable**. Pure linear attention removes softmax and thus loses temperature dependence; standard softmax is too complex for closed-form high-dimensional analysis.

**Goal**: 1) Derive the closed-form generalization error of ICL under distribution shift; 2) Provide an explicit expression for the optimal temperature $\tau_{\text{opt}}$; 3) Link $\tau_{\text{opt}}$ to the moments of the distribution shift; 4) Empirically validate that temperature scaling can remedy ICL performance in LLMs.

**Key Insight**: The paper leverages the **approximate softmax** from Han et al. (2024)—an analytical surrogate that preserves row-wise normalization and temperature dependence highly similar to softmax. Under the high-dimensional asymptotic limit $l, d \to \infty$, the authors use Isserlis' Theorem to calculate higher-order moments, expressing the error as a quadratic rational function of $\tau$, allowing for an explicit derivation of the optimal point.

**Core Idea**: Attention temperature serves as a "training-free lever" to correct distribution shifts at inference time. By linking it to the second-order moments of pre-softmax attention scores, the optimal value can be obtained from a single formula without any fine-tuning.

## Method

### Overall Architecture
Model Setup: A linear regression ICL task where $(\mathbf x_i, y_i)$ are i.i.d. follows $\mathbf x \sim \mathcal{N}(\boldsymbol\mu_x, \boldsymbol\Sigma_x)$, $y = \mathbf w^\top \mathbf x + \epsilon$, and $\mathbf w \sim \mathcal{N}(\boldsymbol\mu_w, \boldsymbol\Sigma_w)$. Token embeddings are $\mathbf Z = [\mathbf x_1\cdots\mathbf x_l; y_1\cdots y_{l-1}\,0]\in\mathbb R^{(d+1)\times l}$ (the last column is the query with a missing label). A single-layer approximate softmax attention is defined as $\mathbf E = \mathbf Z + \mathbf V \mathbf Z\cdot\widehat{\text{softmax}}\big(\frac{(\mathbf K\mathbf Z)^\top(\mathbf Q\mathbf Z)}{\tau}\big)$, predicting $\hat y = E_{d+1,l}$. The parameters $\mathbf V, \mathbf M:=\mathbf K^\top\mathbf Q$ are reparameterized by their roles. The logic follows three steps: (1) Derive the closed-form generalization error under well-behaved data assumptions and high-dimensional limits; (2) Minimize with respect to $\tau$ to obtain $\tau_{\text{opt}}$; (3) Use the parameter configuration corresponding to Bayes-optimal ridge (Proposition 4.4) to explain why unadjusted $\tau$ becomes suboptimal under shift.

### Key Designs

1.  **Approximate Softmax Attention**:
    - **Function**: Retains the normalization and temperature dependence of softmax while remaining analytically tractable.
    - **Mechanism**: Replaces standard softmax with $\widehat{\text{softmax}}$, which is row-wise normalized ($\sum_j \widehat{\text{softmax}}_{ij}=1$). Its dependence on the input divided by $\tau$ is nearly identical to softmax, but its algebraic form allows for the calculation of higher-order moments using Isserlis' formula under Gaussian inputs. Remark 3.4 emphasizes that row normalization makes the model naturally robust to input mean shift (a property linear attention lacks).
    - **Design Motivation**: Linear attention is too weak (no temperature), and standard softmax is too complex (no closed-form). Approximate softmax bridges the two—this "surrogate model designed for analysis" is a common paradigm in high-dimensional statistical ML theory.

2.  **Closed-form Generalization Error and Optimal Temperature Formula**:
    - **Function**: Provide the analytical relationship between $\tau$ and ICL error to identify the optimal temperature.
    - **Mechanism**: Under Assumptions 3.1 (well-conditioned data), 3.2 ($l, d \to \infty$), and 4.1 (parameter norm constraints), Theorem 4.2 gives:
      $\mathcal G(\mathbf V, \mathbf M) = \frac{1}{\tau^2}\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11}) - \frac{1}{\tau}\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top)) + \text{Tr}(\mathbf{AB}) + \sigma^2$
      where $\mathbf A = \boldsymbol\Sigma_x + \boldsymbol\mu_x\boldsymbol\mu_x^\top$ and $\mathbf B = \boldsymbol\Sigma_w + \boldsymbol\mu_w\boldsymbol\mu_w^\top$. This is a quadratic rational in $\tau$. Differentiating with respect to $\tau$ yields Theorem 4.3:
      $\tau_{\text{opt}} = \frac{2\,\text{Tr}(\mathbf A\mathbf M_{11}^\top \mathbf F_1\mathbf M_{11})}{\text{Tr}(\mathbf A(\mathbf F_2\mathbf M_{11} + \mathbf M_{11}^\top \mathbf F_2^\top))}$.
    - **Design Motivation**: Upgrades "temperature tuning" from an engineering heuristic to provable optimal control.

3.  **Bayes-optimal Pretraining Parameter Comparison (Proposition 4.4)**:
    - **Function**: Explains why the "native temperature $\tau=1$" is suboptimal under distribution shift.
    - **Mechanism**: The authors construct parameters that simulate the Bayes-optimal ridge estimator $\hat{\mathbf w}_{\text{Bayes}}$ when $\tau=1$ during pretraining. By analyzing three types of shifts—input mean shift (absorbed by centering), input covariance shift (which disrupts the $\mathbf M_{11}$ fitted to the training covariance), and task/noise shift—they show that covariance shifts genuinely break ICL, and these are precisely the shifts mitigatable by temperature adjustment.
    - **Design Motivation**: Links the theoretical optimal temperature to the actual ICL behavior of pretrained models, making $\tau_{\text{opt}}$ a deployment guide rather than just a mathematical exercise.

### Loss & Training
The theoretical section does not involve a training loss. In the empirical section, inference-time scaling (no retraining) is performed on GPT-2 and Llama2-7B for QA tasks with distribution shifts caused by noisy in-context demonstrations. $\tau_{\text{opt}}$ is estimated using the form of Theorem 4.3 or via grid search.

## Key Experimental Results

### Main Results
Validated on both synthetic linear regression and LLM QA:

| Setting | Unadjusted Temp | Adjusted to $\tau_{\text{opt}}$ | Gap with Bayes-optimal |
| :--- | :--- | :--- | :--- |
| No shift ($\mathcal D^{\text{test}}=\mathcal D^{\text{train}}$) | Already optimal | Equivalent | ≈ 0 |
| Input Covariance Doubled ($\boldsymbol\Sigma_{\text{test}} = 2\boldsymbol\Sigma_{\text{train}}$) | Significant deviation | Nearly recovered | Greatly reduced |
| Task Covariance Doubled ($\boldsymbol\Sigma_w^{\text{test}} = 3\boldsymbol\Sigma_w^{\text{train}}$ + mean shift) | Significant deviation | Near Bayes-optimal | Greatly reduced |
| Noise shift ($\sigma_{\text{train}}=0.1 \to \sigma_{\text{test}}=10$) | Severe degradation | Significant recovery | Significantly reduced |
| Llama2-7B / GPT-2 noisy QA | Baseline performance | Improved | — |

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Linear attention vs. Approx softmax | Linear version is not robust to mean shift and lacks temperature dependence | Row normalization is key |
| Adjusting $\sigma_{\text{test}}$ and $l/d$ | $\tau_{\text{opt}}$ varies smoothly with noise and $l/d$ | High agreement between closed-form and simulation |
| Theorem 4.3 Estimation vs. Grid Search | Nearly identical | The formula is reliable |

### Key Findings
- **Input mean shift is harmless** (absorbed by row-wise normalization), while **input covariance shift is the true killer of ICL**; this provides a clear priority for robustness engineering.
- As $l/d \to \infty$, the effects of task and noise shifts are gradually absorbed by the large context, but covariance shift persists—it must be resolved via temperature adjustment.
- Temperature adjustment is an inference-time, training-free method with near-zero parameter and compute overhead, offering significant practical value for LLM deployment.

## Highlights & Insights
- Using "approximate softmax" as an analytical tool successfully fills the gap between pure linear attention and standard softmax, a paradigm that should be further promoted in Transformer theory.
- The analytical formula for $\tau_{\text{opt}}$ elevates the engineering heuristic of "temperature scaling" to a calculable optimal control problem that can be estimated from data moments.
- The diagonal diagnosis between input mean shift and covariance shift is a clean, useful practical guide: check if the covariance has actually changed before deciding to tune the temperature.

## Limitations & Future Work
- The theoretical analysis is based on the simplified **linear regression ICL** axis; extensions to non-linearities, multi-layer Transformers, multi-head attention, and MLP residuals remain open.
- The assumption of Gaussian inputs and tasks is only a stylized approximation of real LLM text; empirical support is provided via QA experiments, but theoretical guarantees are not yet extended.
- Empirical validation was limited to GPT-2 and Llama2-7B; whether newer models (Llama 3, Qwen 3) benefit similarly or if optimal temperature estimates remain accurate is unverified.
- Estimating $\tau_{\text{opt}}$ requires testing distribution moments; how to approximate these in completely unseen domains remains an open question.

## Related Work & Insights
- **vs. Zhang et al. (2024) Linear attention ICL theory**: This paper uses approximate softmax to capture temperature dependence and relaxes analytical assumptions (not requiring strict $\mathcal N(0, I)$).
- **vs. Veličković et al. (2025) Adaptive temperature**: They propose adaptive temperature at training time; this paper focuses on inference-time closed-form optimal temperature as a post-hoc correction.
- **vs. Han et al. (2024) Approximate softmax**: This paper adopts the architecture but is the first to apply it to the theoretical analysis of ICL under distribution shift.
- **vs. Empirical temperature scaling (Lin, Peng, Zou)**: This paper provides a unified theory for "why/when/how much" to tune, connecting scattered empirical results.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of approximate softmax to ICL temperature theory.
- Experimental Thoroughness: ⭐⭐⭐ Includes both synthetic and LLM QA, but LLM models are somewhat dated.
- Writing Quality: ⭐⭐⭐⭐ Mathematically dense but logically clear; the appendix provides complete proofs.
- Value: ⭐⭐⭐⭐ Provides a simple, deployable inference-time tool for ICL robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)
- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)
- [\[ICML 2026\] GEM: Geometric Entropy Mixing for Optimal LLM Data Curation](gem_geometric_entropy_mixing_for_optimal_llm_data_curation.md)

</div>

<!-- RELATED:END -->
