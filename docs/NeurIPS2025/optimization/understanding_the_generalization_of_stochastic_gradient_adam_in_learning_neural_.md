---
title: >-
  [Paper Note] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks
description: >-
  [NeurIPS 2025][Optimization][Adam] This work presents the first theoretical analysis of the generalization behavior of mini-batch Adam. It proves that large-batch Adam/AdamW converges to solutions with high test error even with weight decay, whereas small-batch variants achieve near-zero test error through a combination of implicit regularization from stochastic gradients and explicit regularization from weight decay. Moreover, the effective weight decay upper bound for Adam is strictly smaller than that for AdamW.
tags:
  - NeurIPS 2025
  - Optimization
  - Adam
  - AdamW
  - batch size
  - weight decay
  - generalization
  - feature learning
date: 2026-05-08
content_hash: be1d7d9ec7a2f59c
---

# Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2510.11354](https://arxiv.org/abs/2510.11354)
**Code**: None
**Area**: Optimization Theory / Deep Learning Theory
**Keywords**: Adam, AdamW, batch size, weight decay, generalization, feature learning

## TL;DR
This work presents the first theoretical analysis of the generalization behavior of mini-batch Adam. It proves that large-batch Adam/AdamW converges to solutions with high test error even with weight decay, whereas small-batch variants achieve near-zero test error through a combination of implicit regularization from stochastic gradients and explicit regularization from weight decay. Moreover, the effective weight decay upper bound for Adam is strictly smaller than that for AdamW.

## Background & Motivation
**Background**: Adam is the most widely used optimizer in deep learning (e.g., GPT, LLaMA, and DeepSeek all rely on Adam), yet most theoretical analyses are restricted to the full-batch setting, whereas practical usage involves stochastic (mini-batch) Adam.

**Limitations of Prior Work**: Unlike SGD, stochastic Adam does not converge to its full-batch counterpart as the learning rate approaches zero—a property unique to Adam. Zou et al. (2023b) showed that full-batch Adam generalizes poorly even with regularization, but this does not necessarily imply the same for mini-batch Adam used in practice.

**Key Challenge**: Why does mini-batch Adam (small batch) perform well in practice while theoretical analysis (full-batch) predicts poor generalization? How does batch size affect Adam's generalization?

**Key Insight**: The analysis is conducted on a two-layer overparameterized CNN with a signal-noise patch data model, examining the convergence and generalization of large-batch and small-batch Adam/AdamW separately.

**Core Idea**: The stochastic gradient noise induced by small batches suppresses Adam's rate of overfitting to noise patches, while weight decay further suppresses residual noise components; the two mechanisms act synergistically to ensure convergence to solutions dominated by true features.

## Method

### Overall Architecture
- **Data model**: Each sample is $x = [x_1^T, x_2^T]^T$, where one patch is a signal $y \cdot v$ (1-sparse) and the other is noise $\xi$ ($s$-sparse Gaussian).
- **Model**: Two-layer CNN $F_j(W, x) = \sum_r [\sigma(\langle w_{j,r}, x_1 \rangle) + \sigma(\langle w_{j,r}, x_2 \rangle)]$ with activation $\sigma(x) = [x]_+^q$ ($q \geq 3$).
- **Optimizers**: Adam (L2 regularization applied in the gradient) vs. AdamW (decoupled weight decay).

### Key Designs

1. **Large-batch analysis (Theorems 4.1, 4.4)**

    - **Finding**: When batch size $B = n$ (or close to $n$), both Adam and AdamW converge to solutions with high test error.
    - **Mechanism**: Full-batch gradients treat all samples uniformly, causing signal and noise patches to be learned at comparable rates; weight decay cannot selectively suppress noise components.

2. **Small-batch analysis (Theorems 4.2, 4.5)**

    - **Finding**: When batch size $B = \text{polylog}(n)$, both Adam and AdamW achieve near-zero test error.
    - **Dual regularization mechanism**: (i) Stochastic gradients implicitly slow down noise fitting, since different mini-batches encounter different noise directions while signal directions remain consistent; (ii) weight decay explicitly suppresses residual noise.
    - **Key condition**: The weight decay $\lambda$ must fall within a specific upper bound.

3. **Weight decay sensitivity: Adam vs. AdamW (Corollaries 4.3, 4.6)**

    - The effective $\lambda$ upper bound for Adam is strictly smaller than that for AdamW.
    - **Reason**: Adam's adaptive gradient normalization amplifies the effective influence of weight decay—$\lambda$ appears in the gradient and is normalized by $\sqrt{v}$, further magnifying the regularization effect.
    - **Practical implication**: Adam requires more careful tuning of $\lambda$, whereas AdamW is more robust to $\lambda$ selection.

4. **SignSGD approximation (Appendix C)**

    - Under appropriate conditions, stochastic Adam $\approx$ SignSGD and stochastic AdamW $\approx$ SignSGDW.
    - **Condition**: Gradient magnitudes dominate optimization noise ($|g_{t,j,r}^{(t)}[k]| \geq \tilde{\Theta}(\eta)$).

### Loss & Training
- **Adam**: $L(W) = \frac{1}{n}\sum L_i(W) + \frac{\lambda}{2}\|W\|_F^2$ (L2 regularization incorporated in the loss).
- **AdamW**: $L(W) = \frac{1}{n}\sum L_i(W)$ (weight decay decoupled from the gradient update).
- Cross-entropy loss is used for classification.

## Key Experimental Results

### Main Results

| Setting | Large-batch Adam | Small-batch Adam | Large-batch AdamW | Small-batch AdamW |
|---------|-----------------|-----------------|------------------|------------------|
| Test error | High (~50%) | **Near 0%** | High (~50%) | **Near 0%** |
| Theory | Thm 4.1 | Thm 4.2 | Thm 4.4 | Thm 4.5 |

### Ablation Study: Weight Decay Sensitivity

| Setting | Adam ($\lambda > 0.05$) | Adam ($\lambda < 0.05$) | AdamW ($\lambda = 0.5$) |
|---------|------------------------|------------------------|------------------------|
| Test error | Catastrophic increase | Normal | Normal, no significant degradation |

### Key Findings
- Batch size is the decisive factor for Adam's generalization—not learning rate or momentum parameters.
- Adam's admissible range for $\lambda$ is strictly narrower than that of AdamW, explaining why AdamW is easier to tune in practice.
- Theoretical predictions are validated on both synthetic and real-world data (e.g., CIFAR-10).

## Highlights & Insights
- **First theoretical explanation of batch size's effect on Adam's generalization**: This work closes a critical gap between full-batch Adam theory and mini-batch Adam practice.
- **Clear articulation of the dual regularization mechanism**: Stochastic noise (implicit) and weight decay (explicit) are both necessary.
- **Theoretical foundation for Adam vs. AdamW tuning sensitivity**: The difference in $\lambda$ upper bounds is characterized with precise mathematics.
- **Direct practical guidance**: When using Adam, $\lambda$ must be tuned carefully (tighter upper bound), or AdamW should be preferred for a wider tuning range.

## Limitations & Future Work
- **Simplified data model**: The 1-sparse signal and $s$-sparse noise setting differs substantially from real-world images.
- **Two-layer CNN limitation**: The analysis does not extend to modern architectures such as Transformers.
- **Activation function $q \geq 3$**: This excludes ReLU ($q = 1$) and GELU.
- **Effects of $\beta_1, \beta_2$ not considered**: These hyperparameters are treated as fixed constants, though they also matter in practice.

## Related Work & Insights
- **vs. Zou et al. (2023b)**: They proved that full-batch Adam generalizes poorly; this work extends the analysis to stochastic Adam and demonstrates that small-batch Adam can generalize well.
- **vs. Li et al. (2025) (SignGD on Transformers)**: They analyzed the poor generalization of SignGD on Transformers; this work shows that stochastic Adam $\approx$ SignSGD (but not exactly).
- **vs. Wilson et al. (2017)**: They empirically found that Adam generalizes worse than SGD; this work provides a finer-grained explanation—batch size is the key factor.

## Rating
- Novelty: ⭐⭐⭐⭐ First analysis of mini-batch Adam's generalization with clear theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic data and real-world benchmarks (CIFAR-10).
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly stated theorems.
- Value: ⭐⭐⭐⭐ Significant for understanding Adam's practical behavior.

## Additional Notes
- The theoretical framework and technical tools developed in this work also offer insights for adjacent research areas.
- The core contribution lies in theoretical understanding, providing a foundation for subsequent practical optimization.
- The work is methodologically complementary to other NeurIPS 2025 papers published concurrently.
- The paper's exposition of problem motivation and technical approach serves as a valuable reference for presentation style.
- Readers are encouraged to consult the appendix for complete experimental details and proofs.

## Extended Reading
- This research direction is closely related to several active topics in the AI community.
- The rigor of the theoretical results provides a solid mathematical foundation for subsequent empirical studies.
- The methodology can be generalized to broader problem settings.
- Follow-up work from this group is worth monitoring.
- For beginners in theoretical research, the proof sketch section provides an excellent technical roadmap.
- From a methodological perspective, the paper demonstrates how careful mathematical modeling can reduce complex problems to tractable analytical frameworks.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise](unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)

<!-- RELATED:END -->
