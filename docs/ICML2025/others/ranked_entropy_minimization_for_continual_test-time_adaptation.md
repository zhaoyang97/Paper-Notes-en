---
title: >-
  [Paper Note] Ranked Entropy Minimization for Continual Test-Time Adaptation
description: >-
  [ICML 2025][Test-Time Adaptation] Proposes Ranked Entropy Minimization (REM), which constructs an explicit ranking structure of prediction difficulty through a progressive masking strategy. Combining masked consistency loss and entropy ranking loss, it solves the model collapse issue of entropy minimization methods in Continual Test-Time Adaptation (CTTA) while maintaining computational efficiency.
tags:
  - "ICML 2025"
  - "Test-Time Adaptation"
  - "Continual Test-Time Adaptation"
  - "Entropy Minimization"
  - "Model Collapse"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 95acea2dc56b3029
---

# Ranked Entropy Minimization for Continual Test-Time Adaptation

**Conference**: ICML 2025  
**arXiv**: [2505.16441](https://arxiv.org/abs/2505.16441)  
**Code**: [https://github.com/pilsHan/rem](https://github.com/pilsHan/rem)  
**Area**: LLM Evaluation  
**Keywords**: Test-Time Adaptation, Continual Test-Time Adaptation, Entropy Minimization, Model Collapse, Vision Transformer

## TL;DR

Proposes Ranked Entropy Minimization (REM), which constructs an explicit ranking structure of prediction difficulty through a progressive masking strategy. Combining masked consistency loss and entropy ranking loss, it solves the model collapse issue of entropy minimization methods in Continual Test-Time Adaptation (CTTA) while maintaining computational efficiency.

## Background & Motivation

### Problem Definition

Test-Time Adaptation (TTA) aims to adapt to distribution shifts in the target domain online during the inference phase. Continual Test-Time Adaptation (CTTA) further requires the model to sequentially adapt to continuous domain shifts without resetting model parameters, which renders the catastrophic forgetting problem more pronounced.

### Limitations of Prior Work

The mainstream methods of CTTA are divided into two main categories:

**Entropy Minimization (EM)**: Represented by Tent, it achieves adaptation by minimizing prediction entropy. It is computationally efficient (updating only BN layers) but poses a risk of **model collapse**—the model converges to a trivial solution that predicts the same category for all inputs.

**Consistency Regularization (CR)**: Represented by CoTTA, it adopts a teacher-student framework to guarantee stability, but requires an extra model and a massive number of forward passes, which is computationally expensive.

Essentially, a **trade-off between efficiency and stability** exists: EM is efficient but unstable, while CR is stable but inefficient.

### Key Observations

The authors discover that the gradient of entropy minimization has two trivial solutions: (1) a uniform distribution $\hat{p}_{t,c} = 1/C$, and (2) a completely confident prediction $\hat{p}_{t,c} \in \{0,1\}$. In practice, the model tends to converge to a completely confident prediction for a single category, leading to model collapse (where Tent's performance degrades drastically after adapting to the impulse noise domain).

### Key Insight

If key objects in an image are masked out, the model's prediction accuracy drops and the prediction entropy rises—this is an intuitive and exploitable explicit ranking relationship. This is inspired by Zeno's paradox of "Achilles and the Tortoise": the original prediction (the tortoise) maintains its lead, while the masked predictions (Achilles) struggle to catch up, thereby preventing a sharp decline in entropy that leads to collapse.

## Method

### Overall Architecture

REM is composed of three core components:

1. **Explicit Mask Chaining**: Utilizes the self-attention mechanism of ViT to progressively mask patches based on attention scores from high to low, constructing a ranking structure of prediction difficulty.
2. **Masked Consistency Loss (MCL)**: Ensures that predictions with high masking ratios remain consistent with those with lower masking ratios.
3. **Entropy Ranking Loss (ERL)**: Ensures that the entropy of predictions with lower masking ratios is lower than that of predictions with higher masking ratios.

The total loss is a linear combination of both, utilizing only a single model and updating only the BN layers.

### Key Designs

#### Explicit Mask Chaining

The self-attention structure of the last layer in ViT is utilized to compute the attention score:

$$A = \sum_{h=1}^{H} \text{Softmax}\left(\frac{Q_{h,cls} K_{h,img}^\top}{\sqrt{d}}\right)$$

where $Q_{h,cls}$ is the query of the CLS token, and $K_{h,img}$ is the key of the image token. Patches with high attention scores are highly likely to contain target objects.

After sorting the attention scores in descending order, a masked chain $\{x_{m_1}, x_{m_2}, \cdots, x_{m_N}\}$ is defined, satisfying $0 \leq m_1 \leq m_2 \leq \cdots \leq m_N \leq 1$. Experiments verify that as the masking ratio increases, error and entropy monotonically increase, showing an approximately linear relationship especially in the low masking ratio region.

**Key Design Points**:

- Masking foregrounds (target regions) instead of backgrounds or random masking, as only foreground masking can establish a reliable ranking relationship.
- By default, $M_N = \{0, 5\%, 10\%\}$ is used, representing the original image plus two levels of masking.
- Leveraging ViT's inherent attention without requiring extra computation.

#### Masked Consistency Loss (MCL)

$$\mathcal{L}_{MCL} = \sum_{i < j}^{M_N} \mathcal{H}(f_t(x_j), \mathbf{sg}(f_t(x_i)))$$

where $\mathcal{H}(p, q)$ is the cross-entropy, and $\mathbf{sg}$ is the stop-gradient operation.

**Design Strategy**:

- Letting high-masking-ratio predictions align with low-masking-ratio predictions, thereby indirectly reducing prediction entropy.
- Compared to direct EM, using cross-entropy as the loss instead of self-entropy avoids abrupt entropy changes and overconfidence.
- Compared to CR methods, there is no need for extra teacher models or uncertainty estimation, as diverse predictions are generated within a single model through the explicit ranking structure.
- Encouraging the model to learn context information of the masked regions.

#### Entropy Ranking Loss (ERL)

$$\mathcal{L}_{ERL} = \sum_{i < j}^{M_N} \max(0, \mathcal{S}(f_t(x_i)) - \mathbf{sg}(\mathcal{S}(f_t(x_j))) + \mathsf{m})$$

where $\mathsf{m}$ is the margin hyperparameter.

**Design Strategy**:

- Maintaining a "low masking $\rightarrow$ low entropy" ranking structure, which prevents high-masking-ratio predictions from being overconfident and biasing towards background information.
- Drawing inspiration from the success of ranking losses in neural network calibration (such as RankMixup), which effectively mitigates overconfidence.
- Directly reducing the entropy of samples that violate the ranking constraints to accelerate adaptation.
- Complementary to MCL: MCL indirectly reduces entropy but may adapt slowly, while ERL directly reduces entropy to supplement the adaptation speed.

### Loss & Training

**Total Loss Function**:

$$\mathcal{L}_{REM} = \mathcal{L}_{MCL} + \lambda \cdot \mathcal{L}_{ERL}$$

**Training Strategy**:

- Modifying only the normalization layer parameters of the ViT (approx. 0.03M).
- Using the Adam optimizer with a learning rate of 1e-3 (ImageNet-C and CIFAR-C).
- Single model, without requiring EMA teacher models or source model storage.
- Each sample requires 3 forward passes (original + 2 levels of masking) and 1 backward pass.
- Default hyperparameters: $\lambda = 1$, $\mathsf{m} = 0$, $M_N = \{0, 5\%, 10\%\}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (Continual-MAE) | Source | Gain (vs Source) |
|---|---|---|---|---|---|
| ImageNetC (CTTA) | Mean Error ↓ | **39.2%** | 42.5% | 55.8% | +16.6% |
| CIFAR10C (CTTA) | Mean Error ↓ | **Best** | - | 28.2% | Significant Gain |
| CIFAR100C (CTTA) | Mean Error ↓ | **Best** | - | 35.4% | Significant Gain |
| ImageNetC (TTA, imbalanced) | Mean Acc ↑ | **63.3%** | - | 29.9% | +33.4% |
| ImageNetC (TTA, BS=1) | Mean Acc ↑ | **60.1%** | - | 29.9% | +30.2% |
| ImageNetC (TTA, mixed L5) | Mean Acc ↑ | **62.4%** | - | 29.9% | +32.5% |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| REM (MCL + ERL) | Best, no collapse | Organic cooperation of both losses, best stability |
| MCL Only (ERL removed) | Performance degrades, early collapse on CIFAR-C | Indirect entropy reduction is insufficient to sustain adaptation speed |
| ERL Only (MCL removed) | Performance degrades, early collapse on CIFAR-C | Lack of consistency regularization leads to instability |
| Foreground Masking (REM) | Best, no collapse | Satisfies the intuitive assumption of explicit ranking |
| Background Masking | Performance degrades | Ranking relationship does not hold |
| Random Masking | Performance degrades | Ranking relationship is unreliable |

### Efficiency Comparison

| Method | Trainable Params | Total Time | Forward Passes | Models | Error (%) |
|---|---|---|---|---|---|
| Tent (EM) | 0.04M | 8m35s | 1 | 1 | 51.0 |
| CoTTA (CR) | 86.4M | 33m23s | 3~35 | 2 | 54.8 |
| ViDA (CR) | 93.7M | 54m48s | 1 | 2 | 43.4 |
| Continual-MAE (CR) | 86.5M | 59m56s | 1 | 2 | 42.5 |
| **REM (Ours)** | **0.03M** | **17m21s** | **3** | **1** | **39.2** |

### Key Findings

1. **Significant Efficiency Advantage**: Compared to Continual-MAE, REM improves performance by 3.3% while consuming only 30% of the computation time and 0.03% of the trainable parameters.
2. **Low Calibration Error**: REM's ECE is 8.7%, the lowest among all high-performance methods (ViDA is 14.6%), indicating that it avoids overconfidence while reducing error rates.
3. **Robustness to Learning Rate**: REM performs stably across different learning rate settings, allowing for flexible selection of adaptation speeds based on application requirements.
4. **Forward/Backward Transfer Trade-off**: A higher learning rate leads to better performance in seen domains (42.1%), whereas a lower learning rate generalizes better in unseen domains (41.4%), with the gap to the supervised upper bound being only 3.5%.
5. **Broad Applicability**: Can be extended to CNN architectures (using Feature Activation or Grad-CAM in place of attention scores) and CLIP vision-language models.

## Highlights & Insights

1. **Simple yet Effective Intuition**: The core idea—masking targets leads to increased entropy—is clear and intuitive, yet can give rise to a complete methodological framework. This approach of "converting uncontrolled distribution shifts into controlled ranking structures" is highly instructive.
2. **Merging the Advantages of EM and CR**: REM replaces data augmentation with a masked chain to achieve consistency, and replaces direct entropy minimization with a ranking loss, preserving the efficiency of EM and the stability of CR.
3. **Single Model Paradigm**: No teacher model, EMA model, or source model storage is required, making it highly attractive from an engineering deployment perspective.
4. **Exploiting the Inherent ViT Structure**: Attention scores naturally indicate target locations; thus, the masking strategy achieves semantically meaningful augmentation without extra computation.

## Limitations & Future Work

1. **Insufficient Theoretical Support**: The assumption that explicit masking causes entropy to increase lacks rigorous mathematical proof; although statistical significance is verified experimentally, counterexamples may exist for certain individual samples.
2. **Learning-rate-sensitive Adaptation Speed Trade-off**: The trade-off between fast adaptation and generalization is not fully resolved.
3. **Limited Adaptation Speed in Simple Domains**: TVD analysis shows that for simpler domains (such as brightness, JPEG), the small prediction differences before and after masking lead to low loss and slow adaptation. The authors suggest introducing adaptive loss weights in the future.
4. **Fixed Masking Ratios**: Currently, a fixed set of masking ratios is used without dynamic adjustments based on domain discrepancies.
5. **ViT Dependency**: Although extendable to CNNs, the core method design is fundamentally based on the ViT attention mechanism, and its generalizability warrants further validation.

## Related Work & Insights

- **Tent (Wang et al., 2021)**: The pioneering work in TTA, which proposes an entropy minimization strategy that updates only BN layers; it is the direct target for improvement in this work.
- **CoTTA (Wang et al., 2022)**: A benchmark CTTA method featuring a teacher-student framework and stochastic parameter restoration, which is stable but inefficient.
- **SAR (Niu et al., 2023)**: First to observe the model collapse phenomenon, proposing sharpness-aware minimization and reliable sample selection.
- **Continual-MAE (Liu et al., 2024)**: The prior SOTA, which uses MC dropout for uncertainty estimation and masked autoencoders, but incurs a heavy computational expense.
- **RankMixup (Noh et al., 2023)**: Application of ranking loss in network calibration, providing inspiration for the design of ERL.

## Rating

| Dimension | Score | Description |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | Converts uncontrolled distribution shifts into controlled ranking structures; the idea is novel and practical |
| Technical Depth | ⭐⭐⭐⭐ | The design of the two complementary losses is sound, and ablation studies thoroughly validate each component |
| Experimental Quality | ⭐⭐⭐⭐⭐ | Comprehensive validation across multiple datasets, scenarios, and architectures with convincing efficiency comparisons |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivation, a vivid analogy of Achilles and the tortoise, and well-designed figures |
| Practical Value | ⭐⭐⭐⭐⭐ | 0.03M parameters, single model, plug-and-play, holding extremely high engineering deployment value |
| **Overall** | **⭐⭐⭐⭐☆** | A simple and elegant method achieving SOTA at extremely low cost; highly worthy of attention |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](../../CVPR2025/others/effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[ICML 2025\] Discrepancy Minimization in Input-Sparsity Time](discrepancy_minimization_in_input-sparsity_time.md)

</div>

<!-- RELATED:END -->
