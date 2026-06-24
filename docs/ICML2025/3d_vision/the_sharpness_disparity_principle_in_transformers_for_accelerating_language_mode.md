---
title: >-
  [Paper Note] The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training
description: >-
  [ICML 2025][3D Vision][Transformer optimization] Reveals a significant and persistent **sharpness disparity** among different types of modules (Emb, QK, FFN, VO, Norm) in Transformers. Based on this, a Blockwise LR strategy is proposed to allocate larger learning rates to low-sharpness modules, achieving nearly **2× acceleration** in LLM pre-training without compromising stability.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "Transformer optimization"
  - "Learning rate scheduling"
  - "Hessian analysis"
  - "Pre-training acceleration"
  - "Blockwise sharpness"
date: 2026-05-08
content_hash: a395ec963cfbdbe7
---

# The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training

**Conference**: ICML 2025  
**arXiv**: [2502.19002](https://arxiv.org/abs/2502.19002)  
**Code**: [GitHub](https://github.com/Wongboo/BlockwiseLearningRate)  
**Area**: LLM Pre-training  
**Keywords**: Transformer optimization, Learning rate scheduling, Hessian analysis, Pre-training acceleration, Blockwise sharpness

## TL;DR

Reveals a significant and persistent **sharpness disparity** among different types of modules (Emb, QK, FFN, VO, Norm) in Transformers. Based on this, a Blockwise LR strategy is proposed to allocate larger learning rates to low-sharpness modules, achieving nearly **2× acceleration** in LLM pre-training without compromising stability.

## Background & Motivation

Transformers exhibit "alloy-like" characteristics: they are composed of heterogeneous modules such as Embedding layers, Normalization layers, Self-Attention (QK/VO), and FFN. Traditional layerwise learning rate adjustment strategies are effective on MLP/CNNs but fail to transfer to deep Transformers. The authors argue that **the root cause is that the heterogeneity of Transformers does not present a consistent pattern at the layer level, but rather exhibits a clear sharpness disparity at the module-type level**.

Prior works (Zhang et al., 2024; Ormaniec et al.) observed the blockwise heterogeneity of the Hessian but did not establish a unified principle across all module types. Building upon these works, this paper systematically compares the sharpness of five categories of modules and establishes a complete sharpness ordering for the first time.

Furthermore, the training process operates under the Edge of Stability (EoS) regime: violent oscillations occur in high-sharpness directions (fast dynamics, which determine training stability), while slow progress is made in low-sharpness directions (slow dynamics, which primarily contribute to loss reduction). If the slow dynamics along low-sharpness directions can be accelerated without perturbing high-sharpness directions, overall training acceleration can be achieved.

## Method

### Overall Architecture

The method consists of two core components:

1. **Sharpness Disparity Principle**: Quantifies sharpness disparities at the module-type granularity using a diagonal approximation of the Fisher Information Matrix, establishing a unified ordinal relationship.
2. **Blockwise LR**: Allocates different learning rate multipliers to different types of modules according to the sharpness disparity principle.

### Key Designs

#### 1. Sharpness Disparity Principle (Principle 1)

Estimating the average sharpness of the five classes of modules reveals that the following relations hold consistently throughout the training process:

$$\mathcal{S}(\text{Emb}) \ll \mathcal{S}(\text{QK}) < \mathcal{S}(\text{FFN}) < \mathcal{S}(\text{VO}) \ll \mathcal{S}(\text{Norm})$$

- **Norm layers have the highest sharpness**: They have very few parameters (only $D$ scalars per layer), and the gradient norm is exceptionally large relative to the parameter norm.
- **Emb layer has the lowest sharpness**: The vocabulary dimension $d$ is extremely large (e.g., 50,304 for the GPT tokenizer) with a massive parameter count, resulting in the smallest sharpness when averaged.
- **QK < FFN < VO**: FFN and VO lie in between, while QK is flatter due to the nature of the softmax operation.

#### 2. Efficient Sharpness Estimation

The diagonal approximation of the Fisher Information Matrix is used to efficiently estimate the Hessian. For each module type $\bullet$, the average sharpness is defined as:

$$\mathcal{S}(\bullet) = \frac{B \|\nabla_{\theta[\bullet]} \hat{\mathcal{L}}_B(\theta)\|_F^2}{\#(\theta[\bullet])}$$

where $B$ is the batch size, and $\#(\theta[\bullet])$ is the parameter count of that module type. The key advantage of this approach is that it only requires mini-batch gradients rather than per-sample gradients, making the computation overhead negligible.

#### 3. Theoretical Explanation

Based on analytical gradient expressions, three sets of theorems (Theorems 4.1–4.3) prove the source of sharpness disparity:

- **FFN vs. Norm (Theorem 4.1)**: $\mathcal{S}(W_\bullet) = \mathcal{O}(\Psi^2 / (D^2 \|W_\bullet\|_F^2))$ and $\mathcal{S}(\gamma) = \mathcal{O}(\Psi^2 / (D \|\gamma\|_F^2))$. Since $D^2 \|W_\bullet\|_F^2 \gg D \|\gamma\|_F^2$ (due to the large parameter size and norm of FFN, plus the Norm parameter $\gamma$ gradually decreasing during training), the sharpness of FFN is far lower than that of Norm.
- **SA vs. Norm (Theorem 4.2)**: QK and VO also have much lower sharpness than Norm for similar reasons.
- **Emb vs. Norm (Theorem 4.3)**: $\mathcal{S}(W_E) = \mathcal{O}(\Psi^2 / (Dd \min_i \|\tilde{w}_{E_i}\|_2^2))$. The vocabulary size $d$ (~50k) makes the denominator extremely large, giving Emb the lowest sharpness.

Core intuition: The multiplicative composition of Transformers yields $\|\nabla_\bullet \mathcal{Q}\| \propto 1/\|\theta[\bullet]\|$, meaning that modules with more/larger parameters maintain lower sharpness.

#### 4. Blockwise LR Strategy

- **Norm Blocks**: Keep the base learning rate $\eta_{\text{Norm}} = \eta_{\text{base}}$ (to maintain stability)
- **Other Blocks**: Multiply the learning rate by a scaling ratio $r(\bullet)$, manually tuned based on the qualitative trend of sharpness disparity.

The default multipliers obtained by tuning (tuned only once on LLaMA 0.25B + MiniPile) are:

| Module Type | Scaling Ratio $r(\bullet)$ | Sharpness Rank |
|---------|---------------------|---------|
| Emb | 10× | Lowest (Flattest) |
| QK | 8× | Lower |
| FFN | 6× | Medium |
| VO | 4× | Higher |
| Norm | 1× (unchanged) | Highest (Sharpest) |

The key advantage of these multipliers is that **they only need to be tuned once and generalize across different models and datasets** with extreme robustness.

### Loss & Training

- **Base Optimizer**: AdamW, $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $\lambda = 0.1$
- **Gradient Clipping**: clip norm = 1.0
- **Learning Rate Schedule**: Linear warm-up + cosine decay (also verified with WSD scheduler)
- **Blockwise LR Integration**: Different LR scaling factors are applied to parameter groups of different module types in AdamW. The implementation is extremely simple and can be directly integrated into frameworks like Megatron.

## Key Experimental Results

### Main Results

Validated across various scales of GPT-2 and LLaMA:

| Model | Parameter Count | Dataset | AdamW loss | + Blockwise LR loss | Speedup |
|------|--------|--------|-----------|---------------------|--------|
| LLaMA | 0.25B | OpenWebText | 2.834 | 2.784 | ~1.8× |
| LLaMA | 0.5B | OpenWebText | — | — | ~1.9× |
| LLaMA | 1.1B | OpenWebText | — | — | **1.92×** |
| LLaMA | 0.25B | MiniPile | — | — | ~1.9× |
| GPT-2 | 0.12B | OpenWebText | — | — | ~1.8× |
| LLaMA | 2B | C4 | — | — | ~2× |

**Downstream Task Evaluation** (LLaMA 1.1B, OpenWebText, 50K steps, 0-shot):

| Task | AdamW | Blockwise LR | Gain |
|------|-------|-------------|------|
| ARC_E | 52.69 | **54.29** | +1.60 |
| ARC_C | 22.87 | **25.34** | +2.47 |
| PIQA | 68.71 | **69.53** | +0.82 |
| HellaSwag | 36.13 | **38.00** | +1.87 |
| OBQA | 19.40 | **22.60** | +3.20 |
| WinoGrande | 55.17 | **59.83** | +4.66 |
| SCIQ | 77.60 | **81.60** | +4.00 |

### Ablation Study

LLaMA 0.25B on OpenWebText, 50K steps:

| Configuration | Terminal Loss | Change | Description |
|------|-------------|------|------|
| AdamW baseline | 2.834 | — | Baseline |
| + Emb only | 2.818 | -0.016 ✓ | Increasing Emb LR is effective |
| + Emb & FFN | 2.791 | -0.043 ✓ | FFN contributes the most (-0.027) |
| + Emb & FFN & QK & VO | 2.784 | -0.050 ✓ | All low-sharpness modules contribute |
| + Norm only (doubled) | 2.837 | +0.003 ✗ | Increasing Norm LR is counter-productive |

### Key Findings

1. **FFN contributes the most**: It accounts for the majority of the parameter count, and increasing its LR leads to the most significant drop in loss (contributing 0.027 individually).
2. **Increasing Norm LR is detrimental**: This validates that high-sharpness directions, where Norm resides, should not be perturbed.
3. **Favorable scaling law**: As model size scales up, the advantage of Blockwise LR **widens** on MiniPile/OpenWebText, while remaining stable on C4.
4. **Cross-optimizer compatibility**:
    - Adam-mini + Blockwise LR = **2× speedup + 2× memory savings**
    - Lion + Blockwise LR = **2× speedup**
    - WSD scheduler + Blockwise LR = **2× speedup**

## Highlights & Insights

1. **From observation to principle**: Instead of simple layerwise analysis, this work establishes a unified sharpness ordering at the **module-type** level, revealing the core of Transformer's "alloy-like" characteristics.
2. **Solid theoretical support**: Three sets of theorems originate from analytical gradient expressions, explaining the sharpness disparity using parameter counts and parameter norms, providing clear intuition (multiplicative composition $\rightarrow$ gradients inversely proportional to parameter norms).
3. **Minimalist implementation**: Requires only 4 additional hyperparameters ($r$ ratios). Once tuned, they transfer across models/datasets and can be directly integrated into PyTorch parameter groups.
4. **Unified EoS perspective**: Explains Blockwise LR as "accelerating slow dynamics along the stable (river) directions," elegantly aligning with the Edge of Stability (EoS) theory.
5. **Sharpness disparity is established extremely early (~2% of steps)**: This implies that Blockwise LR can be deployed from the very beginning, without waiting for the training to stabilize.

## Limitations & Future Work

1. **Validated only on language model pre-training**: Evaluations on Vision Transformers, multi-modality, etc., are currently lacking.
2. **Manual ratio tuning**: Although robust, the ratios could theoretically be automatically derived from sharpness ratios to achieve adaptive blockwise LR.
3. **Compatibility with newer optimizers like Muon remains unverified**.
4. **Natural extensions such as blockwise weight decay and blockwise gradient clipping have not been explored**.
5. **The theoretical proof of the QK < VO relationship relies on Ormaniec et al.**, while this paper only provides empirical validation.
6. **Blockwise LR only significantly outperforms the baseline in the mid-to-late stages of training**, with unpronounced advantages in the early stage for reasons yet to be identified.

## Related Work & Insights

- **Zhang et al. (2024) Adam-mini**: Leverages Hessian blockwise heterogeneity to halve Adam's memory usage; the current work focuses on acceleration on top of this.
- **Ormaniec et al.**: Analyzes QK vs. VO sharpness in single-layer self-attention, inspiring this paper's comprehensive comparison across all module types.
- **Wang et al. (2024) SOAP**: Adjusts LR according to individual parameter sharpness, which incurs heavy computational overhead and unstable estimates; this work scales LR at the module-type level, remaining simple and efficient.
- **EoS Series (Cohen et al., 2021; Wen et al., 2024)**: The fast-slow dynamics landscape provides the theoretical motivation for Blockwise LR.
- **WSD scheduler (Wen et al., 2024)**: Complementary to Blockwise LR, allowing them to be combined.

**Key Insight**: Optimization of deep networks should not treat all variables equally; instead, policies should be customized based on the geometric properties of different components. Sharpness analysis at the module-type level may open up a new paradigm for optimizer design.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The module-type level sharpness principle is a new discovery, and Blockwise LR is simple yet effective. |
| Theoretical Depth | 4 | Three sets of theorems provide rigorous theoretical support. |
| Experimental Thoroughness | 5 | Multi-model (GPT-2/LLaMA), multi-scale (0.12B-2B), multi-dataset, and multi-optimizer. |
| Value | 5 | Simple implementation, strong transferability, significant speedup, and combinable with memory optimizations. |
| Writing Quality | 4 | Clear structure, abundant plots and tables, and smooth logic. |
| **Overall** | **4.5** | High-quality work that combines both theoretical depth and practical value. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Slamming: Training a Speech Language Model on One GPU in a Day](../../ACL2025/3d_vision/slamming_training_a_speech_language_model_on_one_gpu_in_a_day.md)
- [\[ICCV 2025\] 4D Visual Pre-training for Robot Learning](../../ICCV2025/3d_vision/4d_visual_pretraining_for_robot_learning.md)
- [\[ECCV 2024\] Formula-Supervised Visual-Geometric Pre-training (FSVGP)](../../ECCV2024/3d_vision/formula-supervised_visual-geometric_pre-training.md)
- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](../../CVPR2025/3d_vision/unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](../../CVPR2026/3d_vision/e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)

</div>

<!-- RELATED:END -->
