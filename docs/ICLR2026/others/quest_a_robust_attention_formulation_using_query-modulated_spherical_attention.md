---
title: >-
  [Paper Note] QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention
description: >-
  [ICLR 2026][Others][Attention] QUEST normalizes the key vectors in standard scaled dot-product attention to a hypersphere while maintaining the norm degrees of freedom for the queries (i.e., $A=\mathrm{softmax}(Q\bar{K}^\top)$). With a modification of less than one line, it simultaneously eliminates training instability caused by attention logit exp
tags:
  - ICLR 2026
  - Others
  - Attention
  - Vision Transformer
date: 2026-05-08
content_hash: e320f8fed0447038
---
# QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HkztQWZfl2](https://openreview.net/forum?id=HkztQWZfl2)  
**Code**: To be confirmed  
**Area**: Transformer Architecture / Attention Mechanism / Training Stability / Robustness  
**Keywords**: Attention Mechanism, Training Stability, Hyperspherical Normalization, Robustness, Vision Transformer

## TL;DR
QUEST normalizes the key vectors in standard scaled dot-product attention to a hypersphere while maintaining the norm degrees of freedom for the queries (i.e., $A=\mathrm{softmax}(Q\bar{K}^\top)$). With a modification of less than one line, it simultaneously eliminates training instability caused by attention logit explosion and enables the model to learn more dispersed and robust attention. It consistently outperforms standard attention and QKNorm across multiple tasks including ImageNet classification, segmentation, and adversarial attacks.

## Background & Motivation
**Background**: The core of the Transformer is Scaled Dot-Product Attention (SDPA) $A=\mathrm{softmax}(C\,QK^\top)$, where $C=1/\sqrt{D_H}$ is a fixed scaling factor. This formula is adopted by almost all variants like ViT, GPT, PointTransformer, and Conformer. However, large model training frequently encounters instability. The industry relies on initialization tricks, learning rate scheduling, various normalizations, and optimization strategies to mitigate this, yet the underlying cause of "why it is unstable" has not been clearly explained.

**Limitations of Prior Work**: The authors decompose SDPA—letting $q=\|q\|\bar q$ (norm times unit vector), the attention for a single token can be written as $A_i=\mathrm{softmax}\big(C\|q_i\|\|k_j\|(\bar q_i\cdot \bar k_j)\big)_j$. This exposes two overlooked roles: **the query norm $\|q_i\|$ scales all logits for that token, controlling the "sharpness" of its attention distribution** (high norm $\rightarrow$ attention concentrates on few tokens, low norm $\rightarrow$ smoother); meanwhile, **the key norm $\|k_j\|$ amplifies the score of token $j$ in the eyes of all queries**, allowing it to "globally grab attention." When query/key norms grow arbitrarily during training, logits explode, and the model gets stuck in suboptimal solutions.

**Key Challenge**: Existing stabilization schemes like QKNorm (which applies $\ell_2$ normalization to both query and key, then rescales per-dimension or per-head with learnable parameters $C_q, C_k$) do stabilize training. However, they apply **the same scaling factor to all tokens**, forcing every token to have the same attention sharpness, which sacrifices expressivity—sometimes performing worse than standard attention on small models. Thus, a trade-off exists between "training stability" and "attention expressivity."

**Goal**: To find an attention formulation that eliminates norm explosion without sacrificing per-token sharpness control, while serving as a drop-in replacement for any Transformer.

**Key Insight**: Since the query norm is responsible for "useful" sharpness control and the key norm is responsible for "harmful" global attention grabbing, why normalize both? Normalizing only one side can break the query $\leftrightarrow$ key gradient cross-coupling while preserving the norm degree of freedom on one side.

**Core Idea**: Normalize only the keys and keep the queries free—keys are constrained to a hypersphere (sorting is determined entirely by query-key cosine similarity), while the query norm continues to allow each token to independently regulate its own attention sharpness.

## Method

### Overall Architecture
QUEST does not change any other structure of the Transformer; it only replaces the calculation formula in the attention step. In standard multi-head self-attention, each head projects the input sequence $X\in\mathbb{R}^{N\times D}$ into $Q=XW_Q^\top$, $K=XW_K^\top$, and $V=XW_V^\top$, then computes $Z=\mathrm{softmax}(C\,QK^\top)V$. QUEST modifies this to:

$$A=\mathrm{softmax}\big(Q\bar{K}^\top\big),\qquad \bar K = \text{row-wise } \ell_2 \text{ normalization of } K$$

Note two points: **only keys are normalized, queries remain entirely untouched**; and **no scale factor is used** (i.e., $C=1$). Consequently, the ranking of attention logits $= \|q_i\|\,(\bar q_i\cdot \bar k_j)$ is determined entirely by the cosine alignment of queries and keys on the hypersphere, while each query norm $\|q_i\|$ independently controls the sharpness of its own softmax row. This is a highly interpretable modification requiring only a few lines of code, applicable directly to ViT, language models, Graph Transformers, time series, point clouds, and other architectures.

To thoroughly explain the design motivation, the paper also compares a set of proximal variants: standard attention (neither normalized), QNorm (only query normalized), QKNorm-HS (both normalized, one learnable scalar $C\in\mathbb{R}^H$ per head), and QKNorm-DS (learnable $C_q, C_k\in\mathbb{R}^{D_H}$ per feature dimension). QUEST occupies a position previously untried in literature: "only key normalization without additional scaling."

### Key Designs

**1. Spherical Keys, Free Queries: Achieving Stability and Expressivity via Periodic Normalization**

Standard attention is unstable because key norms can grow arbitrarily, allowing a certain token to achieve extremely high scores for all queries, "grabbing attention globally" and pushing logits to explode. QUEST normalizes keys row-wise to the hypersphere ($\|\bar k_j\|=1$), mechanically cutting off the path of "grabbing attention by increasing key norm"—no token can monopolize attention solely due to its norm. However, unlike QKNorm, QUEST **leaves queries untouched**: the preserved query norm $\|q_i\|$ allows each token to still independently choose whether to "look focused or look dispersed." Thus, it does not lock the sharpness of all tokens as QKNorm does, avoiding loss of expressivity. In short, QUEST removes the "harmful degree of freedom" of key norms while keeping the "useful degree of freedom" of query norms.

**2. Why Normalize Keys instead of Queries: Cutting Query $\leftrightarrow$ Key Gradient Coupling**

An accelerator for training failure in standard attention is the "cross-action" between query and key parameter updates—the $\|q_i\|\|k_j\|$ multiplication in the logit means the query norm gradient depends on the key norm and vice-versa, causing them to push each other toward expansion. Normalizing either side breaks this coupling. The paper also evaluates the symmetrical counterpart, QNorm (only query normalized). QNorm indeed mitigates cross-coupling and performs slightly better than standard attention, but it leaves keys free—key norms can still grow and grab attention globally, failing to block the true failure mechanism. QUEST chooses to normalize keys, specifically targeting the root cause of "global attention grabbing" while leaving sharpness control to the queries. Notably, experiments show that although only keys are normalized, query norms and maximum logits are also stabilized as a result.

**3. Elliptical QUEST: Orthogonal Superposition with Elliptical Attention for Further Robustness**

QUEST uses cosine similarity on the hypersphere, while Elliptical Attention (Nielsen et al., 2024) extends the isotropic Gaussian kernel of standard attention to a Mahalanobis-based hyper-ellipsoid. The two operate at different levels and are orthogonal, allowing them to be combined into Elliptical-QUEST: using an elliptical metric instead of pure cosine similarity to measure query-key alignment. Experiments show that while Elliptical itself is robust, it sacrifices classification accuracy on clean data (71.53% vs QUEST's 72.50%). Elliptical-QUEST inherits stronger adversarial robustness while recovering clean accuracy, demonstrating that the "key sphericalization" approach is compatible with other attention improvements.

### Mechanism: Toy Retrieval Task Exposes "Spurious Attention"
The paper constructs a simple retrieval task to visualize the problem. The input is a sequence of vectors $X=[x_1,\dots,x_N]$, where each vector contains a real-valued part $x^k_i$ and an one-hot "answer" part $x^v_i$. Except for the answer token at a random position $L$, the real-valued parts of all non-answer tokens are sampled from $\mathcal{N}(0,I)$. The answer token is "out-of-distribution" ($x^k_L\sim\mathcal{N}(0,\Sigma)$, $\Sigma\neq I$), which is the **always-valid robust signal** that a correct model should learn to locate the answer.

However, the authors inject a **biased signal valid for only about half of the samples**: samples are labeled "biased" with $p=0.5$, and their answer tokens are instead sampled from $\mathcal{N}(b,0.1I)$ with a shared bias vector $b$. This creates a shortcut—if the model increases the magnification of the key weight matrix in direction $b$ to boost the key norm of biased answer tokens, it can globally focus attention on the answer whenever the bias appears, "lazily" solving half the samples. Standard attention and QNorm fall into this trap: the key norm of biased answer tokens grows larger during training, and the model only learns to check for $b$ instead of the robust solution. QKNorm fails entirely because it loses the "answer position distribution anomaly" info in the key norm, regressing to random guessing. QUEST, by prohibiting any token from grabbing attention globally via its norm, increases the success rate from 25% (Standard) and 49% (QNorm) to **58%**, and remains effective across a wider range of learning rates/weight decays.

## Key Experimental Results

### Main Results
ViT-Tiny trained on ImageNet-1K for 300 epochs with DeiT, comparing various QK normalization schemes (mean of multiple runs):

| Attention | IN-val Top-1 | IN-ReaL Top-1 | IN-C MCE ↓ | IN-A Top-1 |
|-----------|--------------|---------------|------------|------------|
| Standard  | 72.6         | 80.4          | 55.7       | 8.2        |
| **QUEST** | **73.4**     | **81.2**      | **55.0**   | 8.5        |
| QNorm     | 72.7         | 80.6          | 55.3       | 8.2        |
| QKNorm-HS | 72.5         | 80.5          | 56.4       | 7.9        |
| QKNorm-DS | 71.6         | 79.6          | 57.4       | 7.2        |
| QKNorm    | 71.9         | 79.0          | 58.1       | 7.0        |

QUEST leads across clean accuracy, ReaL, and corruption robustness (lower MCE is better). The QKNorm family, while stabilizing large models, suffers on small models because sharpness is locked, leading to lower clean accuracy than standard attention.

On larger models (DeiT / DeiT-3 training):

| Model | Attention | IN-val | IN-C MCE ↓ | Notes |
|-------|-----------|--------|------------|-------|
| ViT-S/16 (200ep) | Standard | 79.6 | 44.8 | — |
| ViT-S/16 (200ep) | **QUEST** | **80.2** | **43.2** | — |
| ViT-B/16 (100ep) | Standard | — | — | Crashed |
| ViT-B/16 (100ep) | QKNorm-DS | 79.0 | 44.4 | baseline |
| ViT-B/16 (100ep) | **QUEST** | **79.7** | **42.9** | Stable |
| ViT-L/16 (100ep) | Standard | — | — | Crashed |
| ViT-L/16 (100ep) | QKNorm-DS | 72.5 | 54.4 | baseline |
| ViT-L/16 (100ep) | **QUEST** | **74.9** | **50.3** | +2.4 |

Standard attention diverges/crashes on ViT-B/L, whereas QUEST converges stably and outperforms QKNorm-DS. Under DeiT-3 long training, ViT-B improves from 82.7 to 83.2.

### Ablation Study
Adversarial robustness (ViT-Ti/16, DeiT training, Top-1):

| Config | Clean | FGSM | PGD | Auto |
|--------|-------|------|-----|------|
| Standard | 72.50 | 54.23 | 43.65 | 26.57 |
| **QUEST** | **73.33** | **56.90** | 45.26 | 27.29 |
| Elliptical | 71.53 | 55.96 | 46.30 | 27.35 |
| **Elliptical-QUEST** | 72.48 | 56.39 | **47.25** | **28.54** |

QUEST is more robust than standard attention across all attacks while maintaining higher clean accuracy. Elliptical-QUEST combines the strengths of both, achieving the highest adversarial robustness while recovering clean accuracy lost by pure Elliptical. On ADE20K segmentation, ViT-Ti improved clean mIoU from 37.34 to 38.87 and corruption mIoU from 32.19 to 33.55.

### Key Findings
- **Normalizing only keys stabilizes the entire chain**: Although only keys are constrained, the query norm and maximum attention logit are also stabilized, which is the direct reason QUEST converges while standard attention diverges.
- **Stability $\neq$ Sacrificing Expressivity**: QKNorm exchanges stability for global uniform scaling, losing points on small models. QUEST preserves per-token query norm sharpness, achieving both stability and accuracy.
- **Robustness gains concentrate on IN-C / IN-A and adversarial attacks**: QUEST forces attention to spread more evenly over relevant object regions rather than fixating on a few highly salient instances, making it less prone to misclassification under input perturbation/noise (supported by AG-CAM interpretability maps).

## Highlights & Insights
- **"Decomposing attention to view norm roles" is the most critical step**: By separating query norm (sharpness control, useful) from key norm (global attention grabbing, harmful), the rationality of the method becomes immediate—single-side normalization is a precise excision of harmful degrees of freedom.
- **Near-zero cost drop-in modification**: Removing the scale factor and applying row-wise $\ell_2$ normalization to keys requires just a few lines of code and is universal across vision, language, graphs, etc.
- **Orthogonal and Additive**: Combining with Elliptical Attention into Elliptical-QUEST shows that "key sphericalization" is an independent dimension of improvement that can be stacked with others.
- **Elegant Toy Experiment Design**: Visualizing the failure mechanism of instability/spurious correlation as a growth curve of key norms—using "robust signals vs shortcuts valid for half the samples"—is more persuasive than just reporting metrics.

## Limitations & Future Work
- The authors specifically limit QUEST to **softmax attention**, not covering linear/non-softmax attention; since linear attention also suffers from entropy collapse, expansion is left for future work.
- Experiments are **primarily vision-based**. Language modeling, graphs, and point clouds serve only as proof of "generality," lacking systematic validation on large-scale LLM pre-training.
- Whether removing the scaling factor $C=1$ remains stable beyond the models tested (up to 2B ViT) remains to be verified at larger scales.
- Personal Observation: Robustness gains of QUEST vs. Elliptical-QUEST are inconsistent across different attacks (e.g., Elliptical is stronger on PGD, QUEST has higher clean accuracy on Auto), suggesting selection should be weighed by specific task requirements.

## Related Work & Insights
- **vs Standard Attention (SDPA)**: SDPA relies on fixed $1/\sqrt{D_H}$ scaling but can still suffer logit explosion due to arbitrary growth of query/key norms. QUEST mechanically blocks this via key sphericalization without needing scaling factors.
- **vs QKNorm (HS/DS)**: QKNorm normalizes both and adds learnable scaling, stabilizing large ViTs but locking per-token sharpness with global scaling, causing drops on small models. QUEST normalizes only keys, preserving query norms for stability without loss of expressivity.
- **vs QNorm (Only query normalized)**: QNorm is a symmetric counterpart to QUEST. It mitigates cross-coupling but allows key norms to grow, failing to prevent global attention grabbing and resulting in lower robustness and success rates than QUEST.
- **vs Elliptical Attention**: Elliptical uses Mahalanobis metrics to boost robustness at the cost of clean accuracy. QUEST is orthogonal; Elliptical-QUEST achieves stronger robustness while recovering clean accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear insight into "separating query/key norm roles $\rightarrow$ single-side normalization," filling a gap in literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers classification, segmentation, adversarial, and multiple domains with toy mechanism analysis; lacks large-scale LLM validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from norm decomposition to method; provides excellent clarity and interpretability.
- Value: ⭐⭐⭐⭐ Zero-cost drop-in, cross-domain generality, and stackable with other attention improvements; useful for both engineering and research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hilbert-Guided Sparse Local Attention](hilbert-guided_sparse_local_attention.md)
- [\[ICLR 2026\] Neural Dynamics Self-Attention for Spiking Transformers](neural_dynamics_self-attention_for_spiking_transformers.md)
- [\[CVPR 2026\] Graph Attention Prototypical Network for Robust Few-Shot Classification](../../CVPR2026/others/graph_attention_prototypical_network_for_robust_few-shot_classification.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](../../ACL2025/others/unique_hard_attention_a_tale_of_two_sides.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
