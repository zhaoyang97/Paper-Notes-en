---
title: >-
  [Paper Note] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment
description: >-
  [CVPR 2026][Multimodal VLM][False negatives] This paper proposes FALCON, a **learning-based mini-batch construction strategy** that employs a negative mining scheduler to adaptively balance the trade-off between hard negatives and false negatives, substantially improving cross-modal alignment quality in vision-language pretraining (VLP).
tags:
  - CVPR 2026
  - Multimodal VLM
  - False negatives
  - contrastive learning
  - vision-language pretraining
  - negative mining
  - mini-batch construction
  - scheduler
date: 2026-05-08
content_hash: 238921b3a8830c2c
---

# FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment

**Conference**: CVPR 2026
**arXiv**: [2505.11192](https://arxiv.org/abs/2505.11192)
**Code**: To be confirmed
**Area**: Object Detection / Vision-Language Pretraining
**Keywords**: False negatives, contrastive learning, vision-language pretraining, negative mining, mini-batch construction, scheduler

## TL;DR

This paper proposes FALCON, a **learning-based mini-batch construction strategy** that employs a negative mining scheduler to adaptively balance the trade-off between hard negatives and false negatives, substantially improving cross-modal alignment quality in vision-language pretraining (VLP).

## Background & Motivation

**False negatives as a core challenge in VLP**: Large-scale web-crawled datasets exhibit many-to-many image-text correspondences. Highly similar "negative" samples in contrastive learning may in fact be true matches (false negatives), introducing contradictory supervision signals.

**The dilemma of hard negative mining**: Selecting negatives highly similar to the anchor accelerates learning, but higher similarity implies greater false-negative risk; selecting low-similarity negatives yields insufficient information.

**The optimal similarity range is dynamic**: Anchors with simpler semantics have compact positive distributions and can safely support harder negative mining, whereas semantically complex anchors exhibit noisier embeddings and require more conservative strategies. This optimal range shifts continuously throughout training.

**Pretrained model assistance is insufficient**: Methods such as MAFA filter false negatives using ITM scores from a fixed pretrained model, but tend to assign low scores to complex semantic pairs even when they genuinely match. Fixed thresholds are either too conservative or too permissive.

**Heuristic scheduling lacks flexibility**: Fixed hardness (e.g., $q=1.0$ in GRIT-VLP) or progressive curriculum strategies (Progressive-Hardening/Softening) cannot capture instance-level or training-stage-level dynamics.

**Existing methods rely on hyperparameters and generalize poorly**: Two-stage select-then-filter frameworks are highly sensitive to thresholds, with false-negative rates reaching up to 60%.

## Method

### Overall Architecture

FALCON replaces conventional uniform sampling or fixed hard negative mining in VLP training with a learnable **negative mining scheduler** $\pi_\phi$. Building on GRIT-VLP's grouping strategy, the dataset is partitioned into local search spaces $\{M\}$. The pipeline proceeds as follows:

1. Uniformly sample initial candidates from a local search space $M$ as anchors.
2. The scheduler $\pi_\phi$ takes the **current normalized similarity distribution $\widehat{\mathbf{S}}$** as input and predicts a **hardness quantile** $q \in [0,1]$ for each anchor.
3. Samples at the predicted similarity level are selected from the candidate pool and added to the mini-batch ($q=1.0$ degenerates to GRIT-VLP; $q=0.0$ selects the easiest negatives).
4. This process is applied recursively until the mini-batch reaches size $B$, excluding already-selected samples at each step.
5. The constructed mini-batch is used to update the VLP model parameters $\theta$, and policy gradients are computed to update the scheduler parameters $\phi$.

### Scheduler Design

- **Input construction**: The cosine similarity matrices for I2T and T2I are summed to form a unified similarity matrix $\mathbf{S}$. Each row is compressed by sampling $m$ uniformly spaced quantiles ($m \ll |M|$), followed by row-wise softmax normalization to obtain $\widehat{\mathbf{S}}$, eliminating the effect of similarity scale drift during training.
- **Network architecture**: A lightweight 4-layer residual MLP maps $\widehat{\mathbf{S}}$ to Beta distribution parameters $(\alpha, \beta)$, from which the hardness quantile $q$ is sampled.
- **Permutation equivariance**: Rows of $\widehat{\mathbf{S}}$ are sorted before being fed into the network, achieving permutation equivariance in a lightweight manner without resorting to heavy architectures such as Transformers.
- **Instance-level scheduling**: A hardness quantile is predicted independently for each anchor rather than sharing a single threshold at the batch level.
- **Cache reuse**: The similarity matrix is computed from existing CLS embedding queues, requiring no additional forward passes.

### Loss & Training

The training signal for the scheduler is the **decrease in MLM loss**, used as a proxy for cross-modal alignment improvement:

$$\phi_{k+1} = \phi_k + \gamma \cdot \mathbb{E}_{\pi_{\phi_k}}\left[\Delta_k^{V,T} \cdot \nabla_{\phi_k} \log \pi_{\phi_k}(V,T|\widehat{\mathbf{S}})\right]$$

where $\Delta_k^{V,T} = \mathcal{L}_{\text{MLM}}(V,T;\theta_k) - \mathcal{L}_{\text{MLM}}(V,T;\theta_{k+1})$. MLM is chosen over ITC/ITM as the proxy because contrastive objectives incentivize the scheduler to select trivially easy negatives to minimize loss, which is counterproductive to effective training.

## Key Experimental Results

### Main Results: Comparison with Heuristic Negative Mining Methods (Pretrained on MSCOCO)

| Method | TR R@1 | TR R@5 | IR R@1 | IR R@5 | VQA test-dev | NLVR2 dev |
|--------|--------|--------|--------|--------|-------------|-----------|
| ALBEF | 55.60 | 81.92 | 41.16 | 70.63 | 70.46 | 72.98 |
| GRIT-VLP | 60.60 | 83.52 | 44.61 | 69.54 | 71.04 | 74.63 |
| MAFA | 60.96 | 83.24 | 44.77 | 69.49 | 71.13 | 75.16 |
| **FALCON** | **62.28** | **86.18** | **46.18** | **74.65** | **71.24** | **75.17** |

### Cross-Framework Compatibility (BLIP-2 & SigLIP-2)

| Framework | Baseline COCO TR R@1 | +FALCON TR R@1 | Baseline COCO IR R@1 | +FALCON IR R@1 |
|-----------|----------------------|----------------|----------------------|----------------|
| BLIP-2 | 75.22 | **75.56** | 57.98 | **58.52** |
| SigLIP-2 | 69.96 | **72.96** | 54.21 | 54.15 |

### Ablation Study

**Effect of search space size**:

- $|M|=480$ → TR R@1: 58.48; $|M|=5664$ → 61.72; $|M|=28320$ → 61.94
- FALCON is robust to large search spaces, whereas baseline methods (e.g., GRIT-VLP) degrade due to increased false negatives.

**Training objective selection**:

- $\mathcal{L}_\text{MLM}$ only: TR R@1 = 61.72 (best)
- $\mathcal{L}_\text{ITC}+\mathcal{L}_\text{ITM}$: TR R@1 = 57.64 (significant drop)
- $\mathcal{L}_\text{ITC}+\mathcal{L}_\text{ITM}+\mathcal{L}_\text{MLM}$: TR R@1 = 57.80
- Conclusion: Contrastive objectives induce the scheduler to select trivially easy negatives; the generative objective (MLM) serves as a more principled proxy.

**Scheduling granularity**:

- Instance-level scheduling (61.72) substantially outperforms batch-level scheduling (58.78), confirming that optimal hardness is anchor-dependent and that a uniform threshold cannot accommodate samples of varying semantic complexity.

### Key Findings

- **Adaptive scheduling behavior**: In early training, FALCON tends to sample high quantiles (hard negatives) to accelerate embedding learning; as the embedding space matures and false negatives accumulate at high quantiles, the scheduler automatically lowers its quantile estimates to avoid false-negative contamination.
- **4M standard dataset**: FALCON achieves the best performance on 4M web-noisy data (including CC and SBU), with COCO zero-shot TR R@1 of 74.1 vs. MAFA 72.6 vs. ALBEF 68.7.
- **Convergence efficiency**: Training cost is 0.83C relative to ALBEF, slightly higher than GRIT-VLP (0.65C) and MAFA (0.76C), but the performance-time curve (Recall@1 vs. wall-clock time) consistently dominates all baselines.
- **Robustness to search space**: FALCON improves steadily and then plateaus as $|M|$ grows from 480 to 28320, whereas GRIT-VLP degrades at larger search spaces due to increasing false negatives.

## Highlights & Insights

- **First learning-based negative scheduling method**: Elevates the hard/false-negative trade-off from manual heuristics to a learnable optimization problem, establishing a new paradigm for negative sample management in contrastive learning.
- **Elegant and efficient design**: A 4-layer residual MLP with Beta distribution parameterization and row sorting for equivariance incurs minimal computational overhead and does not become a training bottleneck.
- **Clear theoretical motivation**: Using MLM loss decrease as a cross-modal alignment proxy is well-justified; experiments empirically verify that contrastive objectives induce a trivial-negative trap, providing a sound rationale for the design choices.
- **Broad applicability**: Demonstrated effectiveness across three architecturally distinct VLP frameworks—ALBEF (fusion-based), BLIP-2 (Q-Former), and SigLIP-2 (dual-encoder)—establishing the generality of the approach.
- **Thorough visualization analysis**: Visualizations of scheduler behavior over training, quantile sampling examples, and similarity distribution evolution intuitively illustrate the adaptive mechanism.

## Limitations & Future Work

- Gains on the text side of SigLIP-2 are limited (IR R@1 shows virtually no improvement), as the auxiliary generative loss passes only through the visual encoder, biasing the scheduling signal toward the visual modality.
- Gains are reduced on heavily noisy web datasets compared to the clean MSCOCO setting, as semantically misaligned raw captions interfere with hardness estimation.
- The scheduler requires additional forward passes and updates, resulting in a slightly higher per-epoch training cost than GRIT-VLP and MAFA (0.83C vs. 0.65C/0.76C).
- The similarity matrix is constructed from cached CLS embeddings; embedding staleness may affect precision in early training.
- The proxy signal requires association with both modality encoders to be fully effective; generative objectives confined to a single modality yield limited benefits.
- Evaluation is currently limited to retrieval, VQA, and NLVR2 tasks; validation on broader downstream applications (e.g., visual grounding, image generation) remains to be conducted.

## Related Work & Insights

- **Hard negative mining**: GRIT-VLP (fixed maximum hardness at $q=1.0$ with grouped search spaces), DiHT (debiased contrastive learning), SRCL (self-regularized contrastive learning).
- **False negative handling**: MAFA (ITM-threshold filtering with re-labeling via a pretrained model), FFF (fixing flawed foundations in contrastive pretraining), VL-Match (token- and instance-level matching enhancement).
- **Learning to optimize / meta-learning**: Learning to learn by gradient descent, Neural optimizer search (RL-based optimizer search).
- **Vision-language pretraining**: ALBEF (momentum distillation + ITC/ITM/MLM), BLIP series (unified understanding and generation), SigLIP-2 (sigmoid contrastive + generative objectives), CLIP/ALIGN (large-scale contrastive pretraining).

## Rating

- Novelty: ⭐⭐⭐⭐ — First formulation of negative hardness scheduling as a learnable optimization problem; Beta distribution parameterization and MLM proxy signal are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three VLP frameworks, multiple downstream tasks, comprehensive ablations, training dynamics visualization, and wall-clock time comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Well-motivated (the false-negative rate analysis in Figure 2 is particularly persuasive), complete derivations, and clear figures.
- Value: ⭐⭐⭐⭐ — Plug-and-play applicability across frameworks; the false-negative problem is pervasive in large-scale VLP, giving the method strong practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)
- [\[CVPR 2026\] The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment](the_more_the_merrier_contrastive_fusion_for_higher-order_multimodal_alignment.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noiseaware_fewshot_learning_through_bidirectional.md)
- [\[CVPR 2026\] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](tipsv2_patch_text_alignment.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](../../NeurIPS2025/multimodal_vlm/aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)

<!-- RELATED:END -->
