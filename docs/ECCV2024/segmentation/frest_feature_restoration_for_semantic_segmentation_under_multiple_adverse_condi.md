---
title: >-
  [Paper Note] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions
description: >-
  [ECCV 2024][Segmentation][semantic segmentation] This paper proposes FREST, a source-free domain adaptation semantic segmentation framework for multiple adverse conditions (fog, rain, snow, night). By alternating between learning a condition embedding space (to extract condition-specific information) and feature restoration (to recover adverse condition features back to normal), it progressively eliminates the impact of adverse conditions on features…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "semantic segmentation"
  - "feature restoration"
  - "source-free domain adaptation"
  - "adverse conditions"
  - "robustness"
date: 2026-05-08
content_hash: ca2d3c4de6f42001
---

# FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions

**Conference**: ECCV 2024  
**arXiv**: [2407.13437](https://arxiv.org/abs/2407.13437)  
**Code**: [https://sohyun-l.github.io/frest](https://sohyun-l.github.io/frest)  
**Area**: Image Segmentation  
**Keywords**: semantic segmentation, feature restoration, source-free domain adaptation, adverse conditions, robustness

## TL;DR

This paper proposes FREST, a source-free domain adaptation semantic segmentation framework for multiple adverse conditions (fog, rain, snow, night). By alternating between learning a condition embedding space (to extract condition-specific information) and feature restoration (to recover adverse condition features back to normal), it progressively eliminates the impact of adverse conditions on features, achieving new SOTAs on both ACDC and RobotCar benchmarks.

## Background & Motivation

**Background**: Semantic segmentation exhibits excellent performance under normal conditions but experiences significant degradation under adverse conditions such as fog, rain, snow, and night, which severely limits its applications in safety-critical scenarios like autonomous driving. To address the difficulty of obtaining annotated data, researchers have turned to unsupervised domain adaptation (UDA) and the more practical source-free domain adaptation (SFDA).

**Problem Setup (SFDA to adverse conditions)**: A segmentation model is pre-trained on a labeled source domain (e.g., Cityscapes normal weather) and then fine-tuned using only unlabeled target domain data. The target domain contains images under multiple adverse conditions. Each adverse condition image $I_{\text{adv}}$ has a corresponding normal condition reference image $I_{\text{norm}}$ matched via GNSS. However, they are only roughly aligned (with perspective and temporal differences), both are unlabeled, and the specific condition types are unknown.

**Limitations of Prior Work**: Previous work CMA encourages the features of adverse/normal image pairs to be close to each other through contrastive learning, aiming to learn condition-invariant features. However, this raises two core issues:
   - **Catastrophic Forgetting**: Pulling normal image features closer to adverse features causes the model to forget source domain knowledge, as normal images themselves are inherently close to the source domain.
   - **Alignment Dependency**: Feature alignment heavily relies on rigorous spatial alignment of image pairs. However, GNSS matching only provides coarse alignment, and dynamic objects or imperfect warping introduce content mismatches.

**Core Idea**: Rather than pulling features from both conditions closer to each other, this work proposes a **unidirectional** restoration of adverse condition features toward normal condition features. The key lies in leveraging "condition-specific information"—information that depends solely on image conditions and is independent of semantic content—to guide the restoration process, thereby preventing catastrophic forgetting and reducing the impact of content mismatches.

**Key Insight**: Adverse conditions are treated as "harmful noise," with the goal of removing their effects in the feature space. By designing an embedding space that captures only condition to run the feature restoration within, it ensures that the restoration process only considers condition discrepancies without being distracted by differences in semantic content.

## Method

### Overall Architecture

FREST alternates between two steps in each training iteration:
- **Step 1**: Freeze the segmentation network, train the Condition Strainer and the projection head, and learn the condition embedding space.
- **Step 2**: Freeze the Condition Strainer and the projection head, train the segmentation network, and perform feature restoration in the condition embedding space.

During inference, only the encoder $\phi_{\text{enc}}$ and decoder $\phi_{\text{dec}}$ are utilized, without requiring extra modules like the Condition Strainer.

### Key Designs

#### 1. Condition Strainer —— Extracting Condition Information

**Function**: Extracts condition-specific information that is relevant only to the image conditions and independent of semantic content from the encoder features.

**Mechanism**: Inspired by parameter-efficient fine-tuning (such as Adapter), a small module $\psi_{\text{strainer}}$ is appended alongside each layer of the frozen segmentation encoder $\phi_{\text{enc}}$, generating "condition-infused features" through residual connections:

$$\mathbf{c}^l = \phi_{\text{enc}}^l(\mathbf{c}^{l-1}) + \psi_{\text{strainer}}^l(\mathbf{c}^{l-1})$$

where $\mathbf{c}^l$ represents the condition-infused feature of the $l$-th layer. The Condition Strainer is designed separately from the encoder to prevent the encoder from being contaminated by condition information. Ultimately, the condition-infused feature $\mathbf{c}$ is mapped to the condition embedding space via a projection head $\psi_{\text{proj}}$.

**Design Motivation**: The encoder is pre-frozen with source-domain pre-training, preserving rich semantic knowledge. The Condition Strainer adds only a few parameters (2.1M, accounting for 2.6% of the baseline) to capture condition-specific info that the encoder cannot represent. This design is both parameter-efficient and prevents the destruction of source domain knowledge.

#### 2. Condition Embedding Space Learning (Step 1) —— Disentangling Conditions via Contrastive Learning

**Function**: Learns an embedding space where images of the same condition cluster together, while images of different conditions are separated.

**Mechanism**: Contrastive learning is employed where anchors and positive samples share the same condition but have different semantics, and anchors and negative samples share similar semantics but have different conditions:

$$\mathcal{L}_{\text{spec},i} = -\log \frac{\exp(\mathbf{z}_{\text{adv}}^{i\top} \mathbf{z}_{\text{adv}}^{*} / \tau)}{\exp(\mathbf{z}_{\text{adv}}^{i\top} \mathbf{z}_{\text{adv}}^{*} / \tau) + \exp(\mathbf{z}_{\text{adv}}^{i\top} \mathbf{z}_{\text{norm}}^{i} / \tau)}$$

where the anchor $\mathbf{z}_{\text{adv}}^i$ is the condition embedding of the adverse condition, the negative sample $\mathbf{z}_{\text{norm}}^i$ is the condition embedding of the corresponding normal condition (via warped patch pairs), and the positive sample $\mathbf{z}_{\text{adv}}^*$ is selected from the positive sample queue as the adverse condition embedding **most similar** to the anchor (assuming they share the same condition).

**Design Motivation**: By requiring anchors and negative samples to be semantically similar but conditionally different (since they are adverse/normal image patch pairs of the same spatial location), the embedding space is forced to focus only on condition discrepancies. The strategy of selecting the most similar positive sample ensures that positive samples share the identical adverse condition stage with the anchor, assisting the model in learning finer condition distinctions.

#### 3. Feature Restoration (Step 2) —— Eliminating Adverse Effects in the Condition Space

**Function**: Trains the segmentation network to make the encoder features $\mathbf{f}_{\text{adv}}$ of adverse condition images approximate the condition-infused features of normal conditions $\mathbf{c}_{\text{norm}}$ in the condition embedding space.

**Mechanism - Feature Restoration Loss**: After projecting $\mathbf{f}_{\text{adv}}$ into the condition embedding space, an $\ell_1$ regression loss is utilized to make it approach $\mathbf{z}_{\text{norm}}$:

$$\mathcal{L}_{\text{resto}} = \frac{1}{|\mathcal{W}|} \sum_{i \in \mathcal{W}} |\psi_{\text{proj}}(\mathbf{f}_{\text{adv}}^i) - \mathbf{z}_{\text{norm}}^i|$$

Note that gradients are not backpropagated through $\mathbf{c}_{\text{norm}}$, ensuring a unidirectional restoration of the adverse condition features toward the normal condition elements.

**Auxiliary Design - Adverse Condition Discriminant Loss**: An MLP discriminator $D$ is introduced to distinguish between encoder features $\mathbf{f}_{\text{adv}}^{l,j}$ and condition-infused features $\mathbf{c}_{\text{adv}}^{l,j}$, further driving the encoder features away from adverse condition details:

$$\mathcal{L}_{\text{dis}} = -\frac{1}{|\mathcal{A}|} \sum_{j \in \mathcal{A}} \sum_{l=1}^{L} \{\lambda \log(D(\mathbf{f}_{\text{adv}}^{l,j})) + (1-\lambda) \log(1 - D(\mathbf{c}_{\text{adv}}^{l,j}))\}$$

**Design Motivation**: Performing restoration in the condition embedding space focuses only on condition information while ignoring differences in semantic content. This alleviates the issue of imprecise image pair alignment—even if the contents of the two images do not perfectly match, as long as the condition information is restored. The discriminant loss provides additional gradient signals to push features of each layer to rid themselves of adverse condition feature patterns.

### Loss & Training

**Step 1 Total Loss**: $\mathcal{L}_{\text{step1}} = \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{self}}$

**Step 2 Total Loss**: $\mathcal{L}_{\text{step2}} = \mathcal{L}_{\text{resto}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}} + \mathcal{L}_{\text{self}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$

where $\mathcal{L}_{\text{self}}$ is the self-training loss with pseudo-labels (CBST), and $\mathcal{L}_{\text{ent}}$ is the entropy minimization loss. The two steps alternate, allowing the condition strainer to adapt to the updates of the segmentation network, which in turn improves feature restoration in the subsequent round.

Hyperparameters: $\lambda_{\text{spec}} = 0.01$, $\lambda_{\text{ent}} = 0.01$, $\lambda_{\text{dis}} = 5 \times 10^{-5}$, $\tau = 0.7$. Training spans 8 epochs, with the first 2 epochs training only the Condition Strainer as a warmup.

## Key Experimental Results

### Main Results

**Cityscapes → ACDC (SFDA Setting)**

| Method | mIoU (%) | Gain (vs. Source model) |
|------|----------|---------------------|
| Source model (SegFormer) | 59.4 | - |
| HCL | 60.8 | +1.4 |
| URMA | 65.3 | +5.9 |
| URMA + SimT | 65.7 | +6.3 |
| CMA | 69.1 | +9.7 |
| **FREST (Ours)** | **70.7**| **+11.3** |

FREST improves by 1.6% on this benchmark over the previous SOTA CMA, with significant improvements on fine-grained objects (car +2.9, truck +6.5, bus +2.8).

**Cityscapes → RobotCar (SFDA Setting, 8 Adverse Conditions)**

| Method | mIoU (%) |
|------|----------|
| Source model | 50.0 |
| HCL | 50.1 |
| URMA | 51.6 |
| CMA | 54.3 |
| **FREST (Ours)** | **58.8** |

RobotCar contains more diverse conditions (dawn, dusk, night, night-rain, overcast, rain, snow, sun). FREST achieves an outstanding 4.5% improvement over CMA, indicating that feature restoration grows more effective as the diversity of conditions increases.

**Comparison with UDA Methods (Cityscapes → ACDC)**

| Method | Source-free | mIoU (%) |
|------|------------------|----------|
| Refign (UDA) | No | 65.5 |
| HRDA (UDA) | No | 68.0 |
| HRDA + MIC (UDA) | No | 70.4 |
| CMA (SFDA) | Yes | 69.1 |
| **FREST (SFDA)** | **Yes** | **70.7** |

As an SFDA method, FREST outperforms all UDA methods without requiring access to source domain labels.

### Ablation Study

**Ablation on Loss Functions (Step 1 & Step 2)**

| $\mathcal{L}_{\text{self}}$ | $\mathcal{L}_{\text{spec}}$ | mIoU (%) | Explanation |
|:---:|:---:|---:|------|
| | | 64.3 | No-loss baseline |
| ✓ | | 64.8 | Self-training only |
| ✓ | ✓ | 68.6 | Step 1 complete, condition learning contribution +3.8 |

| $\mathcal{L}_{\text{resto}}$ | $\mathcal{L}_{\text{dis}}$ | mIoU (%) | Explanation |
|:---:|:---:|---:|------|
| | | 62.7 | No restoration loss |
| ✓ | | 67.2 | Restoration loss contribution +4.5 |
| ✓ | ✓ | 68.6 | Discriminant loss extra contribution +1.4 |

**Ablation on Training Strategies & Architectures**

| Condition Strainer | Segmentation Network | Training Strategy | mIoU (%) |
|:---:|:---:|---:|---:|
| | ✓ | Self-training | 62.7 |
| ✓ | | Adapter fine-tuning | 63.1 |
| ✓ | ✓ | Full parameter fine-tuning | 63.2 |
| ✓ | ✓ | **FREST Alternating Training** | **68.6** |

Key Findings: Naively tuning with Adapter only boosts performance by 0.4%, full parameter fine-tuning yields a 0.5% boost, while FREST alternating training strategy contributes a huge 5.9% improvement.

### Key Findings

1. **Feature Restoration Visualization**: Reconstructing images from restored features shows night skies turning blue and snowy ground trees turning green, confirming that feature restoration effectively simulates normal conditions.
2. **Parameter-efficient**: The Condition Strainer requires only 2.1M parameters (2.6% of the baseline), and the projection head requires 1.2M (1.5%), with zero extra parameter overhead during inference.
3. **Positive Sample Selection Strategy**: Selecting the most similar positive sample (HIGHEST) achieves 68.6%, significantly outperforming random selection (62.4%) and the least similar selection (56.6%).
4. **Restored Features vs. Condition-infused Features**: Using restored features $\mathbf{f}_{\text{adv}}$ at inference yields 68.6% mIoU, whereas using condition-infused features $\mathbf{c}_{\text{adv}}$ yields only 59.0%, demonstrating that the restoration process successfully eliminates adverse condition information.
5. **Generalizability**: On the unseen dataset ACG, FREST achieves 52.6% mIoU (CMA 51.3%), and on normal conditions (Cityscapes-lindau40), it reaches 72.5% (on par with the source model), proving that feature restoration does not damage performance under normal conditions.

## Highlights & Insights

1. **Unidirectional Restoration vs. Bidirectional Alignment**: The core insight is formulating the adverse-to-normal translation as a unidirectional process, preventing normal features from being "contaminated" by adverse features, which avoids catastrophic forgetting.
2. **Restoration in Condition Space**: Operating in an embedding space that contains only condition information elegantly bypasses the challenge of imprecise image pair alignment.
3. **Mutual Promotion in Alternating Training**: The two-step alternating training allows the condition space and feature restoration to improve each other—a better condition space guides better restoration, and better-restored features aid in learning a more accurate condition space.
4. **Parameter-Efficient Design**: Drawing inspiration from the Adapter architecture, the Condition Strainer is completely eliminated during inference, entailing zero extra inference cost.

## Limitations & Future Work

1. Currently, it only covers natural adverse conditions like weather and illumination, leaving out broader degradation types like image blur, noise, and camera artifacts.
2. The Condition Strainer depends on the quality of warping alignment; regions with warping failures (confidence < 0.2) cannot participate in learning.
3. The positive sample queue assumes that the most similar embeddings share identical conditions, which may fail when condition distributions are highly mixed.
4. This work does not address complex compound scenarios where multiple adverse conditions occur simultaneously (e.g., snow + night).

## Related Work & Insights

- **CMA [Brüggemann et al.]**: Pioneering work in SFDA under adverse conditions, which proposes condition-invariant learning. FREST improves upon it via unidirectional restoration.
- **Parameter-efficient Fine-tuning**: Architectures like Adapter and LoRA inspired the design of the Condition Strainer.
- **Contrastive Learning**: The positive/negative sample construction strategy for condition-specific learning is highly inspiring, utilizing warped alignments to obtain semantically similar but conditionally different pairs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The feature restoration concept is novel. The design of unidirectional restoration combined with condition embedding space effectively addresses the core shortcomings of existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Demonstrates SOTA performance on two benchmarks, along with UDA comparisons, generalization evaluation, rigorous ablation studies, and qualitative visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, detailed method descriptions, and informative illustrations.
- **Value**: ⭐⭐⭐⭐ — Highly practical for safety-critical scenes like autonomous driving; the SFDA setup is highly realistic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](../../CVPR2026/segmentation/heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[ECCV 2024\] Eliminating Feature Ambiguity for Few-Shot Segmentation](eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [\[ECCV 2024\] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures](representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)
- [\[ECCV 2024\] LiFT: A Surprisingly Simple Lightweight Feature Transform for Dense ViT Descriptors](lift_a_surprisingly_simple_lightweight_feature_transform_for_dense_vit_descripto.md)

</div>

<!-- RELATED:END -->
