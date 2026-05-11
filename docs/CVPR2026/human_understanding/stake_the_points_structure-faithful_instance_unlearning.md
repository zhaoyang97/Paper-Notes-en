---
title: >-
  [Paper Note] Stake the Points: Structure-Faithful Instance Unlearning
description: >-
  [CVPR 2026][Human Understanding][machine unlearning] This paper proposes Structguard, which leverages semantic anchors to preserve the semantic relational structure among retained instances during the forgetting process, thereby preventing structural collapse. The method achieves average improvements of 32.9% / 19.3% / 22.5% across image classification, face recognition, and retrieval tasks.
tags:
  - CVPR 2026
  - Human Understanding
  - machine unlearning
  - instance-level unlearning
  - structural preservation
  - semantic anchors
  - CLIP
date: 2026-05-08
content_hash: b197a5d46f6e0b01
---

# Stake the Points: Structure-Faithful Instance Unlearning

**Conference**: CVPR 2026
**arXiv**: [2603.12915](https://arxiv.org/abs/2603.12915)
**Code**: To be confirmed
**Area**: Human Understanding
**Keywords**: machine unlearning, instance-level unlearning, structural preservation, semantic anchors, CLIP

## TL;DR

This paper proposes Structguard, which leverages semantic anchors to preserve the semantic relational structure among retained instances during the forgetting process, thereby preventing structural collapse. The method achieves average improvements of 32.9% / 19.3% / 22.5% across image classification, face recognition, and retrieval tasks.

## Background & Motivation

1. **Data protection regulations**: Regulations such as GDPR require models to remove the influence of specific user data. Retraining from scratch is prohibitively expensive, motivating the study of Machine Unlearning (MU).
2. **Instance-level unlearning is more practical**: Real-world deletion requests typically target specific individuals rather than entire classes, making instance-level unlearning more realistic than class-level unlearning.
3. **Existing methods neglect semantic structure**: Prior MU methods (e.g., Neggrad, Adv, L2UL) disrupt the semantic relationships among retained instances when erasing target instances, leading to progressive structural collapse in the representation space.
4. **Structural collapse negatively correlates with performance**: Experiments reveal a significant negative correlation between the degree of structural collapse and the forget–retain accuracy balance; better structure preservation leads to better unlearning performance.
5. **No retain set required**: In real-world scenarios, the original training data is often inaccessible due to policy or storage constraints; the proposed method relies solely on the pretrained model and the data to be forgotten.
6. **Knowledge is encoded in relationships**: Knowledge in deep models is not stored in isolation but organized through semantic relationships; the unlearning process must therefore protect this relational structure.

## Method

### Overall Architecture: Structguard

**Core Idea**: Semantic anchors (stakes) are introduced as fixed reference points for retained instances. During unlearning, each instance is bound to an anchor to prevent semantic drift and maintain the structured organization of knowledge.

### 1. Anchor Generation

- For each class $c$, GPT-4o is used to generate attribute descriptions (texture, shape, typical context, etc.).
- The concatenated descriptions are fed into a frozen semantic encoder $T(\cdot)$ (CLIP ViT-B/32) to obtain class-level anchor $a_c$.
- All anchors form the matrix $A \in \mathbb{R}^{b \times d}$, which remains fixed throughout the unlearning process.

### 2. Structure Definition and Proxy Set

- **Original structure** $S^{\text{ori}} = V^{\text{ori}} \cdot A^\top$: affinity matrix between retained instance embeddings and anchors.
- **Proxy set**: Since the retain set is inaccessible, adversarial variants of the forget samples are generated to approximate the embeddings of retained instances.
- **Unlearned structure** $S^{\text{unl}} = V^{\text{unl}} \cdot A^\top$: affinity matrix between embeddings projected through a learnable projector $p_\omega$ and the anchors.

### 3. Structure-Preserving Constraints

**Structure-aware Alignment**:

$$\mathcal{L}_{\text{align}} = -\frac{1}{b} \sum_{i=1}^{b} \cos(S_i^{\text{ori}}, S_i^{\text{unl}})$$

This maximizes the cosine similarity between pre- and post-unlearning structures to preserve relative anchor–instance patterns.

**Structure-aware Regularization**:

$$\mathcal{L}_{\text{reg}} = \frac{1}{2} \sum_i I_i \cdot (\psi_i^{\text{unl}} - \psi_i^{\text{ori}})^2$$

where $I_i$ is the structural importance score of the $i$-th parameter (estimated via the absolute gradient of the alignment loss), which suppresses large updates to structurally critical parameters.

### 4. Overall Loss

- **Retention loss** $\mathcal{L}_{\text{ret}}$: cross-entropy through the projector to preserve semantic relationships.
- **Deletion loss** $\mathcal{L}_{\text{del}}$: negative cross-entropy bypassing the projector to achieve effective erasure.
- Total loss $= \mathcal{L}_{\text{del}} + \mathcal{L}_{\text{ret}} + \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{reg}}$

## Key Experimental Results

### Image Classification (CIFAR-10 / CIFAR-100 / ImageNet-1K)

| Method | CIFAR-10 $\mathcal{A}_{\text{test}}$ (k=256) | CIFAR-100 $\mathcal{A}_{\text{test}}$ (k=256) | ImageNet-1K $\mathcal{A}_{\text{test}}$ (k=256) | $\mathcal{A}_f$ |
|------|:---:|:---:|:---:|:---:|
| L2UL | 45.44 | 48.71 | 31.19 | 100.0 |
| Adv | 36.69 | 46.45 | 21.27 | 100.0 |
| **Structguard** | **56.32** | **56.91** | **41.15** | **100.0** |

- On CIFAR-10 (k=256), Structguard surpasses Oracle by 17.73% ($\mathcal{A}_{\text{test}}$) and 21.77% ($\mathcal{A}_r$).
- On ImageNet-1K, Structguard outperforms all baselines by an average of 21.57% ($\mathcal{A}_{\text{test}}$).
- As $k$ increases, Structguard degrades far less than L2UL (on CIFAR-100, L2UL drops 22.21% vs. Structguard's 9.68%).

### Face Recognition (Lacuna-10)

| Method | k=3 $\mathcal{A}_{\text{test}}$ | k=64 $\mathcal{A}_{\text{test}}$ | $\mathcal{A}_f$ |
|------|:---:|:---:|:---:|
| L2UL | 75.37 | 12.26 | 100.0 |
| **Structguard** | **77.29** | **27.71** | **100.0** |

Structguard outperforms L2UL by an average of 5.92% ($\mathcal{A}_{\text{test}}$) and 5.23% ($\mathcal{A}_r$).

### Ablation Study

| SA | SR | CR | CIFAR-10 $\mathcal{A}_{\text{test}}$ | CIFAR-100 $\mathcal{A}_{\text{test}}$ |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✓ | ✓ | Largest drop | Largest drop |
| ✓ | ✗ | ✓ | Slight drop | Moderate drop |
| ✓ | ✓ | ✗ | Moderate drop | Slight drop |
| ✓ | ✓ | ✓ | **Best** | **Best** |

- **SA (Structure-aware Alignment)** is the most critical component; its removal causes the largest performance degradation.
- On CIFAR-10, CR > SR (classifier regularization is more important when there are fewer classes); on CIFAR-100, SR > CR (parameter constraints are more important when there are more classes).
- Anchor type: semantic anchors outperform visual prototype anchors (+7.84% on CIFAR-10), demonstrating that language-guided semantic anchors provide superior structured reference.

## Highlights & Insights

- **Conceptual novelty**: This work is the first to formalize "structure preservation" as a core objective of MU, revealing a causal relationship between structural collapse and the forget–retain accuracy balance.
- **Elegant semantic anchor design**: LLM-generated attribute descriptions encoded via CLIP construct stable, data-independent reference points.
- **Comprehensive three-task validation**: Significant improvements across classification, recognition, and retrieval demonstrate the generalizability of the method.
- **Excellent representation consistency**: Grad-CAM and cosine similarity analyses show that representations of retained samples are nearly unaffected by the unlearning process.
- **No retain set required**: The method relies solely on the pretrained model and the forget set, making it more applicable to real-world scenarios.

## Limitations & Future Work

- Dependence on CLIP and GPT-4o for anchor generation may be sensitive to model and prompt choices, and increases deployment cost.
- The proxy set approximates the retain set via adversarial samples; approximation quality may degrade when the number of forget samples is small.
- The projector $p_\omega$ introduces additional parameters and computational overhead.
- Evaluation is limited to the ResNet architecture; effectiveness on Transformer architectures such as ViT remains unverified.
- Sequential unlearning scenarios (i.e., whether anchors need to be updated across multiple rounds) are not discussed.
- Class-level anchors have limited capacity to capture intra-class diversity; fine-grained scenarios may require sub-class anchors.

## Related Work & Insights

| Method | Objective | Granularity | Retain Set Required | Structure Preserved |
|------|------|------|:---:|:---:|
| Fisher [Golatkar'20] | undo | instance | ✓ | ✗ |
| UNSIR [Tarun'23] | undo | class | ✓ | ✗ |
| L2UL [Chen'24] | misclassify | instance | ✗ | ✗ |
| LoTUS [Kim'24] | undo | instance | ✓ | ✗ |
| **Structguard** | misclassify | instance | ✗ | **✓** |

Structguard is the first instance-level unlearning method that simultaneously satisfies "no retain set required" and "structure preservation." It shares the misclassification objective and retain-set-free setting with L2UL, but explicitly maintains knowledge structure via semantic anchors, achieving comprehensive improvements across all tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The structure preservation perspective is novel, and the semantic anchor design is particularly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across three tasks with rich ablation studies, visualizations, and anchor analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear figures and rigorously reasoned motivation.
- **Value**: ⭐⭐⭐⭐ — Introduces a new structure-preserving paradigm for the MU field with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[CVPR 2026\] HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](hum4d_markerless_motion_capture.md)
- [\[CVPR 2026\] RAM: Recover Any 3D Human Motion in-the-Wild](ram_recover_any_3d_human_motion_in-the-wild.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)

</div>

<!-- RELATED:END -->
