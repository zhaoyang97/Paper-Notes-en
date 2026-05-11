---
title: >-
  [Paper Note] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection
description: >-
  [AAAI 2026][Object Detection][Open-vocabulary detection] VK-Det is proposed as a framework that leverages only the visual knowledge of VLMs (without any additional supervision signals) to achieve state-of-the-art perform…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Open-vocabulary detection"
  - "aerial remote sensing imagery"
  - "prototype learning"
  - "knowledge distillation"
  - "pseudo-labels"
date: 2026-05-08
content_hash: e20755677ee50f1d
---

# VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection

**Conference**: AAAI 2026
**arXiv**: [2511.18075](https://arxiv.org/abs/2511.18075)
**Code**: None
**Area**: Object Detection
**Keywords**: Open-vocabulary detection, aerial remote sensing imagery, prototype learning, knowledge distillation, pseudo-labels

## TL;DR

VK-Det is proposed as a framework that leverages only the visual knowledge of VLMs (without any additional supervision signals) to achieve state-of-the-art performance in open-vocabulary aerial object detection through Adaptive Selection Knowledge Distillation (ASKD), Prototype-Aware Pseudo-Label generation (PAPL), and Synthetic Matching Inference (SMI), even surpassing methods that rely on extra supervision.

## Background & Motivation

**Aerial Object Detection (AOD)** is a core task in earth observation (security surveillance, disaster response, urban management). Existing deep learning methods can only detect predefined categories and cannot handle the large number of unannotated concepts encountered in real-world deployments. **Open-Vocabulary Aerial Object Detection (OVAD)** has emerged to address this by leveraging the zero-shot capabilities of VLMs to generalize from base categories to novel categories.

**Core problems with existing methods**:

**Limitations of knowledge distillation**: Conventional distillation methods (e.g., ViLD) apply distillation to all proposals, introducing substantial noisy background features and reducing distillation efficiency. The prevalence of small objects, extreme aspect ratios, and high inter-class local similarity in aerial imagery further exacerbates this issue.

**Text dependency in pseudo-labels**: Existing pseudo-label methods rely on VLM text encoders to classify regions into specific novel categories. However, this text dependency introduces **semantic bias**—constraining open-vocabulary extension to textually specified concepts—and text embeddings may produce "hallucinations" over image regions, causing category boundary misalignment.

**Core reflection**: Can novel-category objects be automatically discovered from visual knowledge alone, while maximizing category boundaries to facilitate knowledge transfer?

**Key observation**: Averaging multi-layer attention heatmaps from a VLM visual encoder can distinguish background from informative regions and assigns higher weights to the latter without any labels, revealing the VLM's inherent informative region awareness capability.

## Method

### Overall Architecture

VK-Det consists of three modules: ASKD and PAPL are used during training, while SMI is applied at inference:

1. **ASKD (Adaptive Selection Knowledge Distillation)**: Leverages the VLM visual encoder's informative region awareness to filter high-quality proposals, applying adaptive augmentation and distillation.
2. **PAPL (Prototype-Aware Pseudo-Label)**: Generates unsupervised pseudo-labels via prototype learning, eliminating dependence on text embeddings.
3. **SMI (Synthetic Matching Inference)**: Integrates distillation scores, prototype scores, and localization scores to comprehensively evaluate novel-category objects.

### Key Designs

#### 1. **Adaptive Selection Knowledge Distillation (ASKD)**

**Informative region awareness**: Attention maps from each layer of the VLM visual encoder are averaged to obtain a heatmap focused on informative regions. Informative proposals are filtered via the following steps:

- Attention normalization: $\tilde{Attn} = \sigma(Attn \cdot \lambda)$
- Adaptive offset: $M = \tilde{Attn} + \max(1 - \mathbb{E}[\tilde{Attn}], 0)$
- For each proposal $p_i$, compute the regional average response and retain proposals with mean ≥ 1 as the informative proposal set $P_{inf}$

Compared to threshold- or count-based filtering, this approach better preserves semantic relevance in VLM region embeddings.

**Max-Min Edge Jitter Augmentor**: Targeting objects with extreme aspect ratios in aerial imagery (e.g., ships, bridges), two augmentation strategies are designed:
- **Long-edge jitter**: Perturb the long edge $l_\delta = l + \sigma \cdot s \cdot \epsilon$, fixing the maximum dimension.
- **Short-edge jitter**: Proportionally scale the short edge $s_\delta = s + \sigma \cdot l \cdot \epsilon$, fixing the minimum dimension.

The augmented proposal set $P_{aug}$ enables the model to learn both local and global region views, enhancing feature extraction for objects with extreme aspect ratios.

**Distillation loss**:

$$\mathcal{L}_{distill} = \frac{1}{|P_{aug}|} \sum_i \| f_{roi}(p_i') - v(p_i') \|_1$$

#### 2. **Prototype-Aware Pseudo-Label (PAPL)**

**Unsupervised pseudo-label data generation**:
1. Remove proposals containing base categories from $P_{aug}$, retaining regions likely to contain unknown categories.
2. Apply K-means clustering to their image embeddings, producing $k$ cluster centers $\{v_j\}_{j=1}^k$.
3. For each center, select the top-$n$ nearest-neighbor embedding proposals to form a clean pseudo-label dataset.
4. Label range: unknown-1 to unknown-$k$.

**Trainable class prototypes**: Introduce $k$ trainable class prototypes $\{u_c | c \in \mathcal{C}_U\}$ plus a background prototype $u_{bg}$, replacing frozen text embeddings. The training process encourages the detector to distinguish and exploit inter-class differences in visual features.

**Prototype classifier loss**:

$$\mathcal{L}_{proto} = \mathbb{E}_{(f(p), u)} P(f_{roi}(p_i), u)$$

#### 3. **Synthetic Matching Inference (SMI)**

At inference, three scores are fused:
- **Distillation score** $Score_d$: Similarity between detector RoI features and novel-category text embeddings.
- **Prototype score** $Score_p$: Novel-category text embeddings are mapped to cluster centers via nearest-neighbor matching, and the corresponding prototype is used for classification.
    - Key innovation: $\hat{u}_i = \arg\max_{j} \langle t_c^N, v_j \rangle$ (identifies the cluster center closest to the novel-category text embedding).
- **Localization score** $Score_l$: Objectness score from OLN.

Composite score: $Score_s = \sqrt{Score_l \cdot \sqrt{Score_d \cdot Score_p}}$

### Loss & Training

Two-stage training:
- **Stage 1**: ASKD trains the distillation head for 20 epochs, batch size 32, SGD (lr=1e-3).
- **Stage 2**: Select top-500 proposals and 20 cluster centers to construct pseudo-labels; fine-tune for 12 epochs, batch size 64, with backbone and neck frozen.

Base detector: Faster R-CNN + ResNet-50; VLM: RemoteCLIP-ViT-B32.

## Key Experimental Results

### Main Results

| Method | Extra Supervision | DIOR mAP^N | DIOR HM | DOTA mAP^N | DOTA HM |
|--------|-------------------|-----------|---------|-----------|---------|
| ViLD | ✗ | 7.1 | 12.6 | 3.4 | 6.5 |
| DescReg | ✓ | 7.9 | 14.2 | 4.7 | 8.8 |
| CastDet | ✓ | 29.8 | 42.7 | 14.2 | 23.3 |
| **VK-Det (Ours)** | **✗** | **30.1** | **41.0** | **23.3** | **33.9** |

Key finding: Without any additional supervision, VK-Det surpasses CastDet (which uses extra supervision) on novel-category mAP, with particularly substantial gains on DOTA (+9.1 mAP^N).

### Ablation Study

**Component ablation** (DIOR):

| ASKD | PAPL | SMI | mAP^N | HM | Note |
|------|------|-----|-------|-----|------|
| ✓ | - | - | 7.8 | 14.0 | Distillation only |
| ✓ | ✓ | - | 20.4 | 31.4 | +Prototype classifier |
| ✓ | - | ✓ | 20.1 | 31.1 | +Localization score |
| ✓ | ✓ | ✓ | **30.1** | **41.0** | All components |

**ASKD module ablation**:

| Mask | Enhancer | mAP^N | HM |
|------|----------|-------|-----|
| ✗ | ✗ | 20.0 | 30.5 |
| ✗ | ✓ | 23.2 | 34.1 |
| ✓ | ✗ | 24.5 | 35.5 |
| ✓ | ✓ | **30.1** | **41.0** |

**PAPL vs. externally supervised pseudo-labels**:

| Method | mAP^N | HM | Note |
|--------|-------|-----|------|
| Externally supervised pseudo-labels | 28.1 | 39.0 | Text embeddings introduce hallucinations and noise |
| **Ours (PAPL)** | **30.1** | **41.0** | Prototype learning is more robust |

**SMI score ablation**:

| Score_d | Score_p | Score_l | mAP^N |
|---------|---------|---------|-------|
| ✓ | - | - | 7.8 |
| - | ✓ | - | 9.3 |
| ✓ | ✓ | - | 20.4 |
| ✓ | ✓ | ✓ | **30.1** |

### Key Findings

1. Distillation or classification scores alone converge to local optima (7.8/9.3 mAP^N), while combining all three yields a qualitative leap (30.1%).
2. Informative region awareness is more effective than threshold-based filtering, with mask selection contributing +4.5 mAP^N.
3. Unsupervised pseudo-labels outperform text-supervised pseudo-labels (+2.0 mAP^N), as text embeddings introduce hallucinations over image regions.
4. t-SNE visualizations demonstrate that PAPL-generated pseudo-labels contain rich novel-category annotations, enabling the detector to learn discriminative novel-category features.

## Highlights & Insights

1. **Surpassing supervised methods without extra supervision**: This represents an important paradigm shift, demonstrating that the visual knowledge inherent in VLMs is sufficiently rich on its own.
2. **Informative region awareness**: Proposal filtering leverages the intrinsic properties of VLM attention maps without any additional training.
3. **Prototype learning as a substitute for text embeddings**: Effectively mitigates the text hallucination problem, particularly suitable for aerial scenarios where category names may be ambiguous.
4. **Substantial gains on DOTA**: A 9.1% mAP^N improvement indicates greater advantages in dense small-object scenarios.

## Limitations & Future Work

1. The choice of cluster count $k$ is performance-sensitive; excessively large $k$ may scatter features of the same category across different clusters.
2. The two-stage training pipeline adds complexity; end-to-end alternatives could be explored.
3. The base detector uses Faster R-CNN; more advanced architectures could be investigated.
4. Only RemoteCLIP is used as the VLM; stronger VLMs may yield further gains.
5. The applicability of the method to general (non-aerial) open-vocabulary detection is not discussed.

## Related Work & Insights

- ViLD pioneered the region-level knowledge distillation paradigm from VLMs to detectors; this paper improves the selection of distillation targets.
- CastDet employs a semi-supervised framework but relies on additional supervision; this paper demonstrates that such supervision is not necessary.
- The multi-level scoring mechanism of LP-OVOD inspired the SMI design.
- Prototype learning is widely used in few-shot learning; this paper innovatively applies it to pseudo-label generation for OVAD.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of unsupervised pseudo-labels and informative region awareness is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, extensive ablations, and comprehensive visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; modular design facilitates comprehension.
- Value: ⭐⭐⭐⭐⭐ — Significant practical value in the aerial OVAD domain, surpassing SOTA without additional supervision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](../../CVPR2026/object_detection/beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](../../CVPR2026/object_detection/uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](../../CVPR2026/object_detection/parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)

</div>

<!-- RELATED:END -->
