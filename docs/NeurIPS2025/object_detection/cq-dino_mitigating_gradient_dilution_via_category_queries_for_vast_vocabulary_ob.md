---
title: >-
  [Paper Note] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection
description: >-
  [NeurIPS 2025][Object Detection][Large-vocabulary detection] To address positive gradient dilution and hard-negative gradient dilution in large-vocabulary (>10K category) object detection, this paper proposes CQ-DINO: replacing the classification head with learnable category queries and using image-guided Top-K category selection to reduce the negative space by 100×. CQ-DINO surpasses the previous SOTA by 2.1% AP on V3Det (13,204 categories) while remaining competitive on COC…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "Large-vocabulary detection"
  - "category queries"
  - "gradient dilution"
  - "V3Det"
  - "DINO"
date: 2026-05-08
content_hash: 66a7acea59ed019b
---

# CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection

**Conference**: NeurIPS 2025
**arXiv**: [2503.18430](https://arxiv.org/abs/2503.18430)  
**Code**: [https://github.com/FireRedTeam/CQ-DINO](https://github.com/FireRedTeam/CQ-DINO)  
**Area**: Object Detection
**Keywords**: Large-vocabulary detection, category queries, gradient dilution, V3Det, DINO

## TL;DR

To address positive gradient dilution and hard-negative gradient dilution in large-vocabulary (>10K category) object detection, this paper proposes CQ-DINO: replacing the classification head with learnable category queries and using image-guided Top-K category selection to reduce the negative space by 100×. CQ-DINO surpasses the previous SOTA by 2.1% AP on V3Det (13,204 categories) while remaining competitive on COCO.

## Background & Motivation

**Background**: Object detection has scaled from COCO (80 categories) to V3Det (13,204 categories), representing a two-order-of-magnitude increase in category scale. Existing approaches fall into four main families: classification-head detectors, text-prompt contrastive detectors (e.g., Grounding DINO), language-model-based generative detectors, and open-vocabulary methods.

**Limitations of Prior Work**:
   - **Classification-head methods**: Sigmoid + Focal Loss suffers from severe gradient dilution at large vocabulary sizes—positive-class gradients are overwhelmed by the vast number of negative-class gradients (positive-to-negative gradient ratio $\rho \propto 1/(C \cdot \epsilon)$, so $\rho \to 0$ when $C > 10^4$).
   - **Text-prompt methods**: Input token limits of ~128 make processing 13,204 categories require 331 inference passes, rendering them impractical.
   - **Language-model methods**: Label granularity cannot be controlled (e.g., "Persian cat" may be generated as "cat").

**Key Challenge**: Two gradient dilution problems arise at large vocabulary scales—(1) *positive gradient dilution*: sparse positive-class signals are suppressed by the overwhelming number of negative-class signals; (2) *hard-negative gradient dilution*: informative hard-negative gradients are diluted by the large volume of easy negatives.

**Goal**: Simultaneously resolve positive gradient dilution and hard-negative mining via dynamic sparse category selection.

**Key Insight**: Rather than modifying the loss function (Focal Loss provides limited benefit), fundamentally reduce the negative space—retaining only the Top-K most relevant categories per image ($K=100 \ll 13204$)—thereby naturally achieving gradient rebalancing and implicit hard-negative mining.

**Core Idea**: Replace the fixed classification head with learnable category queries combined with image-guided selection, reframing category prediction as contrastive matching between object queries and category queries.

## Method

### Overall Architecture

Built upon the Grounding DINO architecture. Category encoding: learnable category queries $Q_{cat} \in \mathbb{R}^{C \times D}$ initialized with OpenCLIP, capable of modeling inter-category relationships via self-attention or a hierarchical tree. Image-guided selection: cross-attention computes category–image similarity scores, selecting the Top-$C'$ most relevant categories. Detection output: selected category queries interact with image features through a feature enhancer and cross-modal decoder to produce detection results.

### Key Designs

1. **Image-Guided Category Query Selection**

    - **Function**: Dynamically selects the $C'=100$ most relevant categories per image ($C' \ll C = 13204$).
    - **Mechanism**: Category queries serve as Query; image features serve as Key/Value; similarity is computed via cross-attention followed by Top-K selection. Supervised with Asymmetric Loss.
    - **Design Motivation**: Improves the positive-to-negative gradient ratio by a factor of $C/C' \approx 132$ (derived as $\rho'/\rho \approx C/C'$). Since selected categories are the most image-similar, they inherently include the most confusing hard negatives.
    - **Triple benefit**: Gradient rebalancing + adaptive hard-negative mining + reduced computation.

2. **Explicit Hierarchical Tree Construction (V3Det)**

    - **Function**: Leverages the dataset's hierarchical category structure to build a tree-structured representation of category queries.
    - **Mechanism**: Bottom-up aggregation from leaf nodes; parent query $= (1-\alpha_v) \cdot Q_v + \alpha_v \cdot \text{mean}(Q_{\text{children}})$, where $\alpha_v$ adapts based on the number of child nodes (more children → greater reliance on collective knowledge).
    - **Design Motivation**: Large-vocabulary datasets inherently exhibit hierarchical structure (V3Det provides a category tree); leveraging this prior enables semantically similar categories to share feature representations.

3. **Hierarchical Masking Strategy**

    - **Function**: If a child category appears in the ground truth, all its ancestor categories are excluded from the classification loss.
    - **Design Motivation**: Prevents contradictory supervision signals where "Persian cat" is a positive class while "cat" is incorrectly treated as a negative.

4. **Implicit Relation Learning (datasets without hierarchy, e.g., COCO)**

    - **Function**: For datasets lacking explicit hierarchical structure, self-attention among category queries enables learning of implicit inter-category relationships.
    - **Design Motivation**: Not all datasets provide explicit hierarchies, yet semantic relationships between categories still exist.

### Loss & Training

- Category selection supervision: Asymmetric Loss
- Detection classification: contrastive alignment between object queries and category queries (inherited from Grounding DINO)
- Regression: L1 + GIoU
- Training: 24 epochs, learning rate $10^{-4}$

## Key Experimental Results

### Main Results on V3Det

| Method | Backbone | AP | AP50 | AP75 |
|--------|----------|----|------|------|
| DINO | Swin-B | 42.0 | 46.8 | 43.9 |
| Prova | Swin-B | 44.5 | 49.9 | 46.6 |
| **CQ-DINO** | Swin-B | **46.3** | **51.5** | **48.4** |
| DINO | Swin-L | 48.5 | 54.3 | 50.7 |
| Prova | Swin-L | 50.9 | 57.2 | 53.2 |
| **CQ-DINO** | Swin-L | **53.0** | **58.4** | **55.4** |

### COCO Comparison (verifying no degradation)

| Method | Backbone | AP |
|--------|----------|----|
| DINO | Swin-L | 58.0 |
| Rank-DETR | Swin-L | 58.2 |
| **CQ-DINO** | Swin-L | **58.5** |

### Ablation Study

| Component | V3Det AP | Notes |
|-----------|----------|-------|
| DINO baseline | 42.0 | Baseline |
| + Category queries (no selection) | ~43 | Contrastive matching alone helps |
| + Image-guided selection | ~45 | Primary source of improvement |
| + Hierarchical tree construction | 46.3 | Further gain |

### Key Findings

- **Gradient dilution is the core bottleneck in large-vocabulary detection**: Experiments show that DINO's positive-to-negative gradient ratio on V3Det is extremely low (~0.15); CQ-DINO raises it to ~0.8.
- **Top-K selection achieves 100× gradient rebalancing**: At $C'=100$, $C/C'=132$, consistent with theoretical derivation.
- **No degradation on COCO**: 58.5 AP slightly exceeds DINO (58.0), demonstrating no additional bias is introduced in small-vocabulary settings.
- **Zero-shot methods fail on V3Det**: GenerateU achieves only 0.4 AP and ChatRex only 1.3 AP—fine-grained large-vocabulary detection requires dedicated design.

## Highlights & Insights

- **The gradient dilution analysis is concise yet powerful**: The simple gradient ratio formula $\rho \propto 1/(C \cdot \epsilon)$ clearly explains why large-vocabulary detection is difficult and why Focal Loss is insufficient (it reduces $\epsilon$ but cannot eliminate the effect of $C$). This analytical framework transfers to any classification problem with extreme class imbalance.
- **Image-guided category selection exemplifies a "subtraction" design philosophy**: Rather than training the model to discriminate among all categories, it first narrows the candidate space and then performs fine-grained discrimination—analogous to cascade detectors, but more elegant.
- **Flexibility of learnable category queries**: The same framework accommodates both explicit hierarchical relationships (tree structure) and implicit inter-category associations (self-attention), unifying the two settings.

## Limitations & Future Work

- The Top-K value $C'=100$ is fixed; a dynamically adaptive $K$ may yield better performance.
- Image-guided selection still requires computing cross-attention over all 13,204 categories, leaving room for further efficiency improvement despite being faster than the classification head.
- The adaptive weight $\alpha_v$ in hierarchical tree construction relies on a hand-designed prior (child node count); end-to-end learning may be preferable.
- Validation on larger-scale datasets (e.g., ImageNet-21K-level detection benchmarks) remains unexplored.
- The combination with open-vocabulary detection methods is uninvestigated—can category queries complement text encoders?

## Related Work & Insights

- **vs. Grounding DINO**: Grounding DINO encodes categories with a text encoder, but the 128-token limit prevents handling large vocabularies; CQ-DINO circumvents this bottleneck via learnable queries and Top-K selection.
- **vs. Prova**: Prova employs multimodal image-text prototypes but still relies on a classification-head architecture; CQ-DINO fundamentally replaces the classification head with contrastive matching.
- **vs. Focal Loss**: Focal Loss partially alleviates the problem by reducing $\epsilon$ (activations of easy samples) but cannot eliminate the $C$ term; CQ-DINO addresses the problem by directly reducing $C$ itself.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of gradient dilution analysis and image-guided category selection is highly innovative; however, the concept of category queries has precedent in Query2Label.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both V3Det and COCO with thorough ablation analysis; additional large-vocabulary datasets would strengthen the evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical analysis is clear and well-structured; figures and tables are excellently designed.
- Value: ⭐⭐⭐⭐⭐ Provides a principled solution to large-scale category detection with high practical utility.

## Limitations & Future Work

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[ICCV 2025\] Uncertainty-Aware Gradient Stabilization for Small Object Detection](../../ICCV2025/object_detection/uncertainty-aware_gradient_stabilization_for_small_object_detection.md)
- [\[CVPR 2026\] SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names](../../CVPR2026/object_detection/sra-det_learning_omni-grained_open-vocabulary_detection_beyond_category_names.md)

</div>

<!-- RELATED:END -->
