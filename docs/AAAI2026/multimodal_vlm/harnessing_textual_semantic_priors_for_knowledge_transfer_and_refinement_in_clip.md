---
title: >-
  [Paper Note] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning
description: >-
  [AAAI 2026][Multimodal VLM][Continual Learning] This paper proposes the SECA framework, which leverages the stable semantic priors of the CLIP text branch to guide semantically-aware historical knowledge transfer in the backbone (SG-AKT module), and refines visual prototypes using inter-class semantic relationships derived from text embeddings to build a hybrid classifier (SE-VPR module), achieving state-of-the-art performance on ImageNet-R/A and CIFAR-100.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Continual Learning
  - CLIP
  - Textual Semantic Priors
  - Knowledge Distillation
  - Modality Gap
date: 2026-05-08
content_hash: 14d179a8dfeb5a67
---

# Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning

**Conference**: AAAI 2026
**arXiv**: [2508.01579](https://arxiv.org/abs/2508.01579)
**Code**: [https://github.com/HHHLF/SECA_master](https://github.com/HHHLF/SECA_master)
**Area**: Multimodal VLM
**Keywords**: Continual Learning, CLIP, Textual Semantic Priors, Knowledge Distillation, Modality Gap

## TL;DR
This paper proposes the SECA framework, which leverages the stable semantic priors of the CLIP text branch to guide semantically-aware historical knowledge transfer in the backbone (SG-AKT module), and refines visual prototypes using inter-class semantic relationships derived from text embeddings to build a hybrid classifier (SE-VPR module), achieving state-of-the-art performance on ImageNet-R/A and CIFAR-100.

## Background & Motivation

### State of the Field
The central challenge in Continual Learning (CL) is the **stability-plasticity dilemma**—a model must retain previously acquired knowledge (stability) while adapting to new information (plasticity). With the rise of vision-language models such as CLIP, their strong zero-shot capabilities have made them ideal backbones for continual learning, and PEFT-based methods (e.g., prompt tuning, adapters) have achieved notable progress in this direction.

### Limitations of Prior Work
Existing CLIP-based continual learning methods suffer from two core issues:

**Non-selective knowledge transfer during backbone training**: Most methods (e.g., EWC, AFC) enforce consistency with recent task models via regularization or distillation, without discriminating the semantic relevance of the transferred knowledge. When learning the class "cat," knowledge from "dog" is beneficial, whereas knowledge from "vehicle" introduces interference (semantic interference).

**Modality gap in classifiers**: Pure text classifiers (based on the CLIP text encoder) offer strong generalization but limited plasticity; visual classifiers (prototype-based) can bridge the modality gap, but coarse visual prototypes lack rich and precise semantic information.

### Root Cause
The CLIP text branch maintains consistent semantic representations throughout continual learning (since the frozen text encoder does not change across tasks), yet this valuable **stable semantic prior** is underutilized in existing methods—neither for guiding selective knowledge transfer nor for enhancing the semantic structure of visual classifiers.

### Starting Point
The paper exploits the "forgetting-resistant" and "structured" properties of the CLIP text branch as a unified guidance signal:
- At the backbone level: text cues are used to assess the relevance between new images and historical visual knowledge, enabling instance-adaptive selective distillation.
- At the classifier level: inter-class relationships derived from text embeddings are used to refine visual prototypes, bridging the modality gap.

## Method

### Overall Architecture
SECA is built upon CLIP and comprises frozen visual and text encoders, learnable text prompts and visual adapters, and two core modules:
1. **SG-AKT**: Semantics-Guided Adaptive Knowledge Transfer
2. **SE-VPR**: Semantics-Enhanced Visual Prototype Refinement

### Key Designs

#### 1. **Hybrid PEFT Baseline (H-PEFT)**
- **Text side**: A learnable prompt $\mathbf{P}^s \in \mathbb{R}^{M \times d_T}$ is introduced for each task $s$, concatenated with class names and fed into the frozen text encoder.
- **Visual side**: A shared adapter $\mathcal{A}_l$ is inserted into each transformer block of the visual encoder.
- **Classification probability**: Computed via cosine similarity between visual and text features, using a temperature factor $\tau$.
- Baseline training loss: standard cross-entropy $\mathcal{L}_{ce-T}$.

#### 2. **Semantics-Guided Adaptive Knowledge Transfer (SG-AKT)**
- **Mechanism**: A historical adapter pool $\mathcal{P} = \{\mathcal{A}^1, ..., \mathcal{A}^{|\mathcal{P}|}\}$ is maintained, where each adapter encodes the visual knowledge of its corresponding task. For each new image, semantic relevance to each adapter is assessed using the image's textual semantic vector, and relevant knowledge is aggregated in a weighted manner as a distillation signal.

- **Step 1: Extract multi-perspective knowledge**
    - Visual knowledge: Pass the new image through all historical adapters to obtain $\mathbf{V}_x = [\mathbf{V}_x^{(1)}, ..., \mathbf{V}_x^{(|\mathcal{P}|)}]$.
    - Semantic vectors: Combine class labels with all (historical + current) task prompts to obtain $\mathbf{S}_y = [\mathbf{S}_y^{(1)}, ..., \mathbf{S}_y^{(s)}]$.

- **Step 2: Compute relevance scores**
  $$\alpha_x^{(p)} = \frac{1}{s} \sum_{i=1}^s [\phi(\mathbf{S}_y^{(i)})\mathbf{W}_S]^\top [\phi(\mathbf{V}_x^{(p)})\mathbf{W}_V]$$
  - Learnable projectors $\mathbf{W}_S, \mathbf{W}_V$ map text and visual features into a shared semantic space.
  - LayerNorm $\phi(\cdot)$ stabilizes training.

- **Step 3: Adaptive aggregation and distillation**
  $$\mathbf{V}_x^{agg} = \sum_{p=1}^{|\mathcal{P}|} \frac{\exp(\lambda \alpha_x^{(p)})}{\sum_i \exp(\lambda \alpha_x^{(i)})} \mathbf{V}_x^{(p)}$$
  - The aggregated feature serves as the teacher signal, distilled into the current model via KL divergence.
  - **Design Motivation**: Compared to indiscriminate distillation (Vanilla-KD) or uniform aggregation (Avg-KD), semantics-relevance-based adaptive aggregation prioritizes the transfer of useful knowledge while suppressing interference.

- **Adapter pool management**: The pool size is fixed at $|\mathcal{P}|=5$, with momentum-based updates and pruning guided by a utility score $U^p$. Adapters with the highest utility indicate that their knowledge has been sufficiently transferred to the latest model and can be safely removed.

#### 3. **Semantics-Enhanced Visual Prototype Refinement (SE-VPR)**
- **Mechanism**: Inter-class semantic relationships from text embeddings are used to correct coarse CLIP visual prototypes.

- **Step 1: Compute inter-class affinity matrix**
  $$\mathbf{M}_{k,j} = \exp(-\gamma \|\phi(\mathbf{Z}_k)\mathbf{H}_{proj} - \phi(\mathbf{Z}_j)\mathbf{H}_{proj}\|_2^2)$$
  - A learnable projector $\mathbf{H}_{proj}$ maps class text embeddings into a more expressive latent space.
  - $\gamma$ is a scaling factor.

- **Step 2: Refine visual prototypes**
  $$\hat{\mathbf{c}}_{V,k} = \sum_{j \in \mathcal{Y}^{1:s}} \frac{\mathbf{M}_{k,j}}{\sum_i \mathbf{M}_{k,i}} \mathbf{c}_{V,k}$$
  - The original prototype $\mathbf{c}_{V,k}$ is refined via affinity-weighted aggregation.

- **Prototype consistency regularization**: Prevents old-class prototype drift caused by new task training.
  $$\mathcal{L}_{reg} = \frac{1}{|\mathcal{Y}^{1:s-1}|} \sum_{k \in \mathcal{Y}^{1:s-1}} \|\hat{\mathbf{c}}_{V,k}^s - \hat{\mathbf{c}}_{V,k}^{s-1}\|_2^2$$

- **Design Motivation**: A pure text classifier is limited in plasticity due to the modality gap; refined visual prototypes retain the matching advantage of the visual side while inheriting the semantic structure of the text side.

### Loss & Training
- **Total loss**: $\mathcal{L} = \mathcal{L}_{ce-T} + \underbrace{\mathcal{L}_{agg} + \beta \mathcal{L}_{SG-AKT}}_{SG-AKT} + \underbrace{\mathcal{L}_{ce-V} + \mathcal{L}_{reg}}_{SE-VPR}$
- **Inference**: Hybrid classification—refined visual prototype predictions combined with the average of all task text classifier predictions.

## Key Experimental Results

### Main Results (ImageNet-R & ImageNet-A, CLIP ViT-B/16)

| Method | ER/FR | 10S-ImageNetR Last↑ | 10S-ImageNetR Avg↑ | 10S-ImageNetA Last↑ | 10S-ImageNetA Avg↑ |
|------|-------|--------------------|--------------------|--------------------|--------------------|
| ZS-CLIP | ✗ | 74.93 | 81.56 | 47.33 | 58.35 |
| VPT-NSP | ✓ | 82.48 | 87.94 | 61.42 | 71.76 |
| RAPF | ✓ | 79.62 | 86.28 | 55.37 | 67.32 |
| CLAP | ✓ | 79.98 | 85.77 | 58.66 | 69.35 |
| **SECA (Ours, w/o replay)** | **✗** | **83.18** | **88.58** | **65.09** | **74.45** |
| **SECA++ (Ours, w/ replay)** | **✓** | **83.41** | **88.75** | **65.77** | **74.65** |

### Ablation Study

| Module Combination | 10S-ImageNetA Last↑ | 10S-CIFAR100 Last↑ | 10S-ImageNetR Last↑ |
|----------|--------------------|--------------------|---------------------|
| ZS-CLIP (baseline) | 47.33 | 67.19 | 74.93 |
| +H-PEFT | 55.78 | 73.97 | 80.57 |
| +H-PEFT+SG-AKT | 57.91 (+2.13) | 75.93 (+1.96) | 81.36 (+0.79) |
| +H-PEFT+SE-VPR | 62.62 (+6.84) | 77.13 (+3.16) | 81.30 (+0.73) |
| +H-PEFT+SG-AKT+SE-VPR (**SECA**) | **65.09** (+9.31) | **79.79** (+5.82) | **83.18** (+2.61) |

### Distillation Strategy Comparison (10S-ImageNetA, with SE-VPR)

| Distillation Strategy | Last↑ | Avg↑ |
|----------|------|------|
| Sequential (no distillation) | 62.62 | 73.15 |
| CLIP-KD (global teacher) | 62.39 | 73.01 |
| Vanilla (recent-task distillation) | 63.51 | 73.89 |
| Avg-KD (uniform aggregation) | 64.32 | 74.08 |
| **SG-AKT (Ours)** | **65.09** | **74.45** |

### Classifier Design Comparison (10S-ImageNetA)

| Classifier | Last↑ | Avg↑ |
|--------|------|------|
| Only Text | 57.91 | 68.56 |
| Centroid (CLIP) | 58.55 | 68.30 |
| Centroid (Adapted) | 63.16 | 72.89 |
| Linear | 51.07 | 62.76 |
| **SE-VPR (Ours)** | **65.09** | **74.45** |

### Key Findings
1. **SECA without replay already surpasses replay-based SOTA**: SECA achieves 3.67% higher Last acc than VPT-NSP on 10S-ImageNetA without any replay.
2. **SE-VPR contributes the most**: On ImageNetA, SE-VPR alone yields +6.84%, far exceeding SG-AKT's +2.13%.
3. **Two modules are complementary**: When used jointly, SECA achieves a total gain of +9.31%, substantially exceeding the sum of individual contributions.
4. **Semantics-guided distillation outperforms all variants**: SG-AKT surpasses Vanilla-KD by 1.58% and Avg-KD by 0.77%.
5. **Adapter pool size saturates at 5**: Performance with $|\mathcal{P}| \geq 5$ approaches that of using all historical adapters ("ALL").
6. **Linear classifier performs worst**: This indicates that simple linear layers cannot effectively leverage adapted features for cross-task classification.

## Highlights & Insights
- **Systematic exploitation of textual priors**: This is the first work to simultaneously leverage CLIP textual semantics from two complementary perspectives—knowledge transfer and prototype refinement.
- **Instance-adaptive distillation**: Different images receive different knowledge aggregation weights based on their semantic relevance, offering finer granularity than one-size-fits-all distillation.
- **Utility-score-based pool management**: Momentum tracking monitors the degree to which each adapter's knowledge has been utilized; adapters with the highest utility indicate sufficient knowledge transfer and can be safely pruned—an elegant fixed-capacity strategy.
- **Modality gap bridging**: Rather than naively fusing text and visual features, the relational structure of text is used to "sculpt" visual prototypes.
- **Surpassing SOTA without replay**: This demonstrates that semantics-guided selective transfer can substitute for conventional sample/feature replay strategies.

## Limitations & Future Work
- Validation is limited to classification tasks; extension to more complex continual learning scenarios such as detection and segmentation has not been explored.
- The fixed adapter pool size of 5 may be insufficient for settings with a very large number of tasks (e.g., 100+).
- The approach relies on the quality of CLIP's visual-text alignment; effectiveness in domains where CLIP alignment is weaker (e.g., medical imaging) remains unknown.
- At inference, all historical task prompts and the adapter pool are required, and storage and computational overhead grow with the number of tasks.
- SE-VPR extracts prototypes using the original CLIP encoder (rather than the adapted one), which may discard task-specific useful information.
- $\beta$ is a task-dependent hyperparameter whose growth schedule requires dataset-specific tuning.

## Related Work & Insights
- L2P (Wang et al., CVPR 2022), CODA (Smith et al., CVPR 2023): Prompt-based continual learning methods that do not leverage CLIP textual semantics.
- RAPF (ECCV 2024): Lightweight projector with pseudo-feature replay; SECA already surpasses its replay version without any replay.
- VPT-NSP (NeurIPS 2024): Demonstrates the strong continual learning potential of CLIP; SECA further mines this potential from the textual semantic perspective.
- PROOF (T-PAMI 2025): Cross-attention modules combined with weight averaging, but the classifier remains limited to pure text.
- Insight: In the downstream adaptation of multimodal pretrained models, different modalities each have distinct advantages—the stable semantics of text can guide selective adaptation on the visual side.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The dual-module design leveraging textual priors for both knowledge transfer and prototype refinement is highly insightful.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple benchmarks, extensive ablations, distillation strategy comparisons, and hyperparameter analysis—extremely comprehensive.)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are complete, though the dense notation requires frequent cross-referencing.)
- Value: ⭐⭐⭐⭐⭐ (Provides a new paradigm for CLIP-based continual learning; surpassing SOTA without replay carries significant practical importance.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[CVPR 2026\] KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering](../../CVPR2026/multimodal_vlm/kec_hierarchical_textual_knowledge_clustering.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[NeurIPS 2025\] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](../../NeurIPS2025/multimodal_vlm/hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)

</div>

<!-- RELATED:END -->
