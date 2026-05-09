---
title: >-
  [Paper Note] Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning
description: >-
  [CVPR 2026][Reinforcement Learning][Image Captioning] This paper proposes Cross-modal Identity Mapping (CIM), which quantifies information loss in image captioning by analyzing the representational consistency (GRC) of images retrieved via captions and their relevance to the source image (QIR). These metrics serve as RL reward signals to train LVLMs to generate fine-grained and accurate captions without requiring additional annotations.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Image Captioning
  - Cross-modal Information Loss
  - Retrieval Reward
  - GRPO
date: 2026-05-08
content_hash: 43f1bbcc62dfdb3d
---

# Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2603.01696](https://arxiv.org/abs/2603.01696)
**Code**: To be released (after paper acceptance)
**Area**: Reinforcement Learning
**Keywords**: Image Captioning, Cross-modal Information Loss, Retrieval Reward, Reinforcement Learning, GRPO

## TL;DR
This paper proposes Cross-modal Identity Mapping (CIM), which quantifies information loss in image captioning by analyzing the representational consistency (GRC) of images retrieved via captions and their relevance to the source image (QIR). These metrics serve as RL reward signals to train LVLMs to generate fine-grained and accurate captions without requiring additional annotations.

## Background & Motivation
LVLMs frequently omit or misrepresent key visual content in image captioning tasks. The authors validate this on the Oxford-IIIT Pet dataset through fine-grained classification experiments: while multiple LVLMs (e.g., Qwen3-VL-8B, InternVL3-8B) achieve near-100% accuracy on species classification, breed classification accuracy falls to only 15%–40%, indicating that models tend to describe coarse-grained concepts while neglecting fine-grained details—a manifestation of severe cross-modal information loss.

Existing improvement methods fall into two categories: (1) constructing fine-grained annotated data for SFT, which incurs high annotation costs; and (2) using VLM-based metrics as RL rewards, which are prone to reward hacking due to the limited compositional reasoning capacity of VLMs themselves. The root cause is: how to accurately quantify information loss in image captioning without relying on additional annotations?

The authors propose a key insight: **the more fine-grained a caption, the more consistent the retrieved images; the more accurate a caption, the more similar the retrieved images are to the source image**. Based on this, the information loss of a caption is inferred by analyzing the distribution of retrieval results.

## Method

### Overall Architecture
CIM is an annotation-free RL framework whose core pipeline proceeds as follows: (1) the LVLM generates multiple captions for an input image; (2) each caption is used as a text query to retrieve the top-$K$ relevant image–text pairs; (3) two metrics, GRC and QIR, are computed from the retrieval results as rewards; (4) the LVLM is optimized via GRPO.

### Key Designs

1. **Gallery Representation Consistency (GRC)**:

    - **Function**: Evaluates the internal consistency of the image set retrieved by a caption, reflecting the richness of detail in the caption.
    - **Mechanism**: $GRC(c) = \|\frac{1}{K}\sum_{r=1}^{K}\tilde{v}(x_{i_r})\|_2$, where $\tilde{v}(x_j)$ is the $\ell_2$-normalized embedding of an image extracted by a visual representation model. GRC is essentially the mean resultant length, measuring the concentration of embedding vectors on the hypersphere.
    - **Design Motivation**: The more detailed and specific a caption, the more concentrated the retrieved images are in the visual representation space (higher GRC); the more vague and coarse-grained a caption, the more dispersed the retrieval results.

2. **Query-gallery Image Relevance (QIR)**:

    - **Function**: Measures the relevance between the source image and the retrieved images, reflecting the accuracy of the caption.
    - **Mechanism**: $QIR(v, c) = \sum_{r=1}^{K}\lambda(r) \cdot Cos(\tilde{v}(v), \tilde{v}(x_{i_r}))$, where $\lambda(r) = 1/2^{r-1}$ is an exponentially decaying weight that assigns greater importance to higher-ranked retrieval results.
    - **Design Motivation**: If a caption accurately describes the source image content, the retrieved images should be semantically highly similar to the source; if the caption contains erroneous information, the retrieval results will deviate from the source image.

3. **Cross-modal Identity Mapping Reward**:

    - **Function**: Combines GRC and QIR into an RL reward to optimize the LVLM via GRPO.
    - **Mechanism**: $\Upsilon(v, c) = GRC(c) + \beta \cdot QIR(v, c)$, where $\beta$ balances accuracy and detail richness. $G$ captions are sampled, and the within-group normalized advantage is computed as $A_z = \frac{\Upsilon_z - mean(\{\Upsilon\})}{std(\{\Upsilon\})}$.
    - **Design Motivation**: By reformulating caption quality assessment as an image-to-image similarity problem, the approach circumvents the difficulty of directly measuring cross-modal information loss without requiring additional annotations.

### Loss & Training
GRPO training is conducted using the VERL framework. Training data consists of RefinedCaps (6.5K images), with 5 captions generated per image. Text retrieval uses SBERT (MPNet-base), image encoding uses OpenCLIP ViT-H/14, and the retrieval gallery is augmented with RefinedCaps + DenseFusion-1M. Learning rate is $1 \times 10^{-6}$; training runs for 2 epochs.

## Key Experimental Results

### Main Results

| Model | Dataset | CAPTURE | Relation QA | Gain |
|-------|---------|---------|-------------|------|
| Qwen2.5-VL-7B + CIM | COCO-LN500 | 48.93 | 44.15 | Relation Recall +20.2, QA +20.4 |
| Qwen2-VL-7B + CIM | COCO-LN500 | 48.64 | 38.71 | Relation Recall +10.4, QA +18.2 |
| LLaVA1.5-7B + CIM | COCO-LN500 | 48.62 | 24.98 | Relation Recall +12.6, QA +10.6 |
| InternVL3-8B + CIM | COCO-LN500 | 48.90 | 38.67 | Relation Recall +10.0, QA +12.2 |

### Ablation Study

| Configuration | CAPTURE | Notes |
|---------------|---------|-------|
| GRC only | Improvement but limited | Encourages detail richness only |
| QIR only | Improvement but limited | Constrains accuracy only |
| GRC + QIR | Best | Complementary combination |
| Varying gallery size | Larger is better | Larger gallery provides more reliable retrieval signal |

### Key Findings
- CIM achieves a gain of up to 20.2% in Relation Recall and 20.4% in Relation QA for Qwen2.5-VL-7B on COCO-LN500—a highly significant improvement.
- CIM is effective across multiple architectures (LLaVA, Qwen-VL, InternVL) and versions, demonstrating the generality of the approach.
- Pearson correlation analysis confirms a positive correlation between GRC/QIR and actual caption quality (logit of breed classification accuracy).

## Highlights & Insights
- The cross-modal information loss quantification problem is elegantly reformulated as an image-to-image similarity problem following image retrieval, requiring no additional annotations.
- The design intuition behind GRC and QIR is clear: one governs *detail*, the other governs *accuracy*, aligning with human intuition about caption quality.
- The substantial gains on the Relation dimension suggest that information loss in relational reasoning is the most severe deficiency in existing LVLMs and is also the most amenable to improvement via RL.

## Limitations & Future Work
- The composition and scale of the retrieval gallery directly affect reward signal quality; the approach is sensitive to the choice of retrieval model.
- Training uses only 6.5K images, which, while efficient, may impose an upper bound on performance.
- The computation of GRC and QIR requires an additional retrieval step, increasing training overhead.
- The design of the exponential decay weight $\lambda(r)$ lacks theoretical justification.

## Related Work & Insights
- Similar to CapRL in using RL to optimize captioning, but the reward signal differs: CapRL employs VQA, whereas CIM uses retrieval consistency.
- The self-retrieval reward concept has been explored in prior work; CIM further decomposes it into two dimensions—fine-grainedness (GRC) and accuracy (QIR).
- The idea of reformulating generation quality evaluation as retrieval quality evaluation holds promise for generalization to other cross-modal generation tasks.
- Compared to cycle-consistency methods (reconstructing images from captions), CIM avoids the high overhead of training an image generator.
- SC-Captioner's keyword-checking approach is overly coarse-grained; CIM provides a continuous and comprehensive quality signal via retrieval distributions.

## Supplementary Details
- The mean resultant length underlying GRC measures the concentration of a set of vectors on the unit hypersphere; higher values indicate greater consistency.
- The exponential decay $\lambda(r) = 1/2^{r-1}$ in QIR assigns the greatest contribution to the top-1 retrieval result, consistent with the decreasing reliability of lower-ranked results.
- Applying CIM on top of SFT models yields further improvements, demonstrating that the method is complementary to SFT.
- Experiments are validated across 6 models ranging from LLaVA-1.5-7B to InternVL3-8B.
- Expanding the retrieval gallery (by incorporating DenseFusion-1M) further improves performance, indicating a positive correlation between the reliability of the retrieval signal and gallery scale.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The insight of using retrieval consistency as a proxy for caption quality is novel; the GRC/QIR design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model validation is thorough, with Pearson correlation verification; ablations could be more detailed.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear, validation experiments are well-designed, and figures are of high quality.
- **Value**: ⭐⭐⭐⭐ The annotation-free RL optimization approach is highly practical, with significant gains in relational reasoning.

## Key Terminology
- **Mean Resultant Length**: A measure of vector concentration on the hypersphere.
- **Identity Mapping**: Treats the image-to-caption transformation as an identity-preserving mapping that minimizes information loss.
- **SBERT/OpenCLIP**: Used for text retrieval and image embedding extraction, respectively.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](../../ICCV2025/reinforcement_learning/r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](../../ICLR2026/reinforcement_learning/cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](../../ICLR2026/reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)

<!-- RELATED:END -->
