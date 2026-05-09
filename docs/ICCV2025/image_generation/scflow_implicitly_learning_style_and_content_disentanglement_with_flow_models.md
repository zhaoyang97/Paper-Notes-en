---
title: >-
  [Paper Note] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models
description: >-
  [Image Generation] SCFlow is proposed to learn an invertible merging mapping between style and content via Flow Matching, leveraging the invertibility of the mapping to allow disentanglement to emerge naturally as an implicit property of the merging process, without requiring explicit disentanglement supervision.
tags:
  - Image Generation
date: 2026-05-08
content_hash: fa058fd95a5aac76
---

# SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2508.03402](https://arxiv.org/abs/2508.03402)
- **Code**: [GitHub](https://github.com/CompVis/SCFlow)
- **Area**: Diffusion Models · Style-Content Disentanglement
- **Keywords**: Flow Matching, style-content disentanglement, invertible mapping, CLIP embedding space, dataset construction

## TL;DR
SCFlow is proposed to learn an invertible merging mapping between style and content via Flow Matching, leveraging the invertibility of the mapping to allow disentanglement to emerge naturally as an implicit property of the merging process, without requiring explicit disentanglement supervision.

## Background & Motivation

Explicit disentanglement of style and content faces fundamental difficulties:
1. The two concepts are semantically overlapping, with subjective and ambiguous boundaries.
2. Clean ground-truth annotations for style/content are unavailable.
3. Existing methods (generative or discriminative) require predefined separation criteria.

**Core Insight**: Rather than directly disentangling (which is difficult and ambiguous), it is preferable to learn **merging** (which is well-defined and grounded). If the merging process is invertible, disentanglement emerges naturally.

**Why Flow Matching**:
- Diffusion models and Normalizing Flows require one endpoint to be a Gaussian distribution, making them unsuitable for settings where both endpoints are real data distributions.
- Flow Matching can learn bidirectional ODE mappings between arbitrary distributions, making it perfectly suited for mapping from a disentangled distribution $p_0$ to a merged distribution $p_1$.

## Method

### Overall Architecture

The method operates in the CLIP embedding space (to avoid low-level biases in pixel space), mapping disentangled style/content pairs to merged representations.

### Endpoint Distribution Definition

**Disentangled endpoint $p_0$** (concatenation of two embeddings):
$$x_0 = [z_{c_i, s_*}, z_{c_*, s_j}]$$

**Merged endpoint $p_1$** (repeated identical embedding):
$$x_1 = [z_{c_i, s_j}, z_{c_i, s_j}]$$

$*$ denotes an arbitrary instance. The subtlety of this asymmetric construction is that $x_0$ contains redundant information ($s_*$ and $c_*$), forcing the model to: 1) discard irrelevant information; and 2) extract the useful $s_j$ and $c_i$ from entangled representations.

### Flow Matching Training

Forward path:
$$x_t = (1-t) x_0 + t \cdot x_1$$

Velocity field training objective:
$$\mathcal{L}(\theta) = \int_0^T \mathbb{E}[\|v_\theta(x_t, t) - \dot{\alpha}_t x_0 - \dot{\sigma}_t x_1\|^2] \mathrm{d}t$$

### Bidirectional Inference

**Forward (merging)**: $z_{c_i, s_j} = \text{mean}(\text{ODESolve}([z_{c_i,s_*}, z_{c_*,s_j}])_{[0,1]})$

**Inverse (disentangling)**: $[z_{c_i, \bar{s}}, z_{\bar{c}, s_j}] = \text{ODESolve}(\text{repeat}[z_{c_i, s_j}])_{[1,0]}$

Only the forward direction is trained; the inverse is obtained via reverse integration through the ODE solver.

### Dataset Construction

510,000 samples = 51 styles × 10,000 content instances, with full combinatorial coverage:
- Content images crawled from Pexels
- Style variants generated via ControlNet
- Each style covers all content instances, and each content instance covers all styles

## Experiments

### Quantitative: Embedding Space Quality (NMI + FDR)

| Method | Content NMI↑ | Style NMI↑ | Content FDR↑ | Style FDR↑ |
|--------|-------------|-----------|-------------|-----------|
| CLIP | 0.537 | 0.402 | 0.431 | 0.296 |
| DEADiff | 0.506 | 0.414 | 0.557 | 0.338 |
| CSD | 0.335 | 0.724 | 0.308 | 0.633 |
| **SCFlow** | **0.836** | **0.870** | **2.169** | **3.518** |

SCFlow leads by a large margin on both content and style; the style FDR exceeds CLIP by an order of magnitude.

### Zero-Shot Generalization

| Task | Method | Key Metric |
|------|--------|-----------|
| ImageNet-1k kNN Classification | CLIP | Acc@1=67.10% |
| | **SCFlow** | Acc@1=66.25% |
| WikiArt Style Retrieval | CLIP | Recall@1=59.40% |
| | CSD | Recall@1=64.56% |
| | **SCFlow** | **Recall@1=65.34%** |

Content classification performance is close to CLIP (−0.85%), while style retrieval surpasses all baselines. This demonstrates that the disentangled representations generalize to content and styles unseen during training.

### Key Findings

1. Content representations produced by inverse inference contain no style information, and style representations contain no content-specific information — the disentanglement is highly pure.
2. Linear interpolation yields continuous semantic transitions (whereas CLIP space exhibits abrupt jumps), and t-SNE visualizations show tighter class clustering.
3. NFE=1 already yields strong results by default, indicating that the learned mapping path is nearly linear.

## Highlights & Insights

1. **Philosophical innovation**: The counterintuitive paradigm of "merging rather than disentangling" achieves implicit disentanglement through invertibility.
2. **Elegant data engineering**: The fully combinatorial dataset design enables the model to observe style and content varying independently.
3. **Asymmetric triplets**: $x_0$ deliberately contains redundant information, compelling the model to learn to filter and extract relevant factors.
4. **Zero-shot generalization**: Training exclusively on synthetic data generalizes to ImageNet and WikiArt.

## Limitations & Future Work

- Relies on the CLIP encoder, so representational capacity is bounded by CLIP's pretraining knowledge.
- Visualization depends on an unCLIP decoder, and decoding quality affects result presentation.
- The diversity of 51 styles is limited; scaling to more styles may require additional effort.
- ODE solving in Flow Matching is subject to error accumulation at high NFE.

## Related Work & Insights

- **Flow Matching**: Conditional Flow Matching, Rectified Flow
- **Style Transfer**: Neural Style Transfer, DEADiff, CSGO
- **Contrastive Learning**: CLIP, CSD, self-supervised methods

## Rating
- Novelty: ★★★★★ — The idea of "implicit disentanglement via invertible merging" is highly original.
- Technical Depth: ★★★★☆ — Mathematical modeling is elegant and experimental validation is thorough.
- Practicality: ★★★☆☆ — The disentangled representations are valuable, but downstream application scenarios remain to be further explored.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[ICLR 2026\] Intention-Conditioned Flow Occupancy Models](../../ICLR2026/image_generation/intention-conditioned_flow_occupancy_models.md)

<!-- RELATED:END -->
