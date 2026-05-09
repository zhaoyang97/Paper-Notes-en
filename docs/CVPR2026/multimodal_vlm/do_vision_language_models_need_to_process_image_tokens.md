---
title: >-
  [Paper Note] Do Vision Language Models Need to Process Image Tokens?
description: >-
  [CVPR 2026][Multimodal VLM][vision-language models] This paper systematically demonstrates that image token representations in VLMs stabilize in shallow layers and become functionally interchangeable across deeper layers, while text token representations undergo continuous dynamic reconstruction — the necessity of deep image processing is highly dependent on the output task type.
tags:
  - CVPR 2026
  - Multimodal VLM
  - vision-language models
  - image tokens
  - representation analysis
  - computational efficiency
  - modality redundancy
date: 2026-05-08
content_hash: 654d3e4317d584b5
---

# Do Vision Language Models Need to Process Image Tokens?

**Conference**: CVPR 2026
**arXiv**: [2604.09425](https://arxiv.org/abs/2604.09425)
**Code**: Available
**Area**: Multimodal VLM
**Keywords**: vision-language models, image tokens, representation analysis, computational efficiency, modality redundancy

## TL;DR

This paper systematically demonstrates that image token representations in VLMs stabilize in shallow layers and become functionally interchangeable across deeper layers, while text token representations undergo continuous dynamic reconstruction — the necessity of deep image processing is highly dependent on the output task type.

## Background & Motivation

**Background**: VLMs achieve multimodal reasoning by combining visual encoders with LLMs, yet propagating dense image tokens through deep Transformer layers incurs substantial computational overhead. Recent work suggests that visual signals may be inefficiently utilized in multimodal tasks.

**Limitations of Prior Work**: Whether visual tokens continue to provide meaningful representational transformations in the deeper layers of VLMs remains unclear. Prior work has largely assumed visual redundancy and designed pruning mechanisms accordingly, without a systematic understanding of representational dynamics.

**Key Challenge**: VLMs apply uniform depth of processing to both image and text tokens, yet the representational evolution patterns of the two modalities may be fundamentally different.

**Goal**: To systematically analyze the evolution, interchangeability, task dependence, and recoverability of image tokens in VLMs from a representational perspective.

**Key Insight**: Three metrics — matrix entropy, intrinsic dimensionality, and trajectory curvature — are used to trace representational structure evolution across models ranging from 3B to 72B parameters.

**Core Idea**: Image representations rapidly converge to a bounded-complexity region in shallow layers, with deeper processing primarily preserving rather than reconstructing visual information.

## Method

### Overall Architecture

The study is organized around five research questions: (RQ1) How do representations evolve? (RQ2) Does stabilization imply functional interchangeability? (RQ3) Is the necessity of image tokens task-dependent? (RQ4) Can truncation be recovered via fine-tuning? (RQ5) Can reasoning chains compensate for reduced visual processing?

### Key Designs

1. **Three-Metric Representation Analysis Framework**:

    - Function: Quantifies layer-wise representational dynamics for both image and text tokens.
    - Mechanism: Matrix entropy measures spectral concentration (low = compressed, high = dispersed); intrinsic dimensionality estimates the effective degrees of freedom of the local manifold; trajectory curvature captures the degree of directional reconstruction across layers: $\bar{C}_l = \frac{1}{N}\sum_i \arccos(\frac{\langle v_l^{(i)}, v_{l-1}^{(i)}\rangle}{\|v_l^{(i)}\|\|v_{l-1}^{(i)}\|})$
    - Design Motivation: Any single metric may be biased; consistent findings across three complementary metrics provide more reliable conclusions.

2. **Layer Substitution Protocol**:

    - Function: Tests the functional interchangeability of image tokens at different depths.
    - Mechanism: A hybrid state $Z_{hybrid} = (Z_{l_a}^{img}, Z_{l_b}^{txt})$ is constructed by combining shallow-layer image tokens with deep-layer text tokens and propagating the result, with output semantic similarity evaluated. Image token substitution preserves ~1.0 similarity, while text token substitution exhibits significant degradation as the layer gap increases.
    - Design Motivation: If structural stabilization implies functional interchangeability, shallow-layer image tokens should be substitutable for their deep-layer counterparts without affecting output semantics.

3. **Visual Depth Truncation Analysis**:

    - Function: Quantifies the degree to which different tasks depend on continued image token processing.
    - Mechanism: All image token activations are removed after a cut layer $l_c$. Single-token prediction (MCQ) is relatively robust to truncation, whereas multi-token generation (captioning) is highly sensitive to early truncation, with BLEU/ROUGE scores monotonically increasing with visual depth.
    - Design Motivation: Interchangeability does not imply dispensability; the distinct visual depth requirements of different output structures must be characterized separately.

### Loss & Training

For RQ4, distillation-based LoRA fine-tuning is applied using the full model's output as the target: $y_{target} = f_{base}(x)$, optimizing the truncated model $\tilde{f}_K$ to approximate the behavior of the base model.

## Key Experimental Results

### Main Results

| Experiment | Image Tokens | Text Tokens |
|---|---|---|
| Matrix Entropy | Rapid stabilization | Continuous fluctuation |
| Intrinsic Dimensionality | Early convergence | Alternating expansion and contraction |
| Trajectory Curvature | Near-constant | Large and variable |
| Layer Substitution Similarity | ~1.0 (depth-invariant) | Degrades with layer gap |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| MCQ truncation | Smooth degradation | Single-token prediction is robust |
| VQA truncation | Significant degradation | Exact matching requires deep processing |
| Caption truncation | Severe degradation | Multi-token generation is most sensitive |
| Post-distillation fine-tuning | Good caption recovery | Coarse semantics can be reassigned |
| Post-distillation fine-tuning | Poor ChartQA recovery | Precise visual alignment is irreversible |

### Key Findings

- Image representations exhibit a consistent early-stabilization pattern across all six model families (3B–72B), indicating that this is a structural property of multimodal Transformers rather than a scale-dependent artifact.
- Under deterministic decoding, reducing visual depth perturbs intermediate reasoning trajectories more than final outputs — image tokens influence reasoning structure more than final conclusions.
- Fine-tuning not only recovers average performance but also reduces variance across decoding strategies.

## Highlights & Insights

- **Systematic Evidence of Modality Asymmetry**: Three independent metrics consistently reveal a structural asymmetry in which visual tokens converge early while text tokens continue to evolve throughout all layers.
- **Precise Distinction Between "Interchangeable" and "Dispensable"**: Functional interchangeability implies that deeper processing does not alter semantics, but does not imply that image tokens are unnecessary in deeper layers.
- **Fine-Grained Task Dependency Analysis**: The differing visual depth requirements of single-token prediction, multi-token generation, and open-ended reasoning provide concrete guidance for VLM architectural design.

## Limitations & Future Work

- The analysis is conducted primarily on limited datasets such as BLINK and Flickr8K.
- Truncation experiments employ hard truncation (complete removal of image tokens), without exploring more gradual strategies such as progressive sparsification.
- The indirect influence of visual tokens on text tokens via attention is not examined.

## Related Work & Insights

- **vs. FiT/SparseVLM**: These works assume redundancy and design pruning mechanisms accordingly; this paper provides a representational explanation for why pruning is effective.
- **vs. ShortV**: ShortV explores the limited novelty of visual representations in deep layers; this paper offers a more comprehensive analysis of representational dynamics.

## Rating

- Novelty: ⭐⭐⭐⭐ — A new perspective on understanding VLM efficiency through the lens of representational dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly systematic, covering 6 model families × multiple tasks × multiple metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ — Research-question-driven structure is clear and elegant.
- Value: ⭐⭐⭐⭐ — Offers far-reaching implications for VLM architectural design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[ICLR 2026\] Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](../../ICLR2026/multimodal_vlm/do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)
- [\[CVPR 2026\] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](do_vision-language_models_leak_what_they_learn_adaptive_token-weighted_model_inv.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
