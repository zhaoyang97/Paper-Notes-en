---
title: >-
  [Paper Note] Advancing Textual Prompt Learning with Anchored Attributes
description: >-
  [ICCV 2025][Multimodal VLM][Prompt Learning] This paper proposes ATPrompt, which embeds general-purpose attribute tokens (e.g., color, shape) into textual prompts, extending the learning space of soft prompts from a one-dimensional class level to a multi-dimensional attribute level. ATPrompt serves as a plug-and-play module that integrates seamlessly into existing textual prompt learning methods, consistently improving baseline performance across 11 datasets.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Prompt Learning
  - CLIP
  - Attribute Anchoring
  - Zero-Shot Generalization
  - Differentiable Attribute Search
date: 2026-05-08
content_hash: 9c028e30fec89611
---

# Advancing Textual Prompt Learning with Anchored Attributes

**Conference**: ICCV 2025
**arXiv**: [2412.09442](https://arxiv.org/abs/2412.09442)
**Code**: [https://github.com/zhengli97/ATPrompt](https://github.com/zhengli97/ATPrompt)
**Area**: Multimodal VLM
**Keywords**: Prompt Learning, CLIP, Attribute Anchoring, Zero-Shot Generalization, Differentiable Attribute Search

## TL;DR
This paper proposes ATPrompt, which embeds general-purpose attribute tokens (e.g., color, shape) into textual prompts, extending the learning space of soft prompts from a one-dimensional class level to a multi-dimensional attribute level. ATPrompt serves as a plug-and-play module that integrates seamlessly into existing textual prompt learning methods, consistently improving baseline performance across 11 datasets.

## Background & Motivation
Vision-language models (e.g., CLIP) establish image-text alignment via contrastive learning, and prompt learning methods (e.g., CoOp) efficiently adapt these models to downstream tasks by learning soft prompt tokens. However, a core limitation of existing textual prompt learning methods is that training aligns images only with predefined known classes, making it difficult to establish accurate associations with unseen classes.

This leads to a fundamental problem: models perform well on base classes but exhibit limited generalization to novel classes, essentially overfitting to the classes seen during training.

**Intuition**: When humans encounter unfamiliar categories, they reason through attributes such as color, shape, and texture — "a cheetah is a felid with a small head, short yellowish fur, and black spots" is far more descriptive than "this is a cheetah." Attributes can serve as bridges connecting unknown categories to existing knowledge.

**Key insight**: **General (cross-category) attributes are more efficient and robust than intra-class attributes.** ATPrompt embeds general attributes as learned components within the prompt template rather than as learning targets, thereby expanding the expressive capacity of soft prompts without additional computational overhead.

## Method

### Overall Architecture
ATPrompt transforms the conventional "soft prompt + class token" format into a hybrid format of "attribute soft prompt + attribute token + class soft prompt + class token." Both shallow and deep variants are provided for compatibility with existing methods of varying depth. An automated attribute selection pipeline based on LLM-driven attribute pool construction and differentiable search is also designed.

### Key Designs

1. **Attribute-Anchored Textual Prompt (Shallow Variant)**:

    - **Function**: Embeds fixed attribute tokens and corresponding learnable soft tokens at the input layer of the text encoder.
    - **Mechanism**: Using two general attributes A and B as an example, the textual prompt becomes: $P_T = [T_{a_1}]\ldots[T_{a_m}][\text{A}][T_{b_1}]\ldots[T_{b_m}][\text{B}][T_1]\ldots[T_M][\text{CLS}]$, where attribute tokens are fixed hard tokens, and both attribute-related and class-related soft tokens are learnable parameters.
    - **Design Motivation**: The anchoring effect of attribute tokens guides soft tokens to learn representations that encode not only class-specific information but also general attribute-related knowledge. When encountering unseen classes, these attribute-related tokens provide additional information to facilitate better image-text alignment.

2. **Deep Variant**:

    - **Function**: Introduces soft tokens at deeper Transformer layers, but selectively discards and re-injects only the class-related soft tokens while retaining attribute-related hard and soft tokens across layers.
    - **Mechanism**: The computation at layer $i$ is $[\text{F}_i, \_, \text{CLS}_i] = L_i([\text{F}_{i-1}, \text{T}_{i-1}, \text{CLS}_{i-1}])$, where $\text{F}$ denotes attribute features that are preserved across layers without being discarded, and $\text{T}$ denotes class soft tokens that are discarded and re-injected in the conventional manner.
    - **Design Motivation**: Fully discarding attribute tokens disrupts cross-layer continuity of attribute representations, creating a "gap" between newly introduced lower-layer tokens and existing higher-layer tokens.

3. **Differentiable Attribute Search**:

    - **Function**: Automatically selects the most suitable attribute combination and quantity for a given downstream task.
    - **Mechanism**: (a) An LLM (GPT-4o) is used to first generate descriptions for each known class, then summarize independent attribute bases (e.g., color, shape, size, habitat, behavior), forming a search space of $2^N - 1$ combinations for $N$ bases. (b) Inspired by DARTS, the discrete selection is relaxed into a softmax-weighted sum: $f(x, v; \alpha, \theta) = \sum_{i \in \mathcal{V}} \frac{\exp(\alpha_i)}{\sum_{i'} \exp(\alpha_{i'})} f(x, v_i; \theta)$. Attribute weights $\alpha$ (minimizing validation loss) and soft prompt parameters $\theta$ (minimizing training loss) are optimized alternately to identify the optimal combination.
    - **Design Motivation**: Directly querying an LLM for attributes cannot determine the optimal number, and querying by class name alone may introduce semantic bias. Differentiable search operates at the token level, making it far more efficient than conventional NAS approaches, converging in approximately 5 epochs (under 5 minutes).

### Loss & Training
Training uses the standard cross-entropy loss: $L_{train} = \sum_{x \in D} \text{CE}(f(x; v, \theta), c)$. Attribute search employs bi-level optimization, with $\alpha$ optimized on the validation set and $\theta$ on the training set. Search is performed only once; the selected attributes are then used for formal training. By default, 5 attribute bases (31 candidate combinations) are used.

## Key Experimental Results

### Main Results: Base-to-Novel Generalization (Average HM across 11 Datasets)

| Method | Base | Novel | HM |
|--------|------|-------|-----|
| CoOp | 82.69 | 63.22 | 71.66 |
| CoOp + ATPrompt | 82.68 | 68.04 | **74.65 (+2.99)** |
| CoCoOp | 80.47 | 71.69 | 75.83 |
| CoCoOp + ATPrompt | 81.69 | 74.54 | **77.95 (+2.12)** |
| MaPLe | 82.28 | 75.14 | 78.55 |
| MaPLe + ATPrompt | 82.98 | 75.76 | **79.21 (+0.66)** |
| PromptKD | 86.96 | 80.73 | 83.73 |
| PromptKD + ATPrompt | 87.05 | 81.82 | **84.35 (+0.62)** |

### Ablation Study

| Ablation | Base | Novel | HM | Notes |
|----------|------|-------|-----|-------|
| CoOp + ATPrompt (color, shape) | 76.27 | 70.60 | 73.33 | Default config |
| Class token first | 76.12 | 70.50 | 73.20 | Slight degradation |
| Class token in the middle | 76.13 | 70.29 | 73.09 | Slight degradation |
| Deep: discard and re-inject all | 76.83 | 70.10 | 73.31 | Disrupts continuity |
| Deep: partial discard and re-inject | 76.87 | 70.44 | 73.51 | Still suboptimal |
| Deep: retain all attribute tokens | 76.94 | 70.72 | 73.70 | Best |

### Key Findings
- ATPrompt consistently improves average HM across all 5 baseline methods (+0.62 to +2.99).
- Gains are primarily driven by significant improvements in novel class accuracy, confirming that attributes effectively bridge known and unknown categories.
- Cross-dataset generalization improves by 0.45–1.38%; domain generalization improves by 0.33–0.90%.
- Attribute ordering has minimal impact on performance (variations within a reasonable range), indicating robustness.
- Searched attributes (e.g., color+shape for ImageNet) align with domain intuition and outperform manual selection.

## Highlights & Insights
- **Universal plug-and-play improvement**: Only the textual prompt format is modified without altering model architecture or training pipeline, making ATPrompt a seamless drop-in replacement for any text-prompt-based method.
- **General attributes vs. intra-class attributes**: General attributes need not be re-acquired for new classes; a single search pass is reusable across all categories, making them more practical than intra-class attributes.
- **Differentiable search**: Applying the DARTS paradigm to attribute selection is both novel and efficient.

## Limitations & Future Work
- Improvements are limited to the text branch; visual prompts (e.g., VPT) and joint multimodal prompting are not explored.
- The magnitude of improvement diminishes for more recent methods with additional learnable modules (e.g., PromptKD).
- The attribute space remains constrained by LLM generation quality, and an independent search is required for each task.

## Search Result Examples

| Dataset | Attribute Candidates | Search Result |
|---------|----------------------|---------------|
| ImageNet | color, size, shape, habitat, behavior | (color, shape) |
| Caltech101 | shape, color, material, function, size | (shape, size) |
| OxfordPets | loyalty, affection, energy, playfulness, intelligence | (playfulness, energy) |
| StanfordCars | design, engine, performance, luxury, color | (luxury) |
| Flowers102 | color, flower, habitat, growth, season | (color, habitat, growth) |

The search converges in approximately 5 epochs, taking under 5 minutes on a single A800 GPU. Results are highly consistent with domain intuition and automatically determine the optimal number of attributes.

## Related Work & Insights
- **vs. ArGue**: ArGue leverages LLMs to mine intra-class attributes and constructs multiple text groups for regularization and ensemble prediction, incurring higher computational cost. ATPrompt embeds general attributes directly into the prompt as learned components, resulting in a more lightweight design.
- **vs. VCD**: VCD uses LLMs to decompose class names into intra-class attributes, requiring re-acquisition for novel classes. ATPrompt's general attributes, once searched, apply to all classes without re-querying.
- **vs. KgCoOp/PromptSRC**: These methods mitigate overfitting through regularization but do not address the inherent limitations of the prompt format itself. ATPrompt fundamentally expands the expressive space of the prompt.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of attribute-anchored prompting is novel with clear intuition; the integration of differentiable search is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 datasets × 5 baselines × 3 experimental settings with exhaustive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivation–method–experiment narrative is logically coherent, with intuitive architectural comparisons to prior work.
- **Value**: ⭐⭐⭐⭐ High practical utility as a plug-and-play module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)
- [\[NeurIPS 2025\] Learning Skill-Attributes for Transferable Assessment in Video](../../NeurIPS2025/multimodal_vlm/learning_skill-attributes_for_transferable_assessment_in_video.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)

</div>

<!-- RELATED:END -->
