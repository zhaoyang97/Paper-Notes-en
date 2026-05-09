---
title: >-
  [Paper Note] Interpretable Debiasing of Vision-Language Models for Social Fairness
description: >-
  [CVPR 2026][Multimodal VLM][VLM debiasing] This paper proposes DeBiasLens, which trains a Sparse Autoencoder (SAE) on VLM encoders to localize "social neurons" encoding social attributes, then selectively deactivates these neurons at inference time to mitigate bias. The method reduces Max Skew by 9–16% on CLIP and reduces gender bias rates by 40–50% on InternVL2, while preserving general performance.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM debiasing
  - social fairness
  - sparse autoencoder
  - interpretability
  - neuron manipulation
date: 2026-05-08
content_hash: f535fa8bdf2979d3
---

# Interpretable Debiasing of Vision-Language Models for Social Fairness

**Conference**: CVPR 2026
**arXiv**: [2602.24014](https://arxiv.org/abs/2602.24014)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: VLM debiasing, social fairness, sparse autoencoder, interpretability, neuron manipulation

## TL;DR
This paper proposes DeBiasLens, which trains a Sparse Autoencoder (SAE) on VLM encoders to localize "social neurons" encoding social attributes, then selectively deactivates these neurons at inference time to mitigate bias. The method reduces Max Skew by 9–16% on CLIP and reduces gender bias rates by 40–50% on InternVL2, while preserving general performance.

## Background & Motivation
**State of the Field**: VLMs and LVLMs inherit and amplify social biases from large-scale training data — e.g., CLIP retrieves male-skewed results for "CEO," and InternVL exhibits gender bias in ambiguous contexts. Existing debiasing approaches include fine-tuning, prompt tuning, and pruning.

**Limitations of Prior Work**: Existing debiasing methods address surface-level bias symptoms without targeting the propagation mechanisms of bias in internal representations. Although pruning attempts to identify critical parameters, the polysemantic nature of neurons (a single neuron simultaneously encodes both biased and useful knowledge) means debiasing often sacrifices general performance.

**Root Cause**: Bias and useful knowledge are entangled in model weights, making direct weight modification inevitably lead to performance degradation.

**Paper Goals**: To precisely localize and manipulate bias-related monosemantic features within an interpretable framework without affecting useful knowledge.

**Starting Point**: SAEs are leveraged to decompose the entangled feature space into sparse, monosemantic neurons (satisfying monosemanticity), enabling bias-related "social neurons" to be independently localized and manipulated.

**Core Idea**: SAE decouples polysemantic features into monosemantic ones → social neurons encoding specific social attributes are identified → deactivating these neurons at inference time eliminates bias signals.

## Method

### Overall Architecture
**Three-stage pipeline**: (1) Attach and train an SAE on the last layer of the VLM encoder; (2) Localize social neurons via activation consistency and specificity analysis; (3) Deactivate social neurons at inference time, blending reconstructed features with original features through weighted mixing.

### Key Designs

1. **SAE Training**:

    - **Function**: Decomposes encoder output into high-dimensional sparse representations.
    - **Mechanism**: $\phi(\mathbf{v}) = \sigma(\mathbf{W}_{enc}^\top(\mathbf{v} - \mathbf{b}_1))$, using Matryoshka SAE multi-scale reconstruction loss with an expansion factor of 8.
    - **Design Motivation**: SAE training requires no social attribute labels; training on face/description datasets allows the SAE to automatically capture social attribute features.
    - **Key Findings**: After attaching the SAE, the difference in cosine similarity between image pairs sharing social attributes and random image pairs increases significantly.

2. **Social Neuron Probing**:

    - **Function**: Identifies neurons in the SAE's sparse representation that encode specific social attributes (gender/age/ethnicity).
    - **Mechanism**: Computes each neuron's effectiveness within a specific group (non-zero activation ratio ≥ τ=0.9), derives the group-exclusive neuron set $\mathcal{N}_g = \mathcal{E}_g \setminus \bigcup_{h \neq g} \mathcal{E}_h$, and selects the top neurons with the highest mean activation.
    - **Design Motivation**: Group specificity combined with high consistency identifies monosemantic neurons encoding group-specific social attributes.

3. **Social Neuron Manipulation at Inference**:

    - **Function**: Selectively deactivates social neurons at inference time to eliminate bias signals.
    - **Mechanism**: Social neuron activations in the SAE are set to $\gamma$ (typically 0), yielding reconstructed features $\hat{\mathbf{v}} = \psi(\mathbf{z}')$, which are then blended with the original features as $\mathbf{v}' = \alpha\hat{\mathbf{v}} + (1-\alpha)\mathbf{v}$, with $\alpha=0.6$.
    - **Design Motivation**: Only the SAE's sparse representation is modified without touching the original model weights, minimizing impact on general capabilities.

## Key Experimental Results

### Main Results (CLIP ViT-B/16 Gender Bias, Max Skew↓)

| Method | Interpretable? | Adj | Occup | Act | Ster |
|--------|---------------|-----|-------|-----|------|
| CLIP Baseline | - | 21.9 | 33.5 | 19.8 | 32.5 |
| Bend-VLM | ✗ | 10.8 | 10.2 | 9.8 | 9.1 |
| SANER | ✗ | 8.9 | 14.5 | 7.7 | - |
| **DeBiasLens (T)** | **✓** | **7.1** | **16.2** | **14.2** | **8.1** |
| **DeBiasLens (I)** | **✓** | 14.2 | 21.5 | 20.0 | 18.3 |

### LVLM Debiasing Results

| Configuration | Gender Bias Rate Reduction | General Performance Drop |
|---------------|---------------------------|--------------------------|
| DeBiasLens-InternVL2 (α=0.6) | 40–50% | Only 4–10 points |
| Pruning Methods | Similar | Larger drop |
| Prompt Engineering | Limited | Minimal |

### Key Findings
- DeBiasLens (T) achieves the best results on adjective and stereotype prompts without requiring attribute label supervision during training.
- Deactivating only the top-1 social neuron achieves performance comparable to deactivating all effective neurons, confirming that neurons do not interfere with each other.
- Gender neurons exhibit high specificity — deactivating gender neurons does not affect age bias; however, age neurons show cross-attribute effects (40% of age neurons exhibit gender skew).
- The image encoder is more effective for high-resolution VLMs (ViT-L/14@336), while the text encoder is more effective for standard-resolution models.

## Highlights & Insights
- **Interpretability-driven debiasing** represents a fundamentally new paradigm: rather than mitigating biased outputs in a black-box manner, it precisely localizes and manipulates the internal mechanisms that produce bias.
- The monosemantic property of SAE neurons makes them ideal tools for debiasing: each neuron encodes a single concept, so deactivation produces no cascading effects.
- The framework applies to both encoder-only (CLIP) and encoder-decoder (InternVL2) VLMs, demonstrating strong generality.
- Only intermediate representations are modified without altering model weights, making deployment straightforward.

## Limitations & Future Work
- The social neuron probing stage still requires social attribute labels to partition groups, even though SAE training does not.
- Validation is primarily focused on gender bias; ablation studies on age and ethnicity are relatively limited.
- The SAE expansion factor and threshold τ require hyperparameter tuning.
- Debiasing effectiveness across cultures and languages has not been verified.

## Related Work & Insights
- **vs. Bend-VLM**: Bend-VLM directly debiases embeddings in a black-box manner; DeBiasLens operates through interpretable neuron manipulation, offering transparency and auditability.
- **vs. SANER**: SANER trains residual layers on the text encoder to erase attribute information; DeBiasLens achieves more precise debiasing through SAE-based feature disentanglement followed by selective deactivation.
- **vs. MMNeuron**: MMNeuron identifies attribute-specific neurons in pretrained weights, but weight-level neurons are polysemantic; SAE neurons are monosemantic, enabling more precise debiasing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of SAE for interpretable VLM debiasing, with a unique and well-motivated entry point.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple VLMs/LVLMs, multiple evaluation dimensions, and neuron specificity validation.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly articulated; the concept of "social neurons" is vivid and intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a new interpretable and auditable tool for AI fairness research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[AAAI 2026\] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models](../../AAAI2026/multimodal_vlm/biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)
- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](../../ICCV2025/multimodal_vlm/interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)

<!-- RELATED:END -->
