---
title: >-
  [Paper Note] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability
description: >-
  [CVPR 2025][Image Generation][Diffusion bias] This paper proposes the DiffLens framework, which disentangles the internal neurons of diffusion models into a monosemantic feature space using sparse autoencoders (k-SAEs), and then localizes specific features driving bias generation through gradient-based attribution. This enables fine-grained control and mitigation of social biases such as gender and race while maintaining image quality.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion bias"
  - "mechanistic interpretability"
  - "sparse autoencoders"
  - "bias features"
  - "bias mitigation"
date: 2026-05-08
content_hash: bdd221d61d4a8319
---

# Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability

**Conference**: CVPR 2025  
**arXiv**: [2503.20483](https://arxiv.org/abs/2503.20483)  
**Code**: [Project Page](https://foundation-model-research.github.io/difflens)  
**Area**: Image Generation  
**Keywords**: Diffusion bias, mechanistic interpretability, sparse autoencoders, bias features, bias mitigation

## TL;DR
This paper proposes the DiffLens framework, which disentangles the internal neurons of diffusion models into a monosemantic feature space using sparse autoencoders (k-SAEs), and then localizes specific features driving bias generation through gradient-based attribution. This enables fine-grained control and mitigation of social biases such as gender and race while maintaining image quality.

## Background & Motivation

**Background**: Although diffusion models deliver exceptional generation quality, they often generate content with social biases, such as gender, race, and age stereotypes. Existing bias mitigation methods roughly fall into two categories: training/fine-tuning from scratch (resource-intensive) and guiding/editing the generation process (ignoring internal model mechanisms).

**Limitations of Prior Work**: Guidance-based generation methods do not understand the specific internal mechanisms driving bias within the model, which can lead to over-correction or unintended side effects on non-target attributes. Directly editing the latent space (e.g., h-space) suffers from neuron polysemanticity (where one neuron corresponds to multiple unrelated concepts), meaning editing one attribute inevitably alters others.

**Key Challenge**: There is a need to precisely localize and modify the specific model components that generate biases, but the generative nature of diffusion models and neuron polysemanticity make fine-grained analysis extremely difficult.

**Goal**: To discover the decision-making mechanisms (bias features) that causally drive biased outputs within diffusion models, and to precisely control bias levels by regulating these features.

**Key Insight**: Leveraging sparse autoencoder (SAE) methods from LLM interpretability to disentangle diffusion model neuron activations into a sparse, monosemantic feature space.

**Core Idea**: k-SAE disentanglement $\rightarrow$ gradient attribution to localize bias features $\rightarrow$ scaling bias features to control output distribution.

## Method

### Overall Architecture
A three-stage pipeline: (1) mapping the latent states of the U-Net bottleneck layer to a high-dimensional sparse semantic space using a k-SAE; (2) training a lightweight classifier to estimate the probability of biased attributes, and computing attribution scores of each semantic feature to the bias using integrated gradients; (3) controlling the bias level of generated content by amplifying or suppressing the highest-scoring bias features.

### Key Designs

1. **k-Sparse Autoencoder (k-SAE) Disentanglement**:

    - **Function**: Disentangle polysemantic neuron activations into a monosemantic feature space.
    - **Mechanism**: The encoder $s = \text{TopK}(W_{enc}(h - b_{pre}))$ maps the n-dimensional latent state to an m-dimensional sparse vector (m≫n), keeping only the k hottest features (fired features) and setting the rest to zero. The decoder reconstructs the original latent state. The training objective is to minimize reconstruction MSE.
    - **Design Motivation**: Original neurons are polysemantic, meaning a single neuron participates in encoding multiple unrelated concepts. k-SAE segregates monosemantic features in a high-dimensional space so that each dimension corresponds to a clear semantic concept, laying the foundation for precise editing.

2. **Gradient-Based Bias Feature Localization**:

    - **Function**: Identify features in the semantic space that are most highly correlated with bias generation.
    - **Mechanism**: Define a bias metric $F_x(s) = \Pr(y|s)$ (where y represents gender/race categories) and use integrated gradients to calculate the attribution score for each semantic feature $s_i$: $S(s_i; x) = (s_i - s'_i) \cdot \int_0^1 \frac{\partial F_x(s' + \alpha(s-s'))}{\partial s_i} d\alpha$. By aggregating attribution scores across N generated samples, the top-τ highest attribution features are selected as bias features.
    - **Design Motivation**: Integrated gradients consider both the magnitude of the feature values and their marginal effects on the output, making it more accurate than simple gradients or activation values. This process only needs to be performed once to permanently label bias features.

3. **Bias Feature Scaling Control**:

    - **Function**: Control the level of bias by adjusting the activation values of bias features.
    - **Mechanism**: For each feature in the identified bias feature set A, multiply it by a scaling factor — suppressing it (multiplying by a small coefficient) reduces bias, while amplifying it (multiplying by a large coefficient) increases bias. The modified features are then mapped back to the original latent space via the decoder to continue generation.
    - **Design Motivation**: Since the features are monosemantic, scaling an individual feature does not affect other attributes, thereby achieving fine-grained control.

### Loss & Training
The k-SAE is trained using a reconstruction MSE loss. The bias classifier is trained using attribute labels of generated samples (e.g., a gender classifier).

## Key Experimental Results

### Main Results (CelebA-HQ Unconditional Generation, Gender Bias)

| Method | Bias Mitigation Effect | FID Change | Other Attribute Preservation |
|------|------------|---------|------------|
| Prompt Editing | Moderate | Degraded | Poor (affects other attributes) |
| Attention Editing | Moderate | Slight Degradation | Moderate |
| Fair Diffusion | Good | Degraded | Moderate |
| **DiffLens** | **Best** | **Minimal Degradation** | **Best** |

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| Direct editing of h-space (no SAE) | Changes gender while unintentionally altering hairstyle, skin color, etc. |
| DiffLens editing | Precisely changes gender while other attributes remain unchanged |
| Semantics of different k-SAE features | Discovers that different features control fine-grained attributes such as hairstyle, skin color, and facial structures separately |

### Key Findings
- Bias features discovered by DiffLens indeed control the generation of attributes such as gender and race, showing significant effects after regulation.
- Compared to direct h-space editing, DiffLens editing does not unintentionally alter non-target attributes (e.g., hairstyle remains unchanged when changing gender).
- Features disentangled by the k-SAE show clear semantic correspondence — different features control hair length, skin color, wearing glasses, etc., respectively.
- The method is applicable to both unconditional and conditional (Stable Diffusion) diffusion models.

## Highlights & Insights
- Migrates mechanistic interpretability (SAE) methods from LLMs to diffusion models for bias analysis for the first time, opening up a new research direction.
- The disentanglement of k-SAE on diffusion models is surprisingly effective — different features indeed correspond to different fine-grained facial attributes.
- The combination of integrated gradients and SAE is not only useful for bias mitigation but also serves as a general tool for understanding "how decisions are made" in diffusion models.

## Limitations & Future Work
- The k-SAE needs to be trained on the activations of the target model; replacing the model requires retraining.
- Validation is currently conducted mainly on faces and simple scenes; bias in complex scenes (e.g., occupational stereotypes) has not been fully tested.
- More complex interactions may exist between bias and non-bias features; simple scaling might be insufficient.
- The approach can be extended to more bias dimensions (e.g., age, body shape) and additional generative model architectures (e.g., DiT).

## Related Work & Insights
- **vs Fair Diffusion (guided generation)**: Fair Diffusion injects fairness guidance during sampling without understanding internal model mechanisms, which may affect quality; DiffLens precisely localizes the source of the bias.
- **vs Attention Editing**: Attention editing operates on a macro level, making it difficult to alter only the target attributes; DiffLens operates at the level of fine-grained semantic features.
- **vs Anthropic SAE on LLM**: Migrating the SAE technique from LLM interpretability to visual generative models represents an important extension of this research direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to apply mechanistic interpretability to diffusion model bias analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Unconditional and conditional models, multiple bias attributes, and comprehensive comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and excellent illustrations.
- Value: ⭐⭐⭐⭐⭐ Holds significant importance for both AI safety and model interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](../../ICML2025/image_generation/towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)

</div>

<!-- RELATED:END -->
