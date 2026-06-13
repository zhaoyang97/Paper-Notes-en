---
title: >-
  [Paper Note] All-in-One Slider for Attribute Manipulation in Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Attribute Manipulation] The proposed All-in-One Slider framework trains a lightweight Attribute Sparse Autoencoder on the intermediate layer embeddings of a text encoder. It decomposes attri…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Attribute Manipulation"
  - "Sparse Autoencoder"
  - "Text Embedding Space"
  - "Disentangled Representation"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: d7f9c7e9856cb5cf
---

# All-in-One Slider for Attribute Manipulation in Diffusion Models

**Conference**: CVPR 2026  
**arXiv**: [2508.19195](https://arxiv.org/abs/2508.19195)  
**Code**: Available (provided on project page)  
**Area**: Diffusion Models/Image Generation  
**Keywords**: Attribute Manipulation, Sparse Autoencoder, Text Embedding Space, Disentangled Representation, Zero-shot Generalization

## TL;DR
The proposed All-in-One Slider framework trains a lightweight Attribute Sparse Autoencoder on the intermediate layer embeddings of a text encoder. It decomposes attributes into disentangled directions within a high-dimensional sparse activation space, enabling continuous, fine-grained, and composable control of multiple facial attributes with a single module. It also demonstrates zero-shot continuous manipulation capabilities for unseen attributes (e.g., ethnicity, celebrities).

## Background & Motivation
T2I diffusion models have made significant progress in image generation, but progressive and fine-grained control over specific attributes remains difficult. Traditional prompt engineering (e.g., appending "with a big smile" to a prompt) only allows for coarse-grained, non-continuous manipulation and inevitably affects unrelated attributes (e.g., hairstyle, identity).

Existing attribute manipulation methods follow a **One-for-One** paradigm—training an independent slider module for each attribute (e.g., ConceptSlider, AttributeControl). This leads to three problems: (1) every new attribute requires additional training, accumulating time and compute costs; (2) parameter redundancy as multiple slider parameters pile up; (3) limited flexibility and scalability, making it difficult to handle a large number of different attributes in practice.

The core insight of this paper is "break it down to build it up": drawing on the success of Sparse Autoencoders (SAE) in discovering interpretable semantic units in LLMs, the authors construct a unified, disentangled attribute latent space within the diffusion model's text embedding space, allowing all attributes to share one lightweight module.

## Method

### Overall Architecture
The All-in-One Slider consists of two stages:
- **Stage 1 (Attribute Disentanglement Training)**: An Attribute Sparse Autoencoder is trained (unsupervised) on the intermediate layer embeddings of the SDXL text encoder.
- **Stage 2 (Attribute Manipulation)**: Using the trained SAE, target attribute text is encoded into sparse directions, scaled by a scalar $\lambda$, and added to the original prompt embedding.

### Key Designs
1. **Embedding Extraction**:
    - Function: Extracts hidden states from the intermediate layers of the SDXL dual text encoder.
    - Mechanism: Features are extracted from the 11th layer of the first encoder and the 29th layer of the second encoder, concatenated to form a joint representation $x \in \mathcal{D}$.
    - Design Motivation: Intermediate layers retain both semantic information and identity features; shallow layers are too low-level, while deep layers are too semantic.

2. **Attribute Sparse Autoencoder (SAE)**:
    - Function: Maps text embeddings to a high-dimensional sparse latent space (Attribute Latent Space) to achieve attribute disentanglement.
    - Mechanism: The encoder uses top-k sparse activation:
    $$z_{\text{ALS}} = \text{Top-}k(\text{ReLU}(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}))$$
      The decoder reconstructs the original embedding: $\hat{x} = W_{\text{dec}} z_{\text{ALS}} + b_{\text{pre}}$.
      The training objective is reconstruction loss + auxiliary loss (to activate dead neurons):
    $$\mathcal{L} = \|x - \hat{x}\|_2^2 + \alpha \mathcal{L}_{\text{aux}}$$
      The auxiliary loss selects the $k_{\text{aux}}$ least active neurons to reconstruct the residual $r = x - \hat{x}$, ensuring broad semantic coverage of the latent space.
    - Design Motivation: Top-k sparsity is key to attribute disentanglement—each attribute activates only a small, unique subset of neurons, naturally disentangling different attributes.

3. **Attribute Manipulation**:
    - Function: Achieves continuous, fine-grained attribute control by modifying text embeddings during inference.
    - Mechanism: After encoding target attribute $A$ to obtain a direction in the latent space, it is scaled by $\lambda$ and added to the original embedding:
    $$x_{\text{manipulated}} = x + W_{\text{dec}}(\lambda \times \text{ENC}(x_A))$$
      $\lambda$ controls manipulation intensity (positive for enhancement, negative for reduction).
    - Design Motivation: Because the latent space is compositional, multiple attribute directions can be directly superimposed for combined manipulation; unseen attribute text embeddings still trigger adaptive composite activations in the trained space.

### Loss & Training
- Training Data: 52 controllable facial attributes $\times$ 1000 samples = 52,000 text samples.
- Image generation via SDXL, 50-step sampling, classifier-free guidance = 7.5.
- Multi-subject scenario extension: Introduces an Attention Pooling Aggregator module + consistency loss $\mathcal{L}_{\text{multi}} = \mathcal{L}_{\text{sae}} + \eta \mathcal{L}_{\text{cons}}$.

## Key Experimental Results

### Main Results

| Setting | Method | QS (Old) | IS (Old) | QS (Smile) | IS (Smile) | QS (Makeup) | IS (Makeup) |
|---|---|---|---|---|---|---|---|
| Single Attribute | ConceptSlider | 3.794 | 0.434 | 4.144 | 0.496 | 4.542 | 0.653 |
| Single Attribute | AttControl | 4.039 | 0.601 | 4.395 | 0.695 | 4.268 | 0.604 |
| Single Attribute | **Ours** | **4.049** | **0.716** | 4.265 | 0.637 | 4.291 | **0.742** |
| Multi-Attribute | ConceptSlider | 4.150 | 0.499 | 3.801 | 0.522 | 4.059 | 0.479 |
| Multi-Attribute | AttControl | 3.667 | 0.376 | 4.056 | 0.635 | 4.248 | 0.515 |
| Multi-Attribute | **Ours** | **4.212** | **0.688** | **4.428** | **0.628** | **4.297** | **0.635** |

Advantages are more pronounced in multi-attribute scenarios: Old+Makeup QS 4.428 vs. 4.056 (second best), a Gain of 0.37.

### Ablation Study

| Configuration | QS (Avg) | IS (Avg) | Description |
|---|---|---|---|
| Layer 8/28 | 4.124 | 0.635 | Shallow, insufficient semantics |
| Layer 9/30 | 4.144 | 0.669 | Sub-optimal |
| Layer 10/24 | 4.185 | 0.718 | Good balance |
| **Layer 10/28** | **4.202** | **0.698** | Best overall balance |

$\lambda$ ablation: $\lambda=0.15$ results in under-editing, $\lambda=0.30$ provides stronger attribute expression but identity preservation decreases; the "old" attribute is most deeply entangled with identity features.

### Key Findings
- For multi-attribute combined manipulation, this method avoids conflicts and maintains high semantic consistency due to the disentangled latent space.
- Zero-shot generalization to unseen ethnicities (e.g., African, Chinese, Indian) and celebrity identities (Obama, Einstein).
- Transferable to different diffusion backbones such as SD v1.4 and SDXL-Turbo.
- Equally effective when extended to photographic style manipulation (40 styles including B&W, Golden Hour).

## Highlights & Insights
- Paradigm shift from One-for-One to All-in-One: train once, apply to all attributes forever, greatly lowering the barrier to entry for attribute manipulation.
- Top-k sparsity of SAE is naturally suited for attribute disentanglement, representing a clever innovation migrated from the field of LLM interpretability to visual generation.
- Zero-shot compositional generalization suggests that the attribute correspondences learned by text encoders are well-structured.
- Extremely lightweight module (only SAE encoder/decoder weights), meaning no modifications to the base model are required.

## Limitations & Future Work
- The "old" attribute is deeply entangled with identity features; identity preservation drops noticeably under strong manipulation.
- Currently primarily validated on facial attributes; control effectiveness for full-body or scene-level attributes has not been fully explored.
- Relies on the semantic quality of the pre-trained text encoder; attributes poorly covered by the text encoder may exhibit limited effects.
- Multi-subject scenarios require additional Attention Pooling fine-tuning, not representing a zero-cost solution.

## Related Work & Insights
- The core difference from ConceptSlider (weight space editing) and AttributeControl (embedding space training) is that the former requires 1-LoRA-per-attribute and the latter requires 1-supervised-pair-per-attribute; ours covers all with one unsupervised training session.
- While SAE applications in diffusion (SAeUron, Diffusion Lens, Unpacking SDXL Turbo) focus on interpretability, this paper is the first to use it for controllable attribute manipulation.
- Insight: Similar sparse disentanglement strategies could potentially be used for motion attribute control in video generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use K-SAE for unified attribute manipulation; a paradigm innovation from One-for-One to All-in-One.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive experiments covering single/multi-attribute, zero-shot, cross-model, multi-subject, and style extensions with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic method description, and intuitive chart design.
- Value: ⭐⭐⭐⭐⭐ High practical value as a lightweight, scalable, and zero-shot generalizing attribute control solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)
- [\[ICCV 2025\] CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation](../../ICCV2025/image_generation/compslider_compositional_slider_for_disentangled_multiple-attribute_image_genera.md)
- [\[CVPR 2026\] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[ICLR 2026\] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](../../ICLR2026/image_generation/when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [\[CVPR 2026\] TokenLight: Precise Lighting Control in Images using Attribute Tokens](tokenlight_precise_lighting_control_in_images_using_attribute_tokens.md)

</div>

<!-- RELATED:END -->
