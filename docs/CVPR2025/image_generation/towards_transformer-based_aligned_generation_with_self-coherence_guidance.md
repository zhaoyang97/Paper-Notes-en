---
title: >-
  [Paper Note] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance
description: >-
  [CVPR 2025][Image Generation][Text-to-Image] Proposes Self-Coherence Guidance (SCG), a training-free alignment method tailored for Transformer-structured text-guided diffusion models, which improves attribute binding, fine-grained attribute binding, and style binding by directly optimizing cross-attention maps (rather than latent variables).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-to-Image"
  - "Aligned Generation"
  - "Cross-Attention Optimization"
  - "Transformer Diffusion Models"
  - "Training-Free"
date: 2026-05-08
content_hash: b2ca7bd1a9f6a51c
---

# Towards Transformer-Based Aligned Generation with Self-Coherence Guidance

**Conference**: CVPR 2025  
**arXiv**: [2503.17675](https://arxiv.org/abs/2503.17675)  
**Code**: [Project Page](https://scg-diffusion.github.io/scg-diffusion)  
**Area**: Image Generation  
**Keywords**: Text-to-Image, Aligned Generation, Cross-Attention Optimization, Transformer Diffusion Models, Training-Free

## TL;DR

Proposes Self-Coherence Guidance (SCG), a training-free alignment method tailored for Transformer-structured text-guided diffusion models, which improves attribute binding, fine-grained attribute binding, and style binding by directly optimizing cross-attention maps (rather than latent variables).

## Background & Motivation

Text-guided diffusion models (TGDMs) often fail to generate images that are semantically aligned with complex text prompts, particularly suffering from attribute binding issues (e.g., mismatching colors and objects in "a red bench and a yellow clock"). Existing methods primarily rely on the U-Net architecture, and transferring them directly to Transformer architectures yields poor results:

- **Failure of U-Net Method Transfer**: SOTA U-Net-based methods such as D&B and CONFORM exhibit limited effectiveness and can even degrade generation quality when directly applied to PIXART-α.
- **Different Semantic Distribution in Attention Maps**: The 16×16 cross-attention maps of U-Net possess stronger core semantic information (low entropy), whereas the attention maps across all layers in Transformers share the same resolution and have a more uniform distribution of semantic information.
- **Indirectness of Latent Space Optimization**: U-Net methods transition through loss functions to optimize latent variables for indirectly influencing attention maps; such indirect control becomes even less effective in Transformers.
- **Lack of Research on Fine-Grained and Style Binding**: Existing benchmarks focus solely on coarse-grained attribute binding, lacking evaluations of part-level attributes and style binding.

## Method

### Overall Architecture

SCG is a training-free method that directly modifies cross-attention maps during the inference process of Transformer diffusion models. The core idea is to extract concept masks utilizing the attention maps from the previous denoising step, and then apply these masks to the current step's attention maps, thereby amplifying the attention weights of attribute/style tokens on their corresponding concept regions.

### Key Designs

#### Key Design 1: Self-Coherence Guidance — Direct Optimization of Cross-Attention Maps

**Function**: Dynamically amplifies the attention weights of attribute tokens within the corresponding concept areas, ensuring correct attribute-concept binding.

**Mechanism**: For a concept token $o_i$ and an attribute token $r_i$, a concept mask $M_{t+1}^{o_i}$ is extracted utilizing the cross-attention map from the previous step $t+1$, and the attention map is then directly modified in the current step $t$:

$$(\widehat{A_t})_{p,q} = \begin{cases} c \cdot (A_t)_{p,q} & \text{if } q = r_i \text{ and } M_{t+1}^{o_i}[p] = 1 \\ (A_t)_{p,q} & \text{otherwise} \end{cases}$$

where $c$ is the amplification coefficient (set to 4), $p$ is the spatial position index, and $q$ is the token index.

**Design Motivation**: In the Transformer architecture, all cross-attention maps share the same shape (without downsampling/upsampling), making direct manipulation of attention maps feasible. Compared to U-Net methods that indirectly optimize latent variables via loss functions, directly modifying attention maps is more efficient and provides more precise control.

#### Key Design 2: Mask Extraction Strategy — Clustering for Coarse-Grained, LLM Planning for Fine-Grained

**Function**: Adopts different strategies to extract concept masks from attention maps based on the task type.

**Mechanism**: 
- **Coarse-Grained Attribute Binding & Style Binding**: Applies K-means clustering to the attention maps averaged across all layers, splitting them into two clusters to generate masks.
- **Fine-Grained Attribute Binding**: Since clustering cannot effectively-capture detailed concepts (e.g., the flesh and stem of an apple), an LLM is leveraged to infer part proportions (e.g., "flesh accounts for 80%, stem for 20%"). Corresponding high-attention regions are selected in the attention maps based on these proportions to serve as masks.

**Design Motivation**: Clear boundaries exist between concepts in coarse-grained scenarios, rendering clustering highly effective. Fine-grained scenarios require common-sense reasoning to determine part proportions, where the general knowledge of LLMs perfectly satisfies this requirement.

#### Key Design 3: Cross-Step Self-Coherence — Leveraging Previous Steps for Guidance

**Function**: Leverages the diffusion model's existing capability of "knowing what to draw" to guide "where to draw".

**Mechanism**: Diffusion models can typically generate the parts of individual concepts correctly but fail at the correct attribute assignment. Guidance is performed in a self-coherent manner without external reference images by using the concept attention maps at step $t+1$ (where the model already knows the concept locations) to guide the attribute attention maps at step $t$ (thereby correcting attribute locations).

**Design Motivation**: Distinct from methods like Prompt-to-Prompt that require reference images for guidance, SCG is entirely self-guided, rectifying attribute binding by leveraging the model's own understanding of concept locations during the generation process.

### Loss & Training

No loss function — SCG is an inference-time attention map editing method that does not involve gradient calculation or backpropagation.

## Key Experimental Results

### Performance of Direct Transfer of U-Net Methods to Transformers

| Method | image-text↑ | text-text↑ |
|------|------------|-----------|
| PIXART-α | 0.36 | 0.807 |
| + D&B (Transfer) | 0.35 | 0.807 |
| + CONFORM (Transfer) | 0.36 | 0.814 |

### Text-Text Similarity in Three Binding Tasks

| Method | Coarse-Grained↑ | Fine-Grained↑ | Style↑ |
|------|--------|--------|------|
| D&B (SD) | 0.798 | 0.742 | 0.660 |
| CONFORM (SD) | 0.824 | 0.748 | 0.672 |
| PIXART-α | 0.807 | 0.720 | 0.664 |
| **SCG (Ours)** | **0.854** | **0.765** | **0.698** |

### User Study (Accuracy)

| Method | Coarse-Grained | Fine-Grained | Style |
|------|--------|--------|------|
| PIXART-α | 47.5% | 32.8% | 38.1% |
| D&B (Transfer) | 48.2% | 35.1% | 39.4% |
| **SCG** | **71.3%** | **52.6%** | **58.7%** |

### Key Findings

- Directly transferring SOTA U-Net methods (D&B, CONFORM) to the Transformer architecture provides **almost no improvement**.
- SCG comprehensively outperforms all baselines across three types of binding tasks, including SD-based dedicated U-Net methods.
- Attention entropy analysis confirms that the U-Net 16×16 layers have lower entropy (stronger semantics), whereas the entropy values across Transformer layers are more uniform.
- Directly manipulating attention maps is more effective than optimizing latent variables.
- LLM-assisted proportion inference is crucial for fine-grained attribute binding.

## Highlights & Insights

1. **First systematic analysis of the behavioral differences in attention maps between U-Net and Transformers**, explaining the root cause of method transfer failures.
2. **Direct optimization of attention maps instead of latent variables** — which is more efficient in Transformers since all layers share uniform resolution.
3. **Self-coherence guidance does not require reference images**, enabling self-guidance through the model's internal concept-understanding capabilities.
4. **Integration of LLMs for fine-grained proportion inference** is an ingenious synergy of cross-modal capabilities.

## Limitations & Future Work

- The amplification coefficient $c=4$ is fixed, and different scenarios might require varying values.
- K-means clustering assumes two concepts; scenarios with three or more concepts require adjustments.
- Adding mask extraction and attention formulation at each step increases computational overhead, impacting generation speed.
- Validated only on PIXART-α, leaving applicability to newer architectures like SD3 and FLUX unknown.
- The capability to process compositional scenarios (such as multiple attribute bindings + relational descriptions) remains to be evaluated.

## Related Work & Insights

- **Prompt-to-Prompt**: A classic work for image editing accomplished by editing cross-attention maps.
- **D&B / CONFORM**: SOTA aligned generation methods within the U-Net architecture.
- **PIXART-α**: A high-quality text-to-image model based on DiT.
- **DiT**: The first fully Transformer-based diffusion model.

## Rating

⭐⭐⭐⭐ — Clearly identifies the bottlenecks of transferring U-Net methods to Transformers and proposes a concise and effective solution. The approach of directly optimizing attention maps is both natural and highly efficient within the Transformer architecture. The comprehensive evaluation across three binding tasks and the benchmark construction are valuable contributions. However, verification solely on PIXART-α is somewhat insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[CVPR 2025\] Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects](self-cross_diffusion_guidance_for_text-to-image_synthesis_of_similar_subjects.md)
- [\[CVPR 2025\] Rectified Diffusion Guidance for Conditional Generation](rectified_diffusion_guidance_for_conditional_generation.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[ICCV 2025\] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](../../ICCV2025/image_generation/accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)

</div>

<!-- RELATED:END -->
