---
title: >-
  [Paper Note] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending
description: >-
  [CVPR 2025][Image Generation][Multi-concept customized generation] LaTexBlend achieves high-fidelity, high-efficiency multi-concept customized image generation by representing and blending multiple customized concepts in the latent textual space following the text encoder. It scales linearly in fine-tuning complexity with zero additional inference overhead.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-concept customized generation"
  - "Text-to-Image"
  - "Latent textual space"
  - "Concept blending"
  - "Denoising deviation"
date: 2026-05-08
content_hash: f7883b1140eb1736
---

# LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending

**Conference**: CVPR 2025  
**arXiv**: [2503.06956](https://arxiv.org/abs/2503.06956)  
**Code**: Yes (Project Page)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Multi-concept customized generation, Text-to-Image, Latent textual space, Concept blending, Denoising deviation

## TL;DR
LaTexBlend achieves high-fidelity, high-efficiency multi-concept customized image generation by representing and blending multiple customized concepts in the latent textual space following the text encoder. It scales linearly in fine-tuning complexity with zero additional inference overhead.

## Background & Motivation

1. **Background**: Customized text-to-image (T2I) generation has evolved from single-concept to multi-concept generation, where users aim to integrate multiple personalized subjects into a single scene. Existing methods primarily rely on multi-concept joint training, data augmentation, or merging multiple single-concept diffusion branches.

2. **Limitations of Prior Work**: Existing methods are either computationally inefficient (joint training/data augmentation leads to exponential growth in fine-tuning complexity) or compromise image quality (concept omission, concept mixing, and concept distortion). For instance, Custom Diffusion and MuDI require retraining for each specific concept combination, Mix-of-Show demands additional retraining, and OMG incurs substantial inference overhead.

3. **Key Challenge**: The root cause of quality degradation in multi-concept generation is the "denoising deviation". Customized concepts are fine-tuned using only 3~5 reference images, which are typically single-object-centered and lack context diversity. Consequently, customized concepts fail to achieve the generalization capability of generic concepts in the pretrained model. For multi-concept generation, this deviation worsens due to spatial competition and low co-occurrence probability.

4. **Goal**: (1) How to support arbitrary multi-concept combinations without exponentially increasing the fine-tuning overhead? (2) How to mitigate structural degradation and layout confusion in multi-concept generation?

5. **Key Insight**: The authors observe that in the "latent textual space" after the text encoder + linear projection, the K/V features already contain sufficient concept-specific information. By compressing the concept information into a compact representation within this space, concepts can be freely combined without causing interference during the earlier text encoding phase.

6. **Core Idea**: Independently learn a compact representation for each concept in the latent textual space and store it in a concept library. During inference, seamlessly blend multiple concepts through simple feature replacement in this space.

## Method

### Overall Architecture
LaTexBlend consists of two phases: (1) **Single-Concept Fine-tuning**: Each concept is fine-tuned independently to compress the concept information into the latent textual feature $\mathbf{h}_c$, which is then stored in the concept library. (2) **Multi-Concept Inference**: Multiple latent textual features of target concepts are retrieved from the concept library and integrated into the base text stream via feature replacement (Blend) in the latent textual space, generating multi-concept images. This pipeline completely avoids joint training and incurs zero extra inference overhead.

### Key Designs

1. **Auxiliary Base Text Encoding Flow (Base Flow)**:

    - **Function**: Guide the concentration of concept information into the latent textual feature of the target token during fine-tuning.
    - **Mechanism**: In addition to the standard trainable encoding flow $\mathscr{F}_c$, an extra frozen base flow $\mathscr{F}_b$ utilizing the pretrained projection matrices $\{W_{k_o}, W_{v_o}\}$ is introduced. During fine-tuning, $\mathbf{h}_c$ (the concept token feature) replaces the corresponding position in $\mathbf{h}_b$ (the base flow output), after which the replaced feature is used for denoising reconstruction. Since the gradient is only backpropagated to the trainable branch through $\mathbf{h}_c$, the concept information is forced to concentrate into this compact representation.
    - **Design Motivation**: In standard Custom Diffusion, concept information is scattered across all $M$ token features (since all tokens attend to "V*" in the text encoder). By introducing the base flow, only the concept-related token feature $\mathbf{h}_c$ carries the concept information, realizing information compression and enabling $\mathbf{h}_c$ to independently represent the concept.

2. **Prompt Template Pool for Position Invariance**:

    - **Function**: Ensure that the compact concept representation $\mathbf{h}_c$ functions properly when inserted at different positions.
    - **Mechanism**: A template pool containing 7 prompt templates of varying lengths (e.g., "A {}.", "Photo of {}.") is constructed. During fine-tuning, prompt templates are randomly drawn from the pool to construct input prompts for $\mathscr{F}_b$ and $\mathscr{F}_c$, consistently varying the extraction and insertion position of $\mathbf{h}_c$ to eliminate reliance on position encodings.
    - **Design Motivation**: During inference, $\mathbf{h}_c$ can be inserted at any position in the prompt. If the position is fixed during fine-tuning, artifacts like perspective distortion and layout warping can occur during generation.

3. **Blending Guidance**:

    - **Function**: Correct cross-attention maps during multi-concept inference to reduce concept omission and identity mixing.
    - **Mechanism**: Two guidance terms are designed: $g_1$ maximizes the attention map overlap between each concept's identifier token "V*" and its category descriptor "\<noun\>" (enhancing identity binding); $g_2$ minimizes the attention map overlap between different concept tokens (preventing identity mixing). The gradients of the guidance terms are added to the noise estimation: $\hat{\epsilon}'_t = \hat{\epsilon}_t + \lambda \nabla_{z_t}(g_1 + g_2)$, guiding the sampling process towards the desired direction.
    - **Design Motivation**: Pretrained T2I models can generate erroneous cross-attention maps when handling similar subjects or a large number of subjects. Directly blending is sufficient for a small number of concepts ($\le 3$), but guidance significantly improves generation quality as the concept count increases.

### Loss & Training
- Single-concept fine-tuning employs the standard diffusion model reconstruction loss (Eq. 6), using the blended feature $\mathcal{F}(\mathbf{h}_c)$ as the input condition.
- Each concept is fine-tuned independently. Once completed, $\mathbf{h}_c$ is stored in the concept library.
- Multi-concept inference requires no additional training; concepts are directly retrieved from the library and blended.

## Key Experimental Results

### Main Results

| Evaluation Metric | LaTexBlend | Mix-of-Show | Cones 2 | OMG | MuDI |
|----------|-----------|-------------|---------|-----|------|
| Concept Alignment (User Study) | **4.33** | 2.26 | 1.92 | 2.84 | 3.24 |
| Prompt Alignment (User Study) | **4.16** | 2.55 | 3.26 | 3.66 | 2.83 |
| Overall Quality (User Study) | **4.76** | 2.31 | 3.12 | 1.53 | 3.54 |

The user study involves 20 generation cases evaluated by 25 participants. LaTexBlend achieves the highest score across all three metrics. In quantitative experiments, LaTexBlend also significantly outperforms all baselines on $S_{\text{CLIP}}^I$ and $S_{\text{DINO}}$.

### Ablation Study

| Configuration | $S_{\text{CLIP}}^T$ ↑ | $S_{\text{CLIP}}^I$ ↑ | $S_{\text{DINO}}$ ↑ | Description |
|------|---------|---------|---------|------|
| Full model | 0.3684 | **0.8052** | **0.6564** | Full model |
| w/o base flow | 0.3718 | 0.5861 | 0.4337 | Without the base flow, concept fidelity severely degrades |
| w/o prompt variety | 0.3539 | 0.7155 | 0.5648 | Using fixed templates causes artifacts like perspective distortion |

### Key Findings
- **The base flow is the most critical design**: Removing it causes $S_{\text{DINO}}$ to plummet from 0.6564 to 0.4337 (-34%), leading to a complete breakdown in concept fidelity.
- The fine-tuning complexity of LaTexBlend scales **linearly** with the number of concepts, while scaling up concepts during inference incurs **zero additional overhead**.
- Blending guidance offers marginal improvements when the number of concepts is $\le 3$, but yields significant quality improvements as the concept count grows.
- It is compatible with Layout-to-Image models, demonstrating higher subject occurrence and fidelity under spatial layout conditions.

## Highlights & Insights
- **Elegant choice of latent textual space blending**: This space hits the "sweet spot" where information is abundant but blending cost is negligible—deeper than the token embedding layer and shallower than the U-Net interior, preserving sufficient concept information without requiring computationally expensive branch-merging.
- **Elegant information compression strategy via auxiliary flow + gradient isolation**: By freezing the base flow and cutting off its gradient path, the method cleverly "compresses" concept details into the target token's K/V features, creating a compact yet composable concept representation.
- **Denoising deviation analysis** provides a profound understanding of quality degradation in customized generation, which is generalizable to other generative tasks involving few-shot fine-tuning.

## Limitations & Future Work
- Constrained by the pretrained T2I model's inherent capacity to handle complex scenes and long prompts, further scaling of the concept count is bounded by the underlying base model.
- Each concept in the library requires independent fine-tuning (though scaling is linear); zero-shot or few-shot concept injection remains unexplored.
- Blending guidance requires calculating attention map gradients during inference, translating to a minor inference latency.
- The framework has not been verified on the latest DiT architectures (e.g., Flux, SD3); whether the findings regarding the latent textual space transfer to these new architectures remains to be investigated.

## Related Work & Insights
- **vs Custom Diffusion**: Custom Diffusion fine-tunes token embeddings and projection matrices for customization, but requires joint training for multiple concepts, resulting in exponential complexity. LaTexBlend introduces information compression and independent fine-tuning strategies to completely resolve the scalability issue.
- **vs OMG**: OMG also supports scalable fine-tuning, but requires running independent branches for each concept during inference and then merging them, making inference overhead scale linearly with the concept count. LaTexBlend blends in the latent textual space, ensuring zero extra inference overhead.
- **vs Mix-of-Show**: Mix-of-Show merges multiple LoRAs, but any new combination of concepts requires additional retraining. LaTexBlend enables arbitrary combination without training once the concept library is established.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of blending in the latent textual space is novel, and the design of the information compression strategy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Qualitative and quantitative comparisons are comprehensive, supported by user studies and computational cost analysis, though lacking tests on a larger-scale concept library.
- Writing Quality: ⭐⭐⭐⭐ The analysis of motivation is clear, and the visual analysis of denoising deviation is highly intuitive and convincing.
- Value: ⭐⭐⭐⭐ Successfully addresses the core challenge of multi-concept customized generation, demonstrating high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] MARBLE: Material Recomposition and Blending in CLIP-Space](marble_material_recomposition_and_blending_in_clip-space.md)
- [\[ICLR 2026\] MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](../../ICLR2026/image_generation/mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)

</div>

<!-- RELATED:END -->
