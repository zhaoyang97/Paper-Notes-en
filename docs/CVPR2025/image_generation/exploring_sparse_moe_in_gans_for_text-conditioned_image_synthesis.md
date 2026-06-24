---
title: >-
  [Paper Note] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis
description: >-
  [CVPR 2025][Image Generation][Text-to-image synthesis] This paper proposes Aurora, a text-to-image GAN model based on Sparse Mixture of Experts (Sparse MoE). By incorporating multiple expert networks and a text-aware sparse router in the generator to scale up model capacity, Aurora achieves a zero-shot FID of 6.2 on MS COCO at 64×64 resolution while maintaining an inference speed significantly faster than diffusion models.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-to-image synthesis"
  - "GAN"
  - "Sparse Mixture of Experts"
  - "Large-scale training"
  - "Fast inference"
date: 2026-05-08
content_hash: 42fae49d6f64c483
---

# Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2309.03904](https://arxiv.org/abs/2309.03904)  
**Code**: [https://github.com/zhujiapeng/Aurora](https://github.com/zhujiapeng/Aurora)  
**Area**: Diffusion Models  
**Keywords**: Text-to-image synthesis, GAN, Sparse Mixture of Experts, Large-scale training, Fast inference

## TL;DR
This paper proposes Aurora, a text-to-image GAN model based on Sparse Mixture of Experts (Sparse MoE). By incorporating multiple expert networks and a text-aware sparse router in the generator to scale up model capacity, Aurora achieves a zero-shot FID of 6.2 on MS COCO at 64×64 resolution while maintaining an inference speed significantly faster than diffusion models.

## Background & Motivation

**Background**: Text-to-image (T2I) synthesis is currently dominated by diffusion models (e.g., Stable Diffusion, DALL-E, etc.). Although GANs enjoy extremely rapid inference speeds, interpretable latent spaces, and flexible architectural extensibility (such as 3D-aware generation), they have been progressively marginalized in open-vocabulary T2I tasks.

**Limitations of Prior Work**: The core limitation of GANs lies in the difficulty of scaling model size. Diffusion models can naturally utilize large models through their iterative denoising processes, whereas the generator of a GAN is a feedforward network. Directly increasing layer depth or width introduces serious GPU memory bottlenecks and training instability issues. Furthermore, the community lacks open-source, large-scale GAN T2I models, which severely hinders the development of GAN-related research in T2I synthesis.

**Key Challenge**: GANs require larger model capacities to handle the diverse content of open vocabularies, but directly expanding the parameter size of the feedforward network is computationally infeasible.

**Goal**: To identify a method that effectively scales the capacity of GAN generators under limited computational resources, rendering them capable of large-scale open-vocabulary T2I synthesis.

**Key Insight**: Sparsely-activated Mixture of Experts (Sparse MoE) has demonstrated in the NLP field that ultra-large models can be trained with limited computational resources—only a small subset of experts is activated per forward pass, meaning model capacity grows with the number of experts while computational cost remains almost constant.

**Core Idea**: Integrate Sparse MoE into the FFN layers of the GAN generator. Multiple experts are used to process different feature points, and an adaptive sparse router that fuses text conditions and sampling randomness is designed to route features to these experts.

## Method

### Overall Architecture
The generator of Aurora is built upon a GAN framework. It takes a global latent code $\mathbf{z} \in \mathbb{R}^{512}$ and a text description $\mathbf{c}$ as inputs, and outputs the synthesized image. The generator is formed by stacking multiple generation units of progressively increasing resolution (starting with progressive training from 4×4). Each unit contains convolutional blocks and attention blocks, the latter of which adopt the Sparse MoE mechanism to expand capacity. The discriminator directly inherits the architecture of GigaGAN.

### Key Designs

1. **Text-conditioned Sampling**:

    - Function: Inject text information into the latent space to generate a text-aware global latent code.
    - Mechanism: A CLIP ViT-L/14 is utilized as the text encoder to extract the token sequence $\mathbf{t}_{seq}$ and the global token $\mathbf{t}_g$. The global token $\mathbf{t}_g$ is concatenated with the sampled latent code $\mathbf{z}$ and fed into an MLP mapping network, yielding the disentangled latent code $\mathbf{w} = \text{MLP}(\text{concat}(\mathbf{z}, \mathbf{t}_g))$. The frozen CLIP encoder is stacked with learnable layers to adapt to the T2I task.
    - Design Motivation: Inheriting the mapping network design from StyleGAN yields a more disentangled latent space, while integrating the global text semantics into the sampling process.

2. **Sparse MoE Attention Block**:

    - Function: Scale up the generator capacity without significantly increasing the computational cost.
    - Mechanism: The attention block in each generation unit comprises a self-attention layer, a cross-attention layer (fusing $\mathbf{t}_{seq}$), and an FFN. The key innovation is replacing the FFN with $N$ experts $\{\text{FFN}_j\}_{j=1}^N$, and employing a sparse router to select the most suitable expert for each feature point: $j = \text{Router}(\mathbf{f}_{ca}^{(k)}, \mathbf{w})$. The router dynamically considers both the input features and the text-conditioned global latent code $\mathbf{w}$, enabling routing decisions to be aware of both text semantics and sampling randomness. Visualization demonstrates that pixels with similar visual concepts tend to be routed to the same expert.
    - Design Motivation: Unlike existing Sparse MoE designs that route solely based on input features, the T2I task requires the router to understand "what content is being generated," which necessitates the integration of text conditioning.

3. **Convolution Block and Feature Modulation**:

    - Function: Process features using modulated convolutions in each generation unit.
    - Mechanism: Two Modulated Transition Modules (MTMs) are paired with skip connections: $\mathbf{f}_{conv} = \text{MTM}(\text{MTM}(\mathbf{f}_{in}, \mathbf{w}), \mathbf{w}) + \mathbf{f}_{in}$. Deformable operations with learnable offsets are introduced at low resolutions (≤16×16), and all convolutions adopt sample-adaptive kernel selection.
    - Design Motivation: Inherit the modulated convolution design proven effective in the StyleGAN family, injecting global text semantics via $\mathbf{w}$.

### Loss & Training
Four loss functions are employed: (1) adversarial loss (logistic non-saturating + R1 regularization); (2) match-aware loss (the discriminator rejects mismatched text-image pairs); (3) multi-level CLIP loss (encouraging text-image alignment across all resolutions); (4) MoE balancing loss (preventing some experts from never being activated). The training strategy adopts progressive training (scaling up progressively from 4×4 to 64×64), introducing "reference FID" as an automatic metric to determine when to switch to the next resolution phase—pre-calculating the FID between two sets of real images as a theoretical lower bound, and transitioning to the next phase once the generator's FID outperforms this value. The model is trained on 256 A100 GPUs for one week.

## Key Experimental Results

### Main Results

| Method | Type | FID 10K (Train Set) | Zero-Shot FID 30K (COCO) | Params |
|------|------|-----------------|-------------------------|--------|
| eDiff-I | Diffusion | - | 7.60 | 9.1B |
| BSD (Stable Diffusion) | Diffusion | - | 8.40 | 0.94B |
| StyleGAN-T | GAN | - | 7.30 | 1.02B |
| GigaGAN | GAN | 9.18 | - | 0.65B |
| **Ours (Aurora)** | GAN | **8.28** | **6.45** | 1.16B |

All evaluations are conducted at 64×64 resolution, with Aurora achieving the best results in both domains.

### Ablation Study

| Analytical Dimension | Key Findings |
|---------|-------------|
| Routing Visualization | Pixels with similar visual concepts are assigned to the same expert, holding true across all resolutions. |
| Text Interpolation | Interpolation between two text prompts exhibits smooth semantic transition, maintaining semantic continuity. |
| Latent Code Interpolation | Interpolation in latent space ($\mathcal{Z}$ or $\mathcal{W}$) is not as smooth as expected, behaving more like random sampling. |
| Training Stability | Even with adversarial training and Sparse MoE combined, no instability occurred throughout the entire training process. |

### Key Findings
- The training stability of GANs is remarkably improved after introducing text conditions, without experiencing the mode collapse issues typical of traditional GANs.
- Text prompt interpolation shows more semantic continuity than latent code interpolation, indicating that text token sequences might dominate over global latent codes in T2I GAN generation.
- Routing visualization confirms that the sparse router can automatically cluster based on visual concepts, validating the design of the text-aware routing.

## Highlights & Insights
- The text-aware router is the most elegant design in this paper—integrating $\mathbf{w}$ (composed of text and randomness) into the routing decision allows different text contents and random samples to obtain distinct expert allocation patterns. This is better suited for generation tasks than NLP routers which rely solely on input tokens.
- Progressive training combined with the automatic scheduling of reference FID is a practical engineering contribution, reducing the need for manual hyperparameter tuning.
- Open-sourcing a large-scale GAN T2I model carries significant inherent value—it provides a usable foundational model for utilizing the unique strengths of GANs, such as latent space editing and 3D-aware generation.

## Limitations & Future Work
- The model currently only supports 64×64 resolution, requiring an additional super-resolution model (4× upsampling). End-to-end high-resolution generation has yet to be realized.
- The discontinuity in latent space interpolation remains an unresolved mystery, which might be related to the dominance of text tokens in cross-attention.
- Compared to concurrent diffusion models, a gap in generation quality still exists (in terms of visual diversity and fidelity).
- Future exploration directions: larger-scale training, direct high-resolution outputs, and introducing MoE into the discriminator.

## Related Work & Insights
- **vs StyleGAN-T**: StyleGAN-T also tackles GAN T2I but does not resolve the model capacity scaling problem. Aurora achieves superior FID while scaling up parameters through Sparse MoE.
- **vs GigaGAN**: GigaGAN is the closest work (both focusing on large-scale GAN T2I). Aurora surpasses it in zero-shot FID and offers a fully open-source release.
- **vs Switch Transformer**: Borrowed the sparse routing concepts from Switch Transformer but innovated by incorporating text-condition awareness, rather than directly copying the architecture.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Sparse MoE into the GAN generator and designing a text-aware router is a refreshing and synergistic architectural innovation.
- Experimental Thoroughness: ⭐⭐⭐ Quantitative experiments only focus on the 64×64 resolution, and ablation tables (such as ablating the number of experts or routing strategies) are somewhat lacking.
- Writing Quality: ⭐⭐⭐⭐ Written in a technical report style; descriptions are clear but slightly engineering-heavy, though the discussions offer insightful perspectives on community value.
- Value: ⭐⭐⭐⭐ Open-sourcing a large-scale GAN T2I model is of high importance to the community, although the 64×64 resolution limits its immediate practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)
- [\[CVPR 2025\] Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects](self-cross_diffusion_guidance_for_text-to-image_synthesis_of_similar_subjects.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](../../ICCV2025/image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)

</div>

<!-- RELATED:END -->
