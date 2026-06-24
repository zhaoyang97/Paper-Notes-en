---
title: >-
  [Paper Note] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields
description: >-
  [CVPR 2025][Image Generation][4D Generation] DNF proposes a 4D neural field representation based on dictionary learning. It achieves decoupled and compact encoding of shape and motion through an SVD decomposition-compression-expansion MLP parameter dictionary. Combined with a Transformer diffusion model, it enables unconditional 4D deforming object generation, achieving state-of-the-art (SOTA) performance on DeformingThings4D.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "4D Generation"
  - "Dictionary Learning"
  - "Neural Fields"
  - "Deformation Modeling"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 50e025ae04b3d5cb
---

# DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields

**Conference**: CVPR 2025  
**arXiv**: [2412.05161](https://arxiv.org/abs/2412.05161)  
**Code**: [https://xzhang-t.github.io/project/DNF](https://xzhang-t.github.io/project/DNF)  
**Area**: Image Generation / 3D Vision  
**Keywords**: 4D Generation, Dictionary Learning, Neural Fields, Deformation Modeling, Diffusion Models

## TL;DR

DNF proposes a 4D neural field representation based on dictionary learning. It achieves decoupled and compact encoding of shape and motion through an SVD decomposition-compression-expansion MLP parameter dictionary. Combined with a Transformer diffusion model, it enables unconditional 4D deforming object generation, achieving state-of-the-art (SOTA) performance on DeformingThings4D.

## Background & Motivation

**Background**: Significant progress has been made in 3D generative models, but 4D (3D + time/motion) generation remains highly challenging. Real-world objects are dynamic, requiring simultaneous modeling of shape and motion to support applications like content creation, mixed reality, and simulation.

**Limitations of Prior Work**: (1) Template-based parametric models (e.g., SMPL for humans), while robust, are limited to specific categories and cannot generalize to general deforming objects; (2) coordinate MLPs optimized for single objects can reconstruct high-precision details, but the weight spaces across different objects lack a shared structure, which hinders learning by generative models; (3) multi-object learning based on global latent codes possesses a shared structure but tends to lose high-fidelity details of individual objects; (4) HyperDiffusion directly performs diffusion in the MLP weight space, but its generation quality is limited due to the lack of a shared structure; Motion2VecSets uses vector sets but struggles to balance compression and fidelity.

**Key Challenge**: A 4D representation must simultaneously satisfy three goals—fine-grained detail (high fidelity), continuity (supporting interpolation and generation), and compactness (compressed representation)—which are difficult to reconcile. Global latent codes offer continuity but lack details, whereas per-object optimization provides details but lacks continuity.

**Goal**: Design a 4D representation that balances shape fidelity, representation space continuity, and encoding compactness to support efficient unconditional 4D diffusion-based generation.

**Key Insight**: Treat MLP weights as linear combinations of a dictionary—by decomposing globally optimized MLP parameters via SVD into a shared dictionary (singular vector matrices) and per-instance coefficients (singular value vectors). Freezing the dictionary and only fine-tuning the coefficients maintains the continuity of the shared structure while allowing per-instance detail adaptation.

**Core Idea**: Utilize SVD-based dictionary learning to decouple the 4D representation into a shared dictionary, per-instance latent codes, and coefficient vectors, and then apply a Transformer diffusion model to perform unconditional 4D generation over this compact representation space.

## Method

### Overall Architecture

The method consists of two main stages: (1) 4D representation learning—pre-training shape and motion MLPs (to learn a global latent space), and then establishing dictionaries via SVD to fine-tune coefficient vectors per instance; (2) diffusion generation—using a Transformer diffusion model to unconditionally generate shape and motion representations independently. Ultimately, each 4D sequence is compactly represented as a list of latent codes and coefficient vectors.

### Key Designs

1. **Decoupled Shape-Motion Neural Field**:

    - Function: Decouple 4D deforming objects into two independent latent spaces: static shape and time-varying motion.
    - Mechanism: The shape MLP $f_{\Theta_s}(s_i, x)$ takes a shape latent code $s_i$ and spatial coordinates as inputs to predict SDF values. The motion MLP $f_{\Theta_m}(s_i, m_i^t, x)$ takes the shape latent code, a motion latent code, and coordinates as inputs to predict the 3D flow from the initial frame to the $t$-th frame. It does not assume a canonical pose and directly uses the first frame of the sequence as the canonical shape.
    - Design Motivation: The shape of a deforming object remains constant throughout the sequence, whereas only the motion changes over time. Decoupling them allows generating shape and motion independently, or producing new motions for existing shapes.

2. **SVD Dictionary Learning and Compression-Expansion**:

    - Function: Achieve per-instance high fidelity while maintaining continuity in the weight space.
    - Mechanism: Perform layer-wise SVD on pre-trained MLP weight matrices: $W_\ell = U_\ell \Sigma_\ell V_\ell^T$. The singular vector matrices $U, V$ are treated as a shared dictionary (frozen), and the singular values $\sigma$ are treated as per-instance coefficients (fine-tuned). To improve efficiency and expressiveness, the method first performs **compression**—removing dictionary elements corresponding to small singular values (keeping the top $k$)—and then **expansion**—adding low-rank residual matrices $\Delta\Theta = U_{res} \Sigma_{res} V_{res}^T$, where the singular vectors of the residuals are also added to the dictionary. An orthogonalization loss $\mathcal{L}_{orth}$ constrains the singular vectors of the residual matrices to remain orthogonal, ensuring no dictionary redundancy.
    - Design Motivation: Directly fine-tuning all MLP parameters on all samples destroys the continuity of the weight space (weights of different objects share no structure), making it impossible for generative models to learn a meaningful distribution. The dictionary method limits variation to the coefficients, ensuring continuity, while compression removes redundancy and expansion compensates for details.

3. **Transformer Diffusion Model (Shape + Motion)**:

    - Function: Perform unconditional 4D generation in the dictionary representation space.
    - Mechanism: The shape is represented as $L+1$ tokens (1 latent code + $L$ layer coefficient vectors), which are directly fed into the Transformer decoder. For motion generation, training is performed on sub-sequences (6-frame windows) conditioned on the corresponding shape's latent code (injected via cross-attention), with temporal self-attention added to ensure inter-frame coherence. During inference, longer sequences are generated via sliding-window extrapolation (using the last 2 frames as context to generate the next 4 frames).
    - Design Motivation: The dictionary representation naturally forms token sequences suitable for Transformer processing. Conditioning motion on shape ensures that the generated motion matches the geometry. Sliding-window extrapolation allows generating sequences beyond the training length.

### Loss & Training

Shape reconstruction uses a clamped L1 loss (focusing on regions near the surface), and motion uses an L1 flow loss. During dictionary fine-tuning, shape is trained for 1000 epochs and motion for 400 epochs. The diffusion model uses a simple denoising objective $\mathcal{L}_{simple} = E[||\theta_s - \hat{\theta_s}||^2_2]$. Training takes approximately one day for 1000 epochs on 2x RTX A6000 GPUs. The shape MLP consists of 8 layers with 512 dimensions, while the motion MLP contains 8 layers with 1024 dimensions, with a latent code dimension of 384.

## Key Experimental Results

### Main Results

| Method | MMD ↓ | COV(%) ↑ | 1-NNA(%) ↓ |
|------|------|---------|-----------|
| HyperDiffusion | 16.0 | 45.9 | 63.5 |
| Motion2VecSets | 18.7 | 48.1 | 68.2 |
| **DNF (Ours)** | **15.3** | **54.1** | **58.2** |

### Ablation Study

| Configuration | Description |
|------|------|
| No dictionary (global latent code only) | Shape details are severely lost, and generated object surfaces are overly smooth |
| With dictionary, no compression-expansion | Redundancy in the dictionary leads to low efficiency, with limited improvement in detail |
| With dictionary + compression (no expansion) | More efficient after removing redundancy, but expressiveness is limited |
| Full DNF (compression + expansion + orthogonalization) | Best balance between fidelity and compression rate |

### Key Findings

- DNF significantly outperforms the baselines across all three generation quality metrics: lowering MMD by 4.6% (compared to HyperDiffusion), improving COV by 17.9%, and improving 1-NNA by 8.3%.
- HyperDiffusion performs diffusion directly in the MLP weight space, limiting its generation quality due to the lack of a shared structure. DNF's dictionary approach greatly enhances performance by introducing a shared structure.
- Although Motion2VecSets also uses sets of latent vectors, it underperforms compared to DNF in the unconditional generation setting.
- The compression-expansion strategy of the dictionary is crucial: compression removes redundancy, expansion replenishes details, and both are indispensable.
- The method can generate novel motions for unseen categories during training, demonstrating the generalization capability enabled by the shape-motion decoupling.
- Animation sequences longer than the training window can be generated through diffusion-based extrapolation.

## Highlights & Insights

- The application of **SVD dictionary learning** in neural field representation is highly elegant. Reinterpreting the SVD of MLP weights as a dictionary decomposition (singular vectors = dictionary elements, singular values = coefficients) provides a novel interpretation and extension of low-rank methods like LoRA. This concept can be generalized to any scenario requiring generation over a set of MLP parameters.
- The design of the **compression-expansion strategy** demonstrates strong engineering intuition: streamlining before adding new elements is more effective than using full SVD directly or learning a new dictionary from scratch. The orthogonalization loss ensures dictionary quality.
- **Motion extrapolation** is achieved via sliding windows, which is simple but practical, allowing training on short sequences while generating sequences of arbitrary length during inference.

## Limitations & Future Work

- The evaluation is limited to DeformingThings4D, which mainly contains animal-like deforming objects; generalization to other categories (e.g., clothes, fluids) remains unclear.
- It only supports unconditional generation and lacks conditioning on text or images, which limits its application scenarios.
- It uses SDF for shape representation, which restricts its capability in modeling topological changes.
- The quality of long sequences may degrade as the generated motion length is bounded by the cumulative error of the sliding-window strategy.

## Related Work & Insights

- **vs HyperDiffusion**: HyperDiffusion directly performs diffusion on per-object optimized MLP weights, and the lack of a shared structure yields poor generation quality. DNF introduces a shared structure through its dictionary, presenting a major improvement over HyperDiffusion.
- **vs Motion2VecSets**: Uses latent vector sets to represent 4D but falls short of DNF's dictionary approach in balancing fidelity and compression.
- **vs NPMs**: DNF's shape-motion decoupling follows the philosophy of NPMs but significantly improves detail quality via dictionary learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Elegant application of SVD dictionary learning to 4D neural fields, with a creative compression-expansion strategy.
- Experimental Thoroughness: ⭐⭐⭐ Only evaluated on a single dataset, with a limited number of baselines and lack of detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear description of methods with complete formula derivations.
- Value: ⭐⭐⭐⭐ Proposes an effective representation learning scheme for 4D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [\[CVPR 2025\] AvatarArtist: Open-Domain 4D Avatarization](avatarartist_open-domain_4d_avatarization.md)
- [\[CVPR 2025\] Generation of Maximal Snake Polyominoes Using a Deep Neural Network](generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)
- [\[NeurIPS 2025\] Neural Entropy](../../NeurIPS2025/image_generation/neural_entropy.md)

</div>

<!-- RELATED:END -->
