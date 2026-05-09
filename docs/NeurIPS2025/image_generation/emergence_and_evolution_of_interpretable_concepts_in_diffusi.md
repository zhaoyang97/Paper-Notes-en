---
title: >-
  [Paper Note] Emergence and Evolution of Interpretable Concepts in Diffusion Models
description: >-
  [Image Generation] This work is the first to systematically apply Sparse Autoencoders (SAEs) to multi-step diffusion models (Stable Diffusion v1.4), revealing that image composition emerges as early as the first reverse diffusion step while stylistic concepts form during intermediate stages. Based on these findings, the paper proposes temporally adaptive causal intervention techniques.
tags:
  - Image Generation
date: 2026-05-08
content_hash: effe0d7302c93aa0
---

# Emergence and Evolution of Interpretable Concepts in Diffusion Models

## Basic Information
- **arXiv**: 2504.15473
- **Conference**: NeurIPS 2025
- **Authors**: Berk Tinaz, Zalan Fabian, Mahdi Soltanolkotabi (USC)
- **Institution**: University of Southern California
- **Code**: Not open-sourced

## TL;DR
This work is the first to systematically apply Sparse Autoencoders (SAEs) to multi-step diffusion models (Stable Diffusion v1.4), revealing that image composition emerges as early as the first reverse diffusion step while stylistic concepts form during intermediate stages. Based on these findings, the paper proposes temporally adaptive causal intervention techniques.

## Background & Motivation
Despite the remarkable success of diffusion models in image generation, their internal mechanisms remain largely opaque. SAEs have proven effective in mechanistic interpretability of LLMs (e.g., Anthropic's analysis of Claude), yet they have not been applied to understanding how visual representations evolve over time during the multi-step generation process of diffusion models. Prior work (Surkov et al.) analyzed only single-step distilled diffusion models (SDXL Turbo), making it impossible to capture the temporal evolution of features — which is precisely the most distinctive characteristic of diffusion models.

## Core Problem
1. How much information do image representations already encode in the early stages of generation?
2. How do visual representations evolve across different stages of the diffusion process?
3. Can the discovered interpretable concepts be used to causally guide the generation process?
4. How does the effectiveness of interventions vary with diffusion timestep?

## Method

### 1. SAE Architecture
A $k$-sparse autoencoder (TopK activation) is trained on the residual updates of the U-Net in SD v1.4:
- Encoder: $\mathbf{z} = \text{TopK}(\text{ReLU}(\mathbf{W}_{enc}(\mathbf{x} - \mathbf{b})))$
- Decoder: $\hat{\mathbf{x}} = \mathbf{W}_{dec}\mathbf{z} + \mathbf{b}$
- Concept vectors: $\mathbf{f}_i = \mathbf{W}_{dec}[:, i]$ (columns of the decoder weight matrix)
- Expansion ratio $n_f = 4d = 5120$ ($d = 1280$)

### 2. Temporally Aware Activation Collection
Independent SAEs are trained for each combination of 3 diffusion stages ($t \in \{0.0, 0.5, 1.0\}$) × 3 U-Net blocks (down_block, mid_block, up_block) × 2 conditioning types (cond/uncond). Activations are collected from the residual updates $\Delta_{\ell,t}$ of cross-attention transformer blocks.

### 3. Concept Dictionary Construction (Vision-only Pipeline)
A key novelty is the use of a purely vision-based pipeline for concept annotation, without relying on LLMs:
- RAM (image tagging) → Grounding DINO (open-set detection) → SAM (segmentation)
- IoU between SAE activations and segmentation masks is computed; labels are assigned to concept IDs (CIDs) when IoU exceeds 0.5
- Each concept is represented by the mean Word2Vec embedding of the corresponding object name

### 4. Composition Prediction
A "concept map" is constructed using the concept dictionary: each spatial position is mapped to its top activated concept → Word2Vec embedding → cosine similarity with target object → segmentation prediction.

### 5. Causal Intervention Techniques
**Spatially targeted intervention (composition control)**:
$$\tilde{\Delta}_{\ell,t}[i,j] = \begin{cases} \Delta_{\ell,t}[i,j] + \beta \sum_{c \in C_o} \mathbf{f}_c & \text{if } (i,j) \in S \\ \Delta_{\ell,t}[i,j] - \sum_{c \in C_o} \mathbf{f}_c & \text{otherwise} \end{cases}$$
Activations are manipulated directly rather than through encode-then-decode, avoiding reconstruction error.

**Global intervention (style control)**: $\tilde{\Delta}_{\ell,t}[i,j] = \Delta_{\ell,t}[i,j] + \beta \mathbf{f}_c$

Both interventions incorporate adaptive $\beta$ normalization to stabilize effects across different objects and styles.

## Key Experimental Results

### Emergence Timing of Composition
- $t=1.0$ (first step): mid_block IoU ≈ 0.26 — **scene layout is already predictable**, even though the model output is still pure noise
- $t=0.5$ (middle stage): IoU saturates — composition is largely fixed
- $t=0.0$ (final step): highest IoU, but constrained by annotation pipeline accuracy
- up_block yields the most precise segmentation predictions

### Quantitative Metrics for Temporal Concept Evolution

| Timestep | Concept Cohesion↑ | Inter-concept Similarity↓ |
|---|---|---|
| $t=1.0$ | 0.588 | 0.433 |
| $t=0.5$ | 0.627 | 0.378 |
| $t=0.0$ | 0.664 | 0.344 |
→ Concepts become progressively purer and more discriminative as generation proceeds

### Intervention Effectiveness vs. Diffusion Stage

| Stage | Spatial Intervention Success Rate | Global Intervention Success Rate | LPIPS |
|---|---|---|---|
| Early ($t=1.0$) | **80%** | 78% (alters composition, not style) | 0.653 |
| Middle ($t=0.5$) | 23% (fails) | **93%** (+0.021 ΔCLIP) | 0.385 |
| Final ($t=0.0$) | 25% (fails) | 69% (texture only) | 0.114 |

### Summary of Core Findings
1. **Early stage**: Composition is controllable; style is not (global intervention alters composition rather than style)
2. **Middle stage**: Composition is locked; style is controllable (optimal window for style editing)
3. **Late stage**: Only texture details remain modifiable

## Highlights & Insights
1. **Striking finding**: Image composition emerges at the very first reverse diffusion step, when the model output is still noise
2. **Vision-only annotation pipeline**: Avoids LLM biases and scales to large-scale concept discovery
3. **Temporally adaptive intervention**: First systematic characterization of "what to intervene and when"
4. **Theoretical coherence**: The three-stage evolution (composition → style → texture) aligns with denoising autoencoder (DAE) theory
5. **Causal validity of concept vectors**: Intervention experiments establish causality rather than mere correlation

## Limitations & Future Work
1. Validated only on SD v1.4 (U-Net); not extended to DiT architectures (e.g., FLUX)
2. Skip connections in U-Net cause intervention information leakage, requiring large $\beta$ values
3. Concept dictionary quality depends on external detection and segmentation models
4. Independent SAEs are trained per timestep, precluding direct cross-timestep concept comparison
5. Word2Vec embeddings have limited expressive capacity

## Related Work & Insights
- **vs. Surkov et al. (SDXL Turbo SAE)**: Surkov et al. analyze only a single-step distilled model; this paper analyzes temporal evolution across multi-step generation — a critical distinction
- **vs. Cross-attention visualization (DAAM)**: DAAM uses cross-attention for saliency maps; the SAE concepts proposed here are more fine-grained and support causal intervention
- **vs. h-space/Jacobian editing directions**: Kwon et al. and Park et al. identify editing directions in specific layers; the concept dictionary proposed here is more systematic and interpretable
- **vs. Prompt-to-Prompt**: P2P edits by manipulating attention weights, whereas this work discovers concepts in SAE latent space before intervening — the two approaches are complementary

### Connections and Implications
- **Directions for DiT architectures**: The absence of skip connections in DiT may make interventions more effective. Extending SAEs to FLUX/SD3 is a natural next step
- **Connection to Don't Let It Fade (TTA-Diffusion)**: Both works study the temporal dimension of diffusion — TTA identifies update forgetting, while this paper characterizes the timeline of composition emergence. The early fixation of composition explains why TTA's timestep allocation strategy is effective
- **Implications for controllable generation**: The optimal editing window depends on the type of edit (composition vs. style); a uniform "full-trajectory guidance" strategy may be suboptimal

## Rating
- Novelty: ★★★★★ — Applying SAEs to the temporal evolution of diffusion models is an entirely new perspective
- Technical Depth: ★★★★☆ — The method is clean, and the experimental design is carefully crafted
- Experimental Thoroughness: ★★★★☆ — Qualitative and quantitative results are well integrated, though limited to SD v1.4
- Writing Quality: ★★★★★ — Structure is clear and driven by well-defined scientific questions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)

</div>

<!-- RELATED:END -->
