---
title: >-
  [Paper Note] PersonaBooth: Personalized Text-to-Motion Generation
description: >-
  [CVPR 2025][Human Understanding][Motion Personalization] This work defines a new task of Motion Personalization and proposes PersonaBooth, a multimodal fine-tuning method along with PerMo, a large-scale motion personalization dataset. By employing persona tokens, contrastive learning, and context-aware fusion, the method captures an individual's unique motion style from a few reference motions and generates text-driven personalized motions.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Motion Personalization"
  - "Motion Diffusion Model"
  - "Multimodal Fine-Tuning"
  - "Contrastive Learning"
  - "Context-Aware Fusion"
date: 2026-05-08
content_hash: 972c89ac86aaafdf
---

# PersonaBooth: Personalized Text-to-Motion Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.07390](https://arxiv.org/abs/2503.07390)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Motion Personalization, Motion Diffusion Model, Multimodal Fine-Tuning, Contrastive Learning, Context-Aware Fusion

## TL;DR

This work defines a new task of Motion Personalization and proposes PersonaBooth, a multimodal fine-tuning method along with PerMo, a large-scale motion personalization dataset. By employing persona tokens, contrastive learning, and context-aware fusion, the method captures an individual's unique motion style from a few reference motions and generates text-driven personalized motions.

## Background & Motivation

Motion Style Transfer (MST) utilizes only a single source motion to transfer style, which limits the diversity of the generated motions. Existing methods face the following limitations:

1. **Lack of Personalized Concept**: Existing MST methods focus on abstract style transfer, neglecting the "persona" that reflects an individual's unique expression (e.g., different people express a "playful" style in distinct ways).
2. **Absence of Persona Information in Pre-trained Data**: HumanML3D contains only 15K samples with almost no persona-related data, resulting in a significant distribution gap with persona-focused fine-tuning data.
3. **Content Interference**: Extracting a consistent persona from motions with different contents (e.g., jumping vs. walking) is extremely challenging.
4. **Visual-Only Adaptation in Existing Methods**: Current diffusion-based MST methods do not adapt the text channel, which limits the integration of new persona information.

The authors define a new task called Motion Personalization: given a few reference motions containing a specific persona, the goal is to generate text-driven personalized motions.

## Method

### Overall Architecture

PersonaBooth conducts multimodal fine-tuning on a pre-trained MDM (Motion Diffusion Model). The Persona Extractor extracts visual persona features $V^*$ and a persona token $P^*$ from input motions. $V^*$ is fed into the adaptation layers of the diffusion model, while $P^*$ is integrated into the Personalized Text Encoder to generate personalized text features $T^*$, which together condition the diffusion generation process.

### Key Designs

**1. Persona Extractor and Persona Cohesion Loss**

- **Function**: Extracts content-agnostic persona features from the input motions.
- **Mechanism**: Pre-trained TMR motion-clip models are used to extract general motion features, followed by an additional transformer $\mathcal{E}_P$ to extract persona features $V^*$. $P^* = \text{MLP}(V^*[0])$ serves as the persona token in the text space. Supervised contrastive learning $L_{pc}$ is leveraged to pull features of the same persona with different contents closer, while pushing features of different personas further apart.
- **Design Motivation**: A person's persona manifests differently across various actions (e.g., an elegant ballet dancer's style is expressed in the feet during walking but in the hands during waving). Contrastive learning helps extract consistent persona features across different motion contents.

**2. Dual-Channel Text and Visual Adaptation**

- **Function**: Integrates persona information simultaneously into both the text and visual modalities.
- **Mechanism**: For text adaptation, $[P^*]$ in "$P^*$ person is dancing" is replaced with the persona token, and zero-initialized gated adaptation is used to merge the original and personalized text embeddings: $T^* = \mathcal{X}_{clip}(T_{in}) + s_t \cdot \tanh(\gamma_t) \cdot \mathcal{X}_{clip}(\tilde{T}_{in}, P^*)$. For visual adaptation, an adaptive self-attention layer is inserted into each transformer layer of the diffusion model to inject $V^*$.
- **Design Motivation**: Visual-only adaptation restricts the capacity to integrate personas (since text descriptions in HumanML3D lack personality modifiers). The dual-channel design ensures the integrated delivery of persona information.

**3. Context-Aware Fusion (CAF)**

- **Function**: Weighted fusion of persona features from multiple input motions during inference based on their relevance to the text prompt.
- **Mechanism**: The motion-clip model calculates the cosine similarity $S_i$ between each input motion and the text prompt, selects the top-k most relevant motions, and merges them using softmax weight fusion: $V^* = \sum_i w_i V_i^*$, $P^* = \sum_i w_i P_i^*$.
- **Design Motivation**: A simple average fusion causes motion blending issues (e.g., an unnatural posture where one hand is bent while the other hangs down). CAF selects the most relevant persona cues based on spatial-semantic context.

### Loss & Training

$$L = L_D + \lambda L_{pc}$$

Where $L_D$ represents the diffusion reconstruction loss combined with geometric losses, $L_{pc}$ is the persona cohesion contrastive loss, and $\lambda = 10^{-2}$. Training adopts Classifier-Free Guidance, where the persona conditioning is randomly dropped with a 10% probability.

## Key Experimental Results

### Ablation Study: Component Effectiveness (PerMo Dataset)

| Method | FID↓ | R-Prec Top-1↑ | PRA avg.↑ | Diversity↑ |
|------|------|-------------|-----------|-----------|
| Baseline (Visual adaptation only) | 7.45 | 0.06 | 17.99 | 7.48 |
| + $P^*$ (Text adaptation) | 5.06 | 0.05 | 18.26 | 8.01 |
| + $L_{pc}$ | **3.18** | **0.15** | 18.05 | 7.74 |
| MI + Mean fusion | 3.52 | 0.19 | **19.24** | 7.88 |
| MI + CAF | **2.95** | **0.19** | 18.13 | **8.12** |

### PerMo Dataset Comparison

| Dataset | Actors | Styles | Contents | Clips | Mesh | Text |
|--------|--------|--------|--------|--------|------|------|
| Xia | - | 8 | 6 | 572 | ✗ | ✗ |
| 100Style | 1 | 100 | 8 | 810 | ✗ | ✗ |
| **PerMo** | **5** | **34** | **10** | **6,610** | **✓** | **✓** |

### Key Findings

- Adding text adaptation via the persona token reduces the FID from 7.45 to 5.06 (-32%).
- $L_{pc}$ contrastive learning further decreases the FID to 3.18 while significantly improving R-Precision (from 0.06 to 0.15), validating the importance of decoupling persona from motion content.
- CAF reduces the FID (from 3.52 to 2.95) and boosts Diversity (from 7.88 to 8.12) compared to a simple average fusion.

## Highlights & Insights

1. **Visionary New Task Definition**: Motion Personalization is closer to real-world demands than Style Transfer, holding direct applications in digital humans and metaverse scenarios.
2. **Comprehensive Multimodal Fine-Tuning Strategy**: Dual-channel text and visual adaptation with zero-initialized gating effectively avoids catastrophic forgetting.
3. **Substantial Contribution of the PerMo Dataset**: The first large-scale, multi-actor, multi-style persona motion dataset that includes both meshes and texts.

## Limitations & Future Work

- Currently, it only supports the SMPL-H format, failing to capture facial expressions and finger details.
- The diversity of having only 5 actors might not be sufficient to represent general populations.
- The definition of "persona" remains somewhat ambiguous, with potential overlaps across different styles.
- Future research can extend this work to video-driven motion personalization.

## Related Work & Insights

- **MDM**: The base motion diffusion model on which PersonaBooth is fine-tuned.
- **MoMo**: Zero-shot motion style transfer, which tends to fail when source and target contents differ.
- **SMooDi**: A diffusion fine-tuning method; however, it suffers from slow retraining and forgetting issues.
- **InstantBooth**: Simple averaging strategies commonly used in image-domain personalization are not readily applicable to motions.

## Rating

⭐⭐⭐⭐ — Novel task definition, substantial dataset contribution, and well-designed method. The combination of multimodal fine-tuning and contrastive learning successfully addresses key challenges, with ablation studies thoroughly verifying the contribution of each component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[ICLR 2026\] Motion-Aligned Word Embeddings for Text-to-Motion Generation](../../ICLR2026/human_understanding/motion-aligned_word_embeddings_for_text-to-motion_generation.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)
- [\[CVPR 2025\] Exploring Timeline Control for Facial Motion Generation](exploring_timeline_control_for_facial_motion_generation.md)

</div>

<!-- RELATED:END -->
