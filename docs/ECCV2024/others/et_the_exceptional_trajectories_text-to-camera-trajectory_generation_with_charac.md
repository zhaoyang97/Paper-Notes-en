---
title: >-
  [Paper Note] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness
description: >-
  [ECCV 2024][Camera Trajectory Generation] Proposes the first **camera-character trajectory dataset, E.T.** (115K samples, 11M frames), extracted from real movies, alongside **Director**, a diffusion-based camera trajectory generation method that generates complex camera trajectories based on text descriptions and character trajectories. Additionally, designs **CLaTr**, a contrastive language-trajectory embedding, for trajectory generation quality evaluation.
tags:
  - "ECCV 2024"
  - "Camera Trajectory Generation"
  - "Diffusion Models"
  - "Cinematography"
  - "Character Awareness"
  - "Contrastive Learning Embedding"
date: 2026-05-08
content_hash: 44a2ae46d4ea43a7
---

# ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness

**Conference**: ECCV 2024  
**arXiv**: [2407.01516](https://arxiv.org/abs/2407.01516)  
**Code**: [Project Page](https://www.lix.polytechnique.fr/vista/projects/2024_et_courant)  
**Area**: Others (Cinematography / Camera Trajectory Generation)  
**Keywords**: Camera Trajectory Generation, Diffusion Models, Cinematography, Character Awareness, Contrastive Learning Embedding

## TL;DR

Proposes the first **camera-character trajectory dataset, E.T.** (115K samples, 11M frames), extracted from real movies, alongside **Director**, a diffusion-based camera trajectory generation method that generates complex camera trajectories based on text descriptions and character trajectories. Additionally, designs **CLaTr**, a contrastive language-trajectory embedding, for trajectory generation quality evaluation.

## Background & Motivation

### Background

In cinematography, camera placement and movement are core elements to convey the director's intent. After a century of practice, the film industry has formed a "cinematic grammar" to guide camera movement, but mastering this craft remains difficult. It is particularly overwhelming for novice users who face hundreds of potential camera motion choices.

### Limitations of Prior Work

**Geometric/Rule-based Methods**: Require hand-crafted geometric models or cost functions for each motion type, failing to creatively blend different motions.

**Example-based Methods**: Require carefully selected reference videos, suffering from poor generalization.

**Reinforcement Learning Methods**: Require environment-specific training and lack generation diversity, making them prone to style collapse.

**CCD (Concurrent Work)**:
   - Simplifies the problem by using a character-centric relative coordinate system, which limits the generation capability.
   - Trained purely on synthetic data, with a limited vocabulary of only 48 words.
   - Evaluation metrics are based on an overly simplified camera classifier.

### Dataset Gap

The field of cinematography **lacks large-scale multimodal datasets**. Existing datasets are either synthetic (CCD), omit character information and textual descriptions (RealEstate10K), or suffer from domain mismatch (e.g., human motion datasets like KIT/HumanML3D depict only human movements without camera motion).

### Core Motivation

The fundamental motivation of this work is to **democratize cinematography**—enabling ordinary users to generate professional-grade camera trajectories through natural language descriptions. This requires addressing two key issues:
1. Constructing the first real-movie camera trajectory dataset that includes character trajectories and text descriptions.
2. Designing a trajectory generation model capable of utilizing character-camera relationships.

## Method

### Overall Architecture

The contributions of this work consist of three parts:
1. **E.T. Dataset**: Camera and character trajectories extracted from real movies paired with text descriptions.
2. **Director**: A diffusion-based camera trajectory generation model.
3. **CLaTr**: A contrastive language-trajectory embedding for evaluation metrics.

### Key Designs

#### 1. E.T. Dataset Building Pipeline

**Function**: Extracts 3D trajectories of the camera and characters from real movie clips and generates paired textual descriptions.

**Mechanism**: A three-step pipeline—

**Step 1: Data Extraction and Preprocessing**
- Employs SLAHMR to jointly estimate camera and 3D human poses.
- Performs alignment, filtering, smoothing, and other preprocessing on raw outputs.
- Clips trajectories to a maximum of 300 frames.

**Step 2: Motion Tagging**
- Divides trajectories into pure motion segments, considering 6 base motions (left/right, up/down, forward/backward).
- Results in 27 motion combinations.
- Uses rigid body velocity $\in SE(3)$ for the camera to distinguish similar movements like trucking (moving laterally while facing perpendicularly) and depth (moving in the direction of look).
- Uses hip-centered linear velocity for characters.

**Step 3: Text Description Generation**
- Uses Hitchcock's rule to determine the main character (the character occupying the largest area in the frame).
- Employs the Mistral-7B LLM to convert motion tags into detailed textual descriptions.
- Generates two types of captions: camera-only descriptions and joint camera-character descriptions.

**Design Motivation**: In movies, cameras typically move relative to the filmed characters, making it essential to jointly model their relationship.

**Dataset Scale**: 115K samples, 11M frames, 120 hours, a vocabulary of ~5.4K (far exceeding CCD's 48), and 230K captions.

#### 2. Director Model (Diffusion Transformer for Camera Trajectory Generation)

**Function**: Generates camera trajectories via a diffusion process conditioned on character trajectories and textual descriptions.

**Mechanism**:

**Problem Formulation**: Represents the camera trajectory as a sequence of $N$ consecutive camera poses $\mathbf{x}_{1:N}$, where each pose $\mathbf{x} = [\mathbf{R}|\mathbf{t}]$ contains rotation (represented in continuous 6D) and translation. The conditioning factors include the character trajectory $\mathbf{h}_{1:N}$ (3D hip positions) and the text description $c$.

**Diffusion Framework**: Adopts the EDM paradigm to train a denoiser $D$ with the loss function:

$$\mathcal{L}_{\text{score}} = \frac{D(\mathbf{x}, \mathbf{h}, c; \sigma) - \mathbf{x}}{\sigma^2}$$

The sampling stage employs EDM's second-order deterministic sampling combined with classifier-free guidance.

**Three Conditioning Architectures** (inspired by DiT):
- **Director A** (In-context): Prepends the conditioning factors as context tokens to the transformer input.
- **Director B** (AdaLN): Modulates the transformer blocks using Adaptive Layer Norm (AdaLN) — $(1+\gamma)\text{LN}(X) + \beta$, initialized to zero output.
- **Director C** (Cross-attention): Leverages the full sequence length of conditions, fusing them via cross-attention.

**Design Motivation**: Unlike CCD which uses a character-centric coordinate system, Director operates in a **global coordinate system**, which allows for richer camera-character motion associations.

#### 3. CLaTr (Contrastive Language-Trajectory Embedding)

**Function**: Learns a shared feature embedding space between text and trajectories to evaluate generation quality.

**Mechanism**: Learns a shared embedding space for text and trajectories based on the contrastive learning paradigm of CLIP, utilizing a VAE framework with a trajectory encoder, a text encoder, and a shared feature decoder.

Training includes three losses:
- Reconstruction loss $\mathcal{L}_R$: Reconstruction quality of trajectory and text features.
- KL loss $\mathcal{L}_{KL}$: Regularizes the distribution of each modality and enforces cross-modal similarity.
- Cross-modal embedding similarity loss $\mathcal{L}_E$: Ensures alignment between text and trajectory features.

Based on CLaTr, evaluation metrics such as FD_CLaTr (similar to FID) and CLaTr-Score (similar to CLIP-Score) can be computed.

**Design Motivation**: Existing evaluation methods (e.g., the simple 6-class camera motion classifier used by CCD) cannot capture the true complexity of camera trajectories, necessitating a more robust evaluation tool.

### Loss & Training

**Training Strategy**
- Optimizer: AdamW, lr=1e-4, $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay=0.1
- Learning rate schedule: Cosine decay + 5K warmup steps, totaling 170K steps
- Model configuration: 8-layer Transformer, hidden dim=512, 16 attention heads
- Input: Sequence length of 300, using masking for shorter inputs
- Precision: Mixed precision training with bfloat16

## Key Experimental Results

### Main Results

**Trajectory generation quality comparison on the E.T. Mixed subset**:

| Method | FD_CLaTr ↓ | Precision ↑ | Coverage ↑ | CLaTr-Score ↑ | C-F1 ↑ |
|------|:----------:|:-----------:|:----------:|:------------:|:------:|
| CCD | 35.81 | 0.73 | 0.67 | 6.26 | 0.17 |
| MDM | 6.79 | 0.78 | 0.76 | 18.32 | 0.34 |
| **Director A** | **3.88** | 0.82 | 0.85 | 20.76 | 0.42 |
| **Director B** | 6.10 | 0.78 | 0.78 | 20.78 | 0.39 |
| **Director C** | **3.76** | **0.83** | **0.86** | **21.95** | **0.48** |

Compared to MDM, Director C reduces FD_CLaTr by 3.0 and by 32.1 compared to CCD; CLaTr-Score increases by 3.6 compared to MDM and by 15.7 compared to CCD.

**Comparison on downstream E.T. Pure subset**:

| Method | FD_CLaTr ↓ | Coverage ↑ | CLaTr-Score ↑ | C-F1 ↑ |
|------|:----------:|:----------:|:------------:|:------:|
| CCD | 31.33 | 0.72 | 3.21 | 0.27 |
| MDM | 6.10 | 0.80 | 21.26 | 0.76 |
| **Director C** | **4.57** | **0.87** | 21.49 | 0.80 |
| **Director B** | 6.61 | 0.82 | **23.10** | **0.86** |

### Ablation Study

**Comparison of Director architectural variants** (E.T. mixed):

| Architecture | Conditioning Method | FD_CLaTr ↓ | CLaTr-Score ↑ | C-F1 ↑ | Note |
|------|---------|:----------:|:------------:|:------:|------|
| Director A | In-context | 3.88 | 20.76 | 0.42 | Simple and efficient, performance close to C |
| Director B | AdaLN | 6.10 | 20.78 | 0.39 | Good for simple scenes, poor for complex scenes |
| Director C | Cross-attention | **3.76** | **21.95** | **0.48** | Best performance but has more parameters |

**Key Findings**: AdaLN achieves the best text-trajectory consistency on the E.T. Pure subset (C-F1=0.86) but performs worst on the Mixed subset (0.39). This suggests that AdaLN handles simple conditioning signals well but struggles to capture sequence complexity.

### Key Findings

1. **Director outperforms CCD and MDM across all metrics**, especially in mixed and complex trajectories.
2. **The Cross-attention architecture is best-suited for managing complex conditioning signals**, whereas the in-context approach serves as a simple and efficient alternative.
3. **Character information is crucial**: Camera trajectory generation must take the relationship with character motion into account.
4. **The FD-Score trade-off curve for the CLaTr evaluation metric** demonstrates that Director consistently outperforms MDM.

## Highlights & Insights

1. **Pioneering Dataset Contribution**: E.T. is the first real-world movie dataset containing camera trajectories, character trajectories, and text descriptions simultaneously, filling a major gap in the literature.
2. **Global Coordinate System Design**: Compared to CCD's character-centric coordinate system, the global coordinate system allows for much richer and more diverse camera-character motion relationships.
3. **CLaTr Evaluation Framework**: Establishes a standardized evaluation utility for the camera trajectory generation field, akin to FID's contribution to image generation.
4. **Clear Demonstration of Four Qualitative Strengths**: Controllability, diversity, complexity, and character awareness.

## Limitations & Future Work

1. **Limited Expressiveness of Trajectory Descriptions**: Current captions lack fine-grained information, such as the specific positioning of characters in the frame or descriptive modifiers.
2. **Accuracy of 3D Pose Estimation**: Estimating 3D poses from 2D video is noisy and inherently error-prone, which might affect dataset quality.
3. **Support Only for Single-Character Scenes**: Camera trajectory generation for multi-character interaction scenes remains unexplored.
4. **Lack of Integration with Video Generation**: The model generates abstract 3D trajectories; integrating these with video rendering or generation systems remains a subject for future work.
5. **Dataset Dominated by Western Cinema**: Diversity of cultures and filming styles needs to be further expanded in future iterations.

## Related Work & Insights

- **Human Motion Generation (MDM, HumanML3D)**: Director borrows architectural concepts from human motion diffusion models and successfully adapts them to camera trajectory generation.
- **DiT (Diffusion Transformer)**: The three conditioning injection architectures are directly inspired by DiT.
- **CLIP/TMR**: The contrastive learning framework of CLaTr is derived from image-text and motion-text contrastive learning paradigms.
- **SLAHMR**: A key 3D pose estimation tool that enables data extraction from real movies.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Both the dataset and the method are pioneering, filling an important gap in the field of cinematography.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Offers rich quantitative metrics and strong qualitative analyses, though the ablation studies could be deeper (e.g., examining the impact of character conditioning).
- **Writing Quality**: ⭐⭐⭐⭐ — Features clear structure and a comprehensive narrative from dataset construction to application.
- **Value**: ⭐⭐⭐⭐ — High dataset value that can drive the automation of cinematography, though the application scenarios are relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] Free-Viewpoint Video of Outdoor Sports Using a Flying Camera](free-viewpoint_video_of_outdoor_sports_using_a_flying_camera.md)
- [\[ECCV 2024\] Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese Handwriting](elegantly_written_disentangling_writer_and_character_styles_for_enhancing_online.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](../../ACL2025/others/mapmake_schema_guided_text_to_table_generation.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](../../AAAI2026/others/an_epistemic_perspective_on_agent_awareness.md)

</div>

<!-- RELATED:END -->
