---
title: >-
  [Paper Note] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models
description: >-
  [AAAI 2026][Image Generation][Music Editing] Through systematic probing analysis of attention maps in diffusion models, this work reveals that self-attention maps are critical for preserving the temporal structure of mus…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Music Editing"
  - "Diffusion Models"
  - "Attention Mechanism"
  - "Training-Free"
  - "Self-Attention Manipulation"
date: 2026-05-08
content_hash: bd332d580c06c9d2
---

# Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models

**Conference**: AAAI 2026
**arXiv**: [2511.08252](https://arxiv.org/abs/2511.08252)  
**Code**: None  
**Area**: Image Generation / Music Editing
**Keywords**: Music Editing, Diffusion Models, Attention Mechanism, Training-Free, Self-Attention Manipulation

## TL;DR

Through systematic probing analysis of attention maps in diffusion models, this work reveals that self-attention maps are critical for preserving the temporal structure of music. Based on this finding, Melodia is proposed — a training-free music editing method that achieves an optimal balance between attribute modification and structural preservation by selectively manipulating self-attention maps.

## Background & Motivation

### State of the Field

Text-driven music generation has advanced rapidly, and diffusion-based methods such as AudioLDM 2 are now capable of producing high-quality music. Naturally, researchers have begun exploring **text-guided music editing** — modifying attributes such as instrument type, style, and mood via textual instructions. Music editing falls into two categories: **inter-stem editing** (adding/removing instrument tracks) and **intra-stem editing** (modifying timbre, style, and other features within a single track while preserving melody and structure).

### Limitations of Prior Work

**High training cost**: Existing methods either train dedicated models from scratch (e.g., MusicLM) or fine-tune pretrained models (e.g., Instruct-MusicGen), both of which require substantial computational resources and training data.

**Dependency on source music descriptions**: Most methods (e.g., MusicMagus) require users to provide textual descriptions of the source music to guide editing, which is difficult for general users who cannot accurately characterize musical features.

**Temporal structure degradation**: Edited music frequently loses the melodic and rhythmic structure of the source, which is clearly visible in spectrogram comparisons.

**Insufficient understanding of attention mechanisms**: Although attention manipulation techniques have been developed for image editing (e.g., Prompt-to-Prompt), the functional role of attention in **music diffusion models** has not been systematically studied.

### Root Cause

The fundamental tension in music editing is as follows: effectively modifying musical attributes (e.g., changing drums to violin) necessarily alters the conditioning signals in the generation process, which tends to simultaneously destroy the temporal structure (melody, rhythm) of the source music. How can these two objectives be reconciled?

### Core Idea

**Revealing internal mechanisms of diffusion models through probing analysis**: Cross-attention maps encode rich semantic information about musical attributes (instrument, style, mood), and manipulating them leads to editing failure. Self-attention maps, by contrast, do not encode attribute semantics, yet are critical for preserving temporal structure. Based on this finding, selectively replacing the self-attention maps in the target generation process with those from the source music enables high-quality, training-free music editing without requiring source descriptions.

## Method

### Overall Architecture

Melodia operates through the following pipeline:

1. The source music is encoded into a latent representation $z_0$ via a VAE encoder.
2. **Partial DDIM Inversion** is applied to invert $z_0$ to an intermediate noise state $z_{T_{start}}$, while collecting the self-attention Q and K at each timestep into an **Attention Repository**.
3. During the denoising process conditioned on the target prompt, **Attention-based Structure Retention (ASR)** injects the stored Q and K into the corresponding layers' self-attention computation, replacing the attention maps of the target generation with those of the source music.
4. The edited music is obtained by decoding the resulting latent.

### Key Designs

#### 1. **Probing Analysis**

**Function**: Classifiers are constructed to detect whether attention maps encode musical attribute information.

**Mechanism**: Prompt datasets are constructed along three dimensions — instrument (16 classes), style (11 classes), and mood (8 classes). Cross-attention and self-attention maps from each layer of AudioLDM 2 are extracted, and simple MLP classifiers are trained for classification.

**Key Findings**:
- **Cross-attention maps**: Classification accuracy reaches 70–100%, indicating that they encode rich attribute semantic information → manipulating cross-attention maps leads to editing failure.
- **Self-attention maps**: Classification accuracy is below 40%, indicating that they do not encode attribute semantics → replacement experiments confirm that they are critical for preserving temporal structure.

**Design Motivation**: Analogous findings have been reported in image editing (self-attention preserves spatial structure), but this is the first systematic validation in the music domain, providing a theoretical foundation for the proposed method.

#### 2. **DDIM Inversion with Attention Repository**

**Function**: Self-attention features are collected and stored during the inversion of the source music.

**Mechanism**: Partial DDIM Inversion is adopted (inverting only to $T_{start}$ rather than fully to $T$). At each inversion timestep $t$, the self-attention queries $Q_t^s$ and keys $K_t^s$ are stored in the Attention Repository.

**Theoretical Basis**: Drawing on Content-Style Modeling theory, music is assumed to be decomposable into content (e.g., melodic structure) and style (e.g., timbre). The inverted $z_{T_{start}}$ provides only implicit structural guidance, which is far weaker than the semantic guidance of the target prompt and thus leads to structural deviation. The Attention Repository is therefore introduced to provide **explicit structural guidance**.

#### 3. **Attention-based Structure Retention (ASR)**

**Function**: Source music attention features are converted into structural guidance during the denoising process.

**Mechanism**: At each denoising timestep $t$, the stored source music Q and K are used to compute the self-attention map $M_t'^s$, while the Value is projected from the target latent representation:

$$\phi_t'^s = M_t'^s \cdot V_t'^s$$
$$M_t'^s = \text{Softmax}\left(\frac{Q_t^s {K_t^s}^\top}{\sqrt{d^s}}\right)$$

where $Q_t^s, K_t^s$ are drawn from the Attention Repository (source music) and $V_t'^s$ is derived from the current target denoising process.

**Layer Selection**: This manipulation is applied only to **layers 8–14** of AudioLDM 2. Experiments show that replacing all layers (1–16) retains too much of the source timbre, while replacing only the middle layers yields the best results.

**Design Motivation**: The Q and K of self-attention maps encode relational patterns among spatial/temporal positions (i.e., structure), while V carries the actual feature content of the current generation. Editing is achieved by "borrowing the structural relations of the source music while injecting the content features of the target music."

### Evaluation Metric Contributions

Two new composite evaluation metrics are proposed:
- **ASB (Adherence-Structure Balance Score)**: Balances CLAP (text adherence) and LPAPS (structure preservation) via harmonic mean.
- **AMB (Adherence-Musicality Balance Score)**: Balances CLAP and Chroma (harmonic preservation) via harmonic mean.
- **MEB (Music Editing Balance)**: A balance metric used in subjective evaluation.

### Loss & Training

Melodia is a **completely training-free** method and involves no loss functions or training procedures. Editing is accomplished solely through attention manipulation during inference.

## Key Experimental Results

### Main Results

Objective evaluation results across three datasets (key metrics selected):

| Dataset | Method | CLAP↑ | LPAPS↓ | Chroma↑ | FAD↓ | ASB↑ | AMB↑ |
|--------|------|-------|--------|---------|------|------|------|
| MusicDelta | DDPM-Friendly | 0.35 | 5.66 | 0.27 | 0.88 | 0.58 | 0.74 |
| MusicDelta | **Melodia** | **0.34** | **4.01** | **0.32** | **0.56** | **1.00** | **1.00** |
| ZoME-Bench | DDPM-Friendly | 0.23 | 5.70 | 0.27 | 0.68 | 0.49 | 0.72 |
| ZoME-Bench | **Melodia** | **0.29** | **3.90** | **0.29** | **0.47** | **1.00** | **1.00** |
| MelodiaEdit | DDPM-Friendly | 0.34 | 4.06 | 0.70 | 0.67 | 0.59 | 0.70 |
| MelodiaEdit | **Melodia** | **0.39** | **3.11** | **0.68** | **0.65** | **1.00** | **0.88** |

Subjective evaluation results (5-point Likert scale):

| Dataset | Method | REL↑ | CON↑ | MEB↑ |
|--------|------|------|------|------|
| MusicDelta | DDPM-Friendly | 3.09 | 2.88 | 3.02 |
| MusicDelta | **Melodia** | **3.21** | **3.59** | **3.46** |
| MelodiaEdit | DDPM-Friendly | 2.58 | 2.92 | 2.78 |
| MelodiaEdit | **Melodia** | **3.38** | **3.65** | **3.81** |

### Ablation Study

Layer selection ablation (on the Timbre Transfer task of MelodiaEdit):

| Layer Selection | CLAP↑ | LPAPS↓ | ASB↑ | AMB↑ | Notes |
|--------|-------|--------|------|------|------|
| None | 0.34 | 4.39 | 0.00 | 0.00 | No structural guidance |
| 1–16 (all) | 0.34 | 2.65 | 0.00 | 0.00 | Retains too much source timbre |
| 6–16 | 0.35 | 2.96 | 0.22 | 0.22 | Partially over-constrained |
| **8–14** | **0.42** | **3.49** | **0.68** | **0.57** | **Optimal balance** |
| 10–12 | 0.39 | 3.93 | 0.37 | 0.56 | Insufficient structural guidance |

### Key Findings

1. **Functional division between cross-attention and self-attention**: Cross-attention encodes musical attribute semantics (classification accuracy >70%), while self-attention encodes temporal structure (classification accuracy <40%) — this is the first systematic revelation of such a division in music diffusion models.
2. **Layer selection is critical**: Layers 8–14 are optimal; replacing too few or too many layers degrades editing quality.
3. **Cross-model generalizability**: Experiments on Stable Audio Open demonstrate that Melodia generalizes to different diffusion model architectures and sampling rates.
4. **Necessity of composite metrics**: Traditional single metrics are easily misleading (e.g., MusicMagus achieves high Chroma but fails at editing); ASB/AMB more accurately reflect editing quality.

## Highlights & Insights

1. **Methodological innovation through probing analysis**: Probing analysis methods from NLP are introduced into music diffusion models, providing a new perspective for understanding internal model mechanisms.
2. **Training-free and source-description-free**: The approach significantly lowers the barrier to use, requiring neither additional training nor user-provided descriptions of the source music.
3. **Unity of theory and practice**: The method is designed from empirical findings (probing analysis) rather than intuition followed by post-hoc validation, offering a methodologically instructive paradigm.
4. **New evaluation metrics**: ASB and AMB employ harmonic means to ensure neither dimension is neglected, and the MelodiaEdit benchmark is introduced for standardized evaluation.

## Limitations & Future Work

1. **Dependence on the specific architecture of AudioLDM 2**: The layer selection (8–14) is manually determined for the 16-layer UNet of AudioLDM 2; re-probing is required when applied to other models.
2. **Supports only intra-stem editing**: Inter-stem editing (adding/removing instrument tracks) is not supported.
3. **Requires Partial DDIM Inversion**: The inversion process still incurs non-trivial computational overhead, and the choice of $T_{start}$ requires tuning.
4. **Limited music length**: The maximum editable duration is constrained by the generation length of the underlying model, making long-segment editing difficult.

## Related Work & Insights

- **Image editing analogy**: Works such as Prompt-to-Prompt and MasaCtrl manipulate attention in image diffusion models; this paper transfers similar ideas to the music domain.
- **Content-Style decomposition theory**: Provides a theoretical framework explaining why self-attention replacement preserves structure.
- **Implications for other modalities**: The probing analysis methodology is generalizable to diffusion model editing in other modalities such as video and speech.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic probing analysis of attention mechanisms in music diffusion models; findings are valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, objective and subjective evaluations, extensive ablations, and cross-model validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical flow with a complete argumentation chain from analysis to method design.
- **Value**: ⭐⭐⭐⭐ — The training-free approach offers strong practical utility, though the application scope is relatively narrow (music editing).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](../../CVPR2026/image_generation/ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](../../ICLR2026/image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](../../CVPR2026/image_generation/object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](infinite-story_a_training-free_consistent_text-to-image_gene.md)

</div>

<!-- RELATED:END -->
