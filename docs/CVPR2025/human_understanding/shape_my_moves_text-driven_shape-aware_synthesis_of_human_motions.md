---
title: >-
  [Paper Note] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions
description: >-
  [CVPR 2025][Human Understanding][Human Motion Generation] This paper proposes the ShapeMove framework, which injects continuous body shape information into discretely quantized motion tokens via a Shape-Aware FSQ-VAE, and utilizes a pretrained language model to jointly predict shape parameters and motion tokens. It achieves the first end-to-end shape-aware human motion generation from natural language descriptions.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Human Motion Generation"
  - "Shape-Aware"
  - "Text-Driven"
  - "Quantized Autoencoder"
  - "Large Language Model"
date: 2026-05-08
content_hash: acfe8721dc8b625d
---

# Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions

**Conference**: CVPR 2025  
**arXiv**: [2504.03639](https://arxiv.org/abs/2504.03639)  
**Code**: [https://shape-move.github.io/](https://shape-move.github.io/)  
**Area**: Human Understanding  
**Keywords**: Human Motion Generation, Shape-Aware, Text-Driven, Quantized Autoencoder, Large Language Model

## TL;DR

This paper proposes the ShapeMove framework, which injects continuous body shape information into discretely quantized motion tokens via a Shape-Aware FSQ-VAE, and utilizes a pretrained language model to jointly predict shape parameters and motion tokens. It achieves the first end-to-end shape-aware human motion generation from natural language descriptions.

## Background & Motivation

**Background**: Text-to-motion generation has achieved significant progress. Mainstream methods (such as T2M-GPT, MDM, MotionDiffuse) normalize motion representations to a canonical human model—where motion sequences of individuals with completely different body shapes are mapped to identical motion trajectories.

**Limitations of Prior Work**: In reality, executing the same motion exhibits notable physiological variations across different body shapes—runs of obese and lean individuals differ in gait, arm span, and center of mass transfer. Existing methods overlook these differences, and the generated "one-size-fits-all" motions produce artifacts like self-penetration, foot sliding, and unnatural joint bending when transferred to different body shapes. The root problem lies in the difficulty of aligning continuous shape details with discrete quantized motion representations.

**Key Challenge**: Motion quantization (VQ-VAE) substantially improves generation quality and efficiency through discretization, but discrete codebooks lack the capacity to encode fine-grained variations caused by shape changes. Direct training of the quantizer on shape-aware motions leads to codebook explosion or extremely low utilization.

**Goal**: Build an end-to-end framework to simultaneously predict shape parameters and shape-aware motion sequences from natural language.

**Key Insight**: The authors propose a "content-style separation" strategy—training the discrete quantizer with normalized motions (to learn motion content), and injecting continuous shape parameters as a style condition during the decoding stage, thereby leveraging the efficiency of quantization while retaining continuous shape details.

**Core Idea**: Decouple motion content (discrete tokens) from individual style (continuous body shape parameters) in a quantized network, quantizing canonical motions with FSQ and reconstructing shape-aware motions via conditional decoding.

## Method

### Overall Architecture

The system consists of two stages: Stage 1 is the Shape-Aware FSQ-VAE (SA-VAE), which quantizes normalized motions into discrete tokens and injects shape parameters during decoding to reconstruct shape-aware motions; Stage 2 is ShapeMove, which utilizes a pretrained T5 language model to predict both body shape parameters and motion token sequences from text descriptions. During inference, the motion tokens output by T5 are de-quantized by FSQ and fed alongside the predicted shape parameters into the SA-VAE decoder to generate the final motion.

### Key Designs

1. **Shape-Aware FSQ-VAE (SA-VAE)**:

    - **Function**: Introduces continuous shape information during motion quantization to achieve content-style decoupled motion representation.
    - **Mechanism**: The encoder $\mathcal{E}$ takes normalized motion $X^N \in \mathbb{R}^{T \times D}$ ($D=263$) and encodes it into latent features $Z \in \mathbb{R}^{\tau \times D}$ ($\tau$ is the downsampled length). Finite Scalar Quantization (FSQ) quantizes $Z$ into discrete features $\hat{Z}$ (codebook size $k=1000$, dimensional configurations $\ell=[8,5,5,5]$). The decoder $\mathcal{D}$ receives $\hat{Z}$ concatenated along the temporal dimension with shape features $\tilde{\beta} = P_{\theta_s}(\beta)$ mapped via an MLP, decoding the shape-aware motion $\hat{X}^R = \mathcal{D}(\hat{Z} \oplus \tilde{\beta})$. Crucially, the encoder only encodes normalized motions (free of shape information), ensuring that the codebook learns only motion semantics.
    - **Design Motivation**: Direct quantization of shape-aware motions would require an exponentially larger codebook to cover all shape variations; by encoding canonical motions + conditional decoding, a codebook of 1000 codes can yield reconstruction quality superior to T2M-GPT.

2. **ShapeMove (Shape-Motion Token Predictor)**:

    - **Function**: Simultaneously predicts body shape parameters and motion token sequences from natural language descriptions.
    - **Mechanism**: Adds $k+2$ motion tokens ($k$ codebook codes + start/end tokens) and 1 body shape token `[BETA]` to the vocabulary of a pretrained T5 model. The model takes text as input and autoregressively predicts `{[BETA], \hat{C}}`. Shape parameters are obtained by extracting the embedding $M_\beta$ corresponding to `[BETA]` and projecting it as $\hat{\beta} = P_{\theta_e}(M_\beta)$, exploiting the property that the last embedding layer of the language model retains continuous information. The training objective is cross-entropy loss for motion tokens + L1 loss for shape parameters.
    - **Design Motivation**: Shape parameters are static values rather than sequences, making it impossible to represent them like motion sequences with token streams; extracting continuous information via the embedding space is an elegant piggybacking strategy that avoids designing a separate discretization scheme for body shapes.

3. **Physical Constraint Loss and Shape Data Augmentation**:

    - **Function**: Guarantees the physical plausibility of generated motions and expands the diversity of body shapes in the training data.
    - **Mechanism**: The loss function of SA-VAE includes reconstruction loss $L_r$, floating loss $L_{\text{float}}$ (distance of the lowest joint to the ground), foot sliding loss $L_{\text{slide}}$ (velocity of the contacting foot on the ground), and bone length loss $L_{\text{bone}}$ (difference in bone length between reconstructed and ground truth joints). Data augmentation uses Shapy's A2S model to generate synthetic shape parameters from language attributes, replacing the ground truth shapes of 10% of the training samples.
    - **Design Motivation**: Information loss introduced by quantization can compromise physical plausibility (self-penetration, foot floating/sliding), and physical losses serve as explicit constraints; because existing datasets have limited shape diversity, synthetic augmentation extends the generalization scope of the model.

### Loss & Training

SA-VAE is trained for 300K iterations (200K with lr=2e-4 + 100K with lr=1e-5) with a batch size of 256, taking about 12 hours on a single A100 GPU. ShapeMove utilizes a T5 model (12 layers each for encoder and decoder), pre-trained on joint motion-to-text and text-to-motion tasks for 120K steps (8×A100, 1 day), and then fine-tuned solely on the text-to-motion task for 30K steps (10 hours). Motion sequences are cropped to length $T=64$, downsampling to $\tau=16$.

## Key Experimental Results

### Main Results

| Method | Shape Input | Penetrate↓(cm) | Float↓(cm) | Skate↓(%) | Bone Length Var↓ | FID↓ | R-Precision Top3↑ |
|------|------------|-----------------|------------|-----------|-----------------|------|-------------------|
| T2M-GPT | ✓ | 0.1789 | 0.5241 | 6.162 | 1.176 | 0.269 | 0.683 |
| MotionGPT | ✓ | 0.6986 | 0.2245 | 7.889 | 2.271 | 1.020 | 0.271 |
| MotionDiffuse | ✓ | 0.2401 | 0.2703 | 7.710 | 0.138 | 0.563 | 0.723 |
| **ShapeMove** | **✓** | **0.0268** | 0.2658 | **6.143** | **0.625** | **0.198** | **0.705** |

### Ablation Study

| Configuration | FID↓ | Bone Length Diff↓(mm) | Float↓(cm) | Skate↓(%) |
|------|------|----------------------|-----------|-----------|
| No shape-conditioning | 0.148 | 99.18 | 0.575 | 6.76 |
| + shape-conditioning | 0.105 | 66.41 | 0.567 | 6.60 |
| + $L_{\text{bone}}$ | 0.107 | 45.11 | 0.480 | 7.07 |
| + $L_{\text{bone}} + L_{\text{float}}$ | 0.137 | 45.97 | 0.255 | 7.90 |
| + Full (all losses) | 0.125 | 45.88 | 0.266 | 6.14 |

### Key Findings

- ShapeMove substantially outperforms T2M-GPT on the self-penetration metric (Penetrate) with 0.0268cm versus 0.1789cm (a 6.7-fold improvement), which is close to the reconstruction upper bound of SA-VAE (0.0289cm), demonstrating that shape-aware generation virtually eliminates ground penetration.
- Shape conditioning reduces the bone length error from 99.18mm to 66.41mm, and adding $L_{\text{bone}}$ further reduces it to 45.11mm (halved), proving the efficacy of the decoupled design and physical losses.
- In the quantizer reconstruction comparison, the bone length deviation of SA-VAE is 45.88mm, which is about half that of T2M-GPT (83.42mm), and yields the lowest FID (0.125 vs 0.151), demonstrating the superiority of shape-conditioned decoding.
- In human perceptual evaluations, ShapeMove is close to ground-truth levels in three dimensions—shape-to-text matching, motion-to-text matching, and shape-motion plausibility—outperforming all baselines by 12% to 38%.
- The body shape attribute prediction error is around 1cm (0.58cm for height and 1.06cm for waist circumference), indicating that language model embeddings can effectively encode continuous shape information.

## Highlights & Insights

- **Content-style decoupled quantization strategy** is elegant—the encoder observes only normalized motions to ensure high codebook utilization, while the decoder restores individual differences through shape conditioning. This concept can be transferred to any generation task with a "shared structure + individual variance" formulation (e.g., speaker-styled text-to-speech synthesis).
- **Leveraging language model embeddings to extract continuous information** is an elegant hack—shape parameters cannot be discretized, but the language model's embedding space naturally carries continuous information. This technique of "mining continuous signals from discrete models" has wide potential in multimodal generation.
- FSQ circumvents codebook regularization and codebook collapse compared to VQ-VAE, making it a superior choice for quantization schemes.

## Limitations & Future Work

- Validated only on the HumanML3D dataset, which has limited shape diversity (449 subjects), with synthetic augmentation covering only 10%.
- Relies on the $\beta$ parameters of the SMPL model to represent body shapes, failing to capture shape variations beyond the expressive range of SMPL.
- Physical plausibility constraints are purely geometric (floating, sliding, penetration) without incorporating dynamics constraints (e.g., inertia, muscle forces).
- Shape descriptions in textual prompts are relatively coarse-grained (using adjectives for height and body shape), restricting precise control over the shape.
- Has not explored integrating with a real physics simulator to further improve physical realism.

## Related Work & Insights

- **vs T2M-GPT**: Both use a quantizer + Transformer to predict token sequences, but T2M-GPT cannot recover body shape information after quantizing normalized motions. ShapeMove's SA-VAE achieves a qualitative leap by injecting body shape at the decoding stage.
- **vs MDM / MotionDiffuse**: Diffusion approaches can handle continuous conditional inputs (including shape), but perform significantly worse than ShapeMove in Penetrate and FID, indicating that diffusion models are inferior to quantization-plus-decoding schemes in physical constraint learning.
- **vs HUMOS**: HUMOS also executes shape-aware motion generation but focuses on the physical simulation level, whereas ShapeMove addresses the issue from a learning and quantization perspective, making them complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Content-style decoupled quantization and extracting continuous information from embeddings are creative designs, though the overall framework is combinational.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across quantitative comparisons, ablations, quantizer comparisons, attribute predictions, and human perceptual evaluations.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly explained and figures are abundant, though the Related Work section is somewhat long.
- Value: ⭐⭐⭐⭐ Fills the gap in text-to-shape-aware-motion generation and delivers significant improvements in physical plausibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](../../ECCV2024/human_understanding/humos_human_motion_model_conditioned_on_body_shape.md)
- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2025\] StickMotion: Generating 3D Human Motions by Drawing a Stickman](stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)
- [\[CVPR 2026\] Towards Storytelling Animations: Joint Synthesis of Human and Camera Motions](../../CVPR2026/human_understanding/towards_storytelling_animations_joint_synthesis_of_human_and_camera_motions.md)

</div>

<!-- RELATED:END -->
