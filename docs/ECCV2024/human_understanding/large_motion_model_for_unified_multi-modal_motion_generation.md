---
title: >-
  [Paper Note] Large Motion Model for Unified Multi-Modal Motion Generation
description: >-
  [ECCV 2024][Human Understanding][Motion Generation] This paper proposes the Large Motion Model (LMM), the first motion-centric unified multimodal motion generation foundation model. By constructing the MotionVerse benchmark containing 10 tasks, 16 datasets, and 320K sequences, designing a body-part-aware ArtAttention mechanism, and incorporating a pre-training strategy with random frame rates and masking, LMM achieves high-quality motion generation across diverse tasks.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Motion Generation"
  - "Multimodal"
  - "Diffusion Models"
  - "Unified Model"
  - "large-scale"
date: 2026-05-08
content_hash: 0899567236a955a9
---

# Large Motion Model for Unified Multi-Modal Motion Generation

**Conference**: ECCV 2024  
**arXiv**: [2404.01284](https://arxiv.org/abs/2404.01284)  
**Code**: [Project Page](https://mingyuan-zhang.github.io/projects/LMM.html)  
**Area**: Multimodal VLM  
**Keywords**: Motion Generation, Multimodal, Diffusion Models, Unified Model, large-scale

## TL;DR

This paper proposes the Large Motion Model (LMM), the first motion-centric unified multimodal motion generation foundation model. By constructing the MotionVerse benchmark containing 10 tasks, 16 datasets, and 320K sequences, designing a body-part-aware ArtAttention mechanism, and incorporating a pre-training strategy with random frame rates and masking, LMM achieves high-quality motion generation across diverse tasks.

## Background & Motivation

**Background**: Human motion generation is a core technology in animation and video production, encompassing multiple subtasks such as text-to-motion, music-to-dance, and motion prediction. Specialized models tailor-made for each subtask have achieved promising results.

**Limitations of Prior Work**: These specialist models operate in isolation, being trained only on single tasks and single datasets. They suffer from limited data volume and narrow data domains, leading to restricted model capacity and poor generalization.

**Key Challenge**: Although building a unified motion model can leverage massive multi-source data to improve generalization, it faces three major hurdles: (1) inconsistent motion formats across different datasets (e.g., keypoints vs. rotation representations); (2) diverse evaluation metrics across different tasks; and (3) difficulty in transferring motion knowledge across tasks due to varying frame rates, keypoint counts, and missing body parts.

**Goal**: To unify multimodal and multi-task motion generation into a single generalist model, achieving performance comparable to or exceeding that of specialist models across various tasks.

**Key Insight**: Systematically address the aforementioned challenges from three levels: data (unified representation), architecture (body-part-aware attention), and training strategy (unsupervised pre-training + supervised fine-tuning).

**Core Idea**: Unify heterogeneous motion data from 16 datasets into a body-part-segmented intermediate representation, design the ArtAttention mechanism to handle missing body parts and multi-modal conditional inputs, and leverage large-scale motion data through pre-training with random frame rates and masks.

## Method

### Overall Architecture

LMM is based on a Transformer-based Diffusion Model, with its overall pipeline divided into two stages:

1. **Unsupervised Pre-training Stage**: Utilizes only the motion sequences without conditional signals, enhancing the learning of motion priors through random downsampling and random masking strategies.
2. **Supervised Fine-tuning Stage**: Introduces multimodal conditional signals (text, music, audio, video) to enable the model to learn mapping relationships between conditions and motions.

Architecturally, the model consists of a Read-In Layer (dataset-specific input encoders) $\to$ ArtAttention backbone network $\to$ Read-Out Layer (dataset-specific output decoders).

### MotionVerse Dataset

To address the inconsistency in data formats, the authors construct the MotionVerse benchmark:

- **Scale**: 10 tasks, 16 datasets, 320K sequences, ~100M frames
- **Unified Representation**: Adopts an intermediate format similar to TOMATO, decomposing the motion representation into 10 independent body parts: global orientation/trajectory, facial expression, head, spine, left arm, right arm, left leg, right leg, left hand, and right hand.
- **Handling Missing Data**: Allows missing parts for certain body segments and labels them in the metadata.
- **Evaluation Mapping**: Trains a motion translator to map the unified format back to dataset-specific formats, enabling cross-dataset evaluation.
- **Conditional Alignment**: Uses ImageBind to encode multimodal conditions (text, audio, music, video) into a unified feature space.

### Key Designs

1. **Read-In/Read-Out Layer (Dataset-Adaptive Encoder/Decoder Layer)**: Since distribution discrepancies across different datasets cannot be entirely ignored, dataset-specific encoders and decoders are applied at the input and output stages. During training, there is a 10% probability of replacing the dataset name with "all", enabling the general encoder-decoder to be used in practical deployment.

2. **ArtAttention (Articulated Attention Mechanism)**: The core innovative module, divided into two branches:

    - **Spatial Attention (Body-part Attention)**: For each frame, an attention mechanism is employed along the body-part dimension to model inter-part relationships. Since some body parts are naturally missing or artificially masked during pre-training, fixed attention weights cannot be used; hence, self-attention is applied to dynamically compute inter-part contributions.
    - **Temporal Attention**: Uses multi-head attention where each head corresponds to a specific body part. Key improvements include:
        - Incorporating a Mixture-of-Experts (MoE) to generate a unified Key representation from multimodal conditional features.
        - Separately normalizing motion features $\mathbf{K}_x$ and conditional features $\mathbf{K}_c$ (preventing long conditional sequences from diluting motion autocorrelation).
        - Introducing 64 learnable tokens as placeholders for unconditional generation.
        - Utilizing real time instead of frame indices to support arbitrary frame rates.
    - Final Output: $\mathbf{Y} = \mathbf{Y}_s + \mathbf{Y}_t$

3. **Pre-training Strategy (Random Downsampling + Random Masking)**:

    - **Random Downsampling**: Randomly downsamples the sequences and recalculates the velocity terms to match the downsampling rate, enabling the model to adapt to data with different original frame rates.
    - **Random Masking**: On top of the original missing mask $\mathbf{M}_s$, additional body parts are masked with a certain probability to obtain $\mathbf{M}_t$, where masked parts are replaced with a learnable empty token. When computing the loss, only the regions marked by $\mathbf{M}_s$ are ignored, forcing the model to infer the masked parts using the visible parts.
    - **Crucial Role**: Prevent the model from relying solely on the noisy sequence itself to recover information during diffusion, forcing it to rely more on conditional signals during fine-tuning.

### Loss & Training

- **Standard Diffusion Loss**: Follows the standard DDPM training paradigm.
- **Pre-training**: Adam optimizer, learning rate of $2 \times 10^{-4}$, 80K iterations.
- **Fine-tuning**: Conducted in two stages: first 20K steps (lr=$2 \times 10^{-4}$), then another 20K steps (lr=$2 \times 10^{-5}$).
- **Classifier-free guidance**: During fine-tuning, conditional signals are randomly masked with a 10% probability.
- **Four Model Variants**: LMM-Tiny (90M), LMM-Small (160M), LMM-Base (410M), and LMM-Large (760M).
- **Training Details**: Total batch size of 512, trained on up to 32 V100 GPUs.

## Key Experimental Results

### Main Results

**Text-to-Motion (HumanML3D)**:

| Method | R-Precision Top1↑ | FID↓ | MM Dist↓ | Diversity↑ | MultiModality↑ |
|------|-------------------|------|----------|-----------|----------------|
| T2M-GPT | 0.491 | 0.116 | 3.118 | 9.761 | 1.856 |
| FineMoGen | 0.504 | 0.151 | 2.998 | 9.263 | 2.696 |
| MoMask | 0.521 | 0.045 | 2.958 | - | 1.241 |
| **LMM-Large** | **0.525** | **0.040** | **2.943** | **9.814** | 2.683 |

**Music-to-Dance (AIST++)**:

| Method | FID_k↓ | FID_g↓ | Div_k↑ | BAS↑ |
|------|--------|--------|--------|------|
| Bailando | 28.16 | 9.62 | 7.83 | 0.2332 |
| TM2D | 19.01 | 20.09 | 9.45 | 0.2049 |
| **LMM-Large** | 22.08 | 21.97 | **9.85** | 0.2249 |

### Ablation Study

**Pre-training Strategy Ablation (LMM-Base)**:

| Config | Downsample | Random Mask | Attention | HumanML3D Top1 | HumanML3D FID | AMASS 1000ms |
|------|-----------|-------------|-----------|---------------|---------------|-------------|
| 1 | ✗ | ✗ | ArtAttention | 0.031 | 32.814 | 89.3 |
| 3 | ✗ | ✓ | ArtAttention | 0.515 | 0.151 | 76.1 |
| 4 | ✓ | ✓ | SAMI | 0.400 | 1.866 | 80.9 |
| **5** | **✓** | **✓** | **ArtAttention** | **0.511** | **0.138** | **73.6** |

### Key Findings

1. **Random Masking is a Crucial Component**: Without random masking (Configs 1 and 2), the model fails to perform the text-to-motion task (FID > 30). This is because the representation capability of the diffusion model is strong enough to recover motion solely from noise without relying on conditional signals.
2. **Significant Model Scaling Effects**: Scaling up from Tiny to Large improves R-Precision from 0.496 to 0.525 and reduces FID from 0.415 to 0.040, demonstrating a clear scaling law.
3. **Advantage in Long-term Prediction**: In the motion prediction task, LMM-Large significantly outperforms specialist models at longer time steps (880-1000ms), illustrating that large-scale pre-training provides stronger motion priors.
4. **ArtAttention Outperforms SAMI**: In the context of large models, the independent normalization strategy of ArtAttention is better suited for handling diverse multimodal inputs.
5. **Generalization Capability**: LMM-Large exhibits even more pronounced advantages on 3DPW (an out-of-distribution dataset), validating the generalization capacity yielded by large-scale training.

## Highlights & Insights

1. **Transfer of "Large Model" Paradigm to Motion Generation**: For the first time, the "large data + unified representation + pre-training & fine-tuning" paradigm from the LLM field is systematically applied to human motion generation, establishing a clear scaling path.
2. **Exquisite Body-Part-Segmented Representation**: Decomposing the human body into 10 independent components not only resolves keypoint inconsistencies across datasets but also inherently supports handling of missing parts and part-level controllable generation.
3. **Independently Normalized Conditioning**: Finding that direct concatenation under multiple conditions dilutes motion autocorrelation, the authors propose to normalize motion and conditional features independently. This simple yet effective design is highly insightful.
4. **Dual Role of Random Masking**: It both assists the model in learning from data with missing components and prevents the model from "cheating" by ignoring conditional signals.
5. **Downstream Application Extension**: Generated motion sequences can be mapped onto a 2D plane to serve as guidance signals for video generation, showcasing excellent compatibility with the video generation ecosystem.

## Limitations & Future Work

1. **Limitations of the Intermediate Representation**: Currently, it can only handle cases where entire body parts are missing, lacking the fine-grained ability to handle individual missing keypoints.
2. **Noise Introduced by the Motion Translator**: The process of converting the unified representation back to dataset-specific representations introduces extra errors, degrading motion quality.
3. **Long Sequence Generation**: Constrained by GPU memory, long sequences must be generated using zero-shot methods, which limits practical utility.
4. **Music-to-Dance Correlation**: The proportion of music-to-dance data is relatively low, preventing the FID metric from surpassing that of specialized models.
5. **More Flexible Motion Representations**: More flexible motion representation methods need to be explored to mitigate information loss induced by the translator.

## Related Work & Insights

- **FineMoGen**: The direct baseline of LMM, with ArtAttention being upgraded from its SAMI module.
- **MDM / MoMask**: Representative methods in the text-to-motion field; LMM-Large surpasses them in accuracy.
- **ImageBind**: Utilized to align multimodal conditions into a unified feature space.
- **TOMATO**: The reference format for unified motion representation.
- **Insights**: This work provides data-processing experiences and training strategy baselines for future efforts to construct larger-scale motion foundation models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first to systematically construct a "large model" in the field of motion generation, with innovations in data, architecture, and training strategies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 10 tasks, 9 benchmarks, thorough ablation studies, and comprehensive scaling analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with explicit mapping between motivation and solution.
- **Value**: ⭐⭐⭐⭐ Holds significant infrastructural value for the motion generation community; the MotionVerse dataset itself is a major contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive and Reactive 3D Motion Generation via Rectified Flow](../../ICLR2026/human_understanding/unified_multi-modal_interactive_and_reactive_3d_motion_generation_via_rectified_.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [\[ECCV 2024\] Motion Mamba: Efficient and Long Sequence Motion Generation](motion_mamba_efficient_and_long_sequence_motion_generation.md)
- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](../../ICCV2025/human_understanding/genmo_a_generalist_model_for_human_motion.md)

</div>

<!-- RELATED:END -->
