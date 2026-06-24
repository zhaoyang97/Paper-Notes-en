---
title: >-
  [Paper Note] Human Motion Instruction Tuning
description: >-
  [CVPR 2025][Human Understanding][Human Motion Understanding] LLaMo proposes a multimodal instruction tuning framework that preserves native motion representations (rather than converting them into language tokens), enhancing the model's capability to understand and predict complex human behaviors by simultaneously processing video, motion sequences, and textual inputs.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Human Motion Understanding"
  - "Instruction Tuning"
  - "Multimodal Framework"
  - "Motion Sequence"
  - "Video Analysis"
date: 2026-05-08
content_hash: 8bae970d3eca4d10
---

# Human Motion Instruction Tuning

**Conference**: CVPR 2025  
**arXiv**: [2411.16805](https://arxiv.org/abs/2411.16805)  
**Code**: [https://github.com/ILGLJ/LLaMo](https://github.com/ILGLJ/LLaMo)  
**Area**: Human Understanding / Multimodal VLM  
**Keywords**: Human Motion Understanding, Instruction Tuning, Multimodal Framework, Motion Sequence, Video Analysis

## TL;DR
LLaMo proposes a multimodal instruction tuning framework that preserves native motion representations (rather than converting them into language tokens), enhancing the model's capability to understand and predict complex human behaviors by simultaneously processing video, motion sequences, and textual inputs.

## Background & Motivation

**Background**: Current Multimodal Large Language Models (MLLMs) have made significant progress in handling modalities such as images and text. In the field of human motion understanding, researchers have begun exploring the integration of motion sequences (e.g., skeletal joint sequences) into large language models to enable tasks like motion description generation, action recognition, and behavior prediction.

**Limitations of Prior Work**: Traditional instruction tuning methods typically convert non-linguistic inputs (such as video or motion sequences) into language tokens. This tokenization process loses motion-specific details, such as fine-grained joint trajectories, temporal continuity, and spatial coordination, resulting in insufficient precision when models understand complex human behaviors.

**Key Challenge**: There is a fundamental conflict between the discrete nature of language representation and the continuous nature of motion data. Forcibly mapping continuous motion signals to a discrete token space causes irreversible information loss, which is particularly evident in professional activity analysis (such as sports actions or medical rehabilitation).

**Goal**: To design a multimodal framework capable of processing motion data in its native format, avoiding the information loss caused by tokenization while supporting flexible multimodal instruction interaction.

**Key Insight**: The authors observe that motion data inherently contains rich spatiotemporal structural information. If aligned with the language modality while preserving its native representation, it enables the LLM to better "understand" motion semantics.

**Core Idea**: To replace tokenized motion representations with native motion representations for instruction tuning, building LLaMo—a human motion understanding assistant supporting video, motion, and text modalities simultaneously.

## Method

### Overall Architecture
LLaMo (Large Language and Human Motion Assistant) is a multimodal framework. The input comprises three modalities: video frame sequences, human motion sequences (skeletal joint data), and textual instructions. The framework extracts video and motion features using specialized encoders, maps these features to the LLM's input space via an alignment module, and finally performs unified reasoning and text generation using the large language model.

### Key Designs

1. **Native Motion Representation**:

    - **Function**: Prevents the conversion of motion sequences into discrete language tokens, feeding them directly into the model as continuous vectors.
    - **Mechanism**: Uses a specialized motion encoder to extract spatiotemporal feature representations of motion sequences, preserving the continuity of joint trajectories and spatial relationships. The motion encoder is based on a Transformer architecture, performing feature aggregation across temporal and spatial dimensions on the input skeletal joint sequence $\mathbf{M} \in \mathbb{R}^{T \times J \times 3}$ (where T is the number of frames and J is the number of joints).
    - **Design Motivation**: Discretizing motion signals into tokens in conventional methods leads to the loss of fine-grained motion information. Preserving native representations maintains the spatiotemporal structure more completely, enabling the LLM to reason based on richer motion semantics.

2. **Multimodal Alignment Module**:

    - **Function**: Aligns feature representations from different modalities into a unified semantic space of the LLM.
    - **Mechanism**: Maps video and motion features to a space with the same dimension as the LLM's word embeddings via learnable projection layers. A two-stage training strategy is adopted: the first stage freezes the LLM and only trains the projection layers to complete modal alignment; the second stage performs end-to-end fine-tuning on the entire model.
    - **Design Motivation**: Video and motion data reside in different feature spaces; unified alignment is required for the LLM to simultaneously understand both visual appearance and motion structural information within the same semantic framework.

3. **Video-Motion Co-Analysis**:

    - **Function**: Processes both video and motion data simultaneously to capture complementary information.
    - **Mechanism**: Videos provide appearance, scenes, and contextual information, while motion sequences supply precise body poses and dynamics. The model allows features of the two modalities to interact and fuse within the LLM through a cross-attention mechanism, enabling the reasoning process to comprehensively leverage both appearance and motion cues.
    - **Design Motivation**: Each single modality has its own limitations—videos suffer from information loss during occlusions and in poor lighting, while motion data lacks scene context. Co-analysis compensates for these shortcomings, improving the understanding of complex behaviors.

### Loss & Training
A two-stage training scheme is employed: the pre-training stage uses large-scale motion-text paired data to train the alignment module, predominantly driven by an autoregressive language modeling loss; the instruction-tuning stage performs end-to-end training on high-quality human behavior analysis datasets, using a standard next-token prediction loss. The training data covers multiple domains such as sports analysis and daily behavior recognition.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | LLaMo | Prev. SOTA | Gain |
|------------|------|-------|----------|------|
| Motion Description Generation (HumanML3D) | BLEU-4 | 15.8| 13.2 | +2.6 |
| Motion Description Generation (HumanML3D) | CIDEr | 42.3 | 37.1 | +5.2 |
| Action Recognition (NTU-RGBD) | Top-1 Acc | 89.7% | 86.3% | +3.4% |
| Professional Activity Analysis | F1 Score | 83.5 | 79.2 | +4.3 |

### Ablation Study

| Configuration | BLEU-4 | CIDEr | Description |
|------|--------|-------|------|
| Full LLaMo | 15.8 | 42.3 | Full model (native motion representation) |
| w/ Motion Tokenization | 13.5 | 36.8 | Uses tokenized motion representation, causing a significant drop |
| w/o Video Input | 14.1 | 39.0 | Removes video input, using only motion data |
| w/o Motion Input | 12.7 | 34.5 | Removes motion input, using only video |
| w/o Two-stage Training | 14.3 | 38.7 | Skips pre-training, direct fine-tuning |

### Key Findings
- The gap between native motion representations and tokenized representations is highly significant (a 5.5-point gain in CIDEr), validating the core value of preserving the native form of motion.
- The contribution of motion input is greater than that of video input (removing motion drops performance by 7.8 vs. removing video drops by 3.3), indicating that skeletal data is more critical in motion understanding tasks.
- The two-stage training strategy yields approximately a 3.6-point gain in CIDEr, indicating that modality alignment pre-training is necessary.
- In high-complexity scenarios such as sports analysis and professional activities, LLaMo's advantages are even more pronounced.

## Highlights & Insights
- The design concept of **preserving native representations** is highly generalizable—it is not only applicable to motion sequences but also inspires approaches for interfacing other continuous signals (such as audio and sensor data) with LLMs. Its key value lies in bypassing the information bottleneck of discretization.
- The concept of **video-motion co-analysis** is highly practical. Video and skeletal motion data are often acquired simultaneously in real-world applications (e.g., motion capture systems), making the exploitation of both complementary information sources a logical choice.
- The multimodal instruction tuning paradigm provides a flexible and scalable infrastructure for human-centric AI systems.

## Limitations & Future Work
- The LLaMo GitHub repository currently lacks substantial content (containing only a README), indicating insufficient open-source completeness and making replication difficult.
- The acquisition of motion data relies on skeleton extraction or motion-capture equipment, limiting its applicability in in-the-wild scenarios.
- The capability to distinguish fine-grained motion differences (such as different styles of the same action) is not fully validated.
- Future work could explore extending the framework to motion generation tasks (text-to-motion), establishing a bi-directional understanding-generation capability.

## Related Work & Insights
- **vs MotionGPT**: MotionGPT converts motion into discrete tokens before feeding them to the LLM, whereas LLaMo preserves native representations. LLaMo is superior in preserving motion details, while MotionGPT's tokenization scheme is more flexible for motion generation.
- **vs Video-LLaVA**: Video-LLaVA focuses on video understanding, whereas LLaMo additionally introduces the motion skeletal modality, making it more specialized in human behavior analysis.
- This work offers a valuable technical pathway for applying LLMs to human-centric domains such as motion science, sports analysis, and rehabilitation medicine.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of preserving native motion representations is inspiring, though the overall architecture is a natural extension of the LLaVA-series frameworks.
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple tasks but lacks detailed comparison with more recent baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and comprehensive description of the methodology.
- Value: ⭐⭐⭐⭐ Paves a practical path for the multimodal direction of human motion understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Facial Affective Behavior Analysis with Instruction Tuning](../../ECCV2024/human_understanding/facial_affective_behavior_analysis_with_instruction_tuning.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2025\] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)
- [\[CVPR 2025\] Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic](stochastic_human_motion_prediction_with_memory_of_action_transition_and_action_c.md)

</div>

<!-- RELATED:END -->
