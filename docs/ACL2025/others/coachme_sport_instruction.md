---
title: >-
  [Paper Note] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model
description: >-
  [ACL 2025][sports instruction generation] CoachMe is proposed to automatically generate sports-specific coaching instruction texts by comparing the differences (across both temporal and physical dimensions) between the learner's motion and a reference motion, outperforming GPT-4o by 31.6% in figure skating and 58.3% in boxing (according to G-Eval).
tags:
  - "ACL 2025"
  - "sports instruction generation"
  - "pose analysis"
  - "reference comparison"
  - "motion understanding"
  - "graph convolutional networks"
date: 2026-05-08
content_hash: 7fb38f67c80495ac
---

# CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model

**Conference**: ACL 2025  
**arXiv**: [2509.11698](https://arxiv.org/abs/2509.11698)  
**Code**: [https://motionxperts.github.io/](https://motionxperts.github.io/)  
**Area**: Others  
**Keywords**: sports instruction generation, pose analysis, reference comparison, motion understanding, graph convolutional networks  

## TL;DR
CoachMe is proposed to automatically generate sports-specific coaching instruction texts by comparing the differences (across both temporal and physical dimensions) between the learner's motion and a reference motion, outperforming GPT-4o by 31.6% in figure skating and 58.3% in boxing (according to G-Eval).

## Background & Motivation
**Background**: Vision-language models have made progress in motion captioning, but these models focus on general motion description (e.g., "a person raising their left knee") rather than the error-correcting instructions needed in sports training.

**Limitations of Prior Work**: Each sport has unique movement patterns—in skating, knee-shoulder coordination relates to balance and jump execution, while in boxing, the transfer of force from foot to fist determines striking power. General models lack this domain knowledge and cannot provide actionable suggestions for improvement.

**Key Challenge**: Effective coaching instructions require two elements: (1) identifying the specific body parts and time points of errors, and (2) explaining how to improve. Existing VLMs (such as GPT-4o) can only generate generic responses with a coaching tone but lacking critical information.

**Goal**: How to learn domain knowledge from limited sports data and simulate a coach's thought process (locating errors $\rightarrow$ analyzing causes $\rightarrow$ providing improvement plans)?

**Key Insight**: Through reference action comparison—coaches always compare standard actions with student actions to identify problems. The model can also locate areas for improvement by calculating "difference tokens."

**Core Idea**: Decompose the motion comparison into conceptual differences (temporal alignment + RGB differences) and human pose perception (skeleton graph convolution), and fuse difference information from both modalities to generate instruction texts.

## Method

### Overall Architecture
CoachMe includes three modules: (1) Concept Difference Module aligns learners' and reference videos, calculates RGB-level conceptual differences, and identifies error segments; (2) Human Pose Perception extracts local and global motion tokens from skeleton data; (3) Instruct Motion feeds motion tokens and difference tokens into a T5 language model to generate instruction texts.

### Key Designs

1. **Concept Difference Module**:

    - **Function**: Aligns the learner video and the reference video, calculates frame-level differences, and identifies error segments.
    - **Mechanism**: Extracts frame-level concept embeddings using the CARL video encoder, and finds the segment in the learner video that best matches the reference using Dynamic Time Warping (DTW). The conceptual difference is defined as $c = F(x_r) - F(x_l)$. Then, a Transformer Encoder is used to predict the starting and ending positions of the error segments.
    - **Design Motivation**: The first step for a coach is to determine "when the error occurred," and temporal localization is a prerequisite for follow-up analysis. DTW addresses the issue of different lengths between learner and reference videos.

2. **Human Pose Perception**:

    - **Function**: Extracts motion tokens containing spatial and temporal information from skeleton data.
    - **Mechanism**: Extracts 22 joint coordinates $J$ using HybrIK and calculates joint orientations $O_{a,b} = J_a - J_b$. Three sub-modules are involved: PU (understanding poses using graph convolutions on the skeleton graph $G_S$) $\rightarrow$ PE (extracting local motion tokens + learning an attention map $G_A$ to discover critical joint relationships) $\rightarrow$ PA (propagating information on the learned attention map to obtain global motion tokens). The final $Token = T' \oplus T''$ combines local and global tokens.
    - **Design Motivation**: While STA-GCN only propagates information on fixed skeleton graphs, PA additionally learns the implicit relationships between joints that are not originally connected in the skeleton (such as knee-shoulder coordination).

3. **Instruct Motion**:

    - **Function**: Fuses motion and difference information to generate coaching instructions.
    - **Mechanism**: Calculates the motion token difference $Token^{diff}$ between the learner and the reference, concatenates it with the learner's motion tokens, passes it through max pooling and projection layers, and inputs it to T5 (223M) to generate textual instructions $I = LM(Proj(Pool_{max}(Token \oplus Token^{diff})))$. LoRA fine-tuning is used to adapt to limited sports data.
    - **Design Motivation**: Pre-train Basic CoachMe on the large-scale HumanML3D dataset to learn general motion captioning capabilities, and then fine-tune with small-scale sports data to adapt to specific sports.

## Key Experimental Results

### Main Results
Comparison of instruction generation in figure skating (FS) and boxing (BX) (G-Eval 5-point scale):

| Method | FS G-Eval | BX G-Eval | BX BLEU-4 |
|------|-----------|-----------|-----------|
| GPT-4o | 1.39 | 1.39 | 0.0 |
| LLaMa 3.2 | 1.31 | 1.20 | 0.0 |
| CoachMe (best) | **1.83** | **2.20** | **12.3** |
| Basic CoachMe (no ref) | 1.53 | 1.85 | 9.4 |

### Ablation Study

| CoachMe Configuration | FS G-Eval | BX G-Eval |
|-------------|-----------|-----------|
| With Reference + Aligned Segment | **1.83** | **2.20** |
| With Reference + Error Segment | 1.55 | 1.61 |
| With Reference + GT Segment | 1.37 | - |
| Without Reference (Basic) | 1.53 | 1.85 |
| CoachMe (RGB diff) | 1.21-1.57 | 1.44-1.98 |

### Key Findings
- **CoachMe outperforms GPT-4o on G-Eval by 31.6% (FS) and 58.3% (BX)**: Although GPT-4o possesses a coaching tone, it lacks specific error analysis and actionable suggestions for improvement.
- **Reference motion comparison significantly improves instruction quality**: CoachMe with reference shows obvious improvements compared to Basic CoachMe without reference (FS: 1.83 vs 1.53), demonstrating that difference comparison is effective.
- **Skeleton modality is superior to RGB modality for calculating differences**: CoachMe (skeleton) consistently outperforms CoachMe (RGB) because skeleton-level differences are more accurate for motion analysis.
- **Using the entire aligned segment performs better than using only the error segment**: This may be because the full context helps in understanding the origin and outcome of the errors.
- General motion description does not require references (the quality of Basic CoachMe descriptions drops when references are added), but sports coaching requires references—there is a fundamental difference between the two.

## Highlights & Insights
- **The concept of reference comparison is intuitive and effective**: It perfectly replicates the thought process of a human coach—observing standard actions, observing student actions, identifying differences, and providing guidance. This framework can be transferred to other domains requiring comparative analysis (such as surgical training or musical instrument playing).
- **Works with extremely scarce data**: The FS dataset contains only 177 training videos, and BX contains only 163. Effective adaptation is achieved through pre-training on HumanML3D and using LoRA fine-tuning.
- **Visualized attention maps enhance explainability**: The attention maps learned by the GCN can visually demonstrate which joints and joint relationships the model is focusing on.

## Limitations & Future Work
- The dataset scale is very small, and annotation sources are limited (e.g., FS is annotated by only 1 coach, leading to narrow domain coverage).
- Overall G-Eval scores remain relatively low (the best is only 2.20/5), leaving substantial room for improvement in generation quality.
- Evaluated only on single actions such as jumping and punching; complex combined actions and full-match analysis are not covered.
- No comparison with recent specialized sports analysis tools (such as OpenPose + rule engines).

## Related Work & Insights
- **vs. General VLMs such as GPT-4o/LLaMa**: General models lack domain-specific sports knowledge, producing guidance that sounds like a coach but lacks substance. CoachMe bridges this gap through domain fine-tuning and reference comparison.
- **vs. TM2T/MotionGPT**: These models produce motion captions rather than teaching instruction, and they do not perform reference comparisons. The Basic version of CoachMe also outperforms them on HumanML3D motion captioning.
- This is the first work to explicitly model "reference comparison" as the core mechanism for coaching instruction generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reference-based comparative generation of sports instruction is a novel task definition and methodological design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes human evaluation and detailed ablation, but the dataset is too small and only covers two sports.
- Writing Quality: ⭐⭐⭐⭐ The methodology description is clear but involves many symbols; the interpretation of some experimental results could be deeper.
- Value: ⭐⭐⭐⭐ Contributes an interesting task definition and datasets, but its practicality is constrained by data scale and generation quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation](eru-kg_efficient_reference-aligned_unsupervised_keyphrase_generation.md)
- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)
- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[ACL 2025\] Theoretical Guarantees for Minimum Bayes Risk Decoding](theoretical_guarantees_for_minimum_bayes_risk_decoding.md)
- [\[ACL 2025\] Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)

</div>

<!-- RELATED:END -->
