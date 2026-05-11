---
title: >-
  [Paper Note] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video
description: >-
  [CVPR 2026][Human Understanding][Audio-Visual Reasoning] This paper proposes AVATAR, a framework that improves GRPO through two core components—an off-policy training architecture (hierarchical replay buffer) and Tempora…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Audio-Visual Reasoning"
  - "GRPO Improvement"
  - "Off-Policy Reinforcement Learning"
  - "Temporal Advantage Shaping"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 81b69ea5a695a332
---

# AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video

**Conference**: CVPR 2026
**arXiv**: [2508.03100](https://arxiv.org/abs/2508.03100)
**Code**: [https://people-robots.github.io/AVATAR/](https://people-robots.github.io/AVATAR/)
**Area**: Human Understanding / Multimodal Reasoning
**Keywords**: Multimodal Reasoning, Reinforcement Learning, GRPO, Audio-Visual Understanding, Temporal Advantage Shaping

## TL;DR
This paper proposes AVATAR, a framework that addresses three fundamental limitations of GRPO in multimodal video reasoning—data inefficiency, advantage collapse, and uniform credit assignment—via an off-policy training architecture (hierarchical replay buffer) and a Temporal Advantage Shaping (TAS) strategy. AVATAR significantly outperforms standard GRPO on audio-visual understanding benchmarks (OmniBench +3.7, 5× sample efficiency improvement).

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) require alignment across video, audio, and language modalities to support long-horizon reasoning. GRPO has emerged as an effective RL method for enhancing reasoning, demonstrating strong performance in verifiable domains such as mathematics.

**Limitations of Prior Work**: GRPO exhibits three critical limitations in open-domain video tasks: (1) on-policy training causes data inefficiency, particularly severe given the high annotation cost of video data; (2) **advantage collapse**: when all responses within a group receive identical rewards (all correct or all incorrect), advantages vanish to zero, eliminating the learning signal; (3) **uniform credit assignment**: all tokens receive equal reward regardless of their position in the reasoning chain, ignoring the differential importance of distinct reasoning stages.

**Key Challenge**: In video reasoning, the initial planning stage (localizing sound sources) and the final synthesis stage (identifying speakers by combining audio-visual cues) are critical, yet GRPO treats all tokens uniformly, diluting gradient signals.

**Key Insight**: (1) The attention sink phenomenon in Transformers—initial tokens consistently attract attention as planning anchors; (2) final tokens are critical for answer synthesis.

**Core Idea**: An off-policy architecture with a hierarchical replay buffer addresses data efficiency and advantage collapse; U-shaped parabolic weighting (TAS) emphasizes the beginning and end of the reasoning chain.

## Method

### Overall Architecture
A three-stage RL training pipeline: cold-start SFT (S0) → visual reasoning RL (S1) → audio-visual reasoning RL (S2) → audio source localization RL (S3). Each stage employs distinct datasets and reward configurations.

### Key Designs

1. **Off-Policy Architecture**:

    - **Hierarchical Replay Buffer**: Capacity of 10K, partitioned into three fixed-capacity tiers—Easy (25%), Medium (35%), Hard (40%). Tier assignment is determined by the moving-average reward $\bar{R}(q)$ per prompt, with thresholds set via dynamic quantiles.
    - **Resolving Advantage Collapse**: Training groups sampled from the buffer contain both successful and failed trajectories, ensuring intra-group reward diversity → non-zero advantages → sustained gradient updates.
    - **Hinting Mechanism**: When a prompt remains persistently difficult (low $\bar{R}(q)$) and the policy's KL divergence is low, pre-computed hints are injected to guide exploration.
    - Mixed training objective: $\mathcal{J}_{AVATAR} = \mathcal{J}_{on\text{-}policy} + \alpha \cdot \mathcal{J}_{off\text{-}policy}$, where the off-policy term applies importance sampling to correct for policy drift.

2. **Temporal Advantage Shaping (TAS)**:

    - **Function**: Assigns position-dependent weights to tokens at different positions within the reasoning chain.
    - **Mechanism**: U-shaped parabolic weighting function $w_t = 1.0 + \lambda_{TAS} \cdot (2\tilde{t} - 1)^2$, where $\tilde{t} = t/(L-1) \in [0,1]$. Tokens at the beginning and end of the sequence receive weight $1.0 + \lambda_{TAS}$, while middle tokens receive weight 1.0.
    - Shaped advantage: $A_{i,t}^{TAS} = w_{i,t} \cdot A_i$
    - **Design Motivation**: Grounded in the attention sink phenomenon of Transformers and the critical role of the synthesis stage, TAS amplifies learning signals at the planning and synthesis phases of the reasoning chain.

3. **Multi-Source Reward Functions**:

    - **Format reward** $R_{format}$: Validates the `<think>...</think><answer>...</answer>` structure.
    - **Accuracy reward** $R_{acc}$: Provides dense reward via rMAE for numerical tasks.
    - **Self-reward** $R_{self}$: Generates pseudo-correct answers via majority voting within the group for consensus learning.
    - **Step reasoning judgment** $R_{judge}$: A frozen VLM judge (InternVL3-2B) evaluates reasoning quality.

### Loss & Training
Reward configurations differ across stages: S1 uses 0.5× format + 0.5× accuracy; S2 adds self-reward; S3 incorporates reasoning judgment.

## Key Experimental Results

### Main Results (Qwen2.5-Omni Baseline)

| Model | OmniBench | DailyOmni | AV-Counting | WorldSense |
|-------|-----------|-----------|-------------|------------|
| Qwen2.5-Omni (baseline) | 44.2 | 44.0 | 22.3 | 44.2 |
| + GRPO | 45.4 (+1.2) | 44.8 (+0.8) | 22.8 (+0.5) | 45.1 (+0.9) |
| **+ AVATAR** | **49.1 (+4.9)** | **47.0 (+3.0)** | **23.1 (+0.8)** | **46.0 (+1.8)** |

AVATAR vs. GRPO: OmniBench **+3.7**, Video-Holmes +1.9. Sample efficiency: 5× improvement (80% fewer generations to reach target performance).

### Ablation Study

| Configuration | OmniBench | MMVU | Video-Holmes |
|---------------|-----------|------|-------------|
| GRPO (baseline) | 45.4 | 56.6 | 39.0 |
| + Off-Policy | 47.2 | 57.5 | 39.8 |
| + TAS | 47.8 | 57.9 | 40.2 |
| + Both (AVATAR) | **49.1** | **58.2** | **40.5** |

### Key Findings
- Both the off-policy architecture and TAS contribute independently and yield further gains in combination.
- AVATAR proves effective across two distinct base models (Ola-7B and Qwen2.5-Omni), demonstrating model-agnostic applicability.
- Consistent improvements are observed on more challenging benchmarks such as AV-Odyssey and IntentBench.
- The hierarchical buffer design prevents easy samples from being frequently evicted, maintaining training diversity.

## Highlights & Insights
- **Elegant resolution of advantage collapse**: By introducing intra-group reward diversity through the hierarchical replay buffer, the framework fundamentally avoids zero-advantage gradients—a principle broadly applicable to all GRPO-style methods.
- **Simplicity of TAS**: A single U-shaped parabolic function effectively reinforces the critical stages of the reasoning chain without requiring a learned critic network.
- **Three-stage curriculum training**: The progressive curriculum from visual reasoning → audio-visual reasoning → fine-grained localization incrementally increases task difficulty.

## Limitations & Future Work
- The parabolic shape of TAS is hand-crafted and may not be universally optimal—automatic learning of per-token importance warrants investigation.
- The hinting mechanism relies on pre-computed hints, increasing data preparation overhead.
- Importance sampling ratios in the off-policy term may be unstable in practice, requiring clipping for stabilization.
- Evaluation is limited to multiple-choice formats; performance on open-ended video question answering remains unexplored.

## Related Work & Insights
- **vs. Video-R1**: Video-R1 employs temporal contrastive rewards but retains standard GRPO, leaving the advantage collapse problem unresolved.
- **vs. DAPO**: DAPO mitigates uniform groups through modified sampling but still suffers from zero gradients on hard queries; AVATAR addresses this more fundamentally via replay.
- **vs. HumanOmni**: HumanOmni adopts LLM-based judgment rewards but applies uniform credit assignment; AVATAR resolves this through TAS.

## Rating
- Novelty: ⭐⭐⭐⭐ Off-policy GRPO and TAS are well-motivated, though the individual components are relatively independent.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple baselines, multiple benchmarks, 95% confidence intervals, and sample efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ Offers broadly applicable improvements to the GRPO training paradigm.

---

# AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video

**Conference**: CVPR 2026
**arXiv**: [2508.03100](https://arxiv.org/abs/2508.03100)
**Code**: [https://people-robots.github.io/AVATAR/](https://people-robots.github.io/AVATAR/)
**Area**: Human Understanding / Multimodal Reasoning
**Keywords**: Audio-Visual Reasoning, GRPO Improvement, Off-Policy Reinforcement Learning, Temporal Advantage Shaping, Multimodal Large Language Models

## TL;DR
This paper proposes AVATAR, a framework that improves GRPO through two core components—an off-policy training architecture (hierarchical replay buffer) and Temporal Advantage Shaping (TAS, which applies U-shaped weighting to emphasize the beginning and end of the reasoning chain)—to address data inefficiency, advantage collapse, and uniform credit assignment, achieving significant gains over the GRPO baseline on audio-visual reasoning benchmarks.

## Background & Motivation

1. **Background**: MLLMs must align video, audio, and language modalities to support long-horizon reasoning. GRPO has demonstrated potential for enhancing reasoning as an RL method, but exhibits significant limitations in open-domain video settings.
2. **Three Key Problems with GRPO**:
    - **Data inefficiency**: As an on-policy method, experience is discarded after each update, resulting in substantial waste on costly video-annotated data.
    - **Advantage collapse**: When intra-group reward variance collapses (all correct or all incorrect), advantages reduce to zero and the learning signal vanishes.
    - **Uniform credit assignment**: Identical rewards are applied to all tokens in the reasoning chain, ignoring the critical roles of the planning stage (beginning) and the synthesis stage (end).
3. **Key Insight**: Systematically addressing three structural deficiencies of GRPO from an RL algorithm design perspective.
4. **Core Idea**: An off-policy architecture with a hierarchical replay buffer resolves the first two problems; U-shaped positional weighting via TAS resolves the third.

## Method

### Overall Architecture
A three-stage RL training pipeline: Stage 0 SFT cold start → Stage 1 visual reasoning RL → Stage 2 audio-visual reasoning RL → Stage 3 audio source localization RL. Each stage employs distinct datasets and reward configurations.

### Key Designs

1. **Off-Policy Architecture and Hierarchical Replay Buffer**:
    - **Function**: Reuses historical experience to improve sample efficiency and guarantees intra-group reward diversity.
    - **Mechanism**: Maintains a hierarchical buffer $\mathcal{B}$ with capacity 10K, partitioned into three tiers: Easy (25%), Medium (35%), Hard (40%). The moving-average reward $\bar{R}(q)$ per prompt determines tier assignment. The mixed objective is $\mathcal{J}_{AVATAR} = \mathcal{J}_{on} + \alpha \cdot \mathcal{J}_{off}$, where the off-policy term applies an importance sampling ratio $r_i^{off} = \pi_\theta(o_i|q) / \pi_{\theta_{off}}(o_i|q)$ to correct for policy drift.
    - **Design Motivation**: (1) The Hard tier has the largest capacity, ensuring difficult samples are repeatedly trained on; (2) mixing historical successful and failed trajectories into training groups guarantees non-zero reward variance, resolving advantage collapse.

2. **Hinting Mechanism**:
    - **Function**: Injects pre-computed hints to guide exploration when a prompt remains persistently difficult and the policy ceases to explore.
    - **Mechanism**: Monitors $D_{KL}(\pi_\theta \| \pi_\beta)$; when $\bar{R}(q)$ is low and KL is low, a hint is triggered (e.g., "first localize the sound source, then count"), pre-generated by Qwen2.5-VL-72B.
    - **Design Motivation**: Helps the agent escape local optima and keeps training within a challenging yet solvable regime.

3. **Temporal Advantage Shaping (TAS)**:
    - **Function**: Applies position-dependent advantage weights to tokens at different positions in the reasoning chain.
    - **Mechanism**: U-shaped parabolic weighting $w_t = 1.0 + \lambda_{TAS} \cdot (2\tilde{t} - 1)^2$, where $\tilde{t} = t/(L-1)$. Tokens at the beginning and end of the sequence (planning and synthesis stages) receive the highest weight $1+\lambda_{TAS}$, while middle tokens receive weight 1.0. Token-level advantage: $A_{i,t}^{TAS} = w_{i,t} \cdot A_i$.
    - **Design Motivation**: Grounded in the Transformer attention sink phenomenon (initial tokens persistently attract attention) and the critical role of final tokens in answer synthesis, TAS reinforces learning at both ends of the reasoning chain.

4. **Multi-Reward Function Design**:
    - Format reward $R_{format}$: Validates the reasoning format.
    - Accuracy reward $R_{acc}$: Provides dense reward via rMAE.
    - Self-reward $R_{self}$: Majority-voted pseudo-correct answers provide consistency reward.
    - Reasoning quality judgment $R_{judge}$: Frozen InternVL3-2B evaluates the reasoning process.

### Loss & Training
The standard GRPO objective is augmented with an off-policy term, and uniform advantages are replaced by TAS. The three-stage training progressively transitions from visual reasoning to audio-visual reasoning to fine-grained localization.

## Key Experimental Results

### Main Results (Multi-Benchmark Comparison)

| Model | OmniBench | MMVU | Video-Holmes | AV-Odyssey |
|-------|-----------|------|-------------|------------|
| Qwen2.5-Omni (baseline) | 44.2 | - | - | 29.8 |
| + GRPO | 45.4 (+1.2) | - | - | 31.3 (+1.5) |
| + **AVATAR** | **49.1 (+4.9)** | - | - | **32.1 (+2.3)** |
| Ola-7B (baseline) | 45.3 | - | - | 25.6 |
| + GRPO | 46.8 (+1.5) | - | - | 27.0 (+1.4) |
| + **AVATAR** | **47.2 (+1.9)** | - | - | **28.8 (+3.2)** |

AVATAR vs. GRPO on Qwen2.5-Omni: OmniBench +3.7, Video-Holmes +1.9, while requiring **80% fewer generated completions** to reach target performance.

### Ablation Study

| Component | OmniBench | DailyOmni | Notes |
|-----------|-----------|-----------|-------|
| GRPO (baseline) | 45.4 | 44.8 | |
| + Off-policy only | +1.5 | +1.2 | Contribution of off-policy architecture |
| + TAS only | +1.0 | +0.8 | Contribution of temporal shaping |
| + Both (AVATAR) | **+3.7** | **+2.2** | Complementary gains |

### Key Findings
- AVATAR consistently improves performance across two base models (Qwen2.5-Omni and Ola-7B), confirming model-agnostic applicability.
- 5× sample efficiency improvement: target performance is reached with 80% fewer generated completions.
- Gains from the off-policy architecture and TAS are complementary rather than overlapping.
- All improvements are reported with 95% bootstrap confidence intervals, ensuring statistical reliability.

## Highlights & Insights
- **Systematic resolution of GRPO limitations**: The work effectively translates classical RL challenges (off-policy learning, credit assignment, exploration–exploitation) into engineering solutions for MLLM training.
- **Simplicity and effectiveness of U-shaped TAS weighting**: The approach is theoretically aligned with Transformer attention patterns and requires only a single-line modification to the objective, with no additional networks or critic models.
- **Practical utility of the hinting mechanism**: Using a large model (72B) to pre-compute learning guidance for a smaller model represents a practical teacher–student RL strategy.

## Limitations & Future Work
- The U-shape of TAS is fixed and may not be optimal across different tasks or reasoning lengths; adaptive shaping deserves investigation.
- The hinting mechanism depends on an external large model and is inapplicable in fully autonomous learning settings.
- Evaluation is limited to audio-visual QA tasks; generalization to longer-horizon reasoning (e.g., planning, decision-making) remains unknown.
- The replay buffer size (10K) and tier ratios (25/35/40) are manually specified.

## Related Work & Insights
- **vs. Standard GRPO**: AVATAR is a direct extension of GRPO that preserves its simplicity while addressing three structural deficiencies.
- **vs. Video-R1**: Video-R1 uses temporal contrastive rewards; AVATAR optimizes the training algorithm itself, and the two approaches are potentially complementary.
- **vs. DAPO**: DAPO reduces uniform groups via modified sampling, but AVATAR addresses advantage collapse more fundamentally through off-policy replay.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines existing RL techniques (off-policy learning, credit assignment) with meaningful novelty in their application to MLLM training.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, multiple base models, statistical testing, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis is clear, with an explicit three-limitations-to-three-solutions correspondence.
- Value: ⭐⭐⭐⭐ Offers broadly applicable improvements to MLLM RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification](vision-language_attribute_disentanglement_and_reinforcement_for_lifelong_person_.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[CVPR 2026\] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)

</div>

<!-- RELATED:END -->
