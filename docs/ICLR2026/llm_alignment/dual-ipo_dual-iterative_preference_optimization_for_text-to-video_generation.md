---
title: >-
  [Paper Note] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation
description: >-
  [ICLR 2026][LLM Alignment][Preference Optimization] This paper proposes the Dual-IPO framework, which performs multi-round bidirectional iterative optimization between a reward model and a video generation model. Without large-scale human annotation, the approach continuously improves text-to-video generation quality and human preference alignment, enabling a 2B model to surpass a 5B model.
tags:
  - ICLR 2026
  - LLM Alignment
  - Preference Optimization
  - Video Generation
  - Reward Model
  - DPO
  - Iterative Training
date: 2026-05-08
content_hash: c012f1fb5c8b9c49
---

# Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation

**Conference**: ICLR 2026
**arXiv**: [2502.02088](https://arxiv.org/abs/2502.02088)
**Code**: [https://github.com/SAIS-FUXI/IPO](https://github.com/SAIS-FUXI/IPO)
**Area**: Diffusion Models / Video Generation
**Keywords**: Preference Optimization, Video Generation, Reward Model, DPO, Iterative Training

## TL;DR

This paper proposes the Dual-IPO framework, which performs multi-round bidirectional iterative optimization between a reward model and a video generation model. Without large-scale human annotation, the approach continuously improves text-to-video generation quality and human preference alignment, enabling a 2B model to surpass a 5B model.

## Background & Motivation

**Limitations of State of the Field**: Despite significant advances driven by the DiT architecture, existing video generation models still fall short of user expectations in subject consistency, motion smoothness, and aesthetic quality.

**Data Bottleneck in Preference Learning**: Post-training methods such as DPO and KTO require large volumes of human-annotated preference data, making dataset construction prohibitively expensive.

**Distribution Mismatch of External Reward Models**: Existing general-purpose reward models (e.g., VideoScore, VideoAlign) exhibit significant distribution shift across different video generation models, leading to unreliable reward signals.

**Overfitting on Static Preference Data**: Training on a fixed offline preference dataset tends to cause model overfitting or even collapse.

**Reward Signals Must Co-evolve with the Model**: As training progresses, generation artifacts become increasingly subtle, causing fixed reward models to introduce biased signals.

**Underexplored Potential of Smaller Models**: With effective post-training strategies, smaller models can theoretically approach or even surpass the performance of larger ones.

## Method

### Overall Architecture

Dual-IPO consists of two alternately optimized modules: **(1) Self-Refined Preference Optimization (SRPO)**, which iteratively improves the reward model, and **(2) Iterative Alignment Training**, which optimizes the video generation model based on reward feedback. The two modules complement and reinforce each other across multiple rounds of iteration.

### Key Design 1: Self-Refined Preference Optimization (SRPO)

- **CoT-Guided Annotation**: A VLM is employed for structured reasoning, unlocking reward modeling capability from a small amount of Chain-of-Thought annotated data.
- **Voting-Based Self-Consistency Mechanism**: Multi-path reasoning combined with answer-frequency aggregation automatically constructs stable pseudo-preference labels.
- **Preference Certainty Estimator (PCE)**: Based on the implicit reward formulation of DPO, sample-level confidence is defined as:

$$\text{PCE}(y_w | x) = \mathbb{P}_\theta(R(y_w | x) > Q)$$

where $R(y|x) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ and $Q$ is the average reward over the dataset. Only pseudo-labels with PCE > 0.5 are retained.

### Key Design 2: Iterative Alignment Training

- Both **Diffusion-DPO** (pairwise preference) and **Diffusion-KTO** (pointwise preference) alignment strategies are supported.
- An auxiliary NLL regularization term is incorporated into the DPO loss, using generated samples and real videos respectively to prevent overfitting:

$$\mathcal{L}_{\text{total}}^{\text{DPO}} = L_{\text{dpo}} + \lambda_1 \cdot \mathbb{E}[-\log p_\theta(y^w|x)] + \lambda_2 \cdot \mathbb{E}_{\mathcal{D}^{\text{real}}}[-\log p_\theta(y|x)]$$

- Each iteration follows a closed loop: generate new videos → score with the reward model → construct preference pairs → update the generation model.

### Loss & Training

The final SRPO loss incorporates PCE as a weighting factor into the DPO objective:

$$\mathcal{L}_{\text{SRPO}}(\theta) = \mathbb{E}_{x,y_w,y_l}[\text{PCE}(y_w|x) \cdot \mathcal{L}_{\text{DPO}}(\theta)]$$

## Key Experimental Results

| Model | Total Score | Quality Score | Semantic Score |
|-------|-------------|---------------|----------------|
| CogVideoX-2B (baseline) | 80.91 | 82.18 | 75.83 |
| CogVideoX-2B + IPO-3 rounds | 82.74 | 83.92 | 78.00 |
| CogVideoX-5B (baseline) | 81.61 | 82.75 | 77.04 |
| CogVideoX-5B + IPO-3 rounds | 84.63 | 85.40 | 81.54 |
| Wan-1.3B (baseline) | 84.26 | 85.30 | 80.09 |
| Wan-1.3B + IPO-3 rounds | 86.28 | 86.38 | 85.87 |

| Reward Model | Human Preference Accuracy | VBench Gain |
|--------------|--------------------------|-------------|
| VideoScore | 63.58% | 80.87 (degraded) |
| VideoAlign | 65.21% | 81.27 |
| VisionReward | 68.44% | 81.31 |
| **Ours** | **81.33%** | **81.54** |

**Highlights**: CogVideoX-2B with Dual-IPO surpasses the 5B baseline on VBench (82.74 vs. 81.61); Wan-1.3B reaches 88.32 after 5 iterations, exceeding Sora (84.28).

## Highlights & Insights

1. **Dual Iterative Closed-Loop Design**: The reward model and the generation model mutually promote each other, mitigating the distribution drift caused by static reward signals.
2. **Data Efficiency**: Only 6,000 human-annotated pairs are required for initialization, far fewer than the training data needed by other reward models.
3. **Flexible Preference Strategies**: Both DPO and KTO are supported, accommodating different data formats.
4. **Cross-Architecture Generalization**: The method is effective across both CogVideoX (cross-attention DiT) and Wan (MMDiT) architectures.
5. **"Smaller Model Beats Larger Model"**: The results validate the substantial potential of post-training strategies.

## Limitations & Future Work

1. Training costs remain high (128 GPUs × approximately two weeks per round), making rapid iteration impractical.
2. The PCE threshold (0.5) lacks theoretical justification and may require scenario-specific tuning.
3. The reward model relies on VLMs (VILA 13B/40B), introducing additional computational and storage overhead.
4. Finer-grained quality dimensions (e.g., physical plausibility, causal consistency) are not explored.
5. Performance gains diminish as the number of iterations increases, and the convergence behavior is not thoroughly analyzed.

## Related Work & Insights

- **Diffusion-DPO** (Wallace et al.): Extends DPO to diffusion models; this paper builds upon it by introducing iterative optimization.
- **InstructVideo**: Proposes temporally decayed rewards, but with limited gains; the iterative approach proposed here achieves substantially larger improvements.
- **RLHF for LLM**: The scoring-and-alignment paradigm from the LLM domain is adopted, with the innovative addition of self-iterative reward model refinement.
- Insight: The quality of the reward model is the bottleneck of preference optimization; the self-training paradigm may generalize to other domains such as image generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual iterative closed-loop design and the PCE mechanism are genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-architecture, multi-scale, and multi-round ablation studies are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear and the mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ — The work provides strong practical guidance for post-training of video generation models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ICLR 2026\] Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)
- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[CVPR 2026\] GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering](../../CVPR2026/llm_alignment/glyphprinter_region-grouped_direct_preference_optimization_for_glyph-accurate_vi.md)

<!-- RELATED:END -->
