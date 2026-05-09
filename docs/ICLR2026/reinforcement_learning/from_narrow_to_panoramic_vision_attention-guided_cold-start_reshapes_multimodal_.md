---
title: >-
  [Paper Note] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning
description: >-
  [ICLR 2026][Reinforcement Learning][Visual Attention] This work discovers that the reasoning performance of multimodal LLMs is highly correlated with the Visual Attention Score (VAS) ($r=0.96$), and proposes the AVAR framework, which improves VAS through three stages—visual-anchored data synthesis, attention-guided training objectives, and visual-anchored reward shaping—achieving an average improvement of 7% across 77 benchmarks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Visual Attention
  - Multimodal Reasoning
  - Cold-Start
  - Attention-Guided Training
  - GRPO
date: 2026-05-08
content_hash: 96ff89978ec32b78
---

# From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning

**Conference**: ICLR 2026  
**arXiv**: [2603.03825](https://arxiv.org/abs/2603.03825)  
**Code**: [https://github.com/lrlbbzl/Qwen-AVAR](https://github.com/lrlbbzl/Qwen-AVAR)  
**Area**: Reinforcement Learning  
**Keywords**: Visual Attention, Multimodal Reasoning, Cold-Start, Attention-Guided Training, GRPO

## TL;DR
This work discovers that the reasoning performance of multimodal LLMs is highly correlated with the Visual Attention Score (VAS) ($r=0.96$), and proposes the AVAR framework, which improves VAS through three stages—visual-anchored data synthesis, attention-guided training objectives, and visual-anchored reward shaping—achieving an average improvement of 7% across 77 benchmarks.

## Background & Motivation

**Background**: Multimodal LLMs (e.g., Qwen2.5-VL) have achieved significant progress on reasoning tasks, yet their reasoning processes are often "visually lazy"—models tend to attend to system tokens rather than visual tokens.

**Limitations of Prior Work**: After multimodal cold-start training, the Visual Attention Score (VAS) of models does not improve and may even decline. VAS measures the proportion of attention that reasoning tokens allocate to visual tokens; a low VAS indicates that the model fails to sufficiently utilize image information during reasoning.

**Key Challenge**: "Lazy Attention Localization"—models learn to exploit cues from text descriptions and system instructions to shortcut reasoning, rather than genuinely attending to key information in the image.

**Goal**: How can models be compelled to increase attention to visual tokens during both the cold-start and RL training stages?

**Key Insight**: The observation that VAS is positively correlated with reasoning performance ($r=0.96$), motivating the direct optimization of VAS as a training signal.

**Core Idea**: Apply explicit supervision at the attention level—enhancing image attention while suppressing system-token attention—to force the model to "look back at the image."

## Method

### Overall Architecture
Three-stage training: (1) cold-start SFT using Visual-Anchored Reflective Data (VARD); (2) incorporation of an Attention-Guided Training Objective (AGTO) during SFT; (3) introduction of a Visual-Anchored Reward Shaping (VARS) mechanism during GRPO reinforcement learning.

### Key Designs

1. **Visual-Anchored Reflective Data Synthesis (VARD)**:

    - **Function**: Synthesizes high-quality training data containing explicit "look back at the image" instructions.
    - **Mechanism**: A three-stage synthesis pipeline—Stage 1 uses Gemini 2.5-Pro to generate high-precision visual descriptions; Stage 2 uses Qwen3-235B to generate reasoning chains with reflection; Stage 3 uses Qwen3-32B to insert visual anchoring instructions (e.g., "look back at the position of the triangle").
    - **Design Motivation**: Guide the model at the data level to explicitly reference images during reasoning, analogous to a visual counterpart of Chain-of-Thought.

2. **Attention-Guided Training Objective (AGTO)**:

    - **Function**: Augments the SFT loss with two attention regularization terms.
    - **Mechanism**: (a) Image enhancement loss $L_{\text{enhance-img}}$: maximizes the log of the average attention from reasoning tokens to image tokens across all layers and heads; (b) System suppression loss $L_{\text{suppress-sys}}$: minimizes the log of the average attention from reasoning tokens to system tokens. Total loss $= L_{\text{LM}} + 0.15 \cdot L_{\text{enhance}} + 0.15 \cdot L_{\text{suppress}}$.
    - **Design Motivation**: Imposing constraints directly on attention maps is more direct and effective than indirect data augmentation. After cold-start training, VAS increases from 7.5 to 13.8.

3. **Visual-Anchored Reward Shaping (VARS)**:

    - **Function**: Incorporates VAS as an additional reward signal during GRPO reinforcement learning.
    - **Mechanism**: For correct answers, reward $= r_{\text{accuracy}} + 0.3 \cdot r_{\text{visual}} + 0.1 \cdot r_{\text{format}}$. $r_{\text{visual}}$ is defined as the ratio of reasoning-token attention to image tokens versus system tokens. Visual rewards are granted only for correct answers to avoid reinforcing erroneous visual attention patterns.
    - **Design Motivation**: SFT can only train the model to imitate attention patterns in training data; the RL stage enables the model to autonomously explore superior visual attention strategies via reward signals.

## Key Experimental Results

### Main Results (Average over 77 Benchmarks)

| Model | MathVision | MMMU | HallusionBench | Overall Avg. |
|-------|-----------|------|---------------|-------------|
| Qwen2.5-VL-7B | 25.2% | 58.1% | 50.7% | 49.1% |
| **AVAR-Thinker** | **37.4%** | **63.8%** | **59.5%** | **56.1%** |
| Gain | +12.2% | +5.7% | +8.8% | +7.0% |

### Ablation Study

| Component | VAS | Performance |
|-----------|-----|-------------|
| Baseline | 7.5 | 49.1% |
| +VARD Data | 10.1 | 51.0% |
| +AGTO Attention Guidance | 13.8 | 52.6% |
| **+VARS Reward Shaping** | **18.9** | **56.1%** |

### Key Findings
- The Pearson correlation between VAS and reasoning performance reaches 0.96, demonstrating that visual attention is a critical bottleneck in multimodal reasoning.
- VARS (RL stage) contributes the most (+3.5%), indicating that RL is better suited than SFT for optimizing attention patterns.
- The most significant gains are observed on mathematical reasoning and hallucination detection, tasks that most heavily rely on visual information.
- The three components are complementary and progressive; removing any one substantially degrades performance.

## Highlights & Insights
- **VAS as an Optimizable Metric**: The abstract question of "whether the model is looking at the image" is quantified into a concrete, differentiable VAS metric that is directly optimized. This "measure first, then optimize" paradigm is broadly applicable.
- **Explicit Constraints at the Attention Level**: Most prior methods optimize only at the input/output level; this work directly imposes constraints on attention maps, providing a more direct mechanism.
- **Three-Stage Progressive Design**: Data (VARD) → Training Objective (AGTO) → Reward (VARS) progressively guides the model to increase visual attention from shallow to deep.

## Limitations & Future Work
- The attention-guided loss applies uniform constraints across all layers and heads, whereas the optimal attention distribution may differ across layers and heads.
- VARD data synthesis relies on Gemini 2.5-Pro and Qwen3-235B, incurring substantial synthesis costs.
- Validation is conducted only on Qwen2.5-VL-7B; the effectiveness on larger models remains unknown.
- The VAS metric assumes that higher visual attention is always beneficial, but certain purely text-based reasoning steps may not require visual attention.

## Related Work & Insights
- **vs. Qwen2.5-VL**: This work uses it as the base model and demonstrates the insufficiency of standard cold-start training.
- **vs. GRPO**: This work augments standard GRPO with a visual attention reward, representing an interesting exploration in reinforcement learning from visual feedback (RLVF).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the VAS–reasoning correlation and attention-guided training constitute genuine novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation across 77 benchmarks is highly convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation chain is clear and the VAS analysis is compelling.
- Value: ⭐⭐⭐⭐⭐ Identifies and addresses a fundamental visual attention deficiency in multimodal LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ICLR 2026\] Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)

<!-- RELATED:END -->
