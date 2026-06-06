---
title: >-
  [Paper Note] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Visual Perception] This paper argues that current VLM post-training overemphasizes "long-chain reasoning" while neglecting perception bottlenecks. It explicitly decouples post-training into th…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Perception"
  - "Staged Post-training"
  - "Capability-dimension Curriculum"
  - "RLVR"
  - "Visual Mathematical Reasoning"
date: 2026-05-08
content_hash: 451682e1f639ca00
---

# From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.20177](https://arxiv.org/abs/2605.20177)  
**Code**: https://ucsc-vlaa.github.io/VLM-CapCurriculum/ (Project Page)  
**Area**: Multimodal VLM  
**Keywords**: Visual Perception, Staged Post-training, Capability-dimension Curriculum, RLVR, Visual Mathematical Reasoning

## TL;DR
This paper argues that current VLM post-training overemphasizes "long-chain reasoning" while neglecting perception bottlenecks. It explicitly decouples post-training into three independent stages: "Visual Perception → Textual Reasoning → Visual Reasoning," and uses RLVR (rather than caption SFT) to specifically refine perception. This allows Qwen3-VL-8B to achieve relative gains of approximately +5.9% and +1.2% on visual math and perception benchmarks, respectively, while shortening reasoning traces by 20.8%.

## Background & Motivation

**Background**: Following DeepSeek-R1, the dominant paradigm for VLM post-training is "long-chain CoT + RLVR," where tasks such as VQA, geometry, chart understanding, and visual math are optimized jointly in a single stage (e.g., Mixed Rewards in LLaVA-CoT or VLAA-Thinker), hoping the model improves accuracy by "thinking longer."

**Limitations of Prior Work**: The authors performed an error attribution analysis on Qwen3-VL-8B using Claude-Haiku-4.5 across three visual math datasets, finding that **86.9% of incorrect answers originated from errors in the initial visual perception step** rather than subsequent reasoning. Once perception fails, even long-chain reasoning only serves to "justify" the incorrect premise, sometimes repeatedly yielding the same wrong reading despite revisiting the image.

**Key Challenge**: Merged training (mixing perception and reasoning data) assumes all capabilities can be optimized together by the same reward. However, visual perception is a more fundamental capability than "reasoning" and requires specialized objectives and data. If reasoning is prioritized first, models develop a "long-chain + self-persuasion" behavioral pattern without properly aligning visual features, leading to the "perception tax" where MMStar scores drop by 1.6% after reasoning-only training.

**Goal**: (1) Demonstrate that perception must be trained separately with specialized data; (2) Identify the optimal stage sequence; (3) Compare whether caption-SFT or RLVR is more suitable for learning perception; (4) Integrate "staged capability-dimension" training and traditional "increasing difficulty curriculum learning" into a unified framework.

**Key Insight**: Post-training is viewed as sequentially shaping three capabilities: visual perception, textual reasoning, and visual reasoning. If perception is the "scaffolding" for subsequent reasoning, it should be stabilized before visual reasoning begins. Conversely, learning reasoning first may contaminate the learning signals for subsequent perception.

**Core Idea**: Organize post-training using a "capability-dimension curriculum" $\mathcal{D}_{\text{perc}} \rightarrow \mathcal{D}_{\text{text}} \rightarrow \mathcal{D}_{\text{vis}}$, and employ RLVR instead of caption SFT during the perception stage.

## Method

### Overall Architecture
The approach consists of two main components: **(1) Perception Data Synthesis**—reverse-generating "perception-required" QA from 15K DOCCI image-caption pairs and filtering samples using a "Image vs. Caption" contrast; **(2) Staged GRPO Training**—sequentially running three types of data in the order of $\mathcal{D}_{\text{perc}} \rightarrow \mathcal{D}_{\text{text}} \rightarrow \mathcal{D}_{\text{vis}}$ for the same number of epochs, using identical GRPO hyperparameters for each stage with the vision encoder unfrozen throughout. The output is a VLM that is stronger in both visual math and perception benchmarks with shorter reasoning traces.

### Key Designs

1.  **Perception Data Synthesis + Dual-Model Differential Filtering**:
    *   **Function**: Constructing a perception-focused QA training set $\mathcal{D}_{\text{perc}}$ from open-source image-caption corpora that "requires looking at the image to answer," preventing the model from exploiting linguistic priors.
    *   **Mechanism**: First, Qwen2.5-72B is used to generate perception-focused QA pairs $(Q,A)=f_{\text{gen}}(C)$ based on fine-grained captions from DOCCI. For each candidate, two paths are executed: image-only $\hat{A}_{\text{img}}=f_\theta(I,Q)$ and caption-only $\hat{A}_{\text{cap}}=f_\theta(C,Q)$. Only samples satisfying $\mathbb{I}[\hat{A}_{\text{img}}\neq A]\land\mathbb{I}[\hat{A}_{\text{cap}}=A]$ are retained. Finally, Qwen2.5-VL-7B and Qwen2.5-VL-32B base models are used for cross-filtering, taking the intersection.
    *   **Design Motivation**: This ensures "signal purity." If a sample can be answered via caption alone, it tests language rather than vision. Only samples where the "caption can answer but the image-only model cannot" truly expose perception deficiencies, directly addressing the 86.9% error type identified by the authors.

2.  **Capability-Dimension Staged GRPO (Capability-Dimension Curriculum)**:
    *   **Function**: Using the same GRPO loss $\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{x,y}[\frac{1}{G}\sum_i \min(\rho_i A_i, \text{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)] - \beta\, \text{KL}(\pi_\theta\|\pi_{\text{ref}})$ to train on the three data types sequentially, allowing the model to establish "seeing" before overlaying "reasoning."
    *   **Mechanism**: The reward $R(x,y_i)=r_{\text{acc}}+r_{\text{format}}$ is consistent across all stages. Advantages $A_i=(R-\mu_R)/(\sigma_R+\epsilon)$ are normalized within the group. The steps for the three stages are 90 / 375 / 465 (scaled by epoch), strictly aligned with the 930 steps of the merged baseline for fairness. The vision encoder remains trainable in every stage to prevent drift.
    *   **Design Motivation**: Defining the "capability-dimension curriculum" as a **new dimension orthogonal to the traditional difficulty curriculum**. While traditional curricula rank data by easy→hard, this method ranks by perception→reasoning. Section 4.5 verifies these are stackable; combining both exceeds the merged baseline by +4.43%.

3.  **RLVR Instead of Caption-based SFT in the Perception Stage**:
    *   **Function**: Replacing the traditional next-token SFT paradigm (using captions as targets) with reinforcement learning utilizing verifiable rewards during the perception stage to prevent contamination from low-quality captions.
    *   **Mechanism**: Perception QA naturally yields short answers (colors, spatial relations, attributes), allowing exact matching as $r_{\text{acc}}$, which fits seamlessly with on-policy sampling in GRPO. SFT would require using full captions as targets, which are often lower quality than pre-training corpora and impose off-policy supervision at the token level.
    *   **Design Motivation**: Replacing RL with SFT in the perception stage caused Qwen2.5-VL-7B to drop 8.1% on WeMath and Qwen3-VL-8B to drop 1.6%. This suggests SFT can "cannibalize" existing capabilities, while RLVR keeps the model near its own policy, treating perception as reward-driven fine-tuning.

### Loss & Training
GRPO group size $G=5$, max response length 2048. Steps for the three stages are strictly 90 / 375 / 465 based on constant epochs, totaling 930 steps (matching the merged baseline). The vision encoder is frozen in the merged baseline but unfrozen in the staged approach. Training was conducted on 8×H200 using the EasyR1 framework, with Claude-Haiku-4.5 as the judge model.

## Key Experimental Results

### Main Results

Comparison of Qwen3-VL-8B against same-scale reasoning VLMs (Acc %, selected benchmarks):

| Model | MathVista | WeMath | RWQA | MMStar | Overall AVG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-8B (Base) | 72.40 | 50.86 | 70.85 | 70.00 | 62.19 |
| OneThinker-8B | 75.10 | 54.57 | 71.50 | 70.20 | 64.87 |
| **Qwen3-VL-8B (Staged, Ours)** | **75.90** | **56.10** | **74.51** | **73.07** | **65.77** |

Relative to base: WeMath +5.24, RWQA +3.66, MMStar +3.07. Relative to the strongest baseline OneThinker: WeMath +1.53, RWQA +3.01, MMStar +2.87, Overall +0.90.

### Ablation Study: Training Paradigm + Stage Order

| Configuration (Qwen3-VL-8B) | Vis Math AVG | Perception AVG | Overall | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Base | 45.17 | 79.21 | 62.19 | No post-training |
| Merged | 49.64 | 79.71 | 64.67 | Standard baseline |
| Staged 1→2→3 (Perc→Text→Vis) | **51.10** | **80.44** | **65.77** | Default |
| Order 3→2→1 (Vis→Text→Perc) | 37.70 (7B) | 74.17 (7B) | 55.93 (7B) | ~4.6% drop vs. 1→2→3 |
| Perception stage as Caption SFT | -1.6 on WeMath | — | — | 8B; -8.1% on 7B |

### Key Findings
*   **Perception is the bottleneck**: Error attribution showed 86.9% of Qwen3-VL-8B errors were due to perception. Pure reasoning-only post-training triggers a "perception tax" (MMStar -1.6% for 7B), whereas adding perception data recovers performance (RWQA +3.0%).
*   **Stage order is critical**: Placing perception last (3→2→1) caused Qwen2.5-VL-7B's visual math average to drop from 42.3% to 37.7% (lower than the merged baseline), suggesting low-noise perception gradients are overwritten by pre-established long-chain behaviors.
*   **Shorter reasoning $\neq$ Weaker reasoning**: The staged model's response length in stage 3 was 20.8% shorter than the merged baseline (445 vs. 562 tokens), yet visual math accuracy was 1.46 points higher, and Claude-detected perception errors dropped from 805 to 781. "Seeing clearly" naturally streamlines chain-of-thought.
*   **Curriculum dimensions are stackable**: Capability-dimension curricula (staged) and difficulty-dimension curricula (easy-to-hard) are orthogonal. Combining them yielded a +4.43% gain over the merged baseline, outperforming either dimension used in isolation.

## Highlights & Insights
*   **Deconstructing "Long-chain CoT Omni-potence"**: A simple error attribution experiment (86.9% perception errors) challenges the implicit assumption in the VLM community that "longer reasoning is always better." This "diagnose then treat" paradigm is transferable to any scenario where larger models fail to yield proportional gains.
*   **Differential Filtering for Perception Data**: Using the condition "caption can answer ∧ image cannot" is a clever "reverse engineering" technique. It eliminates the need for manual perception labels by letting the model expose its own blind spots.
*   **Capability-dimension as a New Design Degree of Freedom**: While traditional curricula focus only on difficulty, this work introduces "capability" as an orthogonal dimension. It provides a unified perspective for multi-stage post-training (e.g., decoupling RLHF, DPO, and RLVR).

## Limitations & Future Work
*   Running identical epochs/steps for each stage may not be optimal; adaptive stopping or dynamic weight allocation within stages was not explored.
*   Perception data is based entirely on DOCCI natural image captions, limiting coverage of charts, tables, or dense OCR. Gains on benchmarks like DocVQA were relatively small.
*   Validation was limited to Qwen (7B/8B/32B) and InternVL (3-8B / 3.5-8B) series; scaling laws for 70B+ models or non-Decoder architectures remain untested.
*   The judge model (Claude-Haiku-4.5) has a perception-error labeling accuracy of 82.5% based on human auditing, meaning the 86.9% figure may have systematic bias and potentially overestimates the "perception bottleneck."

## Related Work & Insights
*   **vs. OneThinker / WeThink / GThinker**: These assume perception is a "byproduct of pre-training" and only apply RLVR to reasoning. This paper matches or exceeds them by explicitly modeling perception through staged training.
*   **vs. Curr-ReFT / PC-GRPO**: Traditional curricula rank by difficulty; this work ranks by capability (perception → reasoning) and demonstrates they are stackable, meaning traditional curriculum methods can be layered on top of this approach.
*   **vs. VisOnlyQA / NoReGeo**: While prior works "diagnosed" perception as a bottleneck, this paper provides the "prescription" with a concrete data synthesis and training protocol.

## Rating
*   Novelty: ⭐⭐⭐⭐ Clearly defines capability-dimension as a new orthogonal curriculum dimension.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Cross-backbone testing on 8 benchmarks + stage order + SFT/RL comparisons + error attribution.
*   Writing Quality: ⭐⭐⭐⭐ Strong narrative flow (86.9% perc. errors → decoupling → stage order → RLVR > SFT) with effective visualizations.
*   Value: ⭐⭐⭐⭐ A low-cost, high-reward modification for current VLM post-training pipelines.

## Rating
*   Novelty: TBD
*   Experimental Thoroughness: TBD
*   Writing Quality: TBD
*   Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)
- [\[ICML 2026\] Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training](med-scout_curing_mllms_geometric_blindness_in_medical_perception_via_geometry-aw.md)
- [\[ICML 2026\] Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)
- [\[ICML 2026\] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)

</div>

<!-- RELATED:END -->
