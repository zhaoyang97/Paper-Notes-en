---
title: >-
  [Paper Note] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Multimodal reasoning] Open Vision Reasoner (OVR) employs a two-stage training paradigm—linguistic cold start followed by large-scale multimodal RL—to effectively transfer cognitive behaviors (e.g., backtracking, verification) from language models to visual reasoning. Built on Qwen2.5-VL-7B, OVR achieves 51.8% on MathVision, the first model at this scale to surpass 50%, establishing a new state of the art among same-scale models.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Multimodal reasoning
  - cognitive behavior transfer
  - visual reasoning
  - cold-start fine-tuning
date: 2026-05-08
content_hash: c5fdccb9ccd36381
---

# Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning

**Conference**: NeurIPS 2025  
**arXiv**: [2507.05255](https://arxiv.org/abs/2507.05255)  
**Code**: Open-sourced (model, data, and training dynamics)  
**Area**: Reinforcement Learning  
**Keywords**: Multimodal reasoning, cognitive behavior transfer, reinforcement learning, visual reasoning, cold-start fine-tuning

## TL;DR

Open Vision Reasoner (OVR) employs a two-stage training paradigm—linguistic cold start followed by large-scale multimodal RL—to effectively transfer cognitive behaviors (e.g., backtracking, verification) from language models to visual reasoning. Built on Qwen2.5-VL-7B, OVR achieves 51.8% on MathVision, the first model at this scale to surpass 50%, establishing a new state of the art among same-scale models.

## Background & Motivation

The shift from RLHF to RLVR (Reinforcement Learning from Verifiable Rewards) has endowed LLMs with powerful reasoning capabilities, centered on the acquisition of **cognitive behavior patterns** such as backtracking and subgoal decomposition. The multimodal domain is a natural fit for the RLVR paradigm, as visual facts provide verifiable grounding. However, existing work leaves several core questions unresolved:

**How do linguistic cognitive behaviors transfer to visual reasoning?** Existing approaches either rely on manually constructed behavioral trajectory datasets (complex and unscalable) or adopt RLHF-style learned reward models (susceptible to reward hacking).

**What are the respective roles of cold start and RL?** Prior work lacks systematic analysis of how cognitive behaviors emerge, evolve, and transfer across training stages.

**Insufficient training scale**: Existing open-source RL practice on MLLMs is limited in scale, with the longest RL training spanning only a few hundred steps.

OVR's core motivation is to systematically elucidate the mechanism of cross-modal cognitive behavior transfer on a strong testbed (Qwen2.5-VL-7B) at an unprecedented training scale (2M cold-start samples, 300K RL samples, ~1,000 RL steps).

## Method

### Overall Architecture

OVR adopts the widely used "RL with cold start" paradigm, consisting of two training stages:

- **Stage 1 (Linguistic Cold Start)**: SFT is applied exclusively to the LLM module on large-scale language reasoning data to establish core cognitive behaviors.
- **Stage 2 (Multimodal RL)**: PPO is applied on mixed text and multimodal tasks to promote reasoning generalization and align cognitive patterns to visual contexts.

### Key Designs

1. **Definition of Four Visual Cognitive Behaviors**:  
   Drawing from linguistic cognitive behaviors, the authors define four corresponding visual behaviors:
   - **Visual Reflection**: The model proactively revisits the image upon detecting reasoning inconsistencies, analogous to linguistic "backtracking" (e.g., "Let me look at the image again").
   - **Divide-and-Conquer**: Complex visual problems are decomposed into sub-regions processed sequentially, analogous to "subgoal decomposition."
   - **Visual Verification**: Intermediate conclusions are confirmed against visual evidence, analogous to "verification."
   - **Goal-driven Visual Tracing**: Evidence in the image is located by reasoning backward from the desired conclusion, analogous to "backward induction."

2. **Large-Scale Data Curation**:
   - Cold-start data: ~2M reasoning trajectories distilled from DeepSeek-R1, covering AIME, MATH, Numina-Math, and synthetic logic problems.
   - RL data: ~300K multimodal samples including geometry problems (Geometry3k, GeoQA), visual discrimination (IconQA, ChartQA), visual puzzles (PuzzleVQA), and STEM tasks.
   - Multi-step curation: pretrained model filtering of high-loss noisy samples → rule- and model-assisted removal of undesirable patterns → reweighting for category balance → removal of samples incompatible with the reward function.

3. **PPO + GAE Optimization**:  
   A lightweight PPO algorithm is adopted with Generalized Advantage Estimation (GAE). The advantage estimate is:
   $$\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_{t'} = r_{t'} + \gamma V_\phi(s_{t'+1}) - V_\phi(s_{t'})$$

   The policy update uses the standard PPO clipped objective:
   $$\mathcal{J}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_{\pi_{\text{old}}}\left[\min\left(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

### Loss & Training

- **Reward function**: A minimal rule-based reward—answers are extracted from the model's `\boxed{}` output and awarded 1 for an exact match with the reference, 0 otherwise. No format or style preference rewards are used.
- **Cold-start training**: 5 epochs, batch size 640, sequence length 64k, learning rate $2 \times 10^{-4}$; an aggressive large-batch, high-learning-rate strategy is employed to break inherent model constraints.
- **RL training**: 900 update steps with a sequence length curriculum—24k for the first 300 steps → 32k for steps 300–700 → 48k thereafter. Strict on-policy updates are enforced.
- **Final model**: Uniform averaging of several representative intermediate checkpoints to ensure balanced performance across benchmarks.

## Key Experimental Results

### Main Results: Visual Reasoning Benchmarks

| Model | MathVista | MathVision | MathVerse | DynaMath | WeMath | MMMU-Pro |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B | 69.2 | 25.5 | 41.1 | 31.2 | 53.1 | 47.9 |
| R1-OneVision-7B | 64.1 | 29.9 | 40.0 | - | - | - |
| ReVisual-R1 | 73.1 | 48.8 | 53.6 | 42.0 | - | 52.3 |
| MM-Eureka-7B | 72.6 | 28.1 | 45.4 | 21.8 | - | 46.3 |
| **OVR-7B** | **72.1** | **51.8** | **54.6** | **44.6** | **64.8** | **54.8** |

OVR surpasses the previous SOTA ReVisual-R1 on MathVision (+3.0) and MathVerse (+1.0), and improves MMMU-Pro by +2.5. It is the first model based on Qwen2.5-VL-7B to exceed 50% on MathVision.

### Language Reasoning Benchmarks

| Model | AIME 2024 | AIME 2025 | MATH500 | GPQA | MMLU | MMLU-Pro |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B | 6.7 | 6.7 | 67.4 | 31.8 | 69.6 | 51.7 |
| DeepSeek-R1-Distill-7B | 55.5 | 39.2 | 92.8 | 49.1 | - | - |
| ReVisual-R1 | 53.3 | 43.3 | 89.2 | 47.5 | - | - |
| **OVR-7B** | **63.5** | **52.1** | **95.3** | **49.8** | **77.2** | **67.9** |

OVR outperforms same-scale models on AIME 2024/2025 by more than 10%, achieves 95.3% on MATH500, and approaches GPT-4o-level performance.

### Ablation Study / Cognitive Behavior Analysis

| Training Stage | Visual Reflection Emergence | Backtracking Transfer Rate | Verification Transfer Rate |
|----------|:---:|:---:|:---:|
| Baseline (Qwen2.5-VL-7B) | Low | ~2.5% | ~0% |
| After Cold Start | Significantly increased | ~10% | ~0% |
| After RL | Substantially increased | ~17.3% | ~0% |

| Sequence Length Curriculum | Effect |
|-------------|------|
| 24k (first 300 steps) | Steady growth in reward and sequence length |
| 32k (steps 300–700) | Breaks through plateau, catalyzes a new round of growth |
| 48k (after step 700) | Continued improvement to final performance |

### Key Findings

- **Cognitive behavior transfer emerges very early**: Owing to "mental imagery" in DeepSeek-R1 (e.g., "Let me visualize…"), visual reflection behavior appears in abundance from the very beginning of cold-start training.
- **Cold start memorizes broadly; RL filters selectively**: The cold-start stage indiscriminately memorizes all patterns, while RL acts as a refinement filter, amplifying critical behaviors.
- **Transfer is strategic**: The transfer rate of backtracking grows continuously (2.5%→17.3%), whereas that of verification remains near zero throughout, indicating that transfer prioritizes high-utility behaviors.
- **Perceptual capability degrades then recovers**: Linguistic cold start introduces perceptual degradation, which is effectively restored by multimodal RL.

## Highlights & Insights

- **"Inner vision" in language**: DeepSeek-R1 spontaneously employs mental imagery in mathematical reasoning, which naturally translates into genuine visual interaction in MLLMs.
- **In-depth analysis of training dynamics**: The internal mechanisms of training are revealed across multiple dimensions—behavior emergence, transfer rates, and sequence length curriculum.
- **Large-scale open-source RL practice**: Approximately 1,000 steps of RL training on Qwen2.5-VL-7B represent the largest known open-source multimodal RL experiment.

## Limitations & Future Work

- Scalability is limited when training solely on perceptual tasks with RL—reward signals grow but reasoning complexity does not increase accordingly.
- The cross-modal transfer rate of the Verification behavior remains near zero, suggesting that certain cognitive patterns require more natively visual support.
- The final model requires multi-checkpoint averaging, lacking a single optimal checkpoint.
- Multi-turn or agentic RL frameworks for visual manipulation and imagination remain unexplored.

## Related Work & Insights

- **DeepSeek-R1**: The primary source of inspiration, providing the foundation for cold-start data distillation.
- **Open-Reasoner-Zero**: The basis for the RL training framework.
- **Perception-R1, ReVisual-R1**: Concurrent multimodal RL works; OVR surpasses them in both scale and systematic analysis.
- Key insight: The transfer from language reasoning to visual reasoning is not a simple modality extension, but a structural transmission at the level of cognitive behaviors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically elucidates how linguistic cognitive behaviors transfer to visual reasoning with in-depth analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 8+ visual benchmarks and 6 language benchmarks with detailed training dynamics analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Concepts are clearly articulated; cognitive behavior definitions are presented concisely in tabular form.
- **Value**: ⭐⭐⭐⭐⭐ The largest-scale open-source multimodal RL practice combined with deep mechanistic analysis makes a significant contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)
- [\[NeurIPS 2025\] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [\[AAAI 2026\] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)

</div>

<!-- RELATED:END -->
