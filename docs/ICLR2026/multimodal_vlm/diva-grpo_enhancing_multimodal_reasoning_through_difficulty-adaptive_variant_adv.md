---
title: >-
  [Paper Note] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage
description: >-
  [ICLR 2026][Multimodal VLM][GRPO] This paper proposes DIVA-GRPO, which addresses reward sparsity and advantage vanishing in GRPO training by dynamically assessing question difficulty, adaptively generating semantically consistent variants of varying difficulty, and incorporating difficulty-weighted local-global advantage estimation. The method achieves state-of-the-art multimodal reasoning performance at the 7B model scale.
tags:
  - ICLR 2026
  - Multimodal VLM
  - GRPO
  - reinforcement learning
  - multimodal reasoning
  - difficulty-adaptive
  - advantage vanishing
  - variant augmentation
date: 2026-05-08
content_hash: fb8008950e82258c
---

# DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage

**Conference**: ICLR 2026
**arXiv**: [2603.01106](https://arxiv.org/abs/2603.01106)
**Code**: [Siaaaaaa1/DIVA-GRPO](https://github.com/Siaaaaaa1/DIVA-GRPO)
**Area**: Multimodal VLM
**Keywords**: GRPO, reinforcement learning, multimodal reasoning, difficulty-adaptive, advantage vanishing, variant augmentation

## TL;DR

This paper proposes DIVA-GRPO, which addresses reward sparsity and advantage vanishing in GRPO training by dynamically assessing question difficulty, adaptively generating semantically consistent variants of varying difficulty, and incorporating difficulty-weighted local-global advantage estimation. The method achieves state-of-the-art multimodal reasoning performance at the 7B model scale.

## Background & Motivation

**GRPO is widely adopted for multimodal reasoning**: GRPO enables long-chain reasoning training without a critic model through intra-group relative advantage estimation, and has become the mainstream approach for enhancing MLLM reasoning capabilities.

**Advantage vanishing is the core bottleneck**: When questions are too easy or too hard for the current model, all responses within a group are either entirely correct or entirely incorrect, causing the advantage to collapse to zero and the optimization signal to vanish, severely degrading training efficiency.

**Reward sparsity compounds the problem**: During early training or on difficult questions, only a negligible fraction of reasoning paths receive positive rewards, leading to slow learning due to scarce positive feedback.

**Existing methods each have limitations**: (a) Sample augmentation approaches (e.g., adding prompts, generating variants) do not control difficulty distribution and may exacerbate advantage vanishing; (b) Selective sample utilization discards data, reducing diversity; (c) Indirect reward design may introduce biases misaligned with the final objective.

**Dynamic difficulty shifts are overlooked**: As training progresses and model capability improves, originally moderate questions become easy, causing advantage vanishing to worsen continuously — yet existing methods do not account for this dynamic evolution of difficulty.

**Core Insight**: The key lies in ensuring sufficient variance in the intra-group reward distribution for each question, thereby producing clear optimization signals — which requires dynamically adjusting variant difficulty distribution according to question difficulty.

## Method

### Overall Architecture

DIVA-GRPO consists of three core modules: (1) dynamic difficulty assessment based on historical rollouts; (2) difficulty-adaptive variant generation; and (3) difficulty-weighted local-global advantage balancing with reward-range rescaling. During training, the difficulty of each question is first assessed, variants of appropriate difficulty are then sampled, and the advantage is computed and the policy updated over the expanded space comprising the original question and its variants.

### Key Design 1: Dynamic Difficulty Assessment

- **Function**: Maintains a dynamic difficulty score $D_q \in [D_{\min}, D_{\max}]$ for each training question, updated in real time based on the model's historical performance.
- **Mechanism**: Computes the empirical accuracy $\alpha$ from rollouts and updates difficulty via $D^{\text{new}} = \text{clip}(D^{\text{old}} + \eta \cdot (0.5 - \alpha))$ — difficulty decreases when accuracy is high, increases when accuracy is low, and stabilizes near 50% accuracy.
- **Design Motivation**: Question difficulty is not an intrinsic property but a dynamic quantity relative to current model capability. Recalibrating difficulty each epoch ensures the variant generation strategy always matches the model's current level, preventing advantage vanishing caused by all questions becoming trivially easy in later training stages.

### Key Design 2: Difficulty-Adaptive Variant Generation

- **Function**: Generates semantically consistent variants that preserve the ground-truth answer but differ in difficulty, based on the question's difficulty level.
- **Mechanism**: A three-tier strategy —
    - **Easy questions** ($D_q < D_{\text{mid}}$): Both text and image are perturbed (rotation, noise, blur, etc.) to increase difficulty and produce negative samples.
    - **Medium questions** ($D_q \approx D_{\text{mid}}$): Only textual paraphrase variants are generated, preserving difficulty while increasing surface diversity.
    - **Hard questions** ($D_q > D_{\text{mid}}$): Partial reasoning steps are provided as hints (think-steps) to reduce difficulty and produce positive samples.
- **Design Motivation**: Ensures that each question's variant group simultaneously contains both correct and incorrect responses, maintaining sufficient reward variance within the group and fundamentally resolving advantage vanishing.

### Key Design 3: Difficulty-Weighted Local-Global Advantage Balancing

- **Function**: Computes local (within a single question group) and global (across a question and all its variants) advantages separately, then merges them via batch z-score normalization and difficulty-weighted scaling.
- **Mechanism**: Local and global advantages are first individually normalized via batch-level z-score to eliminate magnitude discrepancies, then difficulty-weighted via $\hat{A} = \exp(k \cdot (D_q^{(i)} - \bar{D}_q) \cdot \text{sgn}(\tilde{A})) \cdot \tilde{A}$ — amplifying the advantage of correct responses and suppressing that of incorrect ones for above-average-difficulty variants, and vice versa.
- **Design Motivation**: (1) Local and global advantages differ in magnitude due to differing sample sizes (global is typically larger); normalization renders them comparable. (2) Difficulty weighting encourages the model to gain larger rewards for correct answers on harder questions, enabling difficulty-adaptive optimization.

### Loss & Training

- The base loss follows the standard GRPO policy gradient objective, with advantages replaced by the difficulty-weighted, normalized values described above.
- Additionally introduces **Reward-Range-Based Advantage Rescaling (RRB)**: $\hat{A}_{\text{range}} = \Delta r_q \cdot \tilde{A}$, where $\Delta r_q = (\max(\mathcal{R}_q) - \min(\mathcal{R}_q)) / R_{\max}$, preventing z-score normalization from amplifying negligible differences when rewards are highly concentrated.
- Base model: Qwen2.5-VL-7B-Instruct; optimizer: AdamW; learning rate: $10^{-6}$; difficulty initialized at $D_q=5$ (range 1–9); $\eta=4$.
- Textual variants and reasoning hints are generated offline by GPT-o3; image perturbations are applied online.

## Key Experimental Results

### Table 1: Main Results on Six Multimodal Mathematical Reasoning Benchmarks

| Model | MathVista | MathVerse | MathVision | OlympiadBench | WeMath | MMK12test | Avg. |
|---|---|---|---|---|---|---|---|
| GPT-4o | 63.8 | 50.2 | 30.4 | 35.0 | 68.8 | 49.9 | 49.68 |
| Qwen2.5-VL-7B (base) | 68.2 | 47.9 | 25.4 | 20.2 | 62.1 | 53.6 | 46.23 |
| Qwen2.5-VL-72B | 74.8 | 57.6 | 38.1 | 40.4 | 72.4 | 70.5 | 59.0 |
| R1-ShareVL-7B | 73.5 | 52.8 | 29.5 | 21.3 | 67.9 | 68.8 | 52.30 |
| MM-Eureka-7B | 71.7 | 50.3 | 26.9 | 20.1 | 66.1 | 64.5 | 49.93 |
| **DIVA-GRPO-7B (Ours)** | **74.2** | **57.6** | **32.1** | **23.1** | **69.3** | **70.2** | **54.58** |

- Achieves state-of-the-art results across all six benchmarks at the 7B scale, with an average score of 54.58.
- Approaches 72B-scale model performance on MathVista, MathVerse, and WeMath.
- Improves over the base Qwen2.5-VL-7B model by an average of **+8.35** points.

### Table 2: Ablation Study Results

| Method | MathVista | MathVerse | MMK12test | Avg. |
|---|---|---|---|---|
| w/o Variant Generation | 70.0 | 53.7 | 61.1 | 61.6 |
| w/o Difficulty-Weighting | 69.9 | 55.7 | 66.5 | 64.0 |
| w/o RRB-Rescaling | 71.5 | 55.2 | 64.7 | 63.8 |
| w/o G-L Balance | 70.8 | 55.4 | 66.0 | 64.1 |
| **Full DIVA-GRPO** | **73.2** | **56.3** | **68.8** | **66.1** |

- Removing any component degrades performance; variant generation has the largest impact (−4.5 avg).
- In terms of training efficiency, the number of steps required to reach optimal performance is reduced by **2.55×**, with an end-to-end speedup of **1.76×**.

## Highlights & Insights

- **Precise problem formulation**: Framing advantage vanishing through the lens of "ensuring sufficient intra-group reward variance" provides a more fundamental solution than the three existing categories of approaches.
- **Closed-loop difficulty adaptation**: Difficulty assessment → variant generation → advantage weighting form a complete closed loop, with difficulty evolving dynamically throughout training.
- **Solid theoretical grounding**: The paper provides a theorem proving that reduced gradient variance accelerates convergence, as well as mathematical analysis showing that optimization signals are strongest when the positive-to-negative sample ratio is approximately 1:1.
- **Significant training efficiency gains**: A 2.55× reduction in required steps and 1.76× end-to-end speedup offer substantial practical value.
- **High generalizability of RRB-Rescaling**: The RRB component can be applied independently to any GRPO framework as a plug-and-play module.

## Limitations & Future Work

- Textual reasoning hints for variants rely on offline generation by GPT-o3, introducing dependence on a closed-source model and additional cost.
- A substantial gap remains on competition-level mathematics (OlympiadBench: 23.1 vs. o1's 68.0), reflecting clear capacity limitations at the 7B scale.
- Image perturbation strategies (rotation, noise, etc.) are relatively simple and may be insufficient for scenarios requiring fine-grained visual understanding.
- Difficulty assessment is based solely on accuracy, lacking discrimination for partially correct responses or cases where the reasoning process is correct but the final answer is wrong.

## Related Work & Insights

- **vs. GRPO/DAPO**: Standard GRPO and DAPO do not account for difficulty adaptation, leading to decaying advantage signals in later training stages; DIVA-GRPO maintains reward variance through variant generation.
- **vs. GSPO**: GSPO introduces semantically consistent variants but does not dynamically adjust difficulty distribution; DIVA-GRPO dynamically matches variant difficulty to the model's current capability.
- **vs. Adora/MM-Eureka**: These methods mitigate the problem via sample selection or indirect rewards, but respectively risk data waste and optimization misalignment.
- **vs. R1-ShareVL**: The closest 7B-scale competitor; DIVA-GRPO demonstrates clear advantages on MathVerse (+4.8) and MMK12test (+1.4).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of difficulty-adaptive variant generation, three-tier strategy, and RRB rescaling is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six benchmarks, detailed ablations, efficiency analysis, and theoretical proofs provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and method motivation is developed in a well-structured, progressive manner.
- Value: ⭐⭐⭐⭐ — Addresses practical pain points in GRPO training; the RRB component is immediately plug-and-play applicable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)

<!-- RELATED:END -->
