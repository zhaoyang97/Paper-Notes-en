---
title: >-
  [Paper Note] ReMoT: Reinforcement Learning with Motion Contrast Triplets
description: >-
  [CVPR 2026][Autonomous Driving][Motion Contrast Triplets] This paper proposes ReMoT, a unified training paradigm that constructs the large-scale ReMoT-16K motion contrast triplet dataset via a rule-driven multi-expert collaborative pipeline, and combines GRPO reinforcement learning with a logical consistency reward and length regularization to systematically address the fundamental deficiencies of VLMs in spatiotemporal consistency reasoning, achieving a 25.1% performance leap on spatiotemporal reasoning tasks.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Motion Contrast Triplets
  - GRPO
  - Spatiotemporal Reasoning
  - VLM
  - Data Construction
date: 2026-05-08
content_hash: 79ae269f85e1a3a1
---

# ReMoT: Reinforcement Learning with Motion Contrast Triplets

**Conference**: CVPR 2026
**arXiv**: [2603.00461](https://arxiv.org/abs/2603.00461)
**Code**: None
**Area**: Autonomous Driving / Vision-Language Models
**Keywords**: Motion Contrast Triplets, Spatiotemporal Reasoning, GRPO, VLM, Reinforcement Learning

## TL;DR

This paper proposes ReMoT, a unified training paradigm that automatically constructs a 16.5K motion contrast triplet dataset (ReMoT-16K) via a rule-driven multi-expert collaborative pipeline, and combines GRPO reinforcement learning with a composite reward (logical consistency + length regularization) to systematically address the fundamental deficiencies of VLMs in spatiotemporal consistency reasoning, achieving a 25.1% performance improvement.

## Background & Motivation

**State of the Field**: Vision-language models (VLMs) such as GPT-4o, Claude, Gemini, and Qwen have evolved into general-purpose perception systems, demonstrating strong performance in static image understanding and semantic alignment, and have been deployed in critical domains including AIGC, embodied intelligence, and autonomous driving.

**Limitations of Prior Work**: (1) Current mainstream VLMs exhibit fundamental deficiencies in spatiotemporal consistency reasoning—frequently confusing camera rotation with object motion, misjudging gripper states, and incorrectly inferring motion direction; (2) existing training data predominantly consists of static image-text pairs, lacking explicit modeling of fine-grained motion attributes; (3) approaches such as architectural modifications or data augmentation provide only sporadic patches and cannot systematically address the problem.

**Root Cause**: VLMs excel at visual-semantic alignment but lack deep understanding of spatial-physical regularities, while existing methods address data, training, and evaluation independently without a unified framework.

**Paper Goals**: To systematically address VLM spatiotemporal reasoning deficiencies across three dimensions: data construction, training optimization, and evaluation benchmarks.

**Starting Point**: (1) Automatically constructing motion contrast triplets from video meta-annotations (camera pose matrices, robot action logs); (2) replacing SFT with GRPO for policy learning optimization; (3) designing a composite reward incorporating logical consistency verification.

**Core Idea**: Formalizing motion understanding as structured learning over contrast triplets, and achieving systematic improvement in VLM spatiotemporal reasoning through rule-driven data construction and GRPO optimization.

## Method

### Overall Architecture

ReMoT comprises three core components: (1) **ReMoT-16K Data Construction**: a multi-expert collaborative pipeline that automatically generates 16.5K motion contrast triplets from video meta-annotations; (2) **Training Optimization**: systematic exploration of SFT, GRPO, and hybrid strategies (sequential SFT→GRPO, alternating SFT↔GRPO); (3) **Evaluation Benchmark**: construction of ReMoT-16K-Test, containing 600 evaluation triplets and 1,776 questions covering navigation, robotic manipulation, and simulation game scenarios.

### Key Designs

1. **Multi-Expert Collaborative Data Construction Pipeline**:
    - Function: Automatically generates large-scale, high-quality motion contrast triplets $(I_{anchor}, I_{pos}, I_{neg})$ from video meta-annotations.
    - Motion Estimation Expert: Domain-specific extractors that compute camera rotation from $SE(3)$ pose matrices, extract end-effector trajectories from robot telemetry, etc., and output composite motion attributes $m$.
    - Triplet Construction Expert: (a) Positive sample selection—filters perceptually salient and coherent transitions via attribute thresholds $\mathcal{T}_m$ (e.g., camera rotation in $[10°, 50°]$); (b) Negative sample generation—synthesizes reversed motion via attribute-conditioned generation $\mathcal{T}_{geo}$, or retrieves visually similar but attribute-conflicting frames via retrieval $\mathcal{R}$.
    - VQA Formulation Expert: Designs multi-perspective reasoning chain questions for each triplet, including multiple-choice, true/false, fill-in-the-blank, and comparative reasoning formats.
    - Design Motivation: Directly using VLMs to generate data results in 55% format errors at high cost, yielding only 632 valid triplets, whereas the multi-expert pipeline produces 16.5K high-quality triplets.

2. **GRPO with Composite Reward Design**:
    - Function: Optimizes VLM motion reasoning capability via reinforcement learning, replacing SFT which offers limited effectiveness.
    - Core Algorithm: Adopts GRPO (Group Relative Policy Optimization), sampling $G$ responses for a given query $q$ and computing group-normalized advantages $\hat{A}_i = \frac{R_i - \bar{R}}{\sigma(\{R_j\})}$.
    - CoT Length Regularization: $R_{length}(o_i) = -\max(0, |o_i^{think}| - L_{target})$, suppressing excessively long reasoning chains.
    - Logical Consistency Reward: Detects logical contradictions in responses (e.g., transitivity violations $L_1 < L_2, L_2 < L_3, L_3 < L_1$), assigning $+1/-1/0$ rewards.
    - Composite Reward: $R_i = R_{task} + \lambda_1 \cdot R_{logic} + \lambda_2 \cdot R_{length}$, with weight ratio 3.5:3.5:1.3:1.7.
    - Design Motivation: Analysis reveals that 31.4% of errors stem from logical inconsistency; the explicit logical reward improves logical correctness from 46.6% to 99.3%.

3. **Hybrid Optimization Strategy**:
    - Function: Explores the optimal combination of SFT and GRPO.
    - Sequential Hybrid (SFT→GRPO): SFT provides a stable initialization before switching to GRPO fine-tuning.
    - Alternating Hybrid (SFT↔GRPO): SFT and GRPO steps alternate every few updates, controlled by $(t \bmod (K_{SFT}+K_{GRPO})) < K_{SFT}$.
    - Design Motivation: The alternating strategy allows language alignment and reward alignment to co-evolve, avoiding pattern forgetting.

### Loss & Training

- SFT phase: Cross-entropy loss computed only on tokens within `<answer>` tags.
- GRPO phase: Standard PPO objective with KL regularization (coefficient 0.01), batch size 16, 4 rollouts/sample.
- Base model: Qwen3-VL-4B-Thinking, retaining its built-in CoT reasoning capability.
- Training configuration: 8×A800 GPUs, mixed precision, 2 epochs.

## Key Experimental Results

### Main Results (ReMoT-16K-Test Benchmark)

| Model | Overall Acc. | Partial Acc. | Navigation (Ov.) | Manipulation (Ov.) | Composite Manipulation (Ov.) |
|-------|-------------|-------------|-----------------|-------------------|------------------------------|
| Qwen2.5-VL-7B | 5.1 | 25.4 | 4.8 | 4.0 | 0.0 |
| Qwen3-VL-CoT-4B (Baseline) | 20.7 | 38.9 | 2.4 | 15.3 | 4.8 |
| InternVL3-8B | 12.2 | 28.9 | 2.8 | 1.6 | 0.0 |
| GRPO | 33.6 | 61.6 | 27.0 | 54.5 | 61.3 |
| SFT→GRPO | 35.0 | 63.3 | 26.6 | 57.3 | 62.9 |
| **SFT↔GRPO (Ours)** | **38.0** | **64.0** | 21.4 | **68.6** | **69.4** |

### Ablation Study (Training Strategy and Data Composition)

| Configuration | Overall Acc. | Partial Acc. |
|--------------|-------------|-------------|
| No training (baseline) | 20.7 | 38.9 |
| Manipulation data only | 23.9 | 46.7 |
| + Navigation data | 32.4 | 57.6 |
| + Simulation data (full) | **38.0** | **64.0** |

| Logical Reward Ablation | Overall | Partial | Logical Correctness |
|------------------------|---------|---------|---------------------|
| Base model | 16.2 | 39.6 | 46.6% |
| GRPO w/o logical reward | 68.6 | 77.3 | 98.6% |
| GRPO w/ logical reward | **78.0** | **81.3** | **99.3%** |

### Key Findings

- The alternating SFT↔GRPO strategy achieves the best overall performance (38.0% Overall), representing a 25.1% relative improvement over the base model.
- ReMoT with 4B parameters outperforms Qwen3-VL-30B-CoT (7.5× larger) on spatiotemporal benchmarks (VLM2: 70.0 vs. 68.2, VSI: 58.8 vs. 56.1).
- The multi-expert pipeline data exhibits smooth scaling behavior, whereas VLM-generated data shows instability and a lower performance ceiling (~0.49 vs. 0.66).
- Performance on general multimodal benchmarks is maintained or improved, demonstrating that enhanced spatiotemporal reasoning does not cause catastrophic forgetting.

## Highlights & Insights

- **Systematic approach**: This is the first work to address VLM spatiotemporal reasoning deficiencies from a unified data/training/evaluation perspective, rather than through isolated patches.
- **Efficient data construction**: The rule-driven pipeline outperforms VLM-generated data by two orders of magnitude in scale (16.5K vs. 632) while achieving higher quality.
- **Logical consistency reward**: The paper identifies and resolves the critical finding that 31.4% of errors originate from logical contradictions; the logical reward improves accuracy by 10.6%.
- **Small model, large capability**: The 4B model surpasses the 30B model and GPT-4o through precise data curation and RL training, validating that data quality and training paradigm matter more than model scale.

## Limitations & Future Work

- Navigation task performance degrades under alternating training (Overall 21.4 vs. 27.0 for GRPO), suggesting potential optimization conflicts across tasks.
- Data construction relies on structured meta-annotations (pose matrices, etc.) and is not applicable to videos lacking such annotations.
- Validation is limited to the 4B model; whether there is a performance ceiling for larger models (7B+) remains unexplored.
- The evaluation benchmark is relatively small in scale (600 triplets), and scenario diversity can be further expanded.

## Related Work & Insights

- **GRPO** (Shao et al.): ReMoT validates the superiority of GRPO over SFT for visual reasoning tasks and introduces logical consistency reward as a novel contribution.
- **SimCLR / Contrastive Learning**: The design of motion contrast triplets draws inspiration from the core principles of contrastive learning.
- **Qwen3-VL**: As one of the strongest open-source VLMs, its Thinking mode provides high-quality initialization for RL training.
- **Insights**: The paradigm of rule-driven data construction combined with RL optimization is generalizable to repairing other capability gaps in VLMs.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

---

# ReMoT: Reinforcement Learning with Motion Contrast Triplets

**Conference**: CVPR 2026
**arXiv**: [2603.00461](https://arxiv.org/abs/2603.00461)
**Code**: None
**Area**: Vision-Language Models / Spatiotemporal Reasoning
**Keywords**: Motion Contrast Triplets, GRPO, Spatiotemporal Reasoning, VLM, Data Construction

## TL;DR

This paper proposes ReMoT, a unified training paradigm that constructs the large-scale ReMoT-16K motion contrast triplet dataset via a rule-driven multi-expert collaborative pipeline, and combines GRPO reinforcement learning with a logical consistency reward and length regularization to systematically address the fundamental deficiencies of VLMs in spatiotemporal consistency reasoning, achieving a 25.1% performance leap on spatiotemporal reasoning tasks.

## Background & Motivation

**State of the Field**: VLMs (e.g., GPT-4o, Claude-Sonnet-4.5, Gemini-2.5-Pro) have become general-purpose perception systems, yet in critical domains involving physical world interaction (autonomous driving, embodied intelligence, robotic manipulation), models must go beyond static single-frame perception to perform spatiotemporal consistency reasoning.

**Limitations of Prior Work**: (1) State-of-the-art VLMs frequently confuse camera rotation with object motion, misjudge gripper states, and incorrectly infer character motion direction—even GPT-4o and Qwen3-VL struggle to correctly reason about cross-frame physical changes. (2) Existing training data is predominantly composed of static image-text pairs, lacking explicit modeling of fine-grained motion attributes. (3) Prior remediation approaches (architectural modifications, data augmentation) are fragmented local patches, lacking a systematic solution spanning the data-training-evaluation pipeline.

**Root Cause**: VLMs are proficient at visual-semantic alignment but exhibit systematic deficiencies in spatiotemporal consistency—capable of recognizing "what" but unable to correctly reason about "how things change."

**Paper Goals**: To systematically enhance VLMs' fine-grained spatiotemporal reasoning capability across three dimensions: data construction, training paradigm, and evaluation benchmark.

**Starting Point**: Motion contrast triplets force models to learn fine-grained motion discrimination rather than relying on surface-level visual patterns; GRPO replaces SFT to improve reasoning consistency.

**Core Idea**: Rule-driven motion contrast data + GRPO reinforcement learning = systematic repair of VLM spatiotemporal reasoning deficiencies.

## Method

### Overall Architecture

ReMoT is organized along three dimensions: (1) **Data**: a multi-expert collaborative pipeline constructs the ReMoT-16K motion contrast triplet dataset from video meta-annotations; (2) **Training**: systematic comparison of SFT, GRPO, and hybrid strategies (sequential/alternating), combined with a composite reward design; (3) **Evaluation**: construction of the ReMoT-16k-Test benchmark, comprising 600 evaluation triplets and 1,776 questions.

### Key Designs

1. **Motion Contrast Triplet Construction (Multi-Expert Collaborative Pipeline)**: Each triplet $(I_{anchor}, I_{pos}, I_{neg})$ consists of an anchor-positive pair exhibiting a specific motion attribute $m$, and an anchor-negative pair that is visually similar but violates that attribute. The pipeline comprises three expert types:
    - *Motion Estimation Expert*: $g: (I_t, I_{t'}, \mathcal{A}) \to m$, extracting motion attributes from structured meta-annotations (e.g., $SE(3)$ pose matrices, robot telemetry data).
    - *Triplet Construction Expert*: Selects salient positive pairs $\phi(I_t, I_{t'}, m)$ via attribute thresholds (e.g., camera rotation in $[10°, 50°]$), then generates hard negatives $\mathcal{N}(I_{anchor}, I_{pos}, m)$ via geometric synthesis or attribute-conflicting retrieval.
    - *VQA Generation Expert*: Designs multi-perspective reasoning chain questions for each triplet, covering multiple-choice, true/false, fill-in-the-blank, and comparative reasoning formats.

2. **GRPO Training with Composite Reward Design**: Using Qwen3-VL-4B-Thinking as the base model with GRPO optimization. Group-normalized advantages $\hat{A}_i = \frac{R_i - \bar{R}}{\sigma(\{R_j\})}$ are computed over $G$ sampled responses. The composite reward $R_i = R_{task} + \lambda_1 R_{logic} + \lambda_2 R_{length}$ comprises:
    - *CoT Length Regularization*: $R_{length}(o_i) = -\max(0, |o_i^{think}| - L_{target})$, suppressing redundant reasoning chains.
    - *Logical Consistency Reward*: Verifies transitivity relations among answers (e.g., $L_1 < L_2, L_2 < L_3$ but $L_3 < L_1$ constitutes a contradiction), with $R_{logic} \in \{-1, 0, +1\}$.
    - Reward weight ratio $3.5:3.5:1.3:1.7$ (format : accuracy : conciseness : logical consistency).

3. **Hybrid Optimization Strategy**: In addition to pure SFT and pure GRPO, two hybrid schemes are explored:
    - *Sequential Hybrid (SFT→GRPO)*: SFT first provides stable initialization, then switches to GRPO refinement.
    - *Alternating Hybrid (SFT↔GRPO)*: SFT and GRPO steps alternate periodically, allowing language alignment and reward alignment to co-evolve.

### Loss & Training

The SFT phase uses cross-entropy loss computed only on tokens within `<answer>` tags: $\mathcal{L}_{SFT} = -\sum_{u \in \text{<answer>}} \log \pi_\theta(y_u | q)$. The GRPO phase uses the standard PPO objective with KL regularization (coefficient 0.01). Training runs for 2 epochs on 8×A800 GPUs with mixed precision, batch size 16, and 4 rollouts per sample.

## Key Experimental Results

### Main Results (ReMoT-16k-Test Benchmark)

| Model | Overall Acc. | Partial Acc. | Navigation | Manipulation | Perception |
|-------|-------------|-------------|-----------|-------------|-----------|
| Qwen2.5-VL-7B | 5.1 | 25.4 | 4.8 | 4.0 | 23.9 |
| Qwen3-VL-CoT-4B (Baseline) | 20.7 | 38.9 | 2.4 | 15.3 | 35.8 |
| InternVL3-8B | 12.2 | 28.9 | 2.8 | 1.6 | 30.6 |
| LLaVA-One-Vision | 9.7 | 27.9 | 2.0 | 10.5 | 32.9 |
| GRPO (Ours) | 33.6 | 61.6 | 27.0 | 54.5 | 44.3 |
| SFT→GRPO (Ours) | 35.0 | 63.3 | 26.6 | 57.3 | 35.8 |
| **SFT↔GRPO (Ours)** | **38.0** | **64.0** | 21.4 | **68.6** | **46.7** |

> On the ReMoT-16k-Test benchmark, the optimal variant SFT↔GRPO achieves +17.3 Overall / +25.1 Partial improvement over the baseline.

### Ablation Study

| Training Data Composition | Overall Acc. | Partial Acc. |
|--------------------------|-------------|-------------|
| No training (baseline) | 20.7 | 38.9 |
| Manipulation only | 23.9 | 46.7 |
| + Navigation | 32.4 | 57.6 |
| + Simulation | **38.0** | **64.0** |

| Logical Reward Ablation | Overall | Partial | Logical Consistency |
|------------------------|---------|---------|---------------------|
| Qwen3-VL-4B baseline | 16.2 | 39.6 | 46.6% |
| GRPO w/o logical reward | 68.6 | 77.3 | 98.6% |
| **GRPO w/ logical reward** | **78.0** | **81.3** | **99.3%** |

> The data composition ablation shows that navigation data contributes most (+ 8.4%), confirming the centrality of spatial relational reasoning. The logical reward yields a +9.4 Overall gain on the manipulation subset, while logical consistency improves from 46.6% to 99.3%.

### Key Findings

- The multi-expert pipeline exhibits smooth data scaling (GRPO reaches 0.61, cross-validated variants reach 0.64–0.66), whereas VLM-generated data scales erratically with a lower ceiling (~0.49).
- ReMoT-4B-CoT surpasses Qwen3-VL-30B-CoT (7.5× larger) on spatiotemporal benchmarks (VLM2/VSI/MMSI: +1.8/+2.7/+2.3%).
- ReMoT maintains or improves performance on general multimodal benchmarks, showing no catastrophic forgetting.
- The 4B model matches or exceeds GPT-4o on spatiotemporal tasks.

## Highlights & Insights

- **Systematic solution**: This is the first work to address VLM spatiotemporal reasoning deficiencies holistically across the data-training-evaluation pipeline, rather than applying isolated patches.
- **Rule-driven vs. VLM-generated data**: The multi-expert pipeline decisively outperforms direct VLM generation (55% format error rate vs. 16.5K high-quality triplets) with superior scalability—a significant insight for AI data production.
- **Logical consistency reward**: Formalizing the verification of transitivity in reasoning chains is a general and elegant solution, transferable to any scenario requiring multi-step reasoning consistency.
- **Small model, large capability**: The 4B model surpasses the 30B model and GPT-4o through careful data curation and training strategy, validating that data quality and training paradigm outweigh model scale.

## Limitations & Future Work

- Validation is limited to Qwen3-VL-4B as the sole base model; effects on larger models (7B/14B) remain unverified.
- Motion contrast triplets primarily cover navigation, manipulation, and simulation domains; complex scenarios (e.g., sports, industrial processes) are not included.
- CoT length regularization may truncate necessary long-chain reasoning.
- The optimal phase lengths $(K_{SFT}, K_{GRPO})$ for the alternating strategy lack theoretical guidance and have not been sufficiently ablated.

## Related Work & Insights

- **GRPO (DeepSeek)**: ReMoT validates the superiority of GRPO over SFT for visual reasoning tasks in the vision domain, and introduces the logical consistency reward as an additional innovation.
- **SpatialVLM / 3D-LLM**: These works enhance spatial understanding via depth maps and scene graphs; ReMoT approaches the problem from a motion contrast perspective, and the two directions are complementary.
- **Insights**: (1) The rule-driven data construction approach for motion contrast triplets is transferable to video generation quality assessment; (2) the logical consistency reward is applicable to RL training for any multi-step reasoning task.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The systematic combination of motion contrast triplets + GRPO + logical reward is highly innovative.
- Technical Depth: ⭐⭐⭐⭐ The multi-expert pipeline is elegantly designed and the training paradigm exploration is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Self-constructed benchmark + multi-benchmark validation + detailed ablation + data scaling analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the overall structure is systematic.
- Value: ⭐⭐⭐⭐⭐ Spatiotemporal reasoning is a core capability bottleneck for VLM applications in autonomous driving and robotics.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SAML: A Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Prediction](../../AAAI2026/autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[CVPR 2026\] SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](sharp_short-window_streaming_for_accurate_and_robust_prediction_in_motion_foreca.md)
- [\[CVPR 2026\] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)

<!-- RELATED:END -->
