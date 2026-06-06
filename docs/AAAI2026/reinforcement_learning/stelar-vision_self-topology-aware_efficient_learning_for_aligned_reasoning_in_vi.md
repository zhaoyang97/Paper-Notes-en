---
title: >-
  [Paper Note] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision
description: >-
  [AAAI 2026][Reinforcement Learning][Topology-aware reasoning] This paper proposes STELAR-Vision, a topology-aware training framework for visual language reasoning. Via the TopoAug data generation pipeline…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Topology-aware reasoning"
  - "vision-language models"
  - "chain/tree/graph of thought"
  - "efficient inference"
date: 2026-05-08
content_hash: ffc3d28c5cafc04d
---

# STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision

**Conference**: AAAI 2026
**arXiv**: [2508.08688](https://arxiv.org/abs/2508.08688)  
**Code**: [stellar-neuron.github.io/stelar-vision](https://stellar-neuron.github.io/stelar-vision/)  
**Area**: Reinforcement Learning
**Keywords**: Topology-aware reasoning, vision-language models, chain/tree/graph of thought, reinforcement learning, efficient inference

## TL;DR

This paper proposes STELAR-Vision, a topology-aware training framework for visual language reasoning. Via the TopoAug data generation pipeline, it introduces diverse reasoning topologies—Chain, Tree, and Graph—and combines SFT with RL (SimPO) post-training. The framework achieves +9.7% accuracy on in-distribution data and up to +28.4% on out-of-distribution benchmarks, while reducing output length by 18.1% through Frugal Learning.

## Background & Motivation

### State of the Field

Current vision-language models (VLMs) for reasoning rely predominantly on the Chain-of-Thought (CoT) paradigm. The authors identify a critical issue: **different problems are better suited to different reasoning topologies**. CoT is only one of many possible reasoning structures; Tree- and Graph-based reasoning demonstrably outperforms CoT on certain problem types.

### Limitations of Prior Work

Through systematic evaluation of Qwen2-VL-7B and GPT-4o-Mini on the MATH-V dataset, the authors find:

- **Chain wins 49% of cases, Tree 28%, Graph 23%**—Tree and Graph together account for the majority
- **Different subjects favor different topologies**: Tree/Graph reasoning is notably superior in graph theory, statistics, and related disciplines
- **Chain reasoning is the most verbose**: its output token length distribution is heavily right-skewed with the highest mean; Tree/Graph distributions are more concentrated and concise

### Root Cause

1. Training data for existing VLMs is almost exclusively CoT-style, causing models to default to chain reasoning even when it is suboptimal
2. CoT reasoning is prone to "overthinking," generating unnecessarily lengthy outputs
3. **Topological diversity correlates with output length**—introducing Tree/Graph topologies naturally reduces output redundancy
4. Hypothesis 1: Training on topologically diverse data (without increasing data volume) enables models to adaptively select the optimal topology
5. Hypothesis 2: Building on this, a mechanism can be designed to encourage concise outputs, substantially improving efficiency with minimal accuracy loss

## Method

### Overall Architecture

STELAR-Vision comprises three core components:
1. **TopoAug**: A synthetic data generation pipeline that produces reasoning responses with multiple topology structures—Chain, Tree, and Graph—for each problem
2. **Two-stage post-training**: SFT → RL (SimPO)
3. **Frugal Learning**: A training variant that incentivizes concise outputs

### Key Designs

#### 1. TopoAug: Topology-Augmented Data Generation Pipeline

For each problem, two models (Qwen2-VL-7B-Instruct and GPT-4o-Mini) are used to repeatedly generate reasoning responses across three topologies: Chain, Tree, and Graph. Each topology supports configurable parameters such as maximum depth, number of child nodes, and number of neighbors.

Two types of labels are computed per problem:

- **Topology label $\mathcal{F}_{q,t}$**: A continuous value in $[0,1]$ representing the accuracy of topology $t$ on problem $q$:
  $$\mathcal{F}_{q,t} = \frac{N_{\text{correct}}(q, t)}{N_{\text{total}}(q, t)}$$

- **Outcome label $\mathcal{H}_r$**: A binary value $\{0, 1\}$ indicating whether each individual response is correct

Problems are categorized into three difficulty levels based on the topology label distribution:
- **Easy**: All three topology scores exceed the 85th percentile
- **Hard**: All three topology scores fall below the 15th percentile
- **Medium**: All remaining cases

Using two models of different scales ensures a balanced distribution of positive and negative samples and promotes diversity in reasoning topologies.

#### 2. Two-Stage Post-Training

**Stage 1: SFT (Supervised Fine-Tuning)**

Data preparation follows a three-step filtering process:
1. Balanced sampling from Easy/Medium/Hard problems
2. Retaining only responses with outcome label $\mathcal{H}_r = 1$
3. Rejection sampling using a 7B ORM (Outcome Reward Model trained on both topology and outcome labels) to select higher-quality samples

TopoAug data is mixed with three general VQA datasets (OKVQA, A-OKVQA, LLaVA-150k), the latter of which receive no topology augmentation. LoRA fine-tuning is applied with standard next-token prediction loss:

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)$$

**Stage 2: RL (Reinforcement Learning)**

Initialized from the SFT checkpoint, preference optimization is performed using SimPO:

$$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

Correct responses serve as preferred responses. A key detail: **topology prompts are removed during training**, compelling the model to autonomously infer the optimal reasoning structure at test time.

#### 3. Frugal Learning: Training Variants for Efficient Inference

Two variants are proposed:

- **STELAR-Vision-Short†**: During SFT, only "short and correct" responses (token length below the 25th percentile) are retained; during RL, "short and correct" responses serve as winners
- **STELAR-Vision-Short‡**: Extends Short† by additionally treating "correct but verbose" responses as losers, penalizing both incorrect and excessively long outputs

### Loss & Training

- Base model: Qwen2VL-7B-Instruct
- SFT data volume: approximately 50K–60K samples
- SFT training time: approximately 5–7 hours (8×A100/H100); RL training time: approximately 8–10 hours
- A single set of weights trained across all 5 OOD datasets is used, without any dataset-specific fine-tuning

## Key Experimental Results

### Main Results

| Model | VLM_S2H | MATH-V | Overall (ID) | Geometry3K | We-Math | PolyMath | SciBench | LogicVista |
|-------|---------|--------|--------------|------------|---------|----------|----------|------------|
| GPT-4o | 32.0 | 28.0 | 30.7 | 57.0 | 66.4 | 25.0 | 31.1 | 34.6 |
| Qwen2VL-7B-Instruct | 21.0 | 13.0 | 18.3 | 35.2 | 46.6 | 16.0 | 10.7 | 17.0 |
| Qwen2VL-72B-Instruct | 21.0 | 20.0 | 20.7 | 50.2 | 60.6 | 13.0 | 25.4 | 28.8 |
| Chain-Only | 25.0 | 21.0 | 23.7 | 31.4 | 42.2 | 17.2 | 10.7 | 25.4 |
| **STELAR-Vision** | **31.0** | **22.0** | **28.0** | 36.8 | 51.0 | 23.8 | 12.4 | 29.0 |

STELAR-Vision surpasses the base model by **+9.7%** on in-distribution data and outperforms the 10× larger Qwen2VL-72B-Instruct by **+7.3%**.

### Ablation Study

| Model | SFT | RL | VLM_S2H | MATH-V | Overall |
|-------|-----|----|---------|--------|---------|
| Qwen2VL-7B-Instruct | × | × | 21.0 | 13.0 | 18.3 |
| Chain-Only-SFT | ✓ | × | 18.5 | 19.0 | 18.7 |
| Chain-Only | ✓ | ✓ | 25.0 | 21.0 | 23.7 |
| STELAR-Vision-SFT | ✓ | × | 28.0 | 24.0 | 26.7 |
| **STELAR-Vision** | ✓ | ✓ | **31.0** | 22.0 | **28.0** |

**Overall gain of topology augmentation over Chain-Only**: 23.7% → 28.0% (+4.3%).

**Frugal Learning efficiency comparison**:

| Model | Accuracy (%) | ID Generated Tokens | OOD Generated Tokens |
|-------|-------------|--------------------|--------------------|
| Qwen2VL-7B-Instruct | 26.2 | 613.5 | 543.3 |
| Chain-Only | 28.7 | 878.4 | 742.6 |
| STELAR-Vision | 31.6 | 556.7 | 523.4 |
| **STELAR-Vision-Short†** | **28.7** | **455.7** | **498.6** |

STELAR-Vision-Short† reduces output length by 18.1% while still outperforming the base model by +2.5%.

### Key Findings

1. **Topological diversity expands the exploration space for RL**: RL yields consistent gains on TopoAug data, whereas its benefit diminishes with Chain-Only data
2. **Post-training models autonomously select topologies** (without explicit prompting): on Geometry3K, 96.4% of responses use Tree; on LogicVista, 61.7% use Chain—indicating that the model has genuinely learned to select the optimal structure based on problem characteristics
3. **SFT may overfit**, causing SFT-only variants to underperform the full model on certain OOD datasets
4. **Frugal Learning fails for Chain-Only-Short†**: after RL fine-tuning, Chain-Only models tend to generate verbose responses, and Frugal Learning alone cannot effectively constrain output length

## Highlights & Insights

- **Systematic validation of the value of reasoning topology diversity**—different problems genuinely require different reasoning structures, offering a compelling rebuttal to the assumption that CoT is universally optimal
- **Punching above its weight**: the 7B model outperforms the 72B model by 7.3%, demonstrating that training paradigm matters more than model scale
- **Frugal Learning is only effective when built on topological diversity**—Chain-only training cannot simultaneously achieve conciseness and accuracy
- The topology selection distributions on OOD datasets closely align with problem structure (simple logic → Chain; complex geometry → Tree/Graph), confirming genuine generalization

## Limitations & Future Work

- The current topology types are predefined as {Chain, Tree, Graph}; more flexible end-to-end topology discovery remains unexplored
- Incompatibility with Qwen2.5-VL (which generates diverse topologies unstably) limits base model selection
- The dynamic relationship between problem structure and optimal topology has not been deeply investigated
- The Short‡ variant of Frugal Learning (simultaneously penalizing verbosity) performs worse, suggesting conflicting optimization signals
- OOD generalization is strong overall, but gains on certain specific datasets (e.g., SciBench) are limited

## Related Work & Insights

- CoT, ToT, and GoT have previously been employed primarily through sampling or rule-based methods; this paper is the first to incorporate multi-topology training into a VLM post-training framework
- The use of SimPO is a judicious choice: it requires no separate reward model and aligns naturally with decoding behavior
- Complementary to CuRPO: CuRPO identifies CoT as harmful in visual grounding and mitigates this via curriculum learning, while this paper finds CoT insufficient and introduces additional topologies as alternatives
- The Frugal Learning direction is related to L1 (RL-based reasoning length control) and SelfBudgeter, but is simpler in design

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing multi-topology reasoning structures into VLM post-training is a novel and persuasive contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 7 datasets (including 5 OOD), multiple baselines, and complete ablations; however, experiments are limited to the 7B scale
- **Writing Quality**: ⭐⭐⭐⭐ — Systematic analysis with rich figures and tables, though the extended version is lengthy
- **Value**: ⭐⭐⭐⭐⭐ — The approach of achieving strong performance with a small model is highly practical, and the TopoAug pipeline can be directly reused

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](../../ICLR2026/reinforcement_learning/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](../../ICML2026/reinforcement_learning/d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)
- [\[NeurIPS 2025\] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](../../NeurIPS2025/reinforcement_learning/open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)

</div>

<!-- RELATED:END -->
