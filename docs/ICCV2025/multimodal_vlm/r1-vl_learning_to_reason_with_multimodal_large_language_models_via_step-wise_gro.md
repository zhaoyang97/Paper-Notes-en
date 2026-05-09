---
title: >-
  [Paper Note] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization
description: >-
  [Multimodal VLM] This paper proposes StepGRPO, an online reinforcement learning framework that introduces two rule-based step-wise reasoning rewards — StepRAR (Step-wise Reasoning Accuracy Reward) and StepRVR (Step-wise Reasoning Validity Reward) — without requiring a process reward model. The framework addresses the sparse reward problem in RL-based MLLM training, enabling models to autonomously explore and improve their reasoning capabilities.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: 7d1d8e45517379cd
---

# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.12937](https://arxiv.org/abs/2503.12937)
- **Code**: [See paper](https://arxiv.org/abs/2503.12937)
- **Area**: Multimodal VLM / MLLM Reasoning
- **Keywords**: MLLM reasoning, online reinforcement learning, step-wise reward, GRPO, sparse reward problem
- **Authors**: Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, Dacheng Tao (NTU Singapore)

## TL;DR

This paper proposes StepGRPO, an online reinforcement learning framework that introduces two rule-based step-wise reasoning rewards — StepRAR (Step-wise Reasoning Accuracy Reward) and StepRVR (Step-wise Reasoning Validity Reward) — without requiring a process reward model. The framework addresses the sparse reward problem in RL-based MLLM training, enabling models to autonomously explore and improve their reasoning capabilities.

## Background & Motivation

- **Background**: The prevailing approach for enhancing MLLM reasoning is supervised fine-tuning (SFT) on high-quality CoT data (e.g., Mulberry). However, SFT only encourages **passive imitation** of correct reasoning paths, leaving the model without understanding of its own errors.
- **Limitations of Prior Work**: Inspired by DeepSeek-R1, applying GRPO-style online RL to MLLMs is a natural extension. However, directly using outcome-level rewards introduces a **sparse reward problem**: small MLLMs achieve low accuracy on long-chain reasoning; few sampled trajectories receive positive rewards; insufficient positive feedback leads to poor exploration efficiency and training instability.
- **Key Challenge**: Outcome-level rewards alone fail to provide dense, fine-grained supervision for intermediate reasoning steps.
- **Goal**: Provide step-level reward signals that are denser and more informative than outcome-level rewards, without training an additional process reward model.

## Method

### Overall Architecture

StepGRPO consists of two stages:
1. **Policy Warm-up**: SFT on CoT data to establish basic reasoning capability.
2. **Step-wise Online Policy Optimization**: Online RL with self-exploration and step-wise reward optimization.

### Step-wise Reasoning Accuracy Reward (StepRAR)

Evaluates whether a reasoning trajectory contains necessary intermediate reasoning steps.

**Key Step Extraction**:
- Key steps $\mathbf{v} = \{v_1, v_2, ...\}$ are pre-extracted from reference CoT reasoning paths in the training data.
- Only core numerical values and equations are retained as key tokens.
- Multiple equivalent formats are augmented (e.g., $\frac{6}{3}=2$ → "6/3 = 2" → "6 divided by 3 equals 2").

**Soft Matching**:
$$k^i = |\mathbf{v}_{match}| / |\mathbf{v}|$$

**StepRAR Definition**:
$$r_{auc}^i = \begin{cases} 1 + \alpha k^i, & \text{correct answer} \\ \alpha k^i, & \text{answer present but incorrect} \\ 0, & \text{no answer} \end{cases}$$

Even when the final answer is wrong, a trajectory containing correct intermediate steps receives a partial reward ($\alpha k^i$), alleviating the sparse reward problem.

### Step-wise Reasoning Validity Reward (StepRVR)

Evaluates the structural completeness and logical consistency of a reasoning trajectory based on two rules:
- **Reasoning Completeness** $\delta^c$: The response must contain contextual analysis, step-by-step reasoning, and a final answer.
- **Reasoning Logicality** $\delta^l$: The contextual analysis must precede the reasoning steps, and the final answer must follow them.

$$r_{val}^i = \begin{cases} 1, & \mathbb{I}(\delta^c) \cdot \mathbb{I}(\delta^l) = 1 \\ 0, & \text{otherwise} \end{cases}$$

### Policy Optimization

The total reward is $r^i = r_{auc}^i + r_{val}^i$, with group-relative advantage:

$$\hat{A}^i = \frac{r^i - \text{mean}(\{r^1, ..., r^M\})}{\text{std}(\{r^1, ..., r^M\})}$$

RL objective with KL divergence regularization:

$$\mathcal{L}_{StepRL} = -\mathbb{E}\left[ \frac{1}{M} \sum_{i=1}^M \frac{\pi_\theta(\mathbf{c}^i|Q)}{[\pi_\theta(\mathbf{c}^i|Q)]_{\text{no grad}}} \hat{A}^i - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

### Training Details

- Base models: Qwen2-VL-2B and Qwen2-VL-7B
- Warm-up: SFT on Mulberry-260k data
- RL: 10K samples randomly drawn from Mulberry-260k
- 4 rollouts per question ($M=4$), temperature 1.2, max length 1024
- 4× H100-80GB GPUs; lr=1e-6, $\alpha=0.1$, $\beta=0.04$

## Key Experimental Results

### Main Results (Table 1: Average over 8 Benchmarks)

| Model | MathVista | MMStar | Math-V | ChartQA | DynaMath | HallBench | MathVerse | AVG |
|------|-----------|--------|--------|---------|----------|-----------|-----------|-----|
| GPT-4o | 63.8 | 63.9 | 30.3 | 85.7 | 63.7 | 55.0 | 39.4 | 56.2 |
| Qwen2-VL-2B | 43.0 | 48.0 | 12.4 | 73.5 | 24.9 | 41.7 | 19.7 | 37.5 |
| **R1-VL-2B** | **52.1** | 49.8 | **17.1** | **75.2** | **29.4** | **44.0** | **26.2** | **41.6** |
| Qwen2-VL-7B | 58.2 | 60.7 | 16.3 | 83.0 | 42.1 | 50.6 | 32.5 | 48.7 |
| **R1-VL-7B** | **63.5** | 60.0 | **24.7** | **83.9** | **45.2** | **54.7** | **40.0** | **52.1** |
| Mulberry-7B | 63.1 | 61.3 | — | 83.9 | 45.1 | 54.1 | — | — |
| LlamaV-o1-11B | 54.4 | 59.4 | — | — | — | 63.5 | — | — |

R1-VL-7B achieves 63.5 on MathVista, approaching GPT-4o (63.8). R1-VL-2B even surpasses LLaVA-CoT-11B (+9.3 on MathVista).

### Ablation Study (Table 2: Contribution of Each Reward Component)

| Warm-up | StepRAR | StepRVR | MathVista |
|---------|---------|---------|-----------|
| ✗ | ✗ | ✗ | 58.2 (baseline) |
| ✓ | ✗ | ✗ | 61.2 |
| ✓ | ✓ | ✗ | 62.4 |
| ✓ | ✗ | ✓ | 61.9 |
| ✓ | ✓ | ✓ | **63.5** |

Each component contributes clearly: warm-up +3.0, StepRAR +1.2, StepRVR +0.7, combined +2.3.

### Step-wise vs. Outcome-level Reward (Table 4)

| Method | MathVista |
|------|-----------|
| Warm-up only | 61.7 |
| Warm-up + Outcome-level reward | 62.3 |
| **Warm-up + Step-wise reward** | **63.5** |

Step-wise rewards yield an additional 1.2% improvement over outcome-level rewards.

### Sensitivity to $M$ (Table 3)

| M (rollouts per question) | 2 | 3 | 4 | 5 | 6 |
|-------------|---|---|---|---|---|
| MathVista | 62.5 | 62.8 | 63.5 | 63.2 | 63.7 |

$M=4$ achieves the best trade-off between performance and computational cost.

### Key Findings

1. **StepGRPO consistently outperforms SFT** at identical training steps.
2. **Significant gains on small models**: R1-VL-2B improves by an average of 4.1% over the Qwen2-VL-2B baseline.
3. **Step-wise rewards alleviate sparsity**: useful intermediate steps receive positive feedback even when the final answer is incorrect.
4. **Rule-based rewards are sufficient**: no additional process reward model training is required.
5. **StepRVR enforces reasoning structure**: the framework requires a logical flow of context analysis → reasoning → conclusion.

## Highlights & Insights

- **Elegant dense step-level reward design**: fine-grained rewards are achieved through rule-based key-step matching and structural validation, without training a Process Reward Model (PRM).
- **Orthogonal decomposition of reasoning quality**: the framework distinguishes reasoning accuracy (StepRAR) from reasoning structural validity (StepRVR) as two independent dimensions.
- **Practical and efficient**: RL training requires only 10K samples and 4 GPUs.
- **Soft key-step matching**: the format augmentation mechanism elegantly handles diverse representations of mathematical expressions.

## Limitations & Future Work

- Key step extraction relies on GPT-4, introducing an external dependency and additional cost.
- Evaluation is primarily on mathematical reasoning benchmarks; performance on spatial reasoning, commonsense reasoning, and other domains remains unexplored.
- Experiments are conducted on Qwen2-VL only; generalizability to other base models is not verified.
- Scaling behavior with larger training datasets beyond 10K has not been investigated.
- StepRVR's rules enforce a rigid reasoning structure, which may limit the diversity of generated reasoning chains.

## Related Work & Insights

- **DeepSeek-R1 → MLLM extension**: StepGRPO extends GRPO from LLMs to MLLMs, with the key contribution being the resolution of sparse reward issues in multimodal settings.
- **Complementary to Mulberry**: Mulberry provides SFT data for the warm-up stage; StepGRPO enables subsequent RL-based self-improvement.
- **PRM-free paradigm**: rule-based step-wise rewards may serve as a more practical alternative to learned process reward models.

## Rating ⭐⭐⭐⭐

A solid and effective extension of GRPO to MLLMs. The step-wise reward design is clean and practical, and the ablation study is clear and well-structured. Consistent improvements across multiple benchmarks are convincing. The primary limitations are the evaluation scope being skewed toward mathematical reasoning and the reliance on an external large model for key step extraction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)

</div>

<!-- RELATED:END -->
