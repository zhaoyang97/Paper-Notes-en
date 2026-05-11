---
title: >-
  [Paper Note] RM-R1: Reward Modeling as Reasoning
description: >-
  [ICLR 2026][Reinforcement Learning][Reward Model] This paper reframes reward modeling as a reasoning task, introducing the RM-R1 family of Reasoning Reward Models (ReasRM). Through reasoning distillation combined with RL…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Reward Model"
  - "Reasoning"
  - "Chain-of-Rubrics"
  - "Generative Reward Model"
  - "RLVR"
date: 2026-05-08
content_hash: fac9e27c6d28db4b
---

# RM-R1: Reward Modeling as Reasoning

**Conference**: ICLR 2026
**arXiv**: [2505.02387](https://arxiv.org/abs/2505.02387)
**Code**: [GitHub](https://github.com/RM-R1-UIUC/RM-R1)
**Area**: Reinforcement Learning
**Keywords**: Reward Model, Reasoning, Chain-of-Rubrics, Generative Reward Model, RLVR

## TL;DR

This paper reframes reward modeling as a reasoning task, introducing the RM-R1 family of Reasoning Reward Models (ReasRM). Through reasoning distillation combined with RL training and a Chain-of-Rubrics (CoR) mechanism, RM-R1 outperforms 70B and GPT-4o models by an average of 4.9% across three major reward model benchmarks.

## Background & Motivation

Reward models are a core component of RLHF for aligning LLMs. Existing approaches fall into two categories: (1) **ScalarRM**—training the RM as a classifier to output scalar scores, which is opaque and lacks explicit reasoning; and (2) **GenRM**—generating textual judgments with some degree of transparency, but whose reasoning is often shallow and unreliable, resulting in performance inferior to ScalarRM.

The authors observe that accurate reward modeling inherently requires reasoning: inferring the latent criteria of a judge, weighing multiple criteria, and simulating potential consequences. The example in Figure 1 illustrates this clearly—a standard instruction-tuned model overfits to surface-level patterns, whereas a reasoning model can evaluate the deeper implications of a response.

Core Problem: **Can reward modeling be treated as a reasoning task?**

This paper proposes **Reasoning Reward Models (ReasRM)** as a new model category, emphasizing the use of long, coherent reasoning chains during the judgment process. A two-stage training pipeline (distillation + RL) and a category-specific Chain-of-Rubrics reasoning strategy are introduced.

## Method

### Overall Architecture

RM-R1 training consists of two stages: (1) reasoning distillation—using oracle models such as o3/Claude to synthesize high-quality reasoning trajectories for training the base model; and (2) reinforcement learning—further optimizing reasoning judgment capability using GRPO with verifiable rewards.

### Key Designs

1. **Reasoning Distillation (Stage 1)**:

    - **Function**: Oracle models synthesize reasoning trajectories $r^{(i)}$, constructing distillation data $y_{\text{trace}}^{(i)} = r^{(i)} \oplus l^{(i)}$.
    - **Mechanism**: Minimizes the NLL loss $\mathcal{L}_{\text{distill}}(\theta) = -\sum \log r_\theta(y_t|x,y_{<t})$ to guide the model toward structured reasoning.
    - **Design Motivation**: Directly applying instruction-tuned models as GenRMs yields poor performance; reasoning trajectory demonstrations are required. Competitive performance is achievable with as few as ~8.7K samples.

2. **Chain-of-Rubrics (CoR) Reasoning Strategy**:

    - **Function**: Applies different reasoning strategies depending on task type.
    - **Mechanism**: The model first classifies the query as Chat or Reasoning—Chat queries trigger rubric generation followed by evaluation; Reasoning queries prompt the model to solve the problem independently before comparing against the candidate answer.
    - **Design Motivation**: Different types of preference judgments focus on distinct criteria—chat tasks emphasize textual standards such as politeness and safety, while reasoning tasks focus on logical correctness and answer accuracy.

3. **GRPO Reinforcement Learning (Stage 2)**:

    - **Function**: Uses judgment correctness as the reward signal and optimizes the policy with GRPO.
    - **Mechanism**: $\mathcal{R}(x,j|y_a,y_b) = \begin{cases} 1 & \text{if } \hat{l}=l \\ -1 & \text{otherwise} \end{cases}$
    - **Design Motivation**: Distillation is prone to overfitting specific patterns; RL enhances generalization and critical thinking through exploration.

### Loss & Training

Stage 1 applies standard NLL loss for reasoning distillation. Stage 2 applies GRPO to maximize $\mathbb{E}[\mathcal{R}(x,j)] - \beta D_{KL}(r_\theta \| r_{\text{ref}})$, where the reference model is the distillation-stage checkpoint. Notably, only a correctness reward is used (no format reward), as the model has already learned proper formatting through distillation.

## Key Experimental Results

### Main Results (Average Across Three Benchmarks)

| Model | RewardBench | RM-Bench | RMB | Average |
|-------|-------------|----------|-----|---------|
| INF-ORM-70B (ScalarRM) | 95.1 | 70.9 | 70.5 | 78.8 |
| GPT-4o (GenRM) | 86.7 | 72.5 | 73.8 | 77.7 |
| Self-taught-eval-70B | 90.2 | 71.4 | 67.0 | 76.2 |
| **RM-R1-14B (ours)** | **88.9** | **81.5** | **68.5** | **79.6** |
| **RM-R1-32B (ours)** | **90.9** | **83.9** | **69.8** | **81.5** |

### Ablation Study (Qwen-2.5-Instruct-32B, RewardBench)

| Method | Chat | Chat Hard | Safety | Reasoning | Average |
|--------|------|-----------|--------|-----------|---------|
| Base Instruct Model | 95.8 | 74.3 | 86.8 | 86.3 | 85.8 |
| +Cold Start RL | 92.5 | 81.5 | 89.7 | 94.4 | 89.5 |
| +RL+Rubrics | 93.0 | 82.5 | 90.8 | 94.2 | 90.1 |
| +RL+Rubrics+QC | 92.3 | 82.6 | 91.6 | 96.3 | 90.8 |
| **RM-R1 (Full)** | **95.3** | **83.1** | **91.9** | **95.2** | **91.4** |

### Key Findings
- RM-R1 surpasses the previous best on RM-Bench by 8.7%, achieving 91.8% on math and 74.1% on code.
- Reasoning capability is critical for reward modeling—distillation provides the foundation while RL further enhances generalization.
- Model scale exhibits strong scaling behavior—near-linear relative gains are observed from 7B to 32B.
- Inference-time reasoning length scaling is also effective—longer reasoning chains yield better judgment performance.

## Highlights & Insights

- **Deep integration of RM and reasoning**: This work is the first to systematically introduce long-chain reasoning into reward modeling, establishing ReasRM as a new model category.
- **High data efficiency**: Only 8.7K distillation samples suffice to achieve competitive performance, far fewer than the 800K samples used in DeepSeek-Distilled.
- **Elegant CoR design**: Differentiating evaluation strategies for chat and reasoning tasks mirrors the actual cognitive process humans employ when scoring responses.
- **Insightful SFT vs. RL comparison**: Table 3 consistently shows that reasoning training via RL outperforms SFT, even when trained on the same distillation data.

## Limitations & Future Work

- The CoR classification (Chat vs. Reasoning) may be overly coarse; finer-grained task categorization could yield further improvements.
- The pipeline depends on oracle models (o3/Claude) for distillation data generation, increasing computational cost.
- The current reward design relies on binary correctness (±1); more fine-grained reward signals may further improve performance.

## Related Work & Insights

- The DeepSeek-GRM series represents a direct competitor, but remains closed-source and relies on substantially more data.
- JudgeLRM also belongs to the ReasRM category but lags considerably in performance, underscoring the importance of the training methodology.
- Key takeaway: In RM training, *how to reason* matters more than *how much data to train on*.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing reasoning into RM is a promising direction, though the distillation + RL framework is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three major benchmarks, detailed ablations, scaling analysis, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the motivating example in Figure 1 is particularly compelling.
- **Value**: ⭐⭐⭐⭐⭐ Establishes the ReasRM paradigm and advances the community through open-source code and models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[AAAI 2026\] TextShield-R1: Reinforced Reasoning for Tampered Text Detection](../../AAAI2026/reinforcement_learning/textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)

</div>

<!-- RELATED:END -->
