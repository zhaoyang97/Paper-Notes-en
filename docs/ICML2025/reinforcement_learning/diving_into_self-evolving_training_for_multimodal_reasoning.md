---
title: >-
  [Paper Note] Diving into Self-Evolving Training for Multimodal Reasoning
description: >-
  [ICML 2025][Reinforcement Learning][Self-Evolving Training] This paper revisits self-evolving training in multimodal reasoning from a reinforcement learning perspective, systematically analyzing three key factors: training methods, reward models, and prompt variations. It proposes an adaptive temperature adjustment mechanism based on Reward-Pass@K to alleviate training saturation, culminating in the M-STaR framework, which achieves consistent improvements across multiple benc…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Self-Evolving Training"
  - "Multimodal Reasoning"
  - "Process Reward Model"
  - "Exploration-Exploitation Balance"
  - "Adaptive Temperature"
date: 2026-05-08
content_hash: f7aaf568e58429f0
---

# Diving into Self-Evolving Training for Multimodal Reasoning

**Conference**: ICML 2025  
**arXiv**: [2412.17451](https://arxiv.org/abs/2412.17451)  
**Code**: [https://mstar-lmm.github.io](https://mstar-lmm.github.io)  
**Area**: Reinforcement Learning  
**Keywords**: Self-Evolving Training, Multimodal Reasoning, Process Reward Model, Exploration-Exploitation Balance, Adaptive Temperature

## TL;DR

This paper revisits self-evolving training in multimodal reasoning from a reinforcement learning perspective, systematically analyzing three key factors: training methods, reward models, and prompt variations. It proposes an adaptive temperature adjustment mechanism based on Reward-Pass@K to alleviate training saturation, culminating in the M-STaR framework, which achieves consistent improvements across multiple benchmarks.

## Background & Motivation

Multimodal reasoning is a fundamental capability for domains such as autonomous agents, robotics, and autonomous driving. However, high-quality, human-annotated Chain-of-Thought (CoT) data is extremely scarce in multimodal scenarios, which severely limits the reasoning capabilities of large multimodal models (LMMs).

**Self-Evolving Training**, where a model iteratively learns from its own outputs, has emerged as a key paradigm to address this issue. However, existing works suffer from several core challenges:

**Research Gap**: Exploration of self-evolving training has mostly focused on pure text domains (e.g., STaR, ReST, ReSTEM), leaving systematic studies on multimodal reasoning almost non-existent.

**Lack of a Unified Framework**: The few attempts in multimodal scenarios lack systematic design principles.

**Performance Saturation**: The exploration capability inevitably declines during training, leading to stagnant performance growth.

The core motivation of this study is: **Can we systematically understand and improve multimodal self-evolving training from an RL perspective?**

## Method

### Overall Architecture

The authors model self-evolving training as a general RL framework. Given a reward function $\mathcal{R}$, the goal of the policy model $\pi_\theta$ is to maximize the expected reward:

$$\pi_\theta^{t+1} = \arg\max_{\pi_\theta^t} \sum_i^L \mathbb{E}_{x,o,a^* \sim \mathcal{D}, \hat{y}_i \sim \pi_\theta^t[\cdot|x,o]}[\mathcal{R}(a^*, \hat{y}_i)]$$

The framework consists of two alternating phases:
- **Generate**: The current policy model samples and generates multiple candidate responses.
- **Improve**: High-quality responses are filtered using reward signals, and the policy model is trained with SFT loss.

Within this framework, the authors identify three key design dimensions: **training methods ($\mathcal{T}$), reward models ($\mathcal{R}$), and prompt variations ($\mathcal{P}$)**, investigating each through large-scale controlled experiments.

### Key Designs

#### Design Dimension 1: Training Methods (Continuous Self-Evolving)

The primary difference among existing iterative training methods lies in their model initialization strategies:
- **Iterative RFT**: Initializes from the previous checkpoint at each iteration, with the optimizer reset.
- **ReSTEM**: Initializes from the initial checkpoint at each iteration to prevent overfitting.

The authors observe a key gap: when the iteration interval is small enough and the optimizer states are inherited across iterations, iterative training approaches online RL. Based on this insight, they propose **Continuous Self-Evolving**:

- Initializes the model ($\pi_\theta^t$) from the previous checkpoint.
- **Inherits optimizer states and the learning rate scheduler** (the key innovation), making the optimization process globally continuous.
- Introduces an adjustable iteration interval to control the proportion of data processed in each iteration.

Experiments on the iteration interval show that **25% is the optimal ratio**: too large approaches offline training, where the model cannot adapt to changes in its own distribution in a timely manner; too small causes excessively frequent switching, leading to training instability.

#### Design Dimension 2: Process Reward Model

Traditional self-evolving training uses a binary exact-match reward $\mathcal{R}(\hat{y}_i) = \mathbb{1}(\hat{a}_i = a^*)$, which only evaluates the correctness of the final answer and ignores the quality of the reasoning process. The authors train the **first Multimodal Process Reward Model (Multimodal PRM)** and integrate it into the training workflow:

$$\mathcal{R}(\hat{y}_i) = \mathcal{H}(\mathbb{1}(a^* = \hat{a}_i) \times \mathcal{R}_p(\hat{y}_i))$$

The process reward scores each reasoning step, taking the minimum value as the overall score:

$$\mathcal{R}_p(\hat{y}_i) = \min(f(s_i^0), f(s_i^1), \ldots, f(s_i^m))$$

Regarding the utilization strategy of PRM, the authors compare two schemes:
- **Top-K**: Selects the $K$ responses with the highest PRM scores among correct responses.
- **Threshold Filtering ($>\alpha$)**: Filters out responses with scores below a threshold.

**Key Finding**: PRM performs far better as a **Reranker** than as a **Verifier**. Specifically:
- PRM underperforms even simple majority voting in Best-of-N and weighted voting metrics.
- However, responses selected by Top-2 exhibit fewer reasoning steps and higher relevance to the questions.
- Threshold filtering tends to either keep all or discard all responses, reducing diversity.

Top-2 is the optimal choice, balancing response quality and diversity.

#### Design Dimension 3: Prompt Variation

The authors investigate whether incorporating unlabeled data can improve training effectiveness:

- **Skyline Experiment**: Using unlabeled prompts with oracle answer feedback $\to$ OOD improves but ID decreases, posing a risk of forgetting.
- **Using Weighted-Voting Pseudo-labels** for unlabeled prompts $\to$ Hurts performance when the PRM's generalization is insufficient.
- **Timing of Introduction**: Incorporating unlabeled data after 75% of the training yields the best results, though primarily due to lower participation.

**Core Conclusion**: When PRM generalization capabilities are limited, unlabeled prompts cause distribution shifts in the policy model. Consequently, the authors opt to use labeled data exclusively.

### Loss & Training

#### Training Dynamics Analysis and Adaptive Temperature Mechanism

The authors propose monitoring three metrics to understand training dynamics:
- **Greedy Accuracy**: Greedy decoding accuracy, which improves progressively.
- **Pass@K**: The proportion where at least one out of $K$ samples is correct, reflecting **exploration capability**.
- **Reward-Pass@2** (a new metric): The proportion of correct answers present within the Top-2 responses ranked by PRM, reflecting **utilization efficiency**.

Key finding: Pass@K continuously declines during training (exploration decay), while Reward-Pass@2 quickly saturates. Higher sampling temperatures can delay exploration decay.

Based on this, they propose an **adaptive temperature adjustment mechanism**:
- Automatically adjusts the sampling temperature every two iterations.
- Temperature ranges from 0.3 to 1.6 with an interval of 0.1.
- Selects the temperature that maximizes Reward-Pass@2 on the validation set.
- Dynamically balances exploration and exploitation, alleviating performance saturation.

#### Final M-STaR Method

Integrating all best practices yields M-STaR (Multimodal Self-evolving Training for Reasoning):
1. Continuous Self-Evolving training method (25% iteration interval)
2. PRM Top-2 reranking to select high-quality training data
3. Exclusive use of labeled data
4. Reward-Pass@2 guided adaptive temperature adjustment

## Key Experimental Results

### Main Results

| Model | Benchmark | Base | +warmup | M-STaR | Gain |
|------|------|------|---------|--------|------|
| MiniCPM-V-2.5 (8B) | MathVista | 52.4 | 52.8 | **59.5** | +6.7 |
| Phi-3.5-Vision (4B) | MathVista | 46.5 | 49.3 | **54.5** | +5.2 |
| InternVL2 (2B) | MathVista | 46.4 | 47.6 | **50.3** | +2.7 |
| MiniCPM-V-2.5 | 5-Benchmark Average | 55.0 | 57.7 | **61.6** | +3.9 |
| Phi-3.5-Vision | 5-Benchmark Average | 46.5 | 55.3 | **59.2** | +3.9 |
| InternVL2-2B | 5-Benchmark Average | 26.2 | 52.8 | **53.3** | +0.5 |

### Ablation Study

| Configuration | MathV360K (ID) | MathVista (OOD) | Description |
|------|----------------|-----------------|------|
| SFT Direct Training | 44.3 | 54.8 | Baseline without iteration |
| Iterative RFT | 42.3 | 55.7 | Initialized from last ckpt, no continuous optimization |
| ReSTEM | 42.3 | 55.1 | Initialized from initial ckpt |
| Cont. Self-Evolving (100%) | 42.2 | 56.7 | Continuous optimization, full-data interval |
| Cont. Self-Evolving (25%) | **43.1** | **57.2** | Optimal interval |
| + PRM Top-2 | **45.3** | **59.2** | PRM reranking brings significant gains |
| + Reward-Pass@2 Adaptive | — | **59.5** | Dynamic temperature yields further improvements |

### Key Findings

1. **Continuous Optimization >> Resetting Optimizer**: Inheriting optimizer states and learn rate schedulers smoothens training, significantly boosting OOD performance.
2. **PRM is a Reranker, not a Verifier**: PRM underperforms majority voting on Best-of-N, but the responses it selects feature fewer reasoning steps and higher relevance.
3. **Unlabeled Data Requires Caution**: Introducing unlabeled data without precise reward signals shifts the policy distribution.
4. **Continuous Decay of Exploration Capability**: Pass@K monotonically decreases during training, which is the root cause of performance saturation.
5. **Model Scale Influences Generalization**: The 8B model improves across all 5 benchmarks, whereas the 2B model struggles to generalize on perception-intensive tasks.

## Highlights & Insights

- **Unified Framework via RL Perspective**: Unifies various self-evolving training variants (STaR, ReST, RFT, etc.) under an RL framework, systematizing design space analysis.
- **Simplicity of Continuous Self-Evolving**: Bridging the gap between iterative training and online RL using a highly simplified modification of inheriting optimizer states.
- **Insightful Role of PRMs**: PRMs are not adept at verifying correctness, but excel at selecting the best response from correct samples (reranking)—providing guidance for future PRM applications.
- **Reward-Pass@K Metric**: Elegantly unifies exploration (can the model generate good answers) and exploitation (can the reward model select good answers) into a single metric.
- **Adaptive Temperature Mechanism**: Uses Reward-Pass@2 as a signal to automatically adjust the temperature, which is theoretically natural and simple to implement.

## Limitations & Future Work

1. **PRM Quality Bottleneck**: Current multimodal PRM verification capabilities are limited (underperforming majority voting). Acquiring higher-quality step-level annotations remains a core challenge.
2. **Poor Generalization in Small Models**: The 2B model even shows negative growth on some benchmarks, suggesting that self-evolving training holds implicit requirements on model capacity.
3. **Coarse Temperature Adjustment**: Adjusting every two iterations among discrete candidates; future endeavors could explore fine-grained continuous adaptation.
4. **Applicable Only to Labeled Data**: The utilization of unlabeled data remains unresolved, limiting the framework's scalability.
5. **No Joint Optimization of Reward Models**: Keeping PRM fixed as a control variable; joint training of PRM and policy model might yield larger improvements.

## Related Work & Insights

- **STaR / ReSTEM**: The core baseline methods of this paper. M-STaR introduces continuous optimization, PRM, and dynamic temperature adjustment on top of them.
- **DeepSeek-R1**: Stimulating reasoning capabilities via pure RL, sharing a similar concept with the RL framework in this paper, but R1 is tailored for the text domain.
- **Process Reward Model** (Lightman et al. 2023; Wang et al. 2024): This paper serves as the first to extend PRM to the multimodal reasoning field.
- **Insights**: The exploration-exploitation balance is a timeless theme in RL. The Proposed Reward-Pass@K offers a practical tool to monitor self-evolving training dynamics, and the finding of PRMs as rerankers invites researchers to rethink the genuine role of reward models in training.

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified framework from an RL perspective and the Reward-Pass@K metric show ingenuity, though each component is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Control experiments on three key factors individually, across three model scales and five benchmarks with extensive dynamics analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-structured, tightly logical, with natural flow from analyses to proposed solutions.
- Value: ⭐⭐⭐⭐ — Provides a systematic guide for multimodal self-evolving training, although some findings might evolve with PRM progress.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] R-Zero: Self-Evolving Reasoning LLM from Zero Data](../../ICLR2026/reinforcement_learning/r-zero_self-evolving_reasoning_llm_from_zero_data.md)
- [\[ICML 2025\] A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization](a_theoretical_study_of_hyper_self-attention_through_the_lens_of_interactions_rep.md)
- [\[ICCV 2025\] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](../../ICCV2025/reinforcement_learning/r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

</div>

<!-- RELATED:END -->
