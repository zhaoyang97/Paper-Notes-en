---
title: >-
  [Paper Note] GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization
description: >-
  [ACL 2025 (Long Paper)][LLM (Other)][Preferential Prompt] This work proposes the GAPO (Generative Adversarial Policy Optimization) framework, which integrates the adversarial training mechanism of GANs with PPO. By replacing the traditional decoder-only architecture with an encoder-only reward model, GAPO introduces a new paradigm called "Preferential Prompt" (modifying constraints in the prompts rather than the responses) to enhance the LLM's capability to understand and adh…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM (Other)"
  - "Preferential Prompt"
  - "GAN"
  - "PPO"
  - "Encoder-only Reward Model"
  - "Constraint Following"
date: 2026-05-08
content_hash: 484b718bc8e05841
---

# GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2503.20194](https://arxiv.org/abs/2503.20194)  
**Code**: [https://github.com/MikeGu721/GAPO](https://github.com/MikeGu721/GAPO)  
**Area**: LLM/NLP  
**Keywords**: Preferential Prompt, GAN, PPO, Encoder-only Reward Model, Constraint Following  

## TL;DR

This work proposes the GAPO (Generative Adversarial Policy Optimization) framework, which integrates the adversarial training mechanism of GANs with PPO. By replacing the traditional decoder-only architecture with an encoder-only reward model, GAPO introduces a new paradigm called "Preferential Prompt" (modifying constraints in the prompts rather than the responses) to enhance the LLM's capability to understand and adhere to fine-grained constraints. It significantly outperforms baselines such as DPO, KTO, and SimPO on the IFEval and product description generation tasks.

## Background & Motivation

In practical applications, LLMs are required to strictly follow predefined constraints (such as format, style, and content), especially in scenarios like legal documents, medical records, and workflow automation. Existing methods primarily fall into two categories: (1) directly synthesizing instruction-response pairs that satisfy the constraints (SFT paradigm), which causes the model to merely learn "what a correct response is" without truly understanding the constraints themselves, rendering it prone to hallucinations and taking shortcuts; (2) preferential response optimization (such as DPO), which aligns the output probability through preferred response pairs under a fixed prompt, but still fails to teach the model to comprehend constraint details.

The common limitation of both approaches is: **the model does not understand constraints**—it only learns what kind of response is preferable given a specific prompt, without learning to distinguish subtle differences among constraints. When encountering new or slightly varied constraints, their performance collapses easily.

## Core Problem

The authors present a key observation: instead of conducting preference learning on the response side (preferential response), it is better to perform preference learning on the prompt side (preferential prompt). Specifically, by keeping the response fixed and modifying constraint conditions in the prompt, the model is trained to distinguish between "prompts that satisfy constraints" and "prompts that violate constraints." This allows the model to deeply understand the semantics of constraints.

However, this direction faces two major technical challenges: (1) Existing LLMs are predominantly decoder-only architectures, whose unidirectional attention mechanism is inherently suboptimal for detecting mismatches between prompts and responses. (2) There are difficulty gaps between constraints of different complexities, necessitating the construction of progressive training samples for bridging, which traditionally requires substantial human intervention.

## Method

### Overall Architecture

GAPO consists of two phases:

**Phase 1 (Warm-up)**: Train an encoder-only reward model (Longformer, 0.4B) using existing preference data. This model takes (prompt, response) pairs and outputs matching scores. The training data comes from constraint-aware data augmentation: modifying or inserting constraints into the original constraint set to generate mismatched rejected prompts, thereby forming (accepted prompt + response, rejected prompt + response) preference pairs.

**Phase 2 (Adversarial Training)**: The Generator (Qwen-2.5-7B) and the Reward Model are trained alternately. The Generator produces a response based on the prompt, and the Reward Model evaluates the generation quality and provides a reward signal, updating the Generator via PPO. Concurrently, the newly generated samples from the Generator are added to the Reward Model's training set, forcing the Reward Model to continuously improve its discriminative capabilities. This adversarial dynamic automatically produces samples of progressive difficulty during training.

### Key Designs

1. **Encoder-only Reward Model**: Longformer-Large-4096 (only 0.4B parameters) is utilized as the Reward Model, replacing the conventional practice of using the LLM itself (7B+) as the reward model. The bidirectional attention mechanism of the Encoder is superior to the unidirectional attention of Decoders in capturing the alignment between prompts and responses—which is critical for handling Preferential Prompt data. The parameter size is reduced by an order of magnitude, drastically lowering computational costs.

2. **Constraint-Aware Data Augmentation**: Instead of manually constructing preference pairs, the data is automatically generated via two actions: (a) **Constraint Modification**—randomly selecting a constraint and modifying it to be incompatible with the original response; (b) **Constraint Insertion**—adding a new constraint that conflicts with existing constraints. Consequently, each data pair becomes (accepted_prompt, response) and (rejected_prompt, response), where the core difference lies solely within the constraint section of the prompts.

3. **GAN-PPO Integration Mechanism**: Unlike standard PPO (where the Reward Model is frozen after training), GAPO alternately trains the Reward Model and the Generator in an adversarial manner. As the Generator produces increasingly better outputs, the Reward Model requires more fine-grained discrimination, which in turn drives the Generator to improve further. Experiments show that different Reward Models stably converge around stage A12, with scores stratifying between 0.2 and 0.95, indicating that the adversarial training successfully establishes a balanced dynamic rather than degradation.

### Loss & Training

The Reward Model is trained using cross-entropy loss:
$$L_R(\theta) = -\mathbb{E}_{(c,t,y)\sim\mathcal{D}'} [y\log R(c,t) + (1-y)\log(1-R(c,t))]$$

The objective function of the Generator takes the standard PPO form, utilizing the advantage function $A_n = Q_\pi(c_n, t_n) - V_\pi(c_n)$, where $Q_\pi$ combines the immediate reward from the Reward Model and discounted future returns.

## Key Experimental Results

| Dataset | Metric | GAPO | PPO | SFT | DPO | KTO | SimPO | ORPO |
|--------|------|------|-----|-----|-----|-----|-------|------|
| IFEval | Overall Accuracy | **83.9%** | 75.6% | 78.3% | 33.3% | 54.4% | 30.6% | 33.9% |
| PDD (GPT-4o Eval) | Score | **90.2%** | 89.7% | 82.6% | 5.4% | - | 2.9% | 7.5% |
| PDD (Human Eval) | Score | **89%** | 81% | 60% | 0% | - | 0% | 0% |

**Preferential Prompt vs Preferential Response Comparison** (6,600 samples, PDD dataset):

| Setting | PPO | GAPO |
|------|-----|------|
| Preferential Response | 78.5% | 82.9% |
| Preferential Prompt | 89.4% | **95.4%** |
| PP Gain over PR | +10.9% | **+12.5%** |

### Ablation Study

- **Preferential Prompt Consistently Outperforms Preferential Response**: Across all sample sizes (2k/4k/6.6k) and both optimization methods (PPO/GAPO), PP significantly outperforms PR, validating the effectiveness of learning constraints from the prompt side.
- **GAPO Achieves Better Scaling Efficiency than PPO**: From 4.2M to 13.0M tokens, GAPO improves by 24.8 percentage points (pp) under the PP setting, whereas PPO only improves by 20.9 pp.
- **Parameter Efficiency of Encoder-only RM**: A 0.4B Longformer acting as the RM outperforms traditional PPO that utilizes a 7B LLM as the RM.
- **Catastrophic Failure of DPO/SimPO/ORPO**: In the Preferential Prompt scenario, these methods collapse almost entirely (with GPT-4o scores all below 10%), because the decoder-only architecture fails to effectively capture subtle differences in constraints within the prompts.
- **Adversarial Training Dynamics**: The Reward Model improves rapidly during stages A1–A7 and converges stably after A12. The final scores of different models form a stratification between 0.2 and 0.95, indicating that training does not degenerate.

## Highlights & Insights

- **Innovative Problem Definition**: The concept of Preferential Prompt is highly elegant. While traditional methods compare response pairs under a fixed prompt, GAPO compares prompt pairs given a fixed response, enabling the model to truly understand constraints rather than merely memorizing answers. This shift in perspective is the most significant contribution of this work.
- **Architectural Choice Insight**: Utilizing a small encoder-only model as the RM instead of the LLM itself not only addresses the fundamental issue that decoder-only architectures are not adept at prompt-response matching, but also drastically reduces computational costs. At 0.4B vs. 7B parameters, it achieves even better performance.
- **Seamless Integration of GAN-PPO**: Traditional PPO requires pre-training the RM before training the Generator. In contrast, GAPO allows them to undergo adversarial training, self-generating training samples of progressive difficulty and eliminating the tedious process of manually constructing intermediate difficulty data.
- **Failure Mode Analysis of DPO-family Methods**: The experiments clearly demonstrate the catastrophic collapse of DPO/SimPO/ORPO in preferential prompt scenarios, offering valuable empirical insights into understanding the limitations of these methods.

## Limitations & Future Work

- **High Computational Overhead**: The adversarial training process, which simultaneously trains the Generator, Reward Model, and Critic Model, demands far more compute than direct optimization methods like DPO, limiting widespread deployment.
- **Dependency on Base Model Capabilities**: GAPO relies on the initial generation quality of the base model. If the base model's generations are poor, it degrades the RM's training quality, leading to a vicious cycle. Thus, it is more suitable as an enhancement tool for models that already possess strong capabilities.
- **Limited Task Diversity**: The approach was primarily validated on two tasks: Product Description Generation (PDD) and instruction-following (IFEval), lacking evaluations on more diverse constraint scenarios (such as code generation constraints, safety alignments, etc.).
- **Context Limitations of Longformer**: Utilizing a window size of 4096 tokens may restrict performance in scenarios with extremely long constraints.

## Related Work & Insights

| Method | Preference Data Type | RM Architecture | Adversarial Training | Performance in PP Setting |
|------|------------|---------|---------|-----------|
| DPO/SimPO/ORPO | Preferential Response | No RM (Implicit) | No | Catastrophic Collapse (<10%) |
| PPO (Standard) | PR or PP | Decoder-only (Frozen) | No | Good but inferior to GAPO |
| **GAPO** | **PP (Core) / PR** | **Encoder-only (Dynamic)** | **Yes** | **Optimal (95.4%)** |

- **vs. DPO/SimPO/ORPO**: These methods perform implicit reward modeling on the response side. The unidirectional attention of decoder-only LLMs cannot capture the fine differences of constraints on the prompt side, rendering them entirely ineffective in PP scenarios.
- **vs. Standard PPO**: Although standard PPO can process PP data, its RM is frozen after initial training, failing to advance its evaluation threshold alongside the Generator. The adversarial mechanism of GAPO allows the RM to continuously evolve, leading to superior final performance.
- **vs. AMaPO (AAAI 2026)**: AMaPO focuses on the adaptive margin issue within the DPO framework, still operating under the preferential response paradigm. In contrast, GAPO fundamentally shifts the subject of preference learning (prompt vs. response).

## Inspirations & Connections

- **Generalization Potential of the Preferential Prompt Paradigm**: The concept of "comparing inputs with a fixed output" could inspire automated prompt engineering—by learning which prompt variations lead to degraded output quality, it can conversely guide the design of better prompts.
- **Trend of Small Models as RMs**: Replacing general LLMs (7B+ decoder) with domain-specialized small models (0.4B encoder) as RMs aligns with the current trend of "utilizing specialized small models to assist large model training." This also inspires the exploration of heterogeneous architectures in other RL+LLM scenarios.
- **Revival of Adversarial Training in LLM Alignment**: While GANs had temporarily waned in popularity in the text generation field, GAPO revitalizes the adversarial training methodology through clever design (encoder-only RM + PPO instead of direct token-level GANs).

## Rating
- Novelty: ⭐⭐⭐⭐ A triple innovation composed of Preferential Prompt, Encoder-only RM, and GAN-PPO integration. The perspective is novel, though each individual component has known precursors.
- Experimental Thoroughness: ⭐⭐⭐⭐ The PR vs. PP comparison, scaling across different sample sizes, and empirical analysis of adversarial dynamics are meticulously conducted, though the choice of tasks is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-formulated motivations and logically designed figures and tables.
- Value: ⭐⭐⭐⭐ The Preferential Prompt paradigm and the concept of encoder-only RMs have broad inspiring potential, but the high computational overhead restricts practical adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)

</div>

<!-- RELATED:END -->
