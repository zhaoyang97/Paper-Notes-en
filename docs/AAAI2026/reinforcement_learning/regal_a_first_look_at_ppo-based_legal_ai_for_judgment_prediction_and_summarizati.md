---
title: >-
  [Paper Note] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India
description: >-
  [AAAI 2026][Reinforcement Learning][Legal AI] This paper presents the first application of PPO-based reinforcement learning (RLAIF) to Indian legal judgment prediction and summarization tasks. Although performance does not surpass SFT or commercial models, this position paper reveals fundamental challenges and future directions for RL in legal NLP.
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Legal AI"
  - "PPO"
  - "RLAIF"
  - "Judgment Prediction"
  - "Legal Document Summarization"
date: 2026-05-08
content_hash: dc70661d740706c7
---

# ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India

**Conference**: AAAI 2026  
**arXiv**: [2512.18014](https://arxiv.org/abs/2512.18014)  
**Code**: [github.com/ShubhamKumarNigam/ReGal](https://github.com/ShubhamKumarNigam/ReGal)  
**Area**: Reinforcement Learning  
**Keywords**: Legal AI, PPO, RLAIF, Judgment Prediction, Legal Document Summarization

## TL;DR

This paper presents the first application of PPO-based reinforcement learning (RLAIF) to Indian legal judgment prediction and summarization tasks. Although performance does not surpass SFT or commercial models, this position paper reveals fundamental challenges and future directions for RL in legal NLP.

## Background & Motivation

### State of the Field

Legal AI holds significant promise for the Indian judicial system, primarily addressing two key tasks:

**Court Judgment Prediction and Explanation (CJPE)**: Predicting case outcomes (accepted/rejected) and generating supporting explanations

**Legal Document Summarization**: Generating concise summaries from lengthy court judgments

Existing approaches rely predominantly on supervised fine-tuning (SFT), with the following limitations:
- Dependence on large-scale annotated datasets
- Inability to dynamically incorporate real-time feedback to enhance interpretability
- Alignment gap between model outputs and legal reasoning objectives

### Core Motivation

RLHF/RLAIF has demonstrated strong capabilities in general-purpose LLM alignment (e.g., ChatGPT), yet remains largely unexplored in the legal domain—particularly in the Indian legal context. The authors seek to answer a central question: **Can PPO-based RLAIF improve reasoning quality and interpretability in legal NLP tasks?**

This is an exploratory study focused not on achieving state-of-the-art results, but on exposing the fundamental challenges encountered when applying RL to legal text.

### Distinction from Prior Work

- Prior legal judgment prediction work (ILDC, PredEx, NyayaAnumana) exclusively employs SFT
- Existing RL-based legal summarization work (e.g., SAC-VAE, DQN) focuses mainly on extractive methods and is not tailored to Indian law
- **This paper is the first to simultaneously apply RLAIF/PPO to both Indian legal judgment prediction and abstractive summarization**

## Method

### Overall Architecture

The ReGal framework adopts a two-stage training pipeline:
1. **Stage 1: Supervised Fine-Tuning (SFT)** → Fine-tune Llama-2-7B on annotated data to obtain a reference policy $\pi^{SFT}$
2. **Stage 2: PPO Reinforcement Learning** → Optimize the SFT model via PPO using AI-generated reward signals

The overall architecture is task-agnostic; the same PPO training pipeline is flexibly applied to both judgment prediction and summarization tasks.

### Key Designs

#### 1. **Base Model and SFT Training**

Llama-2-7B is selected as the base model to enable fair comparison with prior legal judgment prediction literature that employs the same model. Instruction fine-tuning is performed separately for each task:
- CJPE task: Trained on the PredEx dataset for prediction + explanation generation
- Summarization task: Trained on the IL-TUR dataset for abstractive summarization

The fine-tuned model is denoted $\pi^{SFT}$ and serves as the reference policy for the PPO stage.

#### 2. **Task-Specific Reward Models**

Separate reward models (RMs) are constructed for each task:
- **CJPE Reward Model**: Fine-tunes InLegalBERT for prediction correctness classification. The reward is a binary signal: correct prediction → 1, incorrect → 0
- **Summarization Reward Model**: Outputs a scalar reward based on n-gram overlap (ROUGE-style) and shallow semantic similarity against gold summaries

These RMs simulate AI feedback (RLAIF), replacing costly human annotation.

#### 3. **PPO Optimization**

The language model is treated as a learnable policy $\pi_\theta$, and the PPO optimization objective is:

$$\mathcal{L}_{PPO}(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta(x)} \left[ \min \left\{ \frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)} \cdot r(y), \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)}, 1-\epsilon, 1+\epsilon\right) \cdot r(y) \right\} \right]$$

Key parameter interpretations:
- **Probability ratio $\frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)}$**: Measures the deviation of the current policy from the SFT baseline
- **Clipping parameter $\epsilon=0.1$**: Constrains the magnitude of policy updates to maintain training stability
- **$r(y)$**: Task-specific reward signal from the RM
- Taking the minimum of both terms ensures conservative updates, preventing instability from large policy changes

### Loss & Training

- Learning rate: 1.41e-5
- Maximum PPO epochs: 1
- Training batch size: 4; mini-batch size: 2
- Output length constraint: 100–500 tokens
- Mixed-precision training (GradScaler) used for GPU memory efficiency
- Training hardware: NVIDIA A100 80GB; total compute cost approximately $100
- Maximum new tokens at inference: 500

## Key Experimental Results

### Datasets

| Dataset | Scale | Task | Avg. Document Length |
|---------|-------|------|----------------------|
| PredEx | 15,222 samples (12,178/3,044) | Judgment prediction + explanation | ~4,500 tokens |
| In-Abs | 7,130 samples (7,030/100) | Legal summarization | ~4,377 words |

### Main Results

**Judgment Prediction and Explanation Task (PredEx dataset):**

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore | BLANC |
|-------|---------|---------|---------|------|--------|-----------|-------|
| GPT-3.5 Turbo | - | - | - | - | - | - | - |
| LLaMA-2 SFT | **0.50** | **0.43** | **0.44** | **0.25** | **0.36** | **0.69** | **0.28** |
| ReGal (PPO) | 0.19 | 0.04 | 0.12 | 0.01 | 0.10 | 0.50 | 0.02 |

**ILDC Expert dataset:**

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore | BLANC |
|-------|---------|---------|---------|------|--------|-----------|-------|
| GPT-3.5 Turbo | **0.54** | **0.43** | **0.45** | **0.28** | **0.47** | **0.73** | **0.34** |
| LLaMA-2 SFT | 0.49 | 0.38 | 0.40 | 0.29 | **0.51** | 0.69 | 0.36 |
| ReGal (PPO) | 0.25 | 0.05 | 0.16 | 0.01 | 0.16 | 0.50 | 0.03 |

### Ablation Study

**Comparison of inference strategies (PredEx & In-Abs):**

| Strategy | R1 (PredEx) | R1 (In-Abs) | BLEU (PredEx) | METEOR (In-Abs) |
|----------|------------|------------|--------------|----------------|
| Vanilla | 0.39 | **0.47** | 0.07 | **0.34** |
| SFT | **0.42** | 0.44 | **0.12** | 0.34 |
| DPO | 0.38 | 0.44 | 0.08 | 0.34 |
| PPO | 0.30 | 0.41 | 0.05 | 0.31 |

**Base model variant ablation:**
- Phi-3 Mini (smaller model): Performance degrades substantially; unable to handle long and complex legal texts
- LLaMA-2-7B without SFT: Lacking legal domain adaptation, output quality drops significantly

**Reward model variant ablation:**
- Replacing the fine-tuned RM with a pretrained InLegalBERT (not task-specific): PPO performance deteriorates further, producing more incoherent outputs

### Key Findings

1. **PPO consistently underperforms SFT and commercial models in legal NLP**: ReGal scores significantly lower than LLaMA-2 SFT across all metrics
2. **Objective misalignment is the core bottleneck**: The SFT policy is not sufficiently optimized for legal reasoning, leading to a suboptimal PPO starting point
3. **Reward model alignment with legal text is difficult**: The nuance and specialization of legal language exceed the RM's ability to capture
4. **Severe hallucination problem**: The PPO model fabricates legal principles and invents case citations, especially when input is insufficient
5. **RM precision directly determines the PPO ceiling**: A fundamental gap exists between the granularity of reward models and the complexity of legal reasoning

## Highlights & Insights

1. **An honest position paper**: Rather than pursuing SOTA, the paper candidly exposes why RL fails in legal AI, providing valuable lessons for future research
2. **Cross-task validation**: Findings are validated across two structurally distinct tasks—judgment prediction and summarization—strengthening the reliability of conclusions
3. **Systematic failure analysis**: Root causes are analyzed across eight dimensions including objective misalignment, RM limitations, legal complexity, data constraints, and hyperparameter choices
4. **Hallucination case studies**: Concrete examples illustrate the risks of legal AI hallucination (e.g., fabricated privacy law citations)

## Limitations & Future Work

1. **RMs require stronger legal alignment**: High-quality reward models capable of capturing the nuances of legal reasoning must be developed
2. **Incorporate human feedback (RLHF)**: RLAIF alone is insufficient to guide legal text generation; involvement of legal experts is necessary
3. **Legal domain pretraining**: Base models require deeper domain-adaptive pretraining in the legal domain
4. **Anti-hallucination mechanisms**: Factual constraints or hallucination-aware rewards are needed
5. **Expansion to multi-jurisdictional data**: The current dataset covers only the Indian Supreme Court, limiting generalizability

## Related Work & Insights

- **PredEx, ILDC datasets**: Benchmark datasets for Indian legal judgment prediction
- **InLegalBERT**: A pretrained language model for the Indian legal domain
- **RLAIF (Lee et al., 2023)**: A cost-effective alternative to human feedback using AI-generated signals
- Insight: **Legal AI may require a more carefully designed RLHF pipeline** rather than a straightforward application of generic RLAIF frameworks; the quality of reward signals is the decisive factor in the success or failure of RL methods in the legal domain

## Rating

- Novelty: ⭐⭐⭐ (First application of PPO-RLAIF in the Indian legal context)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-dataset evaluation, ablation studies, and failure analysis are comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Clear positioning as a position paper; analysis is candid and well-structured)
- Value: ⭐⭐⭐ (Negative results are equally valuable in pointing future research directions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[ICLR 2026\] Look-ahead Reasoning with a Learned Model in Imperfect Information Games](../../ICLR2026/reinforcement_learning/look-ahead_reasoning_with_a_learned_model_in_imperfect_information_games.md)
- [\[ICLR 2026\] Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning](../../ICLR2026/reinforcement_learning/escaping_policy_contraction_contraction-aware_ppo_cappo_for_stable_language_mode.md)

</div>

<!-- RELATED:END -->
