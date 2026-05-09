---
title: >-
  [Paper Note] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning
description: >-
  [NeurIPS 2025][LLM Safety][contextual integrity] This paper proposes CI-RL, a framework that combines Chain-of-Thought reasoning prompts with GRPO reinforcement learning to train LLMs to understand contextual integrity (CI) using only ~700 synthetic samples. On the PrivacyLens benchmark, it reduces privacy leakage rates by up to 40%, and smaller models trained with CI-RL can surpass larger baseline models.
tags:
  - NeurIPS 2025
  - LLM Safety
  - contextual integrity
  - privacy
  - reinforcement-learning
  - GRPO
  - chain-of-thought
  - information disclosure
date: 2026-05-08
content_hash: dd35fb4ffbd9569a
---

# Contextual Integrity in LLMs via Reasoning and Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.04245](https://arxiv.org/abs/2506.04245)
**Code**: [EricGLan/CI-RL](https://github.com/EricGLan/CI-RL)
**Area**: AI Safety
**Keywords**: contextual integrity, privacy, reinforcement-learning, GRPO, chain-of-thought, information disclosure

## TL;DR

This paper proposes CI-RL, a framework that combines Chain-of-Thought reasoning prompts with GRPO reinforcement learning to train LLMs to understand contextual integrity (CI) using only ~700 synthetic samples. On the PrivacyLens benchmark, it reduces privacy leakage rates by up to 40%, and smaller models trained with CI-RL can surpass larger baseline models.

## Background & Motivation

**Background**: LLM agents are acquiring increasing autonomy (booking appointments, sending emails, managing files), requiring them to interact with the external world on behalf of users and inevitably access and process personal information.

**Limitations of Prior Work**: (a) LLMs lack understanding of contextual integrity (CI)—i.e., what information is appropriate to share within a given context and what is not; (b) even without adversarial attacks, models may inadvertently disclose irrelevant sensitive information; (c) restricting information access is often impractical (e.g., RAG systems require broad access to user files).

**Key Challenge**: LLMs possess knowledge about privacy and sensitive information, yet fail to consistently make correct information disclosure judgments under contextual nuance. This is fundamentally a reasoning problem—the model must reason about which information flows are appropriate within a given context.

**Goal**: (a) Can LLM reasoning capabilities be explicitly guided to judge the appropriateness of information disclosure? (b) Can such reasoning be further reinforced via reinforcement learning? (c) Can capabilities trained on small-scale synthetic data transfer to real-world benchmarks?

**Key Insight**: CI is inherently a reasoning task, analogous to mathematical or code reasoning—the model must analyze the context, evaluate the relevance of each attribute, and make disclosure decisions. This makes the CoT reasoning + RL paradigm a natural fit.

**Core Idea**: Teach LLMs to respect information boundaries while completing tasks by combining explicit CoT reasoning over contextual norms with GRPO reinforcement learning optimized via rule-based reward signals.

## Method

### Overall Architecture

The method consists of three components: (1) **CI-CoT**: a structured prompt template that guides the model to reason about contextual integrity within `<think>` tags before producing a response in `<answer>` tags; (2) **Synthetic Dataset Construction**: a three-stage pipeline generating ~700 training samples covering diverse scenarios, domains, and transmission principles; (3) **CI-RL Training**: reinforcement learning using the GRPO algorithm with rule-based reward functions.

### Key Designs

1. **CI-CoT Reasoning Template**:

   - **Function**: Explicitly guides the model to reason about the contextual appropriateness of each information attribute before responding.
   - **Mechanism**: The prompt template instructs the model to analyze the task context within `<think>...</think>`, assess each personal attribute as "necessary / helpful / optional / inappropriate," and then complete the task within `<answer>...</answer>` using only contextually appropriate information.
   - **Design Motivation**: Inspired by the success of CoT in mathematical reasoning, CI judgments are made explicit as reasoning steps rather than leaving the model to decide implicitly.

2. **Three-Stage Synthetic Dataset Pipeline**:

   - **Function**: Automatically generates diverse CI training scenarios.
   - **Mechanism**: Stage 1 (Initial Seeds): sample combinations of scenario (email/chat) × domain (10 types including medical/financial/education) × transmission principle (confidentiality/proportionality/consent) to produce random seeds; Stage 2 (Vignettes): GPT-4 expands seeds into complete scenarios, populating CI fields (sender/recipient/subject) and generating required/restricted information types; Stage 3 (Final Samples): GPT-4 instantiates vignettes into natural conversational training samples (key-value pairs + flow annotations + keyword match labels).
   - **Design Motivation**: Manual annotation of CI samples is costly and cannot cover a sufficient range of scenarios; synthetic data efficiently explores the scenario space.

3. **GRPO Reinforcement Learning with Rule-Based Rewards**:

   - **Function**: Further optimizes the model's CI reasoning capabilities via RL.
   - **Mechanism**: The GRPO algorithm (no critic network required) is used with the objective:

   $$J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|q)}{\pi_{\text{old}}(a_i|q)}A_i, \text{clip}(\cdot)A_i\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

   The reward function $R$ consists of two components—a format reward (presence of correct think/answer tags) and a CI score:

   $$R = \frac{|A_{\text{present}}|}{|A|} - \frac{|D_{\text{present}}|}{|D|}$$

   where $A$ is the set of required keywords and $D$ is the set of restricted keywords. Higher scores are awarded for including more required information; larger penalties are incurred for leaking more restricted information.
   - **Design Motivation**: Rule-based rewards are more stable and controllable than reward models; GRPO eliminates the critic network to reduce computational overhead; advantage estimation uses within-group normalization $A_i = (r_i - \text{mean}(r)) / \text{std}(r)$.

### Loss & Training

Training uses the VERL framework with 590 training / 66 validation / 73 test samples. The best checkpoint is selected on the validation set, then evaluated on the test set and PrivacyLens. Multiple model families are supported (Qwen2.5 1.5B/3B/7B/14B, Llama-3.1-8B, Mistral-7B).

## Key Experimental Results

### Main Results — Synthetic Test Set

| Model | Integrity ↑ | Utility ↑ | Complete ↑ |
|-------|:-----------:|:---------:|:----------:|
| Qwen2.5-1.5B | 37.5% | 35.9% | 4.7% |
| + CI-RL | **59.4%** | **43.7%** | **26.6%** |
| Qwen2.5-7B | 46.9% | 62.5% | 29.7% |
| + CI-RL | **75.0%** | **67.2%** | **48.4%** |
| Mistral-7B | 38.8% | 67.3% | 24.5% |
| + CI-RL | **89.1%** | **82.8%** | **73.4%** |
| Llama-3.1-8B | 61.9% | 64.3% | 38.1% |
| + CI-RL | **79.7%** | **79.7%** | **62.5%** |
| Qwen2.5-14B | 51.6% | 67.2% | 37.5% |
| + CI-RL | **78.1%** | **64.1%** | **50.0%** |

### PrivacyLens Benchmark Leakage Rates

| Model | LR ↓ | ALR ↓ | Helpful [0-3] ↑ |
|-------|:----:|:-----:|:---------------:|
| Claude 3.7 Sonnet | 30.4% | 35.9% | 2.49 |
| + CI-CoT | **23.1%** | **25.4%** | **2.69** |
| Gemini 2.5 Pro | 37.3% | 38.2% | 2.84 |
| + CI-CoT | **25.3%** | **26.9%** | 2.72 |
| Qwen2.5-7B | 50.3% | 52.4% | 1.99 |
| + CI-RL | **33.7%** | **33.9%** | **2.08** |
| Mistral-7B | 47.9% | 52.1% | 1.78 |
| + CI-RL | **31.2%** | **29.6%** | **1.84** |

### Key Findings

- **Consistent gains from CI-RL**: All models show significant improvements in Integrity and Complete metrics after training while maintaining Utility.
- **Smaller models outperform larger baselines**: Qwen2.5-7B + CI-RL (Integrity 75.0%) surpasses the Qwen2.5-14B baseline (51.6%), demonstrating that RL can bridge or even reverse the scale gap between models.
- **Successful synthetic-to-real transfer**: Training on only ~700 synthetic samples yields up to 40% reduction in leakage rate on the human-annotated PrivacyLens benchmark.
- **Unexpected finding on LRMs vs. LLMs**: DeepSeek-R1 distilled models underperform instruction-tuned LLMs on CI tasks, likely because the distilled models are biased toward scientific and code domains.
- **CI-CoT is effective for frontier models**: Even frontier models such as Claude 3.7 and Gemini 2.5 show significant reductions in privacy leakage rates when augmented with CI-CoT prompting.

## Highlights & Insights

- **CI as a reasoning problem**: Reframing privacy protection from an "alignment/fine-tuning" paradigm to a "reasoning" paradigm is the key insight. CoT enables the model to explicitly deliberate—"Is this information appropriate in the current context?"—before generating output, rather than relying on implicit safety training.
- **Data-efficient training with ~700 samples**: The successful transfer to real-world benchmarks using only ~700 synthetic samples demonstrates that the emergence of CI reasoning capability does not require massive data.
- **Minimalist reward function design**: The entirely rule-based keyword-matching reward avoids the training cost and bias associated with reward models while achieving substantial improvements.
- **Quantitative analysis of the safety-utility tradeoff**: The Adjusted Leakage Rate (ALR) metric, which counts leakage only among helpful responses, provides a fairer evaluation of conservative strategies.

## Limitations & Future Work

- Scenario coverage of the synthetic data remains limited; CI judgment in complex multi-turn dialogues is not addressed.
- The keyword-matching reward function may miss semantically equivalent information leakage (e.g., implying information through context rather than direct mention).
- CI norms are inherently social, subjective, and evolve over time; how models should adapt to dynamic norms is not discussed.
- Evaluation is restricted to English scenarios; cross-lingual and cross-cultural variations in CI are not considered.
- The additional reasoning overhead (generating long CoT) may be unsuitable for latency-sensitive agent deployments.

## Related Work & Insights

- **vs. PrivacyLens (Shao et al., 2024)**: PrivacyLens provides an evaluation benchmark and leakage taxonomy; this work builds on it by offering a training methodology to reduce leakage.
- **vs. DeepSeek-R1**: Both employ GRPO for reasoning-oriented RL, but this work applies the paradigm to privacy rather than mathematical or code reasoning.
- **vs. AirGapAgent**: AirGapAgent protects privacy by restricting information access; this work teaches models to autonomously judge information appropriateness through reasoning. The two approaches are complementary.
- The work has direct implications for the safe deployment of LLM agents; CI reasoning should become a core component of the alignment process.

## Rating

- Novelty: ⭐⭐⭐⭐ First explicit application of RL to CI reasoning; the CoT + GRPO combination is clean and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-scale, multi-benchmark evaluation including frontier model comparisons and ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, methodology is well-structured, and experimental design is sound.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to the safe deployment of agents; the method is lightweight and transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning with Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)
- [\[NeurIPS 2025\] Probabilistic Reasoning with LLMs for K-Anonymity Estimation](probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)
- [\[NeurIPS 2025\] Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](teaming_llms_to_detect_and_mitigate_hallucinations.md)

</div>

<!-- RELATED:END -->
