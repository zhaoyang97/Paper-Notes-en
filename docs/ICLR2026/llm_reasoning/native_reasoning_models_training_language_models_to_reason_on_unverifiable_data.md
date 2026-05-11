---
title: >-
  [Paper Note] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data
description: >-
  [ICLR2026][LLM Reasoning][Reasoning training] This paper proposes NRT (Native Reasoning Training), a framework that treats reasoning chains as latent variables and uses the model's own predictive confidence over referenc…
tags:
  - "ICLR2026"
  - "LLM Reasoning"
  - "Reasoning training"
  - "verifier-free RL"
  - "latent variable reasoning"
  - "GRPO"
  - "reward design"
date: 2026-05-08
content_hash: 16b8bab2c61ba4fc
---

# Native Reasoning Models: Training Language Models to Reason on Unverifiable Data

**Conference**: ICLR2026
**arXiv**: [2602.11549](https://arxiv.org/abs/2602.11549)
**Code**: To be confirmed
**Area**: LLM Reasoning
**Keywords**: Reasoning training, verifier-free RL, latent variable reasoning, GRPO, reward design

## TL;DR
This paper proposes NRT (Native Reasoning Training), a framework that treats reasoning chains as latent variables and uses the model's own predictive confidence over reference answers as an intrinsic reward signal to train LLM reasoning—without external verifiers or expert reasoning demonstrations. On Llama-3.1-8B, NRT achieves an average improvement of 10.2 points across 9 benchmarks (46.0→56.2), surpassing the verifier-dependent RLPR by +5.4 points.

## Background & Motivation

**Background**: Current approaches to improving LLM reasoning follow two main paths—(a) SFT on human- or GPT-4-annotated reasoning chains (e.g., o1 reproductions), and (b) RL with external verifiers (RLVR), such as using final answer correctness as a reward for math problems. Both perform well in verifiable domains such as mathematics and code.

**Limitations of Prior Work**: Many subject-area tasks (history, commonsense, open-domain QA, multi-hop reasoning) have answers that **cannot be programmatically verified**—no deterministic verifier can assess whether the reasoning process is correct. Such "unverifiable data" constitutes the majority of real-world applications, yet existing RLVR methods are inapplicable.

**Key Challenge**: Reasoning ability requires RL training; RL requires reward signals; traditional rewards come from external verifiers—but no such verifiers exist in unverifiable domains. The core question is how to train reasoning without external rewards.

**Goal**: Given only (question, answer) pairs—without reasoning demonstrations or external verifiers—how can one train an LLM to generate effective reasoning chains?

**Key Insight**: Treating reasoning chains $z$ as **latent variables**—a good reasoning chain should increase the model's predicted probability of the correct answer $y^*$. The reward is defined as the token-level probability of the answer predicted by the model after reading the reasoning chain.

**Core Idea**: Using "whether a reasoning chain helps the model better predict the answer" as an intrinsic reward, without relying on any external verification—the model acts simultaneously as reasoner and evaluator.

## Method

### Overall Architecture
Input (question $x$, answer $y^*$) → model samples reasoning chain $z \sim \pi_\theta(z|x)$ → compute token-level probability $c_i = \pi_\theta(y^*_i|x,z,y^*_{<i})$ of predicting $y^*$ conditioned on $z$ → aggregate probabilities into a trace-level reward $R(z,\theta)$ → update $\theta$ via GRPO policy gradient.

### Key Designs

1. **Latent Variable Reasoning Paradigm**:

    - Function: Reasoning chains $z$ are neither externally annotated nor externally evaluated; the model generates and assesses them itself.
    - Mechanism: A good reasoning chain $z$ should increase $\pi_\theta(y^*|x,z)$—i.e., the model should be more "confident" in the correct answer after reading the reasoning.
    - Design Motivation: This is the only self-consistent approach that does not depend on external verification—the model serves as both student and teacher.

2. **Weighted Sum Reward (WS)**:

    - Function: Uses a weighted sum of token-level probabilities as the reward, with weights inversely proportional to the baseline difficulty of each token.
    - Mechanism: Inverse-probability weighting $w_i \propto 1/c_{i,base}$ drives weights for simple tokens (e.g., "the") toward zero while amplifying weights for difficult tokens (e.g., key factual words).
    - Design Motivation: Standard logP rewards are dominated by simple tokens, preventing the model from learning to improve difficult predictions. The $-\log p$ weighting scheme outperforms logP by 3.3 points on Llama-3.1-8B.
    - Theoretical Connection: The $-\log p$ weighting is equivalent to cross-entropy $-\sum c_j \log c_{j,base}$, directly optimizing the reduction of KL divergence on difficult tokens.

3. **Reward Stabilization**:

    - Function: Clipped reward $R' = \max(0, R - R_{base})$ + group-wise normalization.
    - Mechanism: Subtracting the baseline (reward without a reasoning chain) differentiates reward signals; group-wise normalization stabilizes GRPO gradients.
    - Design Motivation: Methods such as RLPR suffer from severe policy collapse (reasoning chain entropy→0, quality→0); NRT maintains high entropy and high quality throughout training.

4. **Format Supervision Loss**:

    - Function: An auxiliary loss ensures the model output contains reasoning wrapped in `<think>...</think>` tags.
    - Weight of 0.3 prevents the model from skipping reasoning and directly outputting answers.

### Loss & Training
$J(\theta) = \mathbb{E}_{z \sim \pi_\theta}[R(z,\theta)]$, optimized via GRPO with importance sampling. The gradient decomposes into a trace policy gradient (reinforcing the entire reasoning chain) and a token prediction gradient (token-level weighted prediction updates). Training data comprises 200K samples from tulu-3-sft-mixture, with an average response length of 415 tokens.

## Key Experimental Results

### Main Results

**Llama-3.1-8B across 9 benchmarks**:

| Method | BBH | MMLU | DROP | GSM8K | MATH | HumanEval | IFEval | Overall Avg. |
|--------|-----|------|------|-------|------|-----------|--------|-------------|
| SFT | 38.0 | 59.2 | 36.7 | 29.0 | 17.8 | 74.7 | 58.3 | 46.0 |
| RLPR* | 41.2 | 58.7 | 32.5 | 65.0 | 27.8 | 77.8 | 61.3 | 50.8 |
| Verifree* | 35.7 | 58.3 | 33.5 | 54.3 | 19.4 | 76.3 | 59.3 | 48.1 |
| NRT-GM | 54.3 | 66.1 | 48.7 | 70.3 | 32.2 | 76.3 | 55.3 | 54.9 |
| **NRT-WS(-logp)** | **51.0** | **66.7** | **52.2** | **76.0** | **30.7** | **77.8** | **59.0** | **56.2** |

**Llama-3.2-3B**:

| Method | Overall Avg. |
|--------|-------------|
| SFT | 36.4 |
| NRT-WS(-logp) | **39.9** (+3.5) |

### Ablation Study

| Reward Aggregation | Llama-3.1-8B Overall |
|-------------------|---------------------|
| logP (log probability) | 52.9 |
| P (probability product) | 51.4 |
| GM (geometric mean) | 54.9 |
| AM (arithmetic mean) | 53.3 |
| WS-1/p (inverse-probability weighting) | 53.3 |
| **WS-(-logp)** | **56.2** |

### Key Findings
- **Policy Collapse Resolved**: Under RLPR, reasoning chain entropy rapidly collapses to 0 during training; NRT maintains high entropy and high-quality reasoning throughout.
- **Targeted Improvement on Difficult Tokens**: The WS weighting scheme yields probability improvements of up to 15% on high-entropy tokens, while RLPR shows virtually no improvement on the same tokens.
- **No Verifiable Data Required**: Large gains on both GSM8K (math, verifiable) and BBH (reasoning, unverifiable) demonstrate that the method is not domain-restricted.
- **Decoupling of Reasoning and Answer**: Lexical analysis reveals that the model autonomously learns to use meta-cognitive vocabulary (e.g., "premise," "reasoning") in reasoning chains while suppressing answer-format tokens.

## Highlights & Insights
- **Paradigm Innovation: Latent Variable Reasoning**: Treating reasoning chains as latent variables and using the model's own predictive confidence as a reward is an elegant design—it requires no external annotation or verifier and extends RL-based reasoning training to all domains.
- **Theoretical Intuition Behind Difficult-Token Weighting**: The $-\log p$ weighting concentrates the reward on tokens where the model is most uncertain, which is consistent with the spirit of curriculum learning and hard example mining. This simple modification yields a significant gain of 3.3 points.
- **Diagnosis and Resolution of Policy Collapse**: The paper clearly demonstrates the collapse phenomenon in RLPR (reasoning entropy→0) and naturally avoids it through intrinsic reward design—collapsed reasoning cannot help predict the answer.

## Limitations & Future Work
- **Hand-crafted Reward Functions**: All five aggregation methods and weighting schemes are manually specified; automatic reward function learning is a promising direction.
- **Limited Sampling Efficiency**: RL training requires extensive sampling (GRPO requires multiple reasoning chains per group), incurring high computational cost.
- **Restricted to Fine-Tuning**: Validation at the pre-training stage is absent; incorporating reasoning training during pre-training may yield further gains.
- **Hallucination Risk**: Case studies show the model may generate non-existent program names in open-domain tasks—intrinsic rewards cannot prevent factual errors.

## Related Work & Insights
- **vs. RLPR (Reasoning via Planning with RL)**: RLPR uses external answer-matching rewards and collapses on unverifiable tasks. NRT uses intrinsic predictive confidence as a reward and is applicable across all settings.
- **vs. Verifree**: Prior verifier-free methods employ simpler reward designs; NRT's token-level weighting scheme achieves substantially better performance (+8.1 on Llama-8B).
- **vs. STaR/Self-Improvement**: STaR relies on filtering reasoning chains by correctness for SFT; NRT directly optimizes reasoning quality via RL, avoiding the distribution-matching issues of SFT.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The latent variable reasoning paradigm and intrinsic reward design represent a fundamentally new perspective, resolving the reasoning training problem in verifier-free domains.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three models × 9 benchmarks × 5 reward variants, with full coverage of training dynamics analysis, token-level analysis, and case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper progresses systematically from problem formulation to theoretical derivation to experimental analysis, with clear mathematical exposition.
- Value: ⭐⭐⭐⭐⭐ Addresses the most critical bottleneck in current reasoning training—extending RL-based reasoning from verifiable domains to arbitrary domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](../../CVPR2026/llm_reasoning/understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ICLR 2026\] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)

</div>

<!-- RELATED:END -->
