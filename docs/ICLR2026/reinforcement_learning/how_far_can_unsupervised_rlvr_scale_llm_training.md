---
title: >-
  [Paper Note] How Far Can Unsupervised RLVR Scale LLM Training?
description: >-
  [ICLR 2026][Reinforcement Learning][unsupervised RLVR] This paper presents a comprehensive analysis of Unsupervised Reinforcement Learning from Verifiable Rewards (URLVR)…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "unsupervised RLVR"
  - "model collapse"
  - "intrinsic rewards"
  - "sharpening mechanism"
  - "test-time training"
date: 2026-05-08
content_hash: 39ccb591dc85f63b
---

# How Far Can Unsupervised RLVR Scale LLM Training?

**Conference**: ICLR 2026
**arXiv**: [2603.08660](https://arxiv.org/abs/2603.08660)
**Code**: [PRIME-RL/TTRL](https://github.com/PRIME-RL/TTRL)
**Area**: Reinforcement Learning
**Keywords**: unsupervised RLVR, model collapse, intrinsic rewards, sharpening mechanism, test-time training

## TL;DR

This paper presents a comprehensive analysis of Unsupervised Reinforcement Learning from Verifiable Rewards (URLVR), demonstrating that all intrinsic reward methods fundamentally operate as a "sharpening" mechanism over the model's initial distribution, leading to an inevitable rise-then-fall collapse pattern. It proposes the Model Collapse Step as a prior-based model indicator and identifies external reward methods as the key direction for overcoming scalability bottlenecks.

## Background & Motivation

RLVR (Reinforcement Learning from Verifiable Rewards) has been a core driver of recent breakthroughs in LLM reasoning capabilities (e.g., DeepSeek-R1, Gemini 2.5, Qwen3). However, supervised RLVR relies on high-quality annotated datasets, and as model capabilities approach or surpass human-level performance, obtaining reliable ground truth becomes increasingly difficult and costly — this constitutes the **supervision bottleneck**.

**Unsupervised RLVR (URLVR)** emerges as a response, aiming to provide reward signals without ground truth labels. Analogous to how pre-training scaling laws convert large amounts of unlabeled data into intelligence, URLVR seeks to extend this paradigm to the post-training stage.

However, existing URLVR methods (e.g., TTRL, RLIF, EM-RL) — while reporting initial performance gains — also exhibit reward hacking and model collapse. The lack of systematic comparison across different methods and settings raises a fundamental question: **can intrinsic rewards truly scale LLM training?**

## Method

### Overall Architecture

This paper establishes a **taxonomy–theory–experiment** unified analytical framework for URLVR:

1. **Taxonomy**: Categorizes URLVR methods into **Intrinsic Rewards** and **External Rewards**
2. **Theoretical Analysis**: Derives the "Sharpening Mechanism" underlying intrinsic rewards
3. **Systematic Experiments**: Validates the rise-then-fall pattern and proposes the Model Collapse Step metric

### Key Designs

1. **Taxonomy of Intrinsic Reward Methods**:

    - **Certainty-Based**: Rewards derived from the model's own confidence (logits), including Self-Certainty (KL divergence from uniform distribution), Token-Level Entropy (negative entropy), Trajectory-Level Entropy (sequence log-probability), and Probability (product of probabilities)
    - **Ensemble-Based**: Rewards based on consistency across multiple rollouts, e.g., Majority Voting in TTRL

2. **Unified Reward Framework**: All intrinsic rewards can be unified as $r_{uni}(x,y) = \psi(\frac{\sigma}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \mathbb{H}(q_i, \pi_\theta^i))$, where $\mathcal{I}$ denotes aggregation granularity, $q$ is the anchor distribution, $\sigma \in \{+1,-1\}$ is a sign factor, and $\psi$ is a monotonic transformation. Different methods are simply distinct instantiations of these four components.

3. **Theoretical Proof of the Sharpening Mechanism** (using TTRL as an example):

    - The optimal policy under the KL-regularized RL objective is $\pi_\theta^*(y|x) \propto \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x,y))$
    - Under binary majority voting rewards, the probability mass on the majority answer is amplified by a factor of $e^{1/\beta}$
    - **Theorem 1**: Under majority stability and effective learning assumptions, $p_{maj}^{(k)}$ converges to 1 at a geometric rate $\rho = e^{-1/\beta}$, and the policy converges to a deterministic distribution concentrated on the initial majority answer

### Loss & Training

- Training uses the veRL framework with the GRPO algorithm
- Default configuration: temperature 1.0, batch size 64, 8 rollouts, no KL regularization
- Base model: Qwen3-1.7B-Base; training set: DAPO-17k

## Key Experimental Results

### Main Results: Rise-Then-Fall Pattern

Across systematic experiments spanning 5 intrinsic reward methods × 4 hyperparameter configurations:

| Method | Collapse Mode | Characteristics |
|--------|--------------|-----------------|
| Self-Certainty | Gradual degradation | Most stable; slowest degradation; Label Accuracy remains relatively high |
| Majority Voting | Gradual degradation | Operates at the answer level, avoiding token-level artifacts |
| Probability | Length collapse | Reward favors short sequences; model output length drops sharply |
| Token-Level Entropy | Repetition collapse | Minimizes entropy by repeating high-probability tokens |
| Trajectory-Level Entropy | Repetition collapse | Same as above; repetitive text fills generated sequences |

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| Hyperparameter tuning | All settings eventually collapse; differences are in *when*, not *whether* collapse occurs |
| Single-question training | Only 3 of 25 questions (12%) flipped correctness; the rest merely sharpened existing preferences |
| Dataset size (32→16384) | ≤128 samples prevent collapse; ≥512 samples lead to inevitable collapse |
| Test-time vs. train-time | Test-time training on small datasets is safe and effective; train-time on large datasets inevitably collapses |
| 32 samples with all-wrong initial majority votes | Still improves OOD performance (AIME24/AMC23) |

### Model Collapse Step

| Model | Collapse Step | GT Gain | Correlation |
|-------|--------------|---------|-------------|
| OLMo series (weak prior) | ~14–22 steps | Low | Strong positive |
| LLaMA series (moderate) | ~19–128 steps | Moderate | Strong positive |
| Qwen series (strong prior) | ~172–195 steps | High | Strong positive |

Computational cost comparison: Model Collapse Step requires only **1/5.6** the computation of GT Gain, and requires no ground truth labels.

### External Rewards: Self-Verification Experiments

| Method | Verification Accuracy (Qwen3-1.7B) | Characteristics |
|--------|-------------------------------------|-----------------|
| Trajectory-Level Entropy | ~40% then collapse | Inherent scalability limitations |
| Self-Verification | ~65% and continuing to rise | Reward Accuracy initially drops then recovers |
| Oracle Supervision | ~70% | Upper bound |

### Key Findings

- **Sharpening nature of all intrinsic reward methods**: Regardless of design differences, all methods converge toward the model's initial distribution
- **Rise-then-fall is intrinsic, not an engineering issue**: Even with optimal hyperparameter combinations, collapse occurs after ~1,000 steps (~4 epochs)
- **Unique value of small datasets**: ≤128 samples avoid collapse through local overfitting rather than global policy shift (KL divergence of only 0.057, vs. more than 2× for large datasets)
- **Counter-intuitive OOD generalization**: Models that answer all training questions incorrectly can still improve test set performance
- **Model prior determines everything**: Qwen > LLaMA; SFT > Base; smaller models are paradoxically more stable
- **Instruction alignment is critical for Self-Verification**: Instruction-aligned models start at 60%+ and are robust to both prompt types; base models are only effective with one prompt type

## Highlights & Insights

- **Unified theoretical framework**: Unifies seemingly diverse intrinsic reward methods as different instantiations of cross-entropy manipulation, revealing their essential equivalence
- **Theoretical proof of "sharpening" as the core mechanism**, redefining the understanding of URLVR — it amplifies existing preferences rather than acquiring new knowledge
- **Model Collapse Step**: A simple, low-cost, GT-free metric for predicting RL trainability, with direct practical value for model selection
- **Bridging theory and practice**: A compelling correspondence between Theorem 1 and the empirically observed rise-then-fall pattern

## Limitations & Future Work

- Experiments are primarily conducted on mathematical reasoning tasks; generalizability to other domains (code, open-ended dialogue, etc.) remains to be verified
- Self-Verification experiments are validated only on the simple Countdown task; more complex scenarios require further exploration
- Systematic investigation of external reward methods is relatively limited, constituting only "preliminary evidence"
- Theoretical assumptions (majority stability, effective learning) are not always satisfied in practice
- Effective combination of intrinsic and external rewards remains unexplored

## Related Work & Insights

This paper spans multiple research directions, including RLVR (DeepSeek-R1, Qwen3, etc.), unsupervised/self-supervised learning, and test-time training. Specific methods such as TTRL, RLIF, and EM-RL are analyzed within the unified sharpening framework. The exploration of Self-Verification suggests a path from the confidence–correctness ceiling of intrinsic rewards toward external rewards — particularly directions that exploit the generation–verification asymmetry (e.g., formal verification, code execution), which hold significant promise.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Unified theoretical framework + Model Collapse Step + systematic experimental analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 methods × 4 hyperparameter settings × multiple model families × multiple datasets)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure; complete logical chain from theory to experiment to practice)
- Value: ⭐⭐⭐⭐⭐ (Provides a clear roadmap and practical tools for the URLVR field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)
- [\[ICLR 2026\] How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](../../ACL2026/reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ICLR 2026\] SUSD: Structured Unsupervised Skill Discovery through State Factorization](susd_structured_unsupervised_skill_discovery_through_state_factorization.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)

</div>

<!-- RELATED:END -->
