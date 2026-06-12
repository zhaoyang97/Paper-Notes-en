---
title: >-
  [Paper Note] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Test-Time RL] This paper proposes the Self-Harmony framework, in which a single model plays two roles—a Solver that addresses the original problem and a Reframer that paraphrases it—an…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Test-Time RL"
  - "Self-Play"
  - "Pseudo-Label"
  - "Harmonic Mean"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: 18f393a27aa5f73b
---

# Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2511.01191](https://arxiv.org/abs/2511.01191)  
**Code**: [Self-Harmony](https://github.com/) (marked as publicly available in the paper)  
**Area**: LLM Reasoning / Test-Time Reinforcement Learning
**Keywords**: Test-Time RL, Self-Play, Pseudo-Label, Harmonic Mean, LLM Reasoning

## TL;DR

This paper proposes the Self-Harmony framework, in which a single model plays two roles—a Solver that addresses the original problem and a Reframer that paraphrases it—and uses the harmonic mean of answer scores across both perspectives as a pseudo-label selection criterion, replacing conventional majority voting. The approach achieves state-of-the-art performance in 28 out of 30 experimental settings with zero training failures.

## Background & Motivation

**Test-Time Reinforcement Learning (TTRL) as a new paradigm for LLM reasoning**: TTRL enables models to self-improve during inference by leveraging unlabeled test data through self-generated feedback signals, without requiring human-annotated data or auxiliary models.

**Fatal flaw of majority voting**: When a model exhibits systematic reasoning defects, incorrect answers may appear more frequently than correct ones. In such cases, majority voting not only fails to correct errors but amplifies them by selecting wrong answers as training targets—creating an "echo chamber" effect. Liu et al. (2025b) theoretically prove that when $p(\text{Correct}|x) < p(\text{Wrong}|x)$, the probability of majority voting recovering the correct answer approaches zero as the number of samples increases.

**Core intuition: correct answers should remain stable across semantically equivalent reformulations**: Humans often verify the robustness of an answer by reconsidering it from a different angle when uncertain. Fragile reasoning paths are easily disrupted by surface-level rephrasing, whereas correct reasoning is invariant to such changes.

**Limitations of existing alternatives**: External verifiers or reward models (Lightman et al., 2024; Khalifa et al., 2025), though effective, violate the principle of a fully self-contained test-time setting.

## Method

### Overall Architecture

Self-Harmony alternates a single model $M_\theta$ between two roles:
- **Solver** $\pi_\theta$: generates answers to the given problem
- **Reframer** $\rho_\theta$: rephrases the problem into a semantically equivalent but differently worded variant

The pipeline proceeds as Solve → Reframe → Solve, and then uses the harmonic mean to select cross-perspective consistent pseudo-labels for reinforcement learning training.

### Key Designs

**1. Dual-Perspective Answer Generation**

- **Function**: Generates answer distributions for each problem from both the original and reframed perspectives.
- **Mechanism**: The original question $x$ produces answer set $\{y_i\}$ and the reframed question $x'$ produces $\{y'_i\}$; empirical frequencies $\hat{p}_0(a)$ and $\hat{p}_1(a)$ are computed for each candidate answer.
- **Design Motivation**: Based on the view-invariance assumption—the probability of a correct answer should be approximately consistent across different formulations, whereas incorrect answers tend to depend on specific phrasing.

Experiments validate this assumption: the consistency of correct answers between original and reframed questions is significantly higher than that of incorrect answers.

**2. Harmonic Mean Selection (HMS) for Pseudo-Labels**

- **Function**: Replaces majority voting with the harmonic mean for selecting training pseudo-labels.
- **Mechanism**: $y^\star = \arg\max_a \frac{2\hat{p}_0(a)\hat{p}_1(a)}{\hat{p}_0(a) + \hat{p}_1(a)}$
- **Design Motivation**: Derived from information theory—the harmonic mean is the second-order approximate optimal solution to the View-Invariant Infomax objective $J_\lambda(a) = I(Z_a; A) - \lambda I(Z_a; X)$ at $\lambda = 2$.

Theoretical guarantee (Theorem 3.2): Under the view-invariance assumption, non-degeneracy, balanced confidence, and uniform view priors, the harmonic mean selector maximizes the second-order approximation of the view-invariant Infomax objective, providing more robust pseudo-labels than majority voting.

Key advantage of the harmonic mean: it is highly sensitive to low values—an answer must appear frequently in both distributions to receive a high score, effectively filtering pseudo-answers produced by fragile reasoning in only one perspective.

**3. Fused Reframe-and-Solve Implementation**

- **Function**: Merges reframing and solving into a single generation step to reduce inference cost.
- **Mechanism**: A system prompt instructs the model to first rephrase the question and then immediately solve it, reducing three model calls to two.
- **Design Motivation**: The three-step pipeline (solve → reframe → solve) requires three model calls; fusion reduces this to two.

### Loss & Training

**Reward for the solve action**: $R_{\text{solve}}(y) = \mathbb{I}[y = y^\star]$

**Reward for the fused reframe-and-solve action**: A gated design is adopted, where answer correctness is a prerequisite before considering format and diversity penalties:
$$R_{\text{fused}}(y') = (1 - w_f R_{\text{format}}^{\text{penalty}}(y'))(1 - w_d R_{\text{div}}^{\text{penalty}}(y', y))\mathbb{I}[y' = y^\star]$$

The diversity penalty employs the Jensen-Shannon divergence between the answer distributions of the original and reframed questions, encouraging the reframing to provide genuinely distinct perspectives.

## Key Experimental Results

### Main Results

Performance of Qwen3-1.7B-Base across multiple benchmarks:

| Method | MATH500 | GSM8K | AIME2024 | AMC | GPQA | MMLU-Pro |
|--------|---------|-------|----------|-----|------|----------|
| Before RL | 42.70 | 65.58 | 3.33 | 26.50 | 20.30 | 16.61 |
| GT-Reward (upper bound) | 71.80 | 85.97 | 20.83 | 53.01 | 53.80 | 85.71 |
| Majority-Voting | 64.64 | 83.80 | 9.37 | 37.65 | 24.68 | 44.82 |
| Co-Reward | 64.67 | 86.59 | 6.67 | 39.75 | 23.66 | 47.14 |
| **Self-Harmony** | **69.60** | **87.47** | **10.00** | **40.51** | **27.92** | **53.66** |

Notable gains on Llama-3.1-8B: GSM8K improves from 60.5% to 91.6%.

Notable gains on Qwen3-4B: MATH500 improves from 60.2% to 78.5%.

### Ablation Study

| Configuration | Result |
|---------------|--------|
| Harmonic mean vs. majority voting | Harmonic mean is superior in nearly all settings |
| Dual-perspective majority voting vs. harmonic mean | Harmonic mean is more stable; dual-perspective majority voting still exhibits failure modes |
| Gated reward vs. additive reward | Gated design avoids rewarding reframings that produce incorrect answers with good formatting |
| Effect of diversity penalty | Encourages reframings that provide genuinely new perspectives rather than simple paraphrases |

### Key Findings

1. **Ranks first in 28 out of 30 experimental settings**: Covering 5 open-source models × 6 reasoning benchmarks.
2. **Zero training failures**: No training collapse observed across all experiments, demonstrating unprecedented stability.
3. **Only 16+16 rollouts required**: 16 original and 16 reframed rollouts are sufficient to achieve strong improvements.
4. **Gap with GT-Reward substantially narrowed**: Self-Harmony performance approaches the upper bound that uses ground-truth labels.
5. **Baselines such as Intuitor and Rent exhibit training instability**: They report peak scores (marked with *), whereas Self-Harmony reports final-step scores.

## Highlights & Insights

1. **Theoretical elegance of the harmonic mean**: The harmonic mean is derived naturally from the View-Invariant Infomax objective rather than introduced as a heuristic, providing a solid theoretical foundation.
2. **Minimalist design with a single model in dual roles**: No auxiliary models or external verifiers are required; role switching is achieved entirely through prompting, preserving simplicity and scalability.
3. **The core intuition that correct answers should be stable across perspectives is both simple and profound**: This observation is rooted in robust verification behavior in human cognition and proves equally applicable to LLMs.
4. **Zero failure rate is an important engineering advantage**: Training instability in TTRL methods is a major obstacle to real-world deployment, making Self-Harmony's stability highly practical.

## Limitations & Future Work

1. **Reframing quality depends on model capability**: If the model's paraphrasing ability is weak (e.g., very small models), the Reframer role may introduce semantic drift.
2. **Approximately doubled computational cost**: Two sets of rollouts (original + reframed) must be generated per question, making inference cost roughly twice that of standard TTRL.
3. **Limitations of the view-invariance assumption**: For tasks that are genuinely sensitive to wording (e.g., logical directionality in natural language inference), correct answers may also be affected by phrasing.
4. **Validated only on reasoning tasks**: The applicability of harmonic mean pseudo-labels to open-ended generation, summarization, and similar tasks remains unexplored.
5. **Sensitivity of hyperparameters $w_f$ and $w_d$**: The optimal weight selection in the fused reward may vary across different tasks and models.

## Related Work & Insights

- **TTRL (Zuo et al., 2025)**: Pioneered the test-time RL paradigm; Self-Harmony addresses its core weakness of majority voting.
- **Co-Reward**: Uses two models to mutually verify each other; Self-Harmony replaces this with two roles of a single model, achieving greater simplicity.
- **Invariant Risk Minimization (Arjovsky et al., 2019)**: Self-Harmony introduces the cross-environment invariance idea from IRM into TTRL pseudo-label selection.
- **FixMatch (Sohn et al., 2020)**: A pioneering work in semi-supervised learning that selects pseudo-labels via strong/weak augmentation consistency—the conceptual predecessor of Self-Harmony in LLM reasoning.
- Insight: The harmonic mean as a measure of cross-perspective consistency may be applicable to other scenarios requiring multi-view alignment, such as multimodal fusion and ensemble learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The combination of harmonic mean pseudo-labels and single-model self-play is highly creative, with elegant theoretical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage across 5 models × 6 datasets × multiple baselines in 30 settings; zero training failures are particularly impressive.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, theoretical proofs are complete, and framework diagrams are intuitive; however, the method section involves dense notation.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the core problem in TTRL (the majority voting trap); its stability and generality make it a strong candidate to become the default method for TTRL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)

</div>

<!-- RELATED:END -->
