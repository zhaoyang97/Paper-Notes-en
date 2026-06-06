---
title: >-
  [Paper Note] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization
description: >-
  [ACL 2026][Reinforcement Learning][Capability Boundary Collapse] RL-PLUS proposes a hybrid-policy optimization method that addresses external data distribution mismatch through Multiple Importance Sampling (MIS) and guid…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Capability Boundary Collapse"
  - "Hybrid-policy Optimization"
  - "Multiple Importance Sampling"
  - "Exploration Advantage Function"
  - "RLVR"
date: 2026-05-08
content_hash: 190e28c8bda8a35a
---

# RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization

**Conference**: ACL 2026  
**arXiv**: [2508.00222](https://arxiv.org/abs/2508.00222)  
**Code**: [GitHub](https://github.com/YihongDong/RL-PLUS)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: Capability Boundary Collapse, Hybrid-policy Optimization, Multiple Importance Sampling, Exploration Advantage Function, RLVR

## TL;DR

RL-PLUS proposes a hybrid-policy optimization method that addresses external data distribution mismatch through Multiple Importance Sampling (MIS) and guides the model to learn low-probability but correct reasoning paths via an Exploration-Based Advantage Function (EAF). It successfully breaks the capability boundary collapse caused by RLVR, achieving SOTA on six mathematical reasoning benchmarks (average 53.4) with consistent improvements across models up to 69.2%.

## Background & Motivation

**Background**: Reinforcement Learning with Verifiable Rewards (RLVR, e.g., GRPO/DAPO) has significantly enhanced the complex reasoning capabilities of LLMs by optimizing long-chain reasoning through rewards given for correct answers.

**Limitations of Prior Work**: RLVR is inherently an on-policy strategy—the model can only learn from trajectories it generates itself. This leads to the "capability boundary collapse" problem: while pass@1 improves, pass@128 actually falls below the base model. That is, RLVR makes the model better at choosing known correct paths (inward exploitation) but narrows the range of solvable problems (capability boundary). Simultaneously, policy entropy drops sharply (entropy collapse), making the model overly deterministic.

**Key Challenge**: Under the vast action space and sparse rewards of LLMs, RLVR cannot effectively guide the model to explore new reasoning paths (outward exploration). SFT can introduce external knowledge but is poor at internalizing reasoning principles. Simple combinations of the two (like GRPO w/ SFT Loss) actually result in performance degradation.

**Goal**: Design an RLVR method that effectively fuses external data and internal exploration to break through the capability ceiling of base models.

**Key Insight**: Drawing from the Confucian proverb, "Learning without thought is labor lost; thought without learning is perilous"—current RLVR is "thought without learning" (relying solely on self-reasoning). It needs to effectively "learn" external knowledge during the RL process. The key challenges are: (1) how to handle distribution mismatch in external data? (2) how to efficiently extract new knowledge from external data?

**Core Idea**: Perform reinforcement learning using a hybrid policy (internal on-policy trajectories + external data), stabilizing off-policy updates via MIS and amplifying learning signals for low-probability correct paths via EAF.

## Method

### Overall Architecture

The training objective of RL-PLUS fuses two parts: $\mathcal{J}_{\text{RL-PLUS}} = \underbrace{\mathbb{E}_{(o_i, A_i) \sim \mathcal{D}_o}[r_{i,t}(\theta) A_i]}_{\text{Internal Exploitation}} + \underbrace{\mathbb{E}_{(e_i, A_{i,t}^c) \sim \mathcal{D}_e}[r_{i,t}^m(\theta) A_{i,t}^c]}_{\text{External Exploration}}$. The first term is standard GRPO (optimizing existing reasoning abilities), and the second term is the core innovation (learning new knowledge from external data), utilizing MIS to correct distribution shifts and EAF to focus on low-probability correct paths.

### Key Designs

1.  **Multiple Importance Sampling (MIS)**:
    - **Function**: Addresses the distribution mismatch between external data and the current policy.
    - **Mechanism**: Instead of directly estimating the unknown external policy $\pi_\omega$, the sampling is treated as coming from a mixture policy $\pi_\omega + \pi_{\theta_{old}}$. The importance weight is defined as $r_{i,t}^m(\theta) = \frac{2\pi_\theta(e_{i,t}|q, e_{i,<t})}{\pi_\omega(e_{i,t}|q, e_{i,<t}) + \pi_{\theta_{old}}(e_{i,t}|q, e_{i,<t})}$. For the unknown $\pi_\omega$, a Bayesian optimal estimate $\hat{\pi}_\omega^* = \frac{1}{2}\pi_{\theta_{old}} + \frac{1}{2}\mathcal{U}$ is adopted (an equal-weighted mixture of the old policy and a uniform distribution), minimizing L2 risk under maximum uncertainty.
    - **Design Motivation**: Standard on-policy IS has systemic bias for external data (Lemma A.5), while direct off-policy IS suffers from excessive variance (Lemma A.7). MIS provides a compromise with low bias and low variance—even if $\pi_\omega$ differs greatly from $\pi_\theta$, the presence of $\pi_{\theta_{old}}$ prevents weight explosion.

2.  **Exploration-Based Advantage Function (EAF)**:
    - **Function**: Guides the model to focus on correct but low-probability reasoning paths, i.e., "new knowledge."
    - **Mechanism**: $A_{i,t}^c = \frac{R_i - \text{mean}(R)}{\text{std}(R)} \cdot C_{i,t}$, where the weight $C_{i,t} = (1 - \text{detach}(\pi_\theta(e_{i,t}|q, e_{i,<t})))^\gamma$. Inspired by Focal Loss—when the model assigns a low probability to a correct token (poor exploration), $C_{i,t}$ increases, amplifying the advantage signal of that path and forcing the model to attend to neglected new reasoning styles.
    - **Design Motivation**: Models naturally gravitate toward high-probability tokens (existing knowledge); new knowledge is often hidden in low-probability paths. Simply introducing external data stably is insufficient; the model must be explicitly guided to "see" and learn these new paths.

3.  **Removal of Clipping Mechanism**:
    - **Function**: Allows for larger gradient steps when optimizing high-value external data.
    - **Mechanism**: Standard GRPO uses $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ to limit update magnitude. The clipping is removed for the external data term because low-probability events (new knowledge) are precisely what require large-step optimization—clipping would suppress learning from these high-information paths.
    - **Design Motivation**: External exploration requires "bolder" policy updates, which conflicts with the objective of clipping.

### Loss & Training

Trained based on Qwen2.5-Math-7B using NuminaMath-1.5 as training data (containing external data). The KL regularization term is omitted (as policies need to deviate significantly from the initial policy in long CoT reasoning scenarios).

## Key Experimental Results

### Main Results

**Six Mathematical Reasoning Benchmarks (Qwen2.5-Math-7B)**

| Method | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base Model | 11.5 | 4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 |
| GRPO | 25.1 | 15.3 | 62.0 | 84.4 | 39.3 | 46.8 | 45.5 |
| LUFFY | 29.4 | 23.1 | 65.6 | 87.6 | 37.5 | 57.2 | 50.1 |
| SFT+GRPO | 25.8 | 23.1 | 62.7 | 87.2 | 39.7 | 50.4 | 48.2 |
| **RL-PLUS** | **33.4** | **25.9** | **68.1** | **90.2** | **43.8** | **58.8** | **53.4** |

**OOD Tasks (Coding + Science QA)**

| Method | HumanEval | LeetCode | LiveCodeBench | ARC-c | GPQA | MMLU-Pro | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GRPO | 63.4 | 21.1 | 15.3 | 81.7 | 40.4 | 47.5 | 44.9 |
| SFT+GRPO | 59.8 | 8.3 | 9.7 | 72.4 | 24.2 | 37.7 | 35.4 |
| **RL-PLUS** | **68.3** | **27.8** | **19.2** | **82.3** | **40.4** | **54.7** | **48.8** |

### Ablation Study

| Method | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RL-PLUS (Full) | **33.4** | **25.9** | **68.1** | **90.2** | **43.8** | **58.8** | **53.4** |
| - EAF | 28.3 | 24.1 | 67.8 | 88.8 | 40.4 | 56.0 | 50.9 |
| - MIS | 25.1 | 15.3 | 62.0 | 84.4 | 39.3 | 46.8 | 45.5 |

### Key Findings

- MIS is the more critical component (dropping 7.9 points when removed vs. 2.5 points for EAF), indicating that the stable introduction of external data is the foundation.
- RL-PLUS achieved an absolute improvement of 11.9 points on LLaMA-3.1-8B (where GRPO was almost ineffective), demonstrating strong generalization.
- Pass@k curves show RL-PLUS consistently outperforming the base model across all k values, proving it truly breaks the capability boundary.
- GRPO w/ SFT Loss performed worse than GRPO alone (40.1 vs. 45.5), showing that simple fusion of external knowledge is difficult.
- Training dynamics show that RL-PLUS's policy entropy does not drop to zero, maintaining exploration capability.

## Highlights & Insights

- The formalization and experimental validation of the "capability boundary collapse" problem are very persuasive—the pass@k curve is intuitive and powerful evidence.
- Rigorous theoretical analysis: The variance boundedness proof of MIS and the Bayesian optimal policy estimation are supported by complete mathematical derivations.
- Solved a practical challenge: How to effectively use external data in RL training without causing distribution mismatch or training collapse.
- Excellent OOD performance indicates that RL-PLUS learns general reasoning abilities rather than task-specific tricks.

## Limitations & Future Work

- The quality and coverage of the external data source (NuminaMath-1.5) affect performance; selection strategies for external data were not explored in depth.
- The equal-weight assumption in Bayesian policy estimation (50% for $\pi_{\theta_{old}}$ and $\mathcal{U}$) may not be optimal.
- Validated only on mathematical reasoning tasks; effectiveness on other reasoning tasks like code generation requires further verification.
- Insufficient sensitivity analysis for the $\gamma$ hyperparameter.

## Related Work & Insights

- **vs LUFFY**: LUFFY selectively imitates high-quality external trajectories but uses hybrid policies in a coarser manner; RL-PLUS provides a theoretically superior distribution correction via MIS.
- **vs ReLIFT**: ReLIFT alternates between RL and online fine-tuning, while RL-PLUS operates simultaneously within a unified framework.
- **vs GRPO w/ SFT Loss**: Directly adding SFT loss is harmful, indicating that the method of fusing external data is crucial.
- **Insight**: The application of Focal Loss concepts to RL advantage functions is an interesting cross-domain transfer.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Problem definition, MIS scheme, and EAF design are original and theoretically deep.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, six OOD tasks, four base models, pass@k analysis, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Solid theoretical derivations, comprehensive experiments, and clear paper structure.
- Value: ⭐⭐⭐⭐⭐ Addresses a core limitation of RLVR with a versatile method, providing significant reference for LLM reasoning training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)
- [\[ACL 2026\] EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning for LLMs](evocot_overcoming_the_exploration_bottleneck_in_reinforcement_learning.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](../../ICML2026/reinforcement_learning/metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)

</div>

<!-- RELATED:END -->
