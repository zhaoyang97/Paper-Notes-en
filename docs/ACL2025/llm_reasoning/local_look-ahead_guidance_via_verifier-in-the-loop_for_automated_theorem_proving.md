---
title: >-
  [Paper Note] Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving
description: >-
  [ACL 2025][Reasoning][Automated Theorem Proving] This paper proposes LeanListener, which introduces a verifier-in-the-loop design in automated theorem proving (ATP). By using the Lean verifier to provide step-level intermediate feedback (changes in the number of sub-goals) rather than merely trajectory-level rewards, the tactic validity rate and proving rate of ReProver are both improved via online GRPO training, achieving a 20% faster proving speed.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Automated Theorem Proving"
  - "Verifier Feedback"
  - "GRPO"
  - "DPO"
  - "Lean"
date: 2026-05-08
content_hash: 1503fe80eb181f59
---

# Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving

**Conference**: ACL 2025  
**arXiv**: [2503.09730](https://arxiv.org/abs/2503.09730)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Automated Theorem Proving, Verifier Feedback, GRPO, DPO, Lean

## TL;DR
This paper proposes LeanListener, which introduces a verifier-in-the-loop design in automated theorem proving (ATP). By using the Lean verifier to provide step-level intermediate feedback (changes in the number of sub-goals) rather than merely trajectory-level rewards, the tactic validity rate and proving rate of ReProver are both improved via online GRPO training, achieving a 20% faster proving speed.

## Background & Motivation

**Background**: LLMs are commonly used in automated theorem proving (ATP) to generate tactics (proof steps), but the training signals are sparse—typically consisting only of binary feedback on whether the entire proof succeeds or fails, lacking step-level signals.

**Limitations of Prior Work**: (a) Sparse rewards make RL training inefficient; (b) generating synthetic training data is highly expensive (requiring extensive rollouts and verification); (c) models frequently generate invalid tactics, wasting time during proof search.

**Key Challenge**: Formal verifiers (such as Lean) can determine whether a tactic is valid at each step and report the remaining number of sub-goals, but existing methods do not fully utilize this step-level feedback signal.

**Goal**: How to convert the step-by-step feedback from the verifier (changes in the number of sub-goals) into effective RL training signals to improve tactic generation quality and theorem-proving efficiency.

**Key Insight**: A reduction in the number of sub-goals indicates progress. Therefore, $\text{softplus}(\mathcal{G}(s_t) - \mathcal{G}(s_{t+1}))$ can be used as a step-level reward, optimized via online GRPO training.

**Core Idea**: Utilizing the changes in the number of sub-goals from the Lean verifier as step-level reward signals to train the tactic generation model online via GRPO.

## Method

### Overall Architecture
Based on ReProver (ByT5 encoder-decoder) as the base model. During training, for each proof state $s_t$, the model samples multiple tactics. The Lean verifier checks the validity of each tactic step-by-step and returns the number of sub-goals for the new state. After calculating the step-level rewards, the policy model is updated via GRPO.

### Key Designs

1. **Sub-goal Number Reward Function**:

    - Function: Provides dense reward signals for each step of tactic generation.
    - Mechanism: $R(a_t; s_t) = \text{softplus}(\mathcal{G}(s_t) - \mathcal{G}(s_{t+1}))$, where $\mathcal{G}(s)$ is the number of sub-goals in state $s$. If the number of sub-goals decreases, the reward is positive; invalid tactics yield a reward of 0. Softplus is used to prevent negative rewards.
    - Design Motivation: Finer grain than binary validity feedback—a tactic that reduces 3 sub-goals is superior to one that only reduces 1.

2. **Online GRPO Training**:

    - Function: Cyclic training involving online sampling, verification, and policy updates.
    - Mechanism: The model weights are updated every 50 steps. Intra-group relative advantage $\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ is used for normalization, with KL regularization added to prevent the policy from shifting too far.
    - Design Motivation: Offline DPO performs poorly (due to distribution shift between data and model), whereas online training is significantly superior.

3. **Premise Dropout Data Augmentation**:

    - Function: Randomly discards some premises during training to increase data diversity.
    - Design Motivation: Prevent the model from overfitting to specific combinations of premises.

### Loss & Training
- Fine-tuning for 10,000 steps with a batch size of 16 and a learning rate of $2.25 \times 10^{-6}$.
- Tactic sampling is performed with a beam width of 8, and the model is updated every 50 steps.
- Approximately 40 A100 GPU-days (with 80% spent on Lean interaction, being CPU-bound).

## Key Experimental Results

### Main Results (LeanDojo Benchmark, Pass@1 %)

| Model | Random Split | Novel Premises |
|------|-------------|----------------|
| ReProver (baseline) | 52.76 | 40.86|
| **LeanListener (GRPO, sub-goals)** | **53.21** | **41.11** |
| LeanListener (DPO, binary, random) | 35.99 | - |

### Tactic Validity Rate

| Method | Prec@8 ↑ | 0-precision ↓ |
|------|----------|---------------|
| ReProver | 40.77 | 12.74% |
| Offline DPO | 37.75 | 29.10% |
| Online DPO | 44.75 | 11.88% |
| **Online GRPO** | **51.01** | **7.44%** |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Offline vs. Online | Online training is critical: Prec@8 is 37.75 for offline vs. 51.01 for online GRPO. |
| DPO (Binary) vs. GRPO (Sub-goals) | DPO with binary feedback actually decreases the proving rate to 35.99% (spurious length bias), while GRPO improves it to 53.21%. |
| DPO Pairing Strategy | Differences between random/zero-acc/hard are <0.65%, showing low sensitivity. |
| Proving Speed | LeanListener is 20% faster within the same time limit. |

### Key Findings
- **Counter-intuitive Failure of DPO**: Although DPO improves tactic validity rate, it leads to a decrease in the overall proving rate. This is because it learns a spurious correlation that "shorter tactics are more likely to be valid"—the model tends to generate excessively short tactics instead of semantically correct ones.
- **Online Training is Core**: Offline DPO is ineffective or even harmful, whereas online DPO/GRPO yields substantial improvements.
- **Sub-goal Rewards Outperform Binary Rewards**: GRPO + sub-goals (53.21%) >> DPO + binary (35.99%).
- **Limitation of One-step Look-Ahead**: Performance drops on theorems requiring multi-step proofs, implying a need for multi-step look-ahead.

## Highlights & Insights
- **Verifier as Reward Signals**: Formal verifiers naturally provide correctness-guaranteed step-level feedback, which can be adapted to code generation, mathematical reasoning, and other scenarios with external verifiers.
- **Profound Lesson from DPO Failure**: Even if preference pairs are quality-reasonable, DPO can still learn spurious correlations (e.g., length bias), serving as a warning for all works utilizing DPO.
- **Dense Rewards > Sparse Rewards**: The concept of using the number of sub-goals as a dense reward can be generalized to any RL scenario involving intermediate state changes.

## Limitations & Future Work
- **Limited to ReProver Architecture**: The generalizability to other ATP models has not been verified.
- **Limitation of One-step Look-Ahead**: Long proofs require multi-step look-ahead, whereas the current method only looks one step ahead.
- **High Computational Cost**: 40 A100 GPU-days, 80% of which is Lean interaction.
- **Weak Cross-domain Generalization**: Slightly lower than the baseline on MiniF2F (36.9% vs 37.7%) and ProofNet (14.4% vs 15.3%).
- Possible Improvements: (a) Introduce n-step look-ahead or Monte Carlo Tree Search; (b) Use more efficient Lean interaction methods to reduce the CPU bottleneck.

## Related Work & Insights
- **vs. ReProver**: The baseline model. LeanListener incorporates step-level verifier feedback on top of it.
- **vs. PRM in LLM Mathematical Reasoning**: PRMs use human/MCTS annotated process rewards, while LeanListener directly obtains them from the verifier—resulting in lower cost and guaranteed correctness.
- **vs. DeepSeek-Prover**: DeepSeek uses synthetic data + RL to train its prover, whereas LeanListener focuses on verifier feedback design rather than data scale.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing step-level verifier feedback for ATP training is novel, and the analysis of DPO failure is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison of multiple methods + detailed ablation studies + speed analysis.
- Writing Quality: ⭐⭐⭐⭐ Explanations of methods and experiments are clear.
- Value: ⭐⭐⭐⭐ Useful reference for both ATP and dense-reward RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Process-Verified Reinforcement Learning for Theorem Proving via Lean](../../ICLR2026/llm_reasoning/process-verified_reinforcement_learning_for_theorem_proving_via_lean.md)
- [\[ICLR 2026\] EvolProver: Advancing Automated Theorem Proving by Evolving Formalized Problems via Symmetry and Difficulty](../../ICLR2026/llm_reasoning/evolprover_advancing_automated_theorem_proving_by_evolving_formalized_problems_v.md)
- [\[ICLR 2026\] Mathesis: Towards Formal Theorem Proving from Natural Languages](../../ICLR2026/llm_reasoning/mathesis_towards_formal_theorem_proving_from_natural_languages.md)
- [\[ICML 2026\] From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates](../../ICML2026/llm_reasoning/from_llm-generated_conjectures_to_lean_formalizations_automated_polynomial_inequ.md)
- [\[ICLR 2026\] Neural Theorem Proving for Verification Conditions: A Real-World Benchmark](../../ICLR2026/llm_reasoning/neural_theorem_proving_for_verification_conditions_a_real-world_benchmark.md)

</div>

<!-- RELATED:END -->
