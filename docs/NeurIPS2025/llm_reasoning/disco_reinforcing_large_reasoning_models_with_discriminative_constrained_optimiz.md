---
title: >-
  [Paper Note] DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization
description: >-
  [NeurIPS 2025][LLM Reasoning][GRPO] This paper analyzes the GRPO objective and reveals two inherent issues: difficulty bias (underweighting questions that are too hard or too easy) and entropy instability. It proposes DisCO, a discriminative constrained optimization framework that addresses these issues via a clip-free scoring function, squared hinge constrained optimization, and distributionally robust optimization (DRO) for imbalanced rollouts. On 1.5B models, DisCO outperforms GRPO by 7% and DAPO by 6% on average.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - GRPO
  - discriminative optimization
  - AUC maximization
  - constrained optimization
  - data imbalance
date: 2026-05-08
content_hash: 2fc80a1b68d76418
---

# DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.12366](https://arxiv.org/abs/2505.12366)
**Code**: [https://github.com/Optimization-AI/DisCO](https://github.com/Optimization-AI/DisCO)
**Area**: LLM Reasoning
**Keywords**: GRPO, discriminative optimization, AUC maximization, constrained optimization, data imbalance

## TL;DR

This paper analyzes the GRPO objective and reveals two inherent issues: difficulty bias (underweighting questions that are too hard or too easy) and entropy instability. It proposes DisCO, a discriminative constrained optimization framework that addresses these issues via a clip-free scoring function, squared hinge constrained optimization, and distributionally robust optimization (DRO) for imbalanced rollouts. On 1.5B models, DisCO outperforms GRPO by 7% and DAPO by 6% on average.

## Background & Motivation

**State of the Field**: The success of DeepSeek-R1 has established GRPO as the core RL training algorithm for large reasoning models (LRMs). Numerous GRPO-based improvements have emerged, including DAPO (decoupled clipping), Dr. GRPO (removing variance normalization), and GPG (simplified REINFORCE).

**Limitations of Prior Work**:
- **Difficulty Bias**: The group relative advantage in GRPO assigns each question a weight $\omega(q) = \sqrt{p(q)(1-p(q))}$, which approaches zero when $p(q) \approx 0$ (too hard) or $p(q) \approx 1$ (too easy), causing the model to effectively ignore these questions. Dr. GRPO's correction only changes the weight to $p(q)(1-p(q))$, mitigating but not eliminating the problem.
- **Entropy Instability**: GRPO's clipping operation leads to entropy collapse (the policy degenerates into deterministic outputs too quickly), while DAPO's decoupled clipping causes entropy explosion (outputs become highly stochastic).
- **Data Imbalance**: For hard questions where $p(q) \ll 1$, negative samples vastly outnumber positive ones; naive AUC-based optimization overlooks the importance of top-ranked negative samples.

**Root Cause**: Existing GRPO variants apply patches to the original framework (modifying clipping parameters, removing normalization, etc.) rather than principled redesign.

**Paper Goals**: To design an optimization framework from scratch that does not inherit GRPO's limitations and simultaneously addresses difficulty bias, entropy instability, and data imbalance.

**Starting Point**: Decomposing the GRPO objective under a binary reward setting reveals that it is essentially a discriminative objective with difficulty-biased weights—increasing scores for correct answers and decreasing scores for incorrect ones—which is closely related to the classical discriminative learning framework of AUC maximization.

**Core Idea**: Replace GRPO's group relative objective with a clip-free discriminative learning objective, substitute KL regularization with constrained optimization for training stability, and apply DRO to handle the imbalance between positive and negative samples in rollouts.

## Method

### Overall Architecture

Given a set of math problems, the policy model (e.g., DeepSeek-R1-Distill) generates multiple responses per question, which are labeled with binary rewards. Rather than computing group relative advantages as in GRPO, DisCO separates positive and negative responses and constructs a discriminative objective—maximizing scores for correct responses while minimizing scores for incorrect ones. It employs clip-free scoring functions (log-likelihood or likelihood ratio), squared hinge penalized constrained optimization to control KL divergence, and DRO to handle positive–negative sample imbalance.

### Key Designs

1. **Discriminative Decomposition of the GRPO Objective (Theoretical Contribution)**

   - **Function**: Proves that the GRPO objective is equivalent to a weighted discriminative objective.
   - **Mechanism**: Proposition 1 establishes that $\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_q \sqrt{p(q)(1-p(q))} \cdot \mathbb{E}_{o \sim \pi^+, o' \sim \pi^-}[s^+(o,q) - s^-(o',q)]$, where $\sqrt{p(q)(1-p(q))}$ is the difficulty bias weight and $s^+, s^-$ are the clipped scoring functions for positive and negative samples, respectively.
   - **Design Motivation**: This decomposition simultaneously identifies the two root causes of GRPO's problems—the weight $\omega(q)$ induces difficulty bias, and the clipped scoring function causes entropy instability.

2. **Clip-Free Scoring Function**

   - **Function**: Replaces GRPO's clipped scoring function with log-likelihood or likelihood ratio.
   - **Mechanism**:
     - Log-likelihood: $s_\theta(o,q) = \frac{1}{|o|} \sum_t \log \pi_\theta(o_t|q, o_{<t})$
     - Likelihood ratio: $s_\theta(o,q) = \frac{1}{|o|} \sum_t \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{old}}(o_t|q, o_{<t})}$
   - **Design Motivation**: GRPO's clipping $\min(x, 1+\epsilon)$ limits update magnitude in the positive sample direction and is the root cause of entropy collapse. DAPO attempts to mitigate this with different $\epsilon$ values but introduces entropy explosion. Clip-free functions fundamentally avoid this dilemma.

3. **Squared Hinge Constrained Optimization (Replacing KL Regularization)**

   - **Function**: Replaces standard KL regularization $\beta \cdot \mathbb{D}_{\text{KL}}$ with a constraint penalty $[\mathbb{D}_{\text{KL}}(\pi_{\text{old}}||\pi_\theta) - \delta]_+^2$.
   - **Mechanism**: When the KL constraint is satisfied ($\mathbb{D}_{\text{KL}} \leq \delta$), the gradient of the penalty term is zero and does not interfere with learning; the penalty is applied only when the constraint is violated, with a magnitude proportional to the square of the violation.
   - **Design Motivation**: Standard KL regularization continuously pulls the model regardless of necessity, and its hyperparameter $\beta$ is difficult to adapt throughout training as model characteristics change dramatically. The squared hinge penalty is self-adaptive—it corrects only when needed, with a force proportional to the degree of deviation.

4. **DRO-Based Imbalanced Rollout Handling (Enhanced DisCO)**

   - **Function**: For each positive sample, rather than treating all negative samples equally, the method focuses on the "most dangerous" negative samples (incorrect answers with the highest scores).
   - **Mechanism**: Partial AUC maximization is formulated as a DRO objective: $\mathcal{J}_2(\theta) = -\mathbb{E}_q \mathbb{E}_{o \sim \pi^+} \tau \log(\mathbb{E}_{o' \sim \pi^-} \exp(\frac{s_\theta(o',q) - s_\theta(o,q)}{\tau}))$. The LogSumExp operator automatically focuses attention on the highest-scoring (most confusing) negative samples.
   - **Design Motivation**: When there is 1 positive sample versus 100 negative samples, AUC may reach 0.99 while the model still tends to generate a particular incorrect answer (with score 0.9 > the correct answer's score of 0.5). DRO reweights the negative sample distribution to focus on the hardest-to-distinguish cases.

### Loss & Training

The full DisCO optimization objective is:
$$\max_\theta \mathcal{J}_2(\theta) - \beta[\mathbb{D}_{\text{KL}}(\pi_{\text{old}}||\pi_\theta) - \delta]_+^2$$

- 128 questions per step, 8 responses each, mini-batch size 32.
- Hyperparameters: $\delta = 10^{-4}$, $\beta = 10^3$ (effective penalty weight ~0.1).
- Both scoring functions are effective: L-ratio ($\tau=1$) and log-L ($\tau=10$).
- Dynamic Sampling (a DAPO technique) is not used, ensuring fair comparison.

## Key Experimental Results

### Main Results (1.5B model, max length 8K)

| Method | AIME'24 | AIME'25 | MATH500 | AMC'23 | Minerva | O-Bench | Avg. |
|--------|---------|---------|---------|--------|---------|---------|------|
| GRPO | 0.277 | 0.242 | 0.838 | 0.647 | 0.276 | 0.462 | 0.457 |
| DAPO | 0.310 | 0.252 | 0.848 | 0.675 | 0.296 | 0.456 | 0.473 |
| Dr. GRPO | 0.252 | 0.238 | 0.831 | 0.631 | 0.268 | 0.440 | 0.443 |
| TRPA | 0.354 | 0.235 | 0.835 | 0.653 | 0.283 | 0.458 | 0.470 |
| **DisCO (log-L)** | **0.404** | **0.317** | **0.876** | **0.758** | **0.333** | **0.509** | **0.533** |

### Ablation Study

| Configuration | Avg. Acc | Notes |
|---------------|----------|-------|
| DisCO (full) | 0.627 (7B) | Full model |
| DisCO-b (base, no DRO) | ~0.60 (7B) | Removing DRO; largest drop on AIME |
| Clipped L-ratio ($\epsilon=0.2$) | Entropy collapse | Same clipping as GRPO |
| Clipped L-ratio ($\epsilon=0.28$) | Entropy too high | Same clipping as DAPO causes entropy explosion |
| KL regularization (replacing constrained opt.) | Worse and unstable | Instability emerges in mid-to-late training on 7B |
| Clip-free L-ratio / log-L | Stable entropy | Both clip-free scoring functions perform well |

### Key Findings

- **DisCO consistently leads across all model scales and benchmarks**: 1.5B (+7% vs. GRPO), 7B (+3.5%), 8B (+5.5%); no baseline outperforms DisCO on any single benchmark.
- **Highly stable training dynamics**: DisCO's entropy remains stable at ~0.22 throughout training, while GRPO variants either suffer entropy collapse or explosion.
- **DRO is most beneficial for hard questions**: The gap between DisCO and DisCO-b is primarily observed on hard benchmarks such as AIME, where positive samples are scarce and DRO's imbalance handling is most effective.
- **DisCO at 8K length surpasses DeepScaleR at 24K length**: This demonstrates that algorithmic improvements can compensate for shorter reasoning chains, enabling more efficient reasoning.
- **Constrained optimization substantially outperforms KL regularization**: Fixed regularization weights are difficult to adapt during training; the "penalize only upon violation" strategy of constrained optimization is considerably more flexible.

## Highlights & Insights

- **The theoretical decomposition of GRPO as a discriminative objective is highly insightful**: It not only explains the mathematical origin of difficulty bias but also reveals the connection to AUC maximization, providing a new perspective for method design. This analytical technique can be applied to other RL-for-LLM objectives.
- **Squared hinge constrained optimization is a practically valuable engineering insight**: Compared to TRPO's complex second-order optimization and PPO's clipping, this approach is simple to implement and more effective. The adaptive "penalize only upon violation" property eliminates the need for painful hyperparameter tuning.
- **DRO for imbalanced rollouts is a natural transfer from discriminative learning**: Borrowing partial AUC / DRO from discriminative learning to address sparse rewards in RL is an excellent example of cross-domain knowledge transfer.
- **Outperforming 24K/32K-length methods under an 8K length constraint**: This indicates that better optimization algorithms can improve reasoning quality without longer reasoning chains, which is particularly valuable in compute-limited settings.

## Limitations & Future Work

- Validation is limited to mathematical reasoning; effectiveness on code generation, logical reasoning, and other tasks remains unknown.
- The binary reward assumption (correct/incorrect) is overly simplistic; extension to tasks requiring fine-grained rewards (e.g., code quality, reasoning process quality) is needed.
- Although DisCO exhibits stable training entropy, it is unclear whether stability is maintained in later training stages (beyond 1,400 steps).
- No comparison with PPO is provided; only GRPO-family baselines are considered.
- The DRO temperature $\tau$ requires different values for different scoring functions (1 for L-ratio, 10 for log-L); automatic tuning of $\tau$ could further improve usability.

## Related Work & Insights

- **vs. GRPO**: DisCO fundamentally resolves GRPO's three core issues (difficulty bias, entropy instability, data imbalance) rather than applying surface-level patches.
- **vs. DAPO**: DAPO improves entropy collapse via decoupled clipping but introduces entropy explosion; DisCO avoids this dilemma entirely through clip-free scoring functions.
- **vs. TRPA**: TRPA can be viewed as a special case of DisCO (using logistic loss, log likelihood ratio scoring, and KL regularization) without addressing imbalanced rollouts.
- **Relationship to SBT**: SBT addresses overthinking from the perspective of reasoning efficiency, while DisCO fundamentally improves reasoning quality from the perspective of training optimization; the two are orthogonal and can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The theoretical connection between GRPO and discriminative learning is highly original; applying DRO to handle imbalanced rollouts is a novel technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 3 model scales × 6 benchmarks × 5 baselines, with comprehensive ablation studies and training dynamics analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The theoretical sections are clear, though the notation is heavy and the first few sections require multiple readings.
- **Value**: ⭐⭐⭐⭐⭐ — A fundamental improvement over GRPO that directly advances LRM training methodology.

## Core Problem

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_reasoning/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties](topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)

<!-- RELATED:END -->
