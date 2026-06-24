---
title: >-
  [Paper Note] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection
description: >-
  [ICLR 2026][Learning Theory][RL Training Dynamics] This paper abstracts LLM reasoning into a two-stage process $q\to r\to a$ ("selecting a reasoning pattern $r$, then deriving an answer $a$ based on it"). It uses a tabular policy and gradient flow to characterize the training dynamics of RLVR and RLIF. The study proves that RLVR stably converges to the reasoning pattern with the highest success rate (strong base models converge quickly, while weak ones undergo an "entanglemen…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "LLM Reasoning"
  - "Reinforcement Learning"
  - "RL Training Dynamics"
  - "RLVR"
  - "RLIF"
  - "Reasoning Pattern Selection"
  - "Convergence Analysis"
date: 2026-05-08
content_hash: 33b8fa81937f2fdc
---

# Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2OO399hRD6](https://openreview.net/forum?id=2OO399hRD6)  
**Code**: To be confirmed  
**Area**: Learning Theory / LLM Reasoning / Reinforcement Learning  
**Keywords**: RL Training Dynamics, RLVR, RLIF, Reasoning Pattern Selection, Convergence Analysis

## TL;DR
This paper abstracts LLM reasoning into a two-stage process $q\to r\to a$ ("selecting a reasoning pattern $r$, then deriving an answer $a$ based on it"). It uses a tabular policy and gradient flow to characterize the training dynamics of RLVR and RLIF. The study proves that RLVR stably converges to the reasoning pattern with the highest success rate (strong base models converge quickly, while weak ones undergo an "entanglement phase"), whereas RLIF improves initially but has a 50% probability of converging to the worst pattern during long-term training, theoretically explaining the observed differences in their empirical curves.

## Background & Motivation
**Background**: Reinforcement Learning (especially Outcome-Reinforced RLVR and Self-Feedback RLIF) has become a key means to enhance reasoning models such as DeepSeek-R1 and Qwen3. However, little is known about "what exactly happens inside the model during RL training." Existing works have made observations from perspectives like pass@k, policy entropy, and sparse key tokens, but lack a unified explanation linking empirical phenomena to training dynamics.

**Limitations of Prior Work**: Empirical observations are isolated—some suggest RL introduces no new reasoning capabilities, others claim RL reduces policy entropy or only modifies a small set of key tokens. These high-level descriptions fail to answer **why RLVR converges stably while RLIF improves and then collapses**, nor can they predict when RL optimization will be difficult. Theoretically, proving optimization guarantees is challenging due to the complexity of real LLMs.

**Key Challenge**: RL training effectiveness highly depends on the quality of the base model, but how "base quality" determines final convergence through training dynamics remains a black box. RLVR and RLIF use the same RL objective but different rewards—one based on correctness and the other only on self-confidence. The mechanism by which this reward difference leads to opposite training curves is missing.

**Goal**: (1) Confirm through clean empirical analysis that "RL only optimizes sparse key tokens, thereby reshaping the reasoning pattern distribution"; (2) Construct an analyzable mathematical framework to prove the training dynamics (convergence points, speed, and degradation conditions) of RLVR and RLIF.

**Key Insight**: The authors observed a critical empirical fact—the **intrinsic success rate of a single reasoning pattern remains largely constant** throughout the training process; what changes is the probability distribution of path selection. This suggests that the "answer derivation" layer can be seen as fixed, allowing "pattern selection" to be treated as the actual object of RL optimization, thereby reducing high-dimensional token-level optimization to an analysis of pattern selection probabilities.

**Core Idea**: By using a two-stage abstraction "Question $\to$ Reasoning Pattern $\to$ Answer" ($q\to r\to a$) and gradient flow on a tabular policy, the paper proves that RL essentially reorders the selection probabilities of various reasoning patterns, providing a unified explanation for RLVR's stable convergence and RLIF's rise and subsequent fall.

## Method

### Overall Architecture
This paper does not propose a new algorithm but instead establishes an "empirical $\to$ theoretical" explanatory framework for the process of training LLMs with RL, divided into two parts.

**Empirical Part**: Focusing on Qwen2.5-3B on the MATH dataset, RLVR and RLIF were executed with a three-layer analysis. ① Training curve layer: RLVR increases monotonically and converges; RLIF rises then falls, even dropping below the base model. ② Reasoning pattern layer: GPT-4o was used to cluster base model answers by keywords and logical structure into "reasoning patterns," tracking their proportions and success rates—it was found that RLVR consistently shifts probability toward high-success patterns, while RLIF oscillates among patterns, even though **the intrinsic success rate of any pattern remains stable**. ③ Token layer: By comparing token probability rankings between base and RL models, it was found that less than 10% of token positions change rankings, proving that RL only acts on sparse critical decision points.

**Theoretical Part**: The empirical facts are formalized. Given a question $q$, the model first selects a reasoning pattern $r\in R$ with $\pi_\theta(r|q)$, then derives an answer $a\in A$ with $\pi_\theta(a|q,r)$; each pattern has a fixed success rate $p^*(r)=\pi_\theta(a^*|q,r)$. Using a tabular policy (each output token corresponds to a trainable logit column, $\pi_\theta(y_l|y_{l-1})=\mathrm{softmax}(\theta_{:,y_{l-1}})_{y_l}$), gradient flow $\frac{d}{dt}\theta(t)=\nabla\phi_{RL}(\theta(t))$ is analyzed under the small learning rate limit. The RL objective follows the standard form:

$$\phi_{RL}(\theta)=\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}[r_\phi(x,y)]-\beta D_{KL}[\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)].$$

Where RLVR reward is $r_\phi=\mathbb{1}[y=\text{ground truth}]$, and RLIF reward is the negative average KL between the next-token distribution and a uniform distribution. The framework provides a general optimal policy and then derives dynamics for RLVR and RLIF.

### Key Designs

**1. Reshaping Pattern Distribution via Sparse Key Tokens: Identifying Exactly What RL Changes**

This part addresses the pain point of isolated empirical observations. The authors conducted controlled experiments at both the pattern and token levels, verifying three facts: RLVR shifts selection probability to high-success patterns; RLIF's pattern distribution is unstable; and **the intrinsic success rate $p^*(r)$ remains nearly constant**. At the token level, the Ranking Change Ratio—measuring the proportion of token positions where probability rankings change post-RL—was mostly below 10% on GSM8K/MATH/AIME (e.g., MATH-RLVR went from 5.3% at step 20 to 6.6% at step 100). These findings support the core theoretical simplification: since pattern success rates are stable, one can fix the "pattern $\to$ answer" layer and analyze only the evolution of "question $\to$ pattern" selection probabilities.

**2. Two-Stage Reasoning Framework & Optimal Policy: Reducing High-Dimensional Optimization to Probability Reordering**

To address the complexity of real LLMs, the authors formalized reasoning as $q\to r\to a$ and introduced Assumption 5.1 (constant pattern success rate $p^*(r)$). Under this framework, the optimal policy for the RL objective was derived (Proposition 5.2):

$$\pi_{opt}(r|q)=\frac{1}{Z}\exp\!\Big(\tfrac{1}{\beta}R(r)\Big)\pi_{ref}(r|q),\qquad Z=\sum_{r\in R}\exp\!\Big(\tfrac{1}{\beta}R(r)\Big)\pi_{ref}(r|q),$$

where $R(r)$ is the reward for the path: $R_{RLVR}(r)=p^*(r)$ (the success rate) and $R_{RLIF}(r)=-\frac{1}{|A|}\sum_{a\in A}\log\pi_\theta(a|q,r)$ (a "confidence" score that **does not distinguish between right and wrong**). This formula is central: it shows that RLVR pushes probability towards high-success patterns for stable gains, whereas RLIF lacks a "correctness" signal, providing no guarantee of accuracy improvement. As $\beta \to 0$, the policy becomes deterministic, selecting $\arg\max_r R(r)$.

**3. Two Convergence Regimes and the "Entanglement Phase" of RLVR: Base Model Strength Dictates Speed**

This design uses gradient flow to characterize the path to the optimal pattern $r^*=\arg\max_r p^*(r)$. **Regime 1 (Theorem 5.3)**: If the base model's accuracy $\mathrm{ACC}_{\theta_{ref}}=\sum_r\pi_{ref}(r|q)p^*(r)$ already exceeds the success rate of all sub-optimal patterns ($>p^*(r),\,\forall r\neq r^*$), $\pi_\theta(r^*|q)$ increases monotonically to 1 with a polynomial convergence time $T_1=O(1/\epsilon)$. **Regime 2 (Theorem 5.4)**: If a sub-optimal pattern $r'$ has a success rate higher than the initial base accuracy ($p^*(r')>\mathrm{ACC}_{\theta_{ref}}>p^*(r)$), the model enters an "entanglement phase" where $r'$ slows down the rise of $r^*$, requiring a transition time $T_0$ to enter Regime 1. Crucially, $T_0$ grows **super-exponentially** with:

$$\gamma_{\pi_{ref}}:=\sum_{r\in R/\{r'\}}\frac{\pi_{ref}(r|q)}{\pi_{ref}(r^*|q)}.$$

If a weak base model gives the optimal pattern $r^*$ a very small initial probability, $T_0$ becomes so large that training effectively stalls.

**4. RLIF's "Rise and Collapse": 100% Initial Gain, 50% Convergence to the Worst Pattern**

The counter-intuitive "rise then fall" of RLIF is clarified in a clean setting (Theorem 5.6): $\beta=0$, tabular policy, uniform pattern selection by the base model, $|A|=2$, and success rates following $U[0,1]$. Under the assumption that the base model assigns higher probability to correct answers (supported by the effectiveness of majority voting), the results show: (1) At $t=0$, the derivative of accuracy is positive with probability 1—explaining early gains; (2) With 0.5 probability, $\arg\max_r R_{RLIF}(r)=\arg\min_r p^*(r)$, meaning long-term training converges to the **worst** pattern half the time. RLIF initially gains by increasing overall confidence, but deterministic convergence eventually locks into the pattern with the highest confidence, which has a 50% chance of being the least accurate.

## Key Experimental Results

### Main Results
Empirical evidence focused on Qwen2.5-3B on MATH for controlled comparison (RLVR vs RLIF), specifically the Ranking Change Ratio.

| Task | Method | Step20 | Step60 | Step100 |
|------|--------|--------|--------|---------|
| GSM8K | RLVR | 5.2% | 6.6% | 7.3% |
| GSM8K | RLIF | 6.8% | 8.9% | 10.1% |
| MATH | RLVR | 5.3% | 6.1% | 6.6% |
| MATH | RLIF | 6.1% | 7.7% | 8.8% |
| AIME24 | RLVR | 5.1% | 5.9% | 6.3% |

Ratios were mostly <10%, proving RL modifies only sparse critical points; RLIF showed higher ratios, corresponding to its unstable distribution.

### Numerical Simulation

| Phenomenon | Theoretical Prediction | Simulation Result |
|------------|------------------------|-------------------|
| RLVR Regime 1 | $T_1=O(1/\epsilon)$ fast convergence | $\pi(r^*)$ monotonic to 1 |
| RLVR Regime 2 | Super-exponential $T_0$ ($\gamma=6$) | Explicit entanglement phase before convergence |
| RLIF (Initial) | $\frac{d}{dt}\mathrm{ACC}|_{t=0}>0$ prob $\to$1 | Approaches 1 as $|R|$ increases |
| RLIF (Long-term) | 50% prob convergence to worst pattern | Improved vs Degraded $\approx$ 48.7% : 51.3% |

### Key Findings
- **Stability of single-pattern success rates** is the empirical cornerstone, justifying the analysis of pattern selection probabilities by freezing the derivation layer.
- **RLVR success is dictated by the base model**: Whether the base accuracy exceeds sub-optimal patterns determines the transition between "instant convergence" and "super-exponential entanglement."
- **Precise characterization of RLIF collapse**: Initial gains are nearly certain, but long-term failure occurs because the reward ignores correctness, locking onto the worst pattern with ~50% probability.

## Highlights & Insights
- **Dimension Reduction of RL Optimization**: By identifying that single-pattern success rates are constant, the authors could freeze $\pi(a|q,r)$ and analyze only $\pi(r|q)$, a clever reduction that makes the dynamics of real LLM RL provable.
- **Unified Policy Formula for Two Rewards**: The formula $\pi_{opt}\propto\exp(R(r)/\beta)\pi_{ref}$ exposes the structural defect of RLIF (ignoring correctness) in a way that is more convincing than empirical observation alone.
- **Super-exponential characterization of $T_0$**: The $\gamma_{\pi_{ref}}$ metric provides a computable criterion for when RLVR optimization will stall.

## Limitations & Future Work
- Assumption 5.1 (constant pattern success rate) is an idealization; in reality, patterns themselves may be fine-tuned during RL.
- Theoretical analysis relies on simplified settings (tabular policy, $\beta=0$, $|A|=2$). Expanding these to general LLMs remains for future work.
- Reasoning patterns are derived via GPT-4o clustering; their interpretability and stability influence all downstream conclusions.

## Related Work & Insights
- **vs Yue et al. 2025a / Huan et al. 2025 (Sparse Token Observation)**: While they observed RL modifies few tokens, this paper proves why this happens and links it to pattern selection dynamics.
- **vs Cui et al. / Zhang et al. (Entropy Perspective)**: This paper replaces the entropy-centric view with "reasoning pattern selection," explaining both RLVR stability and RLIF failure.
- **vs Agarwal et al. 2025 (RLIF Gain via Entropy)**: While they explained why RLIF works initially, this paper explains why it eventually fails by showing the 50% probability of locking into sub-optimal modes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to prove RLVR/RLIF dynamics using a $q\to r\to a$ framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Empirical + simulation verification, though lacks direct validation of theorem predictions on full-scale LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear connection between empirical and theoretical parts, though high-level simplifications increase the difficulty for readers.
- Value: ⭐⭐⭐⭐⭐ Provides computable criteria for RLVR convergence and RLIF collapse, providing direct guidance for post-training practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap](theoretical_modeling_of_large_language_model_self-improvement_training_dynamics_.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution](theoretical_analysis_of_contrastive_learning_under_imbalanced_data_from_training.md)
- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL: A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution](theoretical_analysis_of_contrastive_learning_under_imbalanced_data_from_training.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL：A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)
- [\[ICLR 2026\] High-dimensional Analysis of Synthetic Data Selection](high-dimensional_analysis_of_synthetic_data_selection.md)

</div>

<!-- RELATED:END -->
