---
title: >-
  [Paper Note] Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs
description: >-
  [ACL 2026][Reinforcement Learning][Free Energy Principle] FREIA introduces the Free Energy Principle (FEP) into label-free RL fine-tuning. It utilizes a "consensus + exploration" Free Energy Reward (FER) and Adaptive Adv…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Free Energy Principle"
  - "Unsupervised RL"
  - "Advantage Shaping"
  - "GRPO"
  - "Self-improvement"
date: 2026-05-08
content_hash: 1f3a41b2e8a5a78f
---

# Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs

**Conference**: ACL 2026  
**arXiv**: [2605.04065](https://arxiv.org/abs/2605.04065)  
**Code**: Not yet disclosed  
**Area**: Reinforcement Learning / Unsupervised RL / LLM Reasoning / GRPO  
**Keywords**: Free Energy Principle, Unsupervised RL, Advantage Shaping, GRPO, Self-improvement

## TL;DR
FREIA introduces the Free Energy Principle (FEP) into label-free RL fine-tuning. It utilizes a "consensus + exploration" Free Energy Reward (FER) and Adaptive Advantage Shaping (AAS) based on reward distribution skewness to simultaneously address two issues: premature convergence caused by traditional majority voting or confidence-based rewards, and the mismatch of advantage estimation during the training phase. It achieves performance parity with or even surpasses supervised GRPO across 3 reasoning tasks and 9 datasets.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become a core technology for enhancing LLM reasoning capabilities (e.g., DeepSeek-R1, o1). however, it relies heavily on human-annotated ground-truth. Unsupervised self-improvement has emerged as a research hotspot, primarily divided into two categories: (1) **Intrinsic-trajectory methods** (Entropy, Intuitor, Confidence-is-all-you-need) which use semantic entropy or confidence as rewards; (2) **Group consensus methods** (TTRL, Self-Consistency PO, Co-Reward) which use majority voting as rewards.

**Limitations of Prior Work**: The authors illustrate the flaws of both paradigms with a simple example: if the correct answer is "13" but majority voting assigns 0 points to it while giving full points to the incorrect majority answer "4." Confidence-based methods also tend to assign high scores to "rare but confident" incorrect responses. Furthermore, both approaches apply **static criteria** to **dynamic capabilities**:

- **Early Training (Weak Consensus)**: High-reward answers are scarce. Standard advantage normalization may assign a massive positive advantage to a few outliers, causing the model to overfit to noise prematurely.
- **Late Training (Strong Consensus)**: Majority answers dominate the population. Standard advantage treats occasional minority paths as having large negative advantages, causing the policy to degenerate from "reinforcing strengths" to "simply avoiding mistakes."

**Key Challenge**: Both reward design and advantage estimation are **static**, failing to adjust as model capabilities evolve. Early stages require encouraging exploration, while late stages require consolidating consensus, yet existing methods use the same rules for both.

**Goal**: (1) Design a reward mechanism that adaptively switches between "Consensus $\leftrightarrow$ Exploration"; (2) Design advantage shaping that is aware of the current training phase; (3) Surpass all unsupervised baselines across math, SQL, and multimodal geometry tasks without introducing additional training costs.

**Key Insight**: The authors draw from the **Free Energy Principle (FEP, Friston 2010)** in neuroscience, which posits that the brain minimizes free energy to balance "exploiting existing beliefs" and "active exploration." Conceptualizing unsupervised LLM self-improvement as free energy minimization naturally leads to a dual objective: "consensus alignment + novel path exploration."

**Core Idea**: Use group confidence $C_G \in [0, 1]$ as a gate to adaptively interpolate between consensus and exploration rewards. Use the **skewness** $\mathcal{S}$ of the reward distribution as a training phase indicator to dynamically decay the weights of positive or negative advantages.

## Method

### Overall Architecture

The complete FREIA pipeline (based on the GRPO framework) consists of:

1. **Sampling**: For each input $x$, rollout $G=8$ reasoning paths $Y=\{y_1, ..., y_G\}$, extract the final answers $A$, and count the unique answer set $U=\{u_1, ..., u_M\}$ and their frequencies $D=\{f_1, ..., f_M\}$.
2. **FER Reward Calculation**: Adjust the "consensus/exploration" weights via belief sharpening and group confidence to obtain the reward $R_i$ for each path.
3. **AAS Advantage Shaping**: Compute the group-normalized $\tilde{A}_i$, then use the reward distribution skewness to calculate decay weights $w_{pos}, w_{neg}$, resulting in the final $\hat{A}_i$.
4. **GRPO Policy Update**: Feed $\hat{A}_i$ into the standard GRPO clip-PPO objective $\mathcal{L}(\theta)$ with a KL constraint $\beta D_{KL}(\pi_\theta \| \pi_{ref})$, where $\beta=0.001$.

### Key Designs

1. **Free Energy-Driven Reward (FER) Adaptive Hybrid**:

    - **Function**: Simultaneously represents "following the majority" and "encouraging risk" goals within a single reward formula, with gating weights that adjust dynamically during training.
    - **Mechanism**: First, apply **nonlinear belief sharpening** to answer frequencies: $w_i = f_i^\alpha / \sum_k f_k^\alpha$ (with $\alpha=2$). A larger $\alpha$ strengthens the majority. Group confidence is then defined using normalized Shannon entropy: $C_G = 1 - H(W)/\log M$ (if $M=1$, $C_G=1$). The consensus reward is $r_{cons}(y_i) = \mathbb{1}[a_i = \text{Vote}(A)]$, and the exploration reward is $r_{explore}(y_i) = \tanh(-\log w_i)$ (rare answers score higher, $\tanh$ prevents signal dominance). Finally, $R_i = C_G \cdot r_{cons}(y_i) + (1 - C_G) \cdot r_{explore}(y_i)$.
    - **Design Motivation**: In early training, $C_G$ is low, automatically biasing towards exploration to avoid being locked into incorrect majorities. In late training, $C_G$ is high, shifting towards consensus to consolidate correct reasoning paths. This implements FEP for unsupervised LLMs: exploit when confident, explore when uncertain.

2. **Adaptive Advantage Shaping (AAS) Bi-directional Decay**:

    - **Function**: Uses the **skewness** $\mathcal{S}$ of the reward distribution within a batch to detect the training stage and decay positive or negative advantages accordingly.
    - **Mechanism**: Calculate standard group-normalized advantage $\tilde{A}_i = (R_i - \mu_R)/(\sigma_R + \epsilon)$; calculate sample skewness $\mathcal{S} = \frac{1}{G}\sum_i \tilde{A}_i^3$; then map skewness to decay weights via sigmoid: $w_{pos} = \sigma(-\mathcal{S})$, $w_{neg} = \sigma(\mathcal{S})$. Finally, $\hat{A}_i = w_{pos}\tilde{A}_i$ (if $\tilde{A}_i > 0$) or $w_{neg}\tilde{A}_i$ (if $\tilde{A}_i < 0$).
    - **Design Motivation**: **Positive Skew (Case 1)** indicates that low rewards dominate with rare high-reward answers; here, high values of $\tilde{A}_i$ are likely stochastic outliers. AAS sets $w_{pos} \to 0$ to suppress overfitting to noise. **Negative Skew (Case 2)** indicates high rewards dominate; here, low rewards are likely harmless variations. AAS sets $w_{neg} \to 0$ to prevent over-penalizing minority paths.

3. **Integration with GRPO**:

    - **Function**: Treats the entire FER+AAS module as a plug-in for GRPO without altering the clip + KL regularization structure of PPO.
    - **Mechanism**: Replace the original group-normalized advantage in GRPO with $\hat{A}_i$. The loss remains: $\mathcal{L}(\theta) = \mathbb{E}[\frac{1}{G}\sum_i \frac{1}{|o_i|} \sum_t \min(r_{i,t}(\theta)\hat{A}_i, \text{clip}(r_{i,t}, 1\pm \epsilon)\hat{A}_i) - \beta D_{KL}(\pi_\theta \| \pi_{ref})]$.
    - **Design Motivation**: By only modifying the reward and advantage components, FREIA maintains the default token-level loss and KL terms, allowing for drop-in replacement with nearly zero migration cost and wall-clock time parity with baselines.

### Loss & Training

- **Training**: MATH dataset, AdamW ($lr=1e-6$), 400 steps, batch=512, rollout $G=8$, sampling temperature=1.0; KL coefficient $\beta=0.001$; FER hyperparameter $\alpha=2$.
- **Evaluation**: Pass@1, sampling temperature=0.6, mean over 3 random seeds.
- **Hardware**: 4× A100 40GB GPUs.

## Key Experimental Results

### Main Results

Pass@1 on 6 mathematical reasoning benchmarks (DeepSeek-R1-Distill-Qwen-1.5B):

| Dataset | Base | GRPO (Supervised) | TTRL | Entropy | Intuitor | **Ours (FREIA)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MATH500 | 77.6 | 82.4 | 82.6 | 81.8 | 81.4 | 82.2 |
| AIME24 | 16.7 | 20.0 | 20.0 | 16.7 | 16.7 | **20.0** |
| AIME25 | 16.7 | 20.0 | 20.0 | 16.7 | 16.7 | **20.0** |
| AMC23 | 62.5 | 70.0 | 70.0 | 65.0 | 65.0 | **72.5** |
| Minerva | 27.6 | 30.5 | 30.9 | 29.8 | 29.4 | **31.3** |
| Olympiad | 42.4 | 48.6 | 49.0 | 47.5 | 46.6 | **49.4** |
| **Avg.** | 40.6 | 45.3 | 45.4 | 42.7 | 42.4 | **45.9** |

On Qwen2.5-Math-1.5B-Instruct, the average Pass@1: Base=33.2 $\to$ Entropy=35.7 $\to$ Intuitor=34.8 $\to$ TTRL=38.1 $\to$ **Ours=38.5**, still exceeding supervised GRPO (38.3).

### Ablation Study

| Configuration | Avg Pass@1 (Rel. Change) | Description |
|:---|:---|:---|
| Full FREIA | 45.9 | FER + AAS complete version |
| w/o AAS | ↓ Small | Reverts to static advantage normalization |
| w/o Exploration | ↓↓ | Consensus reward only; premature convergence in early stages |
| w/o Consensus | ↓↓↓ **Max Drop** | Exploration reward only; loses self-improvement drive |

Regarding $\alpha$ sensitivity: optimal performance is found near $\alpha=2$. Small $\alpha$ leads to noisy signals, while large $\alpha$ over-reinforces consensus and converges prematurely to sub-optimal solutions.

### Key Findings

- **Unsupervised parity with supervised GRPO**: FREIA's average score of 45.9 on DeepSeek-1.5B is higher than the supervised GRPO's 45.3, attributed to the "continuous + dense" signals provided by FER compared to sparse binary RLVR signals.
- **Consensus is more critical than exploration**: Ablations show removing consensus causes the largest drop, followed by exploration. This suggests that the primary driver of self-improvement remains the majority signal, while exploration serves as a safety valve.
- **Training dynamics align with FEP**: Results show policy entropy decreasing monotonically while $C_G$ and consensus reward rise smoothly. Exploration reward remains fluctuating, indicating the model maintains exploration while converging.
- **Cross-task Transferability**: FREIA leads all unsupervised baselines in SQL generation (Spider/BIRD) and multimodal geometry (Geometry3K).

## Highlights & Insights

- **First implementation of FEP for unsupervised RL reward design**: FEP was previously used mainly in Active Inference; this work simplifies it into a computable "consensus + exploration" formula gated by $C_G$.
- **Reward distribution skewness as a stage indicator**: Unlike using step counts or entropy, skewness is self-contained within a batch and does not require global state. This can be transferred to any group-based RL method (GRPO/RLOO).
- **"Belief sharpening + soft normalization" combination**: Using power-law sharpening for consensus and $\tanh$ to prevent exploration reward explosion acts as a general template for preventing reward hacking in unsupervised RL.
- **Zero computational overhead**: Since both FER and AAS are $O(G)$ operations within a batch, the wall-clock time is equivalent to baselines, meaning no hidden cost for performance gains.

## Limitations & Future Work

- **Limitations**: (1) Experiments are limited to models up to 3B parameters; (2) $C_G$ only considers the final answer distribution, not intermediate reasoning steps; (3) AAS uses batch-level skewness, which may mask intra-batch heterogeneity.
- **Potential Issues**: (1) Improvement margins are statistical but engineering gains are relatively small (0.5-3.5 points over TTRL); (2) The "consensus = majority" assumption fails in open-ended generative tasks.
- **Future Directions**: (1) Move from batch-level to token-level adaptive shaping; (2) Extend $C_G$ to step-level confidence; (3) Verify the FEP framework on long-CoT tasks where continuous reward signals may offer even greater advantages.

## Related Work & Insights

- **vs TTRL**: TTRL relies purely on majority voting and can be locked into incorrect majorities; FREIA preserves learning signals for minority paths via the exploration term.
- **vs Entropy/Intuitor**: These use trajectory-intrinsic signals which reward "confident but wrong" answers; FREIA uses group consensus as an external anchor.
- **vs Co-Reward**: Co-Reward uses cross-view consensus for robustness but suppresses exploration; FREIA performs both on a single rollout.
- **vs Supervised GRPO**: GRPO uses binary signals; FER uses continuous rewards in $(0, 1)$, providing dense gradient signals for every path.

## Rating
- Novelty: ⭐⭐⭐⭐ FEP is a new perspective for reward design, though the GRPO framework is unchanged.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing across 3 tasks, 9 datasets, 3 models, and 4 baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations and intuitive motivation.
- Value: ⭐⭐⭐⭐ Provides a strong baseline and design paradigm that is easily integrated into GRPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](../../ICLR2026/reinforcement_learning/unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)
- [\[ACL 2026\] Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward](verifier-free_rl_for_llms_via_intrinsic_gradient-norm_reward.md)
- [\[ACL 2026\] LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance](lang_reinforcement_learning_for_multilingual_reasoning_with_language-adaptive_hi.md)
- [\[ICLR 2026\] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](../../ICLR2026/reinforcement_learning/whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)
- [\[ACL 2026\] ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring](argus_policy-adaptive_ad_governance_via_evolving_reinforcement_with_adversarial_.md)

</div>

<!-- RELATED:END -->
