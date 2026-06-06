---
title: >-
  [Paper Note] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models
description: >-
  [ACL 2026][Reinforcement Learning][dLLM] Addressing two reliability bottlenecks in Diffusion Language Model (dLLM) RL—sparse rewards and biased probability estimation—the authors propose d-TreeRPO. This method organizes…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "dLLM"
  - "GRPO"
  - "Tree-structured rollout"
  - "Step-wise advantage"
  - "Self-distillation"
date: 2026-05-08
content_hash: 323b52a8ffcac47b
---

# d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models

**Conference**: ACL 2026  
**arXiv**: [2512.09675](https://arxiv.org/abs/2512.09675)  
**Code**: https://github.com/THU-BPM/d-TreeRPO (Available)  
**Area**: Reinforcement Learning / Diffusion Language Models / Reasoning LLMs  
**Keywords**: dLLM, GRPO, Tree-structured rollout, Step-wise advantage, Self-distillation

## TL;DR
Addressing two reliability bottlenecks in Diffusion Language Model (dLLM) RL—sparse rewards and biased probability estimation—the authors propose d-TreeRPO. This method organizes rollouts into a tree structure, calculating step-wise advantages bottom-up from verifiable rewards at leaf nodes. Simultaneously, it theoretically proves that higher model confidence leads to more accurate single-step forward probability estimation and adopts a time-scheduled self-distillation loss to tighten the policy in later training stages. Ultimately, it achieves gains of +86.2% on Sudoku, +51.6% on Countdown, +4.5% on GSM8K, and +5.3% on Math500 using LLaDA-8B-Instruct.

## Background & Motivation
**Background**: dLLMs (e.g., LLaDA, Dream, Seed Diffusion) replace autoregressive decoding with parallel denoising for faster inference. Reinforcement learning (PPO/GRPO families) has been adapted to enhance their reasoning capabilities, with representative works including Diffu-GRPO, VRPO (LLaDA-1.5), wd1, SAPO, GDPO, and TraceRL.

**Limitations of Prior Work**: dLLM RL faces issues in two core components. (1) **Sparse/Unverifiable Rewards**: Most methods uniformly broadcast the final outcome reward to all tokens, causing the advantage to degenerate into a constant, or they use learned process reward models which risk reward hacking. (2) **Probability Estimation Bias**: Since dLLMs denoise in arbitrary orders, true token probability should be the expectation over all decoding sequences (Eq. 3), which is computationally intractable. ELBO estimation is a biased lower bound requiring multiple forward passes, while single-step estimation is cheap but lacks theoretical guarantees.

**Key Challenge**: Precision vs. Computation—obtaining fine-grained and verifiable advantages requires more rollouts; obtaining accurate $\log\pi$ requires multiple forward passes. Both are expensive and were previously designed independently, lacking a unified framework.

**Goal**: (i) Provide step-wise and verifiable advantages within the same rollout budget; (ii) develop a strategy to improve single-step estimation accuracy without increasing forward passes; (iii) theoretically quantify the relationship between the two.

**Key Insight**: Structuring rollouts as a tree naturally creates state transitions ("denoising segments") between parent and child nodes. Propagating verifiable outcome rewards from leaves bottom-up yields step-wise rewards. Furthermore, the gap between single-step estimation and true expectation narrows monotonically with policy confidence, justifying a time-scheduled self-distillation loss to increase confidence in later stages.

**Core Idea**: Use "tree structure + bottom-up reward" to solve sparse rewards and "time-scheduled self-distillation" to solve probability estimation bias, both synergistic within a single GRPO framework.

## Method

### Overall Architecture
For each prompt $q$, the $N$ denoising steps are compressed into $H=N/s$ tree steps using a merging factor $s$. Each non-leaf node expands into $B$ child nodes, each corresponding to $s$ consecutive denoising steps, resulting in $B^H$ fully generated leaf nodes. Leaf nodes are scored with verifiable outcome rewards. The value of an internal node is the average of its children's values (Eq. 6), and the advantage of a parent-to-child transition is the difference between their values (Eq. 7). The loss is calculated per child node in a GRPO style on "subtrees of depth 1," using single-step forward estimated token probabilities for importance ratios. An additional time-scheduled self-distillation loss is used to make the policy converge toward the token distribution of high-advantage child nodes in later stages.

### Key Designs

1. **Tree-structured Rollout + Bottom-up Step-wise Advantage**:
    - **Function**: Decomposes a single outcome reward into local rewards across parent-child transitions, providing verifiable, fine-grained advantages for every generation step.
    - **Mechanism**: Parent value $V_p = \frac{1}{|C_p|}\sum_{c\in C_p} V_c$; parent-child advantage $A^c_p = V_c - V_p$. Relative advantages are calculated within a group of children sharing the same parent and "broadcast to all newly generated tokens in that child node." Coupled with block-wise decoding (block length $b$, where each tree step covers an integer number of blocks), child nodes from the same parent resolve tokens at the same positions, ensuring comparable single-step probabilities.
    - **Design Motivation**: In dLLMs, "sparse outcome reward + uniform broadcasting" makes all tokens see the same advantage, leaving credit assignment to luck. With tree structuring, the advantage of each step comes from actual value differences, essentially constructing an implicit verifiable process reward and bypassing the hacking risks of training reward models.

2. **Single-step Forward + High-Probability Error Bound**:
    - **Function**: Provides theoretical guarantees for token probability estimation without increasing the number of forward passes.
    - **Mechanism**: Use $\hat{p}_d := f^d_\theta(o_d \mid p)$ as the probability estimate of the child token given the parent state. Theorem 1 proves that for any $\delta\in(0,1)$, there exists a confidence gap $\epsilon_{d,\delta}=\max\{1-\hat{p}_d, 1-q_{d,1-\delta}\}$ such that $\Pr_{\sigma\sim Q}\left[\log\frac{q_{\tau(d,\sigma)}(\sigma)}{\hat{p}_d} \le -\log(1-\epsilon_{d,\delta})\right]\ge 1-\delta$. This implies that as the model becomes more confident ($\hat{p}_d$ approaches 1), the log-ratio bias between path probability and single-step estimation decreases.
    - **Design Motivation**: ELBO is accurate but expensive; single-step estimation is cheap but unproven. Binding the error bound to model confidence creates a feasible optimization path: improving determinism to approach true expectation.

3. **Time-scheduled Self-distillation Loss**:
    - **Function**: Avoids interfering with exploration early in training while sharpening the policy later to tighten the error bound mentioned above.
    - **Mechanism**: In each depth-1 subtree, positive-advantage child nodes $C^+_p$ are selected. Advantage-weighted soft labels $w_c = \frac{\exp(A^c_p/\tau(t))}{\sum_{c'\in C^+_p}\exp(A^{c'}_p/\tau(t))}$ are calculated using temperature $\tau(t)=\tau_{\max}(1-t/T)^\beta$ and aggregated into a target distribution $P^{\sigma_i}_{\text{target}}$ by position-token. The distillation loss is $\mathcal{L}_{\text{distill}} = \lambda(t)\cdot\frac{1}{k}\sum_i \mathrm{KL}(P^{\sigma_i}_{\text{target}}\,\|\,\pi^{\sigma_i}_\theta)$, where $\lambda(t)=\lambda_{\max}\frac{e^{\gamma t/T}-1}{e^\gamma-1}$ increases monotonically and $\tau(t)$ decreases monotonically.
    - **Design Motivation**: Theorem 1 suggests "sharper is more accurate," but early sharpness kills exploration. The time schedule encodes "explore first, converge later" into the loss shape, corresponding to entropy bonus decay, but the target distribution comes from positive advantage nodes, providing RL signals rather than pure entropy.

### Loss & Training
The complete objective $\mathcal{L}_{\text{d-TreeRPO}}$ (Eq. 19) = **policy-gradient loss** (GRPO-style clipped objective within tree steps, using $A^c_p$ as advantage and $f^{d_i}_\theta/f^{d_i}_{\theta_{\text{old}}}$ as importance ratio, with clip range $[1-\epsilon, 1+\epsilon]$) + **KL loss** ($\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ on parent states) + **self-distillation loss** (Eq. 17, with time-increasing weight $\lambda(t)$). Training uses LoRA $r=128, \alpha=64$, lr $3\times 10^{-5}$; $\tau_{\max}=2, \beta=0.7$; $\lambda_{\max}=0.003, \gamma=2$; 256 tokens decoded with block length 32 and 128 denoising steps; tree parameters $H=2, B=4$.

## Key Experimental Results

### Main Results
Main results for LLaDA-8B-Instruct (256 / 512 token columns, relative gains over base in parentheses, excerpt from Table 1):

| Method | Sudoku-256 | Countdown-256 | GSM8K-256 | Math500-256 |
|------|-----------:|--------------:|----------:|------------:|
| LLaDA-8B-Instruct (base) | 6.7 | 19.5 | 76.7 | 32.4 |
| Diffu-GRPO | 12.9 | 31.3 | 79.8 | 34.1 |
| VRPO (LLaDA-1.5) | 12.8 | 22.3 | 80.1 | 35.6 |
| wd1 | 25.2 | 51.2 | 80.8 | 34.4 |
| SAPO | 20.3 | 52.0 | 80.6 | 33.8 |
| GDPO | 25.7 | 64.1 | 81.1 | 37.0 |
| TraceRL | 25.6 | 50.4 | 80.3 | 35.6 |
| **d-TreeRPO** | **92.9 (+86.2)** | **71.1 (+51.6)** | **81.2 (+4.5)** | **37.7 (+5.3)** |

d-TreeRPO also leads on LLaDA-MoE-7BA1B-Instruct and maintains SOTA across various decoding strategies with block-length $\in \{32, 64, 128\}$ (Figure 3).

### Ablation Study
Table 2 compares sparse reward variants under the same rollout budget ($B^H = 4^2 = 16$ leaf nodes per query); Table 3 shows self-distillation ablations.

| Configuration | Sudoku | Countdown | GSM8K | Math500 | Description |
|------|-------:|----------:|------:|--------:|------|
| d-TreeRPO (Full) | 92.9 | 71.1 | 81.2 | 37.7 | Tree-based + Distillation |
| Sparse-Tree (Tree rollouts, outcome reward broadcast) | 22.4 | 38.2 | 80.4 | 35.4 | Confirms fine-grained credit is key, not just budget |
| Sparse-Flat (16 standard rollouts, no tree) | 24.6 | 37.1 | 80.4 | 35.6 | Same conclusion as above |
| d-TreeRPO w/o self-distill | 89.8 | 66.4 | 80.9 | 36.1 | Distillation adds +2–5 points |
| d-TreeRPO + diversity loss (Reverse schedule) | 84.2 | 63.4 | 78.5 | 35.2 | Maintaining high entropy is detrimental |

Probability estimation error ($\log(p_{\text{true}}/\hat{p})$, lower is more accurate, Table 4): Full d-TreeRPO on Sudoku is $1.25 \pm 1.15$, jumping to $2.64 \pm 1.97$ without self-distillation, and further degrading to $2.83 \pm 2.02$ with diversity loss. This directly validates the theorem that "sharper is more accurate."

### Key Findings
- **Fine-grained credit assignment is the root cause of performance**: When the rollout budget is fixed at 16, using tree rollouts without bottom-up advantages (Sparse-Tree) barely beats the original Diffu-GRPO. Only with step-wise advantages do puzzle tasks jump from ~25 to ~90.
- **Self-distillation tunes both RL performance and estimation precision**: Removing it not only lowers rewards but doubles token probability estimation error—a "closed-loop evidence" linking theoretical analysis (Theorem 1) to empirical gains.
- **Forward schedule (exploit later) far outperforms reverse schedule**: While reverse scheduling increases rewards faster initially, it plateaus after 0.75, showing that forced sharpness early on traps the policy in local optima.
- **Controllable training time**: Compared to SAPO (≈72h), d-TreeRPO converges in ≈48h on 8×H20, comparable to GDPO/TraceRL but with far superior results.

## Highlights & Insights
- "Tree rollouts + bottom-up value propagation" grounds the process reward construction entirely in verifiable outcome rewards, bypassing the traditional but messy route of PRM training and reward hacking.
- Theorem 1 links "token confidence" with "single-step estimation error" and uses self-distillation for active control—this "theory $\rightarrow$ engineering knob" loop is rare in dLLM RL and generalizes to any scenario involving an "estimation bias vs. policy entropy" trade-off.
- The time-scheduled $\lambda(t)$ and $\tau(t)$ act as twin "soft schedules," which is more stable than hard-switching between exploration and exploitation. This "exponential rise + polynomial decay" combo could be applied to standard AR LLM RL training.

## Limitations & Future Work
- Evaluation is limited to tasks with automatically verifiable outcome rewards (puzzles + math); in subjective writing/dialogue tasks where verifiable signals are missing, bottom-up credit assignment is harder to apply.
- Tree structures explode exponentially with $B^H$. $H=4$ was too costly for the available resources, meaning deep process rewards remain expensive; hybrid pruning/sampling schemes are needed.
- Self-distillation only votes on positive-advantage children, ignoring negative samples, which may waste information in multi-modal reasoning.
- Experiments used LoRA; consistency in full-parameter fine-tuning remains unverified. The sensitivity to base model "format compliance" (seen with Dream's exclusion) suggests formatting robustness is a prerequisite.

## Related Work & Insights
- **vs. Diffu-GRPO / wd1**: They use single-step estimation for $\log\pi$ without quantifying bias and broadcast outcome rewards directly. d-TreeRPO provides an error bound and step-wise rewards, showing clear performance gaps.
- **vs. VRPO (LLaDA-1.5) / GDPO**: These follow the ELBO route, which is expensive due to multiple forward passes and only provides a lower bound. This paper demonstrates "high confidence makes single-step estimation sufficient," reducing computational overhead.
- **vs. SAPO**: SAPO adds an explicit process reward but still broadcasts it. d-TreeRPO's process reward is implicitly defined by tree structure and leaf rewards, avoiding consistency hacks.
- **vs. TraceRL**: TraceRL uses a learned diffusion value model for token-level advantages, which is differentiable but prone to value bias; this paper uses bottom-up averaging, eliminating extra networks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of tree rollout, estimation error bound, and time-scheduled self-distillation is a first for merging engineering and theory in dLLM RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong main results and controlled budget comparisons, though missing multi-turn dialog validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Pipelines align well with formulas; the appendix provides thorough proofs and baseline descriptions.
- Value: ⭐⭐⭐⭐ Directly sets new records on multiple dLLM reasoning benchmarks; the logic is applicable to all "sparse reward + biased estimation" RL settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](../../ICML2026/reinforcement_learning/learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](../../ICLR2026/reinforcement_learning/fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [\[ACL 2026\] Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](beyond_majority_voting_towards_fine-grained_and_more_reliable_reward_signal_for_.md)
- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](../../ICML2026/reinforcement_learning/d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)
- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)

</div>

<!-- RELATED:END -->
