---
title: >-
  [Paper Note] GRPO is Secretly a Process Reward Model
description: >-
  [ICML 2026][LLM Reasoning][GRPO] This paper theoretically proves that GRPO + ORM is **equivalent** to a process reward RL objective with a Monte-Carlo PRM under the mild condition of "shared prefixes within a group." It reveals a hidden bug in vanilla GRPO—uneven prefix lengths cause majority tokens in high-reward trajectories to receive negative adva
tags:
  - ICML 2026
  - LLM Reasoning
  - GRPO
date: 2026-05-08
content_hash: cc612ddadf3d9239
---
# GRPO is Secretly a Process Reward Model

**Conference**: ICML 2026  
**arXiv**: [2509.21154](https://arxiv.org/abs/2509.21154)  
**Code**: https://github.com/coli-saar/grpo-prm/ (Available)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: GRPO, Process Reward, Advantage Normalization, Mathematical Reasoning, RL Training Acceleration

## TL;DR
This paper theoretically proves that GRPO + ORM is **equivalent** to a process reward RL objective with a Monte-Carlo PRM under the mild condition of "shared prefixes within a group." It reveals a hidden bug in vanilla GRPO—uneven prefix lengths cause majority tokens in high-reward trajectories to receive negative advantages—and proposes $\lambda$-GRPO, which utilizes a PRM-aware normalization to consistently outperform GRPO on reasoning benchmarks with approximately 2x faster training.

## Background & Motivation
**Background**: In RL training for LLM mathematical reasoning, PRMs (Process Reward Models) provide scores for each intermediate step, offering much finer credit assignment than ORM (Outcome Reward), and are typically used with PPO + GAE. The selling point of GRPO (DeepSeekMath) is the **removal of the critic and GAE**, using group-relative reward normalization as the advantage—simple and memory-efficient. Consequently, it is widely adopted (tool use, RLHF, math reasoning), but due to the absence of GAE, almost all GRPO works are restricted to ORMs.

**Limitations of Prior Work**: Integrating PRMs into GRPO requires non-trivial algorithmic modifications (e.g., TreeRPO, GroupPRM, TreeRL), which increases implementation complexity and sacrifices GRPO's simplicity. Furthermore, training neural PRMs is expensive (requiring step-level labels) and prone to reward-hacking.

**Key Challenge**: Traditional approaches treat "GRPO with ORM" and "PRM-aware RL" as distinct entities. However, during the rollout phase, GRPO samples multiple trajectories from the same prompt, which naturally form a prefix-sharing tree. This tree itself carries process-level information that has never been explicitly utilized.

**Goal**: (1) Mathematically prove that vanilla GRPO, under the shared-prefix assumption, is a PRM-aware objective and quantify its underlying PRM; (2) Use this analytical tool to identify hidden bugs in the GRPO objective; (3) Fix these bugs without introducing an explicit PRM.

**Key Insight**: The authors notice a simple fact: a trajectory's advantage $a_i$ in GRPO is uniformly distributed across all its tokens. If a trajectory shares a long prefix with multiple **high-scoring** trajectories in the group, that prefix is actually "good." However, vanilla GRPO misidentifies this prefix as "bad" because it calculates advantages based solely on individual trajectory rewards. Deriving this from a prefix-tree perspective clarifies the issue.

**Core Idea**: View GRPO as "RL performing MC-PRM on a prefix tree," identify the asymmetry in the normalization term, and fix it with a simple $\lambda$ factor correction.

## Method

### Overall Architecture
The paper presents a theoretical chain: "proving equivalence $\rightarrow$ diagnosing bugs via equivalence $\rightarrow$ fixing bugs." First, under two mild assumptions, trajectories from the same prompt are organized into a prefix tree. The authors prove that the vanilla GRPO loss is identically equal to a Monte-Carlo PRM objective built on this tree—meaning GRPO has implicitly been performing process-level credit assignment all along. Second, this PRM perspective identifies asymmetries in vanilla GRPO's normalization. Third, the $\lambda$-GRPO is introduced with a $\lambda$ factor to fix this, requiring only one line of code changes.

### Key Designs

**1. The Prefix Tree $\mathcal B(\mathbb G)$: Translating GRPO into an MC-PRM Objective**
To define the implicit process reward in GRPO, the authors formalize which tokens belong to the same process step. For a group $\mathbb G=\{y^{(1)},\dots,y^{(|\mathbb G|)}\}$, a process set $\mathcal B(\mathbb G)=\{\lambda\subseteq\mathbb G\mid \exists n\geq 0,\forall y^{(i)},y^{(k)}\in\lambda: y_{:n}^{(i)}=y_{:n}^{(k)}\}$ is defined. Each $\lambda$ is a set of trajectories sharing a prefix, forming a tree where node $\lambda$ corresponds to a step spanning $[s(\lambda), e(\lambda))$. The reward for this step is the mean outcome reward of trajectories in that node: $r_\lambda = \frac{1}{|\lambda|}\sum_{y^{(i)}\in\lambda} r^{(i)}$. The key conclusion is that under the assumptions of $\mu=1$ (single update), DAPO-style token-level objective, and ignoring clipping, the MC-PRM-aware loss $L_{\text{PRM}}(\mathbb G)$ is numerically identical to $L_{\text{GRPO}}(\mathbb G)$. This equivalence provides the first PRM semantics for GRPO: process rewards do not require neural PRMs; **as long as trajectories share prefixes during rollout**, MC-PRM signals are provided for free.

**2. Defect Diagnosis: Mismatch between Advantage and Step Frequency**
The PRM perspective exposes a systematic bug in vanilla GRPO. Consider a trajectory with a high reward that shares a prefix with many low-scoring trajectories. In a PRM-aware view, the reward for the shared step is the average of all sub-trajectories, which is pulled down by the low-scoring ones. Consequently, tokens in that prefix receive a **negative** advantage. However, vanilla GRPO applies the same sample-level $a_i$ to all tokens. This violates the PRM requirement that "segmented advantages should be weighted by step frequency." Specifically, the denominator $\sum_{y^{(i)}}\text{len}(y^{(i)})$ injects systematic bias when token counts and step frequencies are misaligned. This mismatch simultaneously hinders **exploitation** (by suppressing known good prefixes) and **exploration** (by distorting search signals).

**3. $\lambda$-GRPO: A PRM-aware Normalization Factor**
The fix is lightweight: maintain the original sample-level advantage but change the denominator during token accumulation from "total group tokens" to a normalization term re-weighted by prefix tree node frequency. This is equivalent to multiplying each token by $\lambda_t = 1/n_t$, where $n_t$ is the occurrence frequency of that token's process step in the group. This ensures that high-frequency shared steps are not repeatedly penalized or rewarded, restoring symmetry. It retains GRPO's efficiency (no critic/GAE) while utilizing the free MC-PRM signal.

### Loss & Training
- Vanilla GRPO (under $\mu=1$, DAPO token-level assumption):
  $$L_{\text{GRPO}}(\mathbb G)=\frac{1}{\sum_{y^{(i)}}\text{len}(y^{(i)})}\sum_{y^{(i)}}\sum_t (P_{i,t}\cdot a_i - D_{i,t})$$, where $a_i=(r^{(i)}-r_{\text{mean}}(\mathbb G))/r_{\text{std}}(\mathbb G)$.
- $\lambda$-GRPO: Replaces the denominator with a PRM-aware normalization sum (weighted by process step frequency).
- Training Setup: Consistent with DeepSeekMath GRPO ($\mu=1$); RL on math reasoning SFT data; TRL framework; 2x acceleration comes from reaching peak validation accuracy faster.

## Key Experimental Results

### Main Results

| Setting | Training Time | Downstream Acc | Convergence Speed |
| :--- | :--- | :--- | :--- |
| Vanilla GRPO | $1\times$ baseline | baseline | baseline |
| $\lambda$-GRPO | Nearly same/step | $>$ baseline | $\sim 2\times$ faster to peak |
| Explicit PRM (PPO+GAE) | Slower | Hit by reward-hack | Slow |

### Ablation Study

| Config | Observation | Explanation |
| :--- | :--- | :--- |
| High shared prefix ratio | Significant gain | Rich implicit PRM signals |
| Low shared prefix (Diverse rollout) | Degenerates to GRPO | Consistent with theory: trivial PRM |
| Remove $\lambda$ (Tree view only) | Performance matches GRPO | Gain comes from $\lambda$ correction |

### Key Findings
- **Implicit PRM in GRPO is non-trivial**: Prefix sharing occurs frequently in real GRPO training, making the analysis practically relevant.
- **Systematic Bias**: Vanilla GRPO tends to assign negative advantages to early shared prefixes of high-reward trajectories, explaining why it sometimes decreases the probability of correct reasoning chains (impaired exploitation).
- **Convergence Acceleration**: The 2x speedup in reaching peak accuracy provides significant GPU time savings for industrial pipelines.
- **Zero Overhead**: No extra labels or forward passes required compared to neural PRMs or explicit MC-PRMs (VineRL).

## Highlights & Insights
- **Algorithm Equivalence as an Analytical Tool**: Rewriting GRPO in a PRM-aware form to diagnose and fix bugs is an elegant "theory-first" paradigm.
- **Free Process Rewards**: Demonstrates that "process rewards do not require PRM training; they just require shared prefixes," which is a disruptive conclusion for cost-sensitive RL.
- **Bug Visualization**: The use of concrete counter-examples makes the abstract normalization mismatch highly intuitive.
- **One-line Implementation**: The $\lambda$ correction can be easily hot-patched into mainstream frameworks like TRL or Verl.

## Limitations & Future Work
- The equivalence proof depends on $\mu=1$ and token-level loss assumptions; performance in sample-level GRPO or high $\mu$ settings needs further discussion.
- Experiments are focused on mathematical reasoning; validation in RLHF, tool use, or agent tasks is lacking.
- Implicit PRM quality depends on prefix sharing density; high temperatures or long trajectories might dilute the benefits.
- Lack of end-to-end comparison with sophisticated variants like TreeRPO or GroupPRM.

## Related Work & Insights
- **vs. TreeRPO / TreeRL**: These explicitly construct tree structures; this paper proves vanilla GRPO is a special case of such structures and achieves similar goals with fewer changes.
- **vs. VinePPO / treeRL**: These use MC rollouts for value estimation with extra forwards; this paper extracts estimates from existing group rollouts for free.
- **vs. DAPO**: DAPO addresses sample-level loss instability; this paper treats DAPO as a prerequisite and orthogonally fixes a normalization bug.
- **Inspiration**: Any algorithm using relative scoring within a group (e.g., DPO, RLAIF) can apply the "shared substructure $\rightarrow$ implicit process signal" perspective to check for systematic normalization bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The equivalence "GRPO is an implicit PRM" is a beautiful and previously unnoticed conclusion.
- Experimental Thoroughness: ⭐⭐⭐ Strong evidence in reasoning, but missing systematic cross-task validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent clarity, especially the use of counter-examples.
- Value: ⭐⭐⭐⭐⭐ ~2x training speedup and stability via a one-line patch is highly valuable for production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
