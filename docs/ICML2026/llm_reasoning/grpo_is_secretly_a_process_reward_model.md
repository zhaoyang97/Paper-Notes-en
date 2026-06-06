---
title: >-
  [Paper Note] GRPO is Secretly a Process Reward Model
description: >-
  [ICML 2026][LLM Reasoning][GRPO] This paper theoretically proves that under the mild condition of "trajectories within a group sharing prefixes…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "GRPO"
  - "Process Reward"
  - "Advantage Normalization"
  - "Mathematical Reasoning"
  - "RL Training Acceleration"
date: 2026-05-08
content_hash: b87f1dae2d1ce066
---

# GRPO is Secretly a Process Reward Model

**Conference**: ICML 2026  
**arXiv**: [2509.21154](https://arxiv.org/abs/2509.21154)  
**Code**: https://github.com/coli-saar/grpo-prm/ (Yes)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: GRPO, Process Reward, Advantage Normalization, Mathematical Reasoning, RL Training Acceleration

## TL;DR
This paper theoretically proves that under the mild condition of "trajectories within a group sharing prefixes," GRPO + ORM is **equivalent** to a process-reward RL objective with a Monte-Carlo PRM. It reveals a hidden bug in vanilla GRPO—unbalanced prefix lengths cause most tokens in high-reward trajectories to receive negative advantages—and proposes $\lambda$-GRPO with a PRM-aware normalization, which consistently outperforms GRPO on reasoning benchmarks and trains approximately 2x faster.

## Background & Motivation
**Background**: In RL training for LLM mathematical reasoning, PRMs (Process Reward Models) can score each intermediate step, providing much finer credit assignment than ORM (Outcome Reward). Consequently, they are typically used with PPO + GAE. GRPO (DeepSeekMath) distinguishes itself by **removing the critic and GAE**, using group-wise reward normalization as the advantage—simple and memory-efficient. While widely applied (tool use, RLHF, math reasoning), the absence of GAE has restricted almost all GRPO work to using ORMs.

**Limitations of Prior Work**: Integrating PRMs into GRPO requires non-trivial algorithmic modifications (e.g., TreeRPO, GroupPRM, TreeRL), increasing implementation complexity and sacrificing GRPO's simplicity. Furthermore, training neural PRMs is expensive (requiring step-level labeling) and prone to reward-hacking.

**Key Challenge**: "GRPO with ORM" and "PRM-aware RL" have long been treated as separate entities. However, during the rollout phase, GRPO samples multiple trajectories from the same prompt, which naturally form a prefix-sharing tree. This tree itself carries process-level information that has never been explicitly utilized.

**Goal**: (1) Mathematically prove that vanilla GRPO is a PRM-aware objective under the shared-prefix assumption and quantify what its corresponding PRM looks like; (2) Use this analytical tool to identify hidden bugs in the GRPO objective; (3) Fix the bug without introducing an explicit PRM.

**Key Insight**: The authors notice a simple fact: a trajectory's advantage $a_i$ in GRPO is distributed uniformly across all its tokens. If this trajectory shares a long prefix with several **highly-scored** trajectories, that prefix is actually "good"—but vanilla GRPO, using only the total reward of a single trajectory for the advantage, may incorrectly calculate that prefix as "bad." Deriving this from a prefix-tree perspective clarifies the issue.

**Core Idea**: Treat GRPO as "RL performing MC-PRM on a prefix tree," identify the asymmetry in the normalization term, and apply a simple $\lambda$ factor correction.

## Method

### Overall Architecture
A two-step approach:

1.  **Theoretical Side** (Section 3): Under two mild assumptions ($\mu=1$ and a DAPO-style token-level objective ignoring clipping), construct a prefix tree $\mathcal B(\mathbb G)$. Each node $\lambda$ represents a set of trajectories sharing a prefix, corresponding to a process step. Step-level rewards are defined using the mean reward of internal trajectories. It is proved that this MC-PRM-aware loss $L_{\text{PRM}}(\mathbb G)$ is numerically identical to $L_{\text{GRPO}}(\mathbb G)$.
2.  **Algorithmic Side** (Section 4–5): Use the prefix-tree perspective to identify the defect where advantages and normalization denominators are mismatched in vanilla GRPO. Propose $\lambda$-GRPO—inserting a PRM-aware normalization factor into the loss to reconcile each process step's effective weight with its actual frequency in the group. This modification can be added to TRL with a single line of code.

### Key Designs

1.  **Construction of Prefix Tree $\mathcal B(\mathbb G)$ and Process Steps**:
    - **Function**: Formalize "which tokens belong to the same process step"—all trajectories under the same prefix share a step, and the step's reward is the mean outcome reward of those trajectories.
    - **Mechanism**: For a group $\mathbb G=\{y^{(1)},\dots,y^{(|\mathbb G|)}\}$, define the process set $\mathcal B(\mathbb G)=\{\lambda\subseteq\mathbb G\mid \exists n\geq 0,\forall y^{(i)},y^{(k)}\in\lambda: y_{:n}^{(i)}=y_{:n}^{(k)}\}$, forming a tree via the $\supseteq$ relation. Each node $\lambda$ corresponds to a step spanning $[s(\lambda), e(\lambda))$; the step-level reward is $r_\lambda = \frac{1}{|\lambda|}\sum_{y^{(i)}\in\lambda} r^{(i)}$, while advantages remain normalized by the group mean.
    - **Design Motivation**: Provide GRPO with PRM semantics for the first time. This equivalence implies that process rewards can be obtained without training neural PRMs or modifying algorithms—**simply by letting trajectories share prefixes during rollout**, MC-PRM signals are obtained for free. Empirical evidence (Section 3.2) shows this prefix sharing is common in real GRPO training, making this "hidden PRM" non-trivial.

2.  **Defect Diagnosis: Mismatch between Advantage and Step Frequency**:
    - **Function**: Use the prefix-tree view to reveal a specific counter-example where most tokens of a high-reward trajectory are assigned negative advantages, thus **decreasing** their probability via RL.
    - **Mechanism**: Consider a trajectory JKLNQU. Suppose its total reward is above the group mean, but its prefix JKL is shared with multiple low-scoring trajectories. In a PRM-aware view, the reward for JKL is the "mean reward of all trajectories under JKL"—which is dragged down by low-scoring paths, giving JKL tokens a **negative** advantage. Only the final token U unique to JKLNQU receives a positive advantage. However, vanilla GRPO treats the trajectory as a whole, sharing the same sample-level $a_i$ across all tokens, which is inconsistent with the PRM view that "segmented advantages should be weighted by step frequency." Specifically, the denominator $\sum_{y^{(i)}}\text{len}(y^{(i)})$ introduces systematic bias when token counts and step frequencies mismatch.
    - **Design Motivation**: This diagnosis converts the intuition that "GRPO occasionally messes up good trajectories" into a formalizable bug.

3.  **$\lambda$-GRPO: PRM-aware Normalization**:
    - **Function**: Add a process-step-frequency-aware normalization factor $\lambda$ to the token-level loss to restore the symmetry that "high-frequency shared steps should not be repeatedly penalized/rewarded."
    - **Mechanism**: Keep original sample-level advantages but replace the summation denominator with a normalization term re-weighted by prefix tree node frequency. This is equivalent to multiplying each token by $\lambda_t = 1/n_t$ (where $n_t$ is the frequency of the process step in the group). One-line patch for TRL's GRPO trainer.
    - **Design Motivation**: Retain the lightweight advantage of GRPO (no critic/GAE) while utilizing free MC-PRM signals. This modification has near-zero overhead but consistently outperforms vanilla GRPO and **converges ~2x faster**.

### Loss & Training
- Vanilla GRPO (under $\mu=1$ and DAPO token-level assumptions):
  $$L_{\text{GRPO}}(\mathbb G)=\frac{1}{\sum_{y^{(i)}}\text{len}(y^{(i)})}\sum_{y^{(i)}}\sum_t (P_{i,t}\cdot a_i - D_{i,t})$$, where $a_i=(r^{(i)}-r_{\text{mean}}(\mathbb G))/r_{\text{std}}(\mathbb G)$.
- $\lambda$-GRPO: Replaces the denominator with a PRM-aware normalization sum (weighted by process step frequency).
- Training Setup: Consistent with DeepSeekMath GRPO, $\mu=1$ update; RL on math reasoning SFT data; TRL framework. The 2x acceleration comes from reaching peak validation accuracy faster.

## Key Experimental Results

### Main Results

| Setup | Training Time | Downstream Reasoning Acc | Convergence Speed |
|-------|---------------|--------------------------|-------------------|
| Vanilla GRPO | $1\times$ baseline | baseline | baseline |
| $\lambda$-GRPO | Nearly same/step | Consistently $>$ baseline | ~2$\times$ faster to peak |
| Explicit PRM (PPO+GAE) | Much slower | Vulnerable to reward-hacking | Slower |

### Ablation Study

| Configuration | Observation | Explanation |
|---------------|-------------|-------------|
| High shared prefix ratio | Significant gain for $\lambda$-GRPO | Rich implicit PRM signals |
| Sparse shared prefixes | $\lambda$-GRPO degrades to GRPO | Consists with theory: no difference for trivial PRMs |
| Remove $\lambda$ weighting | Performance reverts to GRPO | Gain comes from $\lambda$ correction, not just the tree view |

### Key Findings
- **Implicit PRM in GRPO is non-trivial in real training**: Empirical results show prefix sharing occurs frequently in group rollouts.
- **The bug is a systematic counter-example**: Vanilla GRPO tends to assign negative advantages to "early shared prefixes" of high-reward trajectories, explaining why GRPO models sometimes **reduce** the probability of correct reasoning chains.
- **$\lambda$-GRPO's convergence acceleration is more significant than performance gain**: Reaching peak validation accuracy ~2x faster implies massive GPU time savings.
- **Zero additional labels or forward passes**: Compared to neural PRMs, there is zero labeling cost; compared to explicit MC-PRM (VineRL), there are zero extra rollouts.

## Highlights & Insights
- **"Algorithmic Equivalence" as an Analytical Tool**: Rewriting vanilla GRPO in a PRM-aware form to diagnose and fix bugs is an elegant "theory-first" paradigm.
- **Free MC-PRM via Prefix Trees**: Demonstrating that process rewards can be obtained without training PRMs—as long as rollouts share prefixes—is a disruptive conclusion.
- **Concretization of the Bug**: The JKLNQU example makes the abstract normalization mismatch highly readable.
- **One-line Code Patch**: The trick can be hot-patched into mainstream frameworks like TRL/verl with near-zero deployment cost.

## Limitations & Future Work
- Equivalence proof relies on $\mu=1$ and token-level (DAPO) loss assumptions; results under sample-level loss or $\mu>1$ require further investigation.
- Experiments focus on math reasoning; systematic verification on RLHF, tool use, or agents is lacking.
- Quality of implicit PRM depends on prefix-sharing density; high temperature or long trajectories may require strategies to encourage shared prefixes.
- Lacks end-to-end comparison with variants like TreeRPO or GroupPRM to determine if implicit is strictly better than explicit.

## Related Work & Insights
- **vs. TreeRPO / TreeRL**: They explicitly construct tree-structured PRMs; this paper shows vanilla GRPO is naturally a special case of such a PRM.
- **vs. VinePPO / treeRL (MC-based PRM)**: VinePPO uses extra rollouts for step value estimation; this paper hides MC estimates in group-shared prefixes with zero overhead.
- **vs. DAPO**: DAPO addresses instability in sample-level loss; this paper uses DAPO as a premise to fix a separate normalization bug.
- **Insight**: Any algorithm performing relative scoring within a group (e.g., DPO, RLAIF) can apply the "prefix/shared sub-structure $\to$ implicit process signal" view to check for normalization bias.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ "GRPO is an implicit PRM" is an elegant and previously unnoticed equivalence.
- **Experimental Thoroughness**: ⭐⭐⭐ Clear improvements on math benchmarks, but missing systematic verification across other tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The counter-example used to explain the abstract bug is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ A one-line patch providing ~2x acceleration and stable gains is immediately useful for industrial RL pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ICML 2026\] TRACE: Evaluating LLM CoT Reasoning Process Quality with the Toulmin Argumentation Model](trace_toulmin-based_reasoning_assessment_through_constructive_elements_for_llm_c.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)

</div>

<!-- RELATED:END -->
