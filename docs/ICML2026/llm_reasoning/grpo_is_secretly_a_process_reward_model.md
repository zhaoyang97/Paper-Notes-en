---
title: >-
  [Paper Note] GRPO is Secretly a Process Reward Model
description: >-
  [ICML 2026][LLM Reasoning][GRPO] This paper theoretically proves that GRPO + ORM, under the mild condition of "intra-group trajectory shared prefixes…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "GRPO"
  - "process reward"
  - "advantage normalization"
  - "mathematical reasoning"
  - "RL training acceleration"
date: 2026-05-08
content_hash: 7ceac8cad2f3fe37
---

# GRPO is Secretly a Process Reward Model

**Conference**: ICML 2026  
**arXiv**: [2509.21154](https://arxiv.org/abs/2509.21154)  
**Code**: https://github.com/coli-saar/grpo-prm/ (available)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: GRPO, process reward, advantage normalization, mathematical reasoning, RL training acceleration

## TL;DR
This paper theoretically proves that GRPO + ORM, under the mild condition of "intra-group trajectory shared prefixes," is **equivalent to** a process reward RL objective with Monte-Carlo PRM, thereby revealing a hidden bug in vanilla GRPO—uneven prefix lengths cause most tokens in high-reward trajectories to receive negative advantage. The authors propose $\lambda$-GRPO, which performs PRM-aware normalization, consistently outperforming GRPO on reasoning benchmarks and achieving about 2× faster training.

## Background & Motivation
**Background**: In LLM mathematical reasoning RL training, PRM (Process Reward Model) can score each intermediate step, providing much finer credit assignment than ORM (Outcome Reward Model), and is typically used with PPO + GAE. GRPO (DeepSeekMath) is notable for **removing the critic and GAE**, using intra-group reward standardization as advantage—simple and memory-efficient, thus widely adopted (tool use, RLHF, math reasoning). However, without GAE, nearly all GRPO work can only use ORM.

**Limitations of Prior Work**: Integrating PRM into GRPO requires non-trivial algorithmic modifications (e.g., TreeRPO, GroupPRM, TreeRL), increasing implementation complexity and sacrificing GRPO's simplicity. Moreover, neural PRM training is costly (requires step-level annotation) and prone to reward hacking.

**Key Challenge**: The community has long treated "GRPO with ORM" and "PRM-aware RL" as separate issues. Yet, during rollout, GRPO samples multiple trajectories from the same prompt, naturally forming a prefix-sharing tree—this tree inherently carries process-level information, which has never been explicitly utilized.

**Goal**: (1) Mathematically prove that vanilla GRPO under the shared prefix assumption is a PRM-aware objective, and quantify the corresponding PRM; (2) Use this analytical tool to identify hidden bugs in the GRPO objective; (3) Fix this bug without introducing explicit PRM.

**Key Insight**: The authors observe a simple fact—a trajectory's advantage $a_i$ in GRPO is uniformly distributed across all its tokens; if this trajectory shares a long prefix with multiple **high-scoring** trajectories in the group, this prefix is actually "good"—but vanilla GRPO, using only the overall reward of a single trajectory as advantage, misclassifies this prefix as "bad." From the prefix tree perspective, the problem becomes clear.

**Core Idea**: View GRPO as "doing MC-PRM RL on a prefix tree," identify the asymmetry in the normalization term, and correct it with a simple $\lambda$ factor.

## Method

### Overall Architecture
Two steps:

1. **Theoretical Side** (Section 3): Under the mild assumptions of $\mu=1$, DAPO-style token-level objective, and ignoring clip, construct a prefix tree $\mathcal B(\mathbb G)$, where each node $\lambda$ represents a group of trajectories sharing a prefix, corresponding to a process step; define step-level reward as the mean reward of trajectories under that node. Prove that this MC-PRM-aware loss $L_{\text{PRM}}(\mathbb G)$ is numerically identical to $L_{\text{GRPO}}(\mathbb G)$.
2. **Algorithmic Side** (Sections 4–5): Using the prefix tree perspective, identify the mismatch between advantage and normalization denominator in vanilla GRPO, and propose $\lambda$-GRPO—insert a PRM-aware normalization factor into the loss so that each process step's effective weight matches its actual frequency in the group. This modification can be added to TRL with a one-line code change.

### Key Designs

1. **Construction of Prefix Tree $\mathcal B(\mathbb G)$ and Process Steps**:

    - **Function**: Formalizes "which tokens belong to the same process step"—all trajectories under the same prefix share a step, and the step's reward is the mean outcome reward of these trajectories.
    - **Mechanism**: For group $\mathbb G=\{y^{(1)},\dots,y^{(|\mathbb G|)}\}$, define the process set $\mathcal B(\mathbb G)=\{\lambda\subseteq\mathbb G\mid \exists n\geq 0,\forall y^{(i)},y^{(k)}\in\lambda: y_{:n}^{(i)}=y_{:n}^{(k)}\}$, forming a tree under $\supseteq$. Each node $\lambda$ corresponds to a step spanning $[s(\lambda), e(\lambda))$; step-level reward $r_\lambda = \frac{1}{|\lambda|}\sum_{y^{(i)}\in\lambda} r^{(i)}$, with advantage still normalized by group mean. Under $\mu=1$ and token-level loss, it is proven that $L_{\text{PRM}}(\mathbb G)=L_{\text{GRPO}}(\mathbb G)$.
    - **Design Motivation**: This gives "what GRPO is doing" a PRM semantics for the first time. The equivalence means that to obtain process reward, there is no need to train a neural PRM or modify the algorithm—**simply ensure trajectories share prefixes during rollout** to get MC-PRM signals for free. The authors empirically show (Section 3.2) that such prefix sharing is very common in real GRPO training—so this "implicit PRM" is almost always non-trivial.

2. **Defect Diagnosis: Mismatch between Advantage and Step Frequency**:

    - **Function**: Uses the prefix tree perspective to reveal a concrete counterexample in vanilla GRPO—most tokens in high-reward trajectories are assigned negative advantage, thus RL **reduces** their probability.
    - **Mechanism**: Consider trajectory JKLNQU in Figure 1, assuming its overall reward is above the group mean, but its prefix JKL is shared with several low-reward trajectories. In the PRM-aware view, the step-reward for JKL is the mean reward of all trajectories under JKL—this mean is pulled down by low-reward trajectories, causing the three tokens in JKL to receive **negative** advantage; only the final token U, unique to JKLNQU, receives positive advantage. But vanilla GRPO's token-level loss treats the entire trajectory as a unit, all tokens sharing the same sample-level $a_i$, which is inconsistent with the PRM view where "segment advantage should be weighted by step frequency"—specifically, the denominator $\sum_{y^{(i)}}\text{len}(y^{(i)})$ in the loss introduces systematic bias when token count and step frequency are mismatched.
    - **Design Motivation**: This diagnosis formalizes the intuition that "GRPO sometimes messes up good trajectories" into a concrete, formalizable bug.

3. **$\lambda$-GRPO: PRM-aware Normalization**:

    - **Function**: Adds a process-step-frequency-aware normalization factor $\lambda$ to GRPO's token-level loss, restoring the symmetry that "high-frequency shared steps should not be repeatedly penalized/rewarded."
    - **Mechanism**: Keeps the original GRPO sample-level advantage unchanged, but replaces the denominator in token accumulation from "total group token count" to a PRM-aware normalization term reweighted by prefix tree node frequency; equivalently, each token is multiplied by $\lambda_t = 1/n_t$ ($n_t$ is the number of times the token's process step appears in the group). Thus, tokens in high-frequency shared steps are not repeatedly pushed due to multiple occurrences. This can be patched into TRL's GRPO trainer with a single line.
    - **Design Motivation**: Retains GRPO's lightweight advantage of not needing critic/GAE, while leveraging the MC-PRM signal already available for free. The authors empirically show this modification has almost zero training time overhead, but consistently outperforms vanilla GRPO on multiple reasoning benchmarks, and **converges about 2× faster**—indicating cleaner gradient signals after bug fix.

### Loss & Training

- Vanilla GRPO (under $\mu=1$, DAPO token-level assumption):  
  $L_{\text{GRPO}}(\mathbb G)=\frac{1}{\sum_{y^{(i)}}\text{len}(y^{(i)})}\sum_{y^{(i)}}\sum_t (P_{i,t}\cdot a_i - D_{i,t})$, where $a_i=(r^{(i)}-r_{\text{mean}}(\mathbb G))/r_{\text{std}}(\mathbb G)$.
- $\lambda$-GRPO: Replace the denominator with a PRM-aware normalization sum (weighted by process step frequency), all else unchanged.
- Training setup: Same as DeepSeekMath GRPO, $\mu=1$ updates; RL on mathematical reasoning SFT data; TRL framework; 2× training acceleration comes from reaching peak validation accuracy faster.

## Key Experimental Results

### Main Results

| Setting | Training Time | Downstream Reasoning Acc | Convergence Speed |
|---------|--------------|-------------------------|------------------|
| Vanilla GRPO | $1\times$ baseline | baseline | baseline |
| $\lambda$-GRPO | almost same/step | all $>$ baseline | reaches peak $\sim 2\times$ faster |
| Explicit PRM (PPO+GAE) | much slower | affected by reward-hack | slow |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---------------|------------|-------------|
| Groups with high prefix sharing | $\lambda$-GRPO shows clear improvement | Rich implicit PRM signal |
| Sparse prefix sharing (diverse rollout) | $\lambda$-GRPO degrades to GRPO | Consistent with theory: no difference when PRM is trivial |
| Remove normalization reweighting (keep tree view, unchanged loss) | Performance reverts to GRPO | Shows gain comes from $\lambda$ correction, not the perspective itself |

### Key Findings
- **Implicit PRM in GRPO is almost always non-trivial in real training**: Empirical evidence shows prefix sharing in group rollout is frequent (structures like JKLNQU in Figure 1 are the norm, not exceptions), making the analysis practically relevant.
- **Bug direction is a systematic counterexample**: Vanilla GRPO tends to assign negative advantage to "early shared prefixes" of high-reward trajectories, which is a root cause of why GRPO-trained models sometimes **reduce** the probability of correct reasoning chains.
- **$\lambda$-GRPO's convergence acceleration is more significant than its performance gain**: Peak validation accuracy is reached in about half the steps, meaning substantial GPU time savings—highly valuable for industrial RL pipelines.
- **No extra annotation or forward passes required**: Compared to neural PRM, zero annotation cost; compared to explicit MC-PRM (VineRL), zero extra rollout.

## Highlights & Insights
- **"Algorithmic equivalence" as an analytical tool**: Rewriting vanilla GRPO into a PRM-aware form, diagnosing bugs in the rewritten form, then fixing them back in the vanilla framework—this is an elegant "theory-first" analysis paradigm, worth emulating in RL algorithm analysis.
- **Prefix tree perspective provides free MC-PRM**: Reveals that "to get process reward, no need to train PRM, just ensure rollout shares prefixes," a disruptive conclusion for those seeking to save PRM annotation costs.
- **Concrete bug illustration**: The JKLNQU example turns the abstract normalization mismatch into a whiteboard-drawable counterexample, greatly enhancing readability.
- **$\lambda$ correction is a one-line code change**: The trick can be hot-patched into mainstream frameworks like TRL/verl at almost zero cost.

## Limitations & Future Work
- The equivalence proof relies on $\mu=1$ and token-level (DAPO) loss assumptions; under sample-level GRPO or $\mu>1$ multi-update settings, the prefix tree perspective no longer strictly holds, and whether the bug persists in the same direction requires further discussion.
- Experiments are conducted on mathematical reasoning benchmarks; systematic validation on other main GRPO applications (RLHF, tool use, agent) is lacking.
- The quality of implicit PRM depends on prefix sharing density; with long trajectories or diverse sampling (high temperature), sharing becomes sparse; rollout strategies that actively encourage prefix sharing may be needed for $\lambda$-GRPO to consistently benefit.
- Lacks end-to-end comparison with explicit process reward GRPO variants like TreeRPO / GroupPRM, so it is unclear whether explicit or implicit approaches are superior.

## Related Work & Insights
- **vs TreeRPO / TreeRL (feng2025, ji2025)**: These explicitly construct tree-structured PRMs and modify the algorithm; this paper proves vanilla GRPO is a special case of such tree PRMs, and $\lambda$-GRPO achieves similar goals with fewer changes.
- **vs VinePPO / treeRL (MC-based PRM)**: VinePPO uses MC rollout to estimate step value but requires extra forward passes; this paper hides MC estimation in intra-group shared prefixes, incurring zero extra cost.
- **vs DAPO**: DAPO proposes a token-level objective to address instability in sample-level loss training; this paper takes DAPO as a premise and orthogonally fixes another normalization bug.
- **Insights**: Any algorithm that "does relative scoring within group/batch" (e.g., DPO's pairwise comparison, RLAIF's multi-sample aggregation) can use the "prefix/shared substructure → implicit process signal" perspective to check for systematic bias in normalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "GRPO is an implicit PRM" is a beautiful and previously unnoticed equivalence result
- Experimental Thoroughness: ⭐⭐⭐ Clear empirical gains and acceleration on reasoning benchmarks, but lacks systematic cross-task (RLHF/tool use) validation
- Writing Quality: ⭐⭐⭐⭐⭐ The JKLNQU counterexample in Figure 1 thoroughly explains the abstract bug, with a clean chain from hypothesis, proof, to fix
- Value: ⭐⭐⭐⭐⭐ One-line patch yields ~2× training acceleration + stable performance gain, immediately usable in industrial RL pipelines

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](../../AAAI2026/llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)

</div>

<!-- RELATED:END -->
