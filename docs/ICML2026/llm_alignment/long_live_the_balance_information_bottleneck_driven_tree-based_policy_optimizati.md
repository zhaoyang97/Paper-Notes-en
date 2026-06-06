---
title: >-
  [Paper Note] Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization
description: >-
  [ICML 2026][LLM Alignment][Information Bottleneck] This paper proposes IB-Score, a step-level metric derived from Information Bottleneck (IB) theory to quantify the "exploration-exploitation balance." Based on this…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Information Bottleneck"
  - "GRPO"
  - "Tree Search"
  - "Exploration-Exploitation Balance"
  - "Step-level Advantage"
date: 2026-05-08
content_hash: 12090f02057c94aa
---

# Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.28109](https://arxiv.org/abs/2605.28109)  
**Code**: https://github.com/ (The paper claims it is open-sourced, but the repository address is not provided in the main text)  
**Area**: Alignment RLHF / LLM Reasoning / Online Reinforcement Learning  
**Keywords**: Information Bottleneck, GRPO, Tree Search, Exploration-Exploitation Balance, Step-level Advantage

## TL;DR
This paper proposes IB-Score, a step-level metric derived from Information Bottleneck (IB) theory to quantify the "exploration-exploitation balance." Based on this, it designs IB-guided Tree sampling (IBTree) combined with step-level local and global advantages. Experimental results on Qwen3-1.7B/8B show an average improvement of 2.9–3.6% over GRPO, while sampling 50% more trajectories within the same token budget.

## Background & Motivation

**Background**: Current mainstream post-training for LLM reasoning involves online RL, represented by GRPO—where $G$ trajectories are sampled independently for the same problem, using result rewards for group-relative advantage normalization followed by clipped policy gradient updates.

**Limitations of Prior Work**: The authors reproduced GRPO using Qwen3-8B-Base and observed two coupled failure modes: ① **Over-exploitation** — policy entropy plummets after a few training steps, forcing the model into a deterministic local optimum. The $G$ trajectories within a group converge, causing the Effective Rate (Eff-Rate, the proportion of groups with non-zero reward variance) to decline, leading to sparse learning signals; ② **Over-exploration** — using higher clipping or entropy regularization to maintain entropy results in a continued drop in Eff-Rate, or even entropy explosion and training collapse in severe cases. Neither regularization outperformed vanilla GRPO in Table 1.

**Key Challenge**: There is a lack of an objective metric that can **quantify the exploration-exploitation balance at a step-level** rather than at the token or sequence level. Token-level entropy is noisy due to irrelevant tokens, while sequence-level IB regularization (e.g., IBRO) is too coarse to evaluate intermediate reasoning steps.

**Goal**: (1) Provide a step-level, online-estimable balance metric; (2) Integrate it into the GRPO optimization target; (3) Address the implementation bottleneck of the high cost associated with step-level IB estimation.

**Key Insight**: Apply the IB objective $\min I(X;Z) - \beta I(Z;Y)$ to LLM reasoning by treating the reasoning step sequence $\tau=\{s_i\}$ as the bottleneck representation $Z$, the question $q$ as input $X$, and the correct answer $a^*$ as output $Y$. Consequently, the **exploration term** naturally becomes $H(s_i|q,s_{<i})$ (step-level generation entropy), and the **exploitation term** becomes $H(s_i|a^*,q,s_{<i})$ (uncertainty of the step given the correct answer; lower values indicate higher "answer relevance"). The tradeoff between the two using $\beta$ provides a clean balance measure.

**Core Idea**: Use **IB-Score** to simultaneously grade whether a step is "diverse enough" and "aligned enough with the correct answer." Then, construct a search tree that **only branches at nodes with the highest IB-Score**, achieving both efficient step-level IB estimation and high-performance online sampling.

## Method

### Overall Architecture
For each question $q$, IB-TPO executes an IBTree: starting with the root $q$, it grows an initial tree using $B_0$ independent rollouts. This is followed by $L-1$ rounds of expansion, where in each round, the $K$ non-leaf nodes with the highest IB-Scores are selected to branch into $B$ new trajectories (sharing prefixes and reusing vLLM prefix cache). The total number of trajectories is $G = B_0 + (L-1)\cdot K\cdot B$. Once the tree is built, each node is assigned a **global advantage** $A_{GL}$ (the reward density from that node to the answer minus the root) and a **local advantage** $A_{IB}$ (step-level signal from IB-Score). These are weighted and fed into the standard GRPO clipped objective to update the policy.

### Key Designs

1. **IB-Score: Step-level Exploration-Exploitation Balance Metric**:
    - **Function**: Scores any non-leaf node $s_i$ in the reasoning tree. A high score indicates that the position possesses both "exploration diversity and information gain," making it suitable as a branching point and a direction for gradient optimization.
    - **Mechanism**: Starting from the IB objective $J_{IB}(\tau)=(\beta+1)H(\tau|q)-\beta H(\tau|a^*,q)$, it is decomposed into a sum of step-level components. Since direct distribution computation is infeasible, the authors sample $B$ candidate sub-steps $\{s_i^b\}$ under the shared prefix $(q, s_{<i})$ as Monte Carlo samples. Tsallis entropy with $\alpha=2$ is used instead of Shannon entropy for numerical stability with sparse samples, leading to $J_{IB}(s_i)\approx \tfrac{1+\beta}{B}\sum_b \eta_1(s_i^b)\cdot \eta_2(s_i^b)$, where $\eta_1(s_i^b)=\hat p(a^*|s_i^b)/\hat p(a^*|s_{i-1})-(1+1/\beta)$ is the "information gain" from environment feedback, and $\eta_2(s_i^b)=\pi_\theta(s_i^b)$ is the model's "confidence" in that branch. A key insight is that $J_{IB}$ essentially depends on $\mathrm{Cov}(\eta_1,\eta_2)$—balance is not about simply increasing entropy, but strategically allocating confidence to the most informative branches.
    - **Design Motivation**: IB-Score diagnoses GRPO failure modes: after a few training steps, $\mathrm{Cov}(\eta_1,\eta_2)$ collapses from positive to zero, indicating that the confidence distribution has become uniform and decoupled from path correctness. Steps are partitioned using `\n\n`, which is training-agnostic and natural, and ablation (Table 5) proves robustness to segmentation noise.

2. **IBTree: IB-guided Selective Branching Tree Search**:
    - **Function**: Acts as both a sampler and an MC estimator for IB-Score under a fixed token budget, sampling 50% more trajectories than independent sampling.
    - **Mechanism**: Instead of branching at every step (space explosion) or based on token entropy (which can be distracted by irrelevant tokens), it selects top-$K$ nodes based on IB-Score each round to branch into $B$ new rollouts with shared prefixes. In experiments, $(B_0,L,K,B)=(4,9,1,1)$ produced a "tall and thin" tree of 12 trajectories with token consumption comparable to 8 independent trajectories. As new rollouts provide more sub-samples for parent nodes, the estimation accuracy of IB-Score improves, creating a positive feedback loop.
    - **Design Motivation**: Table 3 compares independent, random, fixed-width, entropy-guided, and IB-guided branching. IB-guided branching with $\beta=5$ achieved the highest Eff-Rate (60.2%) and Avg-Rate (23.2%), while consuming 37% fewer tokens than 12 independent samples. This shows that the selection of branching locations is more critical than the branching strategy itself.

3. **Hybrid Step-level IB Local Advantage + Global Advantage**:
    - **Function**: Converts the IB-Score signal into step-level advantages for the GRPO clipped objective, bypassing the sparsity issues of result-only rewards.
    - **Mechanism**: $\tilde J_{IB}(s)=\eta_1(s)\cdot \eta_2(s)$ is rewritten into a policy gradient form $A_{IB}(s)\cdot w(s)$, where $w(s)=\pi_\theta(s)/\pi_{ref}(s)$. The **local advantage** $A_{IB}(s)=\big(\hat p(a^*|s)/\hat p(a^*|s_p)-(1+1/\beta)\big)\cdot \pi_{ref}(s)$ measures whether moving from parent $s_p$ to $s$ increases the probability of finding the correct answer. The **global advantage** $A_{GL}(s)=(\hat p(a^*|s)-\hat p(a^*|q))/\mathrm{std}(\{R(\tau)\})$ measures overall improvement relative to the root. The final advantage is $A(s)=A_{GL}(s)+\lambda\cdot A_{IB}(s)$, with $\lambda=0.1$ being optimal in ablations.
    - **Design Motivation**: Every node in the tree structure naturally has multiple child rollouts, allowing for the calculation of local values like $\hat p(a^*|s)$. Table 2 shows that using IBTree alone improves AMC 24 performance by +4.4% over vanilla GRPO, with an additional +2.2% from IBTPO local advantages. However, replacing IBTree with random or EPTree significantly diminishes the effectiveness of the IB advantage.

### Loss & Training
The base objective follows GRPO's token-level clipping + KL regularization, replacing $A_{i,t}$ with step-level $A(s)=A_{GL}+\lambda A_{IB}$. Training used DAPO-Math-17K (17K math problems with result rewards), Qwen3-1.7B/8B-Base, $lr=10^{-6}$, KL weight $0.001$, 1 epoch, 8×A100. Sampling temperature 0.7, top-p 0.95, max 2K tokens per trajectory. Tree parameters $(B_0,L,K,B)=(4,9,1,1)$, IB weight $\beta=5$, $\lambda=0.1$.

## Key Experimental Results

### Main Results
Benchmarks include MATH-500, AIME 24/25, AMC 23/24 (in-domain math), plus GPQA Diamond and IFEval (out-of-domain), all reported as avg@32:

| Model | Method | MATH-500 | AIME 25 | AMC 24 | GPQA | IFEval | Average |
|------|------|----------|---------|--------|------|--------|------|
| Qwen3-1.7B | Vanilla GRPO | 66.8 | 4.5 | 19.7 | 26.5 | 24.0 | 26.3 |
| Qwen3-1.7B | TreeRL (Prev. SOTA) | 67.2 | 4.6 | 20.6 | 26.8 | 23.5 | 26.8 |
| Qwen3-1.7B | **IBTPO** | **70.1** | **6.7** | **23.4** | **29.0** | **26.9** | **29.2** |
| Qwen3-8B | Vanilla GRPO | 81.5 | 13.6 | 39.4 | 38.1 | 42.0 | 40.7 |
| Qwen3-8B | TreeRL | 82.5 | 14.9 | 40.5 | 39.8 | 42.5 | 42.0 |
| Qwen3-8B | **IBTPO** | **83.3** | **15.3** | **46.0** | **41.7** | **46.2** | **44.3** |

Ours surpassed GRPO by 2.9% and 3.6% on the two scales respectively, also outperforming IBRO (sequence-level IB) and tree-based baselines TreeRL and TreePO.

### Ablation Study

| Config | AIME 25 | AMC 24 | GPQA | Note |
|------|---------|--------|------|------|
| Vanilla GRPO | 13.6 | 39.4 | 38.1 | Baseline |
| + IBTree (Tree Sampling) | 15.0 | 43.8 | 40.8 | Significant gain from sampler alone |
| + IBTPO Adv (Eq 16) | 14.2 | 42.5 | 41.2 | Gain from advantage function alone |
| + RandTree & IBTPO Adv | 14.5 | 39.8 | 37.3 | Gains drop with random tree |
| + EPTree & IBTPO Adv | 15.0 | 42.3 | 40.9 | Entropy-guided tree underperforms |
| **+ IBTree & IBTPO Adv** | **15.3** | **46.0** | **41.7** | **Full Version** |

Branching strategy comparison (Qwen3-8B, 1024 problems):

| Branching Strategy | G | Eff-Rate | Avg-Rate | tokens |
|----------|---|----------|----------|--------|
| Independent | 8 | 54.7% | 19.6% | 7,469 |
| Independent | 12 | 59.8% | 20.1% | 12,035 |
| Random | 12 | 48.4% | 20.0% | 7,579 |
| Entropy (TreeRL) | 12 | 57.8% | 21.6% | 7,784 |
| **IB-guided ($\beta=5$)** | 12 | **60.2%** | **23.2%** | 7,592 |

### Key Findings
- Both IBTree and IBTPO Adv provide gains independently, but **they must be paired**; using IBTPO Adv with RandTree results in a 0.8% drop on GPQA compared to GRPO.
- $\beta=5$ is the sweet spot for Eff-Rate and Avg-Rate; $\lambda=0.1$ is optimal, while $\lambda=0.5$ causes collapse as local signals overwhelm the result reward.
- Training dynamics show that the collapse of $\mathrm{Cov}(\eta_1,\eta_2)$ to zero is the root cause of GRPO's stagnation. IBTPO is the only method that maintains IB-Score and Covariance throughout training.
- Segmentation via `\n\n` is robust; performance remains stable even with 10% random noise in split points.

## Highlights & Insights
- **Step-level + Online IB is key**: Previous work like IBRO used sequence-level advantages, essentially flattening the tree and losing intermediate structure. Ours utilizes Tsallis entropy and MC estimation to make step-level IB online and differentiable.
- **$\mathrm{Cov}(\eta_1,\eta_2)$ explains why entropy bonus fails**: High entropy $\neq$ balance; what matters is whether high confidence is allocated to high information gain branches.
- **Dual Role of IBTree**: One tree serves as both a branching guide and an MC estimator, saving computational costs. Implementing this with prefix cache makes it wall-clock efficient.
- **Low-cost Heuristic**: Using `\n\n` for step segmentation is a simple, effective trick that avoids the need for a separate step segmenter.

## Limitations & Future Work
- **Latency**: Multiple rounds of tree expansion introduce serial sampling overhead. Although parallelized, IBTree can be slightly slower than independent sampling depending on the prefix cache implementation.
- **Domain Focus**: Experiments were limited to math (DAPO-Math-17K) and a few out-of-domain benchmarks. Generalization to code or multimodal reasoning is yet to be verified.
- **MC Variance**: The IB-Score estimation relies heavily on sibling samples ($B$). Using $B=1$ suggests that multi-round expansion implicitly compensates for sample size, which might not hold for shallow trees.
- **Hyperparameters**: $\beta=5$ is determined empirically without an adaptive scheduling strategy for task difficulty or model scale.

## Related Work & Insights
- **vs IBRO (Lei et al., 2025)**: Both use IB, but IBRO is sequence-level and acts as a coarse entropy regularizer. Ours is step-level and directly optimizes the advantage term.
- **vs TreeRL (Hou et al., 2025)**: TreeRL branches based on token entropy, which is sensitive to noise. Ours uses IB-Score, aligning branching with actual decision points.
- **vs TreePO (Li et al., 2025)**: TreePO uses fixed-width branching; ours uses selective branching based on IB-Score to optimize token consumption and performance simultaneously.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Moving IB from sequence/token level to step-level while coupling it with tree search is a complete "diagnosis + solution" loop.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two model scales, seven benchmarks, five branching strategies, and full ablations on $\beta, \lambda$.
- **Writing Quality**: ⭐⭐⭐⭐ Clear derivations for IB and Algorithm 1; effective use of conceptual and dynamic plots.
- **Value**: ⭐⭐⭐⭐ The Eff-Rate and Covariance metrics are valuable for diagnosing GRPO variants. The implementation is based on ms-swift and likely to be adopted by the RLHF community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](../../ICLR2026/llm_alignment/hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](../../ICLR2026/llm_alignment/slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ACL 2026\] Pref-CTRL: Preference Driven LLM Alignment using Representation Editing](../../ACL2026/llm_alignment/pref-ctrl_preference_driven_llm_alignment_using_representation_editing.md)
- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](../../AAAI2026/llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](../../ICLR2026/llm_alignment/learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)

</div>

<!-- RELATED:END -->
