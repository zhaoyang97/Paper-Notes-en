---
title: >-
  [Paper Note] Pareto-Guided Optimal Transport for Multi-Reward Alignment
description: >-
  [ICML 2026][Image Generation][Multi-reward alignment] PG-OT shifts "multi-reward text-to-image alignment" from "weighted global summation" to "constructing a Pareto frontier for each prompt and using Sinkhorn optimal tra…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Multi-reward alignment"
  - "reward hacking"
  - "Pareto frontier"
  - "optimal transport"
  - "JDR/JCR"
date: 2026-05-08
content_hash: 074f41171cb5e93c
---

# Pareto-Guided Optimal Transport for Multi-Reward Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.13155](https://arxiv.org/abs/2605.13155)  
**Code**: None  
**Area**: Text-to-Image Alignment / Multi-Reward Optimization  
**Keywords**: Multi-reward alignment, reward hacking, Pareto frontier, optimal transport, JDR/JCR  

## TL;DR
PG-OT shifts "multi-reward text-to-image alignment" from "weighted global summation" to "constructing a Pareto frontier for each prompt and using Sinkhorn optimal transport to move dominated samples to the frontier," introducing two new metrics, Joint Domination Rate / Joint Collapse Rate, to expose reward hacking masked by averaging. On Parti-Prompts, JDR₂ reaches 47.98%, an 11% improvement over strong baselines, with a human evaluation win rate close to 80%.

## Background & Motivation

**Background**: Post-training preference alignment for text-to-image (T2I) models commonly uses one or more reward models for RLHF-style fine-tuning, with objectives like $\mathcal{L}(x) = C - \sum_k w_k R^k(x)$, treating $C$ as a global upper bound and maximizing the weighted reward.

**Limitations of Prior Work**: (i) **Reward hacking** is prevalent—reward scores increase while image quality collapses; (ii) **Multi-reward fusion methods** rely on weight tuning, which is costly and unstable; (iii) **Mean-based evaluation metrics** (average reward improvement) mask hacking: one dimension may increase while others decrease, yet the mean remains positive.

**Key Challenge**: The root cause is the mismatch between "using a global constant $C$ as the reward upper bound" and "the actual maximum achievable reward varying greatly across prompts." Figure 1 empirically shows that under the ICT reward, the maximum reward across 20 prompts varies widely; using a global $C$ forces all prompts to align to the same upper bound, causing prompts with naturally lower upper bounds to be pushed until shortcuts are taken → reward hacking.

**Goal**: (a) Theoretically prove that "heterogeneous upper bounds + global objective" inevitably pushes some samples toward hacking; (b) Design an optimization strategy that is "prompt-wise upper bound aware"; (c) Propose reliable metrics to detect hacking; (d) Distinguish between strong/weak reward models and design corresponding protection mechanisms.

**Key Insight**: Naturally embed multi-reward alignment into the Pareto optimization framework—since different prompts have different achievable upper bounds, treat the "set of optimal samples within the same prompt" as the prompt's Pareto frontier, and use OT to "transport" non-optimal samples within the same prompt to the frontier; strong reward signals expand the frontier online, weak reward signals lock the frontier offline and use a VLM agent to detect collapse.

**Core Idea**: "Prompt-specific Pareto frontier as the target distribution + OT as the transport operator," with JDR/JCR as Pareto-style metrics to quantify "true gain vs. false hacking."

## Method

### Overall Architecture
The PG-OT training loop operates per prompt $p_i$: (1) Construct the Pareto frontier $\mathcal{R}^{front}(p_i)$ for the prompt—**offline strategy** for weak rewards pre-generates $M$ samples and extracts the frontier using a dominance matrix, **online strategy** for strong rewards dynamically expands the frontier during training; (2) The T2I model generates a batch of samples, from which $n$ dominated samples are identified as the source distribution $\mu_i$, with the frontier as the target distribution $\nu_i$; (3) Use entropy-regularized Sinkhorn to solve for $\gamma^\ast_i$, and backpropagate the transport cost $\sum_{m,j} c(y_i^m, x_i^j)\gamma$ to the T2I model parameters; (4) Use a VLM agent to monitor early collapse patterns for weak rewards, removing the reward and rolling back to a stable checkpoint if triggered; (5) Finally, use JDR/JCR to evaluate genuine improvement.

### Key Designs

1. **Prompt-specific Pareto Frontier Construction (Avoiding Global Upper Bound)**:

    - **Function**: Explicitly encodes the heterogeneity of "different prompts having different upper bounds" as independent optimization targets, eliminating the incentive to "force low-upper-bound prompts to take shortcuts."
    - **Mechanism**: For a given prompt $p_i$, generate $M$ candidate samples $\{x_i^j\}_{j=1}^M$, obtaining the reward vector set $\mathcal{R}_{i,M}^{(pre)} = \{\tilde R(x_i^j)\}$. Construct an $M\times M$ dominance matrix $A$ ($A_{mn}=1$ if $\tilde R(x_i^m)\succ\tilde R(x_i^n)$); the Pareto frontier is the set of samples with zero times being dominated: $\mathcal{R}^{front}(p_i) = \{\tilde R(x_i^j)\mid \sum_m A_{mj}=0\}$. Pareto dominance is defined as "all dimensions ≥ and at least one dimension >."
    - **Design Motivation**: Each prompt receives its own estimate of the "truly achievable upper bound," so the model is no longer pushed toward unattainable global extremes; Figure 1 empirically demonstrates significant prompt-wise upper bound heterogeneity, with the global $C$ being the root cause.

2. **Sinkhorn Optimal Transport Moves Dominated Samples to the Frontier**:

    - **Function**: In reward space, moves currently dominated samples in the batch to the frontier points at minimal total cost, serving as a differentiable training signal.
    - **Mechanism**: Source distribution $\mu_i = \{\tilde R(x_i^j)\mid x_i^j$ is dominated by all points in $\mathcal{R}^{front}\}$, target distribution $\nu_i = \mathcal{R}^{front}(p_i)$. The ground cost is the squared Euclidean distance in reward space $c(y_i^m, x_i^j) = \|\tilde R(y_i^m) - \tilde R(x_i^j)\|_2^2$. Solve the entropy-regularized OT $\gamma^\ast_i = \arg\min_{\gamma\in\Pi(\mu_i, \nu_i)} \sum_{m,j} c(y_i^m, x_i^j)\gamma(y_i^m, x_i^j)$ using the Sinkhorn algorithm. The inner product of $\gamma^\ast$ and $c$ is backpropagated to the T2I model, essentially moving dominated samples toward their nearest frontier points. The training pipeline adopts DRaFT-K-style differentiable reward optimization (reward model is differentiable with respect to the image).
    - **Design Motivation**: OT preserves the geometry of reward space (not simply picking the maximum), avoiding "all samples collapsing to the same target" compared to weighted sum or single-point maximization; Sinkhorn's differentiability allows the entire transport cost to be backpropagated to the generative model, which is essential in practice.

3. **Online / Offline Dual Strategy + VLM Decision Agent**:

    - **Function**: Adopts different frontier construction strategies based on reward model strength, and promptly stops loss when weak rewards are about to collapse.
    - **Mechanism**: The authors calibrate reward accuracy using the Pick-a-Pic and Pick-High high-quality human preference datasets (Table 1: CLIP 60.3%, HPS 72.9%, ICT 87.6%, HP 88.5%), classifying the latter two as "strong" and the former two as "weak." **Strong rewards** use the **online strategy**: dynamically collect samples per prompt during training to expand the frontier, encouraging the T2I model to autonomously explore new Pareto optimal points; **Weak rewards** use the **offline strategy**: pre-generate $M$ samples to compute a fixed frontier, using only this as the target during training to prevent noisy signals from contaminating the frontier. A GPT-4o agent, equipped with a "mild collapse reference set," detects early mild collapse; if triggered, the weak reward is removed and the model rolls back to the last stable checkpoint.
    - **Design Motivation**: Strong rewards align with human preferences, so online frontier expansion is "exploratory + robust"; weak rewards are unreliable, and allowing them to expand the frontier online only introduces more noise, so offline locking + proactive detection and removal is a more stable strategy.

### Loss & Training
The training loss is the total OT transport cost $\sum_{m,j}c(y_i^m, x_i^j)\gamma^\ast(y_i^m, x_i^j)$ backpropagated to the T2I model (using DRaFT-K-style differentiable rewards). The VLM agent triggers collapse checks at each validation step, collecting "mild collapse" cases for each reward as in-context references. In addition to traditional single-reward win rates, evaluation metrics include $\mathrm{JDR}_K = \tfrac{1}{N}\sum_i \mathbb{1}(\mathbf{R}_i\succ\mathbf{R}_{i,b})$ and $\mathrm{JCR}_K = \tfrac{1}{N}\sum_i \mathbb{1}(\mathbf{R}_{i,b}\succ\mathbf{R}_i)$.

## Key Experimental Results

### Main Results
Base model: SD3.5-Turbo; 4 rewards: ICT, HP (strong), CLIP, HPS (weak); evaluated on Parti-Prompts.

| Method | ICT Win Rate | HP Win Rate | CLIP Win Rate | HPS Win Rate | JDR₂ ↑ | JDR₄ ↑ | JCR₄ ↓ |
|---|---|---|---|---|---|---|---|
| +ICT Single Reward | 56.99 | 36.83 | 47.06 | 48.71 | 20.59 | 7.66 | 10.17 |
| +HP Single Reward | 52.45 | **90.26** | 44.30 | 57.29 | 36.15 | 13.73 | 4.11 |
| Weighted 2:3:2:3 | 50.80 | 56.43 | 46.51 | 86.03 | 28.31 | 13.42 | 2.57 |
| Reward Soup 3:2:1:4 | 50.80 | 53.74 | 43.32 | 85.29 | 26.29 | 10.85 | 3.19 |
| Weighted-Sum (w/o OT) | 52.63 | 56.86 | 46.94 | 82.48 | 29.84 | 13.66 | 3.49 |
| **PG-OT** | 56.43 | 85.23 | 43.63 | 61.70 | **47.98** | **17.10** | **2.39** |

**Human evaluation win rate is close to 80%**—one of the paper's strongest selling points. PG-OT does not achieve the highest score on all single rewards (lower than weighted-sum on CLIP/HPS), but achieves the highest JDR₂/JDR₄ and lowest JCR₄, indicating its samples are more broadly superior across dimensions compared to baselines, with minimal collapse.

### Ablation Study

| Variant | Key Observations |
|---|---|
| Global Upper Bound (weighted-sum) | Each single reward increases but JDR is low, JCR is high, indicating hacking risk |
| OT Only, No Pareto Frontier | OT lacks a clear target, results similar to weighted-sum |
| Pareto Only, No OT | Frontier points are discrete, no differentiable signal |
| No Distinction Between Strong/Weak Reward | Weak reward online frontier expansion contaminates the target |
| No VLM Agent Detection | Cannot promptly stop loss after weak reward collapse |
| Full PG-OT | JDR₂ 47.98%, JCR₄ only 2.39%, both improvement and hacking suppression |

Table 2 shows the trend for each reward when optimizing CLIP-only: CLIP increases by +7.27% while HPS drops -2.78% and HP drops -4.38%, a typical example of reward conflict and partial hacking, highlighting the necessity of JDR/JCR detection introduced by PG-OT.

### Key Findings
- Single-reward optimization (e.g., +HP achieving HP win rate of 90.26%) achieves the highest score in that dimension but poor JDR/JCR, indicating that traditional "single reward win rate" metrics are seriously misleading.
- Weighted-sum tuning yields limited gains: across 4 ratios, JDR₄ is only 12.44%–13.66%, far below PG-OT's 17.10%.
- The JCR metric reveals hidden collapse not visible to mean-based metrics: the Separate-Cons configuration achieves an HPS win rate of 61.21%, which seems fine, but JCR₄ is as high as 6.68%, indicating many samples degrade across all dimensions simultaneously.

## Highlights & Insights
- The observation of "prompt-wise heterogeneous upper bounds" is incisive, turning the commonly assumed "global reward" into a provable source of hacking, supported by both theory and empirical evidence.
- Using the Pareto frontier as the OT target distribution is a **conceptual leap**—upgrading from single-point maximization to "distribution-to-distribution transport," offering structural advantages in multi-objective settings.
- JDR/JCR shift multi-reward alignment evaluation from "mean scores" to "Pareto comparison," serving as general diagnostic standards for future multi-reward RLHF work.
- The symmetric treatment of strong/weak rewards (online expansion vs. offline locking) plus dynamic reward pruning by the VLM agent is a pragmatic engineering solution for the uneven quality of rewards in real RLHF training.

## Limitations & Future Work
- The quality of the offline Pareto frontier depends on the number of pre-generated samples $M$ and the reliability of the reward model; if the reward is severely misaligned, the frontier becomes an incorrect target.
- Sinkhorn's computational cost on large batches and sensitivity to the regularization coefficient make it hyperparameter-sensitive; the paper does not provide detailed hyperparameter ablation.
- The VLM agent triggers using a "mild collapse reference set," relying on manually annotated collapse cases, which may fail for unseen new collapse patterns.
- Experiments are limited to 4 rewards and a single SD3.5-Turbo backbone; generalization to more rewards and across diffusion/AR architectures requires further validation.

## Related Work & Insights
- **vs Weighted Sum / Reward Soup**: These methods circumvent conflicts by "tuning weights"; PG-OT directly acknowledges conflicts via the Pareto frontier and uses OT to transport samples, structurally avoiding weight tuning.
- **vs DRaFT / Diffusion-DPO**: Traditional differentiable reward fine-tuning pushes rewards toward a global objective; PG-OT uses prompt-specific frontiers as targets, explicitly suppressing hacking risk.
- **vs Pareto-MTL / Multi-task Learning**: MTL often uses MGDA to find Pareto directions; PG-OT does not search for directions in weight space but uses OT in sample space, avoiding MGDA's instability in high-dimensional tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Prompt-wise Pareto frontier + OT" and "JDR/JCR" are both original, with clear conceptual contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines (single reward/weighted/Reward Soup/OT without Pareto) and human evaluation, but only one backbone
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical foundation and insightful motivation analysis (three types of hacking mechanisms)
- Value: ⭐⭐⭐⭐⭐ Highly generalizable for multi-reward RLHF; JDR/JCR can be directly adopted by the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](../../NeurIPS2025/image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[CVPR 2026\] COT-FM: Cluster-wise Optimal Transport Flow Matching](../../CVPR2026/image_generation/cot-fm_cluster-wise_optimal_transport_flow_matching.md)
- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](../../ICLR2026/image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
