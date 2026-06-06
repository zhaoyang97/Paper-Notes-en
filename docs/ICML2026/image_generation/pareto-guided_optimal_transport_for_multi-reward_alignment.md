---
title: >-
  [Paper Note] Pareto-Guided Optimal Transport for Multi-Reward Alignment
description: >-
  [ICML 2026][Image Generation][Multi-reward alignment] PG-OT reformulates "multi-reward text-to-image alignment" from a "weighted global sum" to "constructing a Pareto frontier for each prompt individually and using Sinkh…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Multi-reward alignment"
  - "reward hacking"
  - "Pareto frontier"
  - "optimal transport"
  - "JDR/JCR"
date: 2026-05-08
content_hash: 1f050839264ef20c
---

# Pareto-Guided Optimal Transport for Multi-Reward Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.13155](https://arxiv.org/abs/2605.13155)  
**Code**: None  
**Area**: Text-to-Image Alignment / Multi-Reward Optimization  
**Keywords**: Multi-reward alignment, reward hacking, Pareto frontier, optimal transport, JDR/JCR  

## TL;DR
PG-OT reformulates "multi-reward text-to-image alignment" from a "weighted global sum" to "constructing a Pareto frontier for each prompt individually and using Sinkhorn optimal transport to move dominated samples toward the frontier." It introduces two new metrics, Joint Domination Rate (JDR) and Joint Collapse Rate (JCR), to expose reward hacking masked by mean values. On Parti-Prompts, it achieves a $\mathrm{JDR}_2$ of 47.98%, an 11% improvement over strong baselines, with a human win rate of nearly 80%.

## Background & Motivation

**Background**: Post-training preference alignment for Text-to-Image (T2I) models generally utilizes RLHF-style fine-tuning with one or more reward models. The objective function typically takes the form $\mathcal{L}(x) = C - \sum_k w_k R^k(x)$, treating $C$ as a global upper bound to maximize weighted rewards.

**Limitations of Prior Work**: (i) **Reward hacking** is prevalent—reward scores continue to rise while image quality collapses; (ii) **Multi-reward fusion methods** rely on weight searching, which incurs high tuning costs and unstable gains; (iii) **Mean-based evaluation metrics** (average gain across rewards) mask hacking, where one dimension rises while others drop, yet the average remains positive.

**Key Challenge**: The authors identify the root cause as the mismatch between "using a global constant $C$ as the reward upper bound" and the "vast differences in the actual maximum achievable rewards across different prompts." Empirical evidence in Figure 1 shows that under the ICT reward, the distribution of maximum rewards across 20 prompts varies significantly. Using a global $C$ forces all prompts toward the same upper bound; for prompts with naturally low upper bounds, gradients continue to push until shortcuts are taken, leading to reward hacking.

**Goal**: (a) Theoretically prove that "heterogeneous upper bounds + global targets" inevitably lead to sample hacking; (b) Design a "per-prompt upper-bound aware" optimization strategy; (c) Propose evaluation metrics that reliably detect hacking; (d) Distinguish behaviors between strong and weak reward models and design corresponding protection mechanisms.

**Key Insight**: Naturally embed multi-reward alignment into a Pareto optimization framework. Since achievable bounds differ by prompt, the "set of optimal samples within the same prompt" is treated as the Pareto frontier for that prompt. OT is used to "transport" non-optimal samples of the same prompt to this frontier. Strong reward signals expand the frontier online, while weak reward signals lock the frontier offline, with a VLM agent detecting potential collapse.

**Core Idea**: Use "prompt-specific Pareto frontiers as target distributions + OT as the transport operator," and quantify "true gains vs. fake hacking" using the Pareto-style metrics JDR and JCR.

## Method

### Overall Architecture
The PG-OT training loop operates for each prompt $p_i$ as follows: (1) Construct the Pareto frontier $\mathcal{R}^{front}(p_i)$ for that prompt; the **offline strategy** pre-generates $M$ samples for weak rewards and extracts the frontier using a domination matrix, while the **online strategy** dynamically collects in-batch samples during training for strong rewards to expand the frontier. (2) Generate a batch of samples using the current T2I model, identifying $n$ samples dominated by the frontier as the source distribution $\mu_i$, with the frontier as the target distribution $\nu_i$. (3) Solve for $\gamma^\ast_i$ using entropy-regularized Sinkhorn, backpropagating the transport cost $\sum_{m,j} c(y_i^m, x_i^j)\gamma$ to the T2I model parameters. (4) Use a VLM agent to monitor early collapse patterns in weak rewards; if triggered, the specific reward is removed, and the model rolls back to a stable checkpoint. (5) Finally, evaluate true gains using JDR/JCR.

### Key Designs

1.  **Prompt-specific Pareto Frontier Construction (Avoiding Global Upper Bounds)**:
    - **Function**: Explicitly encode the heterogeneity of "different prompts having different upper bounds" into independent optimization targets for each prompt, thereby eliminating the incentive for low-upper-bound prompts to take shortcuts.
    - **Mechanism**: Given a prompt $p_i$, $M$ candidate samples $\{x_i^j\}_{j=1}^M$ are generated to obtain a set of reward vectors $\mathcal{R}_{i,M}^{(pre)} = \{\tilde R(x_i^j)\}$. An $M\times M$ domination matrix $A$ is constructed ($A_{mn}=1$ if $\tilde R(x_i^m)\succ\tilde R(x_i^n)$). The Pareto frontier is the set of samples with zero "times being dominated": $\mathcal{R}^{front}(p_i) = \{\tilde R(x_i^j)\mid \sum_m A_{mj}=0\}$. Pareto dominance is defined as "all dimensions $\ge$ and at least one dimension $>$."
    - **Design Motivation**: Each prompt receives its own estimate of the "true achievable upper bound." The model is no longer pushed toward global extrema it cannot reach. Figure 1 experimentally confirms significant prompt-wise heterogeneity in upper bounds, identifying the global $C$ as the core issue.

2.  **Sinkhorn Optimal Transport to the Frontier**:
    - **Function**: Move samples in the current batch that are dominated by the frontier toward frontier points in the reward space with minimal total cost, providing differentiable training signals.
    - **Mechanism**: The source distribution $\mu_i = \{\tilde R(x_i^j)\mid x_i^j \text{ is dominated by all points in } \mathcal{R}^{front}\}$, and the target distribution is $\nu_i = \mathcal{R}^{front}(p_i)$. The ground cost is the squared Euclidean distance in reward space $c(y_i^m, x_i^j) = \|\tilde R(y_i^m) - \tilde R(x_i^j)\|_2^2$. The entropy-regularized OT $\gamma^\ast_i = \arg\min_{\gamma\in\Pi(\mu_i, \nu_i)} \sum_{m,j} c(y_i^m, x_i^j)\gamma(y_i^m, x_i^j)$ is solved quickly via the Sinkhorn algorithm. The inner product of $\gamma^\ast$ and $c$ is backpropagated to the T2I model, essentially moving dominated samples toward the "nearest corresponding points" on the frontier. The pipeline uses differentiable reward optimization similar to DRaFT-K.
    - **Design Motivation**: OT preserves the geometry of the reward space (unlike simply picking the maximum), preventing "all samples collapsing toward a single target." The differentiability of Sinkhorn allows the transport cost to be backpropagated, which is essential for engineering implementation.

3.  **Online/Offline Dual Strategy + VLM Decision Agent**:
    - **Function**: Adopt different frontier construction strategies based on reward model strength and mitigate losses when weak rewards collapse.
    - **Mechanism**: The authors measure reward accuracy using Pick-a-Pic and Pick-High datasets (Table 1: CLIP 60.3%, HPS 72.9%, ICT 87.6%, HP 88.5%), classifying the latter two as "strong" and the former as "weak." **Strong rewards** follow an **online strategy**, dynamically expanding the frontier during training to encourage exploration of new Pareto optima. **Weak rewards** follow an **offline strategy**, using a fixed frontier pre-calculated from $M$ samples to prevent noise from polluting the frontier. Simultaneously, a GPT-4o agent uses a "mild collapse reference set" to detect early collapse. If detected, the weak reward is removed, and the model rolls back.
    - **Design Motivation**: Strong rewards are consistent with human preferences; online expansion provides exploration and robustness. Weak rewards are unreliable, and online expansion would only introduce more noise; thus, offline locking and active detection are safer strategies.

### Loss & Training
The training loss is the total OT transport cost $\sum_{m,j}c(y_i^m, x_i^j)\gamma^\ast(y_i^m, x_i^j)$ backpropagated to the T2I model using DRaFT-K style differentiable rewards. A VLM agent triggers collapse checks at each validation step, using in-context examples of "mild collapse" for reference. Beyond traditional win rates, the metrics $\mathrm{JDR}_K = \tfrac{1}{N}\sum_i \mathbb{1}(\mathbf{R}_i\succ\mathbf{R}_{i,b})$ and $\mathrm{JCR}_K = \tfrac{1}{N}\sum_i \mathbb{1}(\mathbf{R}_{i,b}\succ\mathbf{R}_i)$ are introduced.

## Key Experimental Results

### Main Results
Base model: SD3.5-Turbo; 4 rewards: ICT, HP (strong), CLIP, HPS (weak); evaluated on Parti-Prompts.

| Method | ICT Win Rate | HP Win Rate | CLIP Win Rate | HPS Win Rate | JDR₂ ↑ | JDR₄ ↑ | JCR₄ ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| +ICT Single Reward | 56.99 | 36.83 | 47.06 | 48.71 | 20.59 | 7.66 | 10.17 |
| +HP Single Reward | 52.45 | **90.26** | 44.30 | 57.29 | 36.15 | 13.73 | 4.11 |
| Weighted 2:3:2:3 | 50.80 | 56.43 | 46.51 | 86.03 | 28.31 | 13.42 | 2.57 |
| Reward Soup 3:2:1:4 | 50.80 | 53.74 | 43.32 | 85.29 | 26.29 | 10.85 | 3.19 |
| Weighted-Sum (w/o OT) | 52.63 | 56.86 | 46.94 | 82.48 | 29.84 | 13.66 | 3.49 |
| **PG-OT** | 56.43 | 85.23 | 43.63 | 61.70 | **47.98** | **17.10** | **2.39** |

**Human win rate of nearly 80%**—this is a primary selling point. While PG-OT does not achieve the highest score in every single reward (e.g., lower than weighted-sum in CLIP/HPS), it significantly leads in JDR₂/JDR₄ while maintaining the lowest JCR₄. This indicates that its outputs are multi-dimensionally superior more broadly without the collapse found in baselines.

### Ablation Study

| Variant | Key Observation |
| :--- | :--- |
| Global upper bound (weighted-sum) | Individual rewards rise but JDR is low and JCR is high, proving hacking risk. |
| OT only (no Pareto frontier) | OT lacks a clear target; performance is close to weighted-sum. |
| Pareto only (no OT) | Frontier points are discrete, lacks differentiable signals. |
| No strong/weak separation | Online expansion of weak rewards pollutes the target frontier. |
| No VLM agent detection | Unable to stop loss in time after weak reward collapse. |
| Full PG-OT | JDR₂ 47.98%, JCR₄ only 2.39%, simultaneously improving gains and suppressing hacking. |

Table 2 shows trends in CLIP-only optimization: CLIP rises by +7.27% while HPS falls -2.78% and HP falls -4.38%, a typical example of reward conflict and hacking, highlighting the necessity of JDR/JCR detection.

### Key Findings
- Single-reward optimization (e.g., +HP reaching a 90.26% win rate) yields the highest value in one dimension but poor JDR/JCR, showing that traditional "single-reward win rate" metrics can be highly misleading.
- Weighted-sum weight tuning provides limited gains: across 4 ratios, JDR₄ only reached 12.44%–13.66%, significantly lower than PG-OT's 17.10%.
- JCR reveals hidden collapses that mean-based metrics miss: the Separate-Cons configuration has an HPS win rate of 61.21% which seems acceptable, but its JCR₄ is as high as 6.68%, indicating many samples degraded across all dimensions.

## Highlights & Insights
- The observation of "prompt-wise heterogeneous upper bounds" is incisive, transforming the standard "global reward" assumption into an explicitly provable source of hacking through both theory and empirical evidence.
- Using the Pareto frontier as the OT target distribution is a **conceptual relay**—upgrading from single-point maximization to "distribution-to-distribution transport," which offers structural advantages in multi-objective settings.
- JDR/JCR shift multi-reward alignment evaluation from "average scores" to "Pareto comparison," serving as a potential universal diagnostic standard for future multi-reward RLHF work.
- The asymmetric handling of strong/weak rewards (online vs. offline) combined with active VLM agent pruning is a pragmatic engineering solution for the varying quality of reward models in real-world RLHF.

## Limitations & Future Work
- The quality of the offline Pareto frontier depends on the number of pre-generated samples $M$ and the reliability of the reward model itself; if rewards are severely misaligned, the frontier becomes a false target.
- Sinkhorn computation on large batches and the selection of regularization coefficients are sensitive to hyperparameters; the paper does not provide detailed hyperparameter ablations.
- The VLM agent relies on human-annotated collapse cases for its "mild collapse reference set," which might fail for unseen collapse patterns.
- Experiments are limited to 4 rewards and one SD3.5-Turbo backbone; scalability across more rewards and transferability to Diffusion/Autoregressive architectures requires further validation.

## Related Work & Insights
- **vs. Weighted Sum / Reward Soup**: These methods bypass conflicts by "weight tuning"; PG-OT acknowledges conflicts via the Pareto frontier and transports samples using OT, requiring no structural weight tuning.
- **vs. DRaFT / Diffusion-DPO**: Traditional differentiable reward fine-tuning uses global targets to push rewards; PG-OT uses prompt-specific frontiers as targets to explicitly suppress hacking risks.
- **vs. Pareto-MTL / Multi-task Learning**: MTL often uses methods like MGDA to find Pareto directions; PG-OT does not seek directions in weight space but performs transport in sample space using OT, avoiding the instability of MGDA solvers in high-dimensional tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Prompt-wise Pareto frontier + OT" + "JDR/JCR" are highly original and clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive baselines (single reward/weighted/Reward Soup/OT without Pareto) + human evaluation, though limited backbones.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical groundwork and educational motivation analysis.
- Value: ⭐⭐⭐⭐⭐ Strong universal implications for multi-reward RLHF; JDR/JCR could be widely adopted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](../../NeurIPS2025/image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[CVPR 2026\] COT-FM: Cluster-wise Optimal Transport Flow Matching](../../CVPR2026/image_generation/cot-fm_cluster-wise_optimal_transport_flow_matching.md)
- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](../../ICLR2026/image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICML 2026\] Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation](gradient_preconditioning_for_efficient_and_reliable_reward-guided_generation.md)

</div>

<!-- RELATED:END -->
