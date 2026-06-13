---
title: >-
  [Paper Note] Mixture of Horizons in Action Chunking
description: >-
  [ICML 2026][Robotics][VLA] Addressing the "long-horizon planning vs. short-horizon precision" trade-off caused by action chunking length (horizon) selection in VLA models…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLA"
  - "Action Chunking"
  - "Multi-scale horizon"
  - "Gated fusion"
  - "Dynamic inference"
date: 2026-05-08
content_hash: 6fd1a8be9bed9d78
---

# Mixture of Horizons in Action Chunking

**Conference**: ICML 2026  
**arXiv**: [2511.19433](https://arxiv.org/abs/2511.19433)  
**Code**: To be confirmed  
**Area**: Robotics / VLA / Action Chunking  
**Keywords**: VLA, Action Chunking, Multi-scale horizon, Gated fusion, Dynamic inference

## TL;DR
Addressing the "long-horizon planning vs. short-horizon precision" trade-off caused by action chunking length (horizon) selection in VLA models, this paper proposes Mixture of Horizons (MoH). By decomposing a single action chunk into multiple sub-chunks of varying lengths, predicting them in parallel with a shared action transformer, and fusing them via a 2k-parameter linear gating mechanism—complemented by a load-balancing loss and dynamic inference based on "cross-horizon consensus"—$\pi_{0.5}$ achieves a 99% average success rate on LIBERO for the first time while increasing throughput to 2.5× the baseline.

## Background & Motivation

**Background**: Modern Vision-Language-Action (VLA) models (such as $\pi_0$, $\pi_{0.5}$, OpenVLA-OFT, StarVLA) almost exclusively adopt the action chunking strategy proposed by Zhao et al. This involves predicting future actions for $H$ steps at once, $A_t=(a_t,\dots,a_{t+H-1})$, and processing these action tokens with a lightweight full-attention action transformer. The theoretical basis lies in smooth execution, reduced policy invocation frequency, and the utilization of temporal structural information. The "VLM backbone + chunked action head" has become the de facto standard configuration.

**Limitations of Prior Work**: The authors evaluated $\pi_0$ on LIBERO by setting the horizon to 10, 20, and 30 across four task sets: Spatial, Object, Goal, and Long. They discovered a simple yet often overlooked fact—success rates are extremely sensitive to $H$, and the optimal $H$ varies across tasks. Long tasks favor longer horizons (for planning), while Spatial/Object tasks favor shorter horizons (for precision). Any fixed $H$ is destined to underperform on certain task categories.

**Key Challenge**: Long horizon $\rightarrow$ Distant planning is possible, but per-step accuracy is "diluted"; Short horizon $\rightarrow$ Precise control, but lacks foresight. This is a structural trade-off inherent to chunk-based representations that cannot be solved simply by hyperparameter tuning, especially since horizons cannot be switched online during deployment.

**Goal**: (i) Systematically characterize the impact of horizon on VLA; (ii) Gain the benefits of both long and short horizons within a single model; (iii) Enable the chunk length to scale adaptively during inference based on confidence.

**Key Insight**: Since different horizons have different strengths, one should not choose—incorporate multiple horizons into training simultaneously and let the model learn when to be "long" or "short." The key is to make this practically zero-cost: the computational bottleneck of VLA is the VLM backbone, while the action transformer itself has only ~300M parameters. Parallel forward passes for multiple horizons using tensor parallelism add almost no wall-clock time.

**Core Idea**: Rearrange action chunks into multiple sub-segments based on a set of candidate lengths $\mathcal{H}=\{h_1,\dots,h_N\}$, predict them in parallel using the same action transformer, and perform weighted fusion via a 2k-parameter linear gate across steps and horizons. As a byproduct, prediction consistency across horizons naturally serves as an execution confidence signal to drive dynamic truncation.

## Method

### Overall Architecture
Given multi-view images $V_t$, history $h_{<t}$, instructions $T$, and proprioception $s_t$ at time $t$, the VLM backbone encodes them into a context. MoH then decomposes the target action chunk $A_t\in\mathbb{R}^{H\times d_a}$ into $N$ truncated sub-chunks of increasing length $A_t^{(h)}=(a_{t,1},\dots,a_{t,h})$. Each sub-chunk is padded to $H$ and assigned a horizon-specific attention mask (masking positions $k>h$). All horizons are then **processed in parallel during a single forward pass by a shared action transformer**, yielding horizon-wise predictions $\hat A_t^{(h)}$. Finally, a linear gating head outputs logits $g_{t,k,h}$, which are processed via a masked softmax to obtain fusion weights $\alpha_{t,k,h}$, resulting in the final prediction $\hat a_{t,k}=\sum_{h:k\le h}\alpha_{t,k,h}\hat a_{t,k}^{(h)}$. This design is compatible with flow-matching ($\pi_0$/$\pi_{0.5}$/StarVLA) and one-step regression ($\pi_{\text{reg}}$) and is non-intrusive to the backbone.

### Key Designs

1. **Multi-horizon Action Chunk Rearrangement + Shared Transformer Parallel Processing**:
    - **Function**: Transitions from "selecting one horizon" to "training with multiple horizons simultaneously," allowing a single policy to learn both short-term precision and long-term planning.
    - **Mechanism**: Fix the maximum horizon $H$ and define a candidate set $\mathcal{H}=\{h_1,\dots,h_N=H\}$. For each $h\in\mathcal{H}$, truncate $A_t^{(h)}\in\mathbb{R}^{h\times d_a}$, pad to $H$, and apply a horizon-specific mask for out-of-bounds positions. All horizons share the same action transformer weights and VLM context, computed simultaneously via batching and parallel attention. Two loss terms are calculated: a mixed prediction loss $L_{\text{mix}}$ and independent prediction losses for each horizon $L_{\text{ind}}=\sum_h L^{(h)}$. The former ensures fusion quality, while the latter ensures each horizon branch is independently viable.
    - **Design Motivation**: (a) The VLM forward pass runs once, and since the action transformer is lightweight, the overhead of parallel multi-horizon forward passes is negligible; (b) Shared weights force the network to learn both short and long capabilities rather than simply ensembling independent models; (c) Padding and masking align sequence lengths across horizons, preventing dynamic shapes from slowing down the GPU.

2. **2k-Parameter Linear Gating + Load Balancing Loss**:
    - **Function**: Adaptively fuses predictions at each step $k$ based on which horizons are most credible at that point, while preventing the gate from collapsing into a few horizons.
    - **Mechanism**: A **linear layer** (~2k parameters) is added atop the shared transformer to output logits for each (step, horizon). For each $k$, only valid horizons where $h \ge k$ are kept. A softmax normalization provides $\alpha_{t,k,h}=\exp(g_{t,k,h})/\sum_{h':k\le h'}\exp(g_{t,k,h'})$. To prevent the gate from favoring only specific horizons, a MoE-style load-balancing loss is introduced: the time axis is partitioned into intervals $S_i$ by horizon boundaries, and the average usage $\bar\alpha_h^{(i)}$ of each horizon is calculated in each interval. The loss $L_{\text{bal}}=\frac{1}{|\mathcal{I}|}\sum_i \mathrm{CV}^2(\{\bar\alpha_h^{(i)}\}_h)$ minimizes the squared coefficient of variation to enforce fair allocation. The total objective is $L=L_{\text{mix}}+\lambda_{\text{ind}}L_{\text{ind}}+\lambda_{\text{bal}}L_{\text{bal}}$, with defaults $\lambda_{\text{ind}}=1$ and $\lambda_{\text{bal}}=10^{-3}$.
    - **Design Motivation**: Following Occam's Razor—almost all information is already encoded in the shared transformer's hidden states; the gate only needs to make a lightweight weighted decision. Complex structures would likely result in overfitting. Ablations show that removing $L_{\text{bal}}$ still outperforms the baseline (98.5%), but adding it improves Long tasks by an additional 1.6 points, proving the balancing regularization ensures long horizons contribute effectively.

3. **Dynamic Inference via Cross-Horizon Consensus**:
    - **Function**: Replaces fixed execution of the first $K$ steps with a "confidence-based prefix length," allowing longer execution for simple motions and frequent replanning near decision points, significantly increasing speed.
    - **Mechanism**: At each step $k$, each horizon-wise prediction $\hat a_k^{(h)}$ is treated as a "voter" for the fused result $\hat a$. A weighted $\ell_1$ divergence is defined as $\bar d_k=\sum_{h\in\mathcal{H}_k}\alpha_{k,h}\cdot\|\hat a-\hat a_k^{(h)}\|$ (where $\mathcal{H}_k=\{h\ge k\}$). An adaptive threshold is set as $\textit{thres}=\mathrm{Mean}(\{\bar d_k\}_{k=1}^n)\cdot r$, based on the mean divergence of the first $n$ steps. Starting from $k=n+1$, the sequence is checked incrementally: if the number of valid horizons drops below $m$ or $\bar d_k > \textit{thres}$, execution breaks. The execution prefix $K_{\text{exec}}$ is set there. This naturally results in longer prefixes for stable motions and shorter ones at critical decision points.
    - **Design Motivation**: Previous chunk-based VLAs used hardcoded prefixes (e.g., 5 in LIBERO, 20 in RoboTwin), which are both wasteful and brittle. MoH's multiple horizons naturally provide a "divergence" signal, driving adaptive truncation without additional training. Experiments show $\pi_{0.5}$+MoH surpasses the baseline even at 2.5× throughput.

### Loss & Training
- Total Objective: $L=L_{\text{mix}}+\lambda_{\text{ind}}L_{\text{ind}}+\lambda_{\text{bal}}L_{\text{bal}}$, with $\lambda_{\text{ind}}=1$ and $\lambda_{\text{bal}}=10^{-3}$.
- For flow-matching policies, $L_{\text{mix}}$ and $L^{(h)}$ are velocity-matching losses $\|v_\theta(A_t^{(\tau)},\tau,\cdot)-(A_t-\epsilon)\|_2^2$; for one-step regression, $\ell_1$ is used; for categorical models, cross-entropy is used.
- Default $\mathcal{H}=\{3,6,\dots,30\}$ (stride $d=3$, 10 horizons total), trained for 30k iterations on 4 A100s with batch size 32, taking <10 hours.

## Key Experimental Results

### Main Results

LIBERO (4 task sets, 500 trials/set, uniform execution of first 5 steps):

| Baseline | Spatial | Object | Goal | Long | Average |
|------|---------|--------|------|------|------|
| $\pi_{\text{reg}}$ (3B, 30k) | 97.8 | 98.2 | 94.6 | 90.2 | 95.2 |
| $\pi_{\text{reg}}$ + MoH | 99.0 (**↑1.2**) | 98.8 (**↑0.6**) | 96.4 (**↑1.8**) | 91.4 (**↑1.2**) | **96.4 (↑1.2)** |
| $\pi_0$ (3B, 30k) | 97.4 | 98.2 | 95.4 | 84.2 | 93.8 |
| $\pi_0$ + MoH | 97.6 (**↑0.2**) | 98.8 (**↑0.6**) | 96.4 (**↑1.0**) | 87.4 (**↑3.2**) | **95.1 (↑1.3)** |
| StarVLA (3B, 30k) | 98.0 | 98.2 | 95.8 | 91.4 | 95.9 |
| StarVLA + MoH | 98.4 | 99.6 | 97.6 | 92.4 | **97.0 (↑1.1)** |
| $\pi_{0.5}$ (3B, 30k) | 98.8 | 99.0 | 97.6 | 95.4 | 97.7 |
| $\pi_{0.5}$ + MoH | 98.8 | **100** | 98.8 | 98.4 (**↑3.0**) | **99.0 (↑1.3)** |

$\pi_{0.5}$+MoH sets a new LIBERO SOTA with a 99% average success rate in just 30k iterations (previous best was Spatial Forcing 7B at 98.5%), despite being only 3B parameters. The +3.0 gain in Long tasks is the largest single-point improvement, confirming that MoH effectively addresses the "long planning" bottleneck. On RoboCasa, GR00T+MoH improved by 3.4 points on average (28.0$\rightarrow$31.4), proving effectiveness in non-saturated household scenes. Robust performance was also observed in RoboTwin 2.0.

### Ablation Study

Fixed $H_{\max}=30$, all variants run on $\pi_{0.5}$:

| Config | Spatial | Object | Goal | Long | Average | Note |
|------|---------|--------|------|------|------|------|
| $\pi_{0.5}$ baseline ($\mathcal{H}=\{30\}$) | 98.8 | 99.0 | 97.6 | 95.4 | 97.7 | Single horizon |
| + MoH, $d=10$ (3 horizons) | 98.8 | 99.8 | 97.6 | 96.8 | 98.3 | 0.6 gain with only 3 horizons |
| + MoH, $d=3$ (10 horizons) | 98.8 | 100 | 98.8 | 98.4 | **99.0** | Default config, best |
| + MoH, $d=1$ (30 horizons) | 99.0 | 99.4 | 98.4 | 96.2 | 98.3 | Performance drops if too dense |
| + MoH 10 identical horizons ($H=30$) | 98.6 | 99.4 | 98.6 | 94.8 | 97.9 | Rules out "ensemble effect" |
| + Temporal dim loss reweighting only | 99.2 | 99.6 | 99.2 | 94.4 | 98.1 | Long tasks drop; trade-off unresolved |
| + MoH, mean fusion (no gating) | 98.8 | 99.2 | 98.6 | 96.8 | 98.4 | Simplest MoH is effective |
| + MoH, no $L_{\text{bal}}$ | 98.2 | 100 | 99.0 | 96.8 | 98.5 | Balance loss helps Long tasks |

### Key Findings
- **Horizon diversity is critical, not "multi-branch ensembling"**: 10 identical branches with $H=30$ only raised the average from 97.7% to 97.9%, whereas 10 different horizons reached 99.0%, with the main difference appearing in Long tasks.
- **3 horizons are sufficient, 10 are optimal**: The largest gain occurs moving from 1 to 3 horizons, with performance peaking at 10. 30 horizons led to a decline, suggesting an optimal "density" for the horizon set to avoid signal interference.
- **Loss reweighting cannot replace MoH**: Simply reweighting loss per step improves Spatial/Object/Goal but worsens Long tasks (95.4$\rightarrow$94.4), confirming MoH's gains do not stem from implicit reweighting.
- **Dynamic inference is a free lunch**: $\pi_{0.5}$+MoH with dynamic truncation ($r=1.1$) achieved 2.5× throughput and still outperformed the fixed-prefix baseline.

## Highlights & Insights
- **Converting a "hyperparameter selection" problem into an "in-model decision" problem**: Horizon length has always been treated as a brittle hyperparameter. MoH integrates multiple horizons into training and allows the gate to learn selections, an elegant de-hyperparameterization that could apply to diffusion steps, history length, or temporal stride.
- **MoE applied to the horizon dimension**: This work demonstrates that the MoE template (gating + load balancing) is effective when applied to different axes. The use of $\mathrm{CV}^2$ for load balancing instead of KL/entropy is a noteworthy detail for stability across varying horizon counts.
- **Cross-prediction consistency = Endogenous confidence**: Instead of fixed prefixes, MoH uses multi-horizon prediction divergence as a confidence signal for self-truncation. This **requires zero extra training and zero extra parameters**, making it a valuable byproduct.
- **Almost zero overhead**: With 2k extra parameters and a single shared forward pass, the method is exceptionally friendly to VLA architectures dominated by VLM backbones, suggesting it should be a default component for chunk-based VLAs.

## Limitations & Future Work
- **Only effective for full-attention action transformers**: Pure causal autoregressive models (like some token-level VLAs) cannot obtain parallel predictions for different horizons in one forward pass without architectural changes.
- **Horizon sets still require manual selection**: While the ablation suggests "stride=3, $H_{\max}=30$" is optimal, the best values might vary across platforms or tasks. Ideally, $\mathcal{H}$ should be learnable.
- **Evaluation focused on tabletop manipulation**: LIBERO/RoboTwin/RoboCasa involve short-to-medium horizons. Whether MoH maintains its advantage in truly long-term tasks (e.g., multi-minute room tidying) remains unverified.
- **Limited gating interpretability**: While usage statistics are provided, identifying specific scenarios favoring specific horizons or enabling explicit horizon control via instructions remains for future work.

## Related Work & Insights
- **vs. ACT (Zhao 2023)**: ACT introduced chunk-based prediction with a fixed $H$; this work identifies fixed $H$ as a bottleneck and provides a multi-horizon solution.
- **vs. CogACT (Li 2024)**: CogACT uses similarity-weighted fusion for overlapping frames **within the same horizon**; MoH fuses predictions from **different horizons**, making them orthogonal and complementary.
- **vs. $\pi$ series / OpenVLA-OFT**: These focus on backbones (flow-matching, PaliGemma); MoH is a plug-and-play module for any chunk-based architecture.
- **vs. Switch Transformer / MoE**: The conceptual lineage is clear; the difference is replacing experts with horizons, shifting the goal from "capacity expansion" to "eliminating hyperparameter trade-offs."
- **vs. Dynamic action chunking / replanning**: Previous methods relied on value functions or RL signals; MoH achieves confidence measurement via internal consistency for free.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](../../NeurIPS2025/robotics/reinforcement_learning_with_action_chunking.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)
- [\[ICML 2026\] From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](from_imagined_futures_to_executable_actions_mixture_of_latent_actions_for_robot_.md)

</div>

<!-- RELATED:END -->
