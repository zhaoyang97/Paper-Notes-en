---
title: >-
  [Paper Note] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL
description: >-
  [ICML 2026][Model Compression][Offline GCRL] QHyer replaces trajectory-dependent RTG in Decision Transformers with state-dependent Q-values estimated via Normalizing Flows…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Offline GCRL"
  - "Decision Transformer"
  - "Normalizing Flows"
  - "Mamba"
  - "Trajectory Stitching"
date: 2026-05-08
content_hash: 31073f0d916e7aa7
---

# QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL

**Conference**: ICML 2026  
**arXiv**: [2605.01862](https://arxiv.org/abs/2605.01862)  
**Code**: Not released  
**Area**: Reinforcement Learning / Sequence Modeling / Offline Goal-conditioned RL  
**Keywords**: Offline GCRL, Decision Transformer, Normalizing Flows, Mamba, Trajectory Stitching  

## TL;DR
QHyer replaces trajectory-dependent RTG in Decision Transformers with state-dependent Q-values estimated via Normalizing Flows, while employing a gated hybrid Attention-Mamba backbone to achieve content-adaptive history compression, achieving new SOTA results across non-Markovian and Markovian offline goal-conditioned RL datasets in OGBench/D4RL.

## Background & Motivation

**Background**: Offline Goal-conditioned Reinforcement Learning (Offline GCRL) learns "goal-reaching" policies from static datasets. Current mainstream approaches follow two paths: value-based methods utilizing Bellman backups (e.g., IQL, HIQL) and Decision Transformer (DT) variants that treat decision-making as sequence modeling. The latter naturally handles historical dependencies and is considered more suitable for real-world datasets containing non-Markovian behavioral policies (e.g., OGBench play).

**Limitations of Prior Work**: Directly applying DT to Offline GCRL faces two major hurdles. First, DT uses Return-to-Go (RTG) as a conditioning signal; however, under sparse rewards, RTG collapses into a binary signal indicating whether a trajectory succeeded. Since the same state may have a return of 1 in a successful trajectory and 0 in a failed one, it is **impossible to compare state quality across trajectories**, causing the failure of "stitching" locally useful segments from failed demonstrations. Second, pure attention is insensitive to temporal structures. While LSDT/DMixer use fixed-window causal convolutions for "local branches," play data requires long memory while noisy data requires short memory; a fixed receptive field either wastes capacity or truncates critical dependencies.

**Key Challenge**: These two limitations are **coupled**. Replacing RTG with Q-values while keeping RTG-style fixed windows still suffers from convolutional issues in non-Markovian play; replacing the backbone while keeping RTG still fails at the stitching bottleneck under sparse rewards. Both must be solved simultaneously—requiring both a "state-dependent value signal" and "content-adaptive effective memory."

**Goal**: (i) Identify a conditioning signal for DT that can distinguish state quality under sparse rewards; (ii) design a temporal module for the backbone that can dynamically adjust memory length per token.

**Key Insight**: The goal-conditioned Q-function $Q^\beta(s,a,g)=p^\beta_+(g\mid s,a)$ represents the "probability of reaching goal $g$ from $(s,a)$," which is **trajectory-independent**—exactly the "trajectory-agnostic value metric" needed for stitching. Simultaneously, Mamba's selective SSM makes the discretization step $\Delta_t$ an input-dependent function, allowing the effective memory to drift per token without changing the structure. These two observations address the aforementioned limitations.

**Core Idea**: Utilize **Normalizing Flows to estimate MC Q-values as conditioning tokens to replace RTG**, and replace the pure attention backbone with a **Hybrid Attention-Mamba** structure using gated fusion, truly adapting sequence modeling for Offline GCRL.

## Method

### Overall Architecture
QHyer represents each timestep as a $(Q_t, [s_t;g], a_t)$ triplet: $Q_t=\log p_\theta(g\mid s_t,a_t)$ is the log-probability of "reaching the goal" provided by NFs, and $[s_t;g]$ is a concatenated state-goal token (ensuring the goal signal is visible at each step without increasing sequence length from $3T$ to $4T$). This sequence is fed into $L$ Hybrid Attention-Mamba blocks. Each block contains two parallel branches (Attention for global goal planning and Mamba for temporal compression), with outputs fused via a scalar gate $\alpha=\sigma(\mathbf{w}^\top x + b)$. Training involves joint end-to-end optimization of NFs likelihood, Q-expectile regression, and behavior cloning. Inference follows a two-stage autoregressive process: first predicting the maximum Q, then generating actions conditioned on that maximum Q.

### Key Designs

1.  **NFs-based Q-value replacing RTG**:
    - **Function**: Provides DT with trajectory-agnostic state-action-goal value signals, enabling the model to identify "high-Q segments" in failed demonstrations for stitching.
    - **Mechanism**: Coupling-layer NFs model the conditional density $p_\theta(g\mid s,a)$. The exact log-likelihood is derived via an invertible mapping $f_\theta(\cdot;z)$ and the change-of-variables formula: $Q^\beta_\theta(s,a,g)=\log p_0(f_\theta(g;z))+\log\bigl|\det\partial f_\theta(g;z)/\partial g\bigr|$. Expectile regression $L^2_\tau(u)=|\tau-\mathds{1}(u<0)|\cdot u^2$ (with $\tau\in(0.5,1)$) is then used to learn a transformer-specific $\hat Q_\phi(s,g)$ that converges toward the **in-distribution maximum Q** (Theorem 3.1 shows bias $\epsilon_\tau$ decreases as $\tau$ increases).
    - **Design Motivation**: The authors argue why CVAE (only provides ELBO lower bound), Contrastive RL (density ratio has goal-related shifts), and Diffusion (likelihood requires ODE+Hutchinson estimation introducing variance) are unsuitable. Unlike these, NFs' triangular Jacobian makes log-density **exact and inexpensive**, a property required for cross-goal conditioning in transformers. Empirical tests show NFs have the lowest estimation error (Appendix G.4). Under sparse rewards, RTG coverage is only 25%, while NFs Q-conditioning reaches 92%.

2.  **Hybrid Attention-Mamba Backbone**:
    - **Function**: Uses an attention branch for global goal-oriented reasoning and a Mamba branch for content-adaptive history compression, fused via a learnable gate.
    - **Mechanism**: The Mamba branch extracts local features $x'_t$ via causal convolution, followed by selective SSM: $h_t=\bar A h_{t-1}+\bar B x'_t,\ y_t=Ch_t$, where $\bar A_t=\exp(\Delta_t\cdot A)$ and $\Delta_t=\mathrm{softplus}(\mathrm{Linear}_\Delta(x'_t))$. When $\Delta_t$ is small, $\bar A_t\approx 1$, preserving long history (suitable for play); when $\Delta_t$ is large, $\bar A_t\approx 0$, focusing on local context (suitable for noisy data). The gate $\alpha=\sigma(\mathbf w^\top x+b)$ dynamically allocates capacity between branches.
    - **Design Motivation**: Using convolution for local branches (as in LSDT/DMixer) is limited by fixed kernels; the influence on $j<k$ is a fixed weight $w_j$ with a hard truncation. Mamba provides "input-dependent smooth forgetting," automatically adjusting effective memory across datasets without manual hyperparameter tuning for receptive fields.

3.  **Concatenated State-Goal Tokenization + End-to-End Triple Loss**:
    - **Function**: Embeds goal information into each timestep token while maintaining a sequence length of $3T$, avoiding the quadratic overhead of additional tokens in attention.
    - **Mechanism**: Timestep tokens follow the $(Q_t, [s_t;g], a_t)$ sequence rather than $(Q_t, s_t, g, a_t)$. The training loss is $\mathcal L_{\text{QHyer}}=\lambda_{\text{critic}}\mathcal L_{\text{NFs}}+\lambda_{\text{BC}}\mathcal L_{\text{BC}}+\lambda_Q \mathcal L_Q$, corresponding to NFs maximum likelihood, Q-conditioned behavior cloning, and transformer-side Q-expectile regression.
    - **Design Motivation**: Concatenation instead of separation maintains goal visibility while suppressing computational cost, serving as a key engineering trick to integrate NFs Q-signals into the DT pipeline.

### Loss & Training
NFs are trained with maximum likelihood using hindsight relabeling and $-\log p_\theta(g\mid s_t,a_t)$. Transformer-side BC loss is $\mathcal L_{\text{BC}}=-\mathbb E[\log\pi_\theta(a_t\mid Q_t,[s_t;g])]$. Expectile $\tau=0.9$ is used for low-coverage play data, while $\tau=0.95$ is used for high-coverage noisy data. Inference is two-stage: first generating $\hat Q(s_t,g)$, then generating $a_t$ conditioned on it.

## Key Experimental Results

### Main Results
OGBench manipulation (5 test goals, average success rate %) and D4RL Maze (normalized score).

| Dataset | Task | Second Best | Ours (QHyer) | Gain |
|---|---|---|---|---|
| OGBench cube-play | single | GCIQL 68 | **84** | +16 |
| OGBench cube-play | double | GCIQL 40 | **56** | +16 |
| OGBench cube-noisy | double | GCIQL 23 | **30** | +7 |
| OGBench puzzle-play | 4x5 | GCIQL 14 | **31** | +17 |
| D4RL AntMaze-v2 | large-play | IQL 39.6 | **44.2** | +4.6 |
| D4RL AntMaze-v2 | medium-diverse | LSDT 75.8 | **94.0** | +18.2 |
| D4RL Maze2d | medium | QT 172.0 | **173.0** | +1.0 |

Overall Scores: OGBench cube-play increased from 24 to 152 (vs. HIQL baseline). AntMaze total score increased from 303.6 to 483.4, and Maze2d from 136.5 to 291.5. In large mazes where RTG-series (DT/EDT/DC) scores were near zero, QHyer achieved a breakthrough.

### Ablation Study

| Configuration | cube-single-play | cube-single-noisy | Conclusion |
|---|---|---|---|
| RTG + Attention (≈DT) | Low | Low | RTG failure |
| NFs Q + Attention only | 74 | 60 | Lacks temporal adaptation |
| NFs Q + Mamba only | 80 | 91 | Lacks global reasoning |
| NFs Q + Hybrid (QHyer) | **84** | **95** | Complementary gating |
| Hybrid + No Q | -- | -- | Degenerates to BC |
| Hybrid + CVAE Q | < CRL | < CRL | ELBO distortion |
| Hybrid + CRL Q | < NFs | < NFs | Negative sampling bias |

Expectile $\tau$ shows monotonic improvement from 0.5 up to 0.9, but performance degrades beyond 0.95 due to insufficient coverage.

### Key Findings
- **Innovations are individually necessary and Jointly Optimal**: Independent ablations show that fixed NFs with different backbones, fixed RTG with different backbones, or fixed backbones with different Q-estimators all underperform compared to the joint QHyer architecture.
- **Mamba's $\Delta_t$ drifts according to data characteristics**: On play data, mean $\Delta_t=0.38$, $\bar A_t=0.92$, with an effective memory of ~12 steps and the gate allocating 0.57 capacity to attention. On noisy data, $\Delta_t=1.05$, $\bar A_t=0.61$, with memory of ~3 steps and the gate allocating 0.58 to Mamba.
- **NFs > CRL > CVAE > No-Q**: Precise normalized log-density is the key bottleneck for stitching in sequence modeling.

## Highlights & Insights
- **Clean Argumentation for Coupled Limitations**: The authors explicitly identify failure modes when solving only one side (either non-Markovian issues with convolutions or trajectory-dependency in RTG). This provides a stronger motivation for the dual changes than a simple additive narrative.
- **Structural Argument for NFs Selection**: By moving beyond mere numbers to the normalization requirements of "transformers reading Q-tokens across multiple goals," the authors provide a design principle applicable to broader scenarios.
- **Dual-Level Adaptability**: Coarse-grained adaptation via gating and fine-grained adaptation via $\Delta_t$ adjusting memory length. This hierarchy is a powerful paradigm for datasets with heterogeneous temporal structures, transferable to multi-task robotics or dialogue history compression.

## Limitations & Future Work
- Limited performance on visual-noisy data: Pixel-level NFs density estimation becomes the primary error source, while Markovian behavior offsets the advantages of non-Markovian modeling.
- Theoretical analysis assumes deterministic transitions (inherited from R2CSL); extension to stochastic environments remains an open problem.
- Training cost is higher than pure DT: Integration of NFs critic, Mamba SSM, and expectile regression adds complexity; wall-clock comparisons were not detailed.
- Expectile $\tau$ is strongly coupled with coverage $\tilde c$, requiring manual tuning across datasets.

## Related Work & Insights
- **vs DT/EDT/DC/DMamba**: These use RTG, which degrades to a binary signal in sparse rewards; QHyer's NFs Q enables a qualitative leap in stitching.
- **vs QDT/CGDT/QT/Reinformer/VDT**: These retain RTG and use Q as an auxiliary loss or regularizer; QHyer replaces RTG with Q-tokens entirely.
- **vs LSDT/DMixer**: These supplement local context with fixed convolutions; QHyer utilizes Mamba's selective SSM for content-adaptive memory.
- **vs HIQL/SAW/OTA**: Hierarchical methods assume Markovian transitions between subgoals; QHyer's sequence modeling naturally handles non-Markovianity in play data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to combine NFs Q and Hybrid Attention-Mamba for Offline GCRL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual benchmarks, comprehensive ablations of Q-estimators and backbones, sensitivity analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from "limitation" to "root cause" to "design choice."
- Value: ⭐⭐⭐⭐ Provides a viable path for "sequence modeling + exact density Q" in Offline GCRL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation](../../ACL2026/model_compression/no-worse_context-aware_decoding_preventing_neutral_regression_in_context-conditi.md)
- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](flattengpt_depth_compression_for_transformer_with_layer_flattening.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)
- [\[NeurIPS 2025\] Spark Transformer: Reactivating Sparsity in FFN and Attention](../../NeurIPS2025/model_compression/spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](../../CVPR2025/model_compression/bhvit_binarized_hybrid_vision_transformer.md)
- [\[ACL 2026\] No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation](../../ACL2026/model_compression/no-worse_context-aware_decoding_preventing_neutral_regression_in_context-conditi.md)
- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](../../AAAI2026/model_compression/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)
- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](flattengpt_depth_compression_for_transformer_with_layer_flattening.md)

</div>

<!-- RELATED:END -->
