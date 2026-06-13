---
title: >-
  [Paper Note] Free Energy Mixer
description: >-
  [ICLR 2026][Time Series][Attention mechanism] This paper proposes Free Energy Mixer (FEM), which reframes attention value retrieval as a free energy (log-sum-exp) optimization problem…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Attention mechanism"
  - "free energy"
  - "channel-wise selection"
  - "log-sum-exp"
  - "plug-and-play"
date: 2026-05-08
content_hash: 8eaa99acd3aaef59
---

# Free Energy Mixer

**Conference**: ICLR 2026
**arXiv**: [2602.07160](https://arxiv.org/abs/2602.07160)  
**Code**: Available (linked in paper)  
**Area**: Time Series / LLM Efficiency
**Keywords**: Attention mechanism, free energy, channel-wise selection, log-sum-exp, plug-and-play

## TL;DR
This paper proposes Free Energy Mixer (FEM), which reframes attention value retrieval as a free energy (log-sum-exp) optimization problem, enabling value-aware posterior selection at the per-channel level. FEM addresses the inherent bottleneck of standard attention—lossless storage but lossy reading—and serves as a plug-and-play replacement for softmax/linear attention/RNN/SSM, yielding consistent improvements across NLP, vision, and time series tasks.

## Background & Motivation

**Background**: The attention mechanism in Transformers stores all historical information losslessly via KV-cache, then reads values through a convex combination of probability weights—a pattern of "lossless storage but lossy processing." Existing improvements include sparse attention, low-rank projection, kernelized attention, linear RNN/SSM, and related approaches.

**Limitations of Prior Work**: The reading operation in standard attention applies identical weights to all value dimensions ($\mathbf{o}_t = \sum_i \alpha_{t,i} \mathbf{v}_i$), forcing the output to lie within the convex hull of the value vectors. This means **per-channel index selection is infeasible**—even when different channels need to retrieve information from different historical positions, a single attention head cannot accommodate this.

**Key Challenge**: $H$ attention heads provide at most $t^H$ head-level argmax patterns, far fewer than the $t^D$ patterns required for fully independent per-channel selection (when $H \ll D$). Increasing the number of heads narrows each head's width, and stacking more layers cannot recover the channel-level index information lost after the first convex mixing operation.

**Goal**: How can attention be endowed with per-channel, value-aware selection capability without altering asymptotic complexity?

**Key Insight**: Value retrieval is recast as a selection problem under information constraints—given a prior distribution $p_t$ (derived from Q/K), the goal is to find, for each channel $j$, a posterior distribution $q$ that maximizes expected utility while bounding the KL divergence from the prior. The solution to this variational problem takes exactly the log-sum-exp form.

**Core Idea**: Replace the linear readout in attention with free energy (log-sum-exp), introducing independent value-aware posterior selection for each value channel. This enables a smooth transition from average readout to per-channel hard selection without increasing asymptotic complexity.

## Method

### Overall Architecture
FEM is a plug-and-play module that directly replaces the attention layer within a Transformer block. It takes any selection prior (softmax attention, GLA, Mamba, AFT) as input and enhances the reading capability through four components: (C) low-rank convolutional local conditioning, (L) LSE mixing, (T) linearized temperature learning, and (G) external gating. The output substitutes the original attention output; all other components (MLP, embeddings, etc.) remain unchanged.

### Key Designs

1. **Free Energy Read**:

    - Function: Replaces the standard expected readout $\mu_{t,j} = \sum_i p_t(i) v_{i,j}$ with a per-channel free energy readout.
    - Mechanism: For each channel $j$, a constrained optimization problem is defined as $\max_{q} \mathbb{E}_{i \sim q}[v_{i,j}] \text{ s.t. } \text{KL}(q \| p_t) \leq B_{t,j}$. Introducing a Lagrange multiplier $\beta_{t,j}$ yields the free energy: $\mathcal{F}_{t,j}(\beta) = \frac{1}{\beta} \log \sum_i p_t(i) \exp(\beta v_{i,j})$, with corresponding posterior $q_{t,\beta}^{(j)}(i) \propto p_t(i) \exp(\beta v_{i,j})$. As $\beta \to 0$, this degenerates to the standard mean; as $\beta \to \infty$, it converges to argmax hard selection.
    - Design Motivation: The free energy naturally satisfies $\mathcal{F}_{t,j}(\beta) = \mathbb{E}_{p_t}[v_{i,j}] + \frac{1}{\beta} \text{KL}(p_t \| q^{(\beta)})$, guaranteeing the output is never below the expected value, with the degree of improvement quantified by the KL divergence. Different channels can maintain different posteriors, enabling independent retrieval.

2. **Linearized Temperature Learning (LTL)**:

    - Function: Enables dynamic per-channel temperature without sacrificing single-pass computational efficiency.
    - Mechanism: A maximum inverse temperature $\beta_{\max}$ is fixed; a baseline $\mu_t$ and a high-temperature branch $F_t^{\max}$ are computed, then interpolated via a learnable gate $\lambda_t \in [0,1]^D$: $\tilde{F}_t(\lambda_t) = (1-\lambda_t) \odot \mu_t + \lambda_t \odot F_t^{\max}$. Both terms can be computed in a single pass (by mixing $[v_{i,j}, e^{\beta_{\max} v_{i,j}}]$), preserving the same asymptotic complexity as the prior.
    - Design Motivation: Learning independent $\beta$ values for each (step, channel) pair would break parallelism. By proving via the intermediate value theorem that the map $\lambda \mapsto \beta^*$ is strictly monotone, optimizing $\lambda$ is equivalent to optimizing the implicit temperature.

3. **Dual-Level Gating**:

    - Function: Inner gating (temperature) controls the degree of transition from averaging to selection; outer gating scales the final output magnitude.
    - Mechanism: The final readout is $\mathbf{o}_t = \mathbf{g}_t \odot [(1-\lambda_t) \odot \mu_t + \lambda_t \odot F_t^{\max}]$, where $\mathbf{g}_t$ is parameterized via softplus + RMSNorm.
    - Design Motivation: The outer gate corresponds to applying an exponential scaling $[\sum_i p_t(i) \exp(\beta^* v_{i,j})]^{g_{t,j}}$ on the free energy, expanding the representational space. Setting $\lambda = 0, g = 1$ recovers standard attention.

4. **Low-Rank Convolutional Local Conditioning (Module C)**:

    - Function: Extracts locally position-sensitive features via a lightweight adaptive low-rank convolution to modulate the selection prior and FEM gates.
    - Mechanism: A simple temporal decay kernel with $O(1)$ streaming updates; total cost is $O(TH_c)$ where $H_c = d/16 \ll D$.
    - Design Motivation: Inspired by the local convolution design in Mamba/DeltaNet, this component introduces position sensitivity.

### Loss & Training
- FEM introduces no additional loss terms; standard task losses are used (cross-entropy for language modeling, MSE for time series, etc.).
- Parameter budget strategy (i): $d = D/2$, $r = 4$, maintaining the same $4D^2$ parameter count as standard attention.
- In all experiments, FEM directly replaces the attention layer without modifying other hyperparameters.

## Key Experimental Results

### MAD Synthetic Benchmark — Attention Mechanism Diagnostics

| Model | Compress | Fuzzy Recall | In-Ctx Recall | Memorize | Selective Copy | Avg |
|-------|----------|-------------|--------------|----------|---------------|-----|
| Transformer (SMAttn) | 44.3 | 24.5 | 99.9 | 85.7 | 95.1 | 74.7 |
| DiffTransformer | 42.9 | 39.0 | 99.9 | 83.7 | 95.8 | 76.4 |
| GatedDeltaNet | 45.0 | 29.8 | 99.9 | 80.2 | 94.3 | 74.9 |
| **FEM-SM** | **53.1** | **43.1** | 99.9 | **85.9** | **99.3** | **80.2** |
| **FEM-GLA** | 53.0 | 19.1 | 99.9 | 86.3 | 99.0 | 74.9 |

### Ablation Study (MAD Average Score)

| Configuration | Avg | Notes |
|--------------|-----|-------|
| SMAttn (baseline) | 74.7 | Standard Transformer |
| +C (low-rank conv) | 76.3 | +1.6, local conditioning |
| +C,L (LSE) | **78.8** | +2.5, **largest jump**, LSE is the core |
| +C,L,T (temperature) | 79.4 | +0.6, temperature fine-tuning |
| +C,L,T,G (full FEM) | **80.2** | +0.8, outer gating as additional gain |

### Language Modeling (1.3B parameters, 100B tokens)

| Model | Open LLM Avg Rank↓ | Top-1 Count↑ |
|-------|-------------------|-------------|
| Transformer | 4.56 | 1 |
| **FEM-SM** | **2.06** | **9** |
| GLA | 5.63 | 0 |
| **FEM-GLA** | 3.88 | 1 |

### Time Series Forecasting (MSE)

| Dataset | FEM-SM | iTransformer | PatchTST | DLinear |
|---------|--------|-------------|----------|---------|
| Weather | **0.222** | 0.232 | 0.221 | 0.233 |
| ETTh1 | 0.419 | 0.454 | **0.413** | 0.422 |
| ETTm1 | **0.341** | 0.373 | 0.346 | 0.347 |
| ETTm2 | **0.242** | 0.265 | 0.247 | 0.252 |

### Key Findings
- **LSE mixing (L) is the core component**: It contributes the largest performance jump (+2.5 points) on the MAD benchmark, directly validating the critical role of free energy reading for Compress & Recall tasks.
- **FEM can elevate linear-time methods (GLA, Mamba) to performance levels approaching state-of-the-art attention variants**, closing the gap between linear and quadratic complexity models.
- **Computational overhead is manageable**: Full FEM-SM training latency is 0.041s vs. 0.027s for standard Transformer, with throughput 104K vs. 154K tokens/s—approximately 30% overhead.
- At the 1.3B scale for language modeling, FEM-SM achieves the best result on 9 of 16 benchmarks, with an average rank of 2.06 (vs. 4.56 for Transformer).

## Highlights & Insights
- **The precise diagnosis of the "lossless storage but lossy reading" problem** is a particularly profound contribution: the paper rigorously proves the fundamental limitations of standard attention from both a geometric perspective (convex hull constraint) and an information-theoretic perspective ($t^H$ vs. $t^D$ capacity), and systematically analyzes why "more heads / deeper layers / per-dimension QK / richer mixers" all fail to resolve this issue.
- **Mathematical elegance of the free energy variational framework**: Defining value retrieval as utility maximization under KL constraints naturally yields the log-sum-exp form, with the temperature parameter corresponding to the Lagrange multiplier for the KL budget. This framework unifies a continuous spectrum from mean readout to argmax.
- **The plug-and-play, prior-agnostic design** makes FEM broadly applicable: it is compatible with softmax attention, GLA, Mamba, and AFT, while preserving the original asymptotic complexity. This represents a rare "free lunch" style improvement.
- **The elegant design of LTL**: By proving the monotonicity of the map $\lambda \mapsto \beta^*$, the continuous spectrum of dynamic temperatures is compressed into a two-point interpolation, requiring only a single forward pass.

## Limitations & Future Work
- Validation on large-scale language models (>10B parameters) and very long contexts (>128K) is absent—the authors acknowledge limited computational resources.
- No custom CUDA kernels are implemented; the current 30% computational overhead could be optimized in engineering practice, but this is not addressed in the paper.
- The value space dimension is halved in FEM ($d = D/2$); although total parameter count is matched, the potential degradation in value representational capacity is not thoroughly analyzed.
- The performance advantage on time series forecasting is less pronounced than on NLP and synthetic benchmarks—FEM underperforms PatchTST on ETTh1/ETTh2.

## Related Work & Insights
- **vs. Differential Transformer**: DiffTrans reduces attention noise via a differential mechanism but still operates within the convex hull; FEM breaks the convex hull constraint.
- **vs. Mamba/SSM**: SSMs store history in a fixed-size state, which is inherently lossy storage; FEM achieves lossless reading on top of lossless storage (KV-cache).
- **vs. GatedDeltaNet/DeltaNet**: These linear RNNs update states using the delta rule; FEM can be stacked on top of them (FEM-GLA/Mamba) with significant performance gains.
- The theoretical framework of this paper is well-suited as both a baseline and an analytical tool for future research on attention mechanism improvements.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Both the diagnosis of the lossy reading problem and the free energy solution are highly original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Broad coverage across four domains (synthetic, NLP, vision, time series), but lacks very large-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical analysis is exceptionally rigorous, with a complete logical chain from problem formulation to solution.
- Value: ⭐⭐⭐⭐⭐ — A universal, plug-and-play mechanism improvement with high theoretical and practical value, likely to influence future directions in attention mechanism design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories](../../NeurIPS2025/time_series/xlstm-mixer_multivariate_time_series_forecasting_by_mixing_via_scalar_memories.md)
- [\[NeurIPS 2025\] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](../../NeurIPS2025/time_series/neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)
- [\[CVPR 2026\] SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](../../CVPR2026/time_series/sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)

</div>

<!-- RELATED:END -->
