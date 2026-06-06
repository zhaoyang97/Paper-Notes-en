---
title: >-
  [Paper Note] Memory-Efficient LLM Pretraining via Minimalist Optimizer Design
description: >-
  [ICML 2026][Optimization][Column Normalization] This work identifies the two essential components for LLM pretraining by "deconstructing Adam from the bottom up": column-wise gradient normalization and first-order moment…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Column Normalization"
  - "Last-layer Momentum"
  - "SCALE"
  - "Memory Efficiency"
  - "SGD vs Adam"
date: 2026-05-08
content_hash: 9b30f0f65811f639
---

# Memory-Efficient LLM Pretraining via Minimalist Optimizer Design

**Conference**: ICML 2026  
**arXiv**: [2506.16659](https://arxiv.org/abs/2506.16659)  
**Code**: Available (Note at end of paper: "Code is available at this link")  
**Area**: LLM Optimizers / Memory-Efficient Pretraining / Adam Alternatives  
**Keywords**: Column Normalization, Last-layer Momentum, SCALE, Memory Efficiency, SGD vs Adam  

## TL;DR
This work identifies the two essential components for LLM pretraining by "deconstructing Adam from the bottom up": column-wise gradient normalization and first-order momentum restricted to the final layer. Combining these into the SCALE optimizer achieves Adam-level or superior pretraining perplexity (surpassing Muon/APOLLO) while using memory close to SGD (13.74 GB on LLaMA 7B).

## Background & Motivation

**Background**: Adam is the de facto default optimizer for LLM pretraining, but it maintains two states for every parameter—the first moment $m^t$ and the second moment $v^t$. This results in roughly three times the memory consumption compared to SGD; for a 7B model, Adam states occupy 40 GB. Three primary routes have emerged to reduce this overhead: (i) state compression (Adafactor, SM3, CAME, GaLore via low-rank projection, Fira, APOLLO, and the rank-1 APOLLO-Mini); (ii) complete removal of certain states (Muon with only first-order momentum + Newton-Schulz orthogonalization, Scion, SWAN, SGD-SaI); and (iii) block-wise processing. While these methods introduce various normalization schemes, momentum variants, and low-rank approximations, a systematic deconstruction of which specific components are truly critical is lacking.

**Limitations of Prior Work**: (1) Vanilla SGD fails to converge on LLMs—Figure 2 shows that SGD perplexity does not decrease on LLaMA 130M. (2) To maintain stability, many memory-efficient methods run separate Adam optimizers for the first layer (embedding) and last layer (LM-head). For 60M models, these layers account for 50% of the parameters, essentially negating memory savings on small-to-medium models. (3) Various normalizations and momentum types are mixed arbitrarily without clear justification for their necessity.

**Key Challenge**: There is a fundamental tension between "state compression," which seeks to preserve the full behavior of Adam, and "memory efficiency," which aims to retain only the most essential components. The former inevitably leads to compression loss or extra computation, while the latter requires identifying which components can be safely discarded.

**Goal**: To answer systematically using a "bottom-up minimalist" approach: What are the minimum modifications required to bring vanilla SGD up to Adam's performance? The objective is to determine (a) which gradient normalization to use (singular value / column / row / sign), (b) whether first-order momentum is needed for every layer, and (c) if the second moment is truly necessary.

**Key Insight**: Adam is decomposed into two orthogonal components: the normalization factor $v^t$ and the Exponential Moving Average (EMA). Normalization requires no state, whereas EMA necessitates storing momentum. The strategy is to first stabilize SGD using normalization and then add the minimum amount of EMA required.

**Core Idea**: An optimizer sufficient for LLM pretraining only needs two things: column-wise normalization of gradients by "output dimension" (stateless and near-constant time) and first-order momentum applied only to the last layer (where gradient variance is highest). Other layers run on pure SGD; the second moment is entirely unnecessary.

## Method

### Overall Architecture

The design of SCALE (Stochastic Column-normalized Last-layer momEntum) stems from a three-step empirical chain. First, the authors evaluated SGD with various normalizations (Newton-Schulz singular value, column, row, sign) on LLaMA 60M/130M/350M. They found that singular value and column normalization brought SGD close to Adam's performance, while row and sign normalization performed poorly. Analysis of the LM-head gradient distribution revealed that row normalization produced extreme values (up to 150), destabilizing training. Second, they compared the gains of adding momentum to different layers. They empirically found that the last layer (LM-head) has the highest gradient variance. They used a convergence theorem for multi-layer SGD-M to prove that momentum should be allocated to layers with higher variance. Third, they combined these findings: column normalization (per layer, stateless, near-constant time) + first-order momentum for the last layer only (a negligible fraction of total parameters). SCALE consistently outperformed GaLore, Fira, APOLLO, and APOLLO-Mini, matching the performance of Muon and Adam.

Simplified Workflow: After each forward/backward pass, for each layer $l$: compute the mini-batch gradient $g_l^t$; if $l$ is the last layer, update $m_l^t=\beta\,m_l^{t-1}+(1-\beta)g_l^t$, otherwise $m_l^t=g_l^t$ (stateless); finally, $\theta_l^{t+1}=\theta_l^t-\eta_l\,\mathcal{C}(m_l^t)$, where $\mathcal{C}$ is the column normalization operator.

### Key Designs

1.  **Column-wise normalization as the sole normalization**:
    - **Function**: Each column of a weight matrix $G\in\mathbb{R}^{d_{in}\times d_{out}}$ is divided by its $\ell_2$ norm, resulting in $[\text{col}_1(G)/\|\text{col}_1(G)\|,\dots,\text{col}_n(G)/\|\text{col}_n(G)\|]$, requiring no additional state.
    - **Mechanism**: The four normalizations correspond to steepest descent directions under different matrix norms. On LLaMA 60M/130M/350M, sign (54.36/40.42/27.95) and row (79.27/37.67/21.63) normalization were significantly worse than Adam (30.05/23.13/18.77). Column normalization (39.89/28.85/20.38) and singular value (NS) normalization (34.15/25.25/18.73) performed similarly to Adam. Computationally, for $d{=}4096$, SVD takes 1958.66 ms and Newton-Schulz takes 14.41 ms, whereas column normalization takes only 0.17 ms.
    - **Design Motivation**: Normalization is the ideal starting point for "zero-cost performance gains" as it is stateless. Column normalization was chosen because it hits the sweet spot of being "sufficiently performant" and "computationally negligible," while naturally handling the highly unbalanced gradients of frequent tokens in the LM-head.

2.  **Last-layer momentum (mmt-last)**:
    - **Function**: Intermediate layers do not maintain momentum and use raw gradients; only the LM-head layer maintains $m_L^t=\beta m_L^{t-1}+(1-\beta)g_L^t$. On LLaMA 7B, this layer represents only ~2% of the parameters, making momentum overhead negligible.
    - **Mechanism**: By approximating true gradients with large batches, the authors observed that the "last layer consistently has the highest gradient variance" throughout training (Figure 4a). Theorem 2.1 shows that the multi-layer SGD-M convergence rate includes variance terms weighted by $\frac{1-\beta_l}{1+\beta_l}$. The corollary is that applying a large $\beta$ only to high-variance layers while setting $\beta=0$ elsewhere restores convergence while saving memory. Empirically (Table 3), SCALE matches or exceeds Adam on 60M/130M/350M models. Figure 4b also shows that stabilizing the last layer subsequently reduces variance in the first layer.
    - **Design Motivation**: Since momentum is the primary source of Adam's memory overhead, proving it is only "essential for the noisiest layer" allows compressing Adam's state to nearly zero.

3.  **Minimalist Combination: The SCALE Algorithm**:
    - **Function**: Combines the above components into a complete optimizer with only a few lines of code change relative to Adam and virtually no memory increase over vanilla SGD (only 2% for 7B models).
    - **Mechanism**: The second moment $v^t$ is entirely discarded. Compared to SWAN, which uses both row and singular value normalization (plus Adam for the LM-head), SCALE shows that a single normalization (column) is sufficient. Compared to Scion, which applies momentum to all layers, SCALE demonstrates that only the last layer is necessary. On LLaMA 7B, SCALE uses 13.74 GB of memory vs. 40.43 GB for Adam and 26.95 GB for Muon, placing it on the Pareto frontier with superior perplexity (12.59 vs. Muon’s 12.72).
    - **Design Motivation**: This is the natural conclusion of the bottom-up minimalist approach—incorporating only what is proven necessary to create a "minimal LLM optimizer" baseline.

### Loss & Training

LLaMA 60M-1B models were pretrained on C4 to Chinchilla-optimal token counts (1.4B-20B tokens). The 7B model was trained for 19.7B tokens (150K steps) and subjected to a 100B token stability test using 8 NVIDIA H200 141G GPUs. Hyperparameters follow the settings from Zhao et al. (2024) (GaLore).

## Key Experimental Results

### Main Results

| Model | Adam PPL / Memory | Muon | GaLore | APOLLO-Mini | **SCALE (Ours)** |
|------|------------------|------|--------|-------------|-----------------|
| 60M  | 30.05 / 0.35G | 28.86 / 0.23G | 34.58 / 0.28G | 31.85 / 0.25G | **30.81 / 0.15G** |
| 130M | 23.13 / 0.81G | 22.20 / 0.54G | 25.31 / 0.61G | 23.63 / 0.46G | **22.57 / 0.32G** |
| 350M | 18.77 / 2.21G | 16.70 / 1.47G | 19.37 / 1.59G | 17.11 / 1.00G | **16.32 / 0.80G** |
| 1B   | 15.79 / 8.04G | 13.67 / 5.36G | 15.05 / 4.76G | 13.48 / 3.20G | **13.49 / 2.81G** |
| 7B   | -      | 12.72 / 26.95G | -     | 13.09 / 14.53G  | **12.59 / 13.74G** |

SCALE either achieves Prev. SOTA (350M/7B) or matches the strongest baselines while using only 35-65% of the memory.

### Ablation Study

| Configuration | 60M / 130M / 350M PPL | Description |
|------|--------------------------|------|
| SGD + sign normalization | 54.36 / 40.42 / 27.95 | Sign is too coarse, significantly worse than Adam |
| SGD + row normalization | 79.27 / 37.67 / 21.63 | LM-head gradients explode (magnitude ~150), training diverges |
| SGD + Singular Value (NS) | 34.15 / 25.25 / 18.73 | Good performance but slow (14.41 ms vs 0.17 ms for column) |
| SGD + **Column Normalization** | 39.89 / 28.85 / 20.38 | Best trade-off between performance and speed |
| + Last-layer momentum (SCALE) | 30.81 / 22.57 / 16.32 | Matches or exceeds Adam |
| Adam (Baseline) | 30.05 / 23.13 / 18.77 | Full-state Adam |

### Key Findings
- The last layer is the optimization "choke point": The LM-head dimensions ($d_\text{model}\times |V|$) result in column gradient norms for frequent tokens that far exceed others, dictating the choice of normalization (column > row) and momentum allocation.
- Column normalization provides "free" performance: Unlike SWAN’s complex scheme, column normalization is universally applicable and eliminates the need for Adam on peripheral layers.
- Second moments are not strictly necessary: SCALE validates that removing $v^v$ still matches Adam, supporting the direction of optimizers like Muon while showing that even first moments can be drastically reduced.
- Stabilizing the noisiest source (the last layer) prevents error propagation to upstream layers (Figure 4b).

## Highlights & Insights
- **The "bottom-up" methodology is the most significant contribution**—rather than reinventing an optimizer, the authors stripped Adam to its minimum viable components, proving that much of the existing complexity is redundant. This logic can be applied to other standard components like LayerNorm or Attention.
- **Visual diagnostic of LM-head gradient distribution** is highly persuasive—histograms explain why row normalization fails and column normalization succeeds, moving from empirical selection to mechanistic understanding.
- **Theorem 2.1 provides a theoretical basis for per-layer momentum**: The convergence rate indicates that hyperparameters should be tuned independently based on layer variance rather than using a global $\beta$.

## Limitations & Future Work
- Model scale is capped at 7B and 100B tokens; stability and basin of attraction at 70B+ scales remain unverified.
- Lack of multi-seed runs for the 7B model; the 0.13 PPL difference from Muon may lack statistical significance.
- No direct comparison with the latest Muon variants (e.g., Muon-Clip) or the full Scion recipe.
- Column normalization treats weights as "input x output" matrices; applicability to non-attention architectures (e.g., Mamba or MoE expert weights) requires further discussion.
- Whether post-training (SFT/RL) also only requires last-layer momentum remains an open question.

## Related Work & Insights
- **vs GaLore/Fira/APOLLO**: These methods focus on "state compression" via low-rank projection; SCALE takes a "redesign" approach by discarding the second moment and minimizing momentum, outperforming them in both memory and PPL.
- **vs Muon**: Muon uses global momentum and computationally expensive Newton-Schulz orthogonalization. SCALE uses column normalization (nearly free) and last-layer momentum, requiring only 51% of Muon's memory with better perplexity.
- **vs SWAN**: SCALE simplifies the redundant combination of row and singular value normalization into a single column-wise operation.
- **vs Scion**: While Scion explores layer-wise normalization, SCALE adds selective constraints by demonstrating that momentum is only needed for the highest-variance layer.
- **Implication**: Each new optimizer component should pass an "ablation-by-removal" test. SCALE sets a strong baseline for achieving Adam-level performance with SGD-level memory.

## Rating
- Novelty: ⭐⭐⭐⭐ Stripping Adam rather than adding new features is a sharp and effective perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 60M-7B scales, various normalizations, and 100B token stability, though missing multi-seed for large runs.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain from motivation to theory to algorithm.
- Value: ⭐⭐⭐⭐⭐ Provides a high-performance, memory-efficient baseline that should serve as a default comparison for future LLM optimizer research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Memory-Efficient 4-bit Preconditioned Stochastic Optimization](../../ICCV2025/optimization/memory-efficient_4-bit_preconditioned_stochastic_optimization.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[ICML 2026\] LiMuon: Light and Fast Muon Optimizer for Large Models](limuon_light_and_fast_muon_optimizer_for_large_models.md)
- [\[ICML 2026\] Enhancing LLM Training via Spectral Clipping](enhancing_llm_training_via_spectral_clipping.md)
- [\[ICML 2026\] PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](pathwise_planning_through_world_model_for_automated_heuristic_design_via_self-ev.md)

</div>

<!-- RELATED:END -->
