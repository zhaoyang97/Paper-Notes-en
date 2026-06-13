---
title: >-
  [Paper Note] Score-Based Error Correcting Code Decoder
description: >-
  [ICML2026][Physics & Scientific Computing][Error Correcting Codes] This paper proposes SB-ECC: reformulating soft decoding of binary linear block codes as the reverse denoising of a Variance Exploding (VE) diffusion proc…
tags:
  - "ICML2026"
  - "Physics & Scientific Computing"
  - "Error Correcting Codes"
  - "Score-Based Models"
  - "Probability Flow ODE"
  - "Neural Decoding"
  - "DPM-Solver"
date: 2026-05-08
content_hash: c4c1fedd3d7f4b2b
---

# Score-Based Error Correcting Code Decoder

**Conference**: ICML2026  
**arXiv**: [2605.28358](https://arxiv.org/abs/2605.28358)  
**Code**: https://github.com/alonhelvits/SB-ECC  
**Area**: Signal & Communications / Neural Decoding / Diffusion Models  
**Keywords**: Error Correcting Codes, Score-Based Models, Probability Flow ODE, Neural Decoding, DPM-Solver

## TL;DR
This paper proposes SB-ECC: reformulating soft decoding of binary linear block codes as the reverse denoising of a Variance Exploding (VE) diffusion process. It utilizes a **time-unconditioned** score network that directly processes **signed channel observations** $\mathbf{y}$ to solve a parity-check guided probability flow ODE. SB-ECC achieves SOTA Bit Error Rate (BER) in 39 out of 42 code-SNR configurations, with an average SNR gain of 0.17 dB and a maximum of 0.46 dB.

## Background & Motivation

**Background**: Soft decoding of Error Correcting Codes (ECC) has long been dominated by iterative algorithms like Belief Propagation (BP) on factor graphs. In the deep learning era, research has split into two paths: (a) *model-driven* neural decoders that unfold BP/Min-Sum into trainable networks, preserving the Tanner graph structure; (b) *model-agnostic* decoders that directly learn the $\mathbf{y} \mapsto$ codeword mapping. Transformer-based decoders like ECCT and CrossMPT use parity-check matrix $H$-guided masked attention to integrate inductive biases from (a) into the flexible architectures of (b), surpassing classical BP. Recently, DDECC introduced diffusion concepts to channel decoding, treating "transmission as forward noise injection and decoding as reverse denoising."

**Limitations of Prior Work**: Existing strong baselines (ECCT / CrossMPT / DDECC) share a seemingly harmless but practically restrictive preprocessing choice—mapping channel observations to absolute values $\mathbf{y} \mapsto |\mathbf{y}|$ to obtain "reliability," combined with syndromes calculated from hard decisions. This presents two issues:
1. The mapping $\mathbf{y} \mapsto |\mathbf{y}|$ is **irreversible**: all $2^n$ instances of $\{\mathbf{s} \odot \mathbf{y} : \mathbf{s} \in \{\pm 1\}^n\}$ are mapped to the same input, discarding directional information;
2. These methods directly predict bit logits, whereas diffusion/score models essentially learn a **continuous vector field** in $\mathbb{R}^n$, creating a natural mismatch between discrete hard-decision outputs and continuous ODE solving.

**Key Challenge**: Integrating diffusion models (like DDECC) within the "reliability preprocessing + discrete bit output" framework diminishes the geometric advantages of score-based models. However, retaining geometry faces the risk of overfitting due to the extreme sparsity of training samples relative to the codeword space $|\mathcal{C}|=2^k$ (a classic pain point noted by Bennatan et al., 2018).

**Goal**: (a) Design a decoder that directly processes signed $\mathbf{y}$ and outputs a continuous denoising direction in $\mathbb{R}^n$; (b) eliminate conditional dependence on SNR/time at inference, as actual receivers often lack SNR knowledge; (c) allow for adjustable computational budgets, enabling flexible selection of ODE solver steps based on latency requirements.

**Key Insight**: The authors observe that an AWGN channel $\mathbf{y} = \mathbf{x}_s + \mathbf{z}, \mathbf{z}\sim\mathcal{N}(\mathbf{0},\sigma_{ch}^2 I)$ is itself a marginal of a VE diffusion process at some unknown $t^\star$. Thus, the entire AWGN channel is treated as a "single slice of forward diffusion," making decoding equivalent to reverse integration along the PF-ODE to $t=0$.

**Core Idea**: Use a **time/SNR-independent** score network $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{y}, \mathbf{s})$ that directly takes signed $\mathbf{y}$ and syndromes as input to solve a parity-guided PF-ODE on a uniform $\sigma$-space, fully preserving the channel geometry.

## Method

### Overall Architecture

SB-ECC treats decoding as a reverse denoising trajectory in $\sigma$-space:

1. **Modeling**: After BPSK, $\mathbf{x}_0 \in \{\pm 1\}^n$ passes through AWGN to yield $\mathbf{y} = \mathbf{x}_0 + \sigma_{ch}\boldsymbol{\epsilon}$, which is identical to the VE-SDE marginal $\mathbf{x}_t = \mathbf{x}_0 + \sigma(t)\boldsymbol{\epsilon}$ at $\sigma(t^\star) = \sigma_{ch}$.
2. **Training**: Uniformly sample $t\sim\mathcal{U}(0,1)$ and Gaussian noise $\boldsymbol{\epsilon}$ to construct synthetic received signals $\mathbf{y} = \mathbf{x}_0 + \sigma(t)\boldsymbol{\epsilon}$. Calculate the syndrome $\mathbf{s} = H\,\mathrm{bin}(\mathrm{sign}(\mathbf{y}))^\top$. The network fits $\boldsymbol{\epsilon}$ via noise prediction. Notably, the network **does not receive** $t$ or $\sigma$, thus requiring no SNR estimation during inference.
3. **Inference (Algorithm 2 — Early-Exit Decoding)**: Starting from $\mathbf{x}^{(0)} = \mathbf{y}$, the syndrome is calculated at each step. If it is zero, the current hard decision is a valid codeword, and the process exits early. Otherwise, $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}^{(i)}, \mathbf{s})$ is called to obtain the denoising direction, followed by an Euler step $\mathbf{x}^{(i+1)} = \mathbf{x}^{(i)} - \Delta\sigma\,\hat{\boldsymbol{\epsilon}}$, descending uniformly from $\sigma_{\max} \to \sigma_{\min}$.
4. **Mechanism for Solver Replacement**: Euler steps can be seamlessly replaced by DPM-Solver (facilitated by the linear $\sigma$-schedule), reducing end-to-end latency by 8.86% on average (up to 12.82%) without performance degradation.

### Key Designs

1. **Signed Input + Tanner Graph Masked Attention (Geometry Preservation without Loss of Code Structure)**:
    - **Function**: Replaces the mainstream "$|\mathbf{y}|$ + syndrome" two-channel preprocessing, allowing the network to learn a continuous denoising vector field directly in $\mathbb{R}^n$.
    - **Mechanism**: Adopts the bimodal architecture of CrossMPT (Variable Node (VN) tokens × Check Node (CN) tokens exchanging messages via $H$-guided masked cross-attention), but modifies the VN channel from $|\mathbf{y}|$ to signed $\mathbf{y}$. The CN channel remains the syndrome. The output shifts from "bit flip probabilities/logits" to a "denoising direction in $\mathbb{R}^n$ $\hat{\boldsymbol{\epsilon}}_\theta \in \mathbb{R}^n$."
    - **Design Motivation**: $\mathbf{y} \mapsto |\mathbf{y}|$ is a $2^n$-to-1 folding map that loses per-coordinate direction—exactly the information needed for score field learning. However, purely model-agnostic approaches would be overwhelmed by combinatorial explosion; thus, $H$-guided masked attention is retained to inject parity-check constraints. This is the minimal modification required to preserve both geometry and structure.

2. **Time-Unconditioned + Linear $\sigma$-schedule (No SNR Needed at Inference)**:
    - **Function**: Enables a single trained network to function under any channel SNR without feeding $t$ or $\sigma$ to the model.
    - **Mechanism**: During training, $\sigma(t) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) t$ is a linear function of $t$, with $t \sim \mathcal{U}(0,1)$ covering the full noise spectrum. The network only sees $(\mathbf{y}, \mathbf{s})$, essentially learning a denoising field **shared across all SNRs**. During inference, $\sigma_{ch}$ is not estimated; the model simply follows a fixed $\sigma$-grid over $N_{\text{steps}}$ from $\mathbf{x}^{(0)} = \mathbf{y}$.
    - **Design Motivation**: Real-world receivers often cannot obtain accurate SNR, and SNR-conditioned models are sensitive to SNR bias. While a time-unconditioned model must learn an "average field" robust to various noise levels, the syndrome implicitly provides a strong signal of "distance to the valid codeword," compensating for this. This choice also paves the way for DPM-Solver integration.

3. **Parity-Guided PF-ODE + Early Exit (Adjustable Precision and Controllable Latency)**:
    - **Function**: Injects parity constraints as soft guidance into the PF-ODE integration and utilizes syndrome checks at each step for early stopping.
    - **Mechanism**: The PF-ODE is expressed as $d\mathbf{x}_t = -\tfrac{1}{2} g(t)^2 \nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)\,dt$, with the learned $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, \mathbf{s})$ replacing the score (in VE, these differ only by a time-dependent constant $-1/\sigma(t)$). Re-calculating hard decisions and syndromes at each iteration embeds discrete "constraint satisfaction" events into continuous dynamics. Replacing Euler with DPM-Solver allows the $N_{\text{steps}}$ NFE budget to achieve equivalent $-\ln(\mathrm{BER})$ with fewer evaluations.
    - **Design Motivation**: The tradition of "checking if the syndrome is zero every round" from iterative decoding (BP) naturally becomes an early-exit mechanism in the diffusion framework. This allows users to dynamically balance trade-offs: good channels finish in 5-10 steps, while poor channels utilize the full budget. Combined with SNR-independence, the inference curve becomes a single, slidable precision-latency control knob.

### Loss & Training
A single denoising loss is used: $\mathcal{L}_\epsilon = \mathbb{E}\|\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{y}, \mathbf{s}(\mathbf{y})) - \boldsymbol{\epsilon}\|_2^2$ (Algorithm 1). Each batch randomly samples $t \sim \mathcal{U}(0,1)$ and $\boldsymbol{\epsilon}$ to construct synthetic received signals $\mathbf{y}$ and corresponding syndromes for end-to-end Adam optimization. Architectural hyperparameters (layers, d_model, heads) follow CrossMPT exactly for controlled comparison.

## Key Experimental Results

### Main Results

Evaluated across 14 codes from 5 families (BCH, Polar, LDPC, MacKay, CCSDS) at $E_b/N_0 \in \{4, 5, 6\}$ dB (42 total configurations), using $-\ln(\mathrm{BER})$ as the metric (higher is better):

| Code (n, k) | $E_b/N_0$ | BP-50 | CrossMPT | DDECC | SB-ECC (Ours) | Gain over Strongest Baseline |
|----------|-----------|-------|----------|-------|---------------|--------------|
| BCH(63,45) | 6 dB | 7.26 | 11.62 | 11.41 | **13.17** | +1.55 |
| Polar(128,64) | 6 dB | 6.15 | 14.76 | 16.27 | **16.94** | +0.67 |
| LDPC(121,80) | 6 dB | 17.33 | 18.15 | 18.26 | **20.42** | +2.16 |
| LDPC(121,70) | 6 dB | 15.62 | 17.52 | 17.98 | **19.24** | +1.26 |
| MacKay(96,48) | 6 dB | 12.57 | 15.52 | 16.04 | **16.25** | +0.21 |

Macro view: SB-ECC achieves the **best BER in 39 out of 42 cases**, with an average SNR gain of **0.17 dB** and a maximum of **0.46 dB** compared to the strongest competitor.

### Ablation Study

| Configuration | Change | Key Metric Change |
|------|------|--------------|
| Full SB-ECC (Euler) | signed $\mathbf{y}$ + Tanner Attn + Time-unconditioned + PF-ODE | Baseline $-\ln(\mathrm{BER})$ |
| w/o signed (use $\|\mathbf{y}\|$) | Revert to mainstream reliability preprocessing | Significant BER degradation; loss of geometric info confirmed (Sec 5.3) |
| Time-conditioned | Concatenate $\sigma$ into tokens | Requires SNR estimation; slight performance drop and loss of robustness to "unknown SNR" |
| Euler → DPM-Solver | Solver replacement | $-\ln(\mathrm{BER})$ remains stable; average end-to-end latency ↓8.86% (max ↓12.82%) |

### Key Findings
- **Signed inputs provide the largest gains on medium LDPC/BCH codes**: For codes with larger $k$ and massive $|\mathcal{C}|$, the geometric value of directional information is maximized by the score field. Gains are more modest for short or low-rate codes (e.g., (64,48) Polar).
- **DPM-Solver compatibility stems from linear $\sigma$-scheduling + time-unconditionality**: DPM requires analytically integrable noise schedules. By discretizing according to $\sigma$ rather than $t$, the design assumptions for higher-order DPM terms are directly satisfied.
- **Early-exit maintains average NFE much lower than worst-case $N_{\text{steps}}$**: At high SNR, most samples reach a zero syndrome within a few steps, meaning overall inference latency is dominated by "hard" samples.

## Highlights & Insights
- **"AWGN channel = One time-step of VE Diffusion"** is an elegant unifying perspective. Unlike DDECC, which maps channels to *artificial* discrete diffusion steps, this work aligns directly with the marginal $t^\star$ of a continuous SDE. This ensures mathematical consistency between training and test distributions and automatically supports any SNR.
- **Time-unconditional design is a key prerequisite for DPM-Solver**: The combination of time-unconditionality + linear $\sigma$ $\to$ uniform $\sigma$-grid allows higher-order ODE solvers to be used immediately. This is a classic example of a "design choice for inference robustness accidentally opening up latency optimization space."
- **Syndrome as a *conditional input rather than a supervision signal*** offers an interesting contrast to codeword structure injection. While DDECC uses syndrome as a step index, this work treats it as a token, allowing cross-attention with $H$-masking to automatically route constraint information—effectively leveraging all advancements in Transformer decoders.
- **Transferability to other "constrained denoising problems"**: This template (unconditional score + constraints as tokens + early exit) could be applied to parity-constrained image restoration, CRC-guided speech denoising, or constraint satisfaction problem (CSP) solving.

## Limitations & Future Work
- **Limited architectural innovation**: The network backbone follows CrossMPT exactly, with almost no structural changes. The contribution is more about "redesigning the correct input/output/scheduling" rather than architectural breakthroughs.
- **The 0.17 dB average gain is "significant but not drastic" for engineering**: In certain channels or code families, gains are marginal. Readers expecting a >1 dB leap might be disappointed.
- **No coverage of very long codes (e.g., LDPC(1024,512))**: As $n$ increases, the $\mathcal{O}(n \cdot |E|)$ overhead of CrossMPT attention and the difficulty of score field learning will scale. Long-code scaling remains an open question.
- **Parity guidance remains "soft" with no optimality guarantees**: Compared to lattice-based or generalized sparsity methods that approach ML decoding "theoretically," this is an empirical approximation that lacks convergence or optimality proofs.
- **Future Directions**: (a) Introducing syndrome guidance during *training* (CFG-style) rather than just as a token; (b) using higher-order solvers like DPM++ to further compress NFE; (c) distilling into a single-step decoder via consistency models to compete with low-latency routes like Lei et al. (2025).

## Related Work & Insights
- **vs. DDECC (Choukroun & Wolf, 2022)**: DDECC uses discrete steps and $|\mathbf{y}|$ preprocessing. Ours uses continuous VE diffusion aligned with the channel and retains signed inputs. Outperforming it in 39/42 cases proves "geometry + continuous time" is better suited for decoding.
- **vs. CrossMPT (Park et al., 2024)**: Shares the backbone; differences lie in input channels, output semantics, and the inference paradigm. This work essentially upgrades CrossMPT from "one-shot logits" to "multi-step PF-ODE denoising," showing that representation/dynamics are the real bottlenecks when the backbone is competent.
- **vs. ECCT / Foundation Decoder (Choukroun & Wolf, 2022/2024)**: Those works explore direct decoding with larger Transformers. This work provides an orthogonal direction—**reformulating the task** is more effective than stacking parameters for the same capacity.
- **vs. ECC-Consistency Flow (Lei et al., 2025)**: They distill for single-step low latency. Ours uses DPM-Solver for an 8-13% latency reduction in a "precision-first" mode; the two could be combined for "Accuracy vs. Speed" dual-mode operation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of AWGN ↔ VE alignment + signed input + time-unconditionality is a clever synergy previously unexplored.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of 42 configurations across 5 code families, with latency benchmarks for DPM-Solver.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and pseudocode; preliminaries explain VE-SDE, DSM, and Tanner graphs thoroughly.
- Value: ⭐⭐⭐⭐ Provides a new design template for the neural decoding community, likely to serve as a benchmark for future score-based decoding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/physics/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](../../NeurIPS2025/physics/score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)
- [\[ICML 2026\] Speculative Sampling for Faster Molecular Dynamics](speculative_sampling_for_faster_molecular_dynamics.md)
- [\[ICML 2026\] BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields](ballast_bayesian_active_learning_with_look-ahead_amendment_for_sea-drifter_traje.md)

</div>

<!-- RELATED:END -->
