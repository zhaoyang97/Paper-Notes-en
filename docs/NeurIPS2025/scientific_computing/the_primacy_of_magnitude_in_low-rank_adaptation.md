---
title: >-
  [Paper Note] The Primacy of Magnitude in Low-Rank Adaptation
description: >-
  [NeurIPS 2025][Scientific Computing][LoRA] This paper reveals that weight update magnitude is the fundamental driver of performance in LoRA, unifying the influence of learning rate, scaling factor, and initialization strategy under a single framework. It further proposes LoRAM—an efficient initialization method based on deterministic orthogonal bases and magnitude scaling—that matches or surpasses spectral initialization methods without requiring SVD.
tags:
  - NeurIPS 2025
  - Scientific Computing
  - LoRA
  - parameter-efficient fine-tuning
  - initialization strategy
  - weight update magnitude
  - low-rank adaptation
date: 2026-05-08
content_hash: b1d40a115c5ad56a
---

# The Primacy of Magnitude in Low-Rank Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2507.06558](https://arxiv.org/abs/2507.06558)
**Code**: [GitHub](https://github.com/zhangzicheng-jd/LoRAM)
**Area**: Scientific Computing
**Keywords**: LoRA, parameter-efficient fine-tuning, initialization strategy, weight update magnitude, low-rank adaptation

## TL;DR

This paper reveals that weight update magnitude is the fundamental driver of performance in LoRA, unifying the influence of learning rate, scaling factor, and initialization strategy under a single framework. It further proposes LoRAM—an efficient initialization method based on deterministic orthogonal bases and magnitude scaling—that matches or surpasses spectral initialization methods without requiring SVD.

## Background & Motivation

**Background**: LoRA is the most widely adopted parameter-efficient fine-tuning method, injecting trainable low-rank matrices $B \in \mathbb{R}^{n \times r}$ and $A \in \mathbb{R}^{r \times m}$ to fine-tune large models while updating fewer than 1% of parameters. Recent SVD-based initialization methods such as PiSSA, MiLoRA, and OLoRA have substantially improved convergence speed and downstream performance.

**Limitations of Prior Work**:
   - **Efficiency overhead**: Spectral initialization requires SVD decomposition of pretrained weights, incurring additional computational and memory costs that are impractical in resource-constrained settings (e.g., quantized LoRA, federated learning).
   - **Insufficient understanding**: The success of spectral initialization is commonly attributed to "preserving knowledge in principal components," yet this intuition lacks theoretical grounding—the non-convex optimization of LoRA renders training dynamics difficult to predict.

**Key Challenge**: Spectral initialization methods are effective but costly, and their success mechanism remains unclear. The key question is whether equivalent performance can be achieved without SVD.

**Goal**: (a) Identify the true mechanism underlying spectral initialization; (b) Design an efficient alternative that does not require SVD.

**Key Insight**: The analysis proceeds from the perspective of **weight update magnitude**, $\nu[W_{\text{LoRA}}] = \frac{1}{mn}\|W_{\text{LoRA}}\|_F^2$, examining how various hyperparameters influence performance through their effect on magnitude during LoRA training dynamics.

**Core Idea**: The essence of spectral initialization is not "knowledge preservation" but "magnitude amplification." Its effect can be reproduced using a deterministic orthogonal basis combined with a scaling factor derived from the statistics of pretrained weights.

## Method

### Overall Architecture

Analytical Framework (Magnitude Principle) → Mechanistic Analysis (Demystifying Spectral Gains) → Efficient Solution (LoRAM)

### Key Designs

#### 1. Magnitude Principle

- **Function**: Establishes weight update magnitude as a unified framework for analyzing LoRA training dynamics.
- **Mechanism**: The weight update magnitude of LoRA is given by $\nu[\Delta W_{\text{LoRA}}^{(t)}] \approx r\alpha^2\eta^2(\nu[B^{(t)}]\nu[\nabla_A L^{(t)}] + \nu[\nabla_B L^{(t)}]\nu[A^{(t)}])$, jointly governed by the learning rate $\eta$, the scaling factor $\alpha$, and the initialization magnitude.
- **Key Theorem** (Proposition 1 – Parameter Scaling Equivalence): Proves an exact equivalence among $\alpha$, initialization magnitude, and learning rate—increasing $\alpha$ is equivalent to increasing the initialization magnitude or adjusting the learning rate, as all three fundamentally regulate update magnitude.

#### 2. Magnitude Constraints from Low-Rank Structure (Proposition 2)

- **Function**: Proves that the low-rank structure of LoRA inherently constrains the update magnitude.
- **Key Finding**: $\nu[W_{\text{LoRA}}^{(t)}] \approx k_1 \gamma t$, where $k_1 = r(m\sigma_A^4 + n\sigma_B^4)$. The standard "Noise & Zeros" initialization yields $k_1 = r/m$, which is orders of magnitude smaller than full fine-tuning.
- **Design Motivation**: This explains the slow convergence of LoRA—the low-rank structure suppresses update magnitude relative to full-parameter methods. Any approach that amplifies $k_1$ can improve LoRA.

#### 3. Magnitude Gain Analysis of Spectral Initialization

- **Function**: Reveals the true mechanism behind spectral initialization methods such as PiSSA.
- **Mechanism**: PiSSA initializes with $A^{(0)} = \sqrt{S_r} V_{:,:r}^\top$ and $B^{(0)} = U_{:,:r} \sqrt{S_r}$. Defining the spectral concentration factor $\rho[r] = \mathbb{E}_r[s]^2 / \mathbb{E}_{\mathcal{R}[W]}[s^2]$, PiSSA achieves $k_1 = Q[r](m+n)\nu[W]$, where $Q[r] = \rho[r] \cdot r / \mathcal{R}[W]$ is the "spectral gain factor."
- **Key Argument**: The core advantage of spectral initialization does not lie in aligning LoRA basis directions with principal components ("knowledge preservation"), but rather in amplifying update magnitude through singular value scaling. This is validated via tracking-mode experiments—matching the magnitude of spectral initialization with any orthogonal basis yields comparable performance.

#### 4. LoRAM Initialization

- **Function**: Designs an efficient initialization scheme that does not require SVD.
- **Mechanism**:
    - Uses the Discrete Sine Transform (DST) basis $\Phi_m$ as a deterministic orthogonal basis: $\Phi_m[i,j] = \sqrt{2/(m+1)} \sin((i+1)(j+1)\pi/(m+1))$
    - Approximates the spectral gain factor logarithmically: $Q[r] \approx \log_{\min(n,m)}(r)$
    - Scaling factor: $\beta = (Q[r] \cdot \nu[W] / \nu[\Phi_n \Phi_m^\top])^{1/4}$
    - Initialization: $A^{(0)} = \beta \cdot \Phi_m^\top$, $B^{(0)} = \beta \cdot \Phi_n$, $W \leftarrow W - \beta^2 \cdot \Phi_n \Phi_m^\top$
- **Design Motivation**: The DST basis is analytically defined, requires no storage, and is reproducible across devices. The logarithmic approximation effectively captures the monotonically increasing and concave characteristics of $Q[r]$.

### Loss & Training

- LoRAM modifies only the initialization and leaves the training procedure unchanged, remaining fully compatible with the standard LoRA training pipeline.
- It can be combined with RsLoRA ($\alpha = \sqrt{r}$) for further gains.
- After initialization, $B^{(0)}A^{(0)}$ is absorbed into the frozen weight matrix $W$.

## Key Experimental Results

### Main Results

#### NLG Tasks (LLaMA-2-7B, Table 1)

| Method | GSM8K (r=16) | MATH (r=16) | HumanEval (r=16) | GSM8K (r=128) | MATH (r=128) |
|--------|-------------|-------------|-----------------|-------------|-------------|
| LoRA | 31.51 | 4.16 | 15.98 | 40.27 | 4.72 |
| RsLoRA | 39.04 | 4.94 | 18.85 | 50.38 | 7.32 |
| PiSSA | 37.68 | 5.16 | 18.37 | 51.48 | 7.04 |
| **LoRAM** | **40.32** | **5.30** | **18.92** | 51.12 | **7.25** |

#### NLU Tasks (DeBERTa-v3-base, Table 2)

| Method | MRPC | CoLA | RTE | STS-B |
|--------|------|------|-----|-------|
| LoRA | 84.06 | 63.56 | 50.18 | 87.20 |
| PiSSA | 89.21 | 65.06 | 74.36 | 88.90 |
| **LoRAM** | **89.95** | 65.53 | **74.72** | **89.93** |

#### Multimodal Tasks (LLaVA, Table 3)

| Method | MME_Cog | MMMU | AI2D | ScienceQA |
|--------|---------|------|------|-----------|
| LoRA | 278 | 0.331 | 0.557 | 0.684 |
| PiSSA | 311 | 0.344 | 0.564 | 0.686 |
| **LoRAM** | 308 | **0.350** | **0.571** | **0.700** |

### Ablation Study (Table 4, LLaMA-2-7B NLG)

| Ablation | r=16 GSM8K | r=128 GSM8K | Finding |
|----------|-----------|-------------|---------|
| $Q[r] = \log(r/2)$ | 40.1 | 50.7 | Slightly lower |
| $Q[r] = \log(r)$ (default) | **40.3** | **51.1** | Best log approximation |
| DST basis (default) | 40.3 | 51.1 | — |
| Random orthogonal basis | 36.3 | 50.2 | Basis choice has limited impact |
| Gaussian basis | 35.8 | 49.8 | Orthogonality provides marginal benefit |
| PiSSA tracking | 36.7 | 49.5 | Matching magnitude suffices for comparable performance |
| LoRAM + RsLoRA | **52.1** | **59.4** | Combined use yields further gains |

### Key Findings

1. **Magnitude is the key**: Tracking-mode experiments confirm that matching the magnitude of spectral initialization with a DST basis achieves performance comparable to PiSSA.
2. **Basis choice has limited impact**: Differences among DST, random orthogonal, and Gaussian bases are small, confirming that spectral directions are not critical.
3. **Gains are more pronounced at lower ranks**: The concavity of $Q[r]$ implies that the marginal benefit of magnitude amplification increases as rank decreases.
4. **Combination with RsLoRA**: LoRAM + RsLoRA achieves further improvements on most tasks, though excessive amplification at high rank can be detrimental.
5. **Convergence speed**: The training loss curve of LoRAM closely tracks that of PiSSA, with faster convergence in the early stages.

## Highlights & Insights

1. **A unifying perspective**: The learning rate, scaling factor, and initialization—three seemingly independent tuning dimensions—are unified under the single principle of magnitude control.
2. **Debunking a misconception**: The success of spectral initialization stems not from "preserving knowledge directions" but from simple magnitude amplification—a counterintuitive yet compelling finding.
3. **Minimalist design**: LoRAM requires only a few lines of code (DST basis generation + scaling factor computation), with no SVD, no additional storage, and no modification to the training pipeline.
4. **Rigorous theory**: Proposition 1 (parameter scaling equivalence) and Proposition 2 (magnitude dynamics) provide precise mathematical characterizations.
5. **Strong practicality**: LoRAM retains all efficiency advantages of LoRA (plug-and-play, zero overhead) while matching the performance of spectral initialization.

## Limitations & Future Work

1. **Magnitude is not optimal**: LoRAM emulates the magnitude of spectral initialization rather than seeking an optimal magnitude, and better scaling strategies may exist.
2. **Layer-wise heterogeneity**: Different layers may benefit from different scaling factors; jointly optimizing magnitude, learning rate, and rank remains an open problem.
3. **Incomplete theory of optimization dynamics**: The analysis is primarily based on linear approximations near fixed points and does not deeply address nonlinear training dynamics.
4. **Special case of LoRA-GA**: The tracking mode fails for LoRA-GA, suggesting that directional alignment does matter in certain settings and that the magnitude principle is not universally complete.
5. **Future directions**: Adaptive layer-wise magnitude scheduling; orthogonal combination with other PEFT methods (e.g., LoRA+, DoRA); validation on larger-scale models.

## Related Work & Insights

- **PiSSA (NeurIPS 2024)**: The seminal spectral initialization work; this paper demonstrates that its advantage stems from magnitude rather than direction.
- **RsLoRA**: Proposes $\alpha = \sqrt{r}$ scaling; this paper shows it is essentially equivalent to the learning rate adjustment in LoRA+.
- **LoRA+ (ICML 2024)**: Applies distinct learning rates to the $A$ and $B$ matrices; this paper provides a unified explanation from the magnitude perspective.
- **LoRA-GA (NeurIPS 2024)**: A data-driven initialization approach; this paper proves it maximizes the gradient magnitude of LoRA.
- **Insight**: The core of parameter-efficient fine-tuning may lie not in "directional alignment" but in "magnitude matching"—an insight potentially generalizable to other low-rank methods such as adapters and prefix tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The magnitude principle offers a compelling new perspective, though the conclusion carries some retrospective intuitiveness.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage across NLU, NLG, VLM, and image generation tasks; ablation design is elegant, with the tracking-mode experiment being particularly illuminating.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, though the notation is somewhat heavy, increasing reading overhead.
- Value: ⭐⭐⭐⭐⭐ — Provides an extremely simple yet effective baseline for the LoRA community, with potential to become the new default initialization scheme.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] Collapsing Taylor Mode Automatic Differentiation](collapsing_taylor_mode_automatic_differentiation.md)

</div>

<!-- RELATED:END -->
