---
title: >-
  [Paper Note] Torus Graphs for Large-Scale Neural Phase Analysis
description: >-
  [ICML 2026][torus graph] The authors reduce the inference complexity of the Torus Graph (TG)—an exponential family phase model on the $d$-torus $\mathbb{T}^d$—from $\mathcal{O}(d^6)$ to $\mathcal{O}(d^2)$ using stochasti…
tags:
  - "ICML 2026"
  - "torus graph"
  - "score matching"
  - "phase coupling"
  - "hidden Markov model"
  - "transfer entropy"
date: 2026-05-08
content_hash: ddf4240b93de77a4
---

# Torus Graphs for Large-Scale Neural Phase Analysis

**Conference**: ICML 2026  
**arXiv**: [2606.00496](https://arxiv.org/abs/2606.00496)  
**Code**: https://github.com/jackgoffinet/torus-graphs  
**Area**: Neuroscience / Probabilistic Graphical Models / Circular Statistics  
**Keywords**: torus graph, score matching, phase coupling, hidden Markov model, transfer entropy

## TL;DR
The authors reduce the inference complexity of the Torus Graph (TG)—an exponential family phase model on the $d$-torus $\mathbb{T}^d$—from $\mathcal{O}(d^6)$ to $\mathcal{O}(d^2)$ using stochastic score matching. This enables the first application to thousands of phase variables. They further develop TG-HMM and autoregressive TG extensions, applying them to mouse LFP data to reveal frequency-specific phase reconfiguration during Wake-NREM transitions.

## Background & Motivation

**Background**: EEG/LFP recordings are typically described as a superposition of multiple oscillatory components, each with a continuous phase. Phase relationships are considered core computational variables for communication between brain regions. However, mainstream phase analysis remains limited to pairwise metrics such as Phase Locking Value (PLV): $PLV_{X,Y}=|\mathbb{E}\,e^{i(X-Y)}|$. The Torus Graph, proposed by Klein et al. (2020), is an exponential family model for circular variables where univariate and pairwise potential functions generalize the von Mises distribution. It allows for conditional independence inference, distinguishing "direct coupling" from "spurious coupling mediated by others."

**Limitations of Prior Work**: The normalization constant of the TG is analytically intractable, necessitating score matching for inference. The closed-form solution requires solving a $2d^2\times 2d^2$ linear system and storing $\Gamma\in\mathbb{R}^{2d^2\times 2d^2}$, leading to $\mathcal{O}(d^6)$ time and $\mathcal{O}(d^4)$ memory complexity. Empirically, this fails at $d\approx 100$ on a 24GB GPU. Modern LFP/EEG experiments, however, involve $d=O(10^3)$ phase variables (dozens of channels × dozens of frequency bins).

**Key Challenge**: Pairwise metrics (PLV, coherence) are computationally efficient but cannot distinguish "direct vs. indirect" relations; TG provides this distinction but is computationally prohibitive. Researchers are forced to revert to pairwise analysis for high-dimensional data, losing conditional independence information. Models like Kuramoto or Granger typically model amplitudes or linear Gaussian structures, which are unsuitable for the circular geometry of pure phase variables.

**Goal**: (i) Reduce TG inference complexity to $\mathcal{O}(d^2)$ per step; (ii) develop a dynamic version to capture "temporal state transitions"; (iii) provide an autoregressive version for "directionality" inference and transfer entropy estimation for phase variables.

**Key Insight**: Each term in the sufficient statistics $S(\mathbf{x})$ of a TG depends on at most two phase variables. Thus, although the Jacobian $\nabla_{\mathbf{x}}S(\mathbf{x})$ is formally $\mathcal{O}(d^3)$, it is sparse with only $\Theta(d^2)$ non-zero elements. Consequently, $\bm{\phi}^\top\nabla_{\mathbf{x}}S(\mathbf{x})$ can be computed directly in $\mathcal{O}(d^2)$ time using the vector-Jacobian product (VJP) in reverse-mode automatic differentiation without explicitly constructing the Jacobian.

**Core Idea**: Rewrite the TG score matching objective into a stochastic optimization form relying only on VJP. Combined with Adam, this allows unbiased inference on thousands of phase variables. By layering HMM and autoregressive structures, the authors derive the first family of phase graphical models scalable to thousands of dimensions.

## Method

### Overall Architecture
The approach consists of three layers: (1) Static TG utilizing stochastic score matching; (2) a dynamic extension, TG-HMM, using EM with a discriminative M-step to bypass the log-partition function; (3) a directed extension, AR-TG, which embeds historical phases into TG parameters via $\psi(\theta)=[\cos\theta;\sin\theta]^\top$ and estimates transfer entropy by comparing predictions from two AR-TGs.

The TG density is defined as $p(\mathbf{x};\bm{\phi})\propto\exp(\bm{\phi}^\top S(\mathbf{x}))$, where $S(\mathbf{x})$ includes univariate terms $\cos x_j, \sin x_j$ and pairwise sum/difference terms $\cos(x_j\pm x_k), \sin(x_j\pm x_k)$, with parameter dimension $2d^2$. The implementation uses JAX and runs end-to-end on a single A5000 24GB GPU.

### Key Designs

1. **Stochastic Score Matching for TGs (VJP-based Score Matching)**:
    - **Function**: Converts the TG score matching objective into a stochastic objective optimized via minibatch Adam, reducing per-step complexity from $\mathcal{O}(d^6)$ to $\mathcal{O}(d^2)$ and memory from $\mathcal{O}(d^4)$ to $\mathcal{O}(d^2)$.
    - **Mechanism**: The original objective $J(\bm{\phi})=\mathbb{E}_{\mathbf{x}}[\tfrac{1}{2}\bm{\phi}^\top\Gamma(\mathbf{x})\bm{\phi}-\bm{\phi}^\top\mathbf{h}(\mathbf{x})]$ involves a large matrix $\Gamma=\nabla_{\mathbf{x}}S(\nabla_{\mathbf{x}}S)^\top$. By rewriting $\bm{\phi}^\top\Gamma(\mathbf{x})\bm{\phi}=\|\bm{\phi}^\top\nabla_{\mathbf{x}}S(\mathbf{x})\|_2^2$, it becomes a norm that can be computed in $\mathcal{O}(d^2)$ via a single VJP (backpropagating through the scalar $\bm{\phi}^\top S(\mathbf{x})$). The rewritten objective $J(\bm{\phi})=\mathbb{E}_{\mathbf{x}}\big[\tfrac{1}{2}\|\bm{\phi}^\top\nabla_{\mathbf{x}}S(\mathbf{x})\|_2^2-\bm{\phi}^\top\mathbf{h}(\mathbf{x})\big]$ is unbiased and compatible with $L_2$ and group-$\ell_1$ regularization. This logic extends to conditional TGs where $\bm{\phi}(y)$ can be parameterized by a neural network.
    - **Design Motivation**: The sparsity of TG is a structural fact (each statistic observes two variables), which the original closed-form score matching ignored. Using VJP to "probe" the sparse Jacobian-vector product removes the methodological bottleneck.

2. **Discriminative M-step for TG-HMM**:
    - **Function**: Enables TG to transition dynamically between latent states $z_t\in\{1,\dots,K\}$ to capture state-dependent phase coupling (e.g., spindles in NREM sleep), while sidestepping the intractable log-partition function $A(\bm{\phi}_k)$.
    - **Mechanism**: The emission model is $p(x_t|z_t=k)=\exp(\bm{\phi}_k^\top S(x_t)-A(\bm{\phi}_k))$. Instead of calculating $A(\bm{\phi}_k)$, free parameters $A_k\in \mathbb{R}$ and ridge regularization are introduced to construct a surrogate joint model $\log\tilde{p}(z,x)=\sum_t\log\Pi_{z_{t-1},z_t}+\sum_t[\bm{\phi}_{z_t}^\top S(x_t)-A_{z_t}]$. The E-step uses standard forward-backward on the surrogate. The M-step treats $A_k$ as trainable categorical intercepts, where the objective $Q'(A)$ is equivalent to multinomial logistic regression with soft labels $\gamma_{t,k}$, features $S(x_t)$, and fixed weights $\{\bm{\phi}_k\}$.
    - **Design Motivation**: Direct estimation of partition constants (NCE/MCMC) introduces noise and hyperparameters. The discriminative perspective reduces "constant estimation" to a softmax fit, which is computationally efficient and integrates seamlessly with forward-backward.

3. **AR-TG and Transfer Entropy (Directed Phase Interaction)**:
    - **Function**: Extends TG to an autoregressive form $p(y_t|\mathbf{x}_{<t},y_{<t})\propto\exp[\bm{\phi}(\mathbf{x}_{<t},y_{<t})^\top S(y_t)]$. This enables estimation of transfer entropy $TE_{X\to Y}=\mathbb{H}(Y_t|Y_{<t})-\mathbb{H}(Y_t|Y_{<t},X_{<t})$ to infer directional coupling.
    - **Mechanism**: Parameterize $\bm{\phi}(\mathbf{x}_{<t},y_{<t})=\mathbf{b}+\sum_{\ell=1}^L\big(\mathbf{W}^{(y)}_\ell\psi(y_{t-\ell})+\mathbf{W}^{(x)}_\ell\psi(\mathbf{x}_{t-\ell})\big)$, where $\psi(\theta)=[\cos\theta;\sin\theta]^\top$ embeds phases into $\mathbb{R}^2$ to ensure periodicity while keeping parameters $\mathcal{O}(L)$. TE is estimated by fitting two AR-TGs: $\hat{p}_1(y_t|y_{<t})$ and $\hat{p}_2(y_t|y_{<t},\mathbf{x}_{<t})$, then calculating the difference in log-likelihood on a test set.
    - **Design Motivation**: Directional inference for phase variables is difficult as standard Granger causality uses linear Gaussian assumptions that break periodicity. Using $\psi$ embedding with univariate von Mises conditions (which have tractable $A$) preserves geometry while allowing analytical log-likelihood evaluation during testing.

### Loss & Training
All components are implemented in JAX. Static TGs use stochastic score matching with Adam. TG-HMM uses alternating forward-backward and discriminative M-steps (logistic). AR-TG parameters are estimated via score matching, and TE is evaluated on independent test sets.

## Key Experimental Results

### Main Results
The authors validate parameter recovery on 4-dim and 64-dim synthetic TGs, then perform large-scale visualization and state discovery on mouse LFP data with $d=1860$.

| Dimension $d$ | Inference Method | Complexity / Step | Max Scale | Recovery $R^2$ |
|---------|---------|----------------|-----------|---------------|
| 4 | exact score matching | $\mathcal{O}(d^6)$ | OK | Same as stochastic |
| 64 | exact | $\mathcal{O}(d^6)$ | OK, but slow | Same as stochastic |
| $\sim$100 | exact | $\mathcal{O}(d^6)$ | 24GB GPU OOM | — |
| $\sim$1000+ | **stochastic (Ours)** | **$\mathcal{O}(d^2)$** | **OK** | Matches exact at low dim |
| 1860 | stochastic (Ours) | $\mathcal{O}(d^2)$ | Real LFP data | Reveals frequency reconfiguration |

### Ablation Study
| Configuration | Key Observation | Description |
|------|---------|------|
| Full TG-HMM | Stable extraction of 6 states across 1334 spindles | Discriminative M-step does not hinder recovery |
| TG-HMM (exact) | Accurate at $d\lesssim 100$, OOM at >100 | Not scalable, but validates accuracy at low dim |
| AR-TG vs Multi. Granger | Granger times out after 30h at $d=64$, AR-TG accurate in <1h | Significant causal discovery advantage |
| AR-TG (SM) vs AR-TG (MLE) | Score matching is more stable for bidirectional TE | Differences in partition function handling |

### Key Findings
- Reducing inference bottleneck from $\mathcal{O}(d^6)$ to $\mathcal{O}(d^2)$ is a major shift: on the same hardware, capacity jumps from $\sim$100 to $\sim$1860 variables while being an order of magnitude faster.
- Application to 48h mouse LFP (62 channels × 30 bins = 1860 dims) shows stronger high-frequency (>30 Hz) coupling during Wake and stronger low-frequency (<30 Hz) during NREM, consistent with sleep physiology.
- TG parameters are significantly sparser than empirical PLVs, proving many pairwise synchronies are spurious edges mediated by third parties.
- TG-HMM identifies a "spatially sparse spindle state," contrasting with the "diffuse synchrony increase" shown by PLV, highlighting the benefit of conditional independence.
- AR-TG transfer entropy reveals asymmetric interactions (e.g., prelimbic→striatum, VTA→SNr) that are invisible to PLV/coherence.

## Highlights & Insights
- **Sparsity + VJP as a scaling recipe**: TG's $\Gamma$ has $\mathcal{O}(d^4)$ elements, but physically each statistic only involves two variables. VJP directly exploits this. This pattern is transferable to other exponential family PGMs like discrete MRFs.
- **Discriminative M-step replaces NCE/MCMC**: Using logistic regression to fit log-partition constants $A_k$ transforms intractable constants into learnable offsets, a useful trick for any latent state model with intractable partitions.
- **Geometric consistency**: The $\psi(\theta)$ 2D embedding preserves periodicity while maintaining analytical tractability for von Mises conditions. This explains why the TG family can support sparse inference, dynamic switching, and directed interaction where linear Gaussian models fail.

## Limitations & Future Work
- AR-TG transfer entropy currently requires a univariate target $y_t$ due to dependency on the von Mises partition function; multivariate targets are not yet supported.
- The discriminative M-step for TG-HMM is only "approximately consistent"; formal statistical convergence rates under model misspecification require further study.
- Interpreting TG parameters can be complex for neuroscientists when cross-frequency and within-frequency couplings are mixed; authors suggest restricting the model to within-frequency when cross-frequency is not the focus.
- Like all Granger-like methods, AR-TG directionality is predictive rather than interventional.

## Related Work & Insights
- **vs PLV / coherence**: These are pairwise and cannot eliminate indirect edges; this work provides a scalable conditional independence alternative.
- **vs Klein et al. (2020)**: Original work used closed-form score matching capped at 100 dimensions; this work pushes to 1860 and adds dynamic/directed extensions.
- **vs Kuramoto**: Kuramoto is dynamical and great for large-scale synchrony but is not a probabilistic model for conditional independence inference; TGs provide a complementary statistical approach.
- **vs Multivariate Granger**: Granger violates phase periodicity via linear Gaussian assumptions and times out at $d>64$; AR-TG recovers causal edges reliably at larger scales.

## Rating
- Novelty: ⭐⭐⭐⭐ Stochastic score matching is applied effectively to TG for the first time, integrated with HMM/AR extensions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic recovery and 1860-dim real LFP data with extensive baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation of statistical inference and neuroscience applications.
- Value: ⭐⭐⭐⭐ Provides a scalable statistical toolstack for phase analysis in neuroscience with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](amdp_asynchronous_multi-directional_pipeline_parallelism_for_large-scale_models_.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/others/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2026/others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)
- [\[AAAI 2026\] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](../../AAAI2026/others/intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)

</div>

<!-- RELATED:END -->
