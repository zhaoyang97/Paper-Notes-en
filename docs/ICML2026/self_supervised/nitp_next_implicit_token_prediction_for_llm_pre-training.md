---
title: >-
  [Paper Note] NITP: Next Implicit Token Prediction for LLM Pre-training
description: >-
  [ICML 2026][Self-Supervised Learning][NTP representation degeneration] NITP provides continuous representation space supervision for the final hidden state by using **shallow representations as implicit targets**. This s…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "NTP representation degeneration"
  - "implicit targets"
  - "shallow supervision"
  - "cosine similarity"
date: 2026-05-08
content_hash: c7c32ddf05f05211
---

# NITP: Next Implicit Token Prediction for LLM Pre-training

**Conference**: ICML 2026  
**arXiv**: [2605.24956](https://arxiv.org/abs/2605.24956)  
**Code**: To be confirmed  
**Area**: LLM Pre-training / Representation Learning  
**Keywords**: NTP representation degeneration, implicit targets, shallow supervision, cosine similarity

## TL;DR
NITP provides continuous representation space supervision for the final hidden state by using **shallow representations as implicit targets**. This supplements standard NTP to prevent hidden representation degeneration into low-dimensional anisotropic configurations. It achieves a 5.7% MMLU-Pro improvement on a 9B MoE model and a general 4-6% increase in reasoning tasks, with only ~2% extra computational overhead.

## Background & Motivation

**Background**: Standard Next Token Prediction (NTP) is the mainstream paradigm for LLM pre-training. NTP essentially provides discrete, one-hot supervision in the output logit space.

**Limitations of Prior Work**: Although gradients backpropagate through the output projection to hidden states, NTP objectives primarily constrain representations along the target logit direction, leaving many degrees of freedom weakly constrained in the latent space. This leads to **representation degeneration**, where likelihood-based training collapses learned representations into narrow anisotropic cones, severely limiting expressivity and correlating with degraded downstream performance.

**Key Challenge**: NTP defines "what to predict" but does not constrain "how to represent." Hidden states can adopt various geometrically distinct configurations but fall into representation degeneration in practice—sacrificing semantic richness for discriminative efficiency.

**Goal**: To address the blind spot of NTP in the geometry of hidden representations by using explicit representation-level supervision to guide hidden states toward structured, semantically rich configurations.

**Key Insight**: Instead of working in discrete token space, supervision is performed in continuous representation space by predicting the next token's implicit semantic representation (using the model's own shallow representations as self-supervised targets). Shallow layers are suitable because they preserve rich vocabulary and local semantic details.

**Core Idea**: NITP = NTP (discrete supervision) + NITP (continuous representation space supervision). It uses the next token representation from shallow layers as an implicit target, forcing final hidden states to align via cosine similarity loss. It is parameter-efficient as targets are derived from precomputed intermediate activations without extra forward passes.

## Method

### Overall Architecture
A dual-supervision mechanism is employed: (1) standard NTP $\mathcal{L}_{\text{NTP}}$; (2) NITP auxiliary objective $\mathcal{L}_{\text{NITP}} = 1 - \frac{\mathcal{P}(h_t)^\top z_{t+1}}{\|\mathcal{P}(h_t)\|_2 \cdot \|z_{t+1}\|_2}$; (3) joint optimization $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda \mathcal{L}_{\text{NITP}}$. During a standard forward pass, shallow layers (e.g., layer 4) compute the implicit target $z_{t+1}$ at position $t+1$ (with stop-gradient), which is then predicted from the final hidden state $h_t$ via a projection head.

### Key Designs

1. **Implicit Target Construction**:
    - **Function**: Provides context-dependent, semantically rich supervision for the final hidden state.
    - **Mechanism**: Uses the model's own shallow layer representation (layer 4, ~20% depth) at position $t+1$, $z_{t+1} = \text{sg}[E_{\text{shallow}}(x_{\leq t+1})^{(t+1)}]$, as the implicit target. A stop-gradient operator ensures stability (as shallow layers converge faster) without extra computation.
    - **Design Motivation**: The semantic richness in shallow layers forces deep representations to maintain sufficient expressive power to predict them, preventing anisotropic collapse.

2. **Cosine Similarity Loss**:
    - **Function**: Aligns predicted states with implicit targets in representation space.
    - **Mechanism**: Minimizes cosine similarity loss, where $\mathcal{P}(\cdot)$ is a simple MLP projection head. Cosine similarity is symmetric on $[-1, 1]$ and scale-insensitive.
    - **Design Motivation**: Ablations show cosine loss is more stable than MSE, Smooth-$\ell_1$, or KL divergence. MSE amplifies inter-layer scale mismatches due to quadratic penalties, while KL divergence introduces geometric distortion by treating vectors as distributions.

3. **Self-Supervised Design + Stop-Gradient**:
    - **Function**: Automatically generates supervision signals without external data or encoders while ensuring training stability.
    - **Mechanism**: Implicit targets use `sg` (stop-gradient), so gradients only flow to the final layer and projection head, not backpropagate to shallow layers. Shallow layers act as stable "semantic anchors."
    - **Design Motivation**: This reduces computational cost (~2% extra FLOPs), improves training stability, and remains fully self-supervised.

### Theoretical Analysis: Regulating the Semantic Manifold
Constraints on $h_t$ from the NTP objective primarily originate from its dot product with target token embeddings, leading to Hessian rank deficiency and allowing representation drift in the null space. NITP introduces **positive curvature** to regularize these directions: the NITP Hessian $H_{\text{NITP}}(h) \approx \frac{1}{r^2} P_{\perp u}$ (tangent space projection of the hypersphere). By introducing strictly positive curvature in all orthogonal directions, NITP forces representations to maintain structured geometry.

## Key Experimental Results

### Main Results

| Model | Method | MMLU | MMLU-Pro | C3 | CommonsenseQA | Avg Gain |
|------|------|------|---------|-----|---------|---------|
| 1.9B MoE (0.3B active) | NTP | 31.05 | 7.14 | 32.21 | 25.38 | — |
| 1.9B MoE | NITP | 31.68 | 7.47 | 29.69 | 26.61 | +0.8 |
| 3B MoE | NTP | 34.60 | 11.00 | 39.06 | 34.15 | — |
| 3B MoE | NITP | 37.37 | 12.29 | 44.38 | 37.92 | +2.1 |
| **9B MoE** | **NTP** | 43.71 | 15.29 | 56.65 | 45.70 | — |
| **9B MoE** | **NITP** | **46.14** | **21.00** | **63.01** | **49.96** | **+2.7** |

On the 9B model, MMLU-Pro shows an absolute gain of 5.7%, while reading comprehension and common sense reasoning increase by 6.4% and 4.3% respectively.

### Ablation Study

| Configuration | MMLU | MMLU-Pro | CommonsenseQA | BBH | Average |
|------|------|---------|----------|-----|------|
| Baseline NTP | 34.60 | 11.00 | 34.15 | 21.92 | 25.42 |
| **Shallow (L₄)** | **37.37** | **12.29** | **37.92** | **26.14** | **28.43** |
| Middle (L₈) | 35.33 | 11.57 | 34.72 | 22.07 | 25.92 |
| Deep (L₁₄) | 35.79 | 10.43 | 38.90 | 23.25 | 27.09 |
| Current position t→t | 33.09 | 8.14 | 29.15 | 20.96 | 22.84 |
| MSE Loss | 32.77 | 10.29 | 30.38 | 21.55 | 23.75 |
| Cosine Reg (No Pred) | 34.45 | 10.14 | 33.25 | 22.29 | 25.03 |

### Key Findings
- **Necessity of Shallow Layer Selection**: Using shallow layers (~20% depth) outperforms middle or deep layers, as shallow layers retain richer vocabulary and local semantics.
- **Temporal Structure is Crucial**: Predicting the next token's implicit representation ($t \to t+1$) outperforms same-position alignment ($t \to t$) by 5.6 percentage points.
- **Loss Function Stability**: MSE leads to gradient spikes and temporary divergence; only cosine similarity is fully stable and yields the best performance.
- **Regularization does not equal Prediction**: Generic cosine regularization constrains geometry but does not improve performance; gains derive from "predictive-aligned" semantic supervision.
- **Computational Efficiency**: Extra FLOPs are only ~2%, and $\lambda = 1.0$ proved to be the most robust weighting.

## Highlights & Insights
- **Diagnostic Root Cause of Representation Degeneration**: Visualizations of effective rank and cosine similarity demonstrate how NTP causes representations to drift toward low-dimensional anisotropic configurations; theoretical analysis explains this via the Hessian spectrum.
- **Clever Design of Self-Supervised Implicit Targets**: Using shallow representations as "semantic anchors" requires no external data or models and serves as an ideal supervision signal due to their rich semantic content.
- **Generality and Transferability**: NITP proved effective across MoE and dense models, scale ranges from 0.5B to 9B parameters, and multiple assessment benchmarks.
- **Significant Gains with Minimal Overhead**: A 5%+ improvement in knowledge understanding and 6%+ in reasoning capabilities were achieved at a cost of only ~2% extra training FLOPs.

## Limitations & Future Work
- NITP introduces additional hyperparameters (target layer, weight $\lambda$), and stability across different models requires further verification.
- The explanation for the total failure of current-position alignment needs deepening.
- Applicability to larger scales (> 100B), different architectures, and multimodal models remains to be validated.
- The selection of the 4th layer for the implicit target may not be optimal for all model depths.

## Related Work & Insights
- **vs Multi-token Prediction (MTP)**: MTP extends the prediction range in discrete token space, whereas NITP provides supervision in the representation space; the two are complementary.
- **vs Layer Distillation**: Distillation aligns representations between two different models, while NITP uses shallow layers to guide deep layers within the same model, avoiding external distribution shifts.
- **vs Self-Supervised Contrastive Learning (BYOL)**: Contrastive learning encourages consistency between different views; NITP focuses on prediction in the temporal dimension.
- **Insight**: Representation-level supervision is a viable direction for addressing the incompleteness of LLM pre-training objectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Supplementing NTP with shallow implicit targets and providing a theoretical Hessian-based explanation for degeneration is simple yet profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple model scales, two architectures, extensive ablations, and combines theory with empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Clear logic; theoretical sections are somewhat abstract but highlight key points effectively.
- Value: ⭐⭐⭐⭐⭐ Directly improves LLM pre-training efficiency and performance; high industrial value due to 5%+ gains for 2% cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)
- [\[ICLR 2026\] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](../../ICLR2026/self_supervised/snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)
- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)
- [\[ICML 2026\] LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel.md)

</div>

<!-- RELATED:END -->
