---
title: >-
  [Paper Note] Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts
description: >-
  [ICML 2026][Learning Theory][Sparse MoE] This paper presents the first rigorous geometric and stochastic analysis of the "input-output mapping discontinuity" caused by Top-$k$ routing in Sparse Mixture-of-Experts (SMoE). By classifying discontinuity surfaces based on the "number of tied experts," the authors prove that order-1 surfaces occupy almost all "near-discontinuity" volume, while higher-order volumes are negligible. Using diffusion processes…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "Sparse Mixture-of-Experts"
  - "Routing Continuity"
  - "Sparse MoE"
  - "Top-k Routing"
  - "Discontinuity Analysis"
  - "Measure Theory"
  - "Diffusion Processes"
  - "Local Smoothing"
date: 2026-05-08
content_hash: 0777fdcfa6b7f072
---

# Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2606.19036](https://arxiv.org/abs/2606.19036)  
**Code**: https://github.com/thotranhuu99/Smooth_SMoE  
**Area**: Learning Theory / Sparse Mixture-of-Experts / Routing Continuity  
**Keywords**: Sparse MoE, Top-k Routing, Discontinuity Analysis, Measure Theory, Diffusion Processes, Local Smoothing

## TL;DR
This paper presents the first rigorous geometric and stochastic analysis of the "input-output mapping discontinuity" caused by Top-$k$ routing in Sparse Mixture-of-Experts (SMoE). By classifying discontinuity surfaces based on the "number of tied experts," the authors prove that order-1 surfaces occupy almost all "near-discontinuity" volume, while higher-order volumes are negligible. Using diffusion processes, they demonstrate that random perturbations almost surely hit order-1 surfaces first. Based on these insights, a plug-and-play $\ell_\infty$ local smoothing mechanism, SmoothSMoE, is proposed. It restores continuity to the SMoE mapping and improves performance in language and vision tasks with near-zero additional computational overhead.

## Background & Motivation
**Background**: SMoE replaces the FFN layers of a Transformer with sparsely activated expert modules. By using Top-$k$ gating to activate only $k$ experts, it scales model capacity while maintaining constant computational costs. This architecture is widely used in Large Language Models (e.g., DeepSeek, Switch Transformer) and vision models.

**Limitations of Prior Work**: The sparsity of Top-$k$ gating is achieved through "hard selection," which inherently makes the SMoE input-output mapping **discontinuous**. Two nearly identical inputs falling near a routing boundary may be assigned to completely different sets of experts, causing a sharp jump in output. This poses risks to robustness and adversarial stability. Existing works either merely acknowledge this discontinuity without systematic analysis or avoid it using differentiable routing (e.g., SMEAR merges experts, Soft MoE mixes across tokens), which breaks the causal structure required for autoregressive generation. Alternatives like ReMoE use ReLU gating but require retraining and expensive initialization.

**Key Challenge**: Sparsity (computational efficiency) and continuity (stability) are seemingly contradictory under Top-$k$ gating. Retaining Top-$k$ sparsity and causality requires facing its discontinuities directly, while existing solutions to eliminate them often sacrifice causality or incur high retraining costs. Fundamentally, **prior work has not quantitatively answered** what these discontinuity surfaces look like, how large they are, or if random perturbations hit them.

**Goal**: To rigorously characterize SMoE discontinuities from two perspectives: geometrically (frequency of different tie patterns and the volume of their boundaries) and stochastically (whether random perturbation trajectories hit boundaries and which order they hit). The goal is to translate these theoretical insights into a low-cost, causal-preserving smoothing mechanism.

**Key Insight**: By expressing gating scores as affine functions, Top-$k$ routing partitions the input space into open cells where the active expert set is fixed. Discontinuity surfaces are the boundaries of these cells (where ties in scores occur). By grading boundaries (order-$n$) based on the "number of simultaneously tied experts," measure-theoretic slice arguments can estimate the volume of each order, and random perturbations can be modeled as diffusion processes to calculate hitting/residence times.

**Core Idea**: Use "Geometric Measure (volume estimation) + Stochastic Processes (hitting/residence time)" to prove that "low-order discontinuities dominate and inputs are more likely to fall near order-1 surfaces." Consequently, continuity can be restored by local softening only within a narrow $\ell_\infty$ boundary zone—achieving significant impact with minimal cost.

## Method

### Overall Architecture
The paper follows a "theory first, application second" structure. **Analysis side**: It decomposes the discontinuity set $\Gamma$ into order-1, order-2, etc., based on ties. A slice argument provides an asymptotic upper bound for the volume of the $\epsilon$-thickening of each order, proving higher orders are negligible. Random input perturbations are modeled as Brownian diffusion, proving trajectories almost surely hit the boundary in finite time, specifically hitting an order-1 surface first. **Application side**: Given that inputs most likely reside near low-order surfaces, the non-top-$k$ logits that "just missed the cut" within a narrow $\ell_\infty$ band are softly lifted (log-smoothstep) to ensure continuity. A boundary loss adaptively tunes the band width $\epsilon$ based on an "average of 0.5 additional experts" budget.

### Key Designs

**1. Categorizing Discontinuities by "Number of Tied Experts"**

Discontinuity arises from gating ties. Gating scores are affine: $z_i(x)=\langle W_g^{(i)},x\rangle+b_g^{(i)}$. For each $k$-subset $\mathbb{S}$, an open cell is defined as $\mathcal{C}_{\mathbb{S}}=\{x: z_i(x)>z_j(x),\ \forall i\in\mathbb{S}, j\notin\mathbb{S}\}$. The SMoE mapping is smooth within these cells. Their complement $\Gamma$ is the discontinuity set. An **order-$n$ discontinuity point** occurs when $n+1$ scores tie for the $k$-th and $(k+1)$-th positions:

$$z_{i_1}(x)=\cdots=z_{i_{n+1}}(x)=z_{[k]}(x)=z_{[k+1]}(x)$$

Stacking these $n$ independent equations results in a linear system $A_J x=d_J$, placing the order-$n$ surface on an affine plane $S_J^{(n)}$ of **codimension $n$** (i.e., $D-n$ dimensions). Order-1 is the simplest case (tie between $k$-th and $(k+1)$-th). This "order-based classification" forms the backbone of all subsequent volume and time estimates.

**2. Asymptotic Volume Estimates: Low-Order Dominance**

While $\Gamma$ itself is measure zero, its "near-discontinuity" neighborhood can be large. The authors quantify the volume of the $\epsilon$-thickening $T_\epsilon(\Gamma^{(n)})=\{x:\mathrm{dist}(x,\Gamma^{(n)})<\epsilon\}$. A core theorem proves that within a ball of radius $R$:

$$\frac{\lambda^D(T_\epsilon(\Gamma^{(n)})\cap B^D(0,R))}{\lambda^D(B^D(0,R))}\ \lesssim\ \Big(\frac{\epsilon}{R}\Big)^{n}$$

Intuition: Order-$n$ surfaces lie on codimension-$n$ planes; the normal direction contributes $\epsilon^n$ volume while the tangential direction contributes $R^{D-n}$. The **ratio between orders** $U_n(R)/U_m(R)\sim(\epsilon/R)^{n-m}$ implies that as $R$ increases, higher-order volumes become negligible—**the vast majority of "near-discontinuous" inputs are near order-1 surfaces**. The authors also show $\ell_\infty$ thickening follows the same decay rates but is easier to compute in logit space.

**3. Hitting and Residence Times of Stochastic Perturbations**

The paper models perturbations as Itô diffusion $dx_t=\sigma\,dB_t$ starting from an open cell $\mathcal{C}_{\mathbb{S}}$. Two conclusions: (i) **Hitting Time**: Trajectories almost surely hit the boundary in finite time, and the **first hit is almost surely on an order-1 surface**. (ii) **Residence Time**: The expected time spent in an $\epsilon$-neighborhood of an order-$n$ surface scales with $\epsilon^n$. This physical picture confirms that **random trajectories spend most of their time near low-order surfaces**, justifying smoothing only these narrow bands.

**4. SmoothSMoE: $\ell_\infty$ Local Smoothing + Adaptive Boundary Loss**

Smoothing is applied only to non-top-$k$ logits satisfying $0<z_{[k]}(x)-z_i(x)<\epsilon$. A log-smoothstep $h(u)$ defines the smoothing coefficient $m_i(x)=h((z_i(x)-z_{[k]}(x)+\epsilon)/\epsilon)$, and the smoothed logit becomes $\hat z_i(x)=z_i(x)+m_i(x)$. Within an order-$n$ $\ell_\infty$ thickening, at most $n$ additional experts are activated. Due to rapid decay of higher-order volumes, the average number of extra experts is minimal, **preserving causal structure** and keeping compute overhead low. A boundary loss $\mathcal{L}_{\text{boundary}}=\alpha\,\epsilon\,(\mathcal{K}-k^*)$ adaptively adjusts $\epsilon$, where $\mathcal{K}$ is the current average active experts and $k^* = k + 0.5$.

## Key Experimental Results

### Main Results
SmoothSMoE was applied to existing SMoE/GMoE models across language (WikiText-103), NLU (GLUE), and vision (DomainBed) tasks.

| Task / Metric | SMoE Baseline | ReMoE | SmoothSMoE (Ours) |
|------|------|------|------|
| WikiText-103 Test PPL ↓ | 35.52 | 35.35 | **34.35** |
| Attacked WikiText-103 Test PPL ↓ | 44.18 | 44.00 | **42.85** |
| EnWiki-8 BPC ↓ | 1.153 | — | **1.122** |
| GLUE Avg (K=16, k=2) ↑ | 81.17 | 81.18 | **81.65** |
| GLUE Avg (K=16, k=4) ↑ | 81.14 | 81.16 | **81.73** |
| DomainBed Avg Acc ↑ | GMoE Baseline | — | **+0.56% (vs GMoE)** |

SmoothSMoE consistently outperformed Top-$k$ SMoE and ReMoE. It achieved a 1.17 drop in WikiText-103 test PPL and a 0.57% gain on GLUE (k=4).

### Ablation Study / Analysis

| Experiment | Observation |
|------|------|
| SMoE Output at Boundary | SMoE shows "jumps" in output for tiny input changes |
| SmoothSMoE Output at Boundary | Jumps disappear; mapping is continuous |
| Max Output Diff vs. $\|\Delta x\|$ | In SMoE, diff does not vanish as $\|\Delta x\|\to0$; in Ours, it does |
| Boundary Loss Effectiveness | Successfully maintains average active experts near $k^*$ budget |

### Key Findings
- **Theory Validated by Experiment**: Jumps occur only at boundaries and vanish after smoothing; the max output difference tends to zero with perturbation size, confirming the "order-1 dominance + local smoothing" theory.
- **Robustness in Vision**: On DomainBed, gains were more consistent on larger datasets, interpreted as such data being more likely to have inputs "near the tie boundary."
- **Zero Compute Overhead**: With small $\epsilon$, the average number of active experts is barely above $k$, delivering continuity and performance gains essentially for free.

## Highlights & Insights
- **Quantifying the Discontinuous**: Using "order-based classification + slice measures," the authors turn abstract discontinuities into geometric objects with asymptotic volume formulas—a major step in MoE theory.
- **Geometric + Stochastic Synergy**: Volume estimates show *where* discontinuities are frequent, while diffusion analysis shows *how* perturbations interact with them.
- **Theory-Driven Engineering**: $\ell_\infty$ thickening is chosen as the smoothing criterion specifically because it is efficient in logit space while maintaining necessary decay rates.

## Limitations & Future Work
- The analysis assumes **affine** gating scores; non-linear or normalized gating (e.g., Softmax before Top-$k$) requires further study.
- Random perturbations are modeled as isotropic constant-coefficient Brownian diffusion, which simplifies real-world adversarial attacks or training dynamics.
- Experimental scale is medium (Switch Transformer 16 experts, BERT-large, ViT-S/16). Performance in extremely large MoEs (100B+ parameters) remains to be verified.
- Performance gains are modest (~1 PPL or <1% accuracy), suggesting value lies more in stability and theoretical foundation than massive SOTA breakthroughs.

## Related Work & Insights
- **vs. SMEAR / Soft MoE**: These eliminate hard switches but break causality. Ours performs local smoothing to preserve Top-$k$ causality.
- **vs. ReMoE**: ReMoE requires retraining and expensive initialization. SmoothSMoE is plug-and-play and outperforms ReMoE empirically.
- **vs. Prior Analytical Work**: Previous works only noted the existence of discontinuities; this paper provides the first systematic theory of their structure (orders), volume (asymptotic measures), and stochastic behavior.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorous geometric/stochastic characterization of SMoE discontinuities.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across NLP and Vision, though scales are moderate.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from definitions to theorems to method.
- Value: ⭐⭐⭐⭐ Fills a theoretical gap and provides a causal-preserving, low-cost smoothing mechanism.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Conformal Prediction via Mixture-of-Experts Gating Similarity](../../ICLR2026/learning_theory/adaptive_conformal_prediction_via_mixture-of-experts_gating_similarity.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](robustness_of_mixtures_of_experts_to_feature_noise.md)
- [\[ICLR 2026\] High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification](../../ICLR2026/learning_theory/high-dimensional_analysis_of_single-layer_attention_for_sparse-token_classificat.md)
- [\[ICLR 2026\] Finite-Time Convergence Analysis of ODE-based Generative Models for Stochastic Interpolants](../../ICLR2026/learning_theory/finite-time_convergence_analysis_of_ode-based_generative_models_for_stochastic_i.md)
- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](../../ICLR2026/learning_theory/a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)

</div>

<!-- RELATED:END -->
