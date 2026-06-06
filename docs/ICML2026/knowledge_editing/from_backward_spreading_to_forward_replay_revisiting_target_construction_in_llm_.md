---
title: >-
  [Paper Note] From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing
description: >-
  [ICML 2026][Knowledge Editing][Parameter Editing] This paper systematically analyzes why backward spreading works and its inherent limitations in locate-then-edit. It proposes forward replay: treating the first decisive…
tags:
  - "ICML 2026"
  - "Knowledge Editing"
  - "Parameter Editing"
  - "MEMIT"
  - "Multi-layer Synergy"
  - "Forward Propagation"
  - "Hidden State"
date: 2026-05-08
content_hash: d0eb22878f55f5a9
---

# From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing

**Conference**: ICML 2026  
**arXiv**: [2605.00358](https://arxiv.org/abs/2605.00358)  
**Code**: https://github.com/jugechengzi/FE (Available)  
**Area**: LLM Knowledge Editing / Locate-then-edit  
**Keywords**: Parameter Editing, MEMIT, Multi-layer Synergy, Forward Propagation, Hidden State

## TL;DR
This paper systematically analyzes why backward spreading works and its inherent limitations in locate-then-edit. It proposes forward replay: treating the first decisive layer as an optimization variable and deriving subsequent layer targets via standard forward propagation. This achieves consistent performance gains across MEMIT, RECT, PRUNE, and AlphaEdit without additional computational cost.

## Background & Motivation

**Background**: The locate-then-edit (LTE) paradigm, represented by ROME and MEMIT, has become mainstream for LLM knowledge editing. It first uses causal tracing to locate key tokens and decisive MLP layers, calculates the ideal hidden state $m_L$ for the last layer, and then solves a closed-form rank-one update. To prevent single-layer modifications from being suppressed by regularization, mainstream approaches use backward spreading to linearly interpolate $m_L$ and distribute it to earlier layers (e.g., $m_1 = h_1 + (m_3-h_3)/3$).

**Limitations of Prior Work**: The rationality of backward spreading has lacked systematic verification. It implicitly assumes the residual direction of the last layer is applicable to all earlier layers, but forward dynamics are non-linear, causing directions to rotate across layers. Recent methods like BLUE attempt independent optimization for the first and last layers, doubling computational cost while risking incompatibility between high-dimensional targets.

**Key Challenge**: Closed-form solutions for a single layer rely on an accurate $m_l$, while multi-layer synergy requires compatibility among various $m_l$. Backward spreading uses a single direction to cover all layers, while independent optimization breaks compatibility. The root of the problem is that **target propagation directions are inconsistent with the model's true forward dynamics**.

**Goal**: (1) Formally characterize when backward spreading succeeds or fails; (2) Propose a multi-layer target construction method naturally consistent with forward dynamics without increasing computational overhead.

**Key Insight**: When treating $m_L$ as an optimizable parameter for gradient descent on the final layer, the backward computation graph already implicitly contains information on "where the preceding layers should go" to reach the goal. Thus, by optimizing at the **first decisive layer** and performing a forward propagation replay, this information can be explicitly recovered.

**Core Idea**: Replace backward spreading with forward replay—moving the anchor point from the last layer to the first. Targets for other layers emerge naturally through standard forward propagation, maintaining complexity while ensuring inherent cross-layer compatibility.

## Method

### Overall Architecture
Let the decisive layers to be edited be $W_1,\dots,W_L$, the hidden state of the key token before $W_l$ be $k_l$, and the target hidden state be $m_l$. The closed-form solution $\Delta W^* = (M_I - W K_I) K_I^\top (K_I K_I^\top + \lambda K_J K_J^\top)^{-1}$ for LTE depends on $m_l$ for each layer. Whereas MEMIT computes only $m_L$ and derives other $m_l$ via linear interpolation, the proposed method, FE (Forward propagation Edit), calculates only $m_1$ and obtains subsequent targets via forward replay. The overall pipeline is identical, with only the target construction method changed.

### Key Designs

1. **Theoretical Characterization of Backward Spreading**:
    - **Function**: Explains why backward spreading works and where it fails, providing motivation for the proposed alternative.
    - **Mechanism**: Let $\delta m_l = \Delta W_l k_l$ be the output perturbation of layer $l$. Let $\delta^l_{m_L} = J_{l\to L} \delta m_l$ be the passive change caused at the last layer. Ideally, $\delta^l_{m_L} = \beta \delta m_l$, which is equivalent to $\delta m_l$ being an eigenvector of the Jacobian $J_{l\to L}$ (Theorem 1, a strong condition rarely met). A relaxed condition requires only $\cos(\delta^l_{m_L}, \delta m_l) > 0$, equivalent to the symmetrized Jacobian $\frac{1}{2}(J_{l\to L} + J_{l\to L}^\top)$ being positive definite (Theorem 2, empirically true in most cases for deep networks). Measurements on Llama3-8B show the cosine similarity from layer 4 to 8 increases from $0.34$ to $1.00$, indicating that distant layer influences deviate at the final layer, which is the fundamental reason backward spreading struggles with excessive layers.
    - **Design Motivation**: Establishing why the old method is "partially effective" before designing an alternative provides a more rigorous argument.

2. **Forward Replay for Multi-layer Target Construction**:
    - **Function**: Obtains mutually compatible target hidden states for all decisive layers using a single forward pass.
    - **Mechanism**: MEMIT treats $h_L$ as an optimizable parameter to minimize cross-entropy for $m_L$. FE reverses this: it treats the **first layer** $h_1$ as the optimizable parameter for gradient descent to obtain $m_1$. Then, $m_1$ is injected back into the model for standard forward propagation, recording the hidden states at each decisive layer as the corresponding $m_l$. These $\{m_l\}$ values originate from the same forward trajectory and are naturally compatible. Since the gradient path traverses the entire network, $m_1$ itself implicitly incorporates downstream dynamical constraints.
    - **Design Motivation**: Backward spreading "forces" a last-layer direction onto preceding layers; forward replay allows the first layer to "know what the last layer wants" and lets the network calculate subsequent layer states naturally.

3. **Plug-and-play Compatibility**:
    - **Function**: Enables FE to provide consistent improvements for a wide range of LTE-based methods beyond MEMIT.
    - **Mechanism**: FE only replaces the target propagation mechanism without altering closed-form solutions, initial $m$ optimization, or regularization terms. Consequently, variants like RECT (norm regularization), PRUNE (singular value pruning), and AlphaEdit (null-space projection) can directly adopt the "+FE" suffix.
    - **Design Motivation**: In a fragmented model editing community, improvements that orthogonally stack with existing pipelines have higher impact.

### Loss & Training
No new loss functions are introduced. Cross-entropy $H(Y, Y_{\text{target}})$ is used when optimizing $m_1$. Once $m_1$ is obtained, it serves as the hidden state for the first decisive layer, and a forward pass yields $\{m_l\}_{l=2}^L$. Finally, the standard rank-one closed-form solution is used to update parameters layer by layer. Computational complexity is identical to MEMIT.

## Key Experimental Results

### Main Results
Using Llama3-8B-Instruct with default decisive layers $[4,5,6,7,8]$ via EasyEdit. 2000 knowledge edits performed for MCF and ZsRE in batch editing mode.

| Dataset | Method | Efficacy Success ↑ | Generalization Acc ↑ | DKL ↓ | Top-1 ↑ |
|--------|------|--------------------|------------------------|-------|---------|
| MCF | MEMIT | 97.4 | 48.0 | 0.41 | 77.5 |
| MCF | BLUE (2-layer indep. opt.) | 98.4 | 60.1 | 0.35 | 81.0 |
| MCF | **MEMIT+FE** (Ours) | **99.9** | **61.0** | **0.34** | **82.7** |
| ZsRE | MEMIT | 92.6 | 72.6 | 0.60 | 45.3 |
| ZsRE | BLUE | 94.6 | 78.2 | 0.18 | 66.7 |
| ZsRE | **MEMIT+FE** (Ours) | **97.6** | **84.3** | **0.09** | **75.6** |

### Ablation Study

| Configuration | MCF Efficacy Success | Description |
|------|-----------------------|------|
| OneLayer (Modifying last layer only) | 76.2 | Single layer is insufficient; multi-layer synergy is necessary |
| MEMIT (backward, dividing) | 97.4 | Standard baseline |
| MEMIT (backward, no-dividing) | — | Residual at last layer is 0.23, but the first layer moves away from target |
| **MEMIT+FE** (Ours) | **99.9** | Compatible targets across layers; residuals significantly reduced |

Cosine measurements (Llama3-8B, layer 4→8): $0.34, 0.41, 0.54, 0.72, 1.00$. This quantifies the directional drift of distant layers, explaining why backward spreading's contribution at layer 4 is near zero.

### Key Findings
- The failure of backward spreading in shallow layers is a structural issue rather than an implementation bug: the Jacobian's rotation effect scales linearly with depth difference.
- FE's improvements extend beyond efficacy, significantly reducing DKL (MCF 0.41 → 0.34, ZsRE 0.60 → 0.09). This indicates fewer side effects on unrelated knowledge, as more accurate targets allow closed-form solutions to align better with target directions.
- Applying FE to RECT/PRUNE/AlphaEdit yields consistent gains (e.g., AlphaEdit Efficacy 95.2 → 99.8), validating its value as a general-purpose plugin.

## Highlights & Insights
- Formally decomposes the common assumption of "why backward spreading works" into two hypotheses: positive definiteness of the symmetric Jacobian and sufficiently small step sizes. This "explain then replace" approach is more rigorous than simply introducing a new method.
- The core insight of forward replay: during backpropagation, the model already "calculates" where each layer should move to meet the objective; one only needs a forward pass to extract this information. This is an elegant reuse of implicit backward information.
- As a pure plugin improvement that does not increase computation or modify other components, it can be seamlessly integrated into almost any LTE pipeline in the model editing community.

## Limitations & Future Work
- Experiments were confined to MLP layers; applicability to attention layers remains unverified.
- Scaling performance for extremely large batch sizes (editing $10^4$+ facts simultaneously) was not specifically evaluated. The stability of $m_1$ in larger batches warrants further investigation.
- In very deep models (e.g., 70B+), the forward trajectory from $m_1$ to the final layer might distort. Whether segmented anchors are required remains an open question.
- Potential integration with non-LTE methods (hypernetworks, memory-based WISE) was not discussed.

## Related Work & Insights
- **vs MEMIT**: Retains the MEMIT closed-form solution and regularization, changing only the target source from backward to forward.
- **vs BLUE**: BLUE optimizes the first and last layers independently while ignoring intermediate layers. FE obtains targets for all layers via a single optimization at half the cost, ensuring they are mutually compatible.
- **vs RECT / PRUNE / AlphaEdit**: These methods modify the regularization or subspace of $\Delta W$, which is orthogonal to FE's focus on target sources, allowing for additive improvements.
- **vs WISE / RLEdit**: These follow non-LTE routes (extra memory or hypernetworks). This work does not compete with them but suggests the LTE route has not yet reached its upper limit.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The shift from backward to forward is simple but represents a fundamental paradigm change.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons across two datasets, three models, and 5+ baselines, supported by theoretical proofs in the appendix.
- **Writing Quality**: ⭐⭐⭐⭐ High clarity in the "analyze then propose" structure; Figure 1/2 provide intuitive comparisons.
- **Value**: ⭐⭐⭐⭐⭐ A general plugin improvement that benefits any LTE method, with broad impact on the model editing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ICML 2026\] CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](../../ACL2026/knowledge_editing/clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](reverse-engineering_model_editing_on_language_models.md)

</div>

<!-- RELATED:END -->
