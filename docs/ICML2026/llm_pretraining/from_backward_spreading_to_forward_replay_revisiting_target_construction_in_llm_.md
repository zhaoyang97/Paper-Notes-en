---
title: >-
  [Paper Note] From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing
description: >-
  [ICML 2026][LLM Pretraining][Parameter Editing] This paper systematically analyzes why backward spreading works in locate-then-edit (LTE) editing, why it is insufficient…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Parameter Editing"
  - "MEMIT"
  - "Multi-layer Coordination"
  - "Forward Propagation"
  - "Hidden State"
date: 2026-05-08
content_hash: 49279fb5d07cb87a
---

# From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing

**Conference**: ICML 2026  
**arXiv**: [2605.00358](https://arxiv.org/abs/2605.00358)  
**Code**: https://github.com/jugechengzi/FE (available)  
**Area**: LLM Knowledge Editing / Locate-then-edit  
**Keywords**: Parameter Editing, MEMIT, Multi-layer Coordination, Forward Propagation, Hidden State

## TL;DR
This paper systematically analyzes why backward spreading works in locate-then-edit (LTE) editing, why it is insufficient, and proposes forward replay: treating the first decisive layer as the optimization variable and obtaining subsequent layer targets via standard forward propagation. This approach consistently improves over MEMIT/RECT/PRUNE/AlphaEdit without extra computational cost.

## Background & Motivation

**Background**: The locate-then-edit (LTE) paradigm, exemplified by ROME and MEMIT, has become mainstream in LLM knowledge editing: first, causal tracing is used to locate key tokens and several decisive MLP layers; then, the ideal hidden state $m_L$ at the last layer is computed, followed by a closed-form rank-one update. To avoid regularization suppressing single-layer edits, the mainstream approach linearly interpolates $m_L$ via backward spreading, distributing it to earlier layers (e.g., $m_1 = h_1 + (m_3-h_3)/3$).

**Limitations of Prior Work**: The validity of backward spreading has not been systematically examined. It implicitly assumes that the residual direction at the last layer applies to all earlier layers, but forward dynamics are nonlinear and directions rotate across layers. Recent methods like BLUE attempt independent optimization for the first and last layers, at the cost of doubled computation and potential incompatibility between targets in high-dimensional space.

**Key Challenge**: Single-layer closed-form solutions rely on accurate $m_l$, while multi-layer coordination requires compatibility among $m_l$ across layers; backward spreading uses a single direction for all layers, while independent optimization breaks compatibility. The root problem is that **the direction of target propagation is inconsistent with the model's true forward dynamics**.

**Goal**: (1) Formally characterize when backward spreading is applicable and when it fails; (2) Propose a multi-layer target construction method naturally aligned with forward dynamics and without increased computational cost.

**Key Insight**: Since treating $m_L$ as an optimizable parameter for the last layer already encodes "where earlier layers should go to achieve the goal" in the backward computation graph, optimizing at the **first decisive layer** and then replaying forward propagation can explicitly recover this information.

**Core Idea**: Replace backward spreading with forward replay—move the anchor point from the last to the first layer, and let targets for other layers naturally emerge via standard forward propagation. This maintains complexity while ensuring cross-layer compatibility.

## Method

### Overall Architecture
Let the decisive layers to be edited be $W_1,\dots,W_L$, with the key token's hidden state before $W_l$ as $k_l$, and the target hidden state as $m_l$. The LTE closed-form solution $\Delta W^* = (M_I - W K_I) K_I^\top (K_I K_I^\top + \lambda K_J K_J^\top)^{-1}$ depends on $m_l$ at each layer. MEMIT computes only $m_L$, with other $m_l$ obtained by linear interpolation; the proposed FE (Forward propagation Edit) computes only $m_1$, with the rest obtained via forward replay. The overall pipeline is identical, differing only in target construction.

### Key Designs

1. **Theoretical Characterization of Backward Spreading**:

    - **Function**: Explains why backward spreading works and when it fails, motivating the proposed alternative.
    - **Mechanism**: Let $\delta m_l = \Delta W_l k_l$ be the output perturbation at layer $l$, and $\delta^l_{m_L} = J_{l\to L} \delta m_l$ its induced change at the last layer. Ideally, $\delta^l_{m_L} = \beta \delta m_l$, i.e., $\delta m_l$ is an eigenvector of the Jacobian $J_{l\to L}$ (Theorem 1, a strong and rarely satisfied condition). Relaxing to $\cos(\delta^l_{m_L}, \delta m_l) > 0$ is equivalent to the symmetrized Jacobian $\frac{1}{2}(J_{l\to L} + J_{l\to L}^\top)$ being positive definite (Theorem 2, empirically holds in most deep networks). On Llama3-8B, the cosine from layer 4 to 8 increases from $0.34$ to $1.00$, quantitatively showing that the influence from distant layers drifts in direction at the last layer, fundamentally limiting backward spreading's scalability.
    - **Design Motivation**: By first clarifying why the old method is "partially effective," the case for a replacement is made more robust than simply proposing a new method.

2. **Forward Replay for Multi-layer Target Construction**:

    - **Function**: Uses a single forward pass to obtain mutually compatible target hidden states for all decisive layers.
    - **Mechanism**: MEMIT treats the last layer $h_L$ as an optimizable parameter to minimize cross-entropy and obtain $m_L$, then back-propagates. FE reverses this—treats the **first layer** $h_1$ as the optimizable parameter, performs gradient descent for the same objective to obtain $m_1$; then, $m_1$ is fed into the model, and standard forward propagation records the hidden state at each decisive layer as $m_l$. The resulting $\{m_l\}$ all come from the same forward trajectory and are naturally compatible; since the backward gradient path traverses the entire network, $m_1$ inherently encodes downstream dynamics constraints.
    - **Design Motivation**: Backward spreading forcibly imposes a last-layer direction on earlier layers; forward replay lets the first layer "know what the last layer wants," and the network computes subsequent layers accordingly, following the model's natural dynamics.

3. **Plug-and-Play Compatibility**:

    - **Function**: Enables FE to boost not only MEMIT but also a wide range of subsequent LTE methods.
    - **Mechanism**: FE only replaces the target propagation mechanism, leaving the closed-form solution, initial $m$ optimization, and regularization untouched. Thus, RECT (with norm regularization), PRUNE (singular value pruning), AlphaEdit (null-space projection), etc., can all be directly combined with "+FE."
    - **Design Motivation**: Given the many branches in the model editing community, an improvement orthogonal to all existing pipelines is more impactful than starting anew.

### Loss & Training
No new loss is introduced. When optimizing $m_1$, cross-entropy $H(Y, Y_{\text{target}})$ is still used; after obtaining $m_1$, it is used as the hidden state for the first decisive layer, and a forward pass yields $\{m_l\}_{l=2}^L$; finally, the unified rank-one closed-form solution updates parameters layer by layer. Computational complexity is identical to MEMIT.

## Key Experimental Results

### Main Results
On Llama3-8B-Instruct, using EasyEdit's default decisive layers $[4,5,6,7,8]$, 2000 knowledge edits each for MCF and ZsRE, batch editing.

| Dataset | Method | Efficacy Success ↑ | Generalization Acc ↑ | DKL ↓ | Top-1 ↑ |
|---------|--------|--------------------|---------------------|-------|---------|
| MCF | MEMIT | 97.4 | 48.0 | 0.41 | 77.5 |
| MCF | BLUE (2-layer independent opt.) | 98.4 | 60.1 | 0.35 | 81.0 |
| MCF | **MEMIT+FE** | **99.9** | **61.0** | **0.34** | **82.7** |
| ZsRE | MEMIT | 92.6 | 72.6 | 0.60 | 45.3 |
| ZsRE | BLUE | 94.6 | 78.2 | 0.18 | 66.7 |
| ZsRE | **MEMIT+FE** | **97.6** | **84.3** | **0.09** | **75.6** |

### Ablation Study

| Configuration | MCF Efficacy Success | Description |
|---------------|----------------------|-------------|
| OneLayer (last layer only) | 76.2 | Single-layer insufficient, multi-layer coordination necessary |
| MEMIT (backward, dividing) | 97.4 | Standard baseline |
| MEMIT (backward, no-dividing) | — | Last-layer residual 0.23 but first layer diverges from target |
| MEMIT+FE | **99.9** | Targets compatible across layers, residuals drop significantly |

Cosine measurements (Llama3-8B, layer 4→8): $0.34, 0.41, 0.54, 0.72, 1.00$, quantitatively showing directional drift in distant layers; this explains why backward spreading's contribution at layer 4 is nearly zero.

### Key Findings
- Backward spreading's failure in shallow layers is a structural issue, not an implementation bug: the Jacobian's rotational effect amplifies linearly with depth difference.
- FE's improvement is not only in efficacy but also in significantly reducing DKL (MCF 0.41 → 0.34, ZsRE 0.60 → 0.09), indicating less side effect on unrelated knowledge—more accurate targets make the closed-form solution align better with the intended direction, reducing collateral impact.
- Applying FE to RECT/PRUNE/AlphaEdit also yields consistent gains (AlphaEdit Efficacy 95.2 → 99.8), validating its value as a universal plugin.

## Highlights & Insights
- The commonly assumed effectiveness of "backward spreading" is formally reduced to two assumptions: positive definiteness of the symmetric part of the Jacobian and sufficiently small step size; this "explain then replace" structure is more rigorous than simply proposing a new method.
- The core insight of forward replay: during gradient backpropagation, the model already "computes" where each layer should go to achieve the goal; a single forward pass can read out this information—an elegant "reuse of implicit information from backpropagation."
- A pure plugin-style improvement that does not increase computational cost or alter other components, making it almost seamlessly integrable into any LTE pipeline in the model editing community.

## Limitations & Future Work
- Experiments are limited to MLP decisive layers; applicability to attention layers remains unverified.
- Scaling behavior for extremely large batch sizes (editing $10^4$+ facts simultaneously) is not specifically evaluated; stability of $m_1$ in larger batches warrants further investigation.
- For very deep models (e.g., 70B+), the forward trajectory from $m_1$ to the last layer may itself become distorted, raising the open question of whether segmented anchoring is needed.
- Potential integration with non-LTE methods (hypernetworks, memory-based WISE) is not discussed.

## Related Work & Insights
- **vs MEMIT**: Fully retains MEMIT's closed-form solution and regularization, only changing the source of targets; shifts from backward to forward, reversing the approach.
- **vs BLUE**: BLUE independently optimizes the first and last layers but ignores intermediate layers; FE obtains all layer targets in a single optimization, halving the cost and ensuring mutual compatibility.
- **vs RECT / PRUNE / AlphaEdit**: These methods modify the regularization or subspace of $\Delta W$, while FE changes only the target source; the two are fully orthogonal and can be combined.
- **vs WISE / RLEdit**: Those methods follow non-LTE routes (additional memory or hypernetworks); this work does not compete with them, but rather demonstrates that the LTE route still has untapped potential.

## Rating
- Novelty: ⭐⭐⭐⭐ The backward → forward paradigm shift is simple yet represents a paradigm-level change.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons on two datasets, three models, and 5+ baselines, with theoretical proofs in the appendix.
- Writing Quality: ⭐⭐⭐⭐ The "analyze old method—then propose new method" structure is clear, and Figures 1/2 provide intuitive comparisons.
- Value: ⭐⭐⭐⭐⭐ As a universal plugin-style improvement, any LTE method can directly benefit, with broad impact on the model editing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Mirage of Model Editing: Revisiting Evaluation in the Wild](../../ACL2025/knowledge_editing/the_mirage_of_model_editing_revisiting_evaluation_in_the_wild.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](../../ACL2026/knowledge_editing/clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](../../ACL2025/knowledge_editing/chainedit_propagating_ripple_effects_in_llm.md)
- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](../../ICLR2026/knowledge_editing/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)

</div>

<!-- RELATED:END -->
