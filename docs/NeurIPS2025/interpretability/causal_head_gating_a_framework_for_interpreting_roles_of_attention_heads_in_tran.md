---
title: >-
  [Paper Note] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers
description: >-
  [NeurIPS 2025][mechanistic interpretability] This paper proposes Causal Head Gating (CHG), which learns a differentiable gating scalar for each attention head in a Transformer and applies positive/negative regularization to classify heads into three causal roles—facilitating, interfering, and irrelevant—without requiring manual labels or prompt templates. The framework discovers causal sub-circuits at scale and extends to Contrastive CHG for disentangling independent circuits underlying in-context learning (ICL) and instruction following.
tags:
  - NeurIPS 2025
  - mechanistic interpretability
  - attention head
  - causal taxonomy
  - circuit discovery
  - Llama
date: 2026-05-08
content_hash: 1b92cd63ecfe282c
---

# Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2505.13737](https://arxiv.org/abs/2505.13737)
**Code**: [https://github.com/andrewnam/causal_head_gating](https://github.com/andrewnam/causal_head_gating)
**Area**: Interpretability
**Keywords**: mechanistic interpretability, attention head, causal taxonomy, circuit discovery, Llama

## TL;DR
This paper proposes Causal Head Gating (CHG), which learns a differentiable gating scalar for each attention head in a Transformer and applies positive/negative regularization to classify heads into three causal roles—facilitating, interfering, and irrelevant—without requiring manual labels or prompt templates. The framework discovers causal sub-circuits at scale and extends to Contrastive CHG for disentangling independent circuits underlying in-context learning (ICL) and instruction following.

## Background & Motivation

**Background**: Mechanistic interpretability aims to understand the internal computations of LLMs. Existing approaches fall into two categories: (a) training probes to decode hidden states into interpretable concepts (correlational, not causal); and (b) causal mediation analysis (CMA), which uses carefully designed prompts to localize heads responsible for specific behaviors (causally rigorous but limited in scale).

**Limitations of Prior Work**: (1) Probe-based methods are correlational and cannot demonstrate that the model actually uses the detected features. (2) CMA requires manually designed prompt templates and explicit mechanistic hypotheses, making it difficult to scale to complex tasks such as mathematical reasoning, where problem structures vary widely. (3) Existing head pruning work is primarily conducted on small models or BERT, with limited applicability to modern LLMs. (4) Gumbel-based hard gating methods assume head independence, failing to capture inter-head interactions.

**Key Challenge**: How can one discover causally effective attention heads in LLMs at scale, without relying on manually crafted templates or labels?

**Goal**: Design a scalable method to identify and classify the causal roles of attention heads, applicable to arbitrary tasks and datasets.

**Key Insight**: Gate parameters (one scalar per head) are trained using the next-token prediction objective. Positive and negative regularization induce variation in the gating values, enabling distinction among facilitating, interfering, and irrelevant heads.

**Core Idea**: Two separate fits are performed using positive and negative L1 regularization respectively. Facilitating heads maintain high gate values in both fits; interfering heads are suppressed in both; irrelevant heads diverge between the two fits—thereby establishing a ternary causal taxonomy.

## Method

### Overall Architecture
A gating matrix $G \in [0,1]^{L \times H}$ is defined over the $L \times H$ attention heads of a Transformer, where $G_{\ell,h}$ scales the output of head $(\ell, h)$. All model parameters are frozen; only $G$ is optimized under a NLL + regularization objective. Two separate fits are performed: one with $\lambda > 0$ (encouraging retention, yielding $G^+$) and one with $\lambda < 0$ (encouraging removal, yielding $G^-$). Heads are then classified based on the resulting patterns.

### Key Designs

1. **Gating Mechanism**:

    - **Function**: Applies a learnable scaling coefficient to the output of each attention head.
    - **Mechanism**: $Z_{\ell,h} = G_{\ell,h} \cdot (A_{\ell,h} V_{\ell,h})$, where the gating coefficient $G_{\ell,h}$ is applied after the attention computation but before the output projection. All model parameters $\theta$ are frozen; only $G$ is optimized.
    - **Design Motivation**: Each head introduces only one additional parameter, incurring minimal overhead. The intervention is applied directly within the model's computation graph, ensuring that the findings are causal rather than correlational.

2. **Regularization-Based Separation**:

    - **Function**: Separates irrelevant heads from facilitating and interfering heads via positive and negative regularization.
    - **Mechanism**: The objective is $\mathcal{L} = \text{NLL} - \lambda \sum \sigma^{-1}(G_{\ell,h})$. With $\lambda > 0$, regularization encourages gates toward 1 (retaining all heads), while the NLL gradient suppresses interfering heads, yielding $G^+$. With $\lambda < 0$, regularization encourages gates toward 0 (removing all heads), while the NLL gradient preserves facilitating heads, yielding $G^-$. Key insight: if a head is irrelevant, its expected NLL gradient is zero, so its gate value is determined entirely by regularization—trending toward 1 in $G^+$ and toward 0 in $G^-$, producing divergence.
    - **Design Motivation**: Pure NLL optimization cannot distinguish facilitating heads (gates remain high) from irrelevant heads (gates happen to be high); regularization breaks this confound.

3. **Ternary Causal Classification**:

    - **Function**: Classifies heads into three categories based on the patterns in $G^+$ and $G^-$.
    - **Mechanism**: **Facilitating**: $G^+ \approx 1, G^- \approx 1$ (retained in both fits); **Interfering**: $G^+ \approx 0, G^- \approx 0$ (suppressed in both fits); **Irrelevant**: $G^+ \approx 1, G^- \approx 0$ (regularization dominates). Facilitation score $= G^-$; interference score $= 1 - G^+$; irrelevance score $= G^+ \times (1 - G^-)$.
    - **Design Motivation**: Avoids the independence assumption of Gumbel-based methods. CHG jointly optimizes all gates, capturing inter-head interactions.

4. **Contrastive CHG**:

    - **Function**: Disentangles distinct sub-circuits that implement the same task via different mechanisms (e.g., ICL vs. instruction following).
    - **Mechanism**: Two variants of the same task are constructed (ICL format vs. instruction format). A single gating matrix is fit to "forget" one variant while retaining the other. The objective comprises two terms: maximizing NLL on the forgotten variant (forgetting) and minimizing NLL on the retained variant (remembering).
    - **Design Motivation**: Standard CHG identifies which heads are necessary for a task; Contrastive CHG further distinguishes heads used for "understanding the task format" from those used for "executing the task."

### Loss & Training
NLL + L1 regularization. $G$ is first initialized by fitting with $\lambda = 0$, then separately fit with $\lambda > 0$ and $\lambda < 0$. Gradient clipping ensures that the NLL term remains dominant. Each configuration is fit with 10 random seeds. Experiments are conducted on the Llama-3 series (1B/3B/8B).

## Key Experimental Results

### Main Results: Causal Classification Validation

CHG classifications are validated via targeted ablation—heads are ranked by facilitation/irrelevance/interference scores and ablated sequentially:
- Ablating facilitating heads → performance degrades (negative delta log-prob)
- Ablating irrelevant heads → performance unchanged (delta ≈ 0)
- Ablating interfering heads → performance improves (positive delta log-prob)

All three patterns are consistently validated across 4 models × 3 tasks.

### Head Distribution Analysis

| Task | Always Facilitating | Always Interfering | Notes |
|------|--------------------|--------------------|-------|
| Syntax | <5% | ~0% | Compact, sparse circuit |
| Common Sense | <5% | ~0% | Compact, sparse circuit |
| Math | 38.3% (3B) | 1.3% (3B) | Larger, more rigid circuit |

### Ablation Study: Contrastive CHG

| Task | Retained Format | Forgotten Format Accuracy | Retained Format Accuracy |
|------|----------------|--------------------------|--------------------------|
| antonym (ICL) | ICL | 0% (forgetting successful) | Near baseline |
| singular-plural (Inst) | Instruction | 0% (forgetting successful) | 21% (cross-circuit overlap) |

### Key Findings
- **CHG classifications are highly consistent with ablation results**, validating causality rather than mere correlation.
- **Mathematical reasoning requires a larger and more rigid circuit**: 52.6% of heads are facilitating (vs. 25.6% for syntax), with higher cross-seed consistency.
- **Head roles are not modular**: the same head may be facilitating or irrelevant under different random seeds, depending on the configuration of other heads. The model contains multiple redundant sufficient sub-circuits.
- **ICL and instruction following rely on separable circuits**: Contrastive CHG successfully forgets one while retaining the other, generalizing to unseen tasks.
- **High cross-model consistency**: Pearson correlation of CHG distributions across 1B–8B models reaches 99.2%.
- **Complementarity with CMA is validated**: heads identified by CMA also receive high facilitation scores under CHG ($t(53.77)=11.18$, $p<10^{-15}$).

## Highlights & Insights
- **The symmetric design of positive/negative regularization** is remarkably elegant: a simple idea—using two regularization passes in opposite directions to distinguish three head types—yields a powerful analytical tool. With only one parameter per head, fitting takes just a few minutes, providing excellent scalability.
- **The discovery of "multiple sufficient sub-circuits"** is a profound insight: the model does not rely on a fixed circuit for a given task but instead maintains multiple functionally equivalent sub-circuits. This explains why head pruning typically incurs little performance loss—surviving sub-circuits compensate.
- **Contrastive CHG's disentanglement of ICL and instruction following** is a novel contribution, providing the first evidence that these two capabilities are separable at the level of attention heads.

## Limitations & Future Work
- **Only attention heads are analyzed; MLP layers are not considered**: MLPs also store and process important information. Future work could extend the framework to MLP neurons.
- **The method does not explain what a head computes**: CHG identifies which heads matter but does not characterize the computations they perform. Integration with methods such as CMA is needed.
- **Dependence on the NTP objective**: CHG may not be applicable to tasks that are not well-measured by next-token prediction.
- **Suggested directions**: Extending CHG to MLP layers and feature-level analysis of the residual stream.

## Related Work & Insights
- **vs. CMA**: CMA offers high precision but low scalability (hypothesis-driven); CHG offers high scalability at moderate precision (data-driven). The two approaches are complementary.
- **vs. Sparse Autoencoders**: SAEs discover interpretable features but remain correlational; CHG establishes causal links.
- **vs. Gumbel-based gating**: Gumbel methods assume head independence (factorized distribution); CHG jointly optimizes all gates, capturing inter-head interactions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The positive/negative regularization ternary classification is elegant; Contrastive CHG is a novel contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model, multi-task evaluation; cross-validated with CMA; complete ablation study
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical structure; high-quality figures
- Value: ⭐⭐⭐⭐⭐ Provides a lightweight, scalable, and causally grounded tool for attention head analysis

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)

<!-- RELATED:END -->
