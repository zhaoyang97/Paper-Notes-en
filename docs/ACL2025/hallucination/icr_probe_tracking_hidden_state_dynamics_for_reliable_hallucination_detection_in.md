---
title: >-
  [Paper Note] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs
description: >-
  [ACL 2025][Hallucination Detection][Hidden State Dynamics] Proposes ICR Score (Information Contribution to Residual Stream) to quantify residual stream dynamics by measuring the consistency of the contributions of MHSA and FFN modules to hidden state updates. A lightweight ICR Probe with only 16K parameters is constructed, which consistently outperforms baselines in hallucination detection AUROC across 4 datasets × 3 LLMs.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hidden State Dynamics"
  - "Residual Stream"
  - "ICR Score"
  - "Lightweight Probe"
date: 2026-05-08
content_hash: 9640991b2328f830
---

# ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs

**Conference**: ACL 2025  
**arXiv**: [2507.16488](https://arxiv.org/abs/2507.16488)  
**Code**: [https://github.com/XavierZhang2002/ICR_Probe](https://github.com/XavierZhang2002/ICR_Probe)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Detection, Hidden State Dynamics, Residual Stream, ICR Score, Lightweight Probe

## TL;DR
Proposes ICR Score (Information Contribution to Residual Stream) to quantify residual stream dynamics by measuring the consistency of the contributions of MHSA and FFN modules to hidden state updates. A lightweight ICR Probe with only 16K parameters is constructed, which consistently outperforms baselines in hallucination detection AUROC across 4 datasets × 3 LLMs.

## Background & Motivation
**Background**: LLM hallucination detection methods are categorized into three types: (a) output-consistency-based (requiring multiple generations); (b) logit-probability-based (requiring reference answers); and (c) hidden-state-based (the most practical as it requires no external references).

**Limitations of Prior Work**: Hidden-state-based methods usually focus on static, isolated high-dimensional representations (~4000 dimensions), neglecting the dynamic evolution of hidden states across layers. Direct probing of hidden states, such as in SAPLMA, requires 110K parameters, while SEP utilizes semantic entropy but exhibits poor generalization.

**Key Challenge**: Hidden states themselves are high-dimensional and static, leading to a low signal-to-noise ratio when directly used as classification signals.

**Goal**: To identify a low-dimensional, stable, and cross-dataset consistent hidden state feature to detect hallucinations.

**Key Insight**: Instead of looking at the hidden states themselves, the focus is shifted to their update process—specifically, the contribution proportions of MHSA (context routing) and FFN (knowledge retrieval) to the residual stream.

**Core Idea**: Use JSD to measure the consistency between the hidden state update direction and the attention scores to obtain the ICR Score, followed by cross-layer aggregation, and finally employ a lightweight probe to detect hallucinations.

## Method

### Overall Architecture
At each layer $\ell$ of the LLM, the ICR Score is calculated for each token $i$: (1) extract attention scores $\text{Attn}_i^\ell$; (2) calculate the projection vector $\text{Proj}_i^\ell$ of the hidden state update $\Delta x_i^\ell = x_i^\ell - x_i^{\ell-1}$ onto the hidden state directions of all tokens; (3) measure the consistency between $\text{Proj}$ and $\text{Attn}$ using JSD to obtain the ICR Score. The ICR Scores of all layers are token-averaged and fed into a 4-layer MLP (with 16K parameters) to output the hallucination probability.

### Key Designs

1. **ICR Score Construction**:

    - Function: Quantifying the degree of dominance of MHSA vs FFN on hidden state updates in each layer
    - Mechanism: $\text{ICR}_i^\ell = \text{JSD}(\text{Proj}_i^\ell, \text{Attn}_i^\ell)$. The projection vector $p_{i,j}^\ell = \frac{(\Delta x_i^\ell)^T \cdot x_j^\ell}{\|x_j^\ell\|}$ measures the alignment between the update direction and each token's representation
    - Design Motivation: A low ICR indicates that the update is dominated by MHSA (context routing), whereas a high ICR indicates dominance by FFN (parametric knowledge injection). Hallucinations tend to occur when FFN abnormally injects information
    - Difference from Prior Methods: Instead of directly using the 4000-dimensional hidden states, a 1-dimensional ICR Score is used to compress the information of each layer

2. **Cross-Layer Stability**:

    - Function: Verifying that the layered pattern of ICR Score is consistent across datasets
    - Mechanism: Early layers (0-3) show low ICR (MHSA-dominated local extraction) → middle layers (4-20) exhibit rising ICR (FFN-infused knowledge) → late layers (21+) display falling ICR (MHSA refinement and integration)
    - Design Motivation: This consistent pattern is an intrinsic property of the model rather than being data-specific, thereby ensuring generalization

3. **ICR Probe Architecture**:

    - Function: Using the ICR Scores of all L layers as input, and outputting the hallucination probability via a 4-layer MLP
    - Mechanism: The input is a token-wise averaged $1 \times L$ vector, with an architecture of $(L, 128, 64, 32, 1)$, totalling <16K parameters
    - Design Motivation: The ICR Score is already a powerful feature that does not require a large network; 16K parameters versus SAPLMA's 110K parameters

### Loss & Training
Binary cross-entropy loss + Adam optimizer, standard supervised learning.

## Key Experimental Results

### Main Results

| LLM | Method | HaluEval | SQuAD | HotpotQA | TriviaQA |
|-----|------|----------|-------|----------|----------|
| Gemma-2 | ICR Probe | **0.8436** | **0.8142** | **0.8409** | **0.8001** |
| Gemma-2 | SAPLMA | 0.8101 | 0.7175 | 0.8193 | 0.7751 |
| Qwen2.5 | ICR Probe | **0.8003** | **0.7456** | **0.7917** | 0.7684 |
| Qwen2.5 | SAPLMA | 0.7799 | 0.6929 | 0.7750 | **0.8225** |
| Llama-3 | ICR Probe | **0.7603** | **0.7634** | **0.7982** | 0.7325 |
| Llama-3 | SAPLMA | 0.7238 | 0.7107 | 0.7701 | **0.7650** |

### Ablation Study

| Configuration | AUROC | Description |
|------|-------|------|
| ICR Probe (Attn+Proj) | 0.8436 | Full model |
| Attn Only | 0.5000 | Attention scores alone have no discriminative power |
| Proj Only | 0.5000 | Projection directions alone are also ineffective |
| Attn+Proj (ICR=JSD) | 0.8436 | Only their consistency provides the signal |

### Key Findings
- **ICR Score is an effective hallucination detection signal**: Single-layer AUROC reaches up to 0.769 (layer 11), and is even higher after cross-layer aggregation.
- **Strong cross-dataset generalization**: Cross-domain AUROC drop is only 8.61% vs SAPLMA's 10.18%, thanks to capturing intrinsic properties of the model.
- **Outperforming larger probes with extremely few parameters**: 16K parameters vs SAPLMA's 110K parameters, achieving better performance.
- **Core Insight**: During hallucinations, the contribution patterns of FFN and MHSA are abnormal—FFN excessively injects knowledge in layers where it should not dominate.

## Highlights & Insights
- **Paradigm shift from "observing states" to "observing updates"**: Instead of looking at the hidden states themselves (4000 dimensions with high noise), focusing on features of the update process (1-dimensional ICR Score) significantly increases the signal-to-noise ratio. This "observe changes rather than states" philosophy can be transferred to other model diagnostics tasks.
- **MHSA-FFN contribution inconsistency as a hallucination signal**: This provides a new perspective for understanding internal mechanisms of LLM hallucinations—hallucinations may be associated with abnormal knowledge injection by the FFN.
- **Ingenious use of JSD**: Compressing the comparison of two distributions into a single scalar retains information while reducing dimensionality.

## Limitations & Future Work
- Tests were restricted to LLMs with 7-9B parameters; larger models (70B+) remain unverified.
- Training requires labeled hallucination data, resulting in limited zero-shot detection capability.
- The discriminative power of the ICR Score decreases in the final few layers, indicating a potential need for weighted aggregation instead of simple averaging.
- Only text hallucinations are addressed; multimodal hallucination (VLM) scenarios remain unexplored.

## Related Work & Insights
- **vs SAPLMA**: SAPLMA directly trains a probe with hidden states (110K params), whereas ICR Probe uses ICR Score as a more compact feature (16K params) and achieves better performance.
- **vs SEP**: Semantic entropy probes focus on uncertainty, whereas ICR focuses on information flow dynamics; the two are complementary.
- **vs LN-Entropy**: Training-free methods are simple but yield 10+ points lower AUROC; the training-based ICR Probe offers the best trade-off.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of ICR Score is novel, understanding hallucinations from the perspective of residual stream dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 LLMs × 4 datasets + ablation + generalization analysis + case study.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, rich visualizations, and good interpretability.
- Value: ⭐⭐⭐⭐⭐ A new paradigm for hallucination detection, also contributing to the understanding of the internal mechanisms of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](../../ACL2026/hallucination/multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)
- [\[ACL 2025\] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ACL 2025\] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries](etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)

</div>

<!-- RELATED:END -->
