---
title: >-
  [Paper Note] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][hallucination] This paper identifies the root cause of object hallucination in MLLMs at the representation level—semantic entanglement induced by dataset co-occurrence bias—and proposes a d…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "hallucination"
  - "causal inference"
  - "disentanglement"
  - "MLLM"
  - "co-occurrence bias"
  - "backdoor adjustment"
date: 2026-05-08
content_hash: 584a6f751c8dfeb5
---

# Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.19474](https://arxiv.org/abs/2505.19474)  
**Code**: [https://github.com/IgniSavium/Causal-LLaVA](https://github.com/IgniSavium/Causal-LLaVA)  
**Area**: Multimodal VLM
**Keywords**: hallucination, causal inference, disentanglement, MLLM, co-occurrence bias, backdoor adjustment

## TL;DR
This paper identifies the root cause of object hallucination in MLLMs at the representation level—semantic entanglement induced by dataset co-occurrence bias—and proposes a dual-path causal disentanglement framework (Causal-Driven Projector + Causal Intervention Module). By applying backdoor adjustment at both the projector and the final Transformer layer to decouple co-occurring object representations, the method achieves a 22.6% improvement on MME-Perception.

## Background & Motivation

### State of the Field

**Background**: Object hallucination in MLLMs (i.e., describing objects that do not exist) primarily stems from co-occurrence bias in training data—for example, "dining table" almost always co-occurs with "chair," causing the model to learn spurious correlations. However, prior work has only validated the correlation between co-occurrence frequency and hallucination rate at a **statistical level**, without investigating the underlying mechanisms in the representation space.

A key finding of this paper: by visualizing object representations across layers of LLaVA via PCA, the authors observe the following:

### Limitations of Prior Work

**Limitations of Prior Work**: Object representations are dispersed (normal) at the output of the CLIP encoder.

### Root Cause

**Key Challenge**: After processing by the Projector, representations of high-frequency co-occurring objects cluster sharply together (entanglement emerges).

### Mechanism

**Mechanism**: The entanglement propagates continuously through layers 1–15 of the LLM (the comprehension stage).

### Additional Notes

**Additional Notes**: Significant entanglement persists in the final layer 40 (the prediction stage).

This indicates that the projector is the critical node for bias injection, and once formed, the bias pervades the entire inference process.

### Starting Point

**Goal**: How can the propagation of co-occurrence bias be interrupted at the **representation learning level**, so that the semantic representations of co-occurring objects are disentangled, thereby reducing object hallucination?

## Method

### Overall Architecture
Causal intervention modules are inserted at two key positions within LLaVA: (1) after the Projector (to block the propagation of visual confounders $D_v$ to soft tokens $S$), and (2) at the final Transformer layer of the LLM (to block the propagation of visual/textual confounders $D_v$/$D_t$ to the prediction $W$).

### Key Designs
1. **Causal-Driven Projector**: Based on the backdoor adjustment formula $P(Y|do(X)) \approx_{NWGM} g_f(f_v) + g_z(\mathbb{E}_z[z])$, the original projector output is summed with an estimate of the expected confounder value. The confounder dictionary $D \in \mathbb{R}^{K \times \sigma}$ is constructed from the mean post-projector visual representations of 80 COCO object categories (aggregated from 5,000 samples), and $\mathbb{E}_z[z]$ is dynamically estimated via cross-attention.

2. **Causal Intervention Module (LLM layer)**: Inserted at the final Transformer layer, it applies cross-attention intervention using visual and textual confounder dictionaries $D_v$ and $D_t$ respectively: $\text{CausalIntervention}(h) = \text{CrossAttn}(h, D_v, D_v) + \text{CrossAttn}(h, D_t, D_t)$, decoupling visual and textual co-occurrence biases from the hidden states.

3. **NWGM Approximation**: Exact computation of causal intervention requires enumerating all confounders, which is computationally expensive. The Normalized Weighted Geometric Mean is employed to move the expectation inside the Softmax, simplifying the computation to $\text{Softmax}[g(x, \mathbb{E}_z[z])]$.

### Loss & Training
The original LLaVA training configuration is preserved, with only the following modifications: batch size 256 (2×), learning rate 1e-3 (0.5×). The confounder dictionary is extracted from a checkpoint trained for 0.1 epochs on the non-causal model. Training is conducted on 8×H20 GPUs, with confounder estimation requiring approximately 1 additional hour.

## Key Experimental Results

| Model | LLM | POPE_rnd | MME_P | CHAIR_s↓ | CHAIR_i↓ |
|------|-----|----------|-------|----------|----------|
| LLaVA | LLaMA-2-7B | 71.70 | 714.29 | 33.0 | 9.5 |
| **Causal-LLaVA** | **LLaMA-2-7B** | **72.72** | **757.16** | **30.9** | **9.2** |
| LLaVA | LLaMA-2-13B | 78.60 | 711.22 | 30.3 | 8.7 |
| **Causal-LLaVA** | **LLaMA-2-13B** | **79.54** | **872.09** | **28.2** | **8.5** |
| LLaVA1.5 | Vicuna-1.5-7B | 87.34 | 1508.51 | 52.1 | 14.9 |
| **Causal-LLaVA1.5** | **Vicuna-1.5-7B** | **88.18** | **1522.10** | **51.4** | **14.8** |

Visual understanding capabilities are also improved concurrently: MMBench +2.0%, MM-Vet +4.8%, GQA +2.7%, VizWiz +8.4%.

### Ablation Study
- **Dual-path vs. single-path**: Only-projector (MME 748.70) + Only-transformer (726.15) < Both (757.16), demonstrating complementarity of the two paths.
- **Projection matrix selection**: Shared $W_k/W_v$ yields the best performance (CHAIR_s 27.7); independent $W_q/W_v$ or $W_q/W_o$ leads to catastrophic degradation (CHAIR_i 24.2–24.8).
- **PCA visualization**: After disentanglement, the originally tightly clustered representations of "dining table" and its co-occurring objects are significantly separated across all layers.

## Highlights & Insights
- **Causal analysis at the representation level** is the core contribution—this is the first work to visualize and quantify the propagation of co-occurrence bias across layers of an MLLM.
- The approach is an end-to-end architectural solution that requires no synthetic data, external models, or post-processing.
- The confounder dictionary is an elegant and concise design—mean representations of 80 object categories prove sufficient.
- The visualization analysis is highly thorough (6 sets of PCA plots covering original/disentangled × visual/textual × multiple objects).

## Limitations & Future Work
- Training requires 8×H20 GPUs, imposing substantial computational demands.
- Confounder estimation may be susceptible to noise or dataset distribution shift.
- The method is evaluated on LLaVA (an earlier MLLM) and has not been validated on more recent models (e.g., InternVL, Qwen-VL).
- The improvement in CHAIR on LLaVA 1.5 is modest (52.1→51.4), likely because version 1.5 has already partially mitigated data bias.

## Related Work & Insights
- **vs. VCD/OPERA (contrastive decoding)**: Contrastive decoding intervenes at inference time without modifying representation learning; Causal-LLaVA addresses the root cause during training.
- **vs. LRV/VIGC (data correction)**: Data-centric methods rely on GPT-4 for generation and carry the risk of error propagation; the proposed method requires no additional data.
- **vs. Deconfounded Captioning**: Prior causal methods apply NWGM approximation only at the output Softmax layer; this work extends causal intervention into the feature space with dual-path intervention at both the projector and Transformer layers.

## Related Work & Insights
- The idea of disentangling co-occurrence bias may be applicable to bias mitigation in Scene Graph Generation.
- The confounder dictionary could be dynamically updated—replacing offline statistics with online estimation to accommodate domain shift.
- Connection to BACL (a concurrent note): BACL improves alignment via hard negative samples, while Causal-LLaVA mitigates bias via causal intervention—the two approaches are potentially complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The causal analysis perspective at the representation level is novel, though causal intervention has precedent in VQA/captioning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10 benchmarks, multiple backbones, and extensive visualizations, but lacks comparison with the latest MLLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ The analysis-driven research paradigm is exemplary, with a clear logical progression from phenomenon to cause to solution.
- Value: ⭐⭐⭐⭐ Provides an architecture-level solution to MLLM hallucination grounded in causal theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models](openhoi_open-world_hand-object_interaction_synthesis_with_multimodal_large_langu.md)

</div>

<!-- RELATED:END -->
