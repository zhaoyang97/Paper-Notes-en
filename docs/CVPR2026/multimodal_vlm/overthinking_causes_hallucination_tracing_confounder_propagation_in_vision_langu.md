---
title: >-
  [Paper Note] Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models
description: >-
  [CVPR 2026][Multimodal VLM][VLM hallucination] This paper reveals a novel mechanism underlying VLM hallucinations — *overthinking*: the model generates an excessive number of competing object hypotheses in intermediate decoding layers, and confounders propagate across layers to corrupt the final prediction. The paper proposes the Overthinking Score to quantify inter-layer hypothesis diversity × uncertainty, achieving F1 of 78.9% on MSCOCO and 71.58% on the OOD benchmark AMBER.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM hallucination
  - Overthinking Score
  - confounder propagation
  - LogitLens
  - inter-layer token dynamics
date: 2026-05-08
content_hash: ac7d1794e9a1de07
---

# Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.07619](https://arxiv.org/abs/2603.07619)
**Code**: None
**Area**: Multimodal VLM / Hallucination Detection / Interpretability
**Keywords**: VLM hallucination, Overthinking Score, confounder propagation, LogitLens, inter-layer token dynamics

## TL;DR

This paper reveals a novel mechanism underlying VLM hallucinations — *overthinking*: the model generates an excessive number of competing object hypotheses in intermediate decoding layers, and confounders propagate across layers to corrupt the final prediction. The paper proposes the Overthinking Score to quantify inter-layer hypothesis diversity × uncertainty, achieving F1 of 78.9% on MSCOCO and 71.58% on the OOD benchmark AMBER.

## Background & Motivation

VLMs frequently generate objects that do not exist in the image (hallucinations). Existing detection methods share two fundamental blind spots:

**Failure of attention-based methods**: Methods such as SVAR assume that attention to visual tokens is lower for hallucinated objects than for real ones. However, correlation analysis in this paper reveals the opposite: under strong scene priors (e.g., a kitchen scene), hallucinated objects that are highly contextually plausible (e.g., "dish" in a kitchen) can attract attention equal to or even greater than that of real objects. The authors demonstrate substantial distributional overlap between hallucinated and real objects in the attention score distributions of SVAR and MetaToken, establishing that attention magnitude cannot reliably distinguish hallucinations.

**Blind spot of final-layer uncertainty**: Methods such as MetaToken use the entropy of the final decoding layer to measure output uncertainty, assuming hallucinations are accompanied by high uncertainty. In practice, however, hallucinations may originate much earlier — intermediate layers activate multiple competing object hypotheses (containing confounders), and once the model "commits" to an erroneous hypothesis, it propagates through subsequent layers, resulting in high confidence (low entropy) at the final layer. The near-complete overlap of final-layer entropy distributions between hallucinated and real tokens across three VLMs validates this critical observation.

**Core insight**: The key to detecting hallucinations lies not in the model's final output but in its *reasoning process*. Tracing the evolution of token hypotheses across intermediate layers via LogitLens reveals a previously overlooked behavior: **overthinking**. The model repeatedly revises its object hypothesis across layers — analogous to human over-deliberation leading to indecision and error. For example, intermediate layers successively produce "sink" → "soap" and other confounding concepts, ultimately causing the model to output "dish," a semantically associated but nonexistent object.

## Method

### Overall Architecture

The detection pipeline consists of four steps: (1) **Prefix Prompting** — after the VLM describes the image, a prefix prompt is constructed for each object token and fed back into the model to predict the next token; (2) **Inter-layer tracking via LogitLens** — the final-layer projection matrix maps each intermediate hidden state to the vocabulary space, yielding per-layer token probability distributions; (3) **Feature extraction** — the Overthinking Score, layer-wise entropy vector, image attention vector, and text attention vector are computed and concatenated into a feature vector; (4) **Lightweight classifier** (LR/GB/MLP) performs token-level binary hallucination detection.

### Key Designs

**1. Discovery and Quantification of Confounder Propagation**

LogitLens projects each intermediate hidden state $h_\ell$ into the vocabulary space:
$$p_\ell(v) = \text{softmax}(W \cdot \text{LayerNorm}(h_\ell))$$
obtaining the top-1 predicted token and probability distribution at each layer. The authors find that the semantic alignment between intermediate-layer top-1 tokens and the final-layer token is remarkably high (LLaVA 40.6%, Gemma-3 47.9%, Qwen3-VL 58.6%), indicating that intermediate-layer "thoughts" genuinely influence the final prediction semantically. When intermediate layers activate concepts contextually related to the eventual hallucinated token (i.e., confounders), confounder propagation occurs. Quantitative analysis shows that 63.69% of hallucinations in LLaVA-1.5, 82.73% in Gemma-3, and 85.46% in Qwen3-VL are attributable to this mechanism — making it the primary cause of hallucination.

**2. Systematic Validation of Three Hypotheses (H1→H2→H3)**

- **H1**: Strong scene priors cause the visual attention distributions of hallucinated and real objects to overlap substantially → attention-based methods fail in such settings.
- **H2**: Intermediate-layer tokens semantically influence final-layer predictions, and confounder propagation occurs across layers → inspecting only the final layer is insufficient.
- **H3**: The greater the number of unique object hypotheses in intermediate layers, the higher the probability of confounder occurrence → candidate diversity is positively correlated with propagation rate.

The three hypotheses form a coherent causal chain, progressing from "why existing methods fail" to "the deep mechanism of hallucination" to "how to quantify it."

**3. Overthinking Score ($S_{OT}$)**

$$S_{OT} = \frac{|\{x_\ell \mid \ell \in [1,L]\}|}{L} \cdot \frac{\sum_{\ell=1}^{L} H_\ell}{L}$$

The first term — the number of unique top-1 tokens across layers divided by the total number of layers — measures *how many distinct objects the model considered* (hypothesis diversity). The second term — the mean per-layer entropy — measures *how uncertain the model is at each layer*. Their product captures the overthinking state of "entertaining too many alternatives while remaining uncertain at every layer." SHAP analysis shows that the feature importance of $S_{OT}$ (~0.007) substantially exceeds that of image attention, text attention, and entropy (~0.002–0.004 each), confirming its role as the primary indicator.

**4. Multimodal Attention Features (Auxiliary Signals)**

- Image attention: $\alpha_\ell^{img} = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \max_h A_\ell^{(h)}(t,i)$ — mean attention of the next token over image tokens.
- Text attention: $\alpha_\ell^{text}$ — mean attention of the next token over preceding text tokens.
- High text attention indicates reliance on linguistic priors rather than visual evidence, which is positively correlated with hallucination.

### Loss & Training

The feature vector is $\phi(x_t) = [S_{OT} \| \mathbf{H} \| \boldsymbol{\alpha}^{img} \| \boldsymbol{\alpha}^{text}]$ with dimensionality $3L+1$. Three lightweight classifiers are employed: LR (L-BFGS, 2000 iterations), GB (200 trees, max depth 10, learning rate 0.1), and MLP (128 hidden units + ReLU, 2000 epochs, learning rate 0.01). Hyperparameters are optimized for F1 via grid search on a validation set. Data: 4,000 images from the MSCOCO 2014 validation set, split 90%/10% for training/testing, with token-level hallucination labels annotated by GPT-4o.

## Key Experimental Results

### Main Results: MSCOCO Hallucination Detection (AUC / F1 %)

| Method | Classifier | LLaVA AUC | LLaVA F1 | Gemma-3 AUC | Gemma-3 F1 | Qwen3 AUC | Qwen3 F1 | Avg AUC | Avg F1 |
|--------|------------|-----------|----------|-------------|------------|-----------|----------|---------|--------|
| SVAR | MLP | 85.12 | 69.35 | 74.11 | 47.84 | 75.56 | 50.20 | 78.26 | 55.80 |
| HalLoc | — | 80.38 | 73.68 | 79.27 | 67.11 | 83.85 | 74.75 | 81.17 | 71.85 |
| MetaToken | GB | 88.95 | 75.95 | 77.23 | 67.15 | 84.21 | 74.43 | 83.46 | 72.51 |
| **Ours** | **GB** | **89.66** | **78.95** | **85.59** | **74.54** | **86.65** | **74.43** | **87.30** | **75.97** |
| **Ours** | **MLP** | **89.73** | **75.37** | **85.38** | **72.07** | **86.89** | **71.15** | **87.33** | **72.86** |

OOD generalization (AMBER + LLaVA-1.5): **Ours GB 86.11 AUC / 71.58 F1** vs. MetaToken GB 82.15 / 65.54.

### Ablation Study: Layer Selection and Feature Contribution

| Configuration | AUC | F1 | Notes |
|---------------|-----|-----|-------|
| All Layers [0–31] | **89.73** | **75.37** | Best |
| Layers [19–31] | 88.93 | 74.75 | Deep layers contribute most |
| Layers [5–18] | 87.37 | 71.61 | Middle layers second |
| Layers [0–4] | 85.14 | 67.67 | Shallow layers limited |
| Last Layer Only | 83.79 | 68.76 | Far insufficient |
| Remove $S_{OT}$ | 86.58 | — | **−3.15% AUC, largest drop** |

**Key Findings**:

- $S_{OT}$ is the most critical feature: removing it causes a 3.15% AUC drop, compared to ≤1.4% for any other single feature.
- $S_{OT}$ can be plug-and-played to improve all baselines: SVAR +1.55, HalLoc +8.15, MetaToken +1.55–2.42 AUC.
- All layers contribute, but deeper layers matter more: All (89.73) > [19–31] (88.93) > [5–18] (87.37) > [0–4] (85.14).
- SHAP analysis confirms that Mean Entropy and Unique Token Count each contribute independently, but their product ($S_{OT}$) yields the clearest and most stable signal.

## Highlights & Insights

- **The "overthinking" metaphor is apt**: repeatedly revising object hypotheses across layers → indecision → incorrect output — a perfect analogy to human over-deliberation, highly intuitive.
- **First systematic demonstration of the causal relationship between inter-layer token hypothesis dynamics and VLM hallucination**: 63–85% of hallucinations are attributed to confounder propagation, suggesting hallucination stems primarily from erroneous reasoning rather than perceptual failure.
- **A compelling refutation of attention-based methods**: the case where "book" receives high attention but is misidentified as "laptop" under a strong scene prior is particularly illustrative.
- **$S_{OT}$ is minimal yet powerful**: a single scalar (unique token count × mean entropy) improves every detection baseline, offering strong practical utility.
- **Hypothesis-driven analytical paradigm (H1→H2→H3)**: a methodologically instructive approach — first demonstrate the failure of existing methods, then discover a new mechanism, then quantify it.

## Limitations & Future Work

- **Detection without mitigation**: the overthinking signal has not yet been used for real-time intervention (e.g., early exit or resetting intermediate representations upon detecting high $S_{OT}$); extending from diagnosis to treatment is a natural next step.
- **Reliance on GPT-4o annotations**: token-level hallucination labels are generated by GPT-4o and are subject to its capabilities and biases.
- **LogitLens assumption**: directly applying the final-layer projection matrix to decode intermediate layers assumes a "linear readout" that may not hold for all architectures or layer depths.
- **Limited model scale**: experiments cover only 4B–7B models; overthinking patterns in models with 70B+ parameters may differ substantially.
- **Hallucination types not differentiated**: only object hallucination is analyzed; whether attribute, relation, or counting hallucinations also manifest as overthinking remains unexplored.

## Related Work & Insights

- **vs. SVAR (attention-based)**: SVAR assumes hallucination correlates with low visual attention, which completely breaks down under strong scene priors. This paper demonstrates that inter-layer token dynamics are more fundamental than attention magnitude.
- **vs. MetaToken (final-layer entropy)**: MetaToken relies on final-layer probability distribution features. This paper shows that hallucinations can be output with high confidence, and final-layer entropy distributions overlap extensively.
- **vs. HalLoc (external features)**: HalLoc uses CLIP and VisualBERT for external detection. This paper operates from internal reasoning dynamics and substantially outperforms HalLoc on Gemma-3 and Qwen3.
- **vs. PROJECTAWAY**: both use LogitLens but for different purposes — PROJECTAWAY projects image patches into text space, whereas this paper decodes hidden states to track what the model is "thinking."
- **Potential extension**: combining overthinking detection with inference-time intervention (e.g., adaptive head amplification or early exit) could form a closed-loop diagnosis–mitigation framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The overthinking concept and confounder propagation mechanism represent a genuinely new perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three VLMs, ID+OOD datasets, complete ablation and SHAP analysis; mitigation experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Hypothesis-driven analytical logic is clear; case studies are highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — $S_{OT}$ is plug-and-play and improves all detectors; engineering utility is high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](../../ACL2026/multimodal_vlm/addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/multimodal_vlm/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[CVPR 2026\] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multimodal_la.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)

</div>

<!-- RELATED:END -->
