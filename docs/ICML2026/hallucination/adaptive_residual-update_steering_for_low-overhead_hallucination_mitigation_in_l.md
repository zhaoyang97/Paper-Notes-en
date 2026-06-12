---
title: >-
  [Paper Note] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models
description: >-
  [ICML 2026][Hallucination Detection][LVLM Hallucination] This paper proposes RUDDER, which extracts per-sample visual evidence directions from residual updates during the prefill phase of LVLMs and adaptively injects the…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "LVLM Hallucination"
  - "inference-time steering"
  - "residual stream"
  - "Beta Gate"
  - "visual grounding"
date: 2026-05-08
content_hash: a33cdf990f3c5749
---

# Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models

**Conference**: ICML 2026  
**arXiv**: [2511.10292](https://arxiv.org/abs/2511.10292)  
**Code**: Yes (Referred to as RUDDER, complete URL not provided in cache)  
**Area**: Hallucination Detection  
**Keywords**: LVLM Hallucination, inference-time steering, residual stream, Beta Gate, visual grounding  

## TL;DR
This paper proposes RUDDER, which extracts per-sample visual evidence directions from residual updates during the prefill phase of LVLMs and adaptively injects them via a Beta Gate during decoding to reduce object hallucinations with overhead close to a single forward pass.

## Background & Motivation
**Background**: Large Vision Language Models (LVLMs) typically treat image tokens as prefixes for the language decoder and generate text autoregressively. As the number of generation steps increases, the information from the image prefix is gradually diluted by language priors, making the model prone to adding non-existent objects in its descriptions.

**Limitations of Prior Work**: Existing inference-time intervention methods often perform contrastive decoding on logits or correct outputs through iterative feedback. While these methods reduce hallucinations, they typically require additional forward passes, image perturbations, external classifiers, or multiple rounds of refinement, leading to high latency and throughput overhead. For real-world deployment, especially in long-text generation, this cost is difficult to accept.

**Key Challenge**: Reducing hallucinations requires continuously reminding the model to focus on visual evidence, yet forcing fixed steering may destroy fluency, recall, and general multimodal capabilities. Models need a lightweight control mechanism that "reminds visual evidence only on the appropriate tokens."

**Goal**: The authors aim to transform the visual information already existing in the prefill phase into a sustainable visual anchor and suppress object hallucinations at low cost during decoding, without modifying model weights or adding extra forward passes.

**Key Insight**: The paper observes that the residual update of the self-attention sub-layer during the prefill phase contains the net impact of the image on the text representation. Since prefill is a mandatory step for LVLM generation, caching a visual evidence direction from it incurs almost zero additional cost.

**Core Idea**: Extract CARD visual evidence vectors from prefill residual updates and adaptively inject them per token during decoding using Beta distribution gating.

## Method
The key to RUDDER is not retraining the LVLM but attaching two lightweight modules to the standard generation pipeline. The first module reads the self-attention residual update of a specific decoder layer during the prefill phase and aggregates it into an input-dependent CARD vector. The second module calculates a Beta Gate at each decoding step based on the similarity between the current hidden state and the CARD vector to decide whether and how strongly to inject the CARD into the residual stream.

### Overall Architecture
Given an image and text prompt, the LVLM first executes prefill, processing image and prompt tokens and building the KV cache. RUDDER places a read-only hook at the target layer to collect the self-attention output (residual update) for each token in the prefill span. It performs mean or norm-weighted mean pooling on these updates followed by $L_2$ normalization to obtain the sample-specific visual evidence direction $v_{\mathrm{CARD}}$.

During autoregressive decoding, RUDDER continues to operate at the same target layer. For each generated answer token, it first calculates the cosine similarity $s_t$ between the current hidden state $h_{l,t}$ and $v_{\mathrm{CARD}}$; then $s_t$ is mapped to two parameters of a Beta distribution, with $g_t=\alpha_t/(\alpha_t+\beta_t)$ taken as the gate. The final injected vector is $(\alpha_{\max}g_t)v_{\mathrm{CARD}}$, added to the residual stream after self-attention.

### Key Designs
1.  **CARD Visual Evidence Direction**:
    - **Function**: Extracts a persistent visual anchor for each input without extra forward passes.
    - **Mechanism**: Caches the target layer self-attention residual update $\Delta_i^l$ during the prefill phase, pools them over the prefill token set, and normalizes them as $v_{\mathrm{CARD}}=\mathrm{Pool}(\{\Delta_i^l\})/\|\mathrm{Pool}(\{\Delta_i^l\})\|_2$. Since residual updates represent the newly added information after vision-text fusion, the aggregated direction can be viewed as a summary of the visual evidence for that sample.
    - **Design Motivation**: Hallucinations often stem from the generation process gradually shifting toward language priors; CARD preserves information from the strongest vision-fusion stage, which can repeatedly remind the model later.

2.  **Beta Gate Adaptive Gating**:
    - **Function**: Allows visual reminders to adjust intensity per token, avoiding damage to grammatical tokens and non-visual content by fixed steering.
    - **Mechanism**: Calculates $s_t=\cos(h_{l,t},v_{\mathrm{CARD}})$, then uses $\alpha_t=\mathrm{softplus}(ks_t+c)$ and $\beta_t=\mathrm{softplus}(-ks_t+c)$ to get $g_t=\alpha_t/(\alpha_t+\beta_t)$. High similarity indicates the current generation trajectory reliably follows the visual evidence direction, enhancing the gate; low or negative similarity suppresses injection.
    - **Design Motivation**: It is not an error detector but a trust mechanism. The closer the model's current state is to the visual evidence, the safer it is to continue strengthening it; when the state deviates or is generating grammatical function words, strong injection may instead destroy fluency.

3.  **Single-pass Integration and Lightweight Calibration**:
    - **Function**: Enables deployment feasibility.
    - **Mechanism**: CARD comes from the mandatory prefill, and Beta Gate adds only a small amount of vector operations during decoding. Hyperparameters are calibrated once using 100 held-out MSCOCO images to select the target layer, maximum intensity $\alpha_{\max}$, and sensitivity $k$, while constraining recall to maintain at least 95% of the vanilla performance.
    - **Design Motivation**: If hallucination mitigation relies on multiple forward passes for performance, it is hard to implement in online generation; RUDDER places computation within existing paths, focusing on the effect-efficiency trade-off.

### Loss & Training
RUDDER is a training-free inference-time intervention with no new training loss. Calibration is only used to select deployment hyperparameters: $L=30$ for LLaVA-1.5, $L=28$ for Idefics2, and $L=1$ for InstructBLIP; corresponding $(\alpha_{\max},k)$ are $(20,5.0)$, $(8.0,5.0)$, and $(6.5,8.0)$ respectively. Gating concentration $c=1$, and the gate is clamped to $[0.05, 1]$ to avoid complete closure or saturation.

## Key Experimental Results

### Main Results
The paper evaluates hallucinations, object QA, and general multimodal capabilities on CHAIR, POPE, and MME. Representative results under greedy decoding are extracted below.

| Dataset/Metric | Model | Vanilla | RUDDER-Beta | Gain |
| :--- | :--- | :--- | :--- | :--- |
| CHAIR $C_S/C_I$ ↓ | LLaVA-1.5 | 48.6 / 13.6 | 39.5 / 10.5 | Reduction in both sentence and object levels |
| CHAIR $C_S/C_I$ ↓ | Idefics2 | 46.6 / 14.9 | 28.4 / 10.9 | Most significant reduction in sentence level |
| CHAIR $C_S/C_I$ ↓ | InstructBLIP | 39.2 / 12.8 | 27.1 / 8.5 | Lower hallucination with recall constraint |
| POPE Acc/F1 ↑ | LLaVA-1.5 | 85.34 / 84.91 | 86.53 / 86.03 | Slight improvement in recognition |
| POPE Acc/F1 ↑ | Idefics2 | 78.40 / 74.86 | 78.74 / 76.52 | More significant F1 improvement |
| POPE Acc/F1 ↑ | InstructBLIP | 85.74 / 84.75 | 86.02 / 84.93 | Minimal damage to QA capability |
| MME ↑ | Idefics2 | 1518.84 | 1540.56 | Improvement in general capability |
| MME ↑ | InstructBLIP | 1566.77 | 1592.07 | Improvement in general capability |

### Ablation Study
The analysis focuses on adaptive gating, layer selection, intensity sensitivity, and efficiency.

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| RUDDER-Beta vs RUDDER-Add | Beta is more stable on CHAIR | Token-wise gate is better for precise suppression in open-captioning |
| RUDDER-Add | Sometimes stronger for InstructBLIP on POPE | Fixed steering is effective for short yes/no tasks on some architectures |
| Idefics2 Layer Ablation | Optimal layer around $L=28$ | Mid-late layers best affect output while preserving visual semantics |
| Idefics2 HP Heatmap | $\alpha_{\max}=8.0, k=5.0$ most balanced | Higher intensity reduces CHAIR but excessive intensity hurts recall |
| Throughput tokens/s | Vanilla 56.7, RUDDER-Beta 54.9 | RUDDER-Beta maintains ~96.0% vanilla throughput, faster than multi-forward methods |
| Scaling to 13B/Qwen2.5-VL | LLaVA-13B POPE F1 85.5 | Scalable to larger models and different fusion architectures |

### Key Findings
- **Efficiency**: RUDDER's value lies in efficiency; it matches the hallucination mitigation of VISTA while maintaining ~96% throughput (VISTA is ~58.1%).
- **Per-sample Nature**: CARD's sample-specific property is crucial. It is not an offline-learned general direction but vision evidence extracted from the current residual update.
- **Adaptive Gating**: Ideal for long text and open descriptions as it reinforces vision on content words while avoiding over-intervention on non-visual tokens.

## Highlights & Insights
- Using the prefill residual update as a "visual evidence cache" is ingenious. This signal is already computed; RUDDER explicitly preserves and reuses it.
- The Beta Gate interpretation is more semantic than a sigmoid gate: it treats similarity as a pseudo-count, estimating the confidence that the "current trajectory follows visual evidence."
- The evaluation is restrained, checking not only CHAIR reduction but also recall constraints, POPE, MME, and throughput to ensure low hallucination isn't traded for "talking less" or "slowing down."

## Limitations & Future Work
- Target layers and intensities still need tuning for different architectures; the paper acknowledges hyperparameter sensitivity. Future work could study automatic layer selection.
- CARD comes from a single-layer residual update and may not cover complex errors requiring multi-layer or multi-scale visual reasoning (e.g., relations, counting, OCR).
- The high-similarity enhancement assumption works for object description, but if a model is already generating confidently in a wrong visual direction, simple reinforcement might not correct it.

## Related Work & Insights
- **vs VCD / PAI / HALC**: These methods mostly operate on logits or contrastive contexts and often require extra forwards; RUDDER modifies the hidden residual stream directly.
- **vs VISTA / ASD**: These steering methods also modify representations but rely on predefined directions or extra computation; CARD is extracted on-the-fly.
- **Future Inspiration**: Prefill phases may contain reusable evidence for other tasks, such as multimodal safety, OCR fidelity, and long-context memory in video.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The combination of CARD + Beta Gate is simple yet effective; the source of the signal is clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple models, decoding strategies, hallucination, general capability, and efficiency.
- **Writing Quality**: ⭐⭐⭐⭐☆ Structure is clear, though some tables are dense across models.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses deployment pain points for LVLM hallucination mitigation with a balance of effect and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](../../ICLR2026/hallucination/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
