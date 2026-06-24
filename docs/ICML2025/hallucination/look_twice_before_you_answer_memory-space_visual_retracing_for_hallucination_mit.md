---
title: >-
  [Paper Note] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models
description: >-
  [ICML2025][Hallucination Detection][MLLM] This paper proposes the MemVR decoding paradigm, which reinjects visual tokens as supplementary evidence into intermediate trigger layers through the key-value memory mechanism of FFNs. This "look-twice" mechanism mitigates hallucinations in MLLMs without introducing additional inference overhead.
tags:
  - "ICML2025"
  - "Hallucination Detection"
  - "MLLM"
  - "hallucination"
  - "visual retracing"
  - "FFN key-value memory"
  - "decoding strategy"
date: 2026-05-08
content_hash: da71f9101276a592
---

# Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models

**Conference**: ICML2025  
**arXiv**: [2410.03577](https://arxiv.org/abs/2410.03577)  
**Code**: [GitHub](https://github.com/1zhou-Wang/MemVR)  
**Area**: Hallucination Detection  
**Keywords**: MLLM, hallucination, visual retracing, FFN key-value memory, decoding strategy

## TL;DR
This paper proposes the MemVR decoding paradigm, which reinjects visual tokens as supplementary evidence into intermediate trigger layers through the key-value memory mechanism of FFNs. This "look-twice" mechanism mitigates hallucinations in MLLMs without introducing additional inference overhead.

## Background & Motivation
- MLLMs (e.g., LLaVA) suffer from "forgetting" visual information during generation; the LLM decoder relies increasingly on text tokens as reasoning goes deeper.
- **Finding 1**: Amplifying image features has a greater impact on performance than amplifying text features, which indicates that the text decoder is biased towards text tokens.
- **Finding 2**: Hallucinated tokens exhibit higher uncertainty (higher entropy) when generated in the intermediate and deep layers.
- **Finding 3**: Solely supplementing visual information (rather than text or both) yields the best performance.
- Limitations of Prior Work:
    - Contrastive decoding (e.g., VCD) requires dual-round inference, which doubles the latency.
    - Attention intervention (e.g., OPERA) incurs 3.66 times the latency of the base method.
    - Fine-tuning methods require additional data and training costs.

## Method

### Core Idea: Look-Twice Mechanism
When the model exhibits high uncertainty in the intermediate layers, visual tokens are reinjected as "supplementary evidence" to correct predictions.

### FFN from a Key-Value Memory Perspective
Reinterpreting the FFN as key-value memory retrieval:
$$\text{FFN}(\mathbf{x}) = \sum \phi(\langle \mathbf{x}, \mathbf{k}_i \rangle) \cdot \mathbf{v}_i$$

### Visual Retracing (VR)
Injects visual memory into the FFN of the $l$-th layer:
$$\text{FFN}^{(l)}(\mathbf{x} \propto \mathbf{z}_v) = \alpha \underline{\Delta} + (1-\alpha)\text{FFN}^{(l)}(\mathbf{x})$$

Where the visual retrieval term is formulated as:
$$\underline{\Delta}(\mathbf{z}_v | \mathbf{x}) = \sum_{i=1}^{N_v} \phi(\langle \mathbf{x}, \mathbf{z}_{v,i} \rangle) \cdot \mathbf{z}_{v,i}$$

- $\alpha \in [0,1]$: Injection ratio, proportional to image complexity.
- $N_v \ll D$ (e.g., 256 vs 11008), resulting in negligible computational overhead.

### Trigger Layer Selection
- **Static VR**: Fixed on a specific intermediate layer.
- **Dynamic VR**: Dynamically selects the trigger layer based on uncertainty scores, injecting visual memory into layers with higher uncertainty.

## Key Experimental Results

| Method | Latency (ms/token) | POPE ↑ | MME ↑ | CHAIR ↓ |
|------|-----------------|--------|-------|---------|
| Greedy | 65.71 (1.00×) | Baseline | Baseline | Baseline |
| VCD | 144.62 (2.20×) | +Slight | Negative | +Slight |
| OPERA | 240.59 (3.66×) | +Moderate | Negative | +Moderate |
| MemVR | 68.32 (1.04×) | **+7.0%** | **+32.2** | **-15.6%** |

- MemVR only introduces a 4% latency overhead, whereas VCD increases latency by 120% and OPERA by 266%.
- Memory footprint increases by only 1% (14345 vs 14257 MB).
- MemVR outperforms other baselines across multiple benchmarks such as POPE, MME, and CHAIR.
- Universal effectiveness is validated across multiple MLLMs (LLaVA-1.5, Qwen-VL, GLM4V, etc.).

## Highlights & Insights
- **Ultra-simple and Efficient**: Plug-and-play, training-free, requires no external data, and introduces zero extra parameters.
- **Unique Theoretical Insight**: Interprets VR as information retrieval from the perspective of FFN = key-value memory.
- **Simultaneously Improving Hallucination Mitigation and General Capability**: It is the only method that shows positive impacts on both dimensions simultaneously (Table 2).
- **Modality Imbalance Analysis**: Systematically validates that the root cause of hallucinations is the forgetting of visual information in deeper layers.

## Limitations & Future Work
- $\alpha$ requires manual tuning or heuristic configuration, lacking an adaptive selection mechanism.
- Verified only in image-text scenarios; multi-modal scenarios like video/audio remain to be explored.
- Dynamic VR requires additional computation of uncertainty; although the overhead is minimal, it increases the complexity of the method.
- Its effectiveness on models with very few visual tokens (e.g., highly compressed visual encoders) remains to be validated.
- The selection strategy of trigger layers significantly impacts performance; different model architectures may require different strategies.
- The stability of VR has not been analyzed in long-text generation scenarios (e.g., detailed image description).
- The integration with other methods such as RAG and fine-tuning remains unexplored.
- In multi-image input scenarios (e.g., multi-turn conversations), selecting which image's visual tokens to inject remains an open question.

### Supplementary Analysis
- Normalized entropy is used for uncertainty measurement: $u = \sum -p_i \log p_i / \log N$
- LLaVA-1.5-7B is a 32-layer Transformer with 256 visual tokens.
- The key innovation of MemVR is that it directly modifies the hidden states instead of modifying logits (unlike contrastive decoding methods).
- Experiments show that the uncertainty of intermediate layers (layers 14-18) is most sensitive to hallucination prediction.
- The method can be directly generalized to any MLLM based on a Transformer decoder.

## Related Work & Insights
- **VCD** (Leng et al., 2024): Performs contrastive decoding with noisy visual inputs, but doubles the inference cost.
- **OPERA** (Huang et al., 2024): Attention matrix intervention, suffering from high latency.
- **DoLa** (Chuang et al., 2023): Contrastive decoding between layers, which is ineffective for visual hallucinations.
- **PAI** (Liu et al., 2024): Proposes focusing more on images, which aligns with the findings of this paper.
- Insight: Treating FFNs as memory provides a new perspective for understanding the internal mechanisms of MLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The look-twice + FFN memory explanation framework is highly novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 benchmarks + multiple models + efficiency analysis + GPT-4o evaluation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Solid motivation analysis, clear figures and tables)
- Value: ⭐⭐⭐⭐⭐ (A truly practical method balancing performance and efficiency)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](../../ICLR2026/hallucination/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/hallucination/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](../../ACL2025/hallucination/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2025\] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/visual_evidence_prompting.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)

</div>

<!-- RELATED:END -->
