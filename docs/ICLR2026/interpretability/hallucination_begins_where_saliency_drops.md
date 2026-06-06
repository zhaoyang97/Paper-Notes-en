---
title: >-
  [Paper Note] Hallucination Begins Where Saliency Drops
description: >-
  [ICLR 2026][Interpretability][Hallucination Mitigation] This paper proposes LVLMs-Saliency, a gradient-aware diagnostic framework that quantifies the visual grounding strength of each output token. It identifies a key fi…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Hallucination Mitigation"
  - "Large Vision-Language Models"
  - "Saliency Analysis"
  - "Attention Mechanism"
  - "Inference-Time Intervention"
date: 2026-05-08
content_hash: af97058426f190d5
---

# Hallucination Begins Where Saliency Drops

**Conference**: ICLR 2026
**arXiv**: [2601.20279](https://arxiv.org/abs/2601.20279)  
**Code**: [https://github.com/zhangbaijin/LVLMs-Saliency](https://github.com/zhangbaijin/LVLMs-Saliency)  
**Area**: Interpretability
**Keywords**: Hallucination Mitigation, Large Vision-Language Models, Saliency Analysis, Attention Mechanism, Inference-Time Intervention

## TL;DR

This paper proposes LVLMs-Saliency, a gradient-aware diagnostic framework that quantifies the visual grounding strength of each output token. It identifies a key finding: hallucinations arise when the saliency of previously generated tokens toward the next token prediction drops. Building on this insight, the paper introduces a dual-mechanism inference-time framework combining SGRS (Saliency-Guided Rejection Sampling) and LocoRE (Local Coherence Reinforcement), achieving significant hallucination reduction across multiple LVLMs.

## Background & Motivation

Large vision-language models (LVLMs) such as LLaVA, Qwen2-VL, and InternVL have achieved remarkable progress on cross-modal tasks, yet **hallucination** remains a central challenge—models may generate objects or attributes absent from the input image.

Existing mitigation strategies fall into two categories: methods requiring retraining (e.g., fine-tuning on additional data) and training-free methods (e.g., OPERA, VCD, DOPRA). The latter primarily target the attention sink phenomenon—certain tokens persistently attract high attention weights during subsequent generation, potentially causing hallucinations. However, these approaches share a critical limitation: **attention maps only reflect forward-pass decisions and cannot capture the influence chain from input tokens to final outputs**. Experiments show that attention maps alone are nearly insufficient to distinguish correct token generation patterns from hallucinated ones (Figure 1).

The core insight of this paper is that defining the element-wise product of the attention weight $\mathbf{A}$ and its gradient $\nabla \mathbf{A}$ as **Saliency** reveals a decisive pattern: hallucinations occur precisely when the saliency of preceding output tokens declines. This indicates that the model "forgets" recent context, and the collapse of contextual memory leads to semantically incoherent outputs.

## Method

### Overall Architecture

The overall approach consists of three levels: (1) the LVLMs-Saliency diagnostic framework, which quantifies the hallucination risk of each token; (2) SGRS, which proactively filters low-saliency candidates before they are committed to the sequence; and (3) LocoRE, which reinforces attention to recent context after a token is accepted. SGRS and LocoRE form a closed loop: SGRS acts as a "gatekeeper" preventing low-quality tokens from entering, while LocoRE acts as a "stabilizer" preventing accepted tokens from being forgotten.

### Key Designs

1. **LVLMs-Saliency Diagnostic Framework**: For model $\mathcal{M}$, the saliency matrix is computed at each layer $l$ and attention head $h$ from the attention matrix $\mathbf{A}^{(l,h)}$:

    $\mathbf{S}^{(l,h)} = \text{tril}(|\mathbf{A}^{(l,h)} \odot \nabla \mathbf{A}^{(l,h)}|)$

   The layer-level normalized saliency is obtained by averaging across heads followed by $\ell_2$ normalization: $\bar{\mathbf{S}}^{(l)} = \frac{\sum_h \mathbf{S}^{(l,h)}}{\|\sum_h \mathbf{S}^{(l,h)}\|_2}$. The key finding is that **saliency maps of correct tokens exhibit strong dependence on recent tokens (decaying with distance), while those of hallucinated tokens show a global collapse**. This pattern is validated through statistical analysis over 500 samples and across three models (LLaVA-1.5, Qwen2-VL, InternVL).

2. **Saliency-Guided Rejection Sampling (SGRS)**: At decoding position $P$, after the model produces logits, a candidate set $\mathcal{C}^{(P)}$ is obtained via top-$K$ sampling. For each candidate $c_i$, the hallucination saliency is computed as:

    $\mathcal{S}(c_i) = \frac{1}{|\mathcal{L}_{\text{target}}| \cdot |\mathcal{J}|} \sum_{l \in \mathcal{L}_{\text{target}}} \sum_{j \in \mathcal{J}} \bar{\mathbf{S}}_{P,j}^{(l)}$

   A candidate is accepted only if $\mathcal{S}(c_i) \geq \tau^{(P)}$, where the adaptive threshold is computed from the historical average saliency of the most recent $W$ output tokens: $\tau^{(P)} = \alpha \cdot \frac{1}{|\mathcal{H}|}\sum_{j \in \mathcal{H}} \mathcal{S}(x_j)$, with $\alpha \in (0,1)$ controlling sensitivity. Resampling is performed at most $R$ times; if all candidates are rejected, the one with the highest saliency is selected.

3. **Local Coherence Reinforcement (LocoRE)**: A lightweight plug-and-play module that modifies attention weights at each decoding step. For prediction at position $P+1$, attention to the most recent $w_s$ output tokens is reinforced:

    $\gamma_j^{(P)} = 1 + \beta \cdot \mathbb{I}((P - j) \leq w_s)$

   where $\beta \geq 0$ is the reinforcement strength. This directly amplifies the influence of recent context on the current prediction, counteracting the "forgetting" behavior caused by saliency decay. LocoRE operates purely on the attention structure—requiring no gradient computation or model parameter modification—adding less than 2% latency.

### Loss & Training

This method is a purely inference-time intervention requiring **no training or fine-tuning**. SGRS requires one additional backward pass to compute gradient saliency (adding 30–40% latency), while LocoRE only modifies attention weights (negligible additional latency). In practice, LocoRE alone achieves most of the gains with virtually no speed penalty.

## Key Experimental Results

### Main Results

Comparison on POPE, CHAIR, and MME with LLaVA-1.5-7B as the baseline model:

| Method | POPE F1 | POPE Acc | CHAIR S↓ | CHAIR I↓ | MME Total |
|--------|---------|----------|----------|----------|-----------|
| Beam Search (Baseline) | 85.4 | 84.0 | 51.0 | 15.2 | 565.34 |
| OPERA (CVPR 2024) | 84.2 | 85.2 | 47.0 | 14.6 | 549.00 |
| EAH (EMNLP 2025) | 85.7 | 86.0 | 36.4 | 9.9 | 603.99 |
| CausalLLM (ICLR 2025) | 86.0 | 86.5 | - | - | 656.00 |
| MemVR (ICML 2025) | 87.1 | 87.4 | 46.6 | 13.0 | 648.30 |
| LocoRE (Ours) | 86.9 | 87.3 | 38.4 | 11.2 | 656.66 |
| **SGRS + LocoRE (Ours)** | **87.0** | **87.5** | **35.6** | **8.2** | **668.33** |

Cross-model results (gains of SGRS + LocoRE over baseline):

| Model | LLaVAW | MM-Vet | VizWiz | CHAIR S↓ | POPE Acc |
|-------|--------|--------|--------|----------|----------|
| LLaVA-1.5-7B | +4.2 | +5.5 | +6.4 | +12.4 | +3.5 |
| LLaVA-1.5-13B | +4.3 | +5.9 | +3.5 | +7.4 | +0.4 |
| Qwen2-VL-7B | +4.1 | +4.5 | +3.0 | +5.7 | +1.4 |
| InternVL-7B | +3.9 | +5.0 | +4.5 | +12.2 | +1.5 |

### Ablation Study

Hyperparameter ablation on $\alpha$ (SGRS) and $\beta$ (LocoRE) with LLaVA-1.5-7B:

| $\alpha$ | $\beta$ | CHAIR S↓ | POPE F1 | Note |
|----------|---------|----------|---------|------|
| 0.0 | 0.0 | 48.0 | 85.4 | Baseline |
| 0.0 | 0.15 | 38.4 | 86.9 | LocoRE only |
| 0.6 | 0.0 | 36.5 | 86.9 | SGRS only |
| 0.6 | 0.15 | **35.6** | **87.0** | Full method |
| 0.6 | 1.0 | 50.2 | 60.3 | Degradation from over-reinforcement |

### Key Findings

- **The core pattern is validated across model architectures**: Across LLaVA-1.5, Qwen2-VL, and InternVL, hallucinated tokens consistently exhibit lower saliency than correct tokens; the hallucination rate reaches 68%–76% in the lowest saliency quartile and drops to 18%–28% in the highest
- **Causal validation experiment**: Artificially reducing the saliency of correct tokens (decay factor from 1.0 to 0.2) raises the CHAIR hallucination rate from 35.6 to 56.0, directly establishing a causal relationship
- **LocoRE alone offers the best cost-effectiveness**: Adding less than 2% latency to achieve most of the improvement makes it the preferred choice for practical deployment
- **$\alpha = 0.6$ is the optimal trade-off for SGRS**: It suppresses 28.3%+ of hallucinations while maintaining inference speed and output quality; $\alpha = 0.9$ further reduces hallucinations but incurs 33% additional latency

## Highlights & Insights

- **Paradigm shift from "attention → saliency"**: Attention maps alone cannot distinguish correct from hallucinated tokens, but gradient-weighted saliency achieves clear separation—representing a meaningful advance in understanding LVLM hallucination
- **A concise yet powerful finding**: "Hallucination begins where saliency drops"—when the model forgets recent output context, it generates semantically incoherent content. This finding is also highly intuitive
- **Elegant closed-loop dual-mechanism design**: The collaboration between SGRS (gating) and LocoRE (stabilizing) is well-structured with a clear division of responsibilities
- **High practical utility of LocoRE**: Training-free, requiring no additional model, no gradient computation, plug-and-play, and virtually zero latency—highly suitable for industrial deployment
- **Compelling visualization analysis**: The saliency contrast between correct and hallucinated tokens is visually clear and persuasive

## Limitations & Future Work

- **Latency overhead of SGRS**: Each token requires an additional backward pass (30–40% overhead), making it unsuitable for real-time applications. The paper acknowledges that LocoRE alone is the more practical choice
- **Failure cases**: For "high-confidence hallucinations" (where the model generates incorrect content with high certainty), saliency may remain high and SGRS cannot detect them—consistent with the observation that models can err confidently
- **Context-agnostic generation**: When the model generates content that drifts from the current context (e.g., hallucinating an entirely new topic), SGRS may be insufficient
- **Focus solely on textual output saliency**: Visual saliency is entirely ignored; however, analysis over 500 samples leads the authors to conclude that prompt saliency is not the primary driver of hallucination
- **Hyperparameters require per-model tuning**: LLaVA-1.5 uses $\beta = 0.15$ while Qwen2-VL uses $\beta = 0.20$, necessitating manual search

## Related Work & Insights

- **OPERA, DOPRA**: Mitigate hallucinations by penalizing the logits of attention sink tokens; this paper argues that attention alone is insufficient and gradient information is necessary
- **EAH**: Enhances visual information by replacing shallow attention heads—effective, but may reduce output diversity (lower Recall); LocoRE maintains higher Recall
- **TAME, Farsight**: Analyze local self-attention patterns of anchor tokens, but overlook contextual dependencies in textual outputs
- **Insights**: Gradient-weighted attention as a saliency measure has been applied in NLP for attribution and explanation, but this paper is the first to systematically apply it to LVLM hallucination diagnosis, demonstrating the great potential of "established methods in new settings"
- **Generalizability**: The saliency analysis framework is extensible to quality control in any autoregressive generation model

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](../../CVPR2026/interpretability/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](../../ACL2026/interpretability/mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ICML 2026\] Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](../../ICML2026/interpretability/finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)
- [\[NeurIPS 2025\] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models](../../NeurIPS2025/interpretability/fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)
- [\[NeurIPS 2025\] Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT](../../NeurIPS2025/interpretability/beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)

</div>

<!-- RELATED:END -->
