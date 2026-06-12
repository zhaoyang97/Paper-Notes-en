---
title: >-
  [Paper Note] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations
description: >-
  [NeurIPS 2025][Hallucination Detection][semantic hallucination] This paper identifies a "semantic hallucination" problem in Large Multimodal Models (LMMs) for scene text recognition—where non-semantic text is misread as…
tags:
  - "NeurIPS 2025"
  - "Hallucination Detection"
  - "semantic hallucination"
  - "scene text recognition"
  - "large multimodal models"
  - "attention correction"
  - "training-free"
date: 2026-05-08
content_hash: 8b7e88ed3617f316
---

# When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations

**Conference**: NeurIPS 2025
**arXiv**: [2506.05551](https://arxiv.org/abs/2506.05551)  
**Code**: [GitHub](https://github.com/shuyansy/MLLM-Semantic-Hallucination)  
**Area**: Hallucination Detection
**Keywords**: semantic hallucination, scene text recognition, large multimodal models, attention correction, training-free

## TL;DR

This paper identifies a "semantic hallucination" problem in Large Multimodal Models (LMMs) for scene text recognition—where non-semantic text is misread as semantically plausible words. Analysis reveals that Transformer layers whose attention is more focused on text regions are less prone to hallucination. Based on this finding, the authors propose a training-free framework, ZoomText + Grounded Layer Correction, achieving approximately 4–5% improvement on TextHalu-Bench and approximately 4% on ST-VQA.

## Background & Motivation

LMMs demonstrate strong visual perception and reasoning capabilities, yet tend to produce "semantic hallucinations" when handling visually ambiguous or non-semantic scene text—generating semantically plausible but visually incorrect answers. For example, "MMOTEL" (a semantically meaningless edited string) may be recognized as "MOTEL," and "PULLa" as "PULL." **Key Challenge**: Pre-training on large-scale semantically coherent text instills strong semantic priors in these models, causing them to rely on semantic guessing rather than genuine visual grounding in OCR tasks. Existing hallucination mitigation work focuses primarily on object and factual hallucinations; OCR-specific semantic hallucinations remain largely unexplored. The **Core Idea** is to exploit attention differences across internal LLM layers—layers that attend more to text regions are less prone to hallucination—to guide the decoding process.

## Method

### Overall Architecture

A training-free semantic hallucination mitigation framework consisting of two modules:

- **Input**: Image + text query
- **Output**: Corrected scene text recognition/understanding result
- **Pipeline**: ① ZoomText localizes scene text regions (no external detector required) → ② Grounded Layer Correction selects the hidden states from the optimal layer and fuses them into the decoding process

### Key Designs

1. **Analysis of Semantic Hallucination Causes**:

    - **Hallucination Tendency Score**: For each Transformer layer $\ell$, the output probabilities of the hallucinated token $y_{hal}$ and the ground-truth token $y_{gt}$ are compared: $S_{hal}^{\ell} = P_{hal}^{\ell} / (P_{hal}^{\ell} + P_{gt}^{\ell})$
    - **Text Region Attention Score**: Defined as $A_{\ell} = \frac{\sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{T}} \alpha_{i,j}^{\ell}}{\sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{I}} \alpha_{i,j}^{\ell}}$, measuring the proportion of attention layer $\ell$ allocates to text regions
    - **Key Finding**: Spearman correlation analysis reveals a strong negative correlation between hallucination tendency and text-region attention—layers that concentrate more attention on text regions are less prone to hallucination

2. **ZoomText (Coarse-to-Fine Text Region Localization)**:

    - **Glimpse Step**: Extracts query-to-image cross-attention from the last LLM layer, averaged across all heads and query tokens to obtain a global image attention map $A_{text} = \frac{1}{HQ}\sum_{h=1}^{H}\sum_{q=1}^{Q} A_{q2v}^{(h,q)}$; the top-$K$ tokens are selected as coarse text region candidates
    - **Refocus Step**: Computes a normalized shift score between the self-attention of the first and last Transformer layers $A_{text}^{normalized} = (A_{v2v}^{(L)} - A_{v2v}^{(1)}) / (A_{v2v}^{(1)} + \epsilon)$, filtering out non-semantic tokens whose attention patterns remain stable across layers (global-context "registers") and retaining genuine text regions

3. **Grounded Layer Correction (GLC)**:

    - Selects the layer with the strongest text-region attention: $\ell^{\star} = \arg\max_{\ell} A_{\ell}$
    - Three correction strategies are proposed:
        - **Replacement**: Directly replaces the final layer's hidden states with those from layer $\ell^{\star}$
        - **Selective Replacement**: Applies replacement only to text-region tokens
        - **Fusion (default)**: Weighted fusion $\hat{H}_i = (1-w) \cdot H_i^{(L)} + w \cdot H_i^{(\ell^{\star})}$, with $w=0.1$
    - The Fusion strategy achieves the best balance between hallucination mitigation and preservation of semantic capability

### Loss & Training

The method is entirely training-free and operates as a test-time adaptive plugin. ZoomText uses $K=128$ (top-$K$ image tokens) and a fusion weight of $w=0.1$. No additional modules or trainable parameters are introduced. The framework can be directly integrated into existing LMMs such as Mini-Monkey, Qwen2.5-VL, and LLaVA-NeXT.

## Key Experimental Results

### Main Results

| Model | TextHalu-Bench | ST-VQA | TextVQA | GOT | SEED-Bench |
|------|---------------|--------|---------|-----|------------|
| GPT-4o | 45.3 | - | 71.0 | - | 70.2 |
| Mini-Monkey (baseline) | 46.5 | 66.7 | 74.1 | 88.8 | 83.3 |
| Mini-Monkey + Ours | **50.6 (+4.1)** | **70.6 (+3.9)** | **75.0 (+0.9)** | **89.2 (+0.4)** | **84.5 (+1.2)** |
| Qwen2.5-VL (baseline) | 48.3 | 67.3 | 79.1 | 85.2 | 66.7 |
| Qwen2.5-VL + Ours | **53.8 (+5.5)** | **67.6 (+0.3)** | **80.3 (+1.2)** | **86.0 (+0.8)** | **70.2 (+3.5)** |
| LLaVA-NeXT (baseline) | 27.9 | 65.1 | 65.3 | 41.9 | 50.0 |
| LLaVA-NeXT + Ours | 28.5 (+0.6) | 65.2 (+0.1) | 65.5 (+0.2) | 42.0 (+0.1) | 51.2 (+1.2) |

### Ablation Study

| Configuration | TextHalu-Bench | ST-VQA | Notes |
|------|---------------|--------|------|
| Baseline (Mini-Monkey) | 46.5 | 66.7 | -- |
| Adversarial Training | 47.5 (+1.0) | 66.8 (+0.1) | Training-based approach yields limited gains |
| Chain-of-Thought | 46.8 (+0.3) | 68.2 (+1.5) | CoT helps general tasks but does not address the root cause |
| Ours (Fusion) | **50.6 (+4.1)** | **70.6 (+3.9)** | Best configuration |
| External text detector replacing ZoomText | 50.4 (+3.9) | 70.8 (+4.1) | ZoomText approaches external detector performance |
| w/o Glimpse | 50.2 (+3.7) | 70.2 (+3.5) | Glimpse contributes meaningfully |
| w/o Refocus | 49.8 (+3.3) | 69.5 (+2.8) | Refocus noise filtering is important |
| Replacement strategy | Decline | Decline | Direct replacement disrupts semantics |
| Selective Replacement | Moderate gain | Decline on general tasks | Over-writing hurts alignment |
| Fusion (w=0.1) | **Best** | **Best** | Mild fusion is optimal |

### Key Findings

1. **Semantic hallucination is a fundamental deficiency of LMMs**: Even GPT-4o scores only 45.3 on TextHalu-Bench, far below the human score of 96.8
2. **Hallucination tendency varies significantly across Transformer layers**: Middle layers often predict ground-truth tokens more accurately than the final layers
3. **Attention concentration and hallucination exhibit a strong negative correlation**: Layers that allocate more attention to text regions exhibit lower hallucination probability
4. **Gains are larger for models with stronger OCR capability**: Mini-Monkey and Qwen2.5-VL show substantial improvements, whereas LLaVA-NeXT, which has weaker OCR ability, benefits less
5. **ZoomText achieves comparable performance without external detectors**: The glimpse-refocus strategy effectively leverages the model's own attention

## Highlights & Insights

- This work is the first to systematically define and study "semantic hallucination" in LMMs, with a clear problem formulation that is highly relevant to real-world applications
- The methodology of analyzing hallucination causes through attention mechanisms is instructive—directly quantifying the relationship between layer-wise attention and hallucination probability
- ZoomText's glimpse-refocus strategy provides an elegant text region localization solution requiring no external modules
- The fusion strategy is simple and effective (a single weight $w=0.1$), with minimal computational overhead

## Limitations & Future Work

- The fusion weight $w$ and top-$K$ value require manual tuning and may need adjustment for different models
- The method provides limited benefit for models with inherently weak OCR capability (e.g., LLaVA-NeXT), indicating a dependency on underlying visual encoding ability
- ZoomText assumes text appears on semantically meaningful backgrounds (e.g., signs, posters) and may not generalize well to document-style images containing only text
- TextHalu-Bench comprises only 1,740 samples, offering limited scene coverage

## Related Work & Insights

- Complementary to hallucination mitigation methods such as VCD (Visual Contrastive Decoding)—VCD targets object hallucinations, whereas this work targets textual hallucinations
- Layer-wise attention analysis can generalize to other tasks requiring precise visual grounding (e.g., fine-grained recognition)
- The hidden state fusion mechanism in Grounded Layer Correction can be applied to other scenarios involving model-self-guided decoding
- The evaluation paradigm of TextHalu-Bench (contrasting semantic vs. non-semantic text) offers useful insights for assessing OCR model robustness

## Rating

- Novelty: ⭐⭐⭐⭐ The definition of semantic hallucination and the analytical approach are novel, though the solution (hidden state fusion) is relatively conventional
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple benchmarks with extensive ablations and comparisons against various hallucination mitigation methods
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, visualizations are rich, and the argumentation is logically coherent
- Value: ⭐⭐⭐⭐ Reveals an important deficiency of LMMs in OCR tasks; the training-free solution offers strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](../../ICML2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](../../ICCV2025/hallucination/only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)

</div>

<!-- RELATED:END -->
