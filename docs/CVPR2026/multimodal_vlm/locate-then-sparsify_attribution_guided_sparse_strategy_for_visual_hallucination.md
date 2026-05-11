---
title: >-
  [Paper Note] Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation
description: >-
  [CVPR 2026][Multimodal VLM][Visual Hallucination] This paper proposes LTS-FS (Locate-Then-Sparsify for Feature Steering), a framework that employs causal intervention-based attribution to identify hallucination-relevant…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Visual Hallucination"
  - "Feature Steering"
  - "Layer-wise Attribution"
  - "Sparse Adjustment"
  - "LVLM"
date: 2026-05-08
content_hash: 70a4e8cfb09a8788
---

# Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation

**Conference**: CVPR 2026
**arXiv**: [2603.16284](https://arxiv.org/abs/2603.16284)
**Code**: [https://github.com/huttersadan/LTS-FS](https://github.com/huttersadan/LTS-FS)
**Area**: Multimodal VLM
**Keywords**: Visual Hallucination, Feature Steering, Layer-wise Attribution, Sparse Adjustment, LVLM

## TL;DR
This paper proposes LTS-FS (Locate-Then-Sparsify for Feature Steering), a framework that employs causal intervention-based attribution to identify hallucination-relevant layers and applies layer-wise sparse control over feature steering intensity according to attribution scores, effectively mitigating hallucinations in LVLMs while preserving generalization capability.

## Background & Motivation

**Background**: Despite strong performance on multimodal tasks, large vision-language models (LVLMs) suffer from severe hallucination problems—generating fluent yet visually inconsistent descriptions. Existing mitigation approaches fall into three categories: fine-tuning methods (high cost, impairs generalization), decoding enhancement methods (high inference overhead), and feature steering methods (modifying intermediate layer features).

**Limitations of Prior Work**: Feature steering methods such as Nullu and VTI apply uniform steering intensity across all layers, ignoring inter-layer differences—some layers are highly correlated with hallucinations while others are responsible for general representations. Uniform steering perturbs hallucination-irrelevant layers, disrupts the original feature distribution, and degrades generalization.

**Key Challenge**: There is a fundamental trade-off between hallucination mitigation and generalization—overly strong steering reduces hallucinations but impairs general capability, while overly weak steering yields insufficient effect.

**Goal**: To precisely locate hallucination-relevant layers and apply differentiated steering only where necessary.

**Key Insight**: Drawing on parameter localization techniques, the contribution of each layer to hallucinated outputs is quantified via causal intervention, yielding layer-wise attribution scores.

**Core Idea**: First locate hallucination-relevant layers, then sparsify steering intensity—heavily adjusting high-score layers while leaving low-score layers untouched.

## Method

### Overall Architecture
A three-stage pipeline: (1) construct a dual-granularity hallucination dataset → (2) perform layer-wise attribution via causal intervention → (3) apply sparse, layer-differentiated control of feature steering intensity according to attribution scores. The framework is decoupled from specific steering methods and can be seamlessly integrated with existing approaches such as Nullu and VTI.

### Key Designs

1. **Dual-Granularity Hallucination Dataset Construction**:

    - **Function**: Construct hallucination samples at both the token level and sentence level.
    - **Mechanism**: Token-level samples are derived from yes/no QA benchmarks such as POPE and Antidote, where hallucinated tokens can be identified by rule-based matching; sentence-level samples are sourced from CHAIR benchmark multi-sentence descriptions, where sentences containing hallucinated tokens are identified by splitting and detection.
    - **Design Motivation**: Hallucinations in short responses manifest at individual tokens, whereas in long responses they propagate across entire sentences, necessitating granularity-specific treatment.

2. **Layer-wise Attribution via Causal Intervention**:

    - **Function**: Quantify each layer's contribution to hallucinated outputs.
    - **Mechanism**: At layer $l$, attention outputs are masked head-by-head, and the resulting change in hallucinated token logits is observed. The token-level attribution score is $s_{tok}^l = \sum_{h=1}^H \log \frac{P(y|\mathbf{h}_{l-1}, \mathbf{a}_l)}{P(y|\mathbf{h}_{l-1}, \mathbf{a}_l \odot M^h)}$. Sentence-level attribution aggregates token-level scores using three weighted indicators (cue indicator, position indicator, hallucination indicator), assigning higher weights to later tokens and tokens containing hallucinations.
    - **Design Motivation**: Direct causal intervention more accurately measures a layer's causal contribution than gradient-based analysis.

3. **Layer-wise Sparse Feature Steering**:

    - **Function**: Convert attribution scores into layer-specific steering intensities.
    - **Mechanism**: Hard sparsification combined with soft weighting. Layers with very low attribution scores are filtered out via threshold $\tau = r_s \cdot \frac{1}{L}\sum s^l$ and receive no steering; high-score layers have their steering intensity scaled by normalized scores $\tilde{s}^l$ as $\lambda_l = \lambda \cdot m_l + \lambda \cdot \tilde{s}_l$.
    - **Design Motivation**: Steering low-score layers yields little hallucination reduction but incurs significant generalization cost and should be excluded; differentiated steering on high-score layers enables precise control.

### Loss & Training
- Only 100 sentence-level and 100 token-level hallucination samples are required for attribution computation.
- Once attribution is completed, the strategy is fixed and requires no test-set-specific modification.
- No additional inference overhead is introduced; inference speed is identical to the original model.

## Key Experimental Results

### Main Results (CHAIR Metrics, Lower is Better)

| Model | Method | CS↓ | CI↓ | Recall | Len |
|------|------|-----|-----|--------|-----|
| LLaVA-1.5-7B | Regular | 53.0 | 13.9 | 77.2 | 98.0 |
| LLaVA-1.5-7B | Nullu | 50.2 | 13.7 | 76.9 | 93.3 |
| LLaVA-1.5-7B | LTS-FS(Nullu) | 46.8 | 13.5 | 76.6 | 93.2 |
| LLaVA-1.5-7B | VTI | 47.4 | 13.9 | 76.2 | 88.9 |
| LLaVA-1.5-7B | LTS-FS(VTI) | **35.8** | **11.9** | 75.4 | 82.2 |
| Qwen-VL2.5-7B | LTS-FS(Nullu) | **23.8** | **6.0** | 60.8 | 120.6 |

### Generalization Capability (POPE Accuracy / MMMU, etc.)

| Metric | Nullu | LTS-FS(Nullu) | Note |
|------|-------|---------------|------|
| POPE-popular Acc | Baseline | +2% | Qwen-VL-2.5-7B |
| LLaVA-Bench detailness | 4.72 | 4.92 | Better generalization |
| MMMU | Degraded | Maintained/Improved | General capability preserved |

## Highlights & Insights
- This work is the first to introduce layer-wise sparse steering into hallucination mitigation, and its decoupling from specific steering methods confers broad applicability.
- The causal intervention attribution method is concise and effective, requiring only 200 calibration samples to complete layer-wise attribution.
- The framework is plug-and-play and can directly enhance existing methods such as Nullu and VTI.
- The approach significantly preserves or even improves generalization while mitigating hallucinations (LLaVA-Bench detailness: 4.72 → 4.92).
- LTS-FS(VTI) reduces CHAIR-S from 47.4 to 35.8 on LLaVA-1.5-7B, a 24.5% relative reduction.

## Limitations & Future Work
- Attribution computation requires head-by-head intervention at each layer, incurring considerable attribution overhead and GPU memory usage on larger models (>13B).
- Construction of the dual-granularity dataset relies on existing hallucination benchmarks (POPE, CHAIR, Antidote); applicability to out-of-domain scenarios requires further validation.
- Experiments are currently conducted only on LLaVA and Qwen-VL series; adaptation to additional architectures (e.g., InternVL, Gemma) warrants exploration.
- Refining attribution granularity to the attention-head or neuron level could enable more fine-grained steering control.
- Attribution scores may be inconsistent across tasks (QA vs. captioning); the current strategy of separately using token/sentence-level scores is relatively coarse.
- The optimal choice of threshold parameter $r_s$ may vary across models and tasks.
- The combined effect with decoding enhancement methods (e.g., VCD) merits further investigation.
- Computational efficiency of sentence-level attribution in long-form generation scenarios requires further optimization.

### Additional Model Results
- Effective on LLaVA-1.5-13B as well: CS decreases from 40.8 to 35.7 (LTS-FS+Nullu) and 32.0 (LTS-FS+VTI).
- On Qwen-VL2.5-7B, CHAIR-CI drops from 7.4 to 6.0, reducing hallucinated detailed descriptions by approximately 19%.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](../../ACL2026/multimodal_vlm/spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)
- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](../../ICLR2026/multimodal_vlm/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[CVPR 2026\] EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](ebmc_multimodal_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
