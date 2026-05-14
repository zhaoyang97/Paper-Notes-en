---
title: >-
  [Paper Note] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models
description: >-
  [ICCV 2025][LLM/NLP][Video anomaly detection] This paper proposes VA-GPT, a multimodal large language model for video anomaly event understanding. Through two modules—Spatial Effective Token Selection (SETS) and Temporal…
tags:
  - "ICCV 2025"
  - "LLM/NLP"
  - "Video anomaly detection"
  - "multimodal large language models"
  - "spatial effective tokens"
  - "temporal effective tokens"
  - "cross-domain generalization"
date: 2026-05-08
content_hash: 077b9adddd63361e
---

# VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models

**Conference**: ICCV 2025
**arXiv**: [2508.06350](https://arxiv.org/abs/2508.06350)
**Code**: N/A
**Area**: LLM/NLP
**Keywords**: Video anomaly detection, multimodal large language models, spatial effective tokens, temporal effective tokens, cross-domain generalization

## TL;DR
This paper proposes VA-GPT, a multimodal large language model for video anomaly event understanding. Through two modules—Spatial Effective Token Selection (SETS) and Temporal Effective Token Generation (TETG)—VA-GPT enables MLLMs to precisely align anomaly-relevant information in both spatial and temporal dimensions, achieving state-of-the-art performance on both in-domain and cross-domain anomaly detection benchmarks.

## Background & Motivation

**Background**: Traditional video anomaly detection methods are inherently closed-set detection and classification problems, struggling to handle unseen anomaly types with limited vocabulary. Recent MLLMs, while possessing strong comprehension capabilities, lack sufficient precision in processing anomalous events.

**Limitations of Prior Work**: Anomalous events are sparse in both space and time—only small regions within a few frames contain anomaly-relevant information. Existing video MLLMs treat all visual tokens equally, and the resulting redundant tokens interfere with anomaly localization and description.

**Core Idea**: The paper leverages inter-frame differences to select spatial effective tokens (since anomalies tend to cause localized abrupt changes) and uses the confidence scores of a pretrained classifier to generate temporal effective tokens (encoding prior knowledge of anomalous time intervals), thereby precisely aligning anomaly information along both dimensions.

## Method

### Key Designs

1. **Spatial Effective Token Selection (SETS)**:
   - Extracts patch embeddings of adjacent frames using DINOv2 and computes Manhattan distance as an inter-frame difference map.
   - Selects the top-K proportion of patches with the largest differences as spatial effective tokens.
   - **Design Motivation**: Anomalous events typically induce significant visual changes in localized regions.

2. **Temporal Effective Token Generation (TETG)**:
   - Assigns anomaly probability scores to each frame using a lightweight pretrained anomaly classifier.
   - Encodes the scores as additional temporal tokens injected directly into the LLM in the language space.
   - **Design Motivation**: Provides the LLM with prior knowledge of temporal anomaly locations to enhance temporal reasoning.

3. **Cross-Domain Evaluation Benchmark**: Constructs a new cross-domain evaluation protocol based on XD-Violence, incorporating temporal-localization-oriented QA to assess model transferability across domains.

### Loss & Training
Standard instruction-following training, with fine-tuning performed on a self-constructed anomaly video instruction dataset.

## Key Experimental Results

| Method | LLM | In-Domain Total Acc | In-Domain Temporal Acc | Cross-Domain Total Acc |
|--------|-----|--------------------|-----------------------|-----------------------|
| VA-GPT | Vicuna-7B | **30.69%** | **Highest** | **Highest** |
| Hawkeye | LLaVA-7B | 28.60% | 30.00% | 25.30% |
| Video-ChatGPT | Vicuna-7B | 24.13% | 28.51% | 24.00% |

### Key Findings
- SETS directs the model's attention toward anomalous regions rather than background, significantly improving spatial alignment quality.
- After TETG provides temporal priors, anomaly temporal localization accuracy improves markedly.
- Cross-domain performance demonstrates the generalizability of the approach, confirming that the model learns transferable anomaly patterns rather than merely memorizing training set distributions.

### Ablation Study: SETS Spatial Effective Token Selection

| Top-K Ratio | Total Acc | Temporal Acc | Computation |
|-------------|-----------|-------------|-------------|
| 100% (all) | 28.6 | 28.5 | 1.0x |
| 50% | 29.8 | 30.1 | 0.65x |
| **25%** | **30.7** | **31.2** | **0.45x** |
| 10% | 29.5 | 29.8 | 0.30x |

## Highlights & Insights
- The intuition behind deriving spatial effective tokens from inter-frame differences is clear: anomaly = change → regions of change = important regions.
- Directly injecting temporal prior tokens in the language space is an efficient design choice that avoids the need for additional visual-temporal encoders.

## Limitations & Future Work
- The approach relies on the quality of the pretrained anomaly classifier; classifier bias propagates directly into the temporal effective tokens.
- SETS is based on simple inter-frame differences and may miss static anomalies (e.g., planted explosives).
- Validation is limited to surveillance video scenarios; other domains (e.g., traffic, medical) remain unexplored.
- Absolute performance remains limited, with Total Accuracy around 30%, leaving considerable room for improvement.
- TETG encodes anomaly probabilities directly as tokens; the choice of encoding scheme may influence effectiveness.
- Integration with more recent video large language models (e.g., Qwen2-VL) has not been explored.
- The cross-domain evaluation benchmark is based solely on XD-Violence; generalization to additional domains has not been validated.
- The model may struggle to accurately describe scenarios where multiple anomaly types occur simultaneously.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual selection mechanism of spatial and temporal effective tokens is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers in-domain, cross-domain, and ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and methodology is intuitive.
- **Value**: ⭐⭐⭐⭐ Broad practical application prospects for video anomaly understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[ICCV 2025\] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](any_ssr_how_recursive_least_squares_works_in_continual_learning_of_large_language_models.md)
- [\[NeurIPS 2025\] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](../../NeurIPS2025/llm_nlp/streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](../../NeurIPS2025/llm_nlp/solving_inequality_proofs_with_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/llm_nlp/large_language_models_miss_the_multi-agent_mark.md)

</div>

<!-- RELATED:END -->
