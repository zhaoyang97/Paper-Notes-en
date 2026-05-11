---
title: >-
  [Paper Note] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric
description: >-
  [CVPR 2026][Image Generation][Open-vocabulary HOI detection] This paper proposes SHOE, an evaluation framework that decomposes HOI predictions into verb and object components and computes LLM-driven semantic similarity scores for each independently, replacing the exact-match paradigm of conventional mAP. SHOE achieves 85.73% agreement with human judgments on open-vocabulary HOI detection evaluation, surpassing the average inter-annotator agreement of 78.61%.
tags:
  - CVPR 2026
  - Image Generation
  - Open-vocabulary HOI detection
  - semantic similarity evaluation
  - LLM scoring
  - WordNet
  - evaluation metric
date: 2026-05-08
content_hash: e8fd0a305f2993b5
---

# SHOE: Semantic HOI Open-Vocabulary Evaluation Metric

**Conference**: CVPR 2026
**arXiv**: [2604.01586](https://arxiv.org/abs/2604.01586)
**Code**: [https://github.com/majnoa/SHOE](https://github.com/majnoa/SHOE)
**Area**: Image Generation
**Keywords**: Open-vocabulary HOI detection, semantic similarity evaluation, LLM scoring, WordNet, evaluation metric

## TL;DR

This paper proposes SHOE, an evaluation framework that decomposes HOI predictions into verb and object components and computes LLM-driven semantic similarity scores for each independently, replacing the exact-match paradigm of conventional mAP. SHOE achieves 85.73% agreement with human judgments on open-vocabulary HOI detection evaluation, surpassing the average inter-annotator agreement of 78.61%.

## Background & Motivation

1. **Background**: Human-Object Interaction (HOI) detection is a fundamental task in visual understanding. The standard evaluation metric is mAP, which relies on exact categorical matching between predictions and ground-truth labels.
2. **Limitations of Prior Work**: mAP treats HOI categories as discrete labels, causing semantically similar but lexically different predictions (e.g., "lean on couch" vs. "sit on couch") to be counted as errors. Furthermore, dataset annotations are incomplete, so reasonable but unannotated predictions are penalized as false positives.
3. **Key Challenge**: As VLMs and MLLMs advance, models can generate open-vocabulary predictions beyond fixed label sets, yet existing evaluation protocols cannot fairly assess the quality of such flexible outputs.
4. **Goal**: Design a semantically aware, flexible evaluation framework that supports graded matching for open-vocabulary HOI predictions.
5. **Key Insight**: Decompose HOI into two independent components—verb and object—and compute semantic similarity for each using the averaged scores from multiple LLMs, thereby avoiding the combinatorial explosion of full HOI-pair comparisons.
6. **Core Idea**: Achieve decomposed, flexible HOI matching evaluation via WordNet sense disambiguation combined with multi-LLM semantic scoring.

## Method

### Overall Architecture

Given predicted HOI triplets $(b_h, b_o, v, o)$ and ground-truth HOIs, the framework first performs bounding box matching, then maps verbs and objects to WordNet synsets, queries a precomputed LLM similarity table, synthesizes instance-level similarity scores, and finally aggregates them into Soft-mAP or mF1 scores.

### Key Designs

1. **WordNet Synset Mapping and Disambiguation**:

    - **Function**: Map HOI verb and object labels to semantically unambiguous WordNet synsets.
    - **Mechanism**: Each verb/object is assigned to a sense-specific synset to resolve polysemy. For objects, neighborhood expansion within the WordNet hierarchy (hypernyms and hyponyms) is applied; for verbs, because WordNet's verb taxonomy is shallow and fragmented, approximately 7,150 HOI-relevant verb synsets are manually curated for matching.
    - **Design Motivation**: Direct lexical comparison is susceptible to polysemy; using synsets ensures that semantic comparisons reflect genuine meaning.

2. **Multi-LLM Semantic Similarity Scoring**:

    - **Function**: Compute a 0–4 semantic similarity score for each verb–verb and object–object pair.
    - **Mechanism**: Qwen3-32B first performs full-scale pre-filtering (approximately 850K verb-pair comparisons) to eliminate zero-similarity pairs. The remaining non-zero pairs are then scored by four additional LLMs—DeepSeek-V3, Llama-4-Maverick-17B, Yi-1.5-34B-Chat, and Gemini-2.5-Pro—and scores are averaged. Each LLM rates pairs on a 5-point scale based on synset gloss definitions.
    - **Design Motivation**: Single-LLM scoring introduces bias; averaging across multiple models improves robustness. The inter-model Pearson correlation for verb similarity is relatively low (0.50–0.72), while for objects it is higher (up to $r=0.84$), confirming that verb semantics are inherently more complex.

3. **Decomposed and Scalable Evaluation Design**:

    - **Function**: Decompose HOI similarity as $\text{sim}(p,g) = f(\text{sim}_v(v^p, v^g), \text{sim}_o(o^p, o^g))$.
    - **Mechanism**: Verb and object similarities are aggregated via arithmetic mean with $w=0.5$. This decomposition reduces the number of required similarity computations from $(V \times O)^2$ (brute force) to $V^2 + O^2$, enabling extension from HICO-DET's 600 HOI classes to 38 million semantically relevant HOIs.
    - **Design Motivation**: Brute-force computation of all HOI-pair similarities grows quadratically with vocabulary size; the decomposition strategy makes large-scale open-vocabulary evaluation computationally feasible.

### Loss & Training

SHOE is an evaluation metric framework and involves no model training. It provides two aggregation modes:
- **Confidence-available mode**: Compatible with mAP-style ranked evaluation; computes Soft-AP and Soft-mAP.
- **Confidence-free mode**: Treats all predictions equally and computes soft precision/recall/F1 directly, suitable for VLMs that do not natively produce confidence scores.

## Key Experimental Results

### Main Results

| Method | Type | mAP | SHOE mAP |
|------|------|-----|----------|
| HOLA (ViT-L) | Default | 39.05 | 39.92 |
| LAIN (ViT-B) | Zero-shot | 34.60 | 35.37 |
| THID | Open-Vocab | 22.01 | 22.04 |
| GPT-4.1 + DETR | VLM | 49.50 | 61.67 |
| InternVL3-38B + DETR | VLM | 42.00 | 58.03 |
| Qwen2.5-VL-32B + DETR | VLM | 34.83 | **66.03** |

### Ablation Study

| Metric | Agreement with Human Judgments (%) |
|----------|---------------------|
| SHOE (Standard, arithmetic mean) | **85.73** |
| SHOE (geometric mean) | 84.29 |
| SHOE (minimum) | 84.01 |
| DeepSeek-V3 (direct LLM scoring) | 83.34 |
| Gemini-2.5-Pro | 77.52 |
| CLIP-ViT-B (gloss) | 59.11 |
| WordNet WUP | 57.09 |
| SentenceBERT | 54.09 |
| mAP direct-match | 38.90 |

### Key Findings

- Qwen2.5-VL-32B achieves the lowest standard mAP (34.83) yet the highest SHOE mAP (66.03), indicating that the model possesses strong semantic understanding but does not faithfully reproduce HICO-DET's exact label vocabulary.
- VLM-based methods significantly outperform conventional methods under SHOE mAP, revealing genuine capability differences that mAP fails to capture.
- Hyperparameter analysis shows that for the "same verb, different object" scenario the optimal weight is $w^*=0.267$ (favoring object similarity), while for "different verb, same object" it is $w^*=0.733$ (favoring verb similarity); however, $w=0.5$ is retained due to the limited scale of the user study.
- Among verb pairs filtered out as zero-similarity by Qwen3-32B, the disagreement rates from other LLMs range from only 0.245% to 1.318%, validating the reliability of the filtering strategy.

## Highlights & Insights

- **Elegant decomposition**: Decomposing HOI similarity into independent verb and object comparisons reduces computational complexity from $(V \times O)^2$ to $V^2 + O^2$, enabling the extension of HICO-DET's 600 classes to 38 million classes. This idea generalizes naturally to any evaluation scenario requiring combinatorial semantic comparison.
- **Surpassing human agreement**: SHOE achieves 85.73% agreement with averaged human judgments, whereas inter-annotator agreement averages only 78.61%. This demonstrates that averaging across multiple LLMs yields more stable semantic judgments than individual human annotators.
- **Evaluation metric as infrastructure**: The similarity lookup table is constructed only once and subsequently queried at negligible cost, greatly reducing the overhead of repeated evaluation.

## Limitations & Future Work

- Validation is currently limited to HICO-DET; other HOI datasets (e.g., SWIG-HOI) also suffer from incomplete annotations and require further evaluation.
- The user study is relatively small in scale (500 pairs, 5 annotators); stability under larger-scale human evaluation requires further verification.
- The confidence proxy for VLMs (token probability) may be unreliable; obtaining well-calibrated confidence scores for open-ended generative models remains an open problem.
- The "gold standard" for semantic similarity is inherently subjective and may vary across individuals; HOI evaluation in specialized domains (e.g., medical or legal contexts) may require domain-specific customization.

## Related Work & Insights

- **vs. mAP (standard evaluation)**: mAP enforces strict exact matching, whereas SHOE introduces graded semantic matching. The two are complementary—mAP measures the ability to reproduce exact labels, while SHOE measures semantic comprehension.
- **vs. CLIP-based similarity**: CLIP achieves only 59.11% agreement on HOI-pair comparisons, indicating that general-purpose vision–language embeddings are insufficient to capture the nuances of HOI semantics.
- **vs. direct LLM scoring**: Scoring full HOI pairs directly with a single LLM reaches at most 83.34% agreement, whereas SHOE's decomposition strategy achieves 85.73% with greater scalability.

## Rating

- Novelty: ⭐⭐⭐⭐ The decomposed semantic evaluation approach is novel, though the core mechanism still relies on LLM scoring and averaging.
- Experimental Thoroughness: ⭐⭐⭐⭐ User studies, multi-baseline comparisons, and Qwen filtering validation are all reasonably comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, figures and tables are professional, and mathematical formulations are complete.
- Value: ⭐⭐⭐⭐ Provides a practical tool for open-vocabulary HOI evaluation, though its impact is primarily confined to the HOI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery](opendpr_open-vocabulary_change_detection_via_vision-centric_diffusion-guided_pro.md)
- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[CVPR 2026\] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)

</div>

<!-- RELATED:END -->
