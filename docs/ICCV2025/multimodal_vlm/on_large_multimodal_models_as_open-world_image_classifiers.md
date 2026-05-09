---
title: >-
  [Paper Note] On Large Multimodal Models as Open-World Image Classifiers
description: >-
  [ICCV 2025][Multimodal VLM][large multimodal models] This paper systematically evaluates 13 large multimodal models (LMMs) on open-world image classification, proposes an evaluation protocol comprising four complementary metrics, and reveals systematic error patterns in LMMs regarding granularity judgment and fine-grained discrimination.
tags:
  - ICCV 2025
  - Multimodal VLM
  - large multimodal models
  - open-world classification
  - evaluation protocol
  - LMM
  - image classification
date: 2026-05-08
content_hash: e0e9219bf20f4f2d
---

# On Large Multimodal Models as Open-World Image Classifiers

**Conference**: ICCV 2025
**arXiv**: [2503.21851](https://arxiv.org/abs/2503.21851)
**Code**: [GitHub](https://github.com/altndrr/lmms-owc)
**Area**: Multimodal VLM
**Keywords**: large multimodal models, open-world classification, evaluation protocol, LMM, image classification

## TL;DR

This paper systematically evaluates 13 large multimodal models (LMMs) on open-world image classification, proposes an evaluation protocol comprising four complementary metrics, and reveals systematic error patterns in LMMs regarding granularity judgment and fine-grained discrimination.

## Background & Motivation

Traditional image classification requires a predefined set of categories (closed-world), whereas LMMs natively support open-ended output—directly generating category names in response to queries such as "What is the object in this image?" without requiring a fixed candidate list. While this capability is highly practical, existing research exhibits notable shortcomings:

**Problem 1**: Most LMM classification evaluations remain confined to the closed-world setting (providing candidate category lists to the model), failing to reflect the genuine open-world capability of LMMs. Although Zhang et al. (2024) attempted open-world evaluation, it was limited to four datasets and a single metric (text inclusion).

**Problem 2**: Existing evaluation metrics are overly simplistic. The text inclusion metric merely checks whether the correct label string appears within the prediction, and cannot handle the following scenarios:
- Semantically equivalent but textually different predictions (e.g., *sofa* vs. *couch*)
- Different but reasonable levels of granularity (e.g., *dog* vs. *pug*)
- Literally correct but semantically erroneous matches (e.g., *can* spuriously matching *trash can*)

**Paper Goals**: To provide the first large-scale, multi-dimensional benchmark for evaluating LMMs on open-world classification, analyze error patterns through multiple metrics, and offer directional guidance for future research.

## Method

### Overall Architecture

Rather than introducing a new model, this paper constructs a comprehensive evaluation framework for open-world classification:
1. Formally defines the open-world classification task
2. Proposes four complementary evaluation metrics
3. Conducts large-scale experiments across 10 datasets and 13 models
4. Analyzes error types and mitigation strategies through metric combinations

### Key Designs

1. **Task Formalization (Open-World Classification)**:

    - **Function**: Defines the LMM as a function $f_{\text{LMM}}: \mathcal{X} \times \mathcal{T} \rightarrow \mathcal{T}$, taking an image and a query text as input and producing a textual prediction as output.
    - **Mechanism**: In the closed-world setting, the query includes a candidate set $\mathcal{C}$; in the open-world setting, the output space is unconstrained, and the model freely predicts from all possible semantic concepts $\mathcal{Y}$, where $|\mathcal{C}| \ll |\mathcal{Y}|$.
    - **Design Motivation**: The generative capacity of LMMs should not be constrained by predefined category lists; the open-world setting better reflects real-world application scenarios.

2. **Four Evaluation Metrics**:

    - **Function**: Measure the alignment between predictions and ground truth from different perspectives.
    - **Mechanism**:
        - **Text Inclusion (TI)**: Checks whether the ground-truth label is a substring of the predicted text, $\text{TI}(y, \hat{y}) = \mathbf{1}[y \subseteq \hat{y}]$. Simple but overly strict; semantically equivalent but textually different predictions are penalized.
        - **Llama Inclusion (LI)**: Uses Llama 3.2 3B as a judge to determine whether the prediction is semantically consistent with the ground truth. A classification-specific instantiation of the LLM-as-judge paradigm.
        - **Semantic Similarity (SS)**: $\text{SS} = \langle g_{\text{emb}}(\hat{y}), g_{\text{emb}}(y) \rangle$, computing the cosine similarity between Sentence-BERT embeddings of the prediction and ground truth, yielding a continuous score in $[0, 1]$.
        - **Concept Similarity (CS)**: $\text{CS} = \max_{p \in \text{split}(\hat{y})} \langle g_{\text{emb}}(p), g_{\text{emb}}(y) \rangle$, segmenting the prediction with spaCy and taking the maximum cosine similarity between any segment and the ground truth, addressing the dilution problem caused by evaluating verbose predictions holistically.
    - **Design Motivation**: No single metric provides a complete assessment; inconsistencies across metrics can expose specific error patterns.

3. **Error Analysis Framework**:

    - **Function**: Uses metric discrepancies to localize the type of model error.
    - **Mechanism**:
        - High CS but low LI → granularity error (correct but too generic, e.g., predicting "animal" instead of "Labrador")
        - High LI but low CS → annotation ambiguity or multi-label issues
        - Both low → fine-grained discrimination failure (wrong but specific, e.g., confusing two similar flower species)
    - **Design Motivation**: Rather than simply reporting accuracy, the combined metrics provide actionable directions for improvement.

### Loss & Training

This paper is an evaluation study and does not involve model training. All experiments adopt the standard prompt "What type of object is in this image?" for zero-shot inference in a unified manner.

## Key Experimental Results

### Main Results

**LMM open-world classification performance** (grouped mean by dataset granularity):

| Model | Prototypical TI | Prototypical LI | Fine-grained TI | Very Fine TI | Notes |
|-------|----------------|-----------------|-----------------|--------------|-------|
| Qwen2VL 7B | 46.4 | 78.7 | 34.6 | 0.8 | Best LMM |
| InternVL2 8B | 40.6 | 74.4 | 22.3 | 2.3 | Second best |
| LLaVA-1.5 7B | 34.6 | 63.1 | 8.4 | 0.0 | Weaker performance |
| CaSED (OW baseline) | 24.5 | 46.3 | 27.4 | 0.7 | Contrastive OW method |
| CLIP (closed-world) | 76.4 | — | 85.0 | — | Closed-world upper bound |
| SigLIP (closed-world) | 81.8 | — | 92.6 | — | Closed-world upper bound |

### Ablation Study

**Effect of prompt design on granularity**:

| Prompt Strategy | CS Gain | Notes |
|----------------|---------|-------|
| Default prompt | — | "What type of object?" |
| Fine-grained prompt | +5–10% CS | Adds guidance such as "be specific" |
| CoT reasoning | +2–5% | Adds Chain-of-Thought steps |

**Effect of inference strategy on fine-grained discrimination**:

| Strategy | Fine-grained Improvement | Notes |
|----------|--------------------------|-------|
| Direct prediction | baseline | Default mode |
| Structured reasoning | +3–8% LI | Reduces confusion between similar categories |

### Key Findings

1. **LMMs outperform contrastive baselines in the open world**: Generative LMMs perform better than category-list-free methods such as CaSED and CLIP retrieval, though they remain significantly behind closed-world models that receive candidate lists.
2. **Performance degrades sharply with finer granularity**: Accuracy drops precipitously from prototypical (~46% TI) to very fine-grained (~1% TI).
3. **Granularity errors are the dominant failure mode**: Models tend to produce overly generic predictions (e.g., "bird" instead of "scarlet tanager"); prompt-based guidance can partially mitigate this.
4. **Model scale helps but is not decisive**: InternVL2 2B→8B shows consistent gains, whereas LLaVA-OV 0.5B→7B regresses on some metrics.
5. **Annotation ambiguity is pervasive**: Verification with a tagging model reveals that many "incorrect" predictions are in fact valid, as images contain multiple plausible labels.

## Highlights & Insights

- The contribution lies in **evaluation rather than method** design, yet is equally important—this is the first work to elevate LMM open-world classification from "running a quick benchmark" to rigorous evaluation science.
- The **complementary design** of the four metrics is elegant: TI captures strict matching, LI captures semantic correctness, SS/CS provide continuous scores, and inter-metric discrepancies diagnose specific failure modes.
- The granularity–performance cliff reveals a fundamental bottleneck in LMM visual perception—not an inability to recognize objects, but an inability to articulate precise names.
- The analysis of annotation ambiguity is honest and valuable, cautioning the community against over-interpreting "errors" in open-world evaluation.

## Limitations & Future Work

- The most recent LMMs (e.g., GPT-4o, Gemini 2) are not included; whether conclusions hold for stronger models requires further verification.
- The evaluation prompt is uniformly set to "What type of object," which is unnatural for non-object datasets (e.g., DTD textures, UCF101 actions).
- The Llama inclusion metric relies on the judgment capability of Llama 3.2 3B, which may itself introduce bias.
- Few-shot or retrieval-augmented strategies as potential systematic improvements to fine-grained recognition remain unexplored.

## Related Work & Insights

- Comparison with CaSED (CVPR 2023) highlights the complementary strengths and weaknesses of generative and contrastive models for open-world recognition.
- The connection between granularity errors and prompt engineering suggests the practical value of task-specific prompt template libraries.
- The evaluation framework can be directly applied to open-world evaluation of other vision-language understanding tasks.
- Zhang et al. (2024) is the most direct predecessor; the present work surpasses it comprehensively in both scale and depth.

## Rating

- **Novelty**: ⭐⭐⭐ — An evaluation study rather than a methodological contribution, though the task formalization and metric design exhibit originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 13 models × 10 datasets × 4 metrics; coverage is exceptionally broad.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, progressively layered analysis, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Provides the LMM community with much-needed evaluation infrastructure and insightful error analysis.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/openhoi_open-world_hand-object_interaction_synthesis_with_multimodal_large_langu.md)
- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)

<!-- RELATED:END -->
