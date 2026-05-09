---
title: >-
  [Paper Note] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Interpretability] This paper proposes Tree-of-Evidence (ToE), an inference-time discrete beam search algorithm that formalizes multimodal model interpretability as a discrete optimization problem over coarse-grained evidence units (vital sign time windows, radiology report segments). With only 5 evidence units, ToE retains over 98% of the full-input model's AUROC while generating auditable evidence trace paths.
tags:
  - ACL 2026
  - Multimodal VLM
  - Multimodal Interpretability
  - Evidence Search
  - Clinical Prediction
  - Beam Search
  - Concept Bottleneck
date: 2026-05-08
content_hash: 1d617b25d9c1431f
---

# Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding

**Conference**: ACL 2026
**arXiv**: [2604.07692](https://arxiv.org/abs/2604.07692)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Multimodal Interpretability, Evidence Search, Clinical Prediction, Beam Search, Concept Bottleneck

## TL;DR

This paper proposes Tree-of-Evidence (ToE), an inference-time discrete beam search algorithm that formalizes multimodal model interpretability as a discrete optimization problem over coarse-grained evidence units (vital sign time windows, radiology report segments). With only 5 evidence units, ToE retains over 98% of the full-input model's AUROC while generating auditable evidence trace paths.

## Background & Motivation

**Background**: Large multimodal models (LMMs) have achieved state-of-the-art performance in high-stakes domains such as healthcare, but their reasoning processes remain opaque. Existing interpretability methods include attention visualization, gradient saliency, post-hoc attribution methods (LIME/SHAP), and concept bottleneck models (CBMs).

**Limitations of Prior Work**: (1) Attention weights are often unfaithful to the model's actual decision logic; (2) LIME/SHAP provide approximations without guarantees and cannot produce discrete evidence selection; (3) CBMs require predefined concept annotations and are static at inference time, lacking adaptive search capability; (4) existing rationale extraction methods are typically limited to a single modality (primarily text) and fail to capture cross-modal synergistic dependencies.

**Key Challenge**: Clinical deployment requires that model predictions be explicitly traceable to specific, verifiable evidence, yet existing methods are either unfaithful, do not support multimodal inputs, or cannot provide audit trails.

**Goal**: Design an inference-time search algorithm capable of identifying a compact multimodal evidence set that both reproduces full-input predictions and provides an auditable search process.

**Key Insight**: Drawing on the deliberate branching search concept from Tree-of-Thoughts, this work reframes interpretability as a discrete search problem—a "System 2"-style multi-step deliberate search rather than a "System 1"-style single-pass greedy ranking.

**Core Idea**: Structure the multimodal input space into "global context" (fixed priors, e.g., CXR/ECG baselines) and "searchable evidence" (dynamically varying vital signs and clinical notes). A lightweight Evidence Bottleneck scorer is trained, and beam search is performed at inference time to identify the most compact faithful evidence set.

## Method

### Overall Architecture

The ToE framework consists of three phases. Phase I independently trains modality-specific classifiers (BiGRU for time series, frozen BioClinicalBERT for text). Phase II freezes the encoders and trains a lightweight MLP selector, learning evidence scores via STE top-k masking. Phase III performs beam search at inference time, constructing a compact evidence set by jointly optimizing three objectives: decision consistency, probability stability, and sparsity. Inputs are 24-hour ICU time-series windows and radiology report text segments; outputs are binary predictions along with corresponding evidence traces.

### Key Designs

1. **Evidence Bottleneck Predictor (EB)**:

    - **Function**: Learns interpretable scores for each discrete evidence unit.
    - **Mechanism**: Each modality independently adopts a "selector–predictor" architecture. The selector MLP scores each evidence unit as $s_i = f_\theta(u_i)$, and a differentiable top-k hard mask is realized via the Straight-Through Estimator (STE). The predictor uses only the selected subset for prediction. The two streams are trained separately, and their logits are summed at inference time for fusion.
    - **Design Motivation**: The selector–predictor separation ensures the model cannot "cheat" by accessing unselected information. Phase II updates only the 98K-parameter selector MLP; STE gradient mismatch affects magnitude but not ranking.

2. **Multimodal Role Separation (Context vs. Evidence)**:

    - **Function**: Separates static baseline information from dynamic information to focus the search space.
    - **Mechanism**: CXR/ECG serve as fixed contextual priors concatenated into the representation, while vital sign time windows and clinical notes serve as searchable evidence. The search space is restricted to dynamic evidence; context is always retained.
    - **Design Motivation**: Mirrors clinical reasoning—"given a patient's baseline risk, which dynamic changes explain the outcome?"—and prevents the search from wasting budget on static confirmatory signals.

3. **Inference-Time Beam Search (ToE Search)**:

    - **Function**: Finds a compact and faithful evidence set through multi-step deliberate search at inference time.
    - **Mechanism**: The scoring function is $\text{score}(\mathbf{m}) = C(\mathbf{m}) + \lambda S(\mathbf{m}) - \mu K(\mathbf{m})$, where $C$ denotes decision consistency, $S = 1 - |p_{\text{full}} - p(\mathbf{m})|$ denotes probability stability, and $K$ denotes evidence cost. Starting from the empty set, evidence units are added incrementally; the top-$B$ states are retained, and search terminates when a threshold is met.
    - **Design Motivation**: The probability-space stability term ensures selected evidence is not merely "sufficient" but also faithful to the model's full decision calibration. Beam search captures cross-modal synergistic dependencies that greedy top-k selection cannot discover.

### Loss & Training

Phase I trains both modality streams independently using class-balanced binary cross-entropy. Phase II freezes the encoders and trains only the selector MLP. Inference requires no additional training—only beam search is executed.

## Key Experimental Results

### Main Results

**MIMIC-IV E1: In-hospital Mortality Prediction under Varying Evidence Budgets**

| Method | k=1 AUROC | k=1 Fidelity MAE↓ | k=5 AUROC | k=5 Fidelity MAE↓ |
|--------|-----------|-------------------|-----------|-------------------|
| LIME | 0.564 | 0.229 | 0.695 | 0.171 |
| SHAP | 0.764 | 0.123 | 0.801 | 0.039 |
| ToE | **0.783** | **0.096** | **0.800** | **0.040** |
| Full Model | 0.800 | — | 0.800 | — |

### Ablation Study

**Comparison with LLMs and CBMs**

| Method | Parameters | AUROC | AUPRC |
|--------|------------|-------|-------|
| Hard CBM (24 concepts) | — | 0.775 | 0.349 |
| Med42-v2-70B | 70B | 0.745 | 0.293 |
| ToE (k=5) | 109M | **0.800** | — |

### Key Findings

- ToE retains 98%+ of full-model AUROC with only 5 evidence units, consistently across 6 tasks.
- At k=1, ToE reduces Fidelity MAE by 56% compared to LIME and achieves 22 percentage points higher AUROC.
- Qualitative analysis shows ToE adaptively searches: simple cases rely solely on vital signs, while ambiguous signals trigger inclusion of text.
- Results are stable under cross-center validation (eICU, 208 hospitals) and in a non-medical domain (LEMMA-RCA).

## Highlights & Insights

- The "System 2 search" analogy is apt—interpretability is elevated from passive attribution to active search, with the search process itself being auditable.
- The probability-space stability term is elegantly designed: in ICU settings, most patients have $p$ close to 0 or 1, so logit-space deviations have minimal impact in probability space.
- The 109M-parameter ToE outperforms the 70B Med42, demonstrating that structured approaches substantially outperform general-purpose LLMs on structured prediction tasks.

## Limitations & Future Work

- Evidence unit granularity (1-hour windows, 3-sentence text segments) is predefined; different tasks may require different granularities.
- Beam search yields heuristic rather than global optima, though the gap with exhaustive search is <0.001 AUROC at small $k$.
- Training modality-specific encoders and selectors is required in advance; the approach is not plug-and-play.
- Validation at finer-grained evidence units (e.g., image pixel regions or waveform segments) has not been conducted.

## Related Work & Insights

- **vs. LIME/SHAP**: These are post-hoc approximations without hard selection mechanisms; ToE achieves substantially higher fidelity under sparse evidence budgets.
- **vs. Concept Bottleneck Models**: CBMs require predefined concept annotations and perform static inference; ToE dynamically discovers evidence from learned representations.
- **vs. Tree-of-Thoughts**: ToT searches in the token generation space; ToE searches in the evidence selection space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of inference-time beam search to multimodal interpretability; the framework is complete and original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, 6 tasks, cross-center validation, and comparisons with LLMs and CBMs.
- Writing Quality: ⭐⭐⭐⭐ The System 1/2 analogy is clear; method description is thorough.
- Value: ⭐⭐⭐⭐ Provides a practical and auditable mechanism for deploying multimodal models in high-stakes domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](../../CVPR2026/multimodal_vlm/similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](../../ICLR2026/multimodal_vlm/grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)
- [\[AAAI 2026\] Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts](../../AAAI2026/multimodal_vlm/format_matters_the_robustness_of_multimodal_llms_in_reviewing_evidence_from_tabl.md)

</div>

<!-- RELATED:END -->
