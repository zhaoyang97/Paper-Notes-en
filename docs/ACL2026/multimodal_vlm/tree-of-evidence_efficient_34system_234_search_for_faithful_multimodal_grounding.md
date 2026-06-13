---
title: >-
  [Paper Note] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Interpretability] This paper proposes Tree-of-Evidence (ToE), an inference-time discrete beam search algorithm that formalizes multimodal model interpretability as a discrete optimiz…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal Interpretability"
  - "Evidence Search"
  - "Clinical Prediction"
  - "Beam Search"
  - "Concept Bottleneck"
date: 2026-05-08
content_hash: ff475be35c94f393
---

# Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.07692](https://arxiv.org/abs/2604.07692)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Interpretability, Evidence Search, Clinical Prediction, Beam Search, Concept Bottleneck

## TL;DR

This paper proposes Tree-of-Evidence (ToE), an inference-time discrete beam search algorithm that formalizes multimodal model interpretability as a discrete optimization problem over coarse-grained evidence units (vital sign time windows, radiology report snippets). It preserves over 98% of the full-input model's AUROC using only 5 evidence units while generating auditable evidence traces.

## Background & Motivation

**Background**: Large Multimodal Models (LMMs) have achieved SOTA performance in high-stakes fields such as healthcare, but their reasoning processes remain opaque. Existing interpretability methods include post-hoc attribution methods like attention visualization, gradient saliency, LIME/SHAP, and Concept Bottleneck Models (CBM).

**Limitations of Prior Work**: (1) Attention weights are often unfaithful to the model's actual decision logic; (2) LIME/SHAP provide approximations rather than guarantees and cannot provide discrete evidence selection; (3) CBMs require pre-defined concept annotations and are static during inference, lacking adaptive search capabilities; (4) Existing rationale extraction methods are typically limited to single modalities (primarily text) and fail to capture cross-modal synergistic dependencies.

**Key Challenge**: Clinical deployment requires that model predictions be explicitly traceable to specific, verifiable evidence, but existing methods are either unfaithful, do not support multimodality, or fail to provide an audit trail.

**Goal**: Design an inference-time search algorithm capable of finding compact sets of multimodal evidence that both replicate full-input predictions and provide an auditable search process.

**Key Insight**: Borrowing the deliberate branching search idea from Tree-of-Thoughts, this work treats interpretability as a discrete search problem—a "System 2" style multi-step deliberate search, rather than a "System 1" style single-pass greedy ranking.

**Core Idea**: The multimodal input space is structured into "Global Context" (fixed priors, such as CXR/ECG baselines) and "Searchable Evidence" (dynamically changing vital signs and notes). A compact, faithful evidence set is found by training a lightweight Evidence Bottleneck scorer and executing beam search during inference.

## Method

### Overall Architecture

The ToE framework consists of three phases: Phase I independently trains modality-specific classifiers (BiGRU for time series, frozen BioClinicalBERT for text); Phase II trains a lightweight MLP selector after freezing the encoders, learning evidence scores via STE top-k masking; Phase III executes beam search during inference to construct a compact evidence set by combining three objectives: decision consistency, probability stability, and sparsity. The input consists of 24-hour ICU time-series windows and radiology report text snippets, while the output is a binary classification prediction and its corresponding evidence trace.

### Key Designs

1.  **Evidence Bottleneck Predictor (EB)**:
    *   **Function**: Learns interpretable scores for each discrete evidence unit.
    *   **Mechanism**: A "selector-predictor" architecture is constructed independently for each modality. The selector MLP scores each evidence unit $s_i = f_\theta(u_i)$, implementing differentiable top-k hard mask selection via STE. The predictor only uses the selected subset for prediction. Both streams are trained separately and fused via logit summation during inference.
    *   **Design Motivation**: The selector-predictor separation ensures the model cannot "cheat" by accessing unselected information; Phase II only updates the selector MLP with 98K parameters, where STE gradient mismatch affects magnitude but not ranking.

2.  **Multimodal Role Separation (Context vs. Evidence)**:
    *   **Function**: Separates static baseline information from dynamic changes to focus the search space.
    *   **Mechanism**: CXR/ECG are concatenated into the representation as fixed context priors, while vital sign time windows and clinical notes serve as searchable evidence. The search space is limited to dynamic evidence, while the context is always retained.
    *   **Design Motivation**: Simulates clinical reasoning logic—"Given the patient's baseline risk, what dynamic changes explain the outcome"—preventing the search from wasting the budget on static confirmatory signals.

3.  **Inference-time Beam Search (ToE Search)**:
    *   **Function**: Finds a compact and faithful evidence set through multi-step deliberate search during inference.
    *   **Mechanism**: The scoring function is defined as $\text{score}(\mathbf{m}) = C(\mathbf{m}) + \lambda S(\mathbf{m}) - \mu K(\mathbf{m})$, where $C$ is decision consistency, $S = 1 - |p_{\text{full}} - p(\mathbf{m})|$ is probability stability, and $K$ is evidence cost. The search starts from an empty set, gradually adds evidence while retaining the top-B states, and terminates when a threshold is met.
    *   **Design Motivation**: The probability space stability term ensures that the selected evidence is not only "sufficient" but also faithful to the model's full decision calibration; beam search captures cross-modal synergistic dependencies that greedy top-k cannot discover.

### Loss & Training

Phase I uses category-balanced binary cross-entropy to train the two modality streams independently. Phase II freezes the encoders and trains only the selector MLP. No training is required during inference; only beam search is executed.

## Key Experimental Results

### Main Results

**MIMIC-IV E1: In-hospital mortality prediction, comparison under different evidence budgets**

| Method | k=1 AUROC | k=1 Fidelity MAE↓ | k=5 AUROC | k=5 Fidelity MAE↓ |
| :--- | :--- | :--- | :--- | :--- |
| LIME | 0.564 | 0.229 | 0.695 | 0.171 |
| SHAP | 0.764 | 0.123 | 0.801 | 0.039 |
| **Ours (ToE)** | **0.783** | **0.096** | **0.800** | **0.040** |
| Full Model | 0.800 | — | 0.800 | — |

### Ablation Study

**Comparison with LLMs and CBM**

| Method | Parameters | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| Hard CBM (24 concepts) | — | 0.775 | 0.349 |
| Med42-v2-70B | 70B | 0.745 | 0.293 |
| **Ours (ToE, k=5)** | 109M | **0.800** | — |

### Key Findings

*   ToE retains over 98% of the full model's AUROC using only 5 evidence units, consistent across 6 tasks.
*   At k=1, ToE reduces Fidelity MAE by 56% compared to LIME, with an AUROC 22 percentage points higher.
*   Qualitative analysis shows ToE performs adaptive search: using only vital signs for simple cases, and introducing text when signals are ambiguous.
*   Stability is demonstrated through cross-center validation (eICU, 208 hospitals) and non-medical domains (LEMMA-RCA).

## Highlights & Insights

*   The "System 2 search" analogy is apt—upgrading interpretability from passive attribution to active search, where the search process itself is auditable.
*   The probability space stability term is ingeniously designed—in ICU scenarios where most patients have $p$ near 0 or 1, logit space deviations have minimal impact in the probability space.
*   ToE, with 109M parameters, outperforms the 70B Med42, indicating that structured methods are far superior to general LLMs for structured prediction.

## Limitations & Future Work

*   The granularity of evidence units (1-hour windows, 3-sentence text snippets) is pre-set; different tasks may require different granularities.
*   Beam search is a heuristic optimum rather than a global optimum, though the gap with exhaustive search is $<0.001$ AUROC at small $k$.
*   It requires prior training of modality-specific encoders and selectors and is not plug-and-play.
*   It has not been validated on finer-grained evidence units such as image pixel-level or waveform segment-level data.

## Related Work & Insights

*   **vs. LIME/SHAP**: The latter are post-hoc approximations without a hard selection mechanism; ToE exhibits significantly higher faithfulness under sparse budgets.
*   **vs. Concept Bottleneck Models**: CBMs require pre-defined concept annotations and involve static inference; ToE dynamically discovers evidence from learned representations.
*   **vs. Tree-of-Thoughts**: ToT searches in the token generation space, whereas ToE searches in the evidence selection space.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ First to apply inference-time beam search to multimodal interpretability with a complete, original framework.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 tasks across 3 datasets + cross-center validation + LLM/CBM comparisons.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear System 1/2 analogy and detailed methodological descriptions.
*   **Value**: ⭐⭐⭐⭐ Provides a practical, auditable mechanism for deploying multimodal models in high-stakes fields.

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
