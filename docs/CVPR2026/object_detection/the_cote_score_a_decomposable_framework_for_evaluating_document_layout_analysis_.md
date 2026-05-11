---
title: >-
  [Paper Note] The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models
description: >-
  [CVPR2026][Object Detection][Document Layout Analysis] This paper proposes COTe (Coverage, Overlap, Trespass, Excess), a decomposable evaluation framework for Document Layout Analysis (DLA), along with the concept of Structural Semantic Units (SSUs). Compared to conventional IoU/mAP/F1 metrics, COTe more accurately reflects page parsing quality and reveals model-specific failure modes.
tags:
  - CVPR2026
  - Object Detection
  - Document Layout Analysis
  - Evaluation Metrics
  - Decomposable Metrics
  - Structural Semantic Unit
  - Page Parsing
  - Granularity Robustness
date: 2026-05-08
content_hash: 5ddaa64ac1e288fd
---

# The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models

**Conference**: CVPR2026  
**arXiv**: [2603.12718](https://arxiv.org/abs/2603.12718)  
**Code**: [JonnoB/cotescore](https://github.com/JonnoB/cotescore)  
**Area**: Object Detection / Document Layout Analysis  
**Keywords**: Document Layout Analysis, Evaluation Metrics, Decomposable Metrics, Structural Semantic Unit, Page Parsing, Granularity Robustness

## TL;DR

This paper proposes COTe (Coverage, Overlap, Trespass, Excess), a decomposable evaluation framework for Document Layout Analysis (DLA), along with the concept of Structural Semantic Units (SSUs). Compared to conventional IoU/mAP/F1 metrics, COTe more accurately reflects page parsing quality and reveals model-specific failure modes.

## Background & Motivation

1. **Mismatch between metrics and task**: Current DLA practice relies on general-purpose object detection metrics (IoU, F1, mAP), which were designed for 2D projections of 3D scenes (natural images). Documents are natively 2D irregular tessellation structures, creating a fundamental incompatibility.
2. **Granularity sensitivity**: IoU is highly sensitive to annotation granularity (line-level vs. paragraph-level). When prediction and ground-truth granularity differ (e.g., paragraph-level predictions against line-level ground truth), F1 can drop from 1.0 to 0.32, producing severely misleading results.
3. **Incommensurable annotation schemes**: Major annotation standards such as PAGE, META/ALTO, hOCR, and TEI overlap but are not fully interoperable, and dataset-specific custom schemes further impede cross-dataset comparability.
4. **Indistinguishable failure modes**: Traditional metrics reduce all errors to a binary detected/not-detected judgment, failing to distinguish between overlapping predictions, boundary trespass, and insufficient coverage — fundamentally different page parsing errors.
5. **Zero-shot evaluation demands**: Real-world applications frequently require cross-dataset zero-shot evaluation, yet traditional metrics exhibit severe deficiencies in pragmatic competence under such conditions.
6. **Lack of semantic awareness**: Traditional metrics focus on the physical location of text rather than its semantic structure, failing to capture the core objective of DLA — correctly preserving semantic boundaries.

## Method

### 1. Structural Semantic Unit (SSU)

An SSU is a relational annotation method that aggregates physical text regions according to semantic meaning. Two regions belong to the same SSU if and only if they satisfy four conditions:

- They share the same class (same_class)
- They belong to the same structural unit (e.g., the same column)
- They belong to the same semantic unit (e.g., the same article)
- They are adjacent in reading order

The key design of SSUs lies in allowing multiple predicted bounding boxes to be assigned to a single SSU (many-to-one), with each prediction assigned to the SSU with the maximum area overlap. This fundamentally achieves robustness to annotation granularity differences.

### 2. The Four Components of the COTe Score

Given a page pixel matrix $M$ ($H \times W$), SSU mask $M^S$, and prediction mask $M^p$:

- **Coverage**: $\mathcal{C} = \frac{\sum M^S \odot M^{p,b}}{A^S}$, measuring the degree to which predictions cover the ground truth; $[0,1]$, higher is better.
- **Overlap**: $\mathcal{O} = \frac{\sum M^S \odot (M^p - M^{p,b})}{A^S}$, penalizing redundant coverage of the same region by multiple predictions, which causes OCR text duplication.
- **Trespass**: $\mathcal{T} = \sum_j \frac{\sum M^S_{\setminus i} \odot M^p_j}{A^S}$, penalizing predictions that cross SSU semantic boundaries, causing unrelated text to be merged.
- **Excess**: $E = \frac{\sum \mathcal{N} \odot M^{p,b}}{A^{\mathcal{N}}}$, measuring the area of predictions extending beyond all ground-truth regions; serves as an auxiliary indicator.

### 3. Overall Score Computation

$$\text{COTe} = \mathcal{C} - \mathcal{O} - \mathcal{T}$$

A perfect score is 1 (full coverage, no overlap, no trespass); the score is 0 when no predictions are made, and negative values are permitted for severe errors. The additive structure makes each component directly interpretable.

### 4. Multi-Class Extension

The framework supports extension to $K$ classes, enabling generation of asymmetric confusion matrices to analyze coverage, overlap, and trespass patterns across categories, facilitating fine-grained model diagnosis.

## Key Experimental Results

### Experimental Setup

- **3 datasets**: NCSEv2 (newspaper, 31 pages, with SSU), HNLA2013 (newspaper, 50 pages, with SSU), DocLayNet (multi-format, 4,999 pages, without SSU)
- **5 models**: DocLayout-YOLO (15.4M), Heron (42.9M), PP-DocLayout-L/M/S (30.94M / 5.65M / 1.21M)
- **Zero-shot evaluation**: All models use pretrained weights without fine-tuning

### Main Results

| Dataset | Best Model (COTe) | COTe | Coverage | Overlap | Trespass | Best Model (mAP) | mAP |
|---------|-------------------|------|----------|---------|----------|------------------|-----|
| NCSE | PPDoc-L | 0.72 | 0.80 | 0.05 | 0.03 | Heron | 0.56 |
| HNLA2013 | YOLO | **0.86** | 0.96 | 0.06 | 0.05 | Heron | 0.53 |
| DocLayNet | PPDoc-L | 0.47 | 0.67 | 0.02 | 0.18 | PPDoc-L | 0.01 |

**Key finding**: The best model identified by COTe differs from that identified by traditional metrics across all three datasets.

### Granularity Robustness Comparison (Case Study)

| Metric | GT: Line-level, Pred: Paragraph-level | GT: Paragraph-level, Pred: Line-level |
|--------|---------------------------------------|---------------------------------------|
| F1 | 0.32 | 0.32 |
| Mean IoU | 0.35 | 0.60 |
| COTe | **1.00** | **0.84** |

Under perfect parsing conditions, F1 drops to 0.32 (misleading by 68%), while COTe decreases by at most 16% — a **76%** reduction in relative misinterpretation.

### Ablation Study: Contribution of SSUs

Re-evaluating NCSE and HNLA2013 without SSU annotations results in only a minor decrease in COTe (primarily a slight increase in Trespass). Granularity robustness derives mainly from the many-to-one prediction assignment mechanism rather than expanded semantic area, which lowers the barrier to adoption.

### Failure Mode Analysis

- **DocLayout-YOLO**: High coverage but high overlap and high trespass; in a single-page example, Coverage = 0.98 yet COTe = −0.55.
- **Heron**: Consistently low trespass, but tends toward higher overlap (Overlap = 0.28 on NCSE).
- **PP-DocLayout series**: Model size and decomposed error exhibit a non-linear relationship; PPDoc-M outperforms PPDoc-L in terms of overlap.

## Highlights & Insights

- **Strong originality**: The first decomposable evaluation framework designed natively for 2D media, with a semiotic argument (Peircean triadic sign) grounding the "pragmatic failure" of traditional metrics.
- **High practicality**: Most granularity-robustness benefits are attainable without SSU annotation, significantly lowering the barrier to use.
- **Diagnostic power**: A single COTe score decomposes into Coverage, Overlap, and Trespass, directly localizing specific model failure modes.
- **Open-source ecosystem**: Releases the `cotescore` Python library, SSU-annotated datasets, and an automatic PAGE-format annotator.

## Limitations & Future Work

- **Sensitivity to loose annotations**: When ground-truth regions contain substantial whitespace (e.g., poetry, low-density text), Coverage may be artificially deflated.
- **Limited multi-class evaluation**: Although multi-class extension and confusion matrices are defined, experiments are conducted using only single-class evaluation.
- **Limited data scale**: NCSE contains only 31 pages and HNLA2013 only 50, with a predominance of newspaper layouts, limiting representativeness for modern documents (e.g., table- or formula-intensive formats).
- **No downstream task linkage**: The correlation between COTe and OCR accuracy or information extraction quality is not directly validated.
- **Subjectivity in SSU definition**: The definitions of semantic and structural units within SSUs are left to the user, potentially introducing new inconsistencies.

## Related Work & Insights

- **vs. IoU / F1 / mAP**: COTe is a natively 2D metric robust to annotation granularity; traditional metrics are severely misleading under granularity mismatch (F1 drops 68% vs. COTe drops 16%).
- **vs. PubLayNet / DocLayNet benchmarks**: These benchmarks adopt conventional mAP evaluation; the Spearman correlation between COTe and mAP is only 0.27–0.77, indicating that the two measure different dimensions.
- **vs. parsing failure mode taxonomy [Clausner et al.]**: Prior work proposed a failure mode taxonomy without quantitative metrics; COTe quantifies failure modes as continuous values.
- **vs. dataset shift / construct validity**: This paper supplements ML-community discussions on dataset shift and annotation noise from a semiotic perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first decomposable evaluation framework targeting the tessellation structure of documents; the conceptual design is original.
- Experimental Thoroughness: ⭐⭐⭐ — Coverage across 3 datasets and 5 models is reasonable, but data scale is limited and downstream task linkage is absent.
- Writing Quality: ⭐⭐⭐⭐ — The paper progresses logically from semiotic motivation to mathematical formulation to empirical analysis, with intuitive case studies.
- Value: ⭐⭐⭐⭐ — Addresses a critical pain point in DLA evaluation; the open-source library lowers adoption barriers and makes a tangible contribution to the document understanding community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CompAgent: An Agentic Framework for Visual Compliance Verification](compagent_an_agentic_framework_for_visual_compliance_verification.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)
- [\[ACL 2026\] Evaluating Memory Capability in Continuous Lifelog Scenario](../../ACL2026/object_detection/evaluating_memory_capability_in_continuous_lifelog_scenario.md)
- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](../../NeurIPS2025/object_detection/overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)

</div>

<!-- RELATED:END -->
