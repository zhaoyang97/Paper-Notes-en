---
title: >-
  [Paper Note] Towards Scalable Web Accessibility Audit with MLLMs as Copilots
description: >-
  [AAAI2026][Multimodal VLM][web accessibility] This paper proposes the AAA framework, which operationalizes the WCAG-EM standard through two key innovations—GRASP (Graph-based multimodal page sampling) and MaC (MLLM as Co…
tags:
  - "AAAI2026"
  - "Multimodal VLM"
  - "web accessibility"
  - "WCAG-EM"
  - "multimodal LLM"
  - "graph neural network"
  - "page sampling"
date: 2026-05-08
content_hash: d5d39aefc2cd447c
---

# Towards Scalable Web Accessibility Audit with MLLMs as Copilots

**Conference**: AAAI2026
**arXiv**: [2511.03471](https://arxiv.org/abs/2511.03471)  
**Code**: [eaglelab-zju/AAA](https://github.com/eaglelab-zju/AAA)  
**Area**: Multimodal VLM
**Keywords**: web accessibility, WCAG-EM, multimodal LLM, graph neural network, page sampling

## TL;DR
This paper proposes the AAA framework, which operationalizes the WCAG-EM standard through two key innovations—GRASP (Graph-based multimodal page sampling) and MaC (MLLM as Copilot)—enabling scalable end-to-end web accessibility auditing.

## Background & Motivation
Web accessibility is fundamental to digital inclusion, yet recent surveys reveal that **94.8%** of top-one-million website homepages contain WCAG violations. The root cause lies not in a lack of education or tooling, but in the resource bottleneck of the audit process itself:

- **Limitations of existing tools**: Tools such as WAVE and Axe only perform hard-coded rule checks (e.g., missing alt text, insufficient contrast) and cannot cover semantic or cognitive accessibility issues.
- **Difficulty executing WCAG-EM**: Although W3C's five-step audit methodology is well-standardized, no technical framework exists to support its execution at scale.
- **Inadequate page sampling**: Existing clustering methods (SDC) rely solely on shallow textual statistics, ignoring multimodal semantics such as visual layout and hyperlink structure.
- **Human evaluation bottleneck**: Manually identifying accessibility-critical components (structured pages, complete processes) demands substantial expert effort.

## Method

### Overall Architecture of the AAA Framework
The framework aligns with the five-step WCAG-EM process: website crawling → automated checking → page sampling → manual inspection → reporting/remediation. The core innovations lie in the page sampling and manual inspection stages.

### GRASP: Graph-Based Multimodal Page Sampling
Page representativeness is defined along three dimensions:
1. **Textual semantic representativeness**: BERT is used to extract contextualized semantic representations from DOM text.
2. **Visual layout representativeness**: ViT is used to learn layout-level visual representations from page screenshots.
3. **Link-structural representativeness**: A GNN is applied over the hyperlink graph to learn structural representations.

The fusion pipeline is $\mathbf{X} = \mathbf{H}_t || \mathbf{H}_v$. After GNN message passing, k-means clustering is applied, and the node closest to each cluster centroid is selected as a sampled page. A representativeness-enhanced graph learning module is additionally introduced, which uses clustering results to prune noisy edges and recover semantically similar but non-adjacent edges.

### MaC: MLLM as a Multi-Role Copilot
- **Assistant**: Automatically identifies WCAG-EM-defined structured pages (common/relevant/essential/technology-dependent) to assist individual-feature-based page sampling; pre-extracts accessibility-critical elements (search bars, forms, CAPTCHAs, etc.).
- **Auditor**: Evaluates cognitive accessibility issues overlooked by conventional tools (WCAG 2.2 SC 3.3.8/3.3.9), such as the cognitive load imposed by CAPTCHAs.
- **Consultant**: Provides remediation suggestions (identified as a future direction).

### Four New Datasets
- **TPS**: 97,246 pages from 495 websites, including DOM, screenshots, Axe checks, and adjacency matrices.
- **APR**: 968 pages across 5 website categories, annotated for 4 types of WCAG-EM structured pages.
- **CCT**: 1,985 CAPTCHA images across 17 authentication task types, for evaluating cognitive accessibility.
- **CPE**: 1,199 pages annotated for 5 component categories: search, filter, form, CAPTCHA, and contact information.

## Key Experimental Results

### GRASP Page Sampling (Average over 495 Websites)

| Method | Layout $S_{sampled}$↓ | Layout $D_{intra-inter}$↑ | Text $S_{sampled}$↓ | Text $D_{intra-inter}$↑ |
|---|---|---|---|---|
| SDC_content | 56.66 | 9.96 | 89.29 | 2.73 |
| SDC_tags | 54.18 | 10.76 | 88.76 | 2.12 |
| GRASP_GCN | 51.54 | 13.05 | 86.99 | 1.59 |
| **GRASP_IGNN** | **44.31** | **14.94** | **80.45** | **7.40** |

GRASP_IGNN substantially outperforms baselines in both representation spaces, demonstrating that heterogeneous graph modeling is better suited to website hyperlink structures.

### MaC F1 on APR/CPE
- GPT-4o achieves F1 = 98.01% on search bar identification and F1 = 95.33% on CAPTCHA detection.
- The smaller model Qwen2.5-VL-72B achieves F1 = 80.21% on relevant page identification, surpassing GPT-4o (35.44%).
- For cognitive CAPTCHA classification, fine-tuned InternVL2-8B achieves macro-F1 = 45.58%, outperforming GPT-4o (29.16%).

## Highlights & Insights
- **First end-to-end WAA framework**: Aligns with the full five-step WCAG-EM process, covering the entire audit lifecycle.
- **Multimodal page sampling**: The first approach to jointly integrate textual, visual, and link-structural representativeness; GRASP_IGNN significantly outperforms text-only methods.
- **Multi-role MLLM positioning**: Goes beyond the narrow scope of evaluation and remediation to explore MLLM applications in sampling, pre-audit localization, and cognitive accessibility assessment.
- **Potential of small models**: Experiments demonstrate that fine-tuned 8B models can serve as domain specialists with high cost-effectiveness.

## Limitations & Future Work
- GRASP relies on the quality of BERT/ViT pretraining; its effectiveness on non-English websites has not been validated.
- MLLMs still have substantial room for improvement on tasks such as relevant page identification (GPT-4o F1 only 35%).
- The highest macro-F1 for cognitive CAPTCHA classification is 45.58%, which remains far from practical requirements.
- Dataset scale is limited (APR covers only 968 pages from 5 websites), and generalizability requires further validation.

## Rating
- Novelty: ⭐⭐⭐⭐ — First work to systematically integrate MLLMs and GNNs into the full WCAG-EM audit pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Sampling experiments on 495 websites, comparisons with 5 MLLMs, and 4 datasets provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐ — Framework is clearly presented and well-aligned with the standard, though the paper is detail-heavy.
- Value: ⭐⭐⭐⭐ — Directly applicable to large-scale web accessibility auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach](../../ICLR2026/multimodal_vlm/customizing_visual_emotion_evaluation_for_mllms_an_open-vocabulary_multifaceted_.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)

</div>

<!-- RELATED:END -->
