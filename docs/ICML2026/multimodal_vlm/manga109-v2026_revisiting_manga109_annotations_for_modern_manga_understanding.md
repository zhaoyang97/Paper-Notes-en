---
title: >-
  [Paper Note] Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding
description: >-
  [ICML 2026][Multimodal VLM][Manga Understanding] The authors revisit Manga109, the foundational dataset for manga AI research, identifying five categories of dialogue text annotation issues. By combining commercial OCR…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Manga Understanding"
  - "OCR Evaluation"
  - "Dataset Revision"
  - "Onomatopoeia"
  - "Speech Bubble Segmentation"
date: 2026-05-08
content_hash: 38d2e80442497a94
---

# Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding

**Conference**: ICML 2026  
**arXiv**: [2605.21182](https://arxiv.org/abs/2605.21182)  
**Code**: None (Project Page: https://manga109.github.io/manga109-project-website/en/ )  
**Area**: Multimodal VLM  
**Keywords**: Manga Understanding, OCR Evaluation, Dataset Revision, Onomatopoeia, Speech Bubble Segmentation

## TL;DR
The authors revisit Manga109, the foundational dataset for manga AI research, identifying five categories of dialogue text annotation issues. By combining commercial OCR, dual LLM voting (GPT-5 & Gemini 3 Flash), and human verification, they revised approximately 29,000 annotations (19.6% of the 147,887 text annotations). The resulting Manga109-v2026 improves the end-to-end OCR H-mean from 48.5 to 62.9 (+14.4 pp).

## Background & Motivation

**Background**: Japanese manga is a unique multimodal medium blending visual storytelling, stylized typography, speech bubbles, and onomatopoeia. Manga109 (Matsui 2017; Aizawa 2020) serves as the cornerstone for tasks like manga OCR, translation, transcription, and multimodal understanding, with nearly all manga-AI systems trained or evaluated on it.

**Limitations of Prior Work**: The dialogue text annotations in Manga109 were created over a decade ago based on OCR standards of that time, leading to significant mismatches with modern multimodal systems. Issues such as transcription errors, oversized bounding boxes (encompassing multiple text lines or non-text pixels), missing short emoticons ("!", "..."), overlapping annotations between dialogue and onomatopoeia, and grouping multiple adjacent bubbles into a single text region cause "correct detections" by modern OCR systems to be penalized as errors and introduce inaccurate signals during training.

**Key Challenge**: There is a fundamental mismatch in representation granularity between the expressive structures of manga (onomatopoeia, bubble layout, stylized fonts) and the behavior of modern detectors. Legacy annotations organize "entire dialogues into one box," whereas modern OCR tends to output each bubble as an independent instance. Furthermore, mixing onomatopoeia into dialogue boxes disrupts the rhetorical preservation required for manga translation.

**Goal**: Systematically identify and revise five categories of dialogue annotation issues without altering the overall Manga109 framework, and empirically demonstrate the impact of these revisions on the reliability of OCR evaluation.

**Key Insight**: Rather than treating any single OCR as the ground truth, the authors use commercial Mantra OCR output as a "discrepancy probe." Whenever the OCR output diverges from legacy annotations, the instance is sent to a human/LLM verification pipeline. This transforms the "total manual audit" problem into "audit by discrepancy sampling," significantly reducing human overhead while focusing on areas that truly impact modern systems.

**Core Idea**: Leverage modern AI (commercial OCR + dual LLM voting) for large-scale "candidate discovery," followed by human "final adjudication," applying human-in-the-loop iteration to the generational maintenance of cultural data assets.

## Method

The work does not propose a new model but rather a human-AI collaborative pipeline of **Problem Discovery $\rightarrow$ Categorization $\rightarrow$ Revision $\rightarrow$ Evaluation Verification**. It can be viewed as a "code review pipeline" for manga dialogue annotations: OCR acts as the linter, LLMs act as Reviewer 1 and Reviewer 2, and humans act as the maintainers.

### Overall Architecture
The input consists of all 147,887 dialogue text annotations from the original Manga109. The process follows four steps:
(1) Extract detection/recognition outputs from commercial Mantra OCR for all manga pages;
(2) Perform spatial and text alignment between OCR outputs and original annotations to automatically filter five types of discrepancy candidates;
(3) Execute specific revision sub-flows for each type (LLM voting / region sub-division / manual supplement / overlap removal / bubble re-segmentation);
(4) Run the same OCR output against both old and new annotations using the MangaOCR protocol to compare E2E precision/recall/H-mean. The final Manga109-v2026 includes $\approx 29,000$ revisions, covering about 19.6% of all annotations.

### Key Designs

1. **OCR-as-Probe Strategy for Problem Discovery**:
    - **Function**: Uses a modern commercial OCR (not as GT) as a discrepancy probe to automatically identify where legacy annotations are most likely to mismatch modern systems.
    - **Mechanism**: The authors explicitly state that OCR output is not ground truth; it serves as a "high-recall candidate generator." Consistent parts are deemed reliable, while divergent parts enter the verification queue. Discrepancies are categorized by geometric relationships: misaligned bboxes (Type 1), single bbox covering multiple OCR boxes (Type 2 or Type 5), OCR text with no annotation (Type 3), and bbox overlapping with onomatopoeia boxes (Type 4).
    - **Design Motivation**: Compressing the audit of 148k annotations into a subset that diverges from modern systems is key to completing the work with viable manpower. Excluding OCR from the GT role avoids circular bias in evaluation.

2. **Dual LLM Voting + Human Verification for Transcription (Type 1, $\approx 9,200$ items)**:
    - **Function**: Decides whether to keep the original annotation or adopt the OCR output for Type 1 candidates.
    - **Mechanism**: Both the original annotation and the OCR output are fed to GPT-5 and Gemini 3 Flash to independently select the better transcription. Agreement results in automatic adoption (15,359 cases: 7,156 for OCR, 8,203 for original); 2,051 disagreements are manually adjudicated. Total Type 1 revisions: 9,207. Expected efficiency: Automatic decision rate $\approx 15359/17410 \approx 88.2\%$, with only 11.8% requiring human intervention.
    - **Design Motivation**: A single model might have systemic biases. Dual-model alignment reduces this bias, while manual adjudication of high-discrepancy samples maintains the quality ceiling. This "AI high-throughput + human high-precision" division is the core of the pipeline's efficiency.

3. **Semantic-based Geometric Revisions (Type 2/3/4/5, $\approx 19,800$ items)**:
    - **Function**: Splits geometric revisions into four sub-flows based on problem semantics to avoid over-simplification.
    - **Mechanism**: Type 2 (Oversized Bbox, $\approx 50$) $\rightarrow$ Manually split large boxes into small boxes matching text segments; Type 3 (Missing Text, $\approx 800$) $\rightarrow$ Verify OCR outputs where annotations are missing and add bboxes/transcriptions (focusing on "!", "..."); Type 4 (Text-Onomatopoeia Overlap, $\approx 4,300$) $\rightarrow$ Preserve onomatopoeia annotations added in 2022 and crop overlapping dialogue boxes; Type 5 (Under-segmented Bubbles, $\approx 14,900$) $\rightarrow$ Split legacy bboxes covering multiple OCR bubbes into one bbox per bubble.
    - **Design Motivation**: Type 2/3 fix consistency, Type 4 fixes compatibility for translation systems (treating onomatopoeia separately), and Type 5 fixes evaluation protocol compatibility. Split-track design prevents rule conflicts that would occur with a single IoU threshold.

### Verification Strategy
Instead of a new model, the same commercial OCR output is compared against both original Manga109 and Manga109-v2026 using the MangaOCR (Baek 2026) E2E evaluation protocol. This "fixed OCR, variable annotations" design quantifies the impact of annotation quality on OCR evaluation independently of the specific OCR system used.

## Key Experimental Results

### Annotation Revision Scale

| Type | Description | Revisions |
|------|------|--------|
| Type 1 | Transcription Errors | $\approx 9,200$ |
| Type 2 | Oversized Bounding Boxes | $\approx 50$ |
| Type 3 | Missing Text | $\approx 800$ |
| Type 4 | Dialogue-Onomatopoeia Overlap | $\approx 4,300$ |
| Type 5 | Under-segmented Speech Bubbles | $\approx 14,900$ |
| **Total** | — | **$\approx 29,000$ (19.6% of 147,887)** |

### OCR Evaluation Results (Same OCR output)

| Annotation Version | Precision | Recall | H-mean |
|----------|-----------|--------|--------|
| Original Manga109 | 46.5 | 50.6 | 48.5 |
| Manga109-v2026 | 63.4 | 62.4 | **62.9** |
| **Gain** | +16.9 | +11.8 | **+14.4 pp** |

### Key Findings
- The +14.4 pp H-mean improvement reflects evaluation protocol alignment rather than increased OCR capability—suggesting that prior assessments of manga OCR systems have been systematically underestimated.
- Type 5 (Under-segmented bubbles) accounts for over half of all revisions, indicating that "legacy grouping by dialogue block" is the most severe representation mismatch.
- Dual LLM voting consistency was $\approx 88.2\%$, with 53.4% favoring original annotations and 46.6% favoring OCR—showing comparable quality but high disagreement on stylized or small text.

## Highlights & Insights
- **"Dataset Refinement" as an ICML Paper**: In the LLM/VLM era, generational maintenance of benchmark annotations is methodologically valuable. Reorganizing data to align with modern system behavior can lead to order-of-magnitude changes in evaluation.
- The **OCR-as-Probe + Dual LLM Voting + Human Adjudication** pipeline reduces "total human audit" costs to 11.8%, providing an engineering paradigm for updating other legacy datasets (e.g., VQA, RefCOCO).
- **Expressive Compatibility**: Explicitly addressing the compatibility between expressive structures (onomatopoeia) and evaluation protocols ensures that annotations evolve alongside downstream tasks.

## Limitations & Future Work
- The commercial Mantra OCR is closed-source. Collaborative replication requires substituting it with open-source OCR, which may yield different discrepancy sets.
- Geometric revisions (Types 2-5) relied heavily on manual labor by the four authors; no automated script was provided for replication.
- Only "dialogue text" was revised; higher-level annotations like panels and characters remain untouched, though manga VLM evaluation relies heavily on them.

## Related Work & Insights
- **vs MangaOCR (Baek 2026)**: While MangaOCR improved the model, this work improves the benchmark; the authors utilize the MangaOCR protocol to quantify improvements.
- **vs COO (Baek 2022)**: COO introduced onomatopoeia labels; this work builds on that by resolving spatial conflicts between dialogue and onomatopoeia.
- **Insight**: Using different frontier LLMs as weak-label ensemblers for discrepancy samples is a robust pattern for large-scale data maintenance.

## Rating
- **Novelty**: ⭐⭐⭐ (Methodology of AI-assisted generational maintenance is significant even without a new model.)
- **Experimental Thoroughness**: ⭐⭐⭐ (Sufficient to prove the point, but lacks multi-OCR validation or inter-annotator agreement metrics.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear categorization with helpful statistical breakdowns.)
- **Value**: ⭐⭐⭐⭐⭐ (Manga109 is the de facto benchmark; this update essentially upgrades the reliability of the entire manga AI toolchain.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Revisiting Model Stitching in the Foundation Model Era](../../CVPR2026/multimodal_vlm/revisiting_model_stitching_in_the_foundation_model.md)
- [\[ACL 2026\] Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines](../../ACL2026/multimodal_vlm/reducing_peak_memory_usage_for_modern_multimodal_large_language_model_pipelines.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](../../AAAI2026/multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] Revisiting Model Stitching in the Foundation Model Era](../../CVPR2026/multimodal_vlm/revisiting_model_stitching_in_the_foundation_model.md)
- [\[ACL 2026\] Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines](../../ACL2026/multimodal_vlm/reducing_peak_memory_usage_for_modern_multimodal_large_language_model_pipelines.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](../../AAAI2026/multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/multimodal_vlm/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
