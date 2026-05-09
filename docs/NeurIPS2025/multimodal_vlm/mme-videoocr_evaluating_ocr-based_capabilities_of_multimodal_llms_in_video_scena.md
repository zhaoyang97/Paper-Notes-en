---
title: >-
  [Paper Note] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios
description: >-
  [NeurIPS 2025][Multimodal VLM][video OCR] This paper introduces MME-VideoOCR, a comprehensive video OCR evaluation benchmark comprising 25 tasks, 44 scenarios, 1,464 videos, and 2,000 manually annotated QA pairs, spanning three levels of text recognition, understanding, and reasoning. Evaluation of 18 state-of-the-art MLLMs reveals that the strongest model (Gemini-2.5 Pro) achieves only 73.7% overall, with cross-frame understanding tasks falling below 25%.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - video OCR
  - benchmark
  - cross-frame understanding
  - language prior bias
  - multimodal LLM evaluation
date: 2026-05-08
content_hash: 913cace3e2adef69
---

# MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios

**Conference**: NeurIPS 2025
**arXiv**: [2505.21333](https://arxiv.org/abs/2505.21333)
**Code**: [https://mme-videoocr.github.io/](https://mme-videoocr.github.io/)
**Area**: Multimodal VLM / Video Understanding / OCR Evaluation
**Keywords**: video OCR, benchmark, cross-frame understanding, language prior bias, multimodal LLM evaluation

## TL;DR
This paper introduces MME-VideoOCR, a comprehensive video OCR evaluation benchmark comprising 25 tasks, 44 scenarios, 1,464 videos, and 2,000 manually annotated QA pairs, spanning three levels of text recognition, understanding, and reasoning. Evaluation of 18 state-of-the-art MLLMs reveals that the strongest model (Gemini-2.5 Pro) achieves only 73.7% overall, with cross-frame understanding tasks falling below 25%.

## Background & Motivation

**Background**: MLLMs have achieved promising performance on static image OCR; however, video OCR presents unique challenges—including motion blur, temporal variation, and visual effects—that lead to significant performance degradation.

**Limitations of Prior Work**:
   - OCR Benchmark: only 25 videos and 1 task type, lacking diversity
   - FG Bench: 1,028 videos but uses mixed automatic and manual annotation with only 6 task types
   - Both benchmarks focus predominantly on text perception, neglecting text-based understanding and reasoning

**Three Core Challenges in Video OCR**:
   - (1) Text appears in diverse forms (foreground, background, danmaku, watermarks, etc.), requiring spatiotemporal visual-text association
   - (2) Key textual information is distributed across multiple frames, necessitating cross-frame aggregation and temporal understanding
   - (3) As task complexity increases, reasoning over recognized text becomes essential

## Method

### Task Taxonomy (10 Categories, 25 Sub-tasks)

1. **Text Recognition**: location-specified recognition, attribute-specified recognition
2. **Visual Text QA**: text-centric QA, translation
3. **Text Grounding**: spatial grounding, temporal grounding
4. **Attribute Recognition**: color recognition, named entity recognition, counting
5. **Change Detection & Tracking**: change detection, text tracking
6. **Special Text Parsing**: table/chart/document/mathematical formula/handwriting parsing
7. **Cross-Frame Text Understanding**: scrolling text comprehension, trajectory recognition, disordered assembly
8. **Text-Based Reasoning**: integrating scattered clues, recognizing implicit relations, resolving ambiguity
9. **Text-Based Video Understanding**: subtitle video comprehension, multi-hop needle-in-a-haystack
10. **Robustness Testing**: AIGC video, long video, adversarial video

### Data Construction

**Video Sources** (three pipelines):
- Reconstruction from existing datasets (BOVText, RoadTextVQA, etc.): GPT-4o evaluates visual dynamics and textual semantic quality, followed by filtering
- Manual collection from public platforms (YouTube, Bilibili, Kuaishou)
- AI generation (Wan text-to-video model): 2,000 phrases → scene descriptions → video generation → filtering

**Annotation Pipeline**:
- Manual annotation (not model-generated): 3–4 QA pairs per video → two rounds of expert filtering retaining 1–2 high-quality pairs
- Expert verification: review of ambiguous questions, inaccurate answers, and insufficiently challenging items
- Balanced option distribution + debiasing test

**Debiasing Test**: Without visual input, model accuracy based solely on language priors should approximate chance level (Containment Match 0%, multiple-choice 25.1%), verifying the absence of knowledge leakage and language prior bias.

### Evaluation Protocol
- **Containment Match**: text recognition and handwriting recognition tasks
- **GPT-assisted scoring**: tasks with multiple valid answers such as translation
- **Multiple-choice**: remaining understanding and reasoning tasks

## Key Experimental Results

### Main Results (18 Models)

| Model | Scale | TR | VTQA | TG | AR | CDT | STP | CFTU | TBR | TBVU | RVT | Total |
|-------|-------|-----|------|-----|-----|-----|-----|------|-----|------|-----|-------|
| Gemini-2.5 Pro | - | 83.0 | 91.6 | 64.5 | 74.0 | 70.0 | 84.4 | 48.7 | 74.0 | 56.5 | 72.0 | **73.7** |
| GPT-4o | - | 83.3 | 81.6 | 60.5 | 74.7 | 51.5 | 68.0 | 30.7 | 60.7 | 59.0 | 75.3 | 66.4 |
| Qwen2.5-VL | 72B | 80.7 | 80.0 | 65.0 | 74.0 | 56.5 | 79.6 | 26.7 | 74.7 | 57.0 | 78.7 | 69.0 |
| InternVL3 | 78B | 70.0 | 77.6 | 67.5 | 76.0 | 65.5 | 71.6 | 24.7 | 77.3 | 57.0 | 75.3 | 67.2 |
| InternVL3 | 8B | 61.3 | 72.0 | 60.0 | 69.3 | 56.5 | 62.4 | 23.3 | 57.3 | 55.0 | 71.3 | 59.8 |
| LLaVA-OneVision | 7B | 42.0 | 50.0 | 49.0 | 54.0 | 41.0 | 46.4 | 20.0 | 45.3 | 52.0 | 60.0 | 46.0 |

### Fine-Grained Task Analysis (Top-5 Models)

| Task | Gemini-2.5 Pro | Qwen2.5-VL 72B | InternVL3 78B | GPT-4o |
|------|---------------|----------------|---------------|--------|
| Trajectory Recognition | **0.0%** | 0.0% | 0.0% | 0.0% |
| Disordered Assembly | 76.0% | 16.0% | 4.0% | 30.0% |
| Multi-hop Needle-in-a-Haystack | 27.0% | 18.0% | 18.0% | 25.0% |
| Subtitle Video Comprehension | 86.0% | 96.0% | 96.0% | 93.0% |
| Translation | 84.0% | 66.0% | 68.0% | 70.0% |

### Key Findings

1. **Cross-frame understanding is the most critical bottleneck**: most models score below 25% on Cross-Frame Text Understanding; all Top-5 models achieve **0%** on **trajectory recognition**
2. **Resolution and frame count are critical**: increasing both consistently improves performance, though some models degrade when frame count increases from 32 to 64 due to attention dispersion
3. **Token compression is unsuitable for OCR**: compression-based approaches such as VideoChat-Flash and Slow-fast MLLM perform poorly on OCR tasks
4. **Severe language prior bias**: models tend to "correct" misspellings to semantically plausible words (e.g., "throuh" → "through") rather than faithfully recognizing visual content
5. **Large gap between single-frame and cross-frame tasks**: subtitle comprehension (single-frame information) exceeds 90%, while multi-hop needle-in-a-haystack (cross-frame aggregation) falls below 30%, indicating that models rely on sparse frames rather than genuinely integrating temporal information
6. **Pronounced scaling effects**: Qwen2.5-VL 7B→72B yields 10%+ improvement; InternVL3 8B→78B yields 7%+ improvement

## Highlights & Insights
- ⭐⭐⭐⭐ **Comprehensive task design**: 25 sub-tasks covering the full pipeline from perception → understanding → reasoning, including innovative dimensions such as cross-frame understanding and robustness testing
- ⭐⭐⭐⭐ **Rigorous debiasing design**: debiasing tests, balanced option distribution, and multi-round expert review eliminate language priors and knowledge leakage
- ⭐⭐⭐⭐ **Actionable findings**: results such as 0% trajectory recognition, language prior bias, and token compression deficiencies directly inform directions for model improvement
- ⭐⭐⭐ **Fully manual annotation**: distinguishes this benchmark from those using hybrid annotation, yielding better-controlled quality

## Limitations & Future Work
1. The total of 2,000 QA pairs leaves some sub-categories with limited samples (e.g., approximately 50 for trajectory recognition), potentially causing score volatility
2. Coverage is primarily bilingual (Chinese and English), with no inclusion of additional languages
3. Among the three difficulty tiers (easy/medium/hard), frontier models perform well on easy and medium instances, necessitating continuous supplementation of high-difficulty samples
4. The benchmark does not evaluate models after video OCR fine-tuning—whether it is suitable as a training objective remains unclear
5. Adversarial video testing employs only an all-black frame insertion strategy, limiting the diversity of adversarial forms

## Rating
⭐⭐⭐⭐ A much-needed comprehensive evaluation benchmark for the video OCR domain. The task design spans rich dimensions, annotation quality is high, and debiasing is rigorously handled. The cross-frame understanding bottleneck and language prior bias identified by this benchmark directly guide MLLM optimization. The benchmark demonstrates strong discriminability and challenge.

## Related Work & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[ICCV 2025\] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs](../../ICCV2025/multimodal_vlm/enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->
