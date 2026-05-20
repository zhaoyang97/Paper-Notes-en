---
title: >-
  [Paper Note] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs
description: >-
  [AAAI 2026][Time Series][Generative AI evaluation] This paper presents GAICo (Generative AI Comparator), a deployed, extensible, open-source Python library that provides a unified reference-based evaluation framework for…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Generative AI evaluation"
  - "multimodal comparison"
  - "evaluation framework"
  - "reproducibility"
  - "compound AI systems"
date: 2026-05-08
content_hash: 8f1a9dd78ce60aa0
---

# GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs

**Conference**: AAAI 2026
**arXiv**: [2508.16753](https://arxiv.org/abs/2508.16753)  
**Code**: [github.com/ai4society/GenAIResultsComparator](https://github.com/ai4society/GenAIResultsComparator)  
**Area**: Time Series
**Keywords**: Generative AI evaluation, multimodal comparison, evaluation framework, reproducibility, compound AI systems

## TL;DR

This paper presents GAICo (Generative AI Comparator), a deployed, extensible, open-source Python library that provides a unified reference-based evaluation framework for text, structured data (planning sequences, time series), and multimedia (images, audio), supporting multi-model comparison, visualization, and report generation.

## Background & Motivation

The rapid advancement of generative AI (GenAI) has permeated various high-stakes domains, yet standardization of evaluation methodologies has lagged severely behind, facing the following challenges:

**Fragmented Evaluation**: Developers typically write ad-hoc scripts for evaluation, requiring code rewrites for each change in data modality or metric, resulting in non-reproducible and non-comparable assessments.

**Standard Metrics Ill-Suited for Structured Outputs**: Traditional NLP metrics (BLEU, ROUGE, etc.) cannot effectively evaluate structured outputs such as AI planning sequences or time series forecasts.

**Multimodal Evaluation Demands**: Modern compound AI systems (e.g., AI travel assistants) simultaneously produce text, image, and audio outputs, yet unified tools for cross-modal comparison are lacking.

**Difficulty in Failure Attribution**: In multi-component compound systems, it is difficult to distinguish whether poor performance originates from the orchestrating LLM or downstream specialized models.

**Core Pain Point**: Taking an AI travel assistant as an example (Figure 2), a system comprising an orchestrating LLM (generating itinerary JSON), an image generation model, and an audio generation model requires separately written scripts for JSON parsing, text analysis, planning validation, and multimedia comparison when evaluating different component combinations — a process that is slow, error-prone, and difficult to trace back to the root cause.

## Method

### Overall Architecture

GAICo's architecture is built upon three core components (Figure 1):

1. **BaseMetric Abstract Class**: A unified foundation for all metrics, enforcing the `calculate(generated_texts, reference_texts)` interface.
2. **Comprehensive Metric Library**: A collection of metrics spanning text, structured data, and multimedia.
3. **Experiment Class**: A high-level API encapsulating the end-to-end pipeline from multi-model comparison to visualization and reporting.

Workflow: Multimodal AI model outputs → GAICo computes pairwise similarity scores $s_{kl}$ → generates raw data reports, visualizations, and pass/fail assessments (based on threshold $\delta$).

### Key Designs

#### 1. **Extensible BaseMetric Architecture**

The core design of GAICo is an object-oriented BaseMetric abstract class:
- Every metric, regardless of data modality, implements a unified `calculate()` method.
- Multiple input formats (single items, lists, NumPy arrays) are handled transparently for efficient batch processing.
- **Extension is minimal**: developers need only subclass BaseMetric and implement `calculate()`, and the new metric is immediately integrated into the GAICo ecosystem.

This design guarantees consistency: **once a metric is selected, its application and reporting remain uniform across all scenarios**.

#### 2. **Comprehensive Multimodal Metric Library**

GAICo integrates metrics across three major categories:

**Text Metrics**:
- N-gram based: BLEU, ROUGE
- Text similarity: Jaccard, Cosine
- Semantic understanding: BERTScore

**Structured Data Metrics**:
- Automated planning: PlanningLCS (Longest Common Subsequence), PlanningJaccard (Jaccard similarity, supporting concurrent action comparison)
- Time series: TimeSeriesDTW (Dynamic Time Warping), TimeSeriesElementDiff (element-wise difference)

**Multimedia Metrics**:
- Image: SSIM (Structural Similarity), PSNR (Peak Signal-to-Noise Ratio), AverageHash (perceptual hashing), HistogramMatch (histogram matching)
- Audio: SNR (Signal-to-Noise Ratio), SpectrogramDistance (spectrogram distance)

#### 3. **Workflow Automation via the Experiment Class**

The Experiment class is a practitioner-oriented high-level API that, through its `compare()` method, automatically handles:
- Multi-model score computation
- Generation of publication-quality figures (bar charts or radar charts, Figure 3)
- Pass/fail determination via quality thresholds
- CSV report export

The entire evaluation workflow — previously requiring hundreds of lines of scripts — can be completed in just a few lines of code.

### Deployment & Engineering Practice

- Published on PyPI; installable via `pip install gaico`
- Optional dependency design (e.g., `pip install 'gaico[bertscore]'`) minimizes installation footprint
- Comprehensive test suite (pytest), continuous integration (CI), and pre-commit hooks
- MkDocs documentation and 17 executable Jupyter Notebook examples

### Loss & Training

GAICo, as an evaluation framework, does not involve model training. Its core design philosophy is to **decouple evaluation from LLM inference**:
- Positioned as a post-hoc comparison framework operating on **already-generated** outputs
- Avoids the API costs, rate limits, and non-determinism of LLM-as-a-judge approaches
- All metric computations are deterministic and reproducible

## Key Experimental Results

### Main Results: AI Travel Assistant Case Study

Three pipelines are constructed for evaluation:
- Pipeline A: OpenAI-centric (GPT-5 + DALL-E + OpenAI TTS)
- Pipeline B: Open-source models (Llama 4 + open-source image/audio models)
- Pipeline C: Google-centric (Gemini 2.5 Pro + Google image/audio models)

| Metric | Pipeline A | Pipeline B | Pipeline C |
|--------|-----------|-----------|-----------|
| ROUGE-L (Text) | 1.000 | 0.190 | 0.222 |
| BERTScore-F1 (Text) | 1.000 | 0.599 | 0.613 |
| LCS (Planning) | 1.000 | 0.095 | 0.137 |
| Jaccard (Planning) | 1.000 | 0.083 | 0.117 |
| DTW (Time Series) | 1.000 | 0.122 | 0.367 |
| SSIM (Image) | 1.000 | 0.276 | 0.347 |
| AverageHash (Image) | 1.000 | 0.646 | 0.766 |
| SNR (Audio) | 1.000 | 0.249 | 0.247 |
| SpectDist (Audio) | 1.000 | 0.261 | 0.260 |

### Ablation Study: Two-Stage Evaluation Strategy

| Evaluation Dimension | Finding |
|---------------------|---------|
| Plan Coherence | Pipeline C > Pipeline B (stronger planning capability) |
| Modality Generation Quality | Image: Pipeline C > Pipeline B (higher structural similarity); Audio: both are far below baseline |
| Failure Attribution | Pipeline B's inferior performance stems from a weaker orchestrating LLM combined with suboptimal specialized models (each can be targeted independently for improvement) |

### Key Findings

1. **Necessity of Multi-Metric Evaluation**: Pipeline B outperforms Pipeline C on image HistogramMatch but underperforms on SSIM and AverageHash — a single metric cannot comprehensively assess output quality.
2. **Value of the Two-Stage Evaluation Strategy**: By separating "plan coherence" and "modality generation quality," the framework enables precise diagnosis of whether the orchestrator or the specialized model is responsible for failures.
3. **Community Validation**: Since its release in June 2025, the package has accumulated over 16,000 downloads on PyPI, demonstrating strong real-world demand.
4. **Efficiency Gains**: An evaluation pipeline that previously required numerous independent scripts is unified into a few lines of code.

## Highlights & Insights

1. **Design philosophy of decoupling evaluation from inference**: Avoids the non-determinism and cost of LLM-as-a-judge approaches, focusing instead on deterministic post-hoc comparison.
2. **Innovation in structured data metrics**: PlanningLCS and PlanningJaccard fill a gap in AI planning evaluation tooling.
3. **Failure attribution for compound systems**: The two-stage evaluation strategy (evaluating the orchestrator first, then the specialized models) addresses a genuine need in industrial practice.
4. **High engineering maturity**: Optional dependencies, CI/CD, comprehensive documentation, and 17 example notebooks represent a complete transition from a utility to a production-ready tool.

## Limitations & Future Work

1. **Reference-based evaluation only**: Metrics for fairness, bias, toxicity, and latency are not yet supported.
2. **Static visualizations**: Only static charts (bar charts, radar charts) are currently generated; interactive dashboards are absent.
3. **Limited structured data support**: General comparison of arbitrarily nested JSON, knowledge graphs, and other complex structures is not supported.
4. **No support for multi-turn dialogue evaluation**: Current metrics primarily target single-turn outputs.
5. Future work may integrate with MLOps platforms such as MLflow to enable dynamic result tracking.

## Related Work & Insights

- **HuggingFace Evaluate**: Offers a broad range of NLP metrics but lacks a unified multimodal framework.
- **Ragas / DeepEval**: End-to-end frameworks but tightly coupled to LLM APIs, introducing non-determinism.
- **scikit-learn**: Provides foundational ML metrics but is not tailored to GenAI outputs.
- Insight: In the fast-iterating landscape of GenAI development, **standardized and reproducible evaluation infrastructure** is more valuable than any single metric.

## Rating

- Novelty: ⭐⭐⭐ — The core contribution is engineering integration rather than algorithmic innovation, though the unified multimodal framework carries practical significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The case study is detailed, but large-scale user studies and in-depth comparisons with similar tools are lacking.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, the case study is illustrative, and the scope and contributions are well-defined.
- Value: ⭐⭐⭐⭐ — As a tool paper, 16K downloads substantiate its practical value; it fills a clear gap in multimodal GenAI evaluation tooling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](../../ICLR2026/time_series/rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](../../ICLR2026/time_series/edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](../../CVPR2026/time_series/a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[AAAI 2026\] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)

</div>

<!-- RELATED:END -->
