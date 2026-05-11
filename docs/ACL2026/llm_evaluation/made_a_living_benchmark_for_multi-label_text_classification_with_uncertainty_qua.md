---
title: >-
  [Paper Note] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification
description: >-
  [ACL 2026][LLM Evaluation][Multi-label classification] This paper introduces MADE—a "living" multi-label text classification benchmark built on FDA medical device adverse event reports, featuring 1…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Multi-label classification"
  - "uncertainty quantification"
  - "medical devices"
  - "living benchmark"
  - "long-tail distribution"
date: 2026-05-08
content_hash: a0099335ce9c08bb
---

# MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification

**Conference**: ACL 2026
**arXiv**: [2604.15203](https://arxiv.org/abs/2604.15203)
**Code**: [https://hhi.fraunhofer.de/aml-demonstrator/made-benchmark](https://hhi.fraunhofer.de/aml-demonstrator/made-benchmark)
**Area**: LLM Evaluation
**Keywords**: Multi-label classification, uncertainty quantification, medical devices, living benchmark, long-tail distribution

## TL;DR

This paper introduces MADE—a "living" multi-label text classification benchmark built on FDA medical device adverse event reports, featuring 1,154 hierarchical labels and strict temporal splits. It systematically evaluates 20+ encoder/decoder models across discriminative fine-tuning, generative fine-tuning, and few-shot prompting paradigms, assessing both predictive performance and uncertainty quantification (UQ) capabilities. Key findings reveal critical trade-offs: small discriminatively fine-tuned decoders achieve the best head-to-tail accuracy; generative fine-tuning yields the most reliable UQ; and large reasoning models improve rare-label performance but exhibit surprisingly weak UQ.

## Background & Motivation

**Background**: Multi-label text classification (MLTC) is a core task in healthcare—covering patient triage, clinical coding, and incident reporting—requiring selection of multiple labels from large label sets. Existing benchmarks (e.g., MIMIC-III, EUR-LEX) are approaching saturation and may be contaminated by LLM pretraining data.

**Limitations of Prior Work**: (1) Existing MLTC benchmarks are static, making them susceptible to data contamination that inflates zero-/few-shot performance; (2) real-world MLTC data exhibits severe intra- and inter-class imbalance, with a few frequent classes dominating samples while safety-critical classes reside in the long tail; (3) in high-stakes domains such as healthcare, models must not only achieve strong predictive performance but also provide reliable UQ to support human oversight—yet UQ for MLTC remains almost entirely unstudied.

**Key Challenge**: Practitioners face unresolved questions: which model architecture (encoder vs. decoder) is preferable? Which learning paradigm (fine-tuning vs. in-context learning) offers the best trade-off between frequent and rare classes? How reliable are model predictions? No unified, contamination-free benchmark exists to answer these questions systematically.

**Goal**: (1) Construct a continuously updated, contamination-free MLTC benchmark; (2) establish comprehensive baselines spanning 20+ models; (3) systematically evaluate diverse UQ methods on MLTC.

**Key Insight**: FDA medical device adverse event reports are published on a rolling basis, serving as a continuously updated data source. Strict temporal splits ensure that test data cannot leak into the pretraining of future models.

**Core Idea**: Build a "living" benchmark—as the FDA releases new reports, future models can always be evaluated on data generated after their training cutoff, fundamentally preventing contamination.

## Method

### Overall Architecture

The MADE benchmark comprises three main components: (1) a data pipeline that extracts incident descriptions and IMDRF hierarchical labels from FDA adverse event reports, producing train/validation/test splits via deduplication, downsampling, and temporal partitioning; (2) model baselines covering three paradigms—discriminative fine-tuning (encoder/decoder + classification head), generative fine-tuning (decoder generating label tokens), and few-shot prompting (instruction/thinking models); and (3) UQ evaluation comparing four categories of methods: information-level (entropy, perplexity), consistency-level (graph Laplacian eigenvalues), combined-level, and self-reported uncertainty.

### Key Designs

1. **Living Benchmark Data Construction**:

    - Function: Provide continuously contamination-free MLTC evaluation data
    - Mechanism: Incident descriptions and labels are extracted from FDA adverse event reports spanning 2015–2025. Product problem and patient problem labels for each report are mapped to IMDRF hierarchical codes (3 levels) and propagated upward to all ancestor codes. The training set covers 2015–2023 (298,825 samples), the validation set covers the first half of 2024 (71,271 samples), and the test set covers July 2024–June 2025 (118,177 samples). The final label set contains 1,154 labels, with an average of 8.79 labels per sample and a pronounced long-tail distribution.
    - Design Motivation: As the FDA publishes new reports quarterly, future models can be evaluated on data post-dating their training cutoff, fundamentally eliminating data contamination.

2. **Multi-Paradigm Model Baselines**:

    - Function: Comprehensively compare MLTC performance across different architectures and learning paradigms
    - Mechanism: (a) Discriminative fine-tuning—classification heads are added to Llama 3.2-1B/3B, 3.1-8B, and Ettin 150M/400M/1B, trained with hierarchical BCE loss; (b) Generative fine-tuning—Llama and Ettin decoders generate label tokens, comparing full-parameter and LoRA fine-tuning; (c) Few-shot prompting—10-shot kNN-retrieved prompts are applied to 10+ models including Llama, DeepSeek-R1, Qwen3, and GPT-4.1/5. Labels are partitioned into four frequency tiers based on training set frequency: head (>1%), medium (0.1–1%), tail (0.01–0.1%), and extreme tail (<0.01%).
    - Design Motivation: Practitioners require fair, controlled comparisons to make informed decisions about model size, training cost, and performance trade-offs.

3. **Systematic UQ Evaluation**:

    - Function: Assess the quality of uncertainty estimates across different models and paradigms
    - Mechanism: Per-label entropy is used for discriminative models. For generative models, four categories of UQ are evaluated: information-level $U_{\text{info}}$ (entropy, improbability, avg-log-prob, perplexity), consistency-level $U_{\text{cons}}$ (sum of graph Laplacian eigenvalues across multiple stochastic samples), combined-level $U_{\text{combined}} = U_{\text{info}} \times U_{\text{cons}}$, and self-reported $U_{\text{self}}$ (prompting models to output confidence scores). UQ quality is evaluated using Prediction Rejection Ratio (PRR), Spearman $\rho$, and positive-class ECE$_+$.
    - Design Motivation: In high-risk medical settings, uncertain cases must be routed to human reviewers; UQ quality directly determines system safety.

### Loss & Training

Discriminative fine-tuning employs a hierarchical binary cross-entropy loss, computed separately at each hierarchy level and summed. Training uses AdamW with cosine learning rate scheduling, a batch size of 512, and 20 epochs. Per-label classification thresholds are independently selected on the validation set to maximize F1. Generative fine-tuning uses the standard autoregressive language modeling loss, trained for 4 epochs, supporting both full-parameter and LoRA fine-tuning.

## Key Experimental Results

### Main Results

**Predictive Performance and UQ Quality Across Paradigms (truncated test set, $n=10,288$)**

| Paradigm / Model | Macro F1 | Head F1 | Tail F1 | ET F1 | PRR↑ | ρ↓ |
|---|---|---|---|---|---|---|
| Discriminative Llama-3.1-8B | **0.54** | **0.74** | **0.53** | 0.12 | 0.47 | -0.40 |
| Generative Llama-3.1-70B | 0.53 | 0.73 | 0.51 | 0.16 | 0.55 | -0.27 |
| Generative Llama-3.2-3B | 0.48 | 0.67 | 0.46 | 0.12 | **0.60** | **-0.46** |
| Prompting Qwen3-235B-Think | 0.49 | 0.62 | 0.48 | 0.33 | 0.34 | -0.09 |
| Prompting GPT-5 | 0.54 | 0.68 | 0.53 | **0.34** | N/A | N/A |
| Prompting DeepSeek-R1 | 0.48 | 0.62 | 0.47 | 0.30 | 0.24 | -0.09 |

### Ablation Study

**UQ Method Comparison (Generative Fine-tuning vs. Prompting)**

| UQ Metric | Generative FT PRR | Instruct PRR | Thinking PRR |
|---|---|---|---|
| Avg. Log-Prob | 0.54±0.05 | 0.37±0.25 | 0.18±0.12 |
| Entropy | **0.58±0.03** | **0.45±0.15** | 0.19±0.12 |
| Improbability | 0.54±0.05 | 0.43±0.15 | 0.17±0.12 |
| Perplexity | 0.54±0.06 | 0.37±0.25 | 0.18±0.11 |

### Key Findings

- Discriminative fine-tuning consistently outperforms generative fine-tuning of equivalent size on head-to-tail accuracy (Wilcoxon test $p \leq 0.05$), achieving the best overall F1 with only 8B parameters.
- Generative fine-tuning yields superior UQ—Llama-3.2-3B (generative) achieves the best PRR (0.60) and Spearman $\rho$ (-0.46).
- Reasoning models (GPT-5, Qwen3-235B-Think) excel on extreme tail classes (F1=0.34) but exhibit surprisingly weak UQ (PRR of only 0.21±0.10), while consistently underperforming the best fine-tuned models on head classes.
- Self-reported confidence is not a reliable proxy for uncertainty, showing low correlation with actual error rates.
- Entropy, as a $U_{\text{info}}$ metric, is the best-performing choice for both generative fine-tuned and instruct models.

## Highlights & Insights

- The "living benchmark" concept precisely addresses the fundamental problem of benchmark contamination in the LLM era—leveraging continuously published government data streams to construct a test set that never becomes stale.
- The paper reveals a counterintuitive trade-off between predictive performance and UQ quality: reasoning models excel on rare classes yet exhibit the worst UQ, implying that their "high performance" may be untrustworthy in high-stakes settings.
- Discriminative fine-tuning with only 8B parameters matches or surpasses GPT-5 on overall performance, offering practitioners a highly cost-effective solution.

## Limitations & Future Work

- Label consistency in FDA annotations has not been validated through a formal inter-annotator agreement study, and labels may contain noise.
- Self-reported UQ is evaluated only with simple prompting strategies; more sophisticated calibration prompts may improve results.
- The test set is limited to 10,288 samples to control inference costs, limiting the statistical power of evaluation for extreme tail labels.
- The potential gains from multimodal inputs (e.g., device images) have not been assessed.

## Related Work & Insights

- **vs. MIMIC-III ICD Coding**: MIMIC has served as the standard clinical note ICD coding benchmark for over a decade, but carries a substantial contamination risk; MADE avoids this through continuous temporal splitting.
- **vs. EUR-LEX**: EUR-LEX is a legal document MLTC benchmark with a smaller label space and a different domain; MADE's 1,154 labels and 3-level hierarchy pose greater challenges.
- **vs. Ettin (Weller et al. 2025)**: Ettin provides matched encoder–decoder model comparisons but was not evaluated on MLTC; this paper fills that gap.

## Rating

- Novelty: ⭐⭐⭐⭐ The "living benchmark" concept is both novel and practical; the systematic UQ evaluation on MLTC fills a clear gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison of 20+ models across 4 paradigms and multiple UQ methods, with rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear conclusions, though the high information density requires consulting the appendix for some details.
- Value: ⭐⭐⭐⭐⭐ Provides a practical guide to model selection and UQ methods for high-stakes MLTC applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](../../NeurIPS2025/llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[NeurIPS 2025\] A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification](../../NeurIPS2025/llm_evaluation/a_standardized_benchmark_for_multilabel_antimicrobial_peptide_classification.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)

</div>

<!-- RELATED:END -->
