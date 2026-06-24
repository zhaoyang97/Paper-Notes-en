---
title: >-
  [Paper Note] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects
description: >-
  [ACL 2026][Social Computing][Dialectal Bias] This paper constructs DIA-HARM, the first benchmark to evaluate the robustness of misinformation detection across 50 English dialects. It reveals that human-written dialectal content leads to a performance drop of 1.4-3.6% F1, while fine-tuned Transformers significantly outperform zero-shot LLMs (96.6% vs 78.3%). Furthermore, some models exhibit catastrophic degradation exceeding 33% on mixed content.
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Dialectal Bias"
  - "Misinformation Detection"
  - "Robustness Evaluation"
  - "English Dialects"
  - "Detection Fairness"
date: 2026-05-08
content_hash: baff0b0680ea4d0a
---

# DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects

**Conference**: ACL 2026  
**arXiv**: [2604.05318](https://arxiv.org/abs/2604.05318)  
**Code**: [https://github.com/jsl5710/dia-harm](https://github.com/jsl5710/dia-harm)  
**Area**: Content Safety / Dialectal Robustness  
**Keywords**: Dialectal Bias, Misinformation Detection, Robustness Evaluation, English Dialects, Detection Fairness

## TL;DR

This paper constructs DIA-HARM, the first benchmark to evaluate the robustness of misinformation detection across 50 English dialects. It reveals that human-written dialectal content leads to a performance drop of 1.4-3.6% F1, while fine-tuned Transformers significantly outperform zero-shot LLMs (96.6% vs 78.3%). Furthermore, some models exhibit catastrophic degradation exceeding 33% on mixed content.

## Background & Motivation

**Background**: Harmful content detectors (particularly misinformation classifiers) are primarily developed and evaluated on Standard American English (SAE). Their robustness against various dialectal versions remains largely unexplored.

**Limitations of Prior Work**: (1) Hundreds of millions of English speakers globally use non-SAE dialects, yet detection systems have not been validated on these variations; (2) Dialectal transformation alters morphosyntactic structures while preserving misinformation semantics—if detectors rely on surface patterns rather than deep semantic understanding, dialectal content might bypass detection; (3) Detection failures could systematically leave dialect speakers with less protection.

**Key Challenge**: Misinformation detectors should base their judgments on content veracity (semantics). However, if they rely on surface linguistic patterns, dialectal variants (which change surface forms but retain semantics) will expose this vulnerability.

**Goal**: (1) Construct an evaluation benchmark for misinformation detection across 50 English dialects; (2) Evaluate the robustness of 16 detection models on dialectal variants; (3) Identify patterns of cross-dialectal transfer.

**Key Insight**: By using Multi-VALUE, a linguistic rule-based dialect transformation tool, standard misinformation datasets are converted into 50 dialectal variants to build the D3 corpus (195K samples), enabling a systematic evaluation of detection models.

**Core Idea**: Dialectal variants serve as natural perturbations for misinformation detectors—changing linguistic form without altering content veracity—which can reveal whether detectors understand semantics or merely rely on surface patterns.

## Method

### Overall Architecture

DIA-HARM treats "dialect" as a natural perturbation to misinformation detectors: it preserves content veracity while only altering the linguistic surface form, thereby exposing whether the detector is understanding semantics or capturing surface patterns. The pipeline consists of three steps: first, using rule-based dialect transformation to expand SAE misinformation data into 50 dialects to obtain the D3 corpus (195K samples); second, evaluating 16 detection models on these dialectal variants; and finally, analyzing performance transfer laws across 2,450 dialect pairs.

### Key Designs

**1. Rule-based Dialect Transformation (Multi-VALUE): Creating dialects using linguistic rules rather than random noise**

To evaluate dialectal robustness, "linguistically grounded" dialect data is required; otherwise, conclusions could be contaminated by meaningless noise. Instead of synonym replacement or random perturbations, this work utilizes the Multi-VALUE tool to rewrite text according to morphosyntactic rules—tense markers, pronoun systems, and article usage are all converted according to the grammar of real dialects. This expands SAE text to cover 50 English dialects across the US, UK, Africa, Caribbean, and Asia-Pacific regions. Rule-based methods ensure each variant is a valid dialectal sentence rather than gibberish, allowing performance changes to be cleanly attributed to the "dialect" variable.

**2. Multi-type Detection Model Evaluation: Comparing the vulnerability of fine-tuned models and zero-shot LLMs**

Different detection paradigms may fail in different ways when facing dialects. Fine-tuned models are prone to overfitting to SAE surface writing habits, while zero-shot LLMs might generalize better due to large-scale pre-training. Consequently, this paper evaluates 16 models simultaneously, spanning fine-tuned Transformers (RoBERTa, mDeBERTa, etc.) and zero-shot LLMs (GPT-4, Llama, etc.), and further distinguishes between "human-written" and "AI-generated" dialectal content. The latter is particularly critical: if AI-generated dialects do not suffer performance drops while human-written ones do, it suggests that detectors rely on surface cues remaining in human writing rather than the semantics themselves.

**3. Cross-dialectal Transfer Analysis (2,450 Dialect Pairs): Investigating how detection capability propagates between dialects**

Knowing only the "overall performance drop" is insufficient. Instructionally meaningful insights come from knowing which dialects can transfer to one another and which cause immediate failure. This paper exhaustively analyzes all $50 \times 49 = 2{,}450$ dialect pairs to characterize the transfer map—for example, whether multilingual models (mDeBERTa) are more stable than monolingual models (RoBERTa) across dialects. This transfer map directly serves practical applications by informing deployers on model selection and which dialects should be added to training sets.

### Loss & Training

DIA-HARM is an evaluation benchmark and does not introduce new training objectives. The fine-tuned models under evaluation are trained uniformly on SAE data and then subjected to robustness tests on 50 dialectal variants.

## Key Experimental Results

### Main Results

**Detection Performance Comparison (Best-case F1 %)**

| Model Type | SAE | Dialect (Avg.) | Worst Degradation |
|----------|-----|-----------|---------|
| Fine-tuned Transformer (Best) | 96.6 | 93-95 | -3.6 |
| Zero-shot LLM (Best) | 78.3 | ~76 | -2.4 |
| Monolingual Model (RoBERTa) | High | Severe Degradation | >33% |
| Multilingual Model (mDeBERTa) | 97.2 | 97.2 | Minimal |

### Ablation Study

| Content Type | Dialectal Impact | Description |
|----------|---------|------|
| Human-written | -1.4~3.6% F1 | Dialects significantly impact detection |
| AI-generated | Stable | AI-generated content is unaffected by dialects |
| Mixed Content | >33% Degradation in some models | The most dangerous scenario |

### Key Findings

- Fine-tuned Transformers significantly outperform zero-shot LLMs (96.6% vs 78.3%), but their dialectal vulnerability varies.
- Multilingual models (mDeBERTa: 97.2% average F1) generalize best across dialectal variants.
- Monolingual models (RoBERTa) can suffer catastrophic failure on dialectal inputs (>33% degradation).
- AI-generated dialectal content does not affect detection performance, but human-written dialectal content leads to significant degradation—indicating that detectors partially rely on surface patterns in human writing.
- Certain dialect pairs (e.g., Jamaican Creole) lead to particularly severe detection degradation.

## Highlights & Insights

- First systematic evaluation of the robustness of misinformation detectors across 50 English dialects, with unprecedented scale and coverage.
- The discovery that "AI-generated dialects are stable while human-written dialects degrade" profoundly reveals the dependency patterns of detectors.
- The dialectal robustness of multilingual pre-trained models provides clear guidance for model selection.

## Limitations & Future Work

- Rule-based dialect transformation may not fully capture the complexity of real-world dialects.
- The focus is limited to misinformation detection; other harmful content types (e.g., hate speech) remain to be explored.
- 50 dialects still do not cover all English variants.
- Dialect-aware training as a defense strategy was not investigated.

## Related Work & Insights

- **vs Hate Speech Dialect Studies (Sap et al. 2019)**: Previous work addressed dialectal bias in hate speech detection; DIA-HARM provides the first systematic evaluation for misinformation detection.
- **vs Multi-VALUE**: Multi-VALUE provides dialect transformation tools; DIA-HARM applies them to safety detection evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First robustness benchmark for misinformation detection across 50 dialects.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 models, 50 dialects, 195K samples, and 2450 dialect pair analyses.
- Writing Quality: ⭐⭐⭐⭐ Problem is significant, and analysis is comprehensive.
- Value: ⭐⭐⭐⭐⭐ Reveals critical flaws in detection fairness, with direct impact on safety system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Content Fuzzing for Escaping Information Cocoons on Social Media](content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)
- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](../../ICLR2026/social_computing/socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [\[NeurIPS 2025\] OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](../../NeurIPS2025/social_computing/os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](../../ACL2025/social_computing/taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)
- [\[ACL 2026\] Is this chart lying to me? Automating the detection of misleading visualizations](is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)

</div>

<!-- RELATED:END -->
