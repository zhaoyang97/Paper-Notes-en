---
title: >-
  [Paper Note] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects
description: >-
  [ACL 2026][Social Computing][Dialectal bias] This paper constructs DIA-HARM, the first benchmark to evaluate the robustness of disinformation detection across 50 English dialects. It reveals that human-authored dialectal…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Dialectal bias"
  - "disinformation detection"
  - "robustness evaluation"
  - "English dialects"
  - "detection fairness"
date: 2026-05-08
content_hash: 1fa75ef08dc88b1d
---

# DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects

**Conference**: ACL 2026  
**arXiv**: [2604.05318](https://arxiv.org/abs/2604.05318)  
**Code**: [https://github.com/jsl5710/dia-harm](https://github.com/jsl5710/dia-harm)  
**Area**: Content Safety / Dialectal Robustness  
**Keywords**: Dialectal bias, disinformation detection, robustness evaluation, English dialects, detection fairness

## TL;DR

This paper constructs DIA-HARM, the first benchmark to evaluate the robustness of disinformation detection across 50 English dialects. It reveals that human-authored dialectal content leads to a performance drop of 1.4-3.6% in F1 score. Fine-tuned Transformers significantly outperform zero-shot LLMs (96.6% vs 78.3%), and some models exhibit a catastrophic degradation of over 33% on mixed content.

## Background & Motivation

**Background**: Harmful content detectors (specifically disinformation classifiers) are primarily developed and evaluated on Standard American English (SAE). Their robustness to dialectal variants remains largely unexplored.

**Limitations of Prior Work**: (1) Hundreds of millions of English speakers worldwide use non-SAE dialects, yet detection systems have not been validated on these varieties. (2) Dialectal shifts alter morphosyntactic structures while preserving disinformation semantics; if detectors rely on surface patterns rather than deep semantic understanding, dialectal content may bypass detection. (3) Detection failures could systematically leave dialect users with less protection.

**Key Challenge**: Disinformation detectors should make judgments based on content veracity (semantics), but if they rely on surface linguistic patterns, dialectal variants (which change surface form while preserving semantics) expose this vulnerability.

**Goal**: (1) Construct an evaluation benchmark for disinformation detection across 50 English dialects; (2) Evaluate the robustness of 16 detection models on dialectal variants; (3) Identify patterns of cross-dialectal transfer.

**Key Insight**: Using Multi-VALUE, a linguistically rule-based dialect conversion tool, standard disinformation datasets are transformed into 50 dialectal variants to build the D3 corpus (195K samples) for systematic evaluation of detection models.

**Core Idea**: Dialectal variants serve as natural perturbations for disinformation detectors—altering linguistic form without changing content veracity—revealing whether detectors understand semantics or merely rely on surface patterns.

## Method

### Overall Architecture

DIA-HARM consists of three components: (1) D3 corpus construction, using rule-based dialect conversion to transform SAE disinformation data into 50 dialects; (2) Detection model evaluation, assessing 16 models (fine-tuned Transformers + zero-shot LLMs) on dialectal variants; (3) Cross-dialectal transfer analysis, analyzing performance transfer patterns across 2,450 dialect pairs.

### Key Designs

1. **Rule-based Dialect Conversion (Multi-VALUE)**:
    - **Function**: Generate linguistically valid dialectal variants.
    - **Mechanism**: Use the Multi-VALUE tool to apply morphosyntactic rule transformations (e.g., aspect markers, pronoun systems, article usage) to convert SAE text into 50 English dialects covering the US, UK, Africa, Caribbean, and Asia-Pacific.
    - **Design Motivation**: Rule-based conversion ensures linguistic validity rather than simple noise injection or semantic-equivalent substitution.

2. **Multi-type Detection Model Evaluation**:
    - **Function**: Comprehensively evaluate the dialectal robustness of different detection paradigms.
    - **Mechanism**: Evaluate 16 models including fine-tuned Transformers (RoBERTa, mDeBERTa, etc.) and zero-shot LLMs (GPT-4, Llama, etc.), distinguishing between human-authored and AI-generated dialectal content.
    - **Design Motivation**: Different model types may have distinct vulnerability patterns; fine-tuned models might overfit SAE, while zero-shot LLMs might generalize better.

3. **Cross-Dialectal Transfer Analysis (2,450 Dialect Pairs)**:
    - **Function**: Identify performance transfer regularities between dialects.
    - **Mechanism**: Analyze detection performance shifts across all dialect pairs to identify which dialects transfer well and which lead to degradation. Determine if multilingual models (mDeBERTa) are more robust than monolingual ones.
    - **Design Motivation**: Understanding cross-dialectal transfer helps guide model selection and data augmentation strategies.

### Loss & Training

DIA-HARM is an evaluation benchmark and does not involve training new models. Fine-tuned models are trained on SAE data and then evaluated on dialectal variants.

## Key Experimental Results

### Main Results

**Detection Performance Comparison (Best-case F1 %)**

| Model Type | SAE | Dialect (Avg) | Worst Degradation |
|----------|-----|-----------|---------|
| Fine-tuned Transformer (Best) | 96.6 | 93-95 | -3.6 |
| Zero-shot LLM (Best) | 78.3 | ~76 | -2.4 |
| Monolingual Model (RoBERTa) | High | Severe Degradation | >33% |
| Multilingual Model (mDeBERTa) | 97.2 | 97.2 | Minimal |

### Ablation Study

| Content Type | Dialectal Impact | Description |
|----------|---------|------|
| Human-authored | -1.4~3.6% F1 | Dialects significantly affect detection |
| AI-generated | Stable | AI-generated content is unaffected by dialects |
| Mixed Content | >33% degradation for some models | Most dangerous scenario |

### Key Findings

- Fine-tuned Transformers significantly outperform zero-shot LLMs (96.6% vs 78.3%), but dialectal vulnerability varies.
- Multilingual models (mDeBERTa: 97.2% average F1) generalize best across dialectal variants.
- Monolingual models (RoBERTa) can suffer catastrophic failure (>33% degradation) on dialectal inputs.
- AI-generated dialectal content does not affect detection performance, but human-authored dialectal content causes nodes to degrade—indicating that detectors partially rely on surface patterns in human writing.
- Certain dialect pairs (e.g., Jamaican Patois) result in particularly severe detection degradation.

## Highlights & Insights

- This is the first systematic evaluation of disinformation detector robustness across 50 English dialects, providing unprecedented scale and coverage.
- The discovery that "AI-generated dialects stay stable, whereas human-authored dialects degrade" reveals the dependency patterns of detectors.
- The dialectal robustness of multilingual pre-trained models provides clear guidance for model selection in content safety tasks.

## Limitations & Future Work

- Rule-based dialect conversion may not fully capture the complexity of real-world dialects.
- The study focuses only on disinformation; other harmful content types (e.g., hate speech) remain to be explored.
- 50 dialects still do not cover all possible English variants.
- Dialect-aware training as a defense strategy was not explored.

## Related Work & Insights

- **vs. Hate Speech Dialectal Research (Sap et al. 2019)**: Previous work focused on dialectal bias in hate speech detection; DIA-HARM provides the first systematic evaluation for disinformation detection.
- **vs. Multi-VALUE**: Multi-VALUE provides dialectal transformation tools; DIA-HARM applies them to safety detection evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First benchmark for disinformation detection robustness across 50 dialects.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 16 models, 50 dialects, 195K samples, and 2,450 dialect pair analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Important problem with comprehensive analysis.
- **Value**: ⭐⭐⭐⭐⭐ Reveals critical flaws in detection fairness, directly impacting the deployment of safety systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Content Fuzzing for Escaping Information Cocoons on Social Media](content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)
- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](../../ICLR2026/social_computing/socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [\[NeurIPS 2025\] OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](../../NeurIPS2025/social_computing/os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)
- [\[ACL 2026\] Is this chart lying to me? Automating the detection of misleading visualizations](is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)

</div>

<!-- RELATED:END -->
