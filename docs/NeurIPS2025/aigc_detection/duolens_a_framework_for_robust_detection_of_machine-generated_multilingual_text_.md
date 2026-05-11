---
title: >-
  [Paper Note] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code
description: >-
  [NeurIPS 2025][AIGC Detection][AI-generated text detection] DuoLens is proposed — an AI-generated content detection framework based on dual-encoder fusion of CodeBERT and CodeBERTa — achieving AUROC of 0.97–0.99 on multi…
tags:
  - "NeurIPS 2025"
  - "AIGC Detection"
  - "AI-generated text detection"
  - "code detection"
  - "multilingual"
  - "SLM"
  - "BERT"
date: 2026-05-08
content_hash: d99d2f274efb6e31
---

# DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code

**Conference**: NeurIPS 2025
**arXiv**: [2510.18904](https://arxiv.org/abs/2510.18904)
**Code**: Available (includes training and evaluation scripts)
**Area**: AIGC Detection
**Keywords**: AI-generated text detection, code detection, multilingual, SLM, BERT

## TL;DR
DuoLens is proposed — an AI-generated content detection framework based on dual-encoder fusion of CodeBERT and CodeBERTa — achieving AUROC of 0.97–0.99 on multilingual text (8 languages) and source code (7 programming languages) at significantly reduced computational cost (8–12× lower latency, 3–5× lower VRAM), substantially outperforming large models such as GPT-4o.

## Background & Motivation

**Background**: The proliferation of LLM-generated content has made detection increasingly urgent. Existing detectors are either computationally expensive (LLM-based methods) or insufficiently accurate (perplexity-based methods), and most are only effective for English.

**Limitations of Prior Work**:
   - LLM-based detection methods (Binoculars, AGENT-X) suffer from slow inference and high resource demands
   - Existing code detection datasets exhibit **language imbalance** (Python-dominated), while multilingual text datasets are **domain-narrow**
   - LLMs used for binary classification may be biased (tending to rate AI-generated text as higher quality)

**Key Challenge**: Binary classification does not require LLM-level model complexity; small language models (SLMs) may outperform LLMs on this task while being substantially more efficient.

**Key Insight**: Fine-tuning encoder-only models (RoBERTa/CodeBERT variants) and fusing complementary representations via a dual-encoder architecture.

**Core Idea**: For the binary classification task of AI-generated content detection, carefully fine-tuned SLMs outperform LLMs in both accuracy and efficiency.

## Method

### Overall Architecture
A dual-encoder architecture in which CodeBERT (pre-trained on NL + code) and CodeBERTa (pre-trained on code only) independently encode the input. The two [CLS] vectors are fused via a learned gating mechanism and fed into a linear classifier for binary classification.

### Key Designs

1. **Balanced Dataset Construction**:

    - Code dataset: 7 programming languages × 12,000 samples per language (50% human-written / 50% AI-generated), totaling 84,000 samples
    - Multilingual text: 54,520 samples across 8 languages, spanning multiple domains including news, social media, and QA
    - **Design Motivation**: Eliminate bias introduced by language and label imbalance

2. **DuoLens Dual-Encoder Fusion**:

    - **Function**: Fuses CodeBERT's semantic/linguistic alignment capability with CodeBERTa's syntactic/structural sensitivity
    - **Mechanism**: The gating mechanism learns to suppress redundant features while amplifying encoder-specific cues
    - **Design Motivation**: The two encoders differ in pre-training data (NL+Code vs. code only), thus capturing complementary signals

3. **Cross-Lingual Evaluation**:

    - **Function**: Tests generalization to unseen languages after fine-tuning on specific languages
    - **Design Motivation**: In real-world deployment, it is impractical to train dedicated models for every language

## Key Experimental Results

### Main Results

| Model | Code AUC | Code F1 | Multilingual AUC | Multilingual F1 |
|-------|----------|---------|-----------------|----------------|
| GPT-4o (few-shot) | 0.535 | 0.414 | 0.573 | 0.490 |
| CodeBERT (fine-tuned) | 0.980 | 0.930 | — | — |
| DuoLens (fine-tuned) | **0.985** | **0.937** | **0.975** | **0.911** |
| XLM-R-large (fine-tuned) | — | — | 0.974 | **0.924** |

### Ablation Study / Robustness

| Attack Type | Performance Retained |
|------------|---------------------|
| Paraphrase attack | ≥92% clean AUROC |
| Back-translation attack | ≥92% clean AUROC |
| Code formatting / renaming | ≥92% clean AUROC |

### Key Findings
- GPT-4o **substantially underperforms** fine-tuned SLMs on binary classification (AUC 0.535 vs. 0.985)
- Even **without fine-tuning**, CodeBERT achieves AUC 0.953, far exceeding GPT-4o
- DuoLens achieves 8–12× lower latency and 3–5× lower VRAM usage compared to LLM-based methods
- Performance is maintained at ≥92% under adversarial transformations

## Highlights & Insights
- **Counter-intuitive finding — SLM > LLM**: For binary classification, the inductive bias of encoder models is naturally well-suited to the task, while the generative capability of LLMs proves superfluous. This serves as an important reminder for the AIGC detection community.
- **Dataset engineering**: Carefully balancing language and label distributions is critical — the imbalance in existing datasets is a primary cause of poor detector generalization.

## Limitations & Future Work
- BERT-based models are limited to 512 tokens; long documents require chunking
- Encoder-only models cannot provide sentence-level detection explanations
- The multilingual text dataset exhibits language imbalance
- Among closed-source LLM baselines, only GPT-4o was evaluated

## Related Work & Insights
- **vs. AGENT-X / Binoculars**: LLM-based methods incur high computational cost without necessarily achieving superior accuracy
- **vs. GPT-Sentinel**: Also based on RoBERTa fine-tuning but with poor generalization; DuoLens improves upon this through balanced datasets and dual-encoder fusion
- Practical deployment of AI-generated content detectors should prioritize SLM-based solutions

## Rating
- **Novelty**: ⭐⭐⭐ Dual-encoder fusion is not novel in NLP, but its systematic application to the AIGC detection setting offers practical value
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across multiple natural languages, programming languages, cross-lingual transfer, and adversarial evaluation
- **Writing Quality**: ⭐⭐⭐⭐ Clear and well-organized
- **Value**: ⭐⭐⭐⭐ Directly informative for practical AIGC detection deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CLARC: C/C++ Benchmark for Robust Code Search](../../ICLR2026/aigc_detection/clarc_cc_benchmark_for_robust_code_search.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](../../ACL2026/aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)
- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)
- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)

</div>

<!-- RELATED:END -->
