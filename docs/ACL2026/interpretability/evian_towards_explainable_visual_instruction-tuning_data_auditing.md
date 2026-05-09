---
title: >-
  [Paper Note] Evian: Towards Explainable Visual Instruction-tuning Data Auditing
description: >-
  [ACL 2026][Data Auditing] This paper proposes a Decomposition-then-Evaluation paradigm and the EVIAN framework, which decomposes responses in visual instruction tuning data into three components—visual description, subjective reasoning, and factual claims—and evaluates them along three orthogonal dimensions: image-text consistency, logical coherence, and factual accuracy. Models trained on the small high-quality subset selected by EVIAN outperform those trained on large-scale datasets.
tags:
  - ACL 2026
  - Data Auditing
  - Visual Instruction Tuning
  - Explainable Evaluation
  - Data Quality
  - Multimodal Large Language Models
date: 2026-05-08
content_hash: 8c73501efdb0d378
---

# Evian: Towards Explainable Visual Instruction-tuning Data Auditing

**Conference**: ACL 2026
**arXiv**: [2604.20544](https://arxiv.org/abs/2604.20544)
**Code**: N/A
**Area**: Interpretability
**Keywords**: Data Auditing, Visual Instruction Tuning, Explainable Evaluation, Data Quality, Multimodal Large Language Models

## TL;DR

This paper proposes a Decomposition-then-Evaluation paradigm and the EVIAN framework, which decomposes responses in visual instruction tuning data into three components—visual description, subjective reasoning, and factual claims—and evaluates them along three orthogonal dimensions: image-text consistency, logical coherence, and factual accuracy. Models trained on the small high-quality subset selected by EVIAN outperform those trained on large-scale datasets.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) rely on Visual Instruction Tuning (VIT) to align visual perception with language understanding, yet the quality of training data varies considerably.

**Limitations of Prior Work**: (1) Large-scale data synthesis (e.g., LLaVA-Instruct-150K) improves instruction following but introduces noise; (2) existing filtering methods (e.g., CLIP score) employ coarse-grained, single-dimensional scoring that cannot detect subtle semantic defects such as logical fallacies and factual errors; (3) the LLM-as-a-Judge paradigm suffers from bias, instability, and reasoning shortcuts.

**Key Challenge**: Existing data filtering compresses multiple error types into a single opaque score, making it impossible to distinguish between visual misrepresentation, factual inaccuracy, and reasoning defects.

**Goal**: To construct an explainable, fine-grained data auditing framework that decomposes responses into verifiable cognitive components for multi-dimensional evaluation.

**Key Insight**: Responses are treated as composite structures consisting of visual descriptions, subjective reasoning, and factual claims, rather than indivisible text blocks.

**Core Idea**: By decomposing the complex auditing task into verifiable sub-tasks targeting distinct cognitive components, data quality assessment can be made more precise than coarse-grained scoring, with logical coherence identified as the most critical factor in data quality.

## Method

### Overall Architecture

EVIAN operates in two phases. Phase 1 (Response Decomposition) decomposes responses into a labeled structured form and a pure visual summary via a three-step chain-of-thought (semantic annotation → visual distillation → fluent synthesis). Phase 2 (Multi-dimensional Evaluation) scores responses along three orthogonal dimensions—logical coherence $S_L$, factual accuracy $S_K$, and image-text consistency $S_V$—on a 1–5 scale, with the final score computed as $S_{\text{overall}} = (S_L + S_K + S_V) / 3$.

### Key Designs

1. **Three-Step Chain-of-Thought Decomposition**:

    - Function: Decomposes complex responses into independently verifiable cognitive components.
    - Mechanism: Step 1 (Semantic Annotation) marks subjective reasoning with `<INFER>` tags and factual claims with `<KNOW>` tags, leaving unannotated content as pure visual description; Step 2 (Visual Distillation) removes or rewrites tagged content to retain only objective descriptions; Step 3 (Fluent Synthesis) organizes the fragmented distilled results into coherent paragraphs.
    - Design Motivation: Decomposition enables each component to be evaluated independently along the most appropriate dimension, avoiding the ambiguity of mixed evaluation.

2. **Three-Dimensional Orthogonal Evaluation System**:

    - Function: Separately assesses logical reasoning, factual knowledge, and visual alignment quality.
    - Mechanism: $S_L$ evaluates the logical validity of reasoning within `<INFER>` tags (i.e., whether visual evidence supports the inference); $S_K$ fact-checks the knowledge claims within `<KNOW>` tags; $S_V$ measures the consistency between the pure visual summary and the image, prioritizing consistency over completeness.
    - Design Motivation: Different types of defects require different evaluation criteria; orthogonal separation prevents cross-dimensional interference.

3. **Controlled Defect Injection Benchmark**:

    - Function: Provides a systematic test platform with 300K samples.
    - Mechanism: Fifteen semantic defect categories are designed (5 for visual consistency + 5 for logical coherence + 5 for factual accuracy), and subtle context-dependent defects are injected through a three-stage pipeline (content analysis → context-aware error selection → guided rewriting).
    - Design Motivation: Existing datasets lack systematically injected, controllable errors, making it impossible to quantitatively evaluate the fine-grained detection capability of auditing pipelines.

### Loss & Training

Qwen3-235B is used for response decomposition, and Qwen2.5-VL-7B serves as the automated auditor for scoring. Downstream validation fine-tunes Qwen2-VL-2B on the selected 10K subset. All experiments share the same architecture and SFT procedure.

## Key Experimental Results

### Main Results (Fine-tuning Qwen2-VL-2B on 10K Subset)

| Method | MME | MMBench | ScienceQA | A-OKVQA | POPE | Avg |
|--------|-----|---------|-----------|---------|------|-----|
| Random | 1475.76 | 0.5353 | 0.6614 | 0.7092 | 75.50 | 63.18 |
| Full Data (300K) | 1553.05 | 0.5953 | 0.6267 | 0.6934 | 78.17 | 63.77 |
| SCALE (Prev. SOTA) | 1814.97 | 0.6318 | 0.6916 | 0.7066 | 73.81 | 67.41 |
| EVIAN (Ours) | 1876.89 | 0.6463 | 0.7115 | 0.7493 | 79.87 | 70.20 |

### Ablation Study

| Configuration | Avg | Note |
|---------------|-----|------|
| EVIAN (Full) | 70.20 | Full framework achieves best performance |
| w/o Decomposition | 67.93 | Removing decomposition causes a drop of 2.27 |
| w/o $S_L$ (Logical Coherence) | 57.27 | **Largest drop when logical coherence is removed** (↓12.93) |
| w/o $S_K$ (Factual Accuracy) | 64.21 | Removing factual accuracy causes a drop of 5.99 |
| Only $S_V$ (Image-Text Consistency) | 65.36 | Visual consistency alone is acceptable but POPE drops sharply to 68.56 |

### Key Findings
- **Logical coherence is the most critical dimension**: Removing $S_L$ causes Avg to collapse from 70.20 to 57.27, because relying solely on $S_K$ and $S_V$ selects samples that are factually correct but logically inconsistent, producing contradictory supervision signals.
- **"Less is more"**: The 10K subset selected by EVIAN (3.3% of 300K) yields better training outcomes than the full 300K dataset.
- In the score distribution, 92.3% of original high-quality samples receive scores ≥ 3.0, while defective samples cluster around 3.0 (JSD = 0.35, AUC = 0.86).
- Cross-architecture validation on InternVL2-2B confirms that the gains stem from data quality rather than inductive bias alignment between the auditor and the target model.

## Highlights & Insights
- The central insight of the Decomposition-then-Evaluation paradigm is that decomposing auditing into verifiable sub-tasks renders complex auditing reliable.
- The work challenges the prevailing assumption that more data is better, surpassing full-data training with only 3.3% of the data.
- The counter-intuitive finding that logical coherence—rather than visual alignment or factual accuracy—is the most critical data quality factor carries broad implications.
- The defect injection benchmark features a systematic taxonomy covering three major categories (consistency, reasoning, and knowledge), each with five error subtypes.

## Limitations & Future Work
- The framework depends on large multimodal models for decomposition and evaluation, potentially inheriting their biases and blind spots.
- Errors introduced during the decomposition phase propagate to subsequent evaluation stages, leaving robustness to be improved.
- High computational cost (multiple invocations of large models) limits applicability to very large-scale datasets.
- Other data quality dimensions such as stylistic diversity and pedagogical value are not modeled.

## Related Work & Insights
- **vs. SCALE**: SCALE employs multi-stage filtering (modality quality, relevance, clarity, task rarity) but performs no component-level decomposition; EVIAN achieves more precise fine-grained auditing through cognitive component decomposition.
- **vs. CLIPScore/BLIP**: Similarity-based coarse-grained filtering cannot capture logical fallacies and factual errors.
- **vs. LLM-as-a-Judge**: Directly prompting models for holistic scores introduces bias and instability; EVIAN mitigates this through structured decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐ The Decomposition-then-Evaluation paradigm is novel, and the 15-category defect taxonomy is systematic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes multi-baseline comparison, comprehensive ablation, cross-architecture validation, and a 300K-sample benchmark.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, richly illustrated, and analytically thorough.
- Value: ⭐⭐⭐⭐ Offers important guidance for multimodal data curation; the finding that logical coherence should be prioritized has wide-ranging implications.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] TabReX: Tabular Referenceless eXplainable Evaluation](tabrex_tabular_referenceless_explainable_evaluation.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](../../ICLR2026/interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](../../ICLR2026/interpretability/auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)

<!-- RELATED:END -->
