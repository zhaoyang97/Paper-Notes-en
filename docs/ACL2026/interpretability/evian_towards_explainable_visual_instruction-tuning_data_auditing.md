---
title: >-
  [Paper Note] Evian: Towards Explainable Visual Instruction-tuning Data Auditing
description: >-
  [ACL 2026][Interpretability][Data Auditing] The "Decomposition-then-Evaluation" paradigm and the EVIAN framework are proposed. The approach decomposes visual instruction tuning responses into visual descriptions…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Data Auditing"
  - "Visual Instruction Tuning"
  - "Explainable Evaluation"
  - "Data Quality"
  - "Multi-modal Large Models"
date: 2026-05-08
content_hash: 3861c435ecd74668
---

# Evian: Towards Explainable Visual Instruction-tuning Data Auditing

**Conference**: ACL 2026  
**arXiv**: [2604.20544](https://arxiv.org/abs/2604.20544)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Data Auditing, Visual Instruction Tuning, Explainable Evaluation, Data Quality, Multi-modal Large Models

## TL;DR

The "Decomposition-then-Evaluation" paradigm and the EVIAN framework are proposed. The approach decomposes visual instruction tuning responses into visual descriptions, subjective reasoning, and factual claims. These are evaluated along three orthogonal dimensions: visual consistency, logical coherence, and factual accuracy. Models trained on a small amount of high-quality data selected using this framework outperform models trained on large-scale datasets.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) rely on Visual Instruction Tuning (VIT) to achieve alignment between visual perception and language understanding, but the quality of training data varies significantly.

**Limitations of Prior Work**: (1) Large-scale data synthesis (e.g., LLaVA-Instruct-150K) improves instruction following but introduces noise; (2) existing filtering methods (e.g., CLIP score) use coarse-grained single-dimensional scoring, failing to detect subtle semantic defects like logical fallacies and factual errors; (3) the LLM-as-a-Judge paradigm suffers from biases, instability, and reasoning shortcuts.

**Key Challenge**: Current data filtering compresses multiple error types into a single opaque score, failing to distinguish between different quality issues such as visual misrepresentation, factual inaccuracy, and reasoning flaws.

**Goal**: Build an explainable, fine-grained data auditing framework that decomposes responses into verifiable cognitive components for multi-dimensional evaluation.

**Key Insight**: Responses should be viewed as composite structures consisting of visual descriptions, subjective reasoning, and factual claims, rather than indivisible text blocks.

**Core Idea**: By decomposing complex auditing tasks into verifiable sub-tasks targeting different cognitive components, more precise data quality assessment can be achieved compared to coarse-grained scoring. Furthermore, logical coherence is identified as the most critical factor in data quality.

## Method

### Overall Architecture

EVIAN consists of two phases: Phase 1 (Response Decomposition) utilizes a three-step Chain-of-Thought (Semantic Annotation → Visual Distillation → Fluent Synthesis) to decompose responses into labeled structural forms and pure visual summaries. Phase 2 (Multi-dimensional Evaluation) scores the data along three orthogonal dimensions (1-5 scale): logical coherence $S_L$, factual accuracy $S_K$, and visual consistency $S_V$. The final score is the three-way average: $S_{\text{overall}} = (S_L + S_K + S_V) / 3$.

### Key Designs

1.  **Three-step Chain-of-Thought Decomposition**:
    - **Function**: Decomposes complex responses into independently verifiable cognitive components.
    - **Mechanism**: Step 1 Semantic Annotation—adds `<INFER>` tags for subjective reasoning and `<KNOW>` tags for factual claims, leaving the rest as pure visual description; Step 2 Visual Distillation—removes or rewrites tagged content to retain only objective descriptions; Step 3 Fluent Synthesis—organizes fragmented distillation results into coherent paragraphs.
    - **Design Motivation**: Decomposition allows each component to be evaluated along the most suitable dimension, avoiding the ambiguity of blended assessments.

2.  **Three-dimensional Orthogonal Evaluation System**:
    - **Function**: Independently evaluates the quality of logical reasoning, factual knowledge, and visual alignment.
    - **Mechanism**: $S_L$ assesses the logicality of reasoning in `<INFER>` tags (whether supported by visual evidence); $S_K$ fact-checks knowledge claims in `<KNOW>` tags; $S_V$ measures consistency between the pure visual summary and the image (prioritizing consistency over completeness).
    - **Design Motivation**: Different types of defects require different evaluation criteria; orthogonal separation avoids interference between dimensions.

3.  **Controlled Defect Injection Benchmark**:
    - **Function**: Provides a systematic testing platform with 300,000 samples.
    - **Mechanism**: Design a taxonomy of 15 semantic defects (5 types each for visual consistency, logical coherence, and factual accuracy), injected via a three-stage pipeline (content analysis → contextual error selection → guided rewriting).
    - **Design Motivation**: Existing datasets lack systematic, controlled errors, making it impossible to quantify the fine-grained detection capabilities of auditing pipelines.

### Loss & Training

Qwen3-235B is used for response decomposition, and Qwen2.5-VL-7B serves as the automatic auditor for scoring. Downstream validation is performed by fine-tuning Qwen2-VL-2B on a filtered 10K subset. All experiments share the same architecture and SFT pipeline.

## Key Experimental Results

### Main Results (Qwen2-VL-2B fine-tuned on 10K subset)

| Method | MME | MMBench | ScienceQA | A-OKVQA | POPE | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random | 1475.76 | 0.5353 | 0.6614 | 0.7092 | 75.50 | 63.18 |
| Full Data (300K) | 1553.05 | 0.5953 | 0.6267 | 0.6934 | 78.17 | 63.77 |
| SCALE (Prev. SOTA) | 1814.97 | 0.6318 | 0.6916 | 0.7066 | 73.81 | 67.41 |
| EVIAN (Ours) | 1876.89 | 0.6463 | 0.7115 | 0.7493 | 79.87 | 70.20 |

### Ablation Study

| Configuration | Avg | Description |
| :--- | :--- | :--- |
| EVIAN (Full) | 70.20 | Optimal full framework |
| w/o Decomposition | 67.93 | Loss of 2.27 without decomposition phase |
| w/o $S_L$ (Log. Coherence) | 57.27 | **Largest loss without logical coherence** (↓12.93) |
| w/o $S_K$ (Fact. Acc.) | 64.21 | Loss of 5.99 without factual accuracy |
| Only $S_V$ (Vis. Consist.) | 65.36 | Visual consistency alone is mediocre; POPE drops to 68.56 |

### Key Findings
- **Logical coherence is most critical**: Removing $S_L$ caused the Avg to plummet from 70.20 to 57.27, as relying only on $S_K$ and $S_V$ includes samples that are factually correct but logically inconsistent, producing contradictory supervision signals.
- **"Less is more"**: The 10K subset filtered by EVIAN (3.3% of the 300K) outperformed training on the full 300K dataset.
- In the score distribution, 92.3% of original high-quality samples scored $\ge 3.0$, while defect samples clustered around 3.0 (JSD=0.35, AUC=0.86).
- Cross-architecture validation (InternVL2-2B) shows that improvements stem from data quality rather than the auditor aligning with the target model's inductive biases.

## Highlights & Insights
- Core insight of the "Decomposition-then-Evaluation" paradigm: Decomposing auditing into verifiable sub-tasks makes complex auditing reliable.
- Challenges the "more data is better" paradigm, surpassing full-scale training with only 3.3% of the data.
- Discovers that logical coherence (rather than visual alignment or factual accuracy) is the most critical factor in data quality, a counter-intuitive but significant conclusion.
- The systematic taxonomy design of the defect injection benchmark covers 5 error sub-types across consistency, reasoning, and knowledge.

## Limitations & Future Work
- Reliance on large multi-modal models for decomposition and evaluation may inherit their biases and blind spots.
- Errors in the decomposition stage propagate to subsequent evaluations, necessitating further improvements in robustness.
- High computational costs (multiple LLM calls) limit application to ultra-large-scale datasets.
- Other data quality dimensions, such as stylistic diversity and pedagogical value, are not yet modeled.

## Related Work & Insights
- **vs SCALE**: SCALE uses multi-stage filtering (modality quality, relevance, clarity, task rarity) but lacks component-level decomposition; EVIAN achieves more precise fine-grained auditing through cognitive component decomposition.
- **vs CLIPScore/BLIP**: Coarse-grained filtering based on similarity fails to capture logical fallacies and factual errors.
- **vs LLM-as-a-Judge**: Directly asking models for an overall score introduces bias and instability; EVIAN mitigates this through structured decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐ "Decomposition-then-Evaluation" paradigm is novel; 15-type defect taxonomy is systematic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive baseline comparisons, complete ablation, cross-architecture validation, and a 300,000-sample benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and in-depth analysis.
- Value: ⭐⭐⭐⭐ Significant guidance for multi-modal data curation; the "logical coherence first" finding has a broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](../../ICLR2026/interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ACL 2026\] Diffusion-CAM: Faithful Visual Explanations for dMLLMs](diffusion-cam_faithful_visual_explanations_for_dmllms.md)
- [\[ACL 2026\] Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective](investigating_more_explainable_and_partition-free_compositionality_estimation_fo.md)
- [\[ACL 2026\] The Impact of Off-Policy Training Data on Probe Generalisation](the_impact_of_off-policy_training_data_on_probe_generalisation.md)
- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](../../ICLR2026/interpretability/auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)

</div>

<!-- RELATED:END -->
