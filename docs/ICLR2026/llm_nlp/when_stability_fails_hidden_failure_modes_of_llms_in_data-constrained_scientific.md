---
title: >-
  [Paper Note] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making
description: >-
  [ICLR 2026][LLM/NLP][LLM reliability] Through a controlled behavioral evaluation framework, this paper identifies four hidden failure modes of LLMs in data-constrained scientific decision-making tasks: high stability ≠ c…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "LLM reliability"
  - "stability vs. correctness"
  - "prompt sensitivity"
  - "scientific decision-making"
  - "gene prioritization"
date: 2026-05-08
content_hash: e9b152a7ceb939ca
---

# When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making

**Conference**: ICLR 2026
**arXiv**: [2603.15840](https://arxiv.org/abs/2603.15840)
**Code**: [https://github.com/NaziaRiasat/llm-prompt-sensitivity](https://github.com/NaziaRiasat/llm-prompt-sensitivity)
**Area**: LLM/NLP
**Keywords**: LLM reliability, stability vs. correctness, prompt sensitivity, scientific decision-making, gene prioritization

## TL;DR

Through a controlled behavioral evaluation framework, this paper identifies four hidden failure modes of LLMs in data-constrained scientific decision-making tasks: high stability ≠ correctness, prompt-wording sensitivity, over-selection under relaxed thresholds, and hallucination of invalid identifiers.

## Background & Motivation

LLMs are increasingly deployed as decision-support tools in scientific workflows, including data interpretation, hypothesis generation, and candidate gene prioritization. In such settings, researchers often treat **run-to-run output stability** as a proxy for reliability — if repeated queries return consistent results, the outputs tend to be trusted.

However, **stability does not imply correctness**. This distinction is easily overlooked in unstructured tasks, but can be precisely quantified in statistically grounded scientific tasks. The central question of this paper is:

> When a reliable statistical ground truth exists, does high LLM output stability imply high correctness?

The authors use differential gene expression analysis as a testbed — DESeq2 provides deterministic statistical reference answers, enabling precise comparison between LLM outputs and ground truth.

## Method

### Overall Architecture

A **controlled behavioral evaluation framework** is designed, decomposing LLM decision behavior into four independent dimensions:

1. **Stability**: Output consistency across repeated runs (Jaccard similarity)
2. **Correctness**: Agreement with the DESeq2 statistical reference
3. **Prompt Sensitivity**: Output divergence induced by semantically equivalent but differently worded prompts
4. **Output Validity**: Whether outputs contain gene identifiers that actually exist in the input table

### Key Designs

**Experimental Task**: Given a fixed differential expression results table (containing columns such as gene, log2FoldChange, and padj), LLMs are asked to perform gene prioritization.

**Evaluated LLMs**: ChatGPT (GPT-5.2), Google Gemini 3, and Claude Opus 4.5, all using deterministic decoding (temperature=0).

**Prompt Suite** (P1–P9):
- **P1**: Strict threshold filtering (FDR ≤ 0.05)
- **P5**: Relaxed threshold filtering (0.05 < FDR ≤ 0.10)
- **P6**: Boundary gene ranking (Top-20 from 127 boundary genes)
- **P7a/P7b**: Wording variants (emphasis on statistical significance vs. effect size) to test prompt sensitivity
- **P9**: Explicit ranked output

Each configuration is repeated 10 times to assess stability.

**Evaluation Metrics**:
- Jaccard similarity: $J(A,B) = |A \cap B| / |A \cup B|$
- Overlap coefficient: $O(A,B) = |A \cap B| / \min(|A|, |B|)$

### Loss & Training

This is an evaluation paper; no training is involved. The core contribution is the construction of a **deterministic statistical reference**: the gene sets produced by DESeq2 analysis serve as ground truth, with 0 genes satisfying FDR ≤ 0.05, 35 genes in the 0.05 < FDR ≤ 0.10 range, and 127 genes in the 0.05 < FDR ≤ 0.15 range.

## Key Experimental Results

### Main Results

Behavioral comparison of three LLMs across different prompt configurations:

| Prompt | Task Type | Metric | ChatGPT | Gemini | Claude | Interpretation |
|--------|-----------|--------|---------|--------|--------|----------------|
| P1 (FDR≤0.05) | Threshold filtering | Jaccard vs. ground truth | 1.00 | 1.00 | 0.00 | Claude completely fails |
| P5 (FDR≤0.10) | Relaxed threshold | Jaccard vs. ground truth | 0.47 | 0.28 | 0.00 | General degradation across models |
| P6 (boundary ranking) | Uncertain ranking | Jaccard vs. ground truth | 0.14 | 1.00 | 0.00 | Only Gemini recovers ground truth |
| P6 (stability) | Internal consistency | Pairwise Jaccard | 1.00 | 1.00 | 1.00 | **All models perfectly stable** |
| P7a vs P7b | Prompt sensitivity | Jaccard | 0.74 | 0.08 | 1.00 | Gemini is extremely wording-sensitive |
| P9 (ranking validation) | Output validity | Invalid genes/run | 0 | 0 | 20 | Claude exhibits systematic hallucination |

The most critical finding in the paper is captured in the P6 row: **stability is 1.00 (perfect) across all models, yet correctness values are 0.14, 1.00, and 0.00, respectively** — stability and correctness are completely decoupled.

### Ablation Study

Quantitative analysis of prompt-wording sensitivity (P7a vs. P7b; semantically identical, minimally reworded):

| Model | Jaccard (P7a vs. P7b) | Overlap Coefficient | Interpretation |
|-------|----------------------|---------------------|----------------|
| ChatGPT | 0.74 | 0.85 | Moderate sensitivity |
| Gemini | 0.08 | 0.15 | **Extreme sensitivity** |
| Claude | 1.00 | 1.00 | Insensitive (but outputs are invalid) |

Gemini's Jaccard of 0.08 indicates that two nearly identical prompts produce almost entirely non-overlapping gene selections — minor wording differences lead to drastically different decisions.

### Key Findings

1. **Stability ≠ Correctness**: The central finding. All models achieve near-perfect run-to-run stability, yet agreement with the statistical ground truth can be zero.
2. **Relaxed Thresholds Trigger Over-Selection**: Transitioning from FDR ≤ 0.05 to FDR ≤ 0.10 causes models to over-include genes rather than improve precision, manifesting as either broad inclusion or complete collapse.
3. **Systematic Hallucination in Claude**: In the ranking task, Claude generates 20 gene identifiers absent from the input table per run, and these hallucinations persist across runs — they are not random.
4. **Prompt as an Implicit Decision Variable**: Wording changes are not merely surface noise; they alter the model's interpretation of task objectives, effectively making the prompt itself an overlooked experimental variable.

## Highlights & Insights

- **A concise yet profound contribution**: A single carefully controlled experiment exposes multiple LLM failure modes in scientific settings, which is more persuasive than complex benchmarks.
- The **four-dimensional evaluation framework** — stability, correctness, sensitivity, and validity — provides valuable abstraction, as these dimensions are frequently conflated in prior LLM evaluations.
- The conclusion that **"stability is a necessary but insufficient condition for correctness"** serves as an important warning for researchers employing LLMs in scientific decision-making.
- The choice of differential expression analysis as a testbed is elegant: it provides a deterministic statistical reference, making it ideal for quantitative evaluation.

## Limitations & Future Work

1. **Single dataset**: Only one RNA-seq dataset (GSE239514) is used; generalizability remains to be validated.
2. **Single statistical paradigm**: Only DESeq2 is used as the reference; alternative statistical methods are not explored.
3. **Narrow task scope**: Only gene prioritization is evaluated; cross-domain generalization requires additional evidence.
4. Only three proprietary models are evaluated; open-source models (e.g., Llama, Mistral) are absent.
5. Root causes are not analyzed — why do certain models systematically deviate from statistical ground truth? Deeper mechanistic investigation is needed.

## Related Work & Insights

- **Singhal et al., 2023**: LLM application in clinical reasoning; the failure modes identified here carry even greater implications for clinical settings.
- **Li et al., 2024**: Systematic analysis of LLM hallucination; the gene identifier hallucinations found here represent a concrete instantiation of this phenomenon in scientific contexts.
- **Zhu et al., 2023**: Documentation of prompt sensitivity; this paper precisely quantifies the phenomenon under controlled conditions.
- Key implication: **Any system deploying LLMs in scientific workflows should implement both ground-truth verification and output validity checking**, rather than relying on output consistency alone as a basis for trust.

## Rating

- **Novelty**: 7/10 — The observation that "stability ≠ correctness" is not intuitively surprising, but its precise quantification in a controlled experiment is valuable.
- **Technical Depth**: 5/10 — Primarily an empirical evaluation; theoretical analysis and mechanistic explanation are lacking.
- **Experimental Thoroughness**: 6/10 — The evaluation dimensions are elegantly designed, but the dataset and task scope are limited.
- **Writing Quality**: 7/10 — Well-structured, though some sections are slightly redundant.
- **Value**: 8/10 — Offers immediate practical guidance for researchers integrating LLMs into scientific pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)
- [\[ICLR 2026\] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)
- [\[ICLR 2026\] First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)
- [\[ICLR 2026\] Near-Optimal Online Deployment and Routing for Streaming LLMs](near-optimal_online_deployment_and_routing_for_streaming_llms.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)

</div>

<!-- RELATED:END -->
