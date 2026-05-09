---
title: >-
  [Paper Note] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making
description: >-
  [ICLR2026][LLM/NLP][LLM evaluation] This paper reveals hidden failure modes of LLMs in data-constrained scientific decision-making tasks: models can exhibit near-perfect run-to-run stability while systematically diverging from statistical ground truth, manifesting as over-selection, prompt sensitivity, and hallucinated gene identifiers.
tags:
  - ICLR2026
  - LLM/NLP
  - LLM evaluation
  - scientific decision-making
  - stability
  - correctness
  - gene prioritization
  - differential expression analysis
date: 2026-05-08
content_hash: 08995683bfd4dc26
---

# When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making

**Conference**: ICLR2026
**arXiv**: [2603.15840](https://arxiv.org/abs/2603.15840)
**Code**: [github.com/NaziaRiasat/llm-prompt-sensitivity](https://github.com/NaziaRiasat/llm-prompt-sensitivity)
**Area**: LLM/NLP
**Keywords**: LLM evaluation, scientific decision-making, stability, correctness, gene prioritization, differential expression analysis

## TL;DR
This paper reveals hidden failure modes of LLMs in data-constrained scientific decision-making tasks: models can exhibit near-perfect run-to-run stability while systematically diverging from statistical ground truth, manifesting as over-selection, prompt sensitivity, and hallucinated gene identifiers.

## Background & Motivation
- LLMs are increasingly deployed as decision-support tools in scientific workflows (data interpretation, hypothesis generation, candidate gene prioritization, etc.)
- Evaluation practices typically emphasize **stability** (run-to-run reproducibility) as a reliability proxy
- However, in structured scientific decision-making tasks, stability **does not imply correctness**—when a statistical ground truth exists, stable outputs may systematically deviate from it
- Core question: Can run-to-run consistency serve as a proxy for correctness in scientific tasks?

## Core Problem
When a reliable statistical reference standard is available, is LLM stability sufficient to guarantee output correctness? This paper answers this question through controlled experiments that systematically disentangle four behavioral dimensions.

## Method

### Evaluation Framework Design
Gene prioritization task based on differential expression (DE) analysis:
- **Input**: Fixed DESeq2 differential expression result table (RNA-seq data, GSE239514)
- **Statistical reference**: Deterministic DESeq2 analysis results as ground truth
- **Evaluated models**: ChatGPT (GPT-5.2), Google Gemini 3, Claude Opus 4.5
- **All models use deterministic decoding with temperature = 0**

### Four Evaluation Dimensions
1. **Stability**: Run-to-run output consistency, measured by Jaccard similarity and overlap coefficient
2. **Correctness**: Degree of agreement with the DESeq2 statistical reference
3. **Prompt Sensitivity**: Output variation across semantically equivalent prompts
4. **Output Validity**: Whether model-generated gene identifiers exist in the input table

### Prompt Design
Multiple prompt conditions covering typical analysis scenarios:
- **P1**: Strict threshold (FDR ≤ 0.05)
- **P5**: Relaxed threshold (0.05 < FDR ≤ 0.10)
- **P6**: Boundary Top-20 selection
- **P7a vs P7b**: Semantically equivalent but differently worded (statistical significance vs. effect size priority)
- **P9**: Explicit ranking Top-20

Each configuration is run 10 times with identical input.

### Evaluation Metrics

**Jaccard Similarity**:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Overlap Coefficient** (for handling unequal-sized sets):
$$O(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}$$

### Data Characteristics
- 0 genes satisfy FDR ≤ 0.05
- 35 genes fall within 0.05 < FDR ≤ 0.10
- 127 genes fall within 0.05 < FDR ≤ 0.15

## Key Experimental Results

### Summary of Core Results

| Prompt | Task Type | Metric | ChatGPT | Gemini | Claude | Interpretation |
|--------|-----------|--------|---------|--------|--------|----------------|
| P1 (FDR≤0.05) | Threshold DE | Jaccard vs. truth | 1.00 | 1.00 | 0.00 | Claude fails to recover DE genes |
| P5 (FDR≤0.10) | Relaxed threshold | Jaccard vs. truth | 0.47 | 0.28 | 0.00 | Over-selection / collapse |
| P6 (boundary) | Ranking uncertainty | Jaccard vs. truth | 0.14 | 1.00 | 0.00 | Gemini recovers ground truth |
| P6 (stability) | Within-model | Pairwise Jaccard | **1.00** | **1.00** | **1.00** | Perfect internal stability |
| P7a vs P7b | Prompt sensitivity | Jaccard | 0.74 | 0.08 | 1.00 | High wording sensitivity |
| P9 (ranking) | Validity | Invalid gene IDs/run | 0 | 0 | **20** | Hallucinated identifiers |

### Summary of Key Observations

| Failure Mode | ChatGPT | Gemini | Claude |
|-------------|---------|--------|--------|
| Run-to-run stability | ✓ Perfect | ✓ Perfect | ✓ Perfect |
| Agreement with statistical reference | Partial (P5: 0.47, P6: 0.14) | Partial (P5: 0.28, P6: 1.00) | ✗ Systematic failure |
| Prompt wording sensitivity | Moderate (0.74) | Severe (0.08) | Insensitive (1.00) |
| Output validity | ✓ | ✓ | ✗ (20 invalid IDs per run) |

## Four Failure Modes in Detail

### 1. Stability Does Not Imply Correctness
- All models exhibit near-perfect run-to-run stability (Pairwise Jaccard = 1.00)
- Yet Jaccard agreement with the DESeq2 reference ranges from 0 to 1.00
- Jaccard = 0 indicates **no overlap whatsoever** between the predicted and reference sets
- Conclusion: Deterministic behavior reflects internal consistency, not reliable statistical reasoning

### 2. Over-Selection Under Relaxed Thresholds
- Upon relaxing the threshold from FDR ≤ 0.05 to 0.05 < FDR ≤ 0.10:
    - ChatGPT Jaccard drops from 1.00 to 0.47
    - Gemini drops from 1.00 to 0.28
    - Claude remains at 0.00 throughout
- Models exhibit broad inclusion or complete collapse rather than principled sensitivity–specificity trade-offs

### 3. Prompt Wording Sensitivity
- P7a (statistical significance priority) and P7b (effect size priority) differ only in minor wording:
    - Gemini achieves a Jaccard of only 0.08 (nearly entirely different gene sets)
    - ChatGPT scores 0.74 (moderate divergence)
    - Claude scores 1.00 (insensitive, likely due to systematic failure)
- Prompt wording acts as a **hidden decision variable** rather than a neutral instruction

### 4. Hallucinated Gene Identifiers
- Claude produces 20 gene IDs per run under the P9 setting that do not appear in the input table
- These IDs are syntactically plausible and resemble real gene names, yet are entirely absent from the input data
- ChatGPT and Gemini outputs contain only valid identifiers
- This constitutes a systematic violation of input-domain constraints, not incidental formatting noise

## Highlights & Insights
1. **Precise problem formulation**: The paper clearly disentangles stability, correctness, sensitivity, and validity as four distinct dimensions
2. **Rigorous controlled experimental design**: Fixed input data, deterministic decoding, and a shared statistical reference are maintained throughout
3. **High practical relevance**: The findings raise important warnings for the deployment of LLMs in scientific workflows
4. **Broadly generalizable findings**: The conclusion that stability ≠ correctness is not limited to gene analysis
5. **Discovery of Claude's systematic hallucination**: Exposes a serious reliability concern in constrained generation

## Limitations & Future Work
- Only a single differential expression dataset and a single statistical paradigm are used
- Evaluation covers only three commercial models, excluding open-source alternatives
- Prompt optimization strategies are not explored as potential mitigations for the identified failure modes
- Gene prioritization is a domain-specific bioinformatics task; cross-domain generalization of findings remains unvalidated
- The potential benefits of few-shot prompting or retrieval-augmented generation (RAG) for improving correctness are not discussed
- The edge case of exactly zero significant genes at FDR ≤ 0.05 may amplify certain failure modes

## Related Work & Insights
- Compared to general LLM hallucination research (Li et al. 2024): this paper quantifies hallucination within a controlled statistical task
- Compared to prompt sensitivity studies (Zhu et al. 2023): this paper validates findings in a scientific decision-making setting
- Compared to LLM clinical reasoning evaluations (Singhal 2023): this paper employs a deterministic statistical ground truth
- The key contribution lies in unifying multiple known issues (hallucination, sensitivity, the illusion of stability) within a single controlled analytical framework

## Rating
- Novelty: ⭐⭐⭐ (Observations are valuable, but individual failure modes have been reported separately; the contribution lies in the unified framework)
- Experimental Thoroughness: ⭐⭐⭐ (Rigorous controlled design, but limited to 1 dataset, 3 models, and 1 task)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; the four-dimensional separation is presented compellingly)
- Value: ⭐⭐⭐⭐ (Provides important cautionary guidance for the scientific community's use of LLMs)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)
- [\[ICLR 2026\] Statistical Advantage of Softmax Attention: Insights from Single-Location Regression](statistical_advantage_of_softmax_attention_insights_from_single-location_regress.md)
- [\[ICLR 2026\] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)

<!-- RELATED:END -->
