---
title: >-
  [Paper Note] SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision
description: >-
  [AAAI 2026][LLM Reasoning][Process Reward Model] This paper proposes SPARE, a unified single-pass evaluation framework that simultaneously performs step-to-reference alignment and correctness judgment (with explicit reas…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Process Reward Model"
  - "Automatic Annotation"
  - "Reference-Guided"
  - "Single-Pass Generation"
  - "Data Efficiency"
date: 2026-05-08
content_hash: 2d184d8d21bbde4d
---

# SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision

**Conference**: AAAI 2026
**arXiv**: [2506.15498](https://arxiv.org/abs/2506.15498)  
**Code**: [https://github.com/UKPLab/aaai2026-spare-prm](https://github.com/UKPLab/aaai2026-spare-prm)  
**Area**: LLM Reasoning
**Keywords**: Process Reward Model, Automatic Annotation, Reference-Guided, Single-Pass Generation, Data Efficiency

## TL;DR

This paper proposes SPARE, a unified single-pass evaluation framework that simultaneously performs step-to-reference alignment and correctness judgment (with explicit reasoning) in a single structured generation, requiring no additional training data. SPARE achieves 2.3× speedup over MCTS-based methods and attains OOD generalization with only 16% of the training samples.

## Background & Motivation

Process Reward Models (PRMs) provide step-level supervision signals to guide multi-step reasoning in LLMs, outperforming Outcome Reward Models (ORMs) that only evaluate final answers. However, the core bottleneck of PRMs lies in **acquiring step-level annotation data**—determining the correctness of each reasoning step.

**Limitations of existing automatic annotation approaches**:

**Manual annotation (PRM800K)**: Requires expert mathematicians to evaluate each step, making it prohibitively costly and unscalable.

**MCTS-based methods**: Perform multiple forward rollouts from each intermediate step and infer step quality from final answer accuracy. The computational overhead is enormous—each step requires dozens of complete rollouts.

**Existing reference-guided methods**: Either rely on stronger teacher models to generate synthetic reasoning traces (GenRM, ThinkPRM), or require human step labels for filtering, limiting generalizability.

**Key waste**: All existing methods overlook the reference solutions (ground-truth reasoning traces) already present in SFT datasets—these high-quality step-level traces are left underutilized.

**Core insight of SPARE**: Reference solutions available during SFT training encode rich step-level information. Rather than discarding this information and running MCTS from scratch, SPARE directly prompts an LLM to align and evaluate each step of a candidate output against the corresponding steps of the reference solution—all within a single generation.

## Method

### Overall Architecture

SPARE is a unified single-stage evaluation framework. Given a context $\mathcal{C}$, a reference reasoning path $\mathcal{R}$ (with $m$ steps), and a model-generated output $\mathcal{O}$ (with $n$ steps), SPARE produces an evaluation sequence $\mathcal{E}$ in a single LLM generation, containing alignment information and correctness labels for each step.

Input: $(S, C, R, O) \rightarrow \mathcal{E}$

For each step $o_i$, the evaluation tuple $\varepsilon = (e, c^+, o^+, r^+, \epsilon, y_i)$ consists of:
- $e$: natural language explanation (why the step is correct/incorrect)
- $c^+$: relevant contextual sentences
- $o^+$: related output steps
- $r^+$: aligned reference solution steps
- $\epsilon$: list of error categories (e.g., calculation error, logical leap)
- $y_i \in \{-1, +1\}$: correctness label

### Key Designs

1. **Joint Alignment + Evaluation via ICL**

    - Step evaluation is framed analogously to Natural Language Inference (NLI) with evidence localization: not only determining step correctness, but also identifying the supporting reference evidence.
    - The system prompt encodes instance-agnostic alignment and evaluation criteria.
    - In-context examples demonstrate how to apply these criteria to concrete instances.
    - All steps are evaluated in a single generation pass; computational cost scales only additively with token length of the response and reference.

2. **Explicit Reasoning Annotation**

    - Rather than producing binary labels only, the LLM is required to explain the reasoning behind each step's judgment.
    - An error taxonomy is defined: calculation errors, logical leaps, misalignment with reference, faulty premises, etc.
    - This improves interpretability and debuggability—annotations are no longer a black box.

3. **Two Downstream Applications**

    - **PRM Training** (ranking/aggregation): SPARE-annotated data is used to train a process reward model, which at inference time performs Best-of-N selection or self-consistency voting over $N$ candidate outputs.
    - **Offline RL Fine-tuning**: Step-level signals from SPARE are used for DPO/offline RL to improve greedy decoding quality.
    - Both applications yield consistent gains across 4 datasets.

4. **Zero Additional Data Cost**

    - Reference solutions are directly reused from reasoning traces already present in standard SFT datasets, requiring no additional generation.
    - The entire pipeline requires only a single LLM—no stronger teacher model is needed.

### Loss & Training

PRM training uses a standard binary classification loss. Offline RL fine-tuning uses the DPO objective. SPARE itself is an inference-time ICL framework and requires no training.

## Key Experimental Results

### Main Results (Llama-3 8B Instruct, aggregation/ranking with N=20)

| Method | GSM8K | MATH-500 | MuSiQue | SpaRP |
|--------|-------|----------|---------|-------|
| Self-Consistency | 74.9 | 23.4 | 19.7/25.2 | 25.4/34.4 |
| ORM (BoN) | 79.7 | 20.2 | 33.4/45.4 | 41.7/49.8 |
| ORM + SC | 79.8 | 23.8 | 34.8/44.5 | 41.7/49.8 |
| SPARE (BoN) | **80.0** | 20.9 | **34.9/45.5** | **43.7/50.0** |
| **SPARE + SC** | **80.3** | **24.1** | 32.1/40.4 | 39.6/46.9 |

### Data Efficiency (ProcessBench OOD Generalization)

| Method | Training Data Size | ProcessBench Performance |
|--------|--------------------|--------------------------|
| Human annotation baseline | 100% | Baseline |
| MCTS baseline | 100% | Competitive |
| **SPARE** | **~16%** | **Competitive** |

### Efficiency Comparison

| Method | Total Tokens | Relative Speed |
|--------|-------------|----------------|
| MCTS | Multiple rollouts | 1× |
| **SPARE** | Single-pass generation | **2.3×** |

### Key Findings

- **SPARE and MCTS are complementary**: SPARE yields higher precision but slightly lower recall; MCTS yields higher recall but slightly lower precision—ensembling the two is a natural extension.
- **OOD generalization with only 16% of training data**: Reference-guided alignment substantially reduces data requirements, suggesting that step alignment is more important than annotation volume.
- **Cross-task generalization**: Consistent effectiveness across four diverse reasoning types—mathematical reasoning (GSM8K/MATH), multi-hop QA (MuSiQue), and spatial reasoning (SpaRP).
- **Explicit reasoning improves annotation quality**: Requiring the LLM to explain each step's judgment yields more reliable and interpretable annotations.

## Highlights & Insights

- **Single-pass efficiency paradigm**: Merging step alignment and evaluation into a single LLM call, rather than evaluating each step independently, yields substantial computational gains.
- **Secondary reuse of SFT data**: Reference solutions are used for training during SFT and again for evaluation during annotation—an elegant zero-cost data reuse design.
- **Analogy to NLI**: Step evaluation is framed as natural language inference with evidence localization, offering a new theoretical perspective on process supervision.
- **Practical value of precision–recall complementarity**: Ensembling SPARE with MCTS may represent the most effective combined solution.

## Limitations & Future Work

- Dependent on reference solution quality—errors in reference solutions (e.g., known annotation noise in MATH) propagate into the annotations.
- Step granularity mismatch may hinder alignment—a single candidate step may correspond to multiple reference steps, complicating alignment.
- ICL is constrained by context window length—very long reasoning chains may not be fully processed in a single generation.
- Evaluation is limited to mathematical, QA, and spatial reasoning tasks; other multi-step reasoning domains such as code generation remain unexplored.
- Ensemble strategies combining SPARE and MCTS warrant further investigation.

## Related Work & Insights

- **vs. MCTS-based methods (Math-Shepherd, OmegaPRM)**: MCTS infers step quality from final outcomes via multiple rollouts per step—computationally expensive but with high recall. SPARE uses direct reference alignment for evaluation, achieving 2.3× speedup at the cost of slightly lower recall. The two approaches are complementary.
- **vs. GenRM (Zhang et al. 2025)**: GenRM uses a stronger model to generate synthetic reasoning traces as reference; it functions as an ORM rather than a true PRM. SPARE provides genuine step-level process supervision without requiring a stronger teacher model.
- **vs. ThinkPRM (Khalifa 2025)**: ThinkPRM generates verification reasoning with a stronger model and additionally requires human step labels from PRM800K for filtering. SPARE requires no human step labels whatsoever.

## Rating

- Novelty: ⭐⭐⭐⭐ — Efficient single-pass joint alignment and evaluation design; novel approach to reusing reference solutions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four diverse reasoning benchmarks, two downstream applications (PRM + RL), efficiency analysis, and complementarity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear architectural diagrams; the NLI analogy aids comprehension.
- Value: ⭐⭐⭐⭐⭐ — Substantially reduces the cost of acquiring step-level process supervision, with direct practical utility for PRM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Annotation-Efficient Universal Honesty Alignment](../../ICLR2026/llm_reasoning/annotation-efficient_universal_honesty_alignment.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](../../NeurIPS2025/llm_reasoning/large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)
- [\[AAAI 2026\] Improving Value-based Process Verifier via Low-Cost Variance Reduction](improving_value-based_process_verifier_via_low-cost_variance_reduction.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)

</div>

<!-- RELATED:END -->
