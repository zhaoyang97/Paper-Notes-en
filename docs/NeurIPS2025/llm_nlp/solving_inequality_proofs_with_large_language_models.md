---
title: >-
  [Paper Note] Solving Inequality Proofs with Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][mathematical reasoning] This paper proposes IneqMath, the first large-scale olympiad-level inequality benchmark, formulates inequality proving as two automatically verifiable subtasks (bound estimation and relation prediction), develops a five-module LLM-as-Judge framework, and finds that even o1 achieves an overall accuracy below 10% under step-by-step reasoning scrutiny.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - mathematical reasoning
  - inequality proving
  - LLM-as-Judge
  - benchmark
  - reasoning evaluation
date: 2026-05-08
content_hash: 655a9f6808079aff
---

# Solving Inequality Proofs with Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.07927](https://arxiv.org/abs/2506.07927)
**Code**: [IneqMath](https://ineqmath.github.io/)
**Area**: LLM/NLP
**Keywords**: mathematical reasoning, inequality proving, LLM-as-Judge, benchmark, reasoning evaluation

## TL;DR

This paper proposes IneqMath, the first large-scale olympiad-level inequality benchmark, formulates inequality proving as two automatically verifiable subtasks (bound estimation and relation prediction), develops a five-module LLM-as-Judge framework, and finds that even o1 achieves an overall accuracy below 10% under step-by-step reasoning scrutiny.

## Background & Motivation

Inequality proving is a central challenge in mathematical reasoning, requiring the discovery of tight bounds, strategic application of classical theorems (AM-GM, Cauchy-Schwarz, etc.), and precise symbolic manipulation.

Existing resources are insufficient:

**General ATP datasets** (MiniF2F, ProofNet): contain very few inequalities

**Synthetic datasets** (INT, AIPS): template-based generation, lacking structural diversity

**Manually curated sets** (ChenNEQ): too small in scale

**Format limitations**: most adopt formal representations (Lean/Isabelle), whereas LLMs may perform better in natural language reasoning

Core problem: **evaluating only final answer accuracy severely overestimates LLMs' mathematical reasoning ability**—models may guess the correct answer while the reasoning process is riddled with flaws.

## Method

### Overall Architecture

Three main contributions: (1) informal yet verifiable task definitions; (2) the IneqMath dataset; (3) a five-module LLM-as-Judge evaluation framework.

### Key Design 1: Informal but Verifiable Tasks

Inequality proving is decomposed into two automatically checkable subtasks:

**Bound Estimation**: find the optimal constant for which the inequality holds:
$$C^* = \sup\{C \in \mathbb{R} : f(\mathbf{x}) \geq C \cdot g(\mathbf{x}), \forall \mathbf{x} \in \mathcal{D}\}$$

**Relation Prediction**: determine the relationship between two expressions ($>, \geq, =, \leq, <$)

### Key Design 2: IneqMath Dataset

| Statistic | Count | Bound Estimation | Relation Prediction |
|-----------|:-----:|:----------------:|:-------------------:|
| Test set | 200 | 96 | 104 |
| Dev set | 100 | 50 | 50 |
| Train set | 1,252 | 626 | 626 |
| Theorem categories | 29 | - | - |
| Named theorems | 83 | - | - |
| With theorem annotation | 962 | 482 | 480 |
| Max solutions per problem | 4 | 4 | 4 |

The test set was originally designed by IMO medalists and verified by an independent panel of experts. The training set includes step-by-step solutions and theorem annotations.

### Key Design 3: Five-Module LLM-as-Judge Framework

A solution is considered correct only if it passes all five judges:

1. **Final Answer Judge**: two-stage verification—extract conclusion sentence + check mathematical equivalence (handling $\frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$), F1 = 1.00
2. **Toy Case Judge**: detects errors where general conclusions are drawn solely from specific numerical cases, F1 = 0.91
3. **Logical Gap Judge**: flags missing reasoning steps and unsupported assertions, F1 = 0.96
4. **Numerical Approximation Judge**: flags improper use of numerical approximations, F1 = 0.96
5. **Numerical Computation Judge**: verifies arithmetic step correctness (Python/SymPy), F1 = 0.80

Overall F1 = 0.93.

### Loss & Training

This work presents an evaluation framework and does not involve model training. The step-by-step solutions and theorem annotations in the training set are available for future research.

## Key Experimental Results

### Main Results

Zero-shot evaluation of 29 leading LLMs:

| Model | Answer Accuracy | Overall Accuracy | Drop |
|-------|:--------------:|:---------------:|:----:|
| o1 | ~73.5% | **8.0%** | **-65.5%** |
| o3 | ~65% | ~31% | -34% |
| Qwen2.5-72B | 42.0% | 2.5% | -39.5% |
| GPT-4o | ~40% | ~2% | -38% |
| Llama-4-Maverick | 40.5% | 2.5% | -38% |

### Ablation Study

**Model scale**: larger models improve final answer accuracy but provide limited gains in overall accuracy

**Test-time compute**: o1's overall accuracy remains 8.0% across 5K to 40K max tokens; o3 saturates at approximately 31%

**Improvement strategies**:
- Theorem guidance: overall accuracy of o3-mini improves by **+11%**
- Critic-guided self-refinement: Gemini 2.5 Pro improves by **+5%**

### Key Findings

1. **Correct answers do not imply correct reasoning**: accuracy drops of up to 65.5% are observed
2. **Scaling is not a panacea**: model size and inference time yield minimal improvement in step-level correctness
3. **Taxonomy of LLM reasoning flaws**: logical gaps > toy-case generalization > numerical approximation misuse > arithmetic errors
4. **Theorem knowledge is critical**: providing relevant theorems significantly improves reasoning quality

## Highlights & Insights

1. **Reveals an evaluation blind spot**: the first systematic demonstration of drastic accuracy drops under step-level scrutiny
2. **Modular judge design** outperforms monolithic general-purpose judges
3. **High dataset quality**: original problems by IMO medalists, independently verified, with multi-solution annotations
4. **Points to concrete improvement directions**: theorem-guided reasoning and self-refinement, rather than naive scaling
5. **Comprehensive evaluation scope**: 29 models covering both open-source and proprietary systems

## Limitations & Future Work

1. **LLM-as-Judge is imperfect**: the numerical computation judge achieves only F1 = 0.80
2. **Limited test set size**: 200 problems offer limited statistical power
3. **Zero-shot evaluation only**: few-shot and fine-tuned performance remain unexplored
4. **Inequality types skewed toward competitions**: insufficient coverage of inequalities in analysis and probability theory
5. Hybrid approaches combining formal and informal verification are worth exploring

## Related Work & Insights

- **MiniF2F** (Zheng et al., 2022): formal mathematical benchmark
- **MATH/GSM8K**: conventional math benchmarks focusing on final answers
- **LLM-as-Judge** (Zheng et al., 2024): general-purpose judging framework

## Rating

⭐⭐⭐⭐⭐ (4.5/5)

- **Novelty** ⭐⭐⭐⭐⭐: task definition, evaluation framework, and dataset constitute a unified contribution
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: systematic evaluation across 29 models
- **Impact** ⭐⭐⭐⭐⭐: exposes fundamental deficiencies in LLM mathematical reasoning
- **Writing Quality** ⭐⭐⭐⭐⭐: logically clear with well-crafted figures and tables

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](scaling_up_active_testing_to_large_language_models.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[NeurIPS 2025\] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)

<!-- RELATED:END -->
