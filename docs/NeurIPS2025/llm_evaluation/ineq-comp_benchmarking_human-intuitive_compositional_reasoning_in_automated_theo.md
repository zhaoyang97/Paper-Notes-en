---
title: >-
  [Paper Note] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities
description: >-
  [NeurIPS 2025][LLM Evaluation][automated theorem proving] This paper introduces the Ineq-Comp benchmark, which applies compositionally transformed variants of simple inequality seed problems—variants that humans can reso…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "automated theorem proving"
  - "compositional reasoning"
  - "formal verification"
  - "Lean 4"
  - "mathematical inequalities"
  - "benchmark"
date: 2026-05-08
content_hash: 8808a6030a0303ab
---

# Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities

**Conference**: NeurIPS 2025
**arXiv**: [2505.12680](https://arxiv.org/abs/2505.12680)  
**Code**: [GitHub](https://github.com/haoyuzhao123/LeanIneqComp)  
**Area**: LLM Evaluation
**Keywords**: automated theorem proving, compositional reasoning, formal verification, Lean 4, mathematical inequalities, benchmark

## TL;DR

This paper introduces the Ineq-Comp benchmark, which applies compositionally transformed variants of simple inequality seed problems—variants that humans can resolve with minimal additional effort—to expose fundamental deficiencies in the compositional reasoning of current LLM-based formal theorem provers. Even DeepSeek-Prover-V2-7B suffers a performance drop exceeding 20%.

## Background & Motivation

Recent years have seen remarkable progress in LLM-based formal theorem proving: systems such as AlphaProof and DeepSeek-Prover-V2 have set new records on benchmarks like MiniF2F. Nevertheless, existing benchmarks suffer from the following shortcomings:

1. **MiniF2F** is small in scale and spans an extreme difficulty range, mixing introductory problems with IMO-level challenges.
2. **ProofNet / PutnamBench** focus on high-difficulty problems that current open-source models can solve only in rare cases.
3. None of these benchmarks **explicitly test compositional reasoning**—the ability to chain known simple reasoning steps into more complex arguments.
4. They carry a non-trivial risk of **data contamination**.

The authors' core insight is that the essence of mathematical reasoning lies in "standing on the shoulders of giants": complex arguments are constructed by linking simple, well-understood steps. If a prover can prove a seed problem, it should likewise handle variants obtained by straightforward compositional transformations—variants that require virtually no additional reasoning for a human expert.

## Method

### Overall Architecture

The Ineq-Comp benchmark comprises three subsets:

- **Ineq-Simp**: 150 problems generated from 75 seed problems via two classes of transformations.
- **Ineq-Mix**: A rule-based automatic generation framework that is infinitely extensible.
- **Ineq-Real**: 50 inequality problems sourced from real mathematical competitions.

### Key Designs

**Seed Problems (75 total)**:

| Technique | Count | Description |
|-----------|-------|-------------|
| AM-GM Inequality | 25 | Arithmetic–geometric mean inequality |
| Cauchy–Schwarz | 25 | Cauchy–Schwarz inequality |
| Others (Jensen/Schur/SOS/Induction) | 25 | Jensen (10) + Schur (5) + SOS (5) + Induction (5) |

Each seed problem is accompanied by a verified Lean 4 proof, written and reviewed by personnel with IMO/CMO experience.

**Type I Transformation (Variable Duplication + Composition)**:

Given a seed inequality $f(X) \geq g(X)$ with $X = (x_1, \ldots, x_n)$ and constraint $X \in \mathcal{C}$:

- Duplicate the inequality with fresh variables $Y = (y_1, \ldots, y_n)$.
- If $g(X) \geq 0$: multiplicative composition $f(X) \cdot f(Y) \geq g(X) \cdot g(Y)$.
- If $g(X)$ is not guaranteed non-negative: additive composition $f(X) + f(Y) \geq g(X) + g(Y)$.

This transformation is trivially simple for humans—it merely requires decomposing the problem into two identical parts.

**Type II Transformation (Algebraic Variable Substitution)**:

A transformation $T: \mathbb{R}^n \to \mathbb{R}^n$ (e.g., squaring, taking square roots) is applied to the variables of the seed problem, and one must prove $f(T(X)) \geq g(T(X))$ under the condition $T(X) \in \mathcal{C}$.

**Ineq-Mix Automatic Generation Framework**:

Three classes of rules can be freely combined:
1. **Compositional transformations**: combining inequalities via addition, multiplication, max, or min.
2. **Variable-level transformations**: substituting variables with algebraic expressions.
3. **Problem-level transformations**: applying monotone functions (e.g., $\exp$, $\log$) to both sides.

### Loss & Training

This is a benchmarking paper and involves no training loss functions. Evaluation uses the standard pass@N accuracy: a problem is considered "solved" if and only if at least one correct proof is produced among $N$ generations.

## Key Experimental Results

### Main Results

pass@3200 accuracy (%) on Ineq-Simp:

| Model | AM-GM Seed | AM-GM Type I | AM-GM Type II | Cauchy Seed | Cauchy Type I | Cauchy Type II |
|-------|-----------|-------------|--------------|------------|--------------|---------------|
| DeepSeek-Prover-V2-7B | **96.0** | **84.0** | **76.0** | **76.0** | **52.0** | **48.0** |
| Kimina-7B | 80.0 | 44.0 | 64.0 | 68.0 | 52.0 | 36.0 |
| STP | 64.0 | 44.0 | 40.0 | 44.0 | 24.0 | 28.0 |
| Goedel-Prover-SFT | 60.0 | 4.0 | 24.0 | 40.0 | 32.0 | 28.0 |
| DeepSeek-Prover-V1.5-RL | 60.0 | 4.0 | 24.0 | 24.0 | 12.0 | 12.0 |

**Key finding**: With the exception of DeepSeek-Prover-V2-7B, all models exhibit severe performance degradation from seed to Type I—Goedel-Prover drops precipitously from 60% to 4%.

**Model scaling experiments** (pass@32):

| Model | Seed | Type I | Type II |
|-------|------|--------|---------|
| DeepSeek-Prover-V2-7B | 66.2 | 47.0 | 42.1 |
| DeepSeek-Prover-V2-671B | 88.0 | 68.3 | 70.7 |
| Goedel-Prover-V2-8B | 69.0 | 35.0 | 49.0 |
| Goedel-Prover-V2-32B | 90.0 | 68.0 | 83.0 |

Although scaling up model size improves absolute performance, the gap between seed problems and their transformed variants persists.

### Ablation Study

**In-Context Learning (providing seed proofs)**:

| Model | AM-GM Type I | AM-GM Type II | Cauchy Type I | Cauchy Type II |
|-------|-------------|--------------|--------------|---------------|
| Qwen2.5-Coder-32B (128) | 28.0 | 40.0 | 48.0 | 20.0 |
| DeepSeek-R1-32B w/ thinking (128) | 16.0 | 44.0 | 84.0 | 40.0 |
| GPT-4o (16) | 12.0 | 40.0 | 56.0 | 16.0 |
| Claude 3.7 Sonnet (16) | 36.0 | 28.0 | 32.0 | 20.0 |

Even when complete proofs of the seed problems are provided in context, models still struggle to generalize to transformed variants. **The root cause of the compositional reasoning deficit is not a lack of knowledge, but a lack of compositional generalization ability.**

**Fine-tuning experiments**:

Synthetic data (~8k compilable samples) generated from AM-GM seed problems was used to fine-tune Goedel-Prover-SFT:
- **In-distribution (AM-GM Type I)**: accuracy improves substantially to 56%.
- **Out-of-distribution**: performance shows virtually no improvement, remaining near pre-fine-tuning levels.
- Performance on MiniF2F does not degrade, ruling out catastrophic forgetting.

### Key Findings

1. **Tactic monoculture**: Nearly all models rely heavily on `nlinarith` + `sq_nonneg` (sum-of-squares approach) and rarely attempt to invoke classical inequalities such as AM-GM or Cauchy–Schwarz directly.
2. **Natural language vs. formal reasoning gap**: Models frequently invoke correct high-level strategies (e.g., AM-GM) in natural-language annotations, but abandon these strategies in Lean code, reverting to brute-force algebraic manipulation.
3. **Comparison with O3**: OpenAI's O3 model can readily decompose and solve these compositional problems in natural-language mode, confirming that the bottleneck lies in the formal reasoning system rather than in mathematical understanding per se.
4. **Ineq-Mix harder than Ineq-Real**: Automatically generated compositional problems prove more challenging for models than real-world competition problems—a counterintuitive result that underscores the core difficulty of compositional reasoning.

## Highlights & Insights

1. **Elegant benchmark design**: The bottom-up construction methodology enables controlled, scalable difficulty and allows precise diagnosis of compositional reasoning as a specific capability.
2. **A deep underlying problem revealed**: Current formal provers do not simply lack sufficient intelligence; they lack the ability to compositionally apply existing knowledge—a sharp contrast with the core capacity of human mathematical reasoning.
3. **The O3–formalization gap**: Compositional problems that natural-language reasoning handles effortlessly become a primary bottleneck in the formal setting, suggesting that compositional reasoning patterns are underrepresented in formal training data.
4. **Fine-tuning cannot bridge the gap**: This is not merely a data scarcity issue; moderate-scale SFT can only learn specific patterns without achieving genuine generalization.

## Limitations & Future Work

1. The benchmark focuses exclusively on inequalities; the behavior of compositional reasoning in other mathematical domains—algebra, geometry, number theory—remains unexplored.
2. Seed problem difficulty is limited to introductory competition level; applicability to higher-difficulty problems has not been validated.
3. Newer tree-search-based methods such as HunyuanProver and BFS-Prover are not evaluated, as their code is not publicly available.
4. The random generation process in Ineq-Mix may introduce problems that are formally valid but mathematically trivial.

## Related Work & Insights

- **MiniF2F / ProofNet / PutnamBench**: Existing benchmarks emphasize overall difficulty rather than diagnostic evaluation of specific reasoning capabilities.
- **AlphaProof / DeepSeek-Prover-V2**: The strongest formal proving systems, yet they exhibit notable weaknesses in compositional reasoning.
- **AIPS / inequality solvers**: Targeting higher-difficulty problems, whereas this work exposes a qualitatively different dimension of failure by starting from simple problems.
- **Implication**: Future provers may require explicit decompose-and-compose reasoning modules, or curriculum learning that incorporates compositional reasoning during RL training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Examining formal theorem proving through the lens of compositional reasoning is a genuinely novel perspective.
- **Technical Depth**: ⭐⭐⭐⭐ — The benchmark design is rigorous, the transformation definitions are precise, and the ablation study is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 6+ provers across multiple inference budgets, with validation via ICL, fine-tuning, and model scaling.
- **Value**: ⭐⭐⭐⭐ — Provides the formal theorem proving community with a sustainably extensible diagnostic tool.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive figures, and well-motivated problem framing.
- **Overall**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [\[ICCV 2025\] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](../../ICCV2025/llm_evaluation/hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)
- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](../../ACL2026/llm_evaluation/humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)

</div>

<!-- RELATED:END -->
