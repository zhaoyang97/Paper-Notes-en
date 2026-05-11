---
title: >-
  [Paper Note] Efficient PRM Training Data Synthesis via Formal Verification
description: >-
  [ACL 2026][LLM Reasoning][PRM] This paper proposes FoVer, a framework that leverages formal verification tools (Z3 and Isabelle) to automatically annotate step-level correctness labels for reasoning chains in formal reas…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "PRM"
  - "formal verification"
  - "step-level labels"
  - "Z3"
  - "Isabelle"
date: 2026-05-08
content_hash: da98512eb40b3aaf
---

# Efficient PRM Training Data Synthesis via Formal Verification

**Conference**: ACL 2026
**arXiv**: [2505.15960](https://arxiv.org/abs/2505.15960)
**Code**: [GitHub](https://github.com/psunlpgroup/FoVer)
**Area**: LLM Reasoning
**Keywords**: PRM, formal verification, step-level labels, Z3, Isabelle

## TL;DR

This paper proposes FoVer, a framework that leverages formal verification tools (Z3 and Isabelle) to automatically annotate step-level correctness labels for reasoning chains in formal reasoning tasks. It constructs the FoVer-40K training set and fine-tunes a PRM, demonstrating formal-to-informal transfer capability and cross-task generalization across 12 reasoning benchmarks.

## Background & Motivation

**State of the Field**: Process Reward Models (PRMs) enhance LLM reasoning by providing process supervision over intermediate reasoning steps and have become an important direction for improving reasoning performance. PRMs are typically obtained by fine-tuning LLMs to perform binary classification on the correctness of reasoning steps.

**Limitations of Prior Work**: (1) Manual annotation of step-level labels is prohibitively expensive, with low inter-annotator agreement (Zheng et al. 2025 discarded 30% of annotated data); (2) Monte Carlo roll-outs (Math-Shepherd) require multiple LLM calls per step, incurring high computational cost and noisy labels—estimating intermediate step correctness via the frequency of reaching the correct final answer is inherently imprecise; (3) perturbation-based methods introduce artificially unnatural errors; (4) annotation by stronger LLMs amounts to capability distillation.

**Root Cause**: PRMs require accurate step-level labels, yet existing annotation methods are either expensive (human), imprecise (Monte Carlo), or unnatural (perturbation)—can an annotation approach that is both efficient and precise be found?

**Paper Goals**: To leverage formal verification tools for efficient and precise step-level annotation of formal reasoning tasks, and to verify whether PRMs trained on formal tasks can transfer to informal natural language reasoning tasks.

**Starting Point**: Each step in formal reasoning tasks (logical reasoning, theorem proving) can be precisely verified for correctness by formal verification tools (e.g., SAT solver Z3, theorem prover Isabelle)—providing a zero-cost source of perfect labels.

**Core Idea**: Formal verification tools can transform step-level annotation from "approximate estimation" to "exact determination." PRM capabilities acquired on formal tasks can transfer to informal reasoning tasks, enabling formal-to-informal transfer and cross-task generalization.

## Method

### Overall Architecture

A two-stage framework: Stage 1 prompts an LLM to generate formal reasoning steps (few-shot guided format) → formal verification tools annotate the correctness of each step. Stage 2 uses the annotated data to fine-tune an LLM for step-level binary classification (outputting "correct"/"incorrect"), yielding FoVer-PRM.

### Key Designs

1. **Step-Level Formal Verification**:

    - **Function**: Provides precise correctness labels for each step of a reasoning chain.
    - **Mechanism**: For logical reasoning tasks (FLDx2), each reasoning step is a logical entailment that can be verified by negating it and checking satisfiability—unsatisfiability indicates the step is correct. For example, to verify $(A \to B) \land A \to B$, it is converted to $(\neg A \lor B) \land A \land \neg B$ and checked for satisfiability using Z3. For theorem proving tasks (GSM8K math problems), Isabelle is used to verify each proof step incrementally.
    - **Design Motivation**: Formal verification tools serve as "perfect annotators"—their labels are deterministically correct, free from noise or subjectivity. The key insight is that these tools can perform not only solution-level verification but also step-level verification.

2. **FoVer-40K Dataset Construction**:

    - **Function**: Provides large-scale, precisely annotated PRM training data.
    - **Mechanism**: On FLDx2 (formal logic, multi-step first-order logic derivations) and GSM8K-level math problems, Llama 3.1 8B and Qwen 2.5 7B are used to generate reasoning chains, with Z3 and Isabelle annotating step-level labels. The dataset totals 40K steps with a 50%/50% correct/incorrect split. FLDx2 is selected for its highest diversity of derivation rules.
    - **Design Motivation**: Dataset selection emphasizes diversity and conciseness—FLDx2 provides logical diversity, GSM8K provides mathematical reasoning, and the formal versions of both can be precisely verified by tools.

3. **Formal-to-Informal Transfer Evaluation**:

    - **Function**: Verifies whether PRMs trained on formal tasks can generalize to natural language reasoning.
    - **Mechanism**: FoVer-PRM is evaluated on 12 benchmarks—6 informal variants of training tasks (LogicNLI, AQuA-RAT, AIME, and other math/logic benchmarks) and 6 reasoning benchmarks substantially different from the training tasks (HANS NLI, MMLU, BBH). The standard Best-of-K evaluation protocol is adopted.
    - **Design Motivation**: The practical utility of PRMs lies in improving natural language reasoning; therefore, it is essential to verify whether formal training transfers.

### Loss & Training

Cross-entropy classification loss; the LLM is fine-tuned to output "correct" or "incorrect" for each reasoning step. Base models are Llama 3.1 8B and Qwen 2.5 7B.

## Key Experimental Results

### Main Results

**Best-of-7 Reasoning Performance (12 Benchmarks, 5 Task Categories)**

| PRM | Logic | Math | NLI | MMLU | BBH | Average |
|-----|-------|------|-----|------|-----|---------|
| Llama 3.1 8B (baseline) | 48.9 | baseline | baseline | baseline | baseline | baseline |
| **FoVer-Llama3.1-8B** | **50.6** | gain | gain | gain | gain | **above baseline** |
| Qwen 2.5 7B (baseline) | baseline | baseline | baseline | baseline | baseline | baseline |
| **FoVer-Qwen2.5-7B** | gain | gain | gain | gain | gain | **above baseline** |

### Ablation Study

**Comparison with Existing PRM Training Approaches**

| Method | Annotation Cost | Label Quality | Average Performance |
|--------|----------------|---------------|---------------------|
| Human annotation | Very high | High (but consistency issues) | High |
| Monte Carlo roll-outs | High (multiple LLM calls) | Medium (noisy) | Medium–High |
| Perturbation-based | Medium | Low (unnatural errors) | Medium |
| **FoVer** | **Very low (automated)** | **Perfect (deterministic)** | **Competitive** |

### Key Findings

- FoVer-PRM outperforms the untuned baseline PRM across all 5 task categories, confirming that formal training data genuinely improves natural language reasoning.
- The most surprising gains are observed on NLI and BBH—tasks substantially different from the formal logic and math training data—demonstrating cross-task generalization.
- FoVer labels are deterministically correct, whereas Monte Carlo roll-outs are noisy; even at a modest data scale (40K steps), high-quality labels yield competitive performance.
- Formal-to-informal transfer is the central finding of this paper—step-level verification capability appears to possess task-agnostic generality.

## Highlights & Insights

- Using formal verification tools as "perfect annotators" is an elegant approach that reduces annotation cost to near zero.
- The success of formal-to-informal transfer challenges the assumption that PRMs must be trained on in-distribution data.
- The modular design of the FoVer framework (verification tools are interchangeable) makes it extensible to a broader range of formal reasoning tasks.

## Limitations & Future Work

- FoVer-40K covers only formal logic and GSM8K-level math problems, leaving more complex formal reasoning tasks unaddressed.
- Step-level formal verification requires LLMs to generate reasoning chains in tool-compatible formats; malformed outputs are discarded.
- The possibility of combining FoVer with Monte Carlo roll-outs remains unexplored.
- The theoretical explanation for transfer capability is insufficient—why does training on formal logic improve BBH performance?

## Related Work & Insights

- **vs. Math-Shepherd**: Math-Shepherd estimates labels via Monte Carlo sampling; FoVer determines labels via formal verification—yielding higher quality but narrower applicability.
- **vs. PRM800K (Lightman 2024)**: Relies on human annotation; FoVer is fully automated and produces more precise labels.
- **vs. Strong LLM Distillation**: Annotating with GPT-4 is essentially capability distillation; FoVer does not depend on any stronger model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Using formal verification tools for step-level PRM annotation is an entirely novel approach; the formal-to-informal transfer finding is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 12 benchmarks, though the data scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear; the formal verification examples are intuitive and accessible.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and precise data generation solution for PRM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
