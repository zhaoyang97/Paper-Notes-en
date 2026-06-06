---
title: >-
  [Paper Note] Efficient PRM Training Data Synthesis via Formal Verification
description: >-
  [ACL 2026][LLM Reasoning][PRM] This paper proposes FoVer, a framework that leverages formal verification tools (Z3 and Isabelle) to automatically annotate correctness labels for step-level reasoning chains in formal reas…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "PRM"
  - "Formal Verification"
  - "Step-level Labels"
  - "Z3"
  - "Isabelle"
date: 2026-05-08
content_hash: 7dc4105a36d632d5
---

# Efficient PRM Training Data Synthesis via Formal Verification

**Conference**: ACL 2026  
**arXiv**: [2505.15960](https://arxiv.org/abs/2505.15960)  
**Code**: [GitHub](https://github.com/psunlpgroup/FoVer)  
**Area**: LLM Reasoning  
**Keywords**: PRM, Formal Verification, Step-level Labels, Z3, Isabelle

## TL;DR

This paper proposes FoVer, a framework that leverages formal verification tools (Z3 and Isabelle) to automatically annotate correctness labels for step-level reasoning chains in formal reasoning tasks. By generating the FoVer-40K training set and fine-tuning PRMs, it demonstrates formal-to-informal transfer capabilities and cross-task generalization across 12 reasoning benchmarks.

## Background & Motivation

**Background**: Process Reward Models (PRM) enhance LLM reasoning by providing process supervision for intermediate steps, which has become a key direction for improving reasoning performance. PRMs are typically fine-tuned from LLMs to perform binary classification on the correctness of reasoning steps.

**Limitations of Prior Work**: (1) Manual step-level annotation is extremely expensive and suffers from low inter-annotator agreement (Zheng et al. 2025 discarded 30% of annotated data); (2) Monte Carlo roll-outs (Math-Shepherd) require multiple LLM calls to label each step, resulting in high computational costs and noisy labels—estimating intermediate step correctness via final answer frequency is inherently imprecise; (3) Perturbation-based methods introduce unnatural human-induced errors; (4) Annotation using stronger LLMs amounts to simple capability distillation.

**Key Challenge**: PRMs require accurate step-level labels, but existing annotation methods are either expensive (manual), imprecise (Monte Carlo), or unnatural (perturbation)—is there an efficient yet precise way to annotate?

**Goal**: To utilize formal verification tools to provide efficient and precise step-level annotations for formal reasoning tasks, and to verify whether PRMs trained on formal tasks can transfer to informal natural language reasoning tasks.

**Key Insight**: Each step of formal reasoning tasks (logical reasoning, theorem proving) can be precisely verified for correctness using formal verification tools (e.g., SAT solvers like Z3, theorem provers like Isabelle)—this serves as a zero-cost source of perfect annotations.

**Core Idea**: Formal verification tools can transform step-level annotation from "approximate estimation" into "precise judgment"—the PRM capabilities acquired on formal tasks can be transferred to informal reasoning tasks, achieving formal-to-informal transfer and cross-task generalization.

## Method

### Overall Architecture

A two-stage framework: Stage 1 involves using an LLM to generate formal reasoning steps (guided by few-shot formatting) → using formal verification tools to annotate the correctness of each step. Stage 2 involves fine-tuning the LLM with the annotated data to perform step-level binary classification (outputting "correct"/"incorrect") to obtain FoVer-PRM.

### Key Designs

1.  **Step-level Formal Verification**:

    - **Function**: Provides precise correctness labels for each step in a reasoning chain.
    - **Mechanism**: For logical reasoning tasks (FLDx2), each reasoning step is a logical entailment, which can be verified by negating it and checking satisfiability—unsatisfiability implies the step is correct. For example, to verify $(A \to B) \land A \to B$, it is converted to $(\neg A \lor B) \land A \land \neg B$ and checked with Z3. For theorem proving tasks (GSM8K math problems), Isabelle is used to verify each proof step incrementally.
    - **Design Motivation**: Formal verification tools are "perfect annotators"—their labels are deterministically correct, free from noise or subjectivity. The key insight is that these tools can perform step-level verification as well as solution-level verification.

2.  **FoVer-40K Dataset Construction**:

    - **Function**: Provides a large-scale, precisely annotated dataset for PRM training.
    - **Mechanism**: On FLDx2 (formal logic, multi-step first-order logic derivation) and GSM8K-level math problems, Llama 3.1 8B and Qwen 2.5 7B generate reasoning chains, which are then annotated with step-level labels using Z3 and Isabelle. A total of 40K steps were collected, with a 50% correct/50% incorrect balance. FLDx2 was chosen for its high diversity of derivation rules.
    - **Design Motivation**: Dataset selection emphasizes diversity and simplicity—FLDx2 provides logical diversity, while GSM8K provides mathematical reasoning, both of which can be precisely verified in their formal versions.

3.  **Formal-to-Informal Transfer Evaluation**:

    - **Function**: Verifies whether PRMs trained on formal tasks can generalize to natural language reasoning.
    - **Mechanism**: FoVer-PRM is evaluated on 12 benchmarks—6 are informal variants of the training tasks (LogicNLI, AQuA-RAT, AIME, etc.) + 6 are reasoning benchmarks significantly different from the training tasks (HANS NLI, MMLU, BBH). Evaluation uses the Best-of-K criterion.
    - **Design Motivation**: The practical utility of PRMs lies in improving natural language reasoning, thus formal-to-informal transfer must be validated.

### Loss & Training

Cross-entropy classification loss is used to fine-tune the LLM to output "correct" or "incorrect" for each reasoning step. Base models used are Llama 3.1 8B and Qwen 2.5 7B.

## Key Experimental Results

### Main Results

**Best-of-7 Reasoning Performance (12 Benchmarks, 5 Task Categories)**

| PRM | Logic | Math | NLI | MMLU | BBH | Avg |
|-----|------|------|-----|------|-----|------|
| Llama 3.1 8B (Baseline) | 48.9 | Baseline | Baseline | Baseline | Baseline | Baseline |
| **FoVer-Llama3.1-8B** | **50.6** | Gain | Gain | Gain | Gain | **Above Baseline** |
| Qwen 2.5 7B (Baseline) | Baseline | Baseline | Baseline | Baseline | Baseline | Baseline |
| **FoVer-Qwen2.5-7B** | Gain | Gain | Gain | Gain | Gain | **Above Baseline** |

### Ablation Study

**Comparison with Existing PRM Training Methods**

| Method | Annotation Cost | Label Quality | Avg Performance |
|------|---------|---------|---------|
| Human Annotation | Extremely High | High (but consistency issues) | High |
| Monte Carlo roll-outs | High (multiple LLM calls) | Medium (Noise) | Medium-High |
| Perturbation Methods | Medium | Low (Unnatural errors) | Medium |
| **FoVer** | **Extremely Low (Automated)** | **Perfect (Deterministic)** | **Competitive** |

### Key Findings

- FoVer-PRM outperforms non-fine-tuned baseline PRMs across all 5 task categories—formal training data indeed improves natural language reasoning.
- The most surprising gains are in NLI and BBH—tasks that differ significantly from the formal logic and math problems in the training data, demonstrating cross-task generalization.
- FoVer labels are deterministically correct, whereas Monte Carlo roll-outs are noisy—even with a smaller data scale (40K steps), high-quality labels yield competitive performance.
- Formal-to-informal transfer is the core finding of this paper—step-level verification capability seems to possess task-agnostic universality.

## Highlights & Insights

- Using formal verification tools as "perfect annotators" is an elegant idea—reducing annotation costs to near zero.
- The success of formal-to-informal transfer challenges the assumption that PRMs must be trained on in-distribution data.
- The modular design of the FoVer framework (replaceable verification tools) allows for extension to more formal tasks.

## Limitations & Future Work

- FoVer-40K only includes formal logic and GSM8K-level math, failing to cover more complex formal reasoning.
- Step-level formal verification requires LLMs to generate reasoning chains in a format compatible with the tools; non-compliant generations are discarded.
- The possibility of combining FoVer with Monte Carlo roll-outs has not been explored.
- Theoretical explanation for the transfer capability remains insufficient—why does training on formal logic improve BBH performance?

## Related Work & Insights

- **vs Math-Shepherd**: Math-Shepherd uses Monte Carlo to estimate labels; FoVer uses formal verification to determine them—improving quality but narrowing applicability.
- **vs PRM800K (Lightman 2024)**: Relies on manual annotation, whereas FoVer is fully automated with more precise labels.
- **vs Strong LLM Distillation**: Using GPT-4 for labeling is essentially capability distillation; FoVer does not rely on any stronger models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Using formal verification tools for step-level PRM annotation is a fresh idea; the formal-to-informal transfer finding is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 12 benchmarks, though data scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition; examples of formal verification are intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and precise data generation solution for PRM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] An Information-Theoretic Criterion for Efficient Data Synthesis](../../ICML2026/llm_reasoning/an_information-theoretic_criterion_for_efficient_data_synthesis.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
