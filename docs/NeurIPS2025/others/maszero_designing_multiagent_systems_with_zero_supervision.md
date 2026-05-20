---
title: >-
  [Paper Note] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision
description: >-
  [NeurIPS 2025 (SEA Workshop, Oral)][multi-agent system] MAS-ZERO is the first inference-time automatic MAS design framework. Through a meta-agent that iteratively designs, critiques…
tags:
  - "NeurIPS 2025 (SEA Workshop, Oral)"
  - "multi-agent system"
  - "automatic MAS design"
  - "meta-agent"
  - "zero supervision"
  - "inference-time optimization"
date: 2026-05-08
content_hash: 40ae8aa00630cf88
---

# MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision

**Conference**: NeurIPS 2025 (SEA Workshop, Oral)
**arXiv**: [2505.14996](https://arxiv.org/abs/2505.14996)  
**Code**: [https://github.com/SalesforceAIResearch/MAS-Zero](https://github.com/SalesforceAIResearch/MAS-Zero)  
**Area**: Other
**Keywords**: multi-agent system, automatic MAS design, meta-agent, zero supervision, inference-time optimization

## TL;DR
MAS-ZERO is the first inference-time automatic MAS design framework. Through a meta-agent that iteratively designs, critiques, and refines MAS configurations (including task decomposition and sub-MAS assignment), it requires no validation set or training, and outperforms both manual and automatic MAS baselines on reasoning (+16.69%), programming (+16.66%), and search agent (+5.45%) tasks while maintaining a Pareto-optimal accuracy–cost trade-off.

## Background & Motivation

**Background**: LLM multi-agent systems (MAS) leverage collaboration among multiple agents (debate, division of labor, verification) to tackle complex tasks that single models cannot handle effectively. Current MAS designs fall into two categories: manual design (Debate, Self-Refine, etc.) and automatic design (ADAS, AFlow, which search for optimal configurations using validation sets).

**Limitations of Prior Work**: (1) Manual MAS relies on human-defined roles and communication protocols, making it difficult to align with underlying LLM capabilities and adapt to new tasks. (2) Automatic MAS requires annotated validation sets for tuning and yields fixed architectures (one configuration for all problems), lacking per-instance adaptability. (3) Existing automatic methods cannot dynamically degrade to simpler MAS or a single agent (e.g., CoT) when a complex MAS is unhelpful, causing multiple baselines to underperform CoT on difficult tasks.

**Key Challenge**: Effective automatic MAS must simultaneously satisfy three conditions — (a) decompose complex problems and degrade to simpler systems when unnecessary, (b) autonomously learn LLM agent capabilities and design matching architectures, and (c) adapt per-problem at inference time rather than relying on a validation set. No existing method satisfies all three simultaneously.

**Goal**: Design a framework that requires no validation set and automatically tailors MAS configurations to each problem instance at inference time.

**Key Insight**: Introduce a meta-agent operating at the MAS level (rather than the agent level) to self-evolve through iterative meta-design (problem decomposition + sub-MAS assignment) and meta-feedback (solvability and completeness evaluation), with a final stage that selects the best answer from all candidate outputs (including those from simple building blocks).

**Core Idea**: The meta-agent transforms MAS design into an iterative inference-time optimization process — at each step it designs a MAS (in code form), executes it, refines based on solvability and completeness feedback, and ultimately selects the best answer from building block outputs and multi-round evolved candidates.

## Method

### Overall Architecture
MAS-ZERO proceeds in three stages: (1) **MAS-Init** — runs 4 building blocks (CoT, CoT-SC, Debate, Self-Refine) to collect initial candidate answers; (2) **MAS-Evolve** — the meta-agent iterates for 5 rounds, each time designing a MAS (decomposing the problem + assigning sub-MAS), executing it, evaluating feedback, and storing results in an experience library; (3) **MAS-Verify** — selects the most reliable answer from 9 candidates (4 building blocks + 5 evolved rounds).

### Key Designs

1. **Meta-Design (Task Decomposition + Sub-MAS Assignment)**:

    - Function: Decompose the original problem into manageable sub-tasks and assign a sub-MAS composed of building blocks to each.
    - Mechanism: The meta-agent is prompted to (a) decompose the problem into sub-tasks that are "manageable yet interdependent," and (b) specify a sub-MAS for each sub-task (a single block such as CoT, a chain such as CoT→Self-Refine, or a parallel set such as {CoT ∥ Debate}). Designs are expressed as executable Python code.
    - Design Motivation: The meta-agent may modify connections and parameters between blocks (e.g., temperature, debate rounds) but may not invent new blocks — striking a balance between exploration and reliability.

2. **Meta-Feedback (Solvability + Completeness Evaluation)**:

    - Function: Evaluate the quality of the MAS produced by meta-design and generate improvement feedback.
    - Mechanism: The MAS is executed to obtain two levels of intermediate outputs (sub-task level and agent level), which are evaluated on two criteria:
        - **Solvability**: Is each sub-task independently and completely solved by its sub-MAS? If not solvable, further decomposition or sub-MAS replacement is suggested.
        - **Completeness**: Do all sub-tasks collectively cover all necessary information from the original problem? If gaps exist, modifications to the decomposition strategy are suggested.
    - Design Motivation: Unlike validation-set methods that only observe final answer correctness, the solvability + completeness criteria provide richer process-level feedback.

3. **Experience Library + MAS-Verify**:

    - Experience Library: Each round's MAS design, intermediate outputs, and feedback are stored in an experience library and used as context for subsequent rounds.
    - MAS-Verify: (a) Candidate answers are ranked by final answer frequency (majority vote heuristic); (b) invalid answers are filtered out; (c) the meta-agent selects the best answer from the ranked candidates — including building block outputs — enabling dynamic degradation.

### Loss & Training
Entirely training-free, zero supervision. All operations are implemented via prompting, requiring only black-box LLM access. The same prompt templates are used across all tasks.

## Key Experimental Results

### Main Results (GPT-4o as backbone)

| Method | AIME24 | GPQA | Average | vs MAS-ZERO |
|---|---|---|---|---|
| CoT | 8.33 | 45.78 | 27.06 | -14.91 |
| CoT-SC | 16.67 | 43.37 | 30.02 | -11.95 |
| Debate | 4.17 | 46.99 | 25.58 | -16.39 |
| AFlow (SOTA auto MAS) | 20.83 | 46.99 | 33.91 | -8.05 |
| ADAS | × | 45.20 | — | — |
| **MAS-ZERO** | **33.33** | **50.60** | **41.97** | — |

ADAS completely fails on AIME24 (0%, indicating unreliable validation set). MAS-ZERO surpasses AFlow by 8.05% on average.

### Ablation Study

| Configuration | AIME24 | GPQA | Average | Impact |
|---|---|---|---|---|
| MAS-ZERO | 33.33 | 50.60 | 41.97 | — |
| - MAS-Init | 12.50 | 48.43 | 30.46 | -11.50 (loses building block degradation) |
| - MAS-Evolve | 20.00 | 48.73 | 34.37 | -7.60 |
| - meta-design (no decomposition) | 20.83 | 45.18 | 33.01 | -8.96 |
| - meta-feedback | 25.00 | 42.17 | 33.59 | -8.38 |
| → ensemble feedback | 16.67 | 46.88 | 31.77 | -10.19 (worse) |
| - MAS-Verify (use last round only) | 6.70 | 33.83 | 20.27 | -21.70 (largest drop) |

MAS-Verify is the most critical component — removing it causes a 21.70% performance collapse. An oracle verifier can push GPQA to near 95%, indicating that verification is the primary bottleneck.

### Key Findings
- **Simple strategies are sometimes strongest**: CoT and CoT-SC outperform all complex MAS methods (including automatic ones) on certain benchmarks. MAS-ZERO is the only automatic MAS capable of dynamically degrading to these simple strategies.
- **ADAS completely fails on AIME24**: Even with a validation set, validation-set-based methods can be entirely unreliable on difficult data distributions.
- **Ensemble feedback underperforms single-round feedback**: A counterintuitive finding — the current straightforward meta-feedback is already sufficiently strong, and naive ensembling introduces noise.
- **Heterogeneous agent configurations show potential but are limited**: Using o3-mini as the meta-agent and GPT-4o as individual agents yields limited gains, as overall performance is bottlenecked by individual agent capability.
- **Cost-efficiency is Pareto-optimal**: MAS-ZERO lies on the Pareto frontier in the accuracy–cost scatter plots (Figure 1) across all three benchmarks.

## Highlights & Insights
- **Inference-time MAS design is a new paradigm**: Rather than relying on validation sets or producing fixed architectures, MAS-ZERO tailors a MAS to each problem instance — a natural extension of test-time compute scaling to the multi-agent domain.
- **Code-as-MAS representation**: Expressing MAS designs as executable Python code allows the meta-agent's designs to be directly executed and evaluated, greatly reducing the conversion overhead from "design" to "execution."
- **Dynamic degradation capability**: This is MAS-ZERO's most distinctive advantage over other automatic MAS methods — automatically falling back to CoT/CoT-SC when complex MAS provides no benefit, avoiding unnecessary complexity.

## Limitations & Future Work
- The meta-agent requires a sufficiently capable LLM (GPT-4o level); weaker LLMs (7B–20B) cannot reliably perform meta-level code generation and feedback.
- Inference-time token costs are relatively high (9 candidate answers = 4 building blocks + 5 evolved rounds); although Pareto-optimal, the absolute cost is non-trivial.
- The failure of ensemble meta-feedback indicates that the current feedback mechanism still has room for improvement.
- Gains on search agent tasks are modest (+5.45%), as search errors anchored by retrieved content are difficult for meta-verification to identify.

## Related Work & Insights
- **vs AFlow / ADAS**: AFlow and ADAS use MCTS / rejection sampling to search for optimal MAS on validation sets, yielding fixed architectures. MAS-ZERO designs per-problem at inference time with no validation set. AFlow achieves 20.83% on AIME24; MAS-ZERO achieves 33.33%.
- **vs GiGPO (2505.10978)**: GiGPO addresses step-level credit assignment in agent RL training; MAS-ZERO addresses automatic design of agent configurations. The two are complementary — MAS-ZERO-designed MAS can be further trained with GiGPO.
- **vs DRIFT (2506.12104)**: DRIFT protects agents against external injection attacks, while MAS-ZERO designs how agents collaborate. The two address different layers of agent systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First inference-time automatic MAS design framework; dual-dimensional solvability/completeness feedback; unique dynamic degradation capability.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three LLM backbones, three domains (reasoning/programming/search), eleven baselines, comprehensive ablations, Pareto analysis, and standard deviation reporting.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1–3 present conceptual comparisons and Pareto plots with exceptional clarity; framework description is well-organized.
- Value: ⭐⭐⭐⭐⭐ Reveals important insights — "simple strategies are sometimes strongest" and "validation-set methods can be unreliable" — with open-source code offering direct reuse value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Optimism Without Regularization: Constant Regret in Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](../../AAAI2026/others/designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/others/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)

</div>

<!-- RELATED:END -->
