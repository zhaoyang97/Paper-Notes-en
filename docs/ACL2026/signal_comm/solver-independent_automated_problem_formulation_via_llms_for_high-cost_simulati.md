---
title: >-
  [Paper Note] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design
description: >-
  [ACL 2026][Signal & Communication][Automated Problem Formulation] This paper proposes APF (Automated Problem Formulation), a solver-independent framework that uses LLMs to translate engineers' natural language design requirements into executable mathematical optimization models. Through innovative data generation and test instance annotation pipelines, APF overcomes the difficulty of using solver feedback for data filtering in high-cost simulation scenarios, significantly outperforming existing methods on antenna design tasks.
tags:
  - ACL 2026
  - "Signal & Communication"
  - Automated Problem Formulation
  - High-Cost Simulation
  - LLM Fine-tuning
  - Solver-Independent Evaluation
  - Antenna Design
date: 2025-04-17
content_hash: d933d87cac11527f
---

# Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design

**Conference**: ACL 2026  
**arXiv**: [2512.18682](https://arxiv.org/abs/2512.18682)  
**Code**: N/A  
**Area**: Signal & Communication  
**Keywords**: Automated Problem Formulation, High-Cost Simulation, LLM Fine-tuning, Solver-Independent Evaluation, Antenna Design

## TL;DR

This paper proposes APF (Automated Problem Formulation), a solver-independent framework that uses LLMs to translate engineers' natural language design requirements into executable mathematical optimization models. Through innovative data generation and test instance annotation pipelines, APF overcomes the difficulty of using solver feedback for data filtering in high-cost simulation scenarios, significantly outperforming existing methods on antenna design tasks.

## Background & Motivation

**State of the Field**: High-cost simulation-driven design is prevalent in antenna, aerospace, microelectronics, and robotics domains. The core task is optimizing design parameters so that performance distributions (e.g., frequency-domain radiation efficiency curves) meet design requirements. Since design requirements are typically provided as unstructured natural language, formalizing them into executable mathematical models is the optimization bottleneck.

**Limitations of Prior Work**: (1) Prompt-based methods (e.g., Chain-of-Experts, OptiMUS) struggle to accurately identify objectives and constraints when facing ambiguous or domain-knowledge-dependent natural language requirements; (2) Fine-tuning methods (e.g., ORLM, LLMOPT, SIRL) improve performance but rely on solver feedback for data filtering, which is unavailable in high-cost simulation scenarios; (3) Existing methods primarily focus on linear programming, integer programming, and other operations research problems, differing significantly from high-cost simulation-driven design in problem description and evaluation cost.

**Root Cause**: Fine-tuning LLMs requires high-quality training data, but in high-cost simulation scenarios, verifying the correctness of generated formulations requires expensive physical simulations (e.g., electromagnetic full-wave simulation), making large-scale data quality filtering infeasible. The solver feedback mechanism relied upon by existing fine-tuning methods fails in this scenario.

**Paper Goals**: Develop an automated problem formulation framework that does not depend on solver feedback, capable of automatically generating high-quality training data and fine-tuning LLMs to accurately translate natural language design requirements into executable mathematical optimization models.

**Starting Point**: Introduce test instances as a bridge — by having LLMs rank-annotate test instances, the "semantic alignment between natural language requirements and mathematical formulas" is transformed into a "ranking consistency problem," thereby bypassing expensive solver verification.

**Core Idea**: Through a three-stage pipeline of data generation + test instance annotation + ranking consistency evaluation, build high-quality fine-tuning datasets without invoking expensive solvers, enabling 7B/8B open-source models to match or exceed the modeling accuracy of GPT-4o and other large models.

## Method

### Overall Architecture

APF consists of four modules: (1) Data generation — extracting design requirements from historical simulation data and using LLMs to generate corresponding mathematical formulas; (2) Test instance annotation — LLM-based listwise ranking of test instances to establish reference rankings; (3) Data evaluation and selection — comparing ranking consistency between generated formula execution rankings and reference rankings to filter high-quality samples; (4) Supervised fine-tuning — fine-tuning open-source LLMs on filtered high-quality data.

### Key Designs

1. **Unified Abstract Representation and Data Generation**:

    - Function: Standardize unstructured industrial specifications into processable forms and generate diverse training data
    - Mechanism: Each design requirement is formalized as a structured tuple $r = (\mathcal{Z}, M, \mathcal{C})$, where $\mathcal{Z}$ is the evaluation variable subregion, $M: z \in \mathcal{Z} \to \mathbb{R}$ is the metric function, and $\mathcal{C}$ specifies design intent (e.g., threshold constraint $\min_{z \in \mathcal{Z}} M(z) \geq 1.5$ or optimization objective). Extracting real solvable requirement sets $\mathcal{R} = \{r_1, r_2, \ldots, r_n\}$ from historical simulation records, with LLM generating corresponding mathematical formulas. Data augmentation includes semantic paraphrasing ($v$ equivalent variants) and order permutation (shuffling requirement order)
    - Design Motivation: Extraction from historical simulations guarantees physical feasibility of requirements. Semantic paraphrasing enhances robustness to diverse expressions, and order permutation prevents the model from relying on spurious positional cues

2. **Solver-Independent Evaluation Module**:

    - Function: Evaluate the quality of generated formulas without invoking expensive solvers
    - Mechanism: Test instance set $\mathcal{I}$ is introduced as a bridge. LLMs generate reference rankings $\pi_{\text{LLM}} = \arg\max_\pi P_\theta(\pi | \mathcal{P})$ using a listwise strategy, with prompts containing task instructions, expert examples, instance data tables, and requirement queries. Quality score is defined as the Spearman correlation between execution ranking and reference ranking: $S(E) = \rho(\pi_E, \pi_{\text{LLM}})$, retaining only strongly correlated ($> 0.7$) samples
    - Design Motivation: Directly evaluating semantic alignment between natural language and mathematical formulas is computationally infeasible. Test instances transform the problem into quantifiable ranking consistency comparison. Listwise strategy is two orders of magnitude more efficient than pairwise (1 call vs 105 calls) with comparable ranking quality ($\rho$: 0.8643 vs 0.8536)

3. **Alignment Metrics and Fine-tuning**:

    - Function: Provide comprehensive formula quality evaluation and fine-tune on high-quality data
    - Mechanism: Alignment score consists of objective function alignment and constraint alignment: $A(E) = \alpha A_{\text{obj}}(E) + (1-\alpha) A_{\text{con}}(E)$. Objective function evaluated using Spearman rank correlation: $A_{\text{obj}} = \frac{1}{n_1} \sum_{e_i \in E_{\text{obj}}} \rho(\hat{\pi}_i, \pi^*)$; constraints evaluated using classification accuracy: $A_{\text{con}} = \frac{1}{n_2} \sum_{e_j \in E_{\text{con}}} (1 - \frac{1}{m} \|\hat{\mathbf{y}}_j - \mathbf{y}^*\|_1)$. $\alpha = 0.5$ balances both
    - Design Motivation: Objective functions focus on ranking correctness (relative order), constraint functions focus on feasibility judgment correctness (absolute right/wrong) — both dimensions are indispensable

### Loss & Training

Standard SFT on the filtered high-quality dataset $\mathcal{D}_{\text{HQ}}$ (7,879 samples) for open-source LLMs. 2,300 design requirement sets used, with 300 as test set (zero overlap) and 2,000 for training. Selection threshold 0.7 (strong correlation lower bound). Data generation uses GPT-4o; test instance annotation uses GPT-5 and other strong LLM judges.

## Key Experimental Results

### Main Results

**Overall Formula Quality Comparison**

| Method | $A_{\text{obj}}$ | $A_{\text{con}}$ | $A$ (Overall) |
|------|------|------|------|
| GPT-4o | 0.6055 | 0.7075 | 0.6651 |
| DeepSeek-V3 | 0.7404 | 0.7690 | 0.7518 |
| Claude-sonnet-4.5 | 0.8023 | 0.7880 | 0.7923 |
| Chain-of-Experts | 0.7426 | 0.7453 | 0.7252 |
| OptiMUS | 0.6341 | 0.6986 | 0.6687 |
| LLAMA3.1-8B (Original) | -0.0453 | 0.5029 | 0.2248 |
| **APF + LLAMA3.1-8B** | **0.8012** | **0.7969** | **0.7976** |
| **APF + Qwen2.5-7B** | 0.7990 | 0.7959 | 0.7961 |
| **APF + Mistral-7B** | 0.7974 | 0.7883 | 0.7918 |

### Ablation Study

| Config | $A_{\text{obj}}$ | $A_{\text{con}}$ | $A$ |
|------|------|------|------|
| w/o Augmentation | 0.7656 | 0.7555 | 0.7553 |
| w/o Selection | 0.7603 | 0.7800 | 0.7653 |
| APF (full model) | **0.8009** | **0.7971** | **0.7976** |

**Evaluation Method Comparison**

| Method | Spearman $\rho$ | LLM Calls | Time (s) | Cost ($) |
|------|------|------|------|------|
| Listwise (ours) | 0.8643 | 1 | 97.66 | 0.02 |
| Pairwise | 0.8536 | 105 | 2544.8 | 0.47 |

### Key Findings

- After APF fine-tuning, LLAMA3.1-8B improves from 0.2248 to 0.7976 (+256%), going from nearly unusable to surpassing GPT-4o and Claude-sonnet-4.5
- Three 7B/8B models show highly consistent performance after APF fine-tuning (0.7918–0.7976), demonstrating the universal effectiveness of high-quality data
- LLM judge rankings are highly consistent with human rankings (GPT-5: $\rho = 0.8316$), validating the reliability of solver-independent evaluation
- Listwise evaluation matches pairwise ranking quality but is 26× faster and 23× cheaper
- Selection threshold is highly stable across the 0.6–0.8 range, demonstrating framework insensitivity to hyperparameters
- In actual antenna design, the optimization model driven by APF satisfies all frequency band requirements, while other methods fail on passband and high radiation nulls

## Highlights & Insights

- The "test instances as bridge" approach cleverly transforms the semantic alignment problem into a quantifiable ranking consistency problem, bypassing expensive solver verification
- The listwise vs pairwise efficiency comparison is impressive: comparable quality, 26× speedup, 23× cost reduction
- Demonstrates that on domain-specific tasks, high-quality data + small models can match or even exceed the zero-shot capabilities of general-purpose large models
- Extracting requirements from historical simulation records ensures physical feasibility — this data-driven approach is more reliable than random synthesis

## Limitations & Future Work

- Currently validated only on antenna design; cross-domain generalization to aerodynamics, structural optimization, and other engineering fields needs verification
- Solver-independent evaluation relies on constructing prompts with detailed test instances, limited by LLM context windows
- Data generation depends on strong models like GPT-4o, introducing additional cost and quality dependency
- Only validated on 7B/8B scale models; effects of larger or smaller models are unexplored

## Related Work & Insights

- **vs Chain-of-Experts/OptiMUS (Prompt-based)**: Prompt methods have insufficient accuracy on ambiguous requirements (A: 0.6687–0.7252); APF achieves more accurate requirement understanding through fine-tuning (A: 0.7976)
- **vs ORLM/SIRL (Fine-tuning-based)**: These methods rely on solver feedback for data filtering, which is infeasible in high-cost simulation scenarios; APF achieves solver-independent evaluation through LLM ranking
- **vs GPT-4o/DeepSeek-V3 (Zero-shot)**: 7B fine-tuned models surpass the zero-shot performance of these large models, demonstrating the enormous value of domain fine-tuning

## Rating

- Novelty: ⭐⭐⭐⭐ Test instance bridge and solver-independent evaluation are novel, transforming semantic alignment into ranking consistency
- Experimental Thoroughness: ⭐⭐⭐ Antenna design case validation is thorough, but only a single domain; ablation and sensitivity analyses are complete
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, method motivation is explicit, figures are professional
- Value: ⭐⭐⭐⭐ Provides a practical framework for automated modeling in high-cost simulation domains with broad industrial application prospects

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](../../ICLR2026/signal_comm/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](../../CVPR2026/signal_comm/chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[ACL 2026\] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)
- [\[ICLR 2026\] FASA: Frequency-Aware Sparse Attention](../../ICLR2026/signal_comm/fasa_frequency-aware_sparse_attention.md)

<!-- RELATED:END -->
