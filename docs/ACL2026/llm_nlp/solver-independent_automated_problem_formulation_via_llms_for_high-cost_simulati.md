---
title: >-
  [Paper Note] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design
description: >-
  [ACL 2026][LLM/NLP][Automated problem modeling] This paper proposes APF (Automated Problem Formulation), a solver-independent framework that utilizes LLMs to transform natural language design requirements into executable…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Automated problem modeling"
  - "high-cost simulation"
  - "LLM fine-tuning"
  - "solver-independent evaluation"
  - "antenna design"
date: 2026-05-08
content_hash: cb86df5b85827cac
---

# Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design

**Conference**: ACL 2026 Findings  
**arXiv**: [2512.18682](https://arxiv.org/abs/2512.18682)  
**Code**: None  
**Area**: Signal Communication  
**Keywords**: Automated problem modeling, high-cost simulation, LLM fine-tuning, solver-independent evaluation, antenna design

## TL;DR

This paper proposes APF (Automated Problem Formulation), a solver-independent framework that utilizes LLMs to transform natural language design requirements into executable mathematical optimization models. By employing an innovative data generation and test instance labeling pipeline, it overcomes the difficulty of screening data using solver feedback in high-cost simulation scenarios, significantly outperforming existing methods in antenna design tasks.

## Background & Motivation

**Background**: High-cost simulation-driven design is prevalent in fields such as antenna design, aerospace, microelectronics, and robotics. The core task is to optimize design parameters so that performance distributions (e.g., radiation efficiency curves in the frequency domain) satisfy design requirements. Since requirements are typically provided in unstructured natural language, formalizing them into executable mathematical models remains an optimization bottleneck.

**Limitations of Prior Work**: (1) Prompt-based methods (e.g., Chain-of-Experts, OptiMUS) struggle to accurately identify goals and constraints when facing ambiguous or domain-dependent natural language requirements; (2) Fine-tuning methods (e.g., ORLM, LLMOPT, SIRL) improve performance but rely on solver feedback for data selection, which is unavailable in high-cost simulation scenarios; (3) Existing methods mainly focus on operations research problems like linear or integer programming, which differ significantly from simulation-driven design in terms of problem description and evaluation costs.

**Key Challenge**: Fine-tuning LLMs requires high-quality training data, but in high-cost simulation scenarios, verifying the correctness of generated mathematical formulas requires expensive physical simulations (e.g., electromagnetic full-wave simulation), making large-scale data selection infeasible. The solver feedback mechanisms relied upon by previous fine-tuning methods fail in this context.

**Goal**: Develop a solver-independent automated problem formulation framework capable of automatically generating high-quality training data and fine-tuning LLMs to accurately translate natural language design requirements into executable mathematical optimization models.

**Key Insight**: Introduce test instances as a bridge—by using LLMs to rank and label test instances, the "semantic alignment between natural language requirements and mathematical formulas" is transformed into a "ranking consistency problem," thereby bypassing expensive solver verification.

**Core Idea**: A three-stage pipeline consisting of data generation + test instance labeling + ranking consistency evaluation is used to construct high-quality fine-tuning datasets without calling expensive solvers. This allows 7B/8B open-source models to reach or exceed the modeling accuracy of large models like GPT-4o.

## Method

### Overall Architecture

APF consists of four modules: (1) Data Generation—extracting design requirements from historical simulation data and generating corresponding mathematical formulas using LLMs; (2) Test Instance Labeling—LLMs rank test instances based on a listwise strategy to establish reference rankings; (3) Data Evaluation & Selection—comparing the execution ranking of generated formulas on test instances with the reference ranking to filter high-quality samples; (4) Supervised Fine-Tuning—fine-tuning open-source LLMs on the filtered high-quality data.

### Key Designs

1.  **Unified Abstract Representation and Data Generation**:
    - **Function**: Standardize unstructured industrial specifications into processable forms and generate diverse training data.
    - **Mechanism**: Each requirement is formalized as a structured tuple $r = (\mathcal{Z}, M, \mathcal{C})$, where $\mathcal{Z}$ is the evaluation variable sub-region, $M: z \in \mathcal{Z} \to \mathbb{R}$ is the metric function, and $\mathcal{C}$ specifies the design intent (e.g., threshold constraint $\min_{z \in \mathcal{Z}} M(z) \geq 1.5$ or optimization target). Real solvable requirement sets $\mathcal{R} = \{r_1, r_2, \ldots, r_n\}$ are extracted from historical simulation records, and LLMs generate corresponding mathematical formulas. Data augmentation includes semantic paraphrasing ($v$ equivalent variants) and sequence permutation.
    - **Design Motivation**: Extraction from historical simulations ensures physical feasibility. Semantic paraphrasing enhances robustness to diverse expressions, and sequence permutation prevents the model from relying on spurious positional cues.

2.  **Solver-Independent Evaluation**:
    - **Function**: Evaluate the quality of generated formulas without calling expensive solvers.
    - **Mechanism**: A test instance set $\mathcal{I}$ is introduced as a bridge. LLMs generate reference rankings $\pi_{\text{LLM}} = \arg\max_\pi P_\theta(\pi | \mathcal{P})$ using a listwise strategy. The prompt includes instructions, expert examples, instance data tables, and requirement queries. The quality score is defined as the Spearman correlation coefficient between the execution ranking and the reference ranking: $S(E) = \rho(\pi_E, \pi_{\text{LLM}})$, with only samples showing strong correlation ($> 0.7$) being retained.
    - **Design Motivation**: Direct evaluation of semantic alignment is computationally infeasible. Using test instances transforms the problem into quantifiable ranking consistency. The listwise strategy is two orders of magnitude more efficient than pairwise (1 call vs. 105) while maintaining comparable quality ($\rho$: 0.8643 vs. 0.8536).

3.  **Alignment Metrics and Fine-tuning**:
    - **Function**: Provide comprehensive formula quality evaluation and fine-tune on high-quality data.
    - **Mechanism**: The alignment score consists of objective alignment and constraint alignment: $A(E) = \alpha A_{\text{obj}}(E) + (1-\alpha) A_{\text{con}}(E)$. Objectives are evaluated via Spearman ranking correlation: $A_{\text{obj}} = \frac{1}{n_1} \sum_{e_i \in E_{\text{obj}}} \rho(\hat{\pi}_i, \pi^*)$; constraints use classification accuracy: $A_{\text{con}} = \frac{1}{n_2} \sum_{e_j \in E_{\text{con}}} (1 - \frac{1}{m} \|\hat{\mathbf{y}}_j - \mathbf{y}^*\|_1)$, with $\alpha = 0.5$.
    - **Design Motivation**: Objectives focus on relative ranking correctness, while constraints focus on absolute feasibility judgment; both dimensions are essential.

### Loss & Training

Standard SFT is performed on open-source LLMs using the filtered high-quality dataset $\mathcal{D}_{\text{HQ}}$ (7,879 samples). The dataset uses 2,300 design requirement sets, with 300 for testing (zero overlap) and 2,000 for training. The selection threshold is set at 0.7. Data generation uses GPT-4o, and test instance labeling uses strong LLM judges like GPT-5.

## Key Experimental Results

### Main Results

**Overall Formula Quality Comparison**

| Method | $A_{\text{obj}}$ | $A_{\text{con}}$ | $A$ (Total) |
| :--- | :--- | :--- | :--- |
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

| Configuration | $A_{\text{obj}}$ | $A_{\text{con}}$ | $A$ |
| :--- | :--- | :--- | :--- |
| w/o Augmentation | 0.7656 | 0.7555 | 0.7553 |
| w/o Selection | 0.7603 | 0.7800 | 0.7653 |
| **APF (Full)** | **0.8009** | **0.7971** | **0.7976** |

**Comparison of Evaluation Methods**

| Method | Spearman $\rho$ | LLM Calls | Time (s) | Cost ($) |
| :--- | :--- | :--- | :--- | :--- |
| **Listwise (Ours)** | **0.8643** | **1** | **97.66** | **0.02** |
| Pairwise | 0.8536 | 105 | 2544.8 | 0.47 |

### Key Findings

- After APF fine-tuning, LLAMA3.1-8B improved from 0.2248 to 0.7976 (+256% gain), evolving from nearly unusable to surpassing GPT-4o and Claude-sonnet-4.5.
- Three 7B/8B models showed highly consistent performance (0.7918–0.7976) after APF fine-tuning, proving the general effectiveness of high-quality data.
- LLM judge rankings are highly consistent with human rankings (GPT-5: $\rho = 0.8316$), validating the reliability of solver-independent evaluation.
- Listwise evaluation achieves comparable ranking quality to pairwise while being 26x faster and 23x cheaper.
- Performance remains stable for selection thresholds between 0.6–0.8, showing the framework is insensitive to hyperparameters.
- In practical antenna design, APF-generated optimization models successfully meet all frequency band requirements, while other methods fail at passbands or high radiation nulls.

## Highlights & Insights

- The "test instances as a bridge" concept cleverly transforms semantic alignment into quantifiable ranking consistency, bypassing expensive solver verification.
- The efficiency comparison between listwise and pairwise is impressive: comparable quality with a 26x speedup and 23x cost reduction.
- It proves that for domain-specific tasks, high-quality data combined with small models can match or even exceed the zero-shot capabilities of general-purpose large models.
- Extracting requirements from historical simulations ensures physical feasibility, making this data-driven approach more reliable than random synthesis.

## Limitations & Future Work

- Currently only validated on antenna design; cross-domain generalization in aerodynamics, structural optimization, etc., needs further investigation.
- Solver-independent evaluation relies on constructing prompts with detailed test instances, which is limited by the LLM's context window.
- Data generation depends on powerful models like GPT-4o, introducing additional costs and dependency on the source model's quality.
- Only 7B/8B scale models were verified; the effects on larger or smaller models remain unexplored.

## Related Work & Insights

- **vs. Chain-of-Experts/OptiMUS (Prompt-based)**: Prompting methods lack precision for ambiguous requirements (A: 0.6687–0.7252), whereas APF achieves more accurate understanding through fine-tuning (A: 0.7976).
- **vs. ORLM/SIRL (Fine-tuning-based)**: These methods rely on solver feedback for data filtering, which is infeasible in high-cost simulation scenarios; APF replaces solvers with LLM-based ranking to achieve solver-independent evaluation.
- **vs. GPT-4o/DeepSeek-V3 (Zero-shot)**: The 7B fine-tuned models outperform the zero-shot performance of these large models, demonstrating the significant value of domain-specific fine-tuning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The bridge via test instances and solver-independent evaluation is a novel way to transform semantic alignment into ranking consistency.
- **Experimental Thoroughness**: ⭐⭐⭐ Thoroughly validated in antenna design, though limited to a single domain. Ablation and sensitivity analyses are complete.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, the motivations are well-defined, and the charts are professional.
- **Value**: ⭐⭐⭐⭐ Provides a practical framework for automated modeling in high-cost simulation fields, with broad industrial potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)
- [\[ACL 2026\] Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards](generative_floor_plan_design_with_llms_via_reinforcement_learning_with_verifiabl.md)
- [\[ICML 2026\] Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs](../../ICML2026/llm_nlp/automated_formal_proofs_of_combinatorial_identities_via_wilf-zeilberger_guidance.md)
- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)

</div>

<!-- RELATED:END -->
