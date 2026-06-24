---
title: >-
  [Paper Note] AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts
description: >-
  [ICML 2025 (Workshop on Multi-Agent Systems)][LLM Agent][Multi-Agent Systems] AGACCI proposes a multi-agent evaluation framework consisting of 9 specialized agents. It decomposes the evaluation task of educational programming assignments into roles such as rubric parsing, code execution validation, visual evaluation, and explanatory reasoning assessment. Through collaboration, it achieves more accurate, consistent, and interpretable rubric-aligned feedback than single-model b…
tags:
  - "ICML 2025 (Workshop on Multi-Agent Systems)"
  - "LLM Agent"
  - "Multi-Agent Systems"
  - "Educational Assessment"
  - "Code Evaluation"
  - "rubric alignment"
  - "automated feedback"
date: 2026-05-08
content_hash: 437f79618ce602fa
---

# AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts

**Conference**: ICML 2025 (Workshop on Multi-Agent Systems)  
**arXiv**: [2507.05321](https://arxiv.org/abs/2507.05321)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: Multi-Agent Systems, Educational Assessment, Code Evaluation, rubric alignment, automated feedback

## TL;DR

AGACCI proposes a multi-agent evaluation framework consisting of 9 specialized agents. It decomposes the evaluation task of educational programming assignments into roles such as rubric parsing, code execution validation, visual evaluation, and explanatory reasoning assessment. Through collaboration, it achieves more accurate, consistent, and interpretable rubric-aligned feedback than single-model baselines.

## Background & Motivation

### Limitations of Prior Work

In the domain of AI-assisted educational assessment, existing LLM-based automated evaluation systems suffer from three core issues:

**Low feedback quality**: Single LLMs often generate overly positive evaluations (even when student answers are incorrect) or produce hallucinated reasoning that lacks an empirical basis (Jansen et al., 2024). Feedback is frequently limited to shallow praise or vague suggestions, failing to accurately reflect students' performance or misconceptions.

**Insufficient Rubric Alignment**: Existing methods often ignore the fine-grained criteria defined in pedagogical rubrics, focusing purely on surface-level code correctness or syntax (Phung et al., 2023). This causes the generated feedback to deviate from the instructor's assessment intent.

**Inconsistent Assessment**: LLMs can produce starkly different evaluations for similar or identical submissions. Even with ensemble strategies (Pathak et al., 2025), resolving the structural limitations of single-model systems remains challenging.

### Limitations of Prior Work

- **G-Eval** (Liu et al., 2023): Uses LLMs as evaluators but remains a single-model system lacking structured division of labor.
- **Generate-Evaluate-Regenerate Pipelines** (Guo et al., 2024; Seo et al., 2025): Although iterative optimization is incorporated, error detection remains imperfect, and discovered issues do not always propagate to the final feedback.
- **Agent-as-a-Judge** (Zhuge et al., 2024): Setting up dedicated evaluation agents improves alignment with human graders but is still constrained by the structural limitations of a single evaluation role.

### Motivation

Educational programming assignments (especially those in Jupyter Notebook formats) involve multi-dimensional assessment requirements, including code execution correctness, visual output quality, and depth of explanatory reasoning. A single model struggles to address all these dimensions simultaneously. The authors argue that by systemizing role assignment and structuring the evaluation pipeline, more precise judgments can be achieved on each dimension while maintaining overall consistency.

## Method

### Overall Architecture

AGACCI is built on the **AutoGen** framework, using **GPT-4o mini** as the backbone model for all agents. The system decomposes the evaluation process into a modular agent pipeline consisting of **9 specialized agents**.

The overall workflow consists of three phases:

1. **Input Parsing**: The Rubric Interpreter and Submission Analyzer decompose the task into structured evaluation goals.
2. **Parallel Evaluation**: Three parallel evaluation streams—functional execution evaluation (Execution + Result Evaluator), visualization evaluation (Visualization Evaluator), and reasoning evaluation (Interpretation Evaluator).
3. **Aggregation Output**: The Meta Evaluator checks consistency across streams $\rightarrow$ the Final Judge makes a comprehensive verdict $\rightarrow$ the Summarizer formats the final output.

### Key Designs

#### 1. Rubric Interpreter

- **Function**: Restructures high-level rubric descriptions into actionable evaluation criteria.
- **Key Design Points**: Rather than treating rubrics as static checklists, it transforms them into executable grading goals, identifying implicit dependencies, sequential constraints, and minimum performance expectations.
- **Output Format**: Structured JSON containing `final_objective`, `prerequisite_items`, `subgoals`, and `evidence_types`.

#### 2. Submission Analyzer

- **Function**: Examines the student's submission holistically to identify major objectives, logical structures, and alignment with the rubric criteria.
- **Key Design Points**: Detects the sequence and purpose of code blocks, comments, and outputs, acting as a bridge between human-readable goals and machine-level analysis.
- **Role**: Ensures that downstream evaluation occurs within the correct pedagogical context.

#### 3. Execution Evaluator

- **Function**: Focuses on the functional validity of the code.
- **Checks**: Verifies whether the code runs error-free, whether core computational steps are present, and whether expected outputs (e.g., plots, printed metrics) are generated.
- **Role**: Ensures the reliability of technical performance before qualitative evaluation begins.

#### 4. Result Evaluator

- **Function**: Determines whether the execution results satisfy the quantitative performance standards specified by the rubric.
- **Evaluation Mechanism**: Parses printed outputs, logs, or numerical results to produce a binary judgment (pass/fail).
- **Special Handling**: If execution fails or no measurable results exist, this agent waits for instructions and abstains from making a judgment.

#### 5. Visualization Evaluator

- **Function**: Checks the clarity and appropriateness of visual outputs (graphs, charts).
- **Evaluation Dimensions**: Checks whether the visualization method matches the nature of the data and whether visual components (axes, labels, legends) support interpretability.

#### 6. Interpretation Evaluator

- **Function**: Evaluates the student's ability to reason beyond mere observations.
- **Focus**: Causal or inferential explanations that extract meaning from data patterns, anomalies, or trends.
- **Penalty**: Overly descriptive or poorly argued commentary is flagged.

#### 7. Meta Evaluator

- **Function**: Acts as an internal consistency checker that cross-validates the outputs of various agents.
- **Operation**: Flags contradictory or unsupported assessments and checks the alignment between observed evidence and claimed rubric fulfillment.
- **Authority**: Proposes overrides or adjusts confidence scores.

#### 8. Final Judge

- **Function**: Aggregates all evaluations into a final decision.
- **Output**: Resolves ambiguities in cross-agent outputs, determines binary rubric fulfillment scores (pass/fail), and generates human-readable feedback.

#### 9. Summarizer

- **Function**: Condenses the system's rulings into a compact, learner-oriented summary.
- **Output Format**: Structured JSON containing key findings, recommendations, and rubric scores.

### Architectural Design Choices

- **Parallel + Hierarchical Control**: Three evaluation streams run in parallel to improve efficiency, while the Meta Evaluator and Final Judge provide hierarchical control.
- **Backbone Selection**: GPT-4o mini is selected instead of larger models, considering the limited computational resources, budget constraints, and latency requirements in educational settings.
- **Framework Selection**: Built on AutoGen to achieve modular agent orchestration and flexible interaction modes.

### Loss & Training

No model training is involved in this work. All agents are implemented via prompt engineering using GPT-4o mini, where each agent has a meticulously designed system prompt to define its role and evaluation logic. The complete prompts for all 9 agents are provided in the appendix.

## Key Experimental Results

### Dataset

- **Source**: Student submissions collected from a real university course.
- **Scale**: 60 participants $\times$ 6 programming tasks = **360 submissions**.
- **Task Areas**: Machine Learning (ML), Computer Vision (CV1: face detection, CV2: segmentation), Natural Language Processing (NLP1: text classification, NLP2: summarization, NLP3: chatbot).
- **Annotation**: Domain experts annotated 3 binary rubric scores + qualitative feedback.
- **Language**: Korean.

### Evaluation Strategy

- **Quantitative**: Rubric classification accuracy (modeled as a multi-label binary classification problem).
- **Qualitative**: G-Eval-based 4-dimensional evaluation (each feedback evaluated 20 times and averaged, using GPT-4o as the scorer).
- **Baseline**: SLI (Single-model baseline) utilizing the same GPT-4o mini model.
- **Repetitions**: Each system was run independently for 6 rounds.

### Main Results

#### Overall Rubric Accuracy Comparison

| System | Average Rubric Accuracy |
|------|-------------------|
| SLI (Single-model) | ~48% |
| **AGACCI** | **~60%** |

AGACCI outperforms the baseline in overall rubric accuracy by **approximately 12 percentage points**.

#### Fine-grained Rubric Accuracy by Task Domain (Selected from Table 4 / Table 2)

| Task | Rubric Item | AGACCI (mean±std) | SLI (mean±std) |
|------|----------|-------------------|----------------|
| ML | Preprocessing, Training, and Visualization | **0.734±0.098** | 0.174±0.018 |
| ML | Kaggle Submission Status | 0.473±0.011 | **0.587±0.059** |
| ML | Leaderboard Accuracy Threshold | 0.239±0.000 | **0.685±0.042** |
| CV1 | Natural Alignment of Face Stickers | **0.746±0.027** | 0.179±0.048 |
| CV2 | Portrait Mode Error Resolution | **0.680±0.044** | 0.386±0.052 |
| CV2 | Clear Localization of Portrait Errors | **0.654±0.009** | 0.434±0.046 |
| NLP1 | Word2Vec Accuracy Improvement | **0.651±0.019** | 0.406±0.011 |
| NLP2 | Extractive vs. Generative Comparison | **0.867±0.020** | 0.454±0.051 |
| NLP3 | Stabilizing Transformer Convergence | **0.969±0.020** | 0.577±0.092 |
| NLP3 | Korean Response Generation Model | **0.959±0.000** | 0.510±0.096 |

### Qualitative Evaluation Results (G-Eval 4 Dimensions, 5-point Scale)

| Dimension | AGACCI | SLI |
|------|--------|-----|
| Feedback Accuracy | **Higher** | Lower |
| Consistency | **Higher, lower variance** | Lower, higher variance |
| Coherence | **Higher** | Lower |
| Relevance | Comparable (slightly higher variance) | Comparable |

### Ablation Study

While the paper does not present strict ablation experiments (such as removing agents one by one), a fine-grained analysis of different task domains and rubric items indirectly reveals the contributions of individual modules:

| Analysis Dimension | Finding |
|---------|------|
| High-complexity rubric items (requiring multi-step reasoning) | AGACCI achieves an average accuracy >0.73, significantly outperforming SLI |
| Low-complexity/external verification rubric items (e.g., Kaggle status) | SLI performs better instead |
| NLP tasks (requiring explanatory reasoning) | AGACCI shows the most pronounced advantage |
| ML tasks (requiring external behavior verification) | Performances are comparable, or SLI slightly wins |
| Role of Meta Evaluator | The improvement in consistency and reduction in variance are attributed to this module |
| Role of Rubric Interpreter | Stable relevance scores are attributed to the structuring of rubrics by this module |

### Key Findings

1. **AGACCI exhibits its most significant advantages on high-complexity rubric items requiring multi-step reasoning and structured understanding**: This includes visual consistency, error diagnosis, comparative summarization strategies, and deep learning model convergence.
2. **External verification criteria represent a pronounced bottleneck**: For rubric items involving behavior outside the code (such as Kaggle submission status and leaderboard validation), AGACCI cannot infer these steps, resulting in performance inferior to the baseline.
3. **Consistency improvement is driven by the Meta Evaluator**: By reconciling contradictory evaluations among individual agents prior to finalized outputs, AGACCI maintains a highly stable grading stance.
4. **Reason for higher variance in Relevance**: AGACCI tends to provide forward-looking suggestions and reflective comments beyond the strict scope of the rubrics. While pedagogically beneficial, this may lead to penalties under strict rubric alignment evaluations.

## Highlights & Insights

1. **The role decomposition paradigm is highly referenceable**: Disassembling the evaluation task into a pipeline of parsing, analysis, execution validation, visual evaluation, reasoning evaluation, meta-checking, final grading, and summarization provides clear role boundaries and single responsibilities. This decomposition holds value for other agentic workflows requiring multi-dimensional judgments.
2. **Ingenious Meta Evaluator design**: Introducing a consistency checker to cross-validate multiple agent outputs effectively mitigates contradictions and groundless judgments, serving as a highly generalizable design pattern for multi-agent systems.
3. **Validation in practical educational settings**: Evaluating with 360 real student submissions from university courses rather than synthetic data bolsters the practical relevance of the findings.
4. **Comprehensive evaluation of feedback quality**: Adopting G-Eval's four dimensions (accuracy, relevance, consistency, coherence) and averaging each feedback over 20 repeated evaluations exemplifies a rigorous evaluation methodology.
5. **Pragmatic choice of GPT-4o mini**: Prioritizing a lightweight model over larger frontiers aligns with real-world deployment challenges regarding budget and latency in educational settings.

## Limitations & Future Work

1. **Inability to handle external verification criteria**: The system evaluates based solely on code and rubric contexts, failing to verify criteria requiring external information (e.g., Kaggle standings). This could be resolved by integrating external APIs or utilizing screenshot-parsing visual agents.
2. **Lack of strict ablation experiments**: Individual agents were not systematically removed to test their independent contributions, leaving potential redundancies among the 9 agents unquantified.
3. **Limited data scale**: The evaluation relies on 360 submissions from a single university course involving 60 participants and 6 tasks, leaving generalizability to other courses or programming languages unverified.
4. **Workshop paper limitations**: Published in the ICML 2025 Multi-Agent Systems Workshop rather than the main conference, leaving the depth of experiments and paper length somewhat constrained.
5. **Korean context constraints**: All experimental materials were in Korean, and evaluations utilized GPT-4o's Korean language capability. Cross-lingual generalizability remains unverified.
6. **Performance drop under vague or contradictory rubrics**: As acknowledged by the authors, when rubrics lack clarity or contradict themselves, inter-agent consistency decreases.
7. **Lack of cost analysis**: The cost and latency overhead of calling 9 agents in parallel/serial configurations are not quantified. Compared to single-model baselines, total token consumption likely increases several-fold.
8. **Lack of stronger baselines**: The system is only compared against a single GPT-4o mini, without comparisons to single-model GPT-4o, Claude, or alternative multi-agent evaluation frameworks.
9. **Unverified student feedback experience**: Whether students genuinely benefit more from AGACCI's feedback remains untested by a formal user study.

## Related Work & Insights

- **G-Eval** (Liu et al., 2023): The foundational framework of LLM-as-Grade, which underpins AGACCI's qualitative assessment methodology.
- **Agent-as-a-Judge** (Zhuge et al., 2024): Assigns the grading task to a dedicated agent, which AGACCI expands into a multi-agent framework.
- **AutoGen** (Wu et al., 2023): Serves as AGACCI's underlying architecture, enabling multi-agent conversation and orchestration.
- **VISTA** (Lee et al., 2024): Demonstrates that separating task-specific LLM components improves educational content generation, inspiring AGACCI's modular layout.
- **Rubric is All You Need** (Pathak et al., 2025): Reinforces rubric-based LLM code evaluation, aligning with AGACCI's criteria-centric approach.

**Insights**: Multi-agent systems possess inherent advantages in tasks requiring multi-dimensional evaluation. AGACCI's execution-to-reasoning role decomposition can be mapped to domains such as code review, paper peer review, and interview evaluation. Furthermore, the consistency checking pattern of the Meta Evaluator is worth promoting in broader multi-agent tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The role division of the multi-agent grading framework is well-designed, and the Meta Evaluator is a notable highlight. However, utilizing multi-agent collaboration for evaluation is not an entirely new concept.
- Experimental Thoroughness: ⭐⭐⭐ The data scale is limited (360 submissions), ablation analysis and cost tracking are missing, and evaluation is limited to a single-model baseline—reflecting typical workshop-level experimental depth.
- Writing Quality: ⭐⭐⭐⭐⭐ The writing is clearly structured, role definitions for each agent are precise, and the appendix supplies complete prompts and run examples to facilitate reproducibility.
- Value: ⭐⭐⭐⭐ It holds direct application value for the Educational AI domain, and its role-decomposition pattern can be migrated to other multi-dimensional grading contexts, though the paper's depth is restricted by its workshop format.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EpiAgent: An Agent-Centric System for Ancient Inscription Restoration](../../CVPR2026/llm_agent/epiagent_agent_centric_system_for_ancient_inscription_restoration.md)
- [\[ICLR 2026\] Terminal-Bench: Benchmarking Agents on Difficult, Real-World Tasks in the Command Line Interface](../../ICLR2026/llm_agent/terminal-bench_benchmarking_agents_on_hard_realistic_tasks_in_command_line_inter.md)
- [\[ICLR 2026\] SciNav: A General Agent Framework for Scientific Coding Tasks](../../ICLR2026/llm_agent/scinav_a_general_agent_framework_for_scientific_coding_tasks.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](../../ACL2026/llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](../../ICLR2026/llm_agent/membership_privacy_risks_of_sharpness_aware_minimization.md)

</div>

<!-- RELATED:END -->
