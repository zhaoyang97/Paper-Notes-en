---
title: >-
  [Paper Note] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents
description: >-
  [ACL 2025][LLM Agent][Guideline Following] This paper proposes GuideBench, a benchmark designed to systematically evaluate the capability of LLMs in following domain-oriented guidelines. It covers 1,272 instances across 7 task categories and evaluates 18 LLMs along three dimensions: rule compliance, robustness to rule updates, and alignment with human preferences. The results indicate significant room for improvement for current models when following complex domain rules.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Guideline Following"
  - "benchmark"
  - "Domain-Oriented Agent"
  - "Instruction Following"
  - "Rule Compliance"
date: 2026-05-08
content_hash: bb791eb6b512a449
---

# GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents

**Conference**: ACL 2025  
**arXiv**: [2505.11368](https://arxiv.org/abs/2505.11368)  
**Code**: [https://github.com/Dlxxx/GuideBench](https://github.com/Dlxxx/GuideBench)  
**Area**: LLM Agent  
**Keywords**: Guideline Following, benchmark, Domain-Oriented Agent, Instruction Following, Rule Compliance

## TL;DR

This paper proposes GuideBench, a benchmark designed to systematically evaluate the capability of LLMs in following domain-oriented guidelines. It covers 1,272 instances across 7 task categories and evaluates 18 LLMs along three dimensions: rule compliance, robustness to rule updates, and alignment with human preferences. The results indicate significant room for improvement for current models when following complex domain rules.

## Background & Motivation

LLMs are increasingly deployed as autonomous agents to perform tasks in fields such as operations, auditing, and process optimization. Existing instruction-following benchmarks mainly focus on general domains, relying on the LLM's built-in common-sense knowledge to evaluate capabilities. However, the core challenges faced by **domain-oriented agents** are fundamentally different:

**Domain Rules**: Guidelines contain many rules based on domain knowledge, which may involve compositional, conditional, or nested relationships, and can conflict with the LLM's common-sense knowledge. For example, in e-commerce, whether two products are "identical" may be defined by specific business rules rather than common-sense judgment.

**Frequent Updates**: Domain guidelines are constantly updated due to changes in standards and regulations, and LLM agents must adapt to these changes.

As shown in Figure 1, for the same task of determining whether products are identical, the correct answer can change completely after guidelines are updated from Rule #1 to Rule #2. Currently, there is a lack of comprehensive benchmarks targeted at such scenarios, which hinders the effective evaluation and improvement of LLM agents in domain deployments.

## Method

### Overall Architecture

The construction and evaluation of GuideBench consist of the following core components:

- **7 Task Categories**: audit algorithm, price matching, text relevance, math, agent chatting, summarization, hallucination detection.
- **1,272 Task Instances**: Generated via automated synthesis combined with human refinement.
- **Three Evaluation Dimensions**: (i) rule compliance, (ii) robustness to rule updates, and (iii) human preference alignment.
- **Two Task Formats**: Multiple-choice questions (agent chatting, summarization, hallucination detection) and question answering (audit algorithm, price matching, text relevance, math).

### Key Designs

#### Data Construction Pipeline

The data construction process of GuideBench consists of four stages:

**1. Data Collection**:
- Identify 7 categories that are highly valuable yet understudied in operational applications.
- Manually extract seed instructions from practical use cases.
- Derive domain-specific instructions and base guidelines.

**2. Guideline Rule Generation**:
- Extract key elements based on system prompts, including task objectives, input/output specifications, and rule construction requirements.
- Each rule consists of a condition part (trigger conditions) and an operation part (execution action).
- Categorize and diversify rules, followed by GPT-4o deduplication and human review.
- Obtian 537 guideline rules in total.

**3. Guideline Construction**:
- **Random Selection**: Randomly select $k$ rules within the same domain.
- **Diversity-based Selection**: Prioritize selecting different types of rules to ensure diversity.
- **Semantic-based Selection**: Leverage LLMs to select semantically coherent rules based on the overall instruction.
- Use LLMs to modify rules to simulate guideline update scenarios.

**4. Multi-Response Generation**:
- Use the generated guidelines as part of the prompt to let LLMs generate context.
- Assemble questions and generate multiple-choice answers.

#### Data Quality Control

- **LLM Filtering**: Eliminate duplicate and low-quality rules during the rule generation stage.
- **Human Annotation**: Domain experts with backgrounds in AI and computer science perform comprehensive reviews, correcting optimal options and reference answers.

#### Task Design Methodology

Each task contains four components:
- **Instruction**: Overall task objective.
- **Guidelines**: A set of domain-specific rules.
- **Context**: Relevant text passages.
- **Multiple Options** (optional): Diverse responses generated by LLMs.

### Loss & Training

GuideBench is an evaluation benchmark and does not involve model training. The evaluation uses accuracy as the core metric, assessing selection accuracy for correct choices in multiple-choice questions, and the consistency between generated answers and reference answers for question-answering tasks.

## Key Experimental Results

### Main Results

**Evaluation Results of 18 LLMs (Overall Accuracy %)**:

| Model | Overall | Audit | Price | Text | Math | Agent | Summ. | Halluc. |
|------|------|-------|-------|------|------|-------|-------|---------|
| Deepseek-R1 | **87.26** | 93.04 | 80.32 | 84.90 | **65.38** | 98.89 | **89.66** | 96.61 |
| GPT-4o | 86.48 | **96.52** | 84.84 | 81.25 | 13.46 | **100** | 82.76 | 94.92 |
| Deepseek-V3 | 83.96 | 97.39 | **91.18** | 53.65 | 5.77 | 98.89 | 77.59 | 94.92 |
| o1 | 79.17 | 73.48 | 76.24 | 79.69 | 48.08 | 92.78 | 81.03 | 92.37 |
| GPT-4o* | 80.90 | 94.78 | 74.66 | 80.21 | 7.69 | 95.56 | 68.97 | 94.07 |
| Gemini2.5-pro-exp | 80.90 | 90.00 | 75.79 | 85.94 | 44.23 | 80.00 | 87.93 | 93.22 |
| Yi-1.5-6B | 56.05 | 50.43 | 66.29 | 43.75 | 7.69 | 66.11 | 20.69 | 72.03 |
| Mistral-7B | 69.58 | 86.52 | 66.06 | 77.60 | 1.92 | 58.33 | 58.62 | 88.98 |
| Gemma-3-4b-it | 61.71 | 58.70 | 56.11 | 75.00 | 0 | 76.67 | 72.41 | — |

**Key Observations**:
- **Math task is the most challenging**: Almost all models score extremely low on Math, with GPT-4o at only 13.46% and Deepseek-V3 at only 5.77%, while only Deepseek-R1 reaches 65.38%.
- **Agent Chatting is relatively easy**: GPT-4o achieves 100% on this task.
- **Small models perform significantly worse than large models**: Yi-1.5-6B only achieves 56.05% overall.

### Key Findings

1. **Crucial Role of Guidelines**: Without guidelines, models can only rely on common sense, which may lead to judgments that conflict with domain rules. Experiments show that performance drops significantly when guidelines are removed, indicating that external domain knowledge is vital for correct decision-making.

2. **Benefits of Chain-of-Thought**: CoT brings significant improvements in complex tasks (e.g., math, audit algorithm), but yields limited benefits on simple tasks. This suggests that explicit chain-of-thought is necessary for domain rules requiring multi-step reasoning.

3. **Robustness of Rule Updates**: After rules are modified, models must adapt to the new rules instead of relying on prior knowledge. Experiments show that current LLMs generally struggle with rule updates, especially when new rules conflict with common sense.

4. **Error Analysis**:
    - Rule conflict errors: Models tend to follow internal common sense rather than external rules.
    - Rule omission errors: Models ignore key rules among a large number of rules.
    - Reasoning chain breakage: Models are prone to making mistakes on conditional nested rules.

## Highlights & Insights

1. **Addressing practical deployment pain points**: The biggest challenge for domain agents is not whether they can understand instructions, but whether they can strictly adhere to constantly updated business rules. GuideBench precisely targets this challenge.
2. **Systematic evaluation system**: A three-dimensional evaluation (compliance, robustness, and preference alignment) combined with two task formats (multiple-choice and QA) forms a comprehensive evaluation matrix.
3. **Automated data construction pipeline**: The fully automated pipeline from rule generation to quality validation offers excellent scalability, allowing for easy expansion into new task domains.
4. **Profound insights**: The extremely low scores on the Math task expose fundamental deficiencies of LLMs in strict logical reasoning—even the strongest reasoning model, Deepseek-R1, only reaches 65.38%.

## Limitations & Future Work

1. **Limited domain coverage**: Although 7 categories are covered, realistic domains are far more extensive. Highly specialized fields such as medicine, law, and finance are not yet included.
2. **Limited rule complexity**: While 537 rules are substantial, actual business systems can contain thousands of rules with more complex dependencies.
3. **Dependency on LLMs for evaluation**: Some parts of quality control and evaluation rely on GPT-4o, which may introduce biases associated with the evaluated models.
4. **Static evaluation**: The benchmark does not consider the dynamic process of agents progressively understanding and applying rules in multi-turn interactions.

## Related Work & Insights

- **Instruction-following benchmarks**: IFEval (Zhou et al., 2023a), ComplexBench (Wen et al., 2024), and RuleBench (Sun et al., 2024) focus on general instruction following, while this paper extends this to domain-specific rules.
- **LLM Agents**: Many agent studies have explored the application of LLMs in engineering, natural sciences, and social sciences, but systematic evaluation of rule compliance remains missing.
- **Counterfactual reasoning**: Xu et al. (2024b) and Xie et al. (2024) explore LLMs' ability to process information that conflicts with common sense, which is relevant to the domain rule conflict issues discussed in this paper.
- **Insights**: For the deployment of LLM agents, merely improving general instruction-following capability is insufficient. It is crucial to develop specialized domain-rule adaptation capabilities, which may require new training paradigms (such as rule-conditioned instruction tuning).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 3.5 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Value | 4.5 |
| Writing Quality | 4 |
| Overall Rating | 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)
- [\[CVPR 2026\] Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent](../../CVPR2026/llm_agent/universal_guideline-driven_image_clustering_via_a_hybrid_llm_agent.md)
- [\[ACL 2025\] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)

</div>

<!-- RELATED:END -->
