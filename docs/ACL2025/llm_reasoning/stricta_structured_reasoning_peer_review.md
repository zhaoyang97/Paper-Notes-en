---
title: >-
  [Paper Note] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond
description: >-
  [ACL 2025][Reasoning][Structured Reasoning] This paper proposes the STRICTA framework, which models expert text assessment (e.g., peer review) as a step-by-step reasoning graph based on Structural Causal Models (SCMs). By collecting over 4,000 reasoning steps from more than 40 biomedical experts across 22 papers, the study reveals that differences in prior knowledge are the primary cause of review disagreement, and that writing style has an outsized impact on final decisions.…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Structured Reasoning"
  - "Peer Review"
  - "Causal Models"
  - "Text Assessment"
  - "Human-AI Collaboration"
date: 2026-05-08
content_hash: 474d4db1aa82e105
---

# STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond

**Conference**: ACL 2025  
**arXiv**: [2409.05367](https://arxiv.org/abs/2409.05367)  
**Code**: [https://github.com/UKPLab/acl2025-stricta](https://github.com/UKPLab/acl2025-stricta)  
**Area**: LLM Reasoning  
**Keywords**: Structured Reasoning, Peer Review, Causal Models, Text Assessment, Human-AI Collaboration

## TL;DR
This paper proposes the STRICTA framework, which models expert text assessment (e.g., peer review) as a step-by-step reasoning graph based on Structural Causal Models (SCMs). By collecting over 4,000 reasoning steps from more than 40 biomedical experts across 22 papers, the study reveals that differences in prior knowledge are the primary cause of review disagreement, and that writing style has an outsized impact on final decisions. Additionally, it highlights that LLMs can effectively assist in structured assessment when under human supervision.

## Background & Motivation

**Background**: Text quality assessment (such as peer review, fact-checking, and essay grading) is a core activity in many professional domains. Existing work typically treats this as a black-box problem—taking a document as input to directly predict a quality score or generate review comments.

**Limitations of Prior Work**: (a) The process of how experts step-by-step reason from a document to a final judgment is entirely non-transparent; (b) the lack of formal models to describe the assessment reasoning process hinders the development of explainable AI-assisted peer review; (c) existing automatic review generation focuses solely on the final output, ignoring fine-grained factors in the decision-making process.

**Key Challenge**: To achieve transparent and reliable human-AI collaborative review, it is essential to first understand "how experts make judgments." However, there is currently neither a formal framework to describe this reasoning process nor datasets that record expert reasoning steps.

**Goal**: To formalize the reasoning process of expert text assessment and construct an analyzable, automatable structured reasoning framework.

**Key Insight**: Leveraging Pearl's causal inference theory, the assessment process is modeled as a Structural Causal Model (SCM), where each reasoning step serves as a node in a causal graph, and the causal relationships between steps are represented by directed edges.

**Core Idea**: Externalize the "mental process" of text assessment into an analyzable reasoning graph using Structural Causal Models, making the review decision process observable, comparable, and automatable.

## Method

### Overall Architecture
Input: A document to be assessed (e.g., a biomedical paper). Output: A final quality judgment along with a complete reasoning process (answers to 45 interconnected steps). STRICTA operates in three stages: designing the SCM structure $\rightarrow$ populating the SCM with human data $\rightarrow$ analysis and automation.

### Key Designs

1. **STRICTA Problem Formalization (Based on SCM)**:

    - Function: Defines text assessment as an SCM $\mathcal{M} = (U, V, F, P_\mathcal{M})$, where $V$ contains input nodes $I$ (document text), reasoning components $C$ (intermediate steps), and a final judgment $T$.
    - Mechanism: The input is the root node of the causal graph, the final judgment is the terminal node, and intermediate reasoning steps form a directed acyclic graph. Background variables $U$ capture the subjective differences among reviewers (e.g., prior knowledge, preferences). The structural equation for each step, $v_i = f_i(\text{pa}_i, u_i)$, describes how the answer for the current step is derived from its parent nodes.
    - Design Motivation: Causal models not only describe correlations but also support intervention and counterfactual analysis—answering questions such as "how would the final judgment change if a certain assessment criterion were altered?" This is something purely statistical models cannot achieve.

2. **Workflow Design (Paper Review Case Study)**:

    - Function: Designs a 45-step review reasoning workflow through expert interviews.
    - Mechanism: Interviews two senior biomedical researchers to extract their cognitive processes when reviewing papers. This is structured into three types of steps: **read** (reading specific parts of the paper), **extract** (extracting key information from the text), and **infer** (reasoning and making judgments based on existing information). The workflow begins with the methodology section, proceeds through figure/table quality assessment, results analysis, and conclusion consistency checks, and ultimately converges to the final paper quality judgment.
    - Design Motivation: A fixed reasoning structure enables quantitative comparison across different papers and reviewers, which stands in stark contrast to previous approaches that dynamically construct custom reasoning chains for each instance (e.g., reasoning graphs in fact-checking).

3. **Data Collection and SCM Population**:

    - Function: Organizes over 40 biomedical researchers to review 22 papers following the workflow, collecting 4,371 reasoning step answers.
    - Mechanism: Each paper is reviewed by at least 3 researchers (with 5 reviewers for 11 of the papers). Reviewers are forced to answer questions in the topological order of the causal graph during annotation, ensuring compliance with causal constraints. Gaussian process classifiers are used to fit the structural equations of Boolean nodes.
    - Design Motivation: Redundant annotations (multiple answers for the same step) are used to estimate background noise/the level of subjectivity. Krippendorff's $\alpha = 0.42$, which aligns with typical agreement levels of peer review scores.

4. **Causal Analysis**:

    - Function: Explores which factors influence the final judgment through Average Causal Effect (ACE) and counterfactual analysis.
    - Mechanism: Simulates 200 samples on the Boolean-node SCM to calculate the ACE of each step on the final judgment. Counterfactual analysis focuses on negative judgment cases to explore whether altering steps related to figure/table quality can flip the decision.
    - Design Motivation: ACE measures causal effect rather than correlation, truly revealing which factors drive decisions during the review process.

### Loss & Training
The structural equations of SCM are fitted from human data using Gaussian process classifiers. LLM experiments employ zero-shot prompting, involving no training.

## Key Experimental Results

### Main Results

| Workflow Step | ACE (Causal Effect on Final Judgment) |
|-----------|--------------------------|
| Consistency between conclusion and research question (step33) | **0.37** (Highest positive impact) |
| Relevance of conclusions (step46) | 0.20 |
| Clarity of writing (step48) | 0.20 |
| Whether it is a methodology paper (step4) | 0.02 (Almost no impact) |
| Whether figure selection is reasonable (step19) | -0.01 |

**Counterfactual Analysis**: Among 25 negative judgments, 60% could be flipped to positive by improving just one step: "consistency between figures and discussion" (step12).

### Ablation Study

| Model/Setting | BERT-F1↑ | SummaC↑ | F1 (Boolean Decision)↑ |
|-----------|----------|---------|----------------|
| Human Baseline | 0.799 | -0.151 | 0.801 |
| GPT-4o (Independent Reasoning) | 0.780 | -0.186 | 0.720 |
| GPT-4o (Human-Supervised) | - | - | Significant Gain |
| Llama3 (Independent Reasoning) | 0.752 | -0.274 | 0.170 |
| Mixtral (Independent Reasoning) | 0.761 | -0.149 | 0.559 |

### Key Findings
- **Prior knowledge is the primary cause of disagreement**: Reasoning steps involving background knowledge (infer-knowledge) exhibit the largest variance in answers, indicating that differences in reviewers' prior experiences are the main source of review inconsistency.
- **The causal impact of writing style is unexpectedly large**: With an ACE of 0.20, it is on par with the relevance of conclusions. This indicates that reviewers have a positive bias toward good writing styles, even if the scientific content is average.
- **LLMs are prone to error propagation**: In independent reasoning settings, LLMs' F1 scores for Boolean decisions are far below those of humans, as errors cascade and amplify along the causal graph.
- **Human supervision effectively mitigates the issue**: When humans review each step and provide corrected inputs, LLM performance is significantly enhanced, validating the value of human-AI collaboration within the STRICTA framework.
- **Figure quality is the most "cost-effective" point for improvement**: 60% of negative judgments can be flipped by improving a single figure-related step.

## Highlights & Insights
- **Applying causal models to text assessment is a groundbreaking contribution**: Instead of merely "decomposing review into multiple steps", it leverages a rigorous causal framework (SCM) to support intervention and counterfactual analysis. This formalization provides a rigorous way to answer questions like "would the paper have been accepted if the authors had improved the figures?"
- **Fixed reasoning structures vs. dynamic reasoning chains**: Unlike Chain-of-Thought (CoT), STRICTA's reasoning graph is fixed, with each step holding explicit semantics, which enables quantitative comparison across different instances. This is particularly valuable for assessment tasks that demand consistency and reproducibility.
- **Transferable to other assessment scenarios**: Paper review is only one instantiation; the STRICTA framework is equally applicable to any domain requiring structured judgments, such as fact-checking, essay grading, and medical report evaluation.

## Limitations & Future Work
- **Workflow design relies heavily on expert interviews**: Currently, domain experts are required to hand-design the reasoning graphs. Automated causal discovery (automatically extracting causal structures from review report text) is an important future direction.
- **Limited to the biomedical domain**: The workflow is domain-specific, and adapting it to other disciplines (such as computer science) requires redesigning the workflow.
- **Over-simplification of Boolean decisions**: Many review steps are naturally continuous or multi-valued; simplifying them to Boolean values may result in the loss of fine-grained information.
- **Error propagation in LLMs remains unresolved**: Although human supervision is effective, better error mitigation strategies are still needed for fully autonomous scenarios.

## Related Work & Insights
- **vs. Automatic review generation** (e.g., reviews written by GPT-4): Prior works directly generate final review text, whereas STRICTA focuses on the structured reasoning leading to the decision, making them complementary.
- **vs. Fact-checking reasoning chains** (e.g., ProgramFC): Fact-checking dynamically constructs verification programs for each claim, while STRICTA leverages a fixed reasoning structure to support cross-instance comparison. Their applicable scenarios differ.
- **vs. Graph-of-Thought (GoT)**: GoT also utilizes graph structures for reasoning but constructs them dynamically. STRICTA's fixed graph structure can be combined with adaptive strategies of GoT to achieve more reliable, long-chain reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing causal models to text assessment reasoning is pioneering work, with a mathematically rigorous framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale study involving 40+ experts and 4000+ reasoning steps, though restricted to the biomedical domain.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formal definitions, showing a logical continuity from the framework to empirical analysis and application.
- Value: ⭐⭐⭐⭐⭐ Establishes a solid theoretical foundation and empirical support for AI-assisted review, opening up new research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](../../AAAI2026/llm_reasoning/text-to-scene_with_large_reasoning_models.md)
- [\[ICML 2026\] What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code](../../ICML2026/llm_reasoning/what_really_improves_mathematical_reasoning_structured_reasoning_signals_beyond_.md)
- [\[NeurIPS 2025\] GPO: Learning from Critical Steps to Improve LLM Reasoning](../../NeurIPS2025/llm_reasoning/gpo_learning_from_critical_steps_to_improve_llm_reasoning.md)
- [\[ACL 2025\] Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation](beyond_the_answer_advancing_multi-hop_qa_with_fine-grained_graph_reasoning_and_e.md)

</div>

<!-- RELATED:END -->
