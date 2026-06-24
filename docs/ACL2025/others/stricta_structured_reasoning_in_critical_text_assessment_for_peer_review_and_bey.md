---
title: >-
  [Paper Note] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond
description: >-
  [ACL 2025][structured reasoning] This paper proposes the STRICTA framework, which models text assessment as an explicit, step-by-step reasoning graph (workflow) based on Structured Causal Models (SCMs). By collecting a dataset of over 4,000 reasoning steps from more than 40 experts reviewing biomedical papers, the study finds that differences in prior knowledge are the primary cause of expert disagreement, and writing style has a causal impact on the final evaluation. Further…
tags:
  - "ACL 2025"
  - "structured reasoning"
  - "peer review"
  - "causal model"
  - "text assessment"
  - "human-AI collaboration"
date: 2026-05-08
content_hash: 2d1022893b15be5d
---

# STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond

**Conference**: ACL 2025  
**arXiv**: [2409.05367](https://arxiv.org/abs/2409.05367)  
**Code**: [GitHub](https://github.com/UKPLab/acl2025-stricta)  
**Area**: Others  
**Keywords**: structured reasoning, peer review, causal model, text assessment, human-AI collaboration

## TL;DR
This paper proposes the STRICTA framework, which models text assessment as an explicit, step-by-step reasoning graph (workflow) based on Structured Causal Models (SCMs). By collecting a dataset of over 4,000 reasoning steps from more than 40 experts reviewing biomedical papers, the study finds that differences in prior knowledge are the primary cause of expert disagreement, and writing style has a causal impact on the final evaluation. Furthermore, while LLMs suffer from error propagation, this can be effectively mitigated with human supervision.

## Background & Motivation

**Background**: Critical text assessment (such as peer review, fact-checking, and essay grading) is at the core of expert activities. Existing automated methods treat assessment as a black box, failing to reveal how experts arrive at their decisions.

**Limitations of Prior Work**: (a) There is a lack of formal models to capture the reasoning process of experts; (b) Existing explanations for fact-checking only target short claims and do not handle subjectivity; (c) The boundaries of LLM capabilities in text assessment remain unclear.

**Key Challenge**: To achieve transparent human-AI collaborative assessment, it is necessary to first understand "how experts reason to reach a judgment."

**Goal**: (a) Formalize the reasoning process of assessment; (b) Provide an empirical dataset to study human reasoning patterns; (c) Evaluate the performance of LLMs in structured reasoning.

**Key Insight**: Represent assessment reasoning using SCMs—where each reasoning step serves as a variable, dependencies between steps are represented as causal edges, and background variables account for subjectivity.

**Core Idea**: Decompose the "reading $\to$ thinking $\to$ scoring" process into a structured workflow with causal dependencies, enabling both causal analysis and fine-grained human-AI collaboration.

## Method

### Overall Architecture
STRICTA is defined as an SCM $\mathcal{M} = (U, V, F, P_\mathcal{M})$, where $V$ contains input nodes $I$ (texts), reasoning steps $C$, and final judgments $T$; $F$ represents the structural equations; and $U$ denotes background variables (the source of subjectivity).

### Key Designs

1. **Three-step Framework Instantiation**

    - **Step 1: SCM Structure Design**: Identified 45 interconnected reasoning steps through expert interviews, categorizing them into three activities: read, extract, and infer.
    - **Step 2: Data Population**: Over 40 experts reviewed 22 papers following the prescribed workflow, yielding 4,371 reasoning step responses.
    - **Step 3: Analysis**: Gaussian Process Classifiers were used to estimate structural equations for boolean nodes, supporting Average Causal Effect (ACE) and counterfactual queries.

2. **Causal Analysis**

    - **ACE Estimation**: The alignment between conclusions and research questions (step33) has the greatest impact on the final judgment (ACE = 0.37); writing clarity has an ACE of 0.20—demonstrating that **writing style has a direct causal influence on assessment outcomes**, rather than being merely associated with them.
    - **Counterfactual Analysis**: 60% of negative judgments could be reversed simply by improving "figure-text alignment."
    - The background knowledge step shows the highest variance $\to$ prior knowledge is the primary cause of expert disagreement.

3. **LLM-Assisted Reasoning**

    - Execute the workflow step-by-step as an LLM program (abductive reasoning: reconstructing intermediate reasoning given the paper and the final decision).
    - LLMs achieve factual consistency close to humans but lag significantly in boolean decision-making (tending to "restate" rather than focus on key arguments).
    - **Human Supervision Mitigates Error Propagation**: Under the input-output conditioning (io-condition), the performance of all LLMs improves significantly.

## Key Experimental Results

### Causal Effect (ACE)

| Reasoning Step | ACE | Explanation |
|---------|-----|------|
| Alignment between conclusion and research question (step33) | **0.37** | Largest impact |
| Relevance of conclusion (step46) | 0.20 | |
| Writing clarity (step48) | 0.20 | Style has a causal effect |
| Whether it is a methodology paper (step4) | 0.02 | Almost no effect |

### LLM Performance (Abductive Reasoning)

| Model | BERT-F1↑ | Boolean F1↑ |
|------|----------|-------------|
| Human baseline | 0.55 | **0.71** |
| GPT-4o | 0.48 | 0.52 |
| GPT-3.5 | 0.46 | 0.48 |
| Mixtral | 0.45 | 0.45 |
| LLama3-8B | 0.42 | 0.40 |

### Key Findings
- **Prior knowledge is the main cause of disagreement**: The semantic similarity of the "infer-knowledge" steps is the lowest (~0.45), while that of the "extract" steps is the highest (~0.65).
- **Writing style has a causal effect**: Good writing directly improves assessment outcomes in a causal manner, rather than just through indirect association.
- **"Reversal through figure improvement"**: 60% of negative reviews can be reversed by intervening on just one node (figure-text alignment).
- **LLM "restating" tendency**: High factual coverage but low judgment accuracy—revealing that LLMs weigh factors differently compared to humans.
- **Human supervision is crucial**: Step-by-step correction significantly improves LLM performance; STRICTA naturally supports efficient human-AI collaboration.

## Highlights & Insights
- **Innovative combination of causal frameworks and NLP**: First study to systematically apply SCMs to text assessment—moving beyond "which features are important" to "what changes would alter the outcome."
- **Practical value of counterfactual analysis**: Can inform authors that "if figure-text alignment is improved, the assessment will flip"—which is much more actionable than generic feedback.
- **Academic implication of "writing style having a causal effect"**: Empirical evidence reveals that reviewers' scientific judgment is indeed causally influenced by writing quality.
- **LLM Program Architecture**: Using the workflow as an execution scaffold naturally prevents "skipping steps" and "hallucinations."

## Limitations & Future Work
- **Limited to the biomedical domain**: The reasoning structure for assessments in other fields may differ.
- **Limited data scale**: 22 papers / 93 executions is relatively small for robust SCM estimation.
- **Lack of comparison with existing review-assistance systems**.
- **Insufficient LLM evaluation metrics**: Correlation between automated metrics and human judgment is moderate.

## Related Work & Insights
- **vs. Automated Peer Review Generation (D'Arcy et al., 2024)**: They generate complete reports, whereas this work focuses on the underlying reasoning process—which is deeper.
- **vs. CoT Prompting**: CoT is a free-form chain of thought, while STRICTA is a structured reasoning process with causal constraints—making it more controllable.
- **Insights**: The SCM + LLM program paradigm can be extended to other tasks requiring structured judgment, such as medical diagnosis and legal arguments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of SCM and text assessment is entirely novel; the formalized framework has theoretical depth, and the dataset is unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Involves 40+ experts, 4 LLMs, causal analysis, and collaboration simulation, though the dataset is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logic chain from Definition $\to$ Instantiation $\to$ Analysis $\to$ LLM.
- Value: ⭐⭐⭐⭐⭐ Pioneering contribution to AI-based peer review; the causal findings have real-world implications for academic publishing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] Graph-Structured Trajectory Extraction from Travelogues](graph-structured_trajectory_extraction_from_travelogues.md)
- [\[ACL 2025\] DRS: Deep Question Reformulation With Structured Output](drs_deep_question_reformulation_with_structured_output.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)

</div>

<!-- RELATED:END -->
