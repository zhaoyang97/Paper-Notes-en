---
title: >-
  [Paper Note] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents
description: >-
  [ACL 2025][LLM (Other)][student simulation] To address the difficulty of LLMs in simulating the erroneous behaviors of low-performing students, this paper proposes a training-free framework based on knowledge graph cognitive prototypes. By utilizing a three-stage process—cognitive state modeling $\to$ behavior prediction $\to$ beam search self-refinement—the framework generates realistic student responses, achieving a 100% improvement in simulation accuracy on the Student_100…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "student simulation"
  - "cognitive prototype"
  - "knowledge graph"
  - "beam search refinement"
  - "LLM agent"
date: 2026-05-08
content_hash: a5911c2a2ec9c25c
---

# Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents

**Conference**: ACL 2025  
**arXiv**: [2505.19997](https://arxiv.org/abs/2505.19997)  
**Code**: [https://mccartney01.github.io/student_sim](https://mccartney01.github.io/student_sim)  
**Area**: LLM NLP / AI in Education  
**Keywords**: student simulation, cognitive prototype, knowledge graph, beam search refinement, LLM agent

## TL;DR
To address the difficulty of LLMs in simulating the erroneous behaviors of low-performing students, this paper proposes a training-free framework based on knowledge graph cognitive prototypes. By utilizing a three-stage process—cognitive state modeling $\to$ behavior prediction $\to$ beam search self-refinement—the framework generates realistic student responses, achieving a 100% improvement in simulation accuracy on the Student_100 dataset.

## Background & Motivation

**Background**: LLMs are widely applied in educational scenarios for personalized tutoring, curriculum design, and adaptive assessment. Simulating student behavior using LLM-based agents is a critical approach for evaluating teaching strategies and testing intelligent tutoring systems.

**Limitations of Prior Work**: Current LLMs are fundamentally trained to be "helpful assistants" and tend to generate correct answers. When prompted to simulate students with low cognitive levels, they still generate overly "advanced" responses, failing to reproduce the natural errors and imperfect performances of actual students during learning. Experiments show that in naive prompt simulations of 15 students with varying proficiency, LLMs systematically overestimate the cognitive scores of low-ability students.

**Key Challenge**: There is a fundamental conflict between the training objectives of LLMs (generating correct and helpful answers) and the need to simulate "error-making" students. While fine-tuning methods (such as MalAlgoPy) can introduce error patterns, they risk contaminating the model's own knowledge and overlook the personalized characteristics of errors—the error patterns of different students should be determined by their individual cognitive states.

**Goal**: (1) How to enable LLMs to simulate students of diverse cognitive levels without modifying model weights? (2) How to accurately predict what specific errors a particular student will make on a new task? (3) How to generate realistic student answers that contain both correct explanations and plausible errors?

**Key Insight**: Departing from cognitive science, this work explicitly models students' mastery of various knowledge concepts (cognitive prototypes) using knowledge graphs, rather than relying on implicit neural network representations. The natural language nodes of knowledge graphs are inherently suitable for interacting with LLMs, enabling precise behavior prediction in a training-free manner.

**Core Idea**: Build student cognitive prototypes using knowledge graphs to predict behavior, and then apply beam search self-refinement to generate responses consistent with the predicted behavior, achieving training-free, multi-level student simulation.

## Method

### Overall Architecture
The input consists of $M=40$ historical learning records of a student, denoted as $P=\{(t_i, b_i, s_i)\}$ (tasks, behaviors, and responses), and the output is the simulated responses for $N=10$ new tasks. The three-stage pipeline includes: (1) constructing a knowledge graph-based cognitive prototype from historical records; (2) mapping the prototype to new tasks to predict behavior; and (3) generating responses via beam search self-refinement based on the predicted behavior.

### Key Designs

1. **Knowledge Graph-based Cognitive Prototype**:

    - **Function**: Constructs an explicit and interpretable representation of cognitive states from a student's historical records.
    - **Mechanism**: Iteratively processes each learning record via four steps: (a) Concept Extraction: generating high-level task descriptions $d_i = \pi_{desc}(t_i)$ to identify advanced concepts, and then extracting multi-level knowledge concepts combined with the record; (b) Relation Extraction: establishing four types of relations (Prerequisite_of / Used_for / Hyponym_of / Part_of) between concepts as graph edges; (c) Local Cognitive Analysis: determining the mastery level (Good/Bad) of each concept based on the student's performance; (d) Global Prototype Construction: summarizing the global cognitive state by synthesizing the historical frequencies of Good/Bad for each concept after processing all records.
    - **Design Motivation**: Compared to implicit neural networks, natural language knowledge graphs offer strong interpretability and can be directly used as context inside LLM prompts without requiring any training.

2. **Concept-aware Behavior Prediction**:

    - **Function**: Predicts the expected behavior of a student on a new task (correctness/error and the specific error type).
    - **Mechanism**: Generates concept descriptions for a new task, calculates similarity with knowledge graph nodes, selects the top-$p=5$ most relevant concepts and their cognitive states, retrieves historical records containing the most overlap with these concepts as references, and comprehensively predicts the behavior as $\hat{b}_j = \pi_{pred}(t_j, [C_1,...,C_p], P_{\hat{j}})$.
    - **Design Motivation**: Traditional methods retrieve tasks based on textual similarity, which easily mismatches tasks requiring entirely different knowledge concepts due to surface-level similarity (e.g., retrieving "calculate double" for "calculate factorial"). Concept mapping, however, matches tasks at a deeper knowledge level.

3. **Self-Refinement with Beam Search**:

    - **Function**: Generates student responses that precisely align with the predicted behaviors.
    - **Mechanism**: Generates an initial weak response $\hat{s}_j^1$ and iteratively refines it over $L=3$ rounds. In each round, it samples $B=2$ candidates, which are scored by $\pi_{value}$ (0-1 score assessing behavior alignment), and selects the highest-scoring candidate for the next round until the score exceeds the threshold $\delta=0.9$ or the maximum iteration limit is reached.
    - **Design Motivation**: Single-turn generation by LLMs struggles to accurately simulate specific error patterns. Iterative refinement paired with self-evaluation can progressively correct biases, and multi-candidate sampling in beam search increases the probability of finding a suitable response.

### Loss & Training
The entire framework is training-free. All components ($\pi_{desc}$, $\pi_{node}$, $\pi_{edge}$, $\pi_{local}$, $\pi_{global}$, $\pi_{pred}$, $\pi_{refine}$, $\pi_{value}$) utilize the same LLM. A temperature of 0 is applied to ensure reproducibility, with diverse sampling enabled only for $\pi_{refine}$.

## Key Experimental Results

### Main Results
Evaluated on the Student_100 dataset (consisting of 100 students and 5,000 Python programming learning records), the evaluation metrics include behavior prediction accuracy (Acc), behavior consistency (Con1), and response consistency (Con2, scored 1-5):

| Model | Method | Acc | Con1 | Con2 |
|------|------|-----|------|------|
| GPT-4o | Similarity + IO | 0.47 | 2.62 | 2.65 |
| GPT-4o | Prototype Mapping + Refine (Ours) | **0.94** | **3.77** | **3.65** |
| Claude-3.5 | Similarity + IO | 0.61 | 3.03 | 2.82 |
| Claude-3.5 | Prototype Mapping + Refine (Ours) | **0.65** | **3.09** | **3.09** |
| LLaMA-3.3-70B | Similarity + Refine | 0.41 | 2.45 | 1.99 |
| LLaMA-3.3-70B | Prototype Mapping + Refine (Ours) | **0.61** | **2.99** | **2.69** |

### Ablation Study

| Configuration | GPT-4o Acc | GPT-4o Con2 | Description |
|------|-----------|-------------|------|
| Full model | 0.94 | 3.65 | Full framework |
| w/o Knowledge Graph (text similarity retrieval only) | 0.47 | 2.65 | Acc drops by 50%, highlighting the criticality of cognitive prototype |
| w/o Global Cognitive Construction (local state only) | 0.66 | 2.89 | Global synthesis is necessary |
| w/o Behavior Prediction (direct generation of response) | - | 2.70 | Lacks behavioral description guidance, leading to poor response quality |
| w/o Self-Refinement (IO only) | 0.94 | 3.50 | Refinement contributes +0.15 |
| w/o Self-Evaluation (refinement only, no grading) | 0.94 | 3.52 | Guidance from self-evaluation is important |

### Key Findings
- GPT-4o's behavior prediction accuracy jumps from 0.47 in the baseline to 0.94 (+100%), confirming the core value of cognitive prototype.
- Stronger LLMs benefit more from self-refinement (GPT-4o: +0.15 vs. LLaMA: +0.08) due to their superior self-evaluation capabilities.
- Simulating students with high cognitive levels is easier than simulating those with low levels (Con2 is positively correlated with cognitive score), as generating correct solutions is simpler than simulating specific errors.
- As historical records increase from 10 to 40, Acc steadily rises from ~0.7 to 0.94, indicating that more data helps construct more precise cognitive prototypes.

## Highlights & Insights
- The cognitive prototype construction uniting knowledge graphs and LLMs is an ingenious "change the input, not the model" approach. Natural language graph nodes seamlessly embed into prompts to achieve training-free personalized simulation, which can be extended to any scenario requiring personalized behavior modeling.
- Instead of aiming to "generate better responses," the beam search self-refinement aims to "generate poorer but plausible responses." This reverse optimization goal design is highly novel.
- Replacing text similarity retrieval with concept-level behavior prediction is fundamentally about "understanding what the student does not know" rather than "finding similar questions." This paradigm shift can be transferred to personalized problem generation in intelligent tutoring systems.

## Limitations & Future Work
- The method is evaluated only in the Python programming domain; subjects requiring different reasoning chains, such as mathematics, may require adjustments in concept extraction strategies.
- The framework relies on multiple LLM calls across 8 components, incurring high computational costs, as experimenting with 100 students requires a substantial number of API invocations.
- The concept granularity in the knowledge graph is determined by the LLM, lacking standard ontology constraints, which may impact cross-student comparability.
- Multimodal signals (e.g., student operation behaviors, coding temporal patterns) are not considered, as the model only relies on text-based behavioral modeling.

## Related Work & Insights
- **vs. MalAlgoPy**: MalAlgoPy defines 20 types of equation transformation errors and fine-tunes the model, which risks contaminating the model's knowledge and lacks personalization. Ours is training-free and adaptively introduces errors matching individual student cognitive states.
- **vs. Knowledge Tracing Methods (e.g., DKT)**: Traditional methods parameterize cognitive states using implicit neural networks and lack interpretability. Ours explicitly models cognitive states using natural language knowledge graphs, which can directly interact with LLMs.
- **vs. Level-based Methods**: Merely estimating student levels based on historical accuracy is too coarse-grained and ignores the concept-specificity of errors. The cognitive prototype can capture a fine-grained understanding of each knowledge concept.

## Rating
- Novelty: ⭐⭐⭐⭐ The training-free framework combining cognitive prototypes with LLMs is novel, though beam search self-refinement is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 LLMs, 18 configuration combinations, ablation studies, parameter analysis, and human evaluation, and extends to Java/C++.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, intuitive illustrations, and detailed method description.
- Value: ⭐⭐⭐⭐ Direct practical value for personalized simulation in AI in education, with a generalizable framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility](leveraging_human_production-interpretation_asymmetries_to_test_llm_cognitive_pla.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)

</div>

<!-- RELATED:END -->
