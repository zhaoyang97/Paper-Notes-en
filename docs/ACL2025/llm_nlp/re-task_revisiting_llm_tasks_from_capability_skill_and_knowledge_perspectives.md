---
title: >-
  [Paper Note] Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives
description: >-
  [ACL 2025][LLM (Other)][Chain-of-Thought] Drawing on Bloom's Taxonomy and Knowledge Space Theory, this paper proposes the Re-TASK framework to revisit LLM tasks from a three-layer perspective of "capability item - skill - knowledge." It designs the Re-TASK prompting strategy to enhance Chain-of-Thought (CoT) performance on domain tasks through targeted knowledge injection and skill adaptation, achieving up to a 45% improvement on legal tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Chain-of-Thought"
  - "Capability Decomposition"
  - "Knowledge Injection"
  - "Skill Adaptation"
  - "Bloom's Taxonomy"
  - "Knowledge Space Theory"
  - "Domain Tasks"
date: 2026-05-08
content_hash: 571a813fe8a7ff88
---

# Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives

**Conference**: ACL 2025  
**arXiv**: [2408.06904](https://arxiv.org/abs/2408.06904)  
**Authors**: Zhihu Wang (Huawei), Shiwan Zhao (Nankai Univ.), Yu Wang (Xi'an Jiaotong Univ.) et al.
**Code**: [GitHub](https://github.com/Uylee/Re-TASK)  
**Area**: LLM/NLP  
**Keywords**: Chain-of-Thought, Capability Decomposition, Knowledge Injection, Skill Adaptation, Bloom's Taxonomy, Knowledge Space Theory, Domain Tasks

## TL;DR

Drawing on Bloom's Taxonomy and Knowledge Space Theory, this paper proposes the Re-TASK framework to revisit LLM tasks from a three-layer perspective of "capability item - skill - knowledge." It designs the Re-TASK prompting strategy to enhance Chain-of-Thought (CoT) performance on domain tasks through targeted knowledge injection and skill adaptation, achieving up to a 45% improvement on legal tasks.

## Background & Motivation

### Background
Chain-of-Thought (CoT) has become the core paradigm for LLMs to solve complex problems by decomposing a complex task into a sequence of subtasks to achieve divide-and-conquer. However, on domain-specific tasks (Law, Finance, STEM), CoT often fails. LLMs struggle to both accurately decompose tasks and effectively execute subtasks, fundamentally due to a lack of domain knowledge and specialized capabilities.

### Limitations of Prior Work
- **CoT only provides a workflow perspective**: It focuses on "how to decompose steps," ignoring the specific capabilities, knowledge, and skills required for each step.
- **RAG only addresses knowledge absence**: Retrieval-Augmented Generation can inject knowledge, but models may lack the skill adaptation capability to effectively apply the retrieved knowledge.
- **Few-shot ICL exemplars lack targeted specificity**: Randomly selected few-shot exemplars do not necessarily cover the critical capability items required for the task.
- **Existing evaluation frameworks like KoLA are limited to evaluation**, failing to propose systematic capability enhancement methods.

### Design Motivation
Drawing from educational theories, this work proposes a novel **Chain-of-Learning (CoL)** perspective: the successful completion of a task depends on sequential mastery of multiple capability items, each composed of knowledge and skills. By identifying and reinforcing these capability items, CoT performance can distinguishably and systematically improve.

## Method

### Overall Architecture: Re-TASK Theoretical Model

The Re-TASK framework decomposes LLM tasks into four hierarchical concepts:

1. **Task**: A mapping from input $x$ to output $y$, formulated as $\mathbf{T}(ctx; I; x) = y$, where $I$ represents instructions and $ctx$ represents optional context.
2. **Capability Item**: Specific demonstrations or exercises required to complete the task, guiding the LLM to apply specific skills to relevant knowledge, achieving knowledge-skill adaptation.
3. **Knowledge**: Includes factual, conceptual, and procedural domain knowledge (corresponding to the knowledge dimension of Bloom's Taxonomy).
4. **Skill**: Corresponds to the cognitive process dimension of Bloom's Taxonomy, including remembering/retrieving, understanding, applying, etc.

The completion of task $\mathbf{T}$ requires the sequential mastery of multiple capability items $C_{ij}$, where $i$ denotes the subtask number and $j$ represents the capability item index associated with that subtask. $C_{01}$ represents the overall procedural knowledge, and $C_{02}$ represents the application of this knowledge (similar to the CoT process).

### Key Designs 1: Capability Item Construction

Identification and construction of three core capability items:

- **Knowledge Retrieval**: Identifies task-relevant knowledge points and retrieves them from external sources, or recalls knowledge stored within the LLM. Knowledge itself is treated as a special capability item (with "recalling/retrieving" as the default skill).
- **Instances of Conceptual Knowledge**: Illustrating conceptual knowledge through concrete examples to reinforce understanding.
- **Execution of Procedural Knowledge**: Demonstrating how to execute procedural knowledge in sequential steps, such as the legal sentencing reasoning process.

Construction pipeline: First, an LLM is used for task decomposition to obtain overall procedural knowledge $C_{01}$, and then a CoT demonstration is generated as the knowledge application capability item $C_{02}$. Finally, relevant knowledge $C_{i1}$ and application demonstrations $C_{i2}$ are generated for each subtask.

### Key Designs 2: Re-TASK Prompting Strategy

Two versions of prompting strategies are designed:

**Re-TASK (Lite)**: Contains only capability items for the overall task.
- Places the overall procedural knowledge $C_{01}$ (knowledge injection) and its application demonstration $C_{02}$ (skill adaptation) into the prompt.
- Equivalent to a "single demonstration + structured knowledge," with a token overhead comparable to One-shot CoT.

**Re-TASK (Full)**: Contains all available capability items.
- Arranges capability items $C_{ij}$ of each subtask according to dependency relationships (Chain-of-Learning).
- For each subtask, knowledge recall capability items are presented first, followed by comprehension/application capability items.
- Finally, the overall procedural knowledge $C_{01}$ and application $C_{02}$ are provided.

### Key Designs 3: CoT Failure Attribution Analysis

The framework attributes CoT failure to two types of capability deficiencies:
- **Knowledge insufficiency**: Missing knowledge due to a lack of domain-specific data in LLMs or data freshness issues.
- **Skill adaptation insufficiency**: Even if knowledge is available, the LLM fails to apply it effectively. This explains why pure RAG knowledge injection has limited effectiveness and calls for skill adaptation demonstrations.

## Key Experimental Results

### Legal Domain (Sentencing Prediction Task, CAIL Dataset)

| Method | Llama3-Chinese-8B | Yi-1.5-9B | Qwen1.5-7B | Avg. Gain |
|------|-------------------|-----------|------------|---------|
| Zero-shot CoT | 54.00 | 40.00 | 33.50 | - |
| Zero-shot CoT + SC | 54.50 | 40.50 | 33.50 | +0.33 |
| One-shot CoT | 53.67 | 66.50 | 36.17 | +9.61 |
| Three-shot CoT | 56.33 | 70.17 | 38.50 | +12.50 |
| Step-Back | 72.50 | 72.50 | 36.50 | +18.00 |
| Re-TASK (+K0, Knowledge Only) | 60.50 | 57.50 | 44.00 | +11.50 |
| **Re-TASK (Lite)** | **78.50** | **85.00** | **45.50** | **+27.17** |

The legal domain shows the most significant improvement, with Re-TASK (Lite) boosting Yi-1.5-9B by 45 percentage points and achieving an average gain of 27.17%, far exceeding all baselines.

### Financial Domain (FinanceIQ Exam Task)

| Method | Llama3-Chinese-8B | Yi-1.5-9B | Qwen1.5-7B | Avg. Gain |
|------|-------------------|-----------|------------|---------|
| Zero-shot CoT | 36.52 | 53.93 | 43.82 | - |
| Three-shot CoT | 34.27 | 63.82 | 46.07 | +3.30 |
| Step-Back | 30.90 | 66.85 | 44.38 | +2.62 |
| Re-TASK (Lite) | 38.20 | 61.80 | 49.44 | +5.06 |
| **Re-TASK (Full)** | **52.81** | **73.60** | **51.69** | **+14.61** |

In the financial domain, Re-TASK (Full) achieves an average gain of 14.61%, which is significantly higher than Three-shot CoT's 3.30%, verifying the added value of capability items at the subtask level.

### STEM Domain (MMLU Math/Physics/Biology)

| Domain | Method | Llama3-8B | Mistral-7B | Qwen1.5-7B | Avg. Gain |
|------|------|-----------|------------|------------|---------|
| Mathematics | Zero-shot CoT | 40.58 | 24.28 | 36.96 | - |
| Mathematics | Re-TASK (Lite) | 51.81 | 28.99 | 43.84 | +7.61 |
| Physics | Zero-shot CoT | 57.84 | 37.25 | 42.16 | - |
| Physics | Re-TASK (Lite) | 60.78 | 44.12 | 50.98 | +6.21 |
| Biology | Zero-shot CoT | 76.39 | 57.64 | 59.72 | - |
| Biology | Re-TASK (Lite) | 88.19 | 79.17 | 81.25 | +18.29 |

The biological domain experiences the largest improvement (+18.29%). In contrast, the performance of Step-Back on STEM drops significantly (-15.28%), indicating that the high-level principles generated by small models themselves are unreliable in quality.

## Key Findings

1. **Knowledge injection is necessary but not sufficient**: Re-TASK (+K0) only injects knowledge, leading to an average gain of 11.50%. When skill adaptation demonstrations are added (Re-TASK Lite), the gain jumps to 27.17%, illustrating that skill adaptation is a key bottleneck.
2. **Capability items are more effective than random exemplars**: Under the same token budget, Re-TASK (Lite) outperforms random exemplars of One-shot CoT using only a single capability item demonstration. Re-TASK (Full) with a combination of three capability items significantly exceeds Three-shot CoT (14.61% vs. 3.30%).
3. **Model scaling cannot replace capability enhancement**: During the scaling of Qwen1.5 from 7B to 14B and 32B, both Zero-shot CoT and Re-TASK (Lite) improve in parallel, suggesting that the benefits of Re-TASK remain effective on larger models.
4. **Domain knowledge intensity determines the magnitude of improvement**: Law (+27.17%) > Biology (+18.29%) > Mathematics (+7.61%). The more knowledge-intensive and professional the domain is, the more significant the Re-TASK improvement becomes.
5. **Step-Back is unstable on small models**: Small models struggle to generate effective high-level 'abstraction principles', leading to a sharp decline in STEM accuracy.

## Highlights & Insights

- **Innovative integration of educational theory and LLMs**: It introduces the "knowledge dimension × cognitive process dimension" matrix of Bloom's Taxonomy and the learning path concept of Knowledge Space Theory into LLM task analysis, establishing a theoretical framework that transcends pure prompt engineering.
- **Novel concept of Chain-of-Learning (CoL)**: Compared to the "workflow perspective" of CoT, CoL provides a "learning perspective", revealing the hierarchical dependency structure of task capabilities.
- **High practicality**: The process of constructing capability items is straightforward (decompose task → identify knowledge → generate demonstrations), allowing direct application to any domain task.
- **Good efficiency**: The token overhead of Re-TASK (Lite) is comparable to One-shot CoT but yields vastly superior results, proving highly cost-effective.
- **Diagnostic value**: The framework not only enhances performance but also diagnoses the root causes of CoT failure—whether it is a lack of knowledge or insufficient skill adaptation.

## Limitations & Future Work

- **Unoptimized capability item generation**: It relies on LLMs to directly generate capability items without introducing retrieval processes or offline knowledge base matching, which necessitates integration with RAG for practical deployment.
- **Lack of in-depth cross-domain analysis**: The level of improvement varies widely across different domains (Law 27% vs. Math 7%), but the reasons have not been systematically analyzed.
- **Limited testing on small open-source models**: Experiments are confined to open-source models at the 7B-32B scale and have not been validated on closed-source LLMs such as GPT-4.
- **Manual predefinition of capability item types**: It requires predefining capability item categories (knowledge retrieval/comprehension/application) for each task, limiting the level of automation.
- **Small scale of legal/financial datasets**: With only 200 test cases for Law and 178 for Finance, statistical significance awaits verification on larger scales.

## Related Work & Insights

- **Bloom's Taxonomy → LLM Capability Modeling**: The KoLA benchmark has utilized a simplified version of Bloom's Taxonomy (four levels: remember, understand, apply, create) for LLM evaluation. Re-TASK further applies it to capability enhancement.
- **Skill-it (Chen et al. 2024)**: Formalizes skill concepts and sequential skill learning from the training data perspective. In contrast, Re-TASK achieves skill enhancement from the perspective of inference-time prompt design.
- **RAG → A special case of knowledge injection**: Re-TASK views RAG as a specific capability item of "knowledge retrieval" and points out that knowledge injection alone is insufficient without skill adaptation.
- **Insights for CoT research**: Most existing CoT improvements (Self-Consistency, Plan-and-Solve, Step-Back) remain at the workflow level. Re-TASK suggests a new paradigm of redesigning prompts from the perspective of capability dependencies.

## Rating

- Novelty: ⭐⭐⭐⭐ — The interdisciplinary integration of educational theory and LLM prompt engineering is unique, and the Chain-of-Learning concept is innovative.
- Experimental Thoroughness: ⭐⭐⭐ — Covers 5 datasets across 3 domains, but the dataset scale is relatively small, and it has not been verified on closed-source LLMs.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical framework is clearly structured with rigorous definitions, though some concepts (such as the essential difference between capability items and few-shot exemplars) could be further clarified.
- Value: ⭐⭐⭐⭐ — Provides a systematic methodology to analyze and enhance LLM capabilities on domain tasks, which is practical and easy to generalize.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)
- [\[ACL 2025\] A Modular Dataset to Demonstrate LLM Abstraction Capability](a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)
- [\[ICLR 2026\] Discovering Novel LLM Experts via Task-Capability Coevolution](../../ICLR2026/llm_nlp/discovering_novel_llm_experts_via_task-capability_coevolution.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)

</div>

<!-- RELATED:END -->
