---
title: >-
  [Paper Note] ExpeTrans: LLMs Are Experiential Transfer Learners
description: >-
  [ACL 2025][LLM (Other)][Experience Transfer] ExpeTrans proposes an autonomous experience transfer framework. It mimics human cognitive intelligence to automatically transfer problem-solving experience from existing source tasks to newly encountered target tasks. This framework effectively improves LLM performance across 13 datasets without requiring manual experience collection for every new task.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Experience Transfer"
  - "Task Generalization"
  - "Prompt Engineering"
  - "Cognitive Intelligence"
  - "Autonomous Learning"
date: 2026-05-08
content_hash: 4db1be71852b1679
---

# ExpeTrans: LLMs Are Experiential Transfer Learners

**Conference**: ACL 2025  
**arXiv**: [2505.23191](https://arxiv.org/abs/2505.23191)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Experience Transfer, Task Generalization, Prompt Engineering, Cognitive Intelligence, Autonomous Learning

## TL;DR

ExpeTrans proposes an autonomous experience transfer framework. It mimics human cognitive intelligence to automatically transfer problem-solving experience from existing source tasks to newly encountered target tasks. This framework effectively improves LLM performance across 13 datasets without requiring manual experience collection for every new task.

## Background & Motivation

**Background**: Recent studies show that providing textual task-solving experiences to LLMs via prompts can effectively improve their performance. These experiences typically include task descriptions, problem-solving strategies, and common error patterns, which assist LLMs in better understanding and completing tasks.

**Limitations of Prior Work**: Existing approaches to obtaining such experiences require extensive manual annotation or lengthy automated collection processes—where experience must be systematically gathered from scratch for every new task. As the types of tasks in user queries become increasingly diverse, this task-by-task collection approach is practically unfeasible.

**Key Challenge**: While experience yields clear improvements in LLM performance, there is a fundamental conflict between the cost of acquisition and task diversity. Humans can flexibly transfer experiences learned in one domain to new domains, but LLMs currently lack this autonomous transfer capability.

**Goal**: Design a framework that enables LLMs to autonomously transfer accumulated source-task experiences to newly encountered target tasks, bypassing the limitations of task-by-task experience collection.

**Key Insight**: The authors approach this from the perspective of human cognitive intelligence: when faced with new problems, humans automatically recall relevant experiences and perform adaptive transfer. As models with extensive knowledge, LLMs should theoretically possess similar capabilities for experience transfer.

**Core Idea**: Build an autonomous experience transfer framework consisting of four stages: experience extraction, experience selection, experience adaptation, and experience application. This framework enables LLMs to distill generalizable experiences from source tasks and transfer them to target tasks.

## Method

### Overall Architecture

The pipeline of ExpeTrans consists of four stages: (1) extracting structured problem-solving experience from existing source tasks; (2) automatically selecting the most relevant source task experience for a given target task; (3) adapting the selected experience to the specific requirements of the target task; (4) integrating the adapted experience into the prompt to guide the LLM in completing the target task. The input consists of a set of existing source tasks along with their experience library, and a new target task; the output is the enhanced target task prompt and its solution.

### Key Designs

1. **Experience Extraction**:

    - **Function**: Distill structured, transferable experience from the problem-solving processes of source tasks.
    - **Mechanism**: LLMs are leveraged to analyze and summarize the problem-solving records of source tasks, extracting general strategies, key step patterns, and common error avoidance methods. This experience is organized into a structured textual representation, containing dimensions such as strategy descriptions, applicability conditions, and key execution points.
    - **Design Motivation**: Raw problem-solving records contain numerous task-specific details, making direct transfer too noisy. Structured extraction retains the core transferable strategies while removing task-specific surface features.

2. **Experience Selection**:

    - **Function**: Select the most relevant source-task experiences from the experience library for the target task.
    - **Mechanism**: Based on the characteristic description of the target task, semantic matching is used to calculate the relevance between the target task and various source task experiences. The matching strategy considers task type similarity, overlapping capability requirements, and analogical relations in problem structures. The top-k most relevant source task experiences are selected for the next stage.
    - **Design Motivation**: Not all source task experiences are helpful for the target task. Incorrect experience transfer can introduce noise (negative transfer). Precise selection is key to ensuring transfer effectiveness.

3. **Experience Adaptation & Application**:

    - **Function**: Adapt the selected experiences to the specific requirements of the target task and integrate them into the prompt.
    - **Mechanism**: LLMs analyze the differences between the source task experiences and the target task, targetedly modifying terminology and strategy details within the experience. For example, a source task experience of "focus on emotional vocabulary" for sentiment analysis can be adapted to "focus on topic indicator words" for a target task of topic classification. The adapted experiences are formatted as part of the prompt, serving as reference guidelines when the LLM solves the target task.
    - **Design Motivation**: Directly using source task experiences might be inapplicable due to domain gaps. The adaptation step bridges the semantic gap between source and target.

### Loss & Training

ExpeTrans is a pure inference-time framework and does not involve model parameter updates. All modules are implemented based on the in-context learning capability of LLMs, guiding the model to execute each stage of the task through carefully designed prompt templates.

## Key Experimental Results

### Main Results

Performance comparison across 13 datasets (covering various task types such as classification, reasoning, and generation):

| Method | Average Accuracy | Relative Improvement | Notes |
|------|-----------|---------|------|
| Base LLM (zero-shot) | Baseline | - | No experience prompt |
| Base LLM + Manual Experience | Improved | High manual cost | Manually written task-by-task |
| ExpeTrans (Auto Transfer) | Significant Improvement | Exceeds zero-shot | Autonomous experience transfer |

ExpeTrans outperforms the zero-shot baseline on most datasets, and approaches or even surpasses the performance of manually written experiences on some tasks.

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Full ExpeTrans | Optimal | Synergy of the four modules |
| w/o Experience Selection | Decreased | Randomly selected experience introduces noise |
| w/o Experience Adaptation | Decreased | Direct use of source experience without adaptation |
| w/o Experience Extraction | Significantly Decreased | Using raw problem-solving records is too verbose |

### Key Findings

- The experience selection module has the greatest impact—selecting the wrong experience (negative transfer) not only yields no help but actually degrades performance, demonstrating that precise matching is key to transfer effectiveness.
- The experience adaptation step is particularly crucial for cross-domain transfer; when the domain gap between source and target tasks is large, the performance gains after adaptation are more pronounced.
- The framework demonstrates strong robustness across 13 datasets, finding transferable general strategies even when task types differ significantly.
- Detailed module analysis provides quantitative evidence for the contribution of each component, with 12 charts presenting comprehensive experimental analysis.

## Highlights & Insights

- The entry point of **analogizing human cognitive intelligence** is highly convincing—human experience transfer capability is a core feature of cognitive intelligence, and bringing this capability into LLM systems is a natural and meaningful exploration.
- The **pure inference-time framework** requires no fine-tuning of model parameters and has extremely low deployment costs, allowing plug-and-play integration with any LLM. This "experience-as-prompt" approach offers excellent scalability.
- The framework decomposes experience transfer into four clear stages: extraction, selection, adaptation, and application. Each stage can be independently improved, and this modular design facilitates subsequent iterations.

## Limitations & Future Work

- The framework is entirely based on the in-context learning of LLMs, which is limited by the context window size, preventing the transfer of excessively long experience texts.
- The quality and coverage of the experience library directly affect the transfer performance; if the source task library is too small or biased, it may be difficult to find high-quality transfer sources.
- Computational overhead is not discussed in the paper—all four stages require LLM inference, totaling at least 4 LLM calls, which is significantly slower than simple zero-shot inference.
- Future work can explore experience retrieval combined with vector databases, as well as efficient selection strategies when the experience library scales up.

## Related Work & Insights

- **vs Self-Consistency/CoT**: These methods improve performance by enhancing reasoning pathways, whereas ExpeTrans enhances performance by introducing external experiential knowledge; the two approaches are complementary.
- **vs Few-shot learning**: Traditional few-shot approaches guide the model using examples, whereas ExpeTrans uses distilled strategic experience, providing higher information density.
- **vs Manual prompt engineering**: ExpeTrans automates the acquisition and transfer of experience, reducing manual effort.

## Rating

- Novelty: ⭐⭐⭐⭐ The experience transfer framework introduces cognitive science concepts to LLM prompt design, offering a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 datasets + 12 charts, providing broad coverage and detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ The 9-page paper is compactly structured with clear motivations.
- Value: ⭐⭐⭐⭐ Provides a practical, low-cost solution to improve the task generalization capabilities of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mixtures of In-Context Learners](mixtures_of_in-context_learners.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)

</div>

<!-- RELATED:END -->
