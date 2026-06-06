---
title: >-
  [Paper Note] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems
description: >-
  [ACL2026][Multi-Agent][Research Roadmap Generation] This paper proposes the RoadMap benchmark for research roadmap generation and the RoadMapper multi-agent system. By forming a closed loop with knowledge retrieval…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Research Roadmap Generation"
  - "DPO Evaluator"
  - "Structured Content Generation"
  - "RoadMap Benchmark"
date: 2026-05-08
content_hash: 1851d550311539d1
---

# RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems

**Conference**: ACL2026  
**arXiv**: [2604.27616](https://arxiv.org/abs/2604.27616)  
**Code**: https://github.com/BUPT-Reasoning-Lab/RoadMapper  
**Area**: LLM Evaluation / Multi-Agent Systems / AI for Science  
**Keywords**: Research Roadmap Generation, Multi-Agent, DPO Evaluator, Structured Content Generation, RoadMap Benchmark

## TL;DR
This paper proposes the RoadMap benchmark for research roadmap generation and the RoadMapper multi-agent system. By forming a closed loop with knowledge retrieval, logical/granularity critiques, revision, and a DPO-trained evaluator, the system achieves an average improvement of approximately 7-9 points over direct prompting on complex Chinese and English research problems, significantly reducing the time cost for experts to design roadmaps.

## Background & Motivation
**Background**: Researchers frequently use tables, flowcharts, knowledge graphs, or mind maps to organize complex knowledge. However, these structured contents primarily serve information display rather than decomposing a research problem into an actionable, hierarchically clear solution route. While existing LLMs can write plans, list steps, or perform long-form Q&A, a specialized task definition and evaluation benchmark for "how to solve a complex research problem step-by-step" is still lacking.

**Limitations of Prior Work**: Manual roadmap creation requires experts to consult vast amounts of specialized knowledge and undergo repeated revisions, which is costly and time-consuming. When prompted directly to generate roadmaps, LLMs tend to suffer from three issues: insufficient specialized knowledge, unreasonable task decomposition, and chaotic logical relationships between nodes. Especially in scientific scenarios, a roadmap is not a simple checklist; it must balance intellectual depth, breadth, execution sequence, and coverage of key steps.

**Key Challenge**: Roadmap generation requires both expert-level domain knowledge and project-management-style structural consistency and logical progression. A single prompt struggle to simultaneously handle knowledge completion, hierarchical decomposition, logical verification, granularity control, and termination judgment. Thus, it is necessary to decompose the complex generation process into multiple collaborative sub-roles.

**Goal**: The authors first define the research roadmap generation task and construct the RoadMap benchmark. Based on this, they design RoadMapper, allowing the LLM to enter a "Critique-Revise-Evaluate" iterative loop after knowledge enhancement. Finally, the quality and efficiency of the roadmaps are validated through automatic metrics, ablation studies, and expert evaluations.

**Key Insight**: The paper observes that high-quality roadmaps often resemble the abstraction of graduate theses: theses naturally contain research problems, key technologies, and solution paths. Therefore, the authors extract research problems and Skill Points from Master's and PhD theses, followed by expert-guided generation of "gold" roadmaps, creating a data source with both professional depth and structural constraints.

**Core Idea**: A multi-agent system decomposes roadmap generation into six roles: "Initial Draft, Knowledge Enhancement, Logic Critique, Granularity Critique, Revision, and Evaluation." A DPO-trained evaluator learns expert preferences, enabling the system to generate structured answers closer to expert roadmaps within limited iterations.

## Method
The key to RoadMapper is not simply applying an agent framework, but formalizing the "research roadmap" into a verifiable tree-structured Markdown format, then designing data, retrieval, evaluation, and iteration mechanisms around this format. The process consists of two layers: an offline layer to build RoadMap and Skill-Repo, and an online layer where RoadMapper generates roadmaps for new research problems.

### Overall Architecture
The input is a complex research problem, and the output is a tree-like roadmap adhering to Markdown hierarchical rules. The system starts with an Init Agent generating an initial draft, followed by a Knowledge Agent retrieving relevant skill points from Skill-Repo to supplement the roadmap. Then, it enters an iterative loop (up to 5 rounds): a Logic Critique Agent checks the logic between parent-child and sibling nodes; a Granularity Critique Agent checks if nodes are too coarse or fine; a Revise Agent rewrites the roadmap based on both critiques; and an Evaluate Agent determines if the passing standard is met. If the evaluator accepts the result, it stops early; otherwise, it continues to the next round.

The RoadMap benchmark is derived from Master's and PhD theses from ProQuest and CNKI. Papers were filtered by publication time, author identity, and university prestige. Gemini 2.5 Flash extracted core research problems and skill points, which were then refined by experts into gold roadmaps. The final dataset includes 1,705 gold roadmaps, 8,436 skill points, 10 research fields, and 5 research types, covering both Chinese and English.

### Key Designs
1. **RoadMap Benchmark and Format Constraints**:
	- **Function**: Transforms the vague "research roadmap" into an evaluable, trainable, and iteratively optimizable generation task.
	- **Mechanism**: Each roadmap is represented as a Markdown tree, with each line containing a level, number, and title (e.g., hierarchical numbering like `1.3.2`). Evaluation considers not just text, but depth, out-degree, and key step coverage.
	- **Design Motivation**: Without strong format constraints, models tend to output simple suggestion lists, making structural quality comparison difficult. The tree format enables structural metrics like `DegreeScore` and `DepthScore`.

2. **Knowledge Enhancement and Dual Critique Agents**:
	- **Function**: Supplement specialized knowledge, fix logical errors, and control node granularity respectively.
	- **Mechanism**: The Knowledge Agent retrieves Top-K skill points from Skill-Repo to extend the draft. The Logic Critique Agent checks if parent-child nodes have a direct refinement relationship and if siblings are parallel or sequential. The Granularity Critique Agent checks for over-decomposition or info-density issues.
	- **Design Motivation**: Roadmap errors vary; missing knowledge, logical chaos, and improper granularity require different diagnostic perspectives. Splitting logic and granularity critiques prevents a single critic from conflating multiple issues.

3. **DPO-Trained Evaluate Agent and Early-Stopping Loop**:
	- **Function**: Informs the system when to stop iterating and reduces the cost of ineffective multi-turn dialogues.
	- **Mechanism**: The authors used Qwen3-32B to generate candidate evaluations, then had 7 experts vote for the optimal and suboptimal evaluations to construct 818 preference pairs. The DPO objective aligns the evaluator with expert standards. During inference, the evaluator outputs a score and reasoning; it stops if the score reaches a threshold of 80.
	- **Design Motivation**: A bottleneck in multi-agent systems is "modifying more without necessarily getting better." By aligning the evaluator with expert preferences, early stopping saves time and tokens while avoiding over-revision that might damage the structure.

### Loss & Training
Training is focused solely on the Evaluate Agent. Given preferred and non-preferred samples from expert preferences, DPO is applied to increase the probability of optimal outputs relative to a reference model. Negative samples were specifically chosen as the "second-highest vote" rather than the worst candidate to force the model to learn fine-grained evaluation criteria. During inference, the maximum iterations are set to 5, the passing score to 80, and the Knowledge Agent retrieves up to 30 skill points.

## Key Experimental Results

### Main Results
The paper compares Direct Prompting, Best-of-N, CoT, ReConcile, DyLAN, and RoadMapper across 11 open-source and closed-source models using StepScore, LogicScore, DegreeScore, DepthScore, and Average Score.

| Base Model | Method | EN Avg | CN Avg | Overall Avg | Key Conclusion |
|-----------|--------|--------|--------|-------------|----------------|
| Llama 3.1 8B | Direct Prompting | 61.24 | 60.01 | 60.62 | Small models are weak in roadmap structure and content. |
| Llama 3.1 8B | RoadMapper | 67.68 | 67.95 | 67.82 | Overall Gain: 7.20 points. |
| Llama 3.3 70B | Direct Prompting | 63.16 | 61.04 | 62.10 | StepScore for CN split is only 38.16. |
| Llama 3.3 70B | RoadMapper | 70.69 | 70.20 | 70.45 | EN Gain: 7.53, CN Gain: 9.16. |
| DeepSeek-V3.2 | Direct Prompting | 71.03 | 73.20 | 72.12 | Strong models are still limited by direct prompting. |
| DeepSeek-V3.2 | RoadMapper | 79.14 | 80.08 | 79.61 | EN Gain: 8.11, CN Gain: 6.88. |

### Ablation Study

| Configuration | StepScore | LogicScore | DegreeScore | DepthScore | Avg. | Note |
|---------------|-----------|------------|-------------|------------|------|------|
| w/o Knowledge Agent | 45.44 | 59.60 | 73.11 | 88.76 | 66.73 | StepScore drops by 5.84 without knowledge. |
| w/o Logic Critique Agent | 48.29 | 56.94 | 74.17 | 88.46 | 66.97 | LogicScore shows the most significant drop. |
| w/o Granularity Critique Agent | 47.51 | 58.74 | 74.80 | 87.37 | 67.11 | Lack of granularity control affects overall structure. |
| w/o DPO | 49.83 | 60.20 | 74.67 | 88.25 | 68.24 | DPO contributes approx. 2.21 points. |
| DPO-14B | 49.77 | 60.69 | 75.30 | 88.87 | 68.66 | Smaller evaluator remains effective but lower than 32B. |
| RoadMapper | 51.28 | 62.37 | 77.37 | 90.77 | 70.45 | Complete system achieves highest scores on all four metrics. |

### Key Findings
- **Knowledge Agent**: Vital for key step coverage; removing it caused StepScore to drop from 51.28 to 45.44, indicating that roadmaps cannot be written well based on structural templates alone—professional skill points are essential.
- **Logic Critique Agent**: Contributes most to LogicScore; removing it dropped the score to 56.94, validating the necessity of parent-child/sibling relationship checks.
- **Efficiency**: The early stopping mechanism requires only ~1.64 iterations on average to reach high quality, saving 36.5% of time and 45.8% of tokens compared to ReConcile.
- **Expert Alignment**: Pairwise evaluations from 7 experts showed GPT-4o mini has a 93% match rate with experts, and RoadMapper outperformed base methods in 86% of cases.

## Highlights & Insights
- The most valuable contribution is the complete closed loop of "Task Definition + Benchmark + Method + Evaluation Metrics." While many agent papers only show system performance, this work defines the roadmap structure and evaluation dimensions, providing a reusable baseline for future research.
- The split of dual critique agents is practical: logical relationships and granularity are two different axes of roadmap quality; merging them into one critic often leads to vague feedback. This idea can be transferred to generating course syllabi, experimental plans, or review outlines.
- Selecting the "second-highest vote" as a DPO negative example is clever. Roadmap evaluation is rarely about obvious right/wrong but rather subtle professional judgment. Training the model to distinguish optimal from suboptimal is more realistic than distinguishing good from catastrophically bad.
- The paper treats multi-agent systems as more than just prompt engineering, incorporating system designs like evaluator early-stopping, knowledge base retrieval, and structural metrics, making it more controllable than typical "round-table" agents.

## Limitations & Future Work
- **Data Source**: RoadMap primarily stems from graduate theses. While professional, it may bias toward thesis-style narratives; real-world research projects, industrial R&D, and interdisciplinary problems might require different roadmap forms.
- **Evaluation Bias**: Content metrics (StepScore and LogicScore) rely on GPT-4o mini. Despite high expert alignment, they might still inherit the biases and stability issues of LLM-as-a-judge.
- **Cost**: RoadMapper is more computationally expensive than direct prompting. Early stopping mitigates multi-turn overhead, but the complex system is less suitable for batch calls on low-value, short questions.
- **Model Coverage**: 11 LLMs were tested, but resources limited coverage of all expensive models. Future research could explore smaller evaluators, distilled RoadMapper versions, or lightweight workflows for specific domains.

## Related Work & Insights
- **vs. Structured Content Generation**: Early text-to-table/mindmap/procedural graph tasks focused more on information display. RoadMapper emphasizes actionable roadmaps for "guiding problem solving," extending evaluation from surface format to key steps and logic.
- **vs. General Multi-Agent Discussion**: Methods like ReConcile and DyLAN improve answers via discussion but lack task-specific logic/granularity roles and expert-aligned evaluators. RoadMapper's advantage lies in binding agent roles to specific roadmap error types.
- **vs. Long-form QA Evaluation**: This work draws on LLM-as-a-judge but goes further by introducing structural metrics and key step labeling, demonstrating that complex generation tasks require dual evaluation of content and structure.
- **Insight**: For AI-for-Science systems, "generating an answer" is not always the optimal goal. It is more valuable to generate intermediate structures that are checkable, iterative, and takeover-ready by experts. Roadmaps, experimental plans, and review checklists can all follow this design pattern.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The task definition and benchmark are fresh; multi-agent components are standard but combined effectively for the scientific roadmap scenario.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Main experiments, ablations, efficiency, and expert evaluations are comprehensive, though automated content evaluation still relies on a single strong judge.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure; data construction and agent roles are well-explained; tables strongly support the claims.
- **Value**: ⭐⭐⭐⭐⭐ Directly inspiring for research assistance, educational planning, and complex task decomposition. The RoadMap benchmark itself has significant long-term utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](../../ICML2026/multi_agent/engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[ACL 2026\] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](a_multi-agent_framework_for_feature-constrained_difficulty_control_in_reading_co.md)

</div>

<!-- RELATED:END -->
