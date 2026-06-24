---
title: >-
  [Paper Note] ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development
description: >-
  [ACL 2025][LLM (Other)][ComfyUI] This paper proposes ComfyUI-Copilot, an LLM-based hierarchical multi-agent framework. Served as a ComfyUI plugin, it provides intelligent node/model recommendations and one-click workflow generation. Powered by a knowledge base covering 7K nodes, 62K models, and 9K workflows, it has served over 19K users across 22 countries and processed more than 85K queries online.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "ComfyUI"
  - "Multi-Agent Framework"
  - "Workflow Generation"
  - "LLM Agent"
  - "AIGC"
date: 2026-05-08
content_hash: 7779c138fca2a832
---

# ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development

**Conference**: ACL 2025  
**arXiv**: [2506.05010](https://arxiv.org/abs/2506.05010)  
**Code**: [https://github.com/AIDC-AI/ComfyUI-Copilot](https://github.com/AIDC-AI/ComfyUI-Copilot)  
**Area**: Others  
**Keywords**: ComfyUI, Multi-Agent Framework, Workflow Generation, LLM Agent, AIGC

## TL;DR

This paper proposes ComfyUI-Copilot, an LLM-based hierarchical multi-agent framework. Served as a ComfyUI plugin, it provides intelligent node/model recommendations and one-click workflow generation. Powered by a knowledge base covering 7K nodes, 62K models, and 9K workflows, it has served over 19K users across 22 countries and processed more than 85K queries online.

## Background & Motivation

ComfyUI is the most popular open-source, low-code AI workflow platform in the AIGC field, boasting over 4 million active users and more than 12K community-contributed components. Users build workflows for multimodal tasks (such as text-to-image, face swapping, and video editing) by dragging and dropping components. However, ComfyUI faces several barriers to usability:

**High barrier to entry for beginners**: Documentation is scattered across forums and GitHub issues, lacking a unified tutorial system.

**Complex node/model configuration**: Selecting appropriate nodes and models requires significant domain expertise, and compatibility dependencies exist between different models (e.g., a specific LoRA only works with a specific diffusion model).

**High cost of workflow design**: Even for experienced users, debugging and constructing a complete workflow requires considerable time.

**Inadequacy of existing automation solutions**: Prior research on automated workflow construction suffers from instability (generating unparseable workflows) and limited scope (only supporting text-to-image tasks).

The goal of ComfyUI-Copilot is to lower the entry barrier for beginners while enhancing the efficiency of expert users.

## Method

### Overall Architecture

The framework adopts a hierarchical multi-agent architecture: a **central assistant agent** (LLM-based) is responsible for task dispatching and response integration, while **specialized worker agents** (categorized into workflow, node, and model) handle specific task execution. The system is supported by three underlying knowledge bases. Implemented using LangChain, the assistant agent autonomously selects the appropriate worker agent based on user instructions and short-term memory (dialogue history).

### Key Designs

1. **Knowledge Base Construction (7K Nodes + 62K Models + 9K Workflows)**: Data is sourced from popular resource platforms, GitHub repositories, and the official ComfyUI website, with NSFW content filtered out. For nodes lacking documentation, an automated document generation pipeline is designed: set up a ComfyUI sandbox environment $\rightarrow$ clone GitHub repositories and install dependencies $\rightarrow$ import nodes to extract metadata $\rightarrow$ code chunking + BGE-M3 embedding retrieval $\rightarrow$ LLM generates documentation by combining metadata and code $\rightarrow$ quality audit. For workflows and models in the community lacking functional descriptions, the multimodal understanding capability of GPT-4o is leveraged to supplement instructions using text, sample images, and JSON files. **The knowledge base is continuously updated weekly** to ensure coverage of the latest modules. Key incentive: The AIGC field evolves extremely rapidly, and a static knowledge base quickly becomes obsolete.

2. **Three-Stage Recommendation Pipeline (Coarse-to-Fine)**: 

    - **Stage 1 - Intent Expansion**: Uses an LLM/LMM to expand vague user instructions into detailed task descriptions. For instance, when a portrait is identified in the original image, the expanded intent will emphasize maintaining subject consistency.
    - **Stage 2 - Hybrid Retrieval**: Uses OpenAI's `text-embedding-3-small` to compute semantic similarity $\text{sim}_S$ and token-level overlap ratio $\text{sim}_L$. The combined score is computed as $\text{sim}_O = 0.7 \times \text{sim}_S + 0.3 \times \text{sim}_L$, yielding the top 30 candidates.
    - **Stage 3 - Re-ranking**: Uses the GTE-Rerank model to select the top 3 candidates from the 30 choices, which are then sorted by popularity metrics (likes, downloads, stars). This coarse-to-fine design strikes a good balance between efficiency and accuracy.

3. **Workflow Generation (Retrieval + Generation from Scratch)**: In addition to retrieving existing workflows through the pipeline, generating workflows from scratch using code LLMs is also explored. Workflows support mutual conversion among three formats: ComfyUI flowcharts $\leftrightarrow$ JSON $\leftrightarrow$ code (Pythonic style), with code as the primary representation (as it is rich in logic and semantic information, and naturally compatible with LLM code-generation capabilities). Close-source LLMs are prompted using retrieved nodes and code examples to generate workflows. It is also discovered that a fine-tuned Qwen2.5-Coder-7B can achieve performance close to Claude-3.7-Sonnet in terms of pass rates and node selection.

4. **Additional Features**: 

    - **Prompt Optimization**: Assists users in refining simple descriptions (e.g., "a cat") into vivid, detailed prompts.
    - **Parameter Search**: Supports parallel experimentation with different parameter combinations (e.g., cfg/denoise) to generate images in batches for comparison.
    - **Multilingual Support**: Supports multilingual queries and responses, including languages like Polish.

### User Interface

Serving as a ComfyUI sidebar plugin, it activates a chat interface with one click, supporting multi-turn dialogue and backend LLM switching (DeepSeek-V3/GPT-4o). Users can click on any node in the canvas to ask questions and enjoy one-click loading of recommended workflows and nodes onto the canvas.

## Key Experimental Results

### Main Results

| Task | LLM Backend | Recall@3 |
|------|-------------|----------|
| Node Recommendation (104 instructions) | DeepSeek-V3 | 88.5% |
| Node Recommendation (104 instructions) | GPT-4o | 89.4% |
| Workflow Recall (130 instructions) | DeepSeek-V3 | 90.0% |
| Workflow Recall (130 instructions) | GPT-4o | 89.2% |

Both LLM backends achieve a Top-3 recall rate of over 88.5%, demonstrating the robustness of the framework.

### Online User Feedback

| Metric | Value |
|------|------|
| Node recommendation acceptance rate | 65.4% |
| Workflow recommendation acceptance rate | 85.9% |
| Total query volume | 85K+ |
| Number of users | 19K |
| Countries covered | 22 |
| GitHub Stars | 1.6K+ |

The acceptance rate for workflow recommendations (85.9%) is significantly higher than that for node recommendations (65.4%), indicating that workflow recommendation carries higher practical utility.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Fine-tuned Qwen2.5-Coder-7B vs Claude-3.7 | Close pass rate + node selection | Open-source models can replace closed-source models |
| Error Analysis | Unrecalled workflows still satisfy user intent | Indicates that the actual performance is better than the figures suggest |

### Key Findings

1. The acceptance rate for workflow recommendations (85.9%) is significantly higher than that for node recommendations (65.4%), reflecting a greater user demand for 'end-to-end solutions'.
2. Even when the target workflow is not accurately recalled, the recommended alternative workflows usually still satisfy user requirements.
3. The fine-tuned 7B open-source code model achieves workflow generation performance close to that of commercial closed-source LLMs, reducing deployment costs.

## Highlights & Insights

- **Highly engineered systems paper**: Covers the complete product lifecycle from knowledge base construction, automated documentation generation, and multi-agent architecture to front-end plugin integration, validated with real-world user data.
- **Automated node documentation generation**: A reusable technical solution that automatically supplements missing documentation through a pipeline of sandbox environments, code analysis, and LLM comprehension.
- **Continuous knowledge base update mechanism**: The weekly update design ensures coverage of cutting-edge modules, addressing the core challenge of rapid iteration in the AIGC domain.
- **Code as workflow representation**: Using code-generation capabilities of LLMs to generate graphical workflows is an ingenious bridging strategy.

## Limitations & Future Work

- The pass rate of generating workflows from scratch (mentioned in the paper as 'having significant room for improvement') is currently the biggest bottleneck; correct generation of complex workflows remains challenging.
- The online acceptance rate for node recommendations (65.4%) still has substantial room for improvement, likely requiring better context understanding.
- Evaluation primarily relies on recall rates and user acceptance rates, lacking systematic evaluation of the quality of generated workflows (executability, image quality, etc.).
- Despite the large coverage of the knowledge base (7K nodes / 62K models), the ComfyUI ecosystem iterates exceptionally fast, and coverage for long-tail nodes may still be insufficient.
- LLM invocation costs and latency are not discussed, leaving scalability in high-frequency usage scenarios questionable.

## Related Work & Insights

- Prior works like ComfyGen only support text-to-image tasks, whereas ComfyUI-Copilot expands coverage to the full spectrum of conditional multimodal generation tasks.
- The design philosophy of the multi-agent architecture (central planner + specialized workers) is a general paradigm in the LLM Agent field, which this paper successfully applies to the AIGC workflow scenario.
- Insight: For component-based and visual tools (not limited to ComfyUI), the LLM agent-assisted pipeline of 'intent understanding $\rightarrow$ component retrieval $\rightarrow$ automated assembly' serves as a general solution.

## Rating

- **Novelty**: ⭐⭐⭐ — The technical framework is relatively standard (retrieval + LLM + multi-agent), but its application to the ComfyUI scenario is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐ — Features both offline evaluation and online feedback, but evaluation metrics are relatively single-dimensional, lacking in-depth evaluation such as workflow executability.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear systematic description, rich functional demonstrations, and strong practicality.
- **Value**: ⭐⭐⭐⭐⭐ — The real-world deployment data of 1.6K stars, 19K users, and 85K queries demonstrates extremely high practical value, contributing to the AIGC community far beyond the paper itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)
- [\[ACL 2025\] Achieving Certification-by-Design Through Model-Driven Development](achieving_certification-by-design_through_model-driven_development.md)
- [\[ACL 2025\] ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)

</div>

<!-- RELATED:END -->
