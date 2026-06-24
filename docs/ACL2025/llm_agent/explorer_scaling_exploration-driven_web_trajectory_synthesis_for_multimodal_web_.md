---
title: >-
  [Paper Note] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents
description: >-
  [ACL 2025][LLM Agent][Web Agent] Proposes Explorer, a scalable multi-agent pipeline that synthesizes large-scale multimodal web trajectory datasets (94K successful trajectories, 49K+ URLs, 720K screenshots) through autonomous web exploration and step-by-step refinement. The trained Explorer-7B matches or exceeds GPT-4 performance on benchmarks like Mind2Web-Live and MiniWob++.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Web Agent"
  - "Trajectory Synthesis"
  - "Multimodal"
  - "Data Scaling"
  - "Multi-Agent"
date: 2026-05-08
content_hash: 1af87ce756b7a189
---

# Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents

**Conference**: ACL 2025  
**arXiv**: [2502.11357](https://arxiv.org/abs/2502.11357)  
**Code**: [https://osu-nlp-group.github.io/Explorer/](https://osu-nlp-group.github.io/Explorer/)  
**Area**: LLM Agent  
**Keywords**: Web Agent, Trajectory Synthesis, Multimodal, Data Scaling, Multi-Agent

## TL;DR
Proposes Explorer, a scalable multi-agent pipeline that synthesizes large-scale multimodal web trajectory datasets (94K successful trajectories, 49K+ URLs, 720K screenshots) through autonomous web exploration and step-by-step refinement. The trained Explorer-7B matches or exceeds GPT-4 performance on benchmarks like Mind2Web-Live and MiniWob++.

## Background & Motivation

**Background**: Training multimodal web agents requires large-scale, high-quality trajectory-level data (task $\rightarrow$ action sequence $\rightarrow$ screenshot). However, human annotation is extremely costly, and existing synthetic methods suffer from either limited scale or insufficient diversity.

**Limitations of Prior Work**: (a) Human-annotated trajectory data (e.g., Mind2Web) is small in scale and covers limited websites; (b) existing synthesis methods (e.g., AgentTrek) rely on tutorial-based knowledge guidance and are restricted to predefined task templates; (c) most methods only support textual modalities, neglecting the critical role of visual screenshots in GUI interaction.

**Key Challenge**: The web environment is extremely diverse (49K+ URLs). Implementing automatic synthesis of high-quality, diverse task-trajectory pairs without relying on human templates is highly challenging.

**Goal** To design an automated pipeline capable of large-scale multimodal web trajectory synthesis without human-designed task templates.

**Key Insight**: A bottom-up approach: allowing agents to freely explore and execute actions on web pages first, then extracting high-level task descriptions from the execution history, and using a verifier to filter out inconsistent trajectories.

**Core Idea**: Utilizing four LLM agents (Proposer, Refiner, Summarizer, Verifier) to autonomously explore web pages and synthesize high-quality task trajectory data bottom-up from low-level actions.

## Method

### Overall Architecture
The input consists of a large set of website URLs (from Similarweb Top 100 + Tranco list), and the output is a triplet of <task description, action sequence, screenshot sequence>. Four GPT-4o-driven agents collaborate: the Task Proposer proposes abstract tasks on the homepage and executes the first action $\rightarrow$ the Task Refiner incrementally refines the task description and predicts the subsequent action $\rightarrow$ the Task Summarizer generates high-level task descriptions from the complete trajectory $\rightarrow$ the Task Verifier validates whether the trajectory successfully completed the task.

### Key Designs

1. **Task Proposer Agent**:

    - **Function**: Given a website homepage (screenshot + accessibility tree), it proposes diverse high-level abstract tasks and executes the initial action.
    - **Mechanism**: Instead of using fixed task templates, the LLM is allowed to autonomously propose reasonable tasks based on web content. It automatically terminates upon encountering CAPTCHA, login, or payment pages.
    - **Design Motivation**: Bottom-up exploration covers diverse interactions of real websites much better than top-down, template-driven methods.

2. **Task Refiner Agent**:

    - **Function**: After each step, it refines the task description based on the current state and predicts the next action.
    - **Mechanism**: It takes the initial/previous refined task description + action history as input, outputting an updated task description and the next action. This ensures the refined task remains consistent with the executed action history.
    - **Design Motivation**: Constructing task descriptions incrementally during exploration prevents premature over-specification from restricting the exploration space.

3. **Task Summarizer**:

    - **Function**: Processes the complete action and screenshot history to output a pure high-level task description (describing "what to do" rather than "how to do it").
    - **Design Motivation**: While the Refiner's output tasks may contain procedural details, the Summarizer provides clean task definitions.

4. **Task Verifier**:

    - **Function**: Evaluates whether the trajectory has successfully completed the task.
    - **Mechanism**: Integrates the task description, action history, screenshots, and the markdown of the final page to determine success. Inconsistent trajectories are discarded.
    - **Design Motivation**: Autonomous exploration inevitably yields incoherent trajectories, making the verifier crucial for quality control. It achieves an $81\%$ agreement rate with human judgment.

### Dataset Scale
- 175K total trajectories, 94K successful trajectories (passed verification)
- 49,494 unique URLs, 720K screenshots, 33M web elements
- Average of 7.7 steps per trajectory, 46.3 elements per screenshot
- Cost of only **$0.28 per successful trajectory**, completed in 50 hours with 60 parallel processes

### Loss & Training
- Fine-tuned Phi-3.5V (4B) and Qwen2-VL-7B on the synthetic dataset.
- Input format includes screenshots + accessibility tree + task descriptions.

## Key Experimental Results

### Main Results

**Mind2Web-Live (83 tasks)**

| Model | Avg Step SR | Full Task SR |
|------|------------|-------------|
| GPT-4o | 58.5% | 25.3% |
| Qwen2-VL-7B (base) | 40.2% | 14.5% |
| **Explorer-7B** | **45.3%** | **19.3%** |
| Phi-3.5V (base) | 28.5% | 2.4% |
| **Explorer-4B** | **44.0%** | **18.1%** |

**MiniWob++ (46 tasks, zero-shot)**

| Model | Accuracy |
|------|---------|
| GPT-4 | 53.04% |
| **Explorer-7B** | **53.26%** |
| AgentTrek-7B | 45.28% |
| **Explorer-4B** | **46.74%** |

Explorer-7B matches or even exceeds GPT-4 on MiniWob++ ($53.26\%$ vs $53.04\%$).

### Ablation Study

| Configuration | Full Task SR (Mind2Web-Live) |
|------|---------------------------|
| Explorer-4B (multimodal) | 18.1% |
| Phi-3-mini (text-only) | 13.3% |
| LLaVA-Mistral-7B | 4.8% |
| 25% data → 100% data | Consistent improvements |

### Key Findings
- **Data scale is a key driver**: From 25% to 100% data volume, all metrics consistently improve.
- **Visual modality is critical**: Removing screenshots degrades the Full Task SR from 18.1% to 13.3% (-4.8pp).
- **Comparison with AgentTrek**: Using the same 7B backbone, Explorer (54.3%) > AgentTrek (53.2%) on Multimodal-Mind2Web, demonstrating that autonomous exploration synthesis is more effective than tutorial-driven synthesis.
- **Small models can perform strongly**: Explorer-4B already outperforms AgentTrek-7B and multiple other 7B models.

## Highlights & Insights
- **Bottom-up trajectory synthesis paradigm**: Proposes tasks emergent from exploration without predefining task templates. This "do first, summarize later" data generation strategy can be transferred to other domains requiring large-scale interaction data (e.g., app testing, robotic manipulation).
- **Cost efficiency of $0.28 per trajectory**: Significantly lower than human annotation (typically $5-$20 per trajectory), making web-scale data synthesis viable.
- **Clear role division among 4 agents**: The pipeline design of Proposer $\rightarrow$ Refiner $\rightarrow$ Summarizer $\rightarrow$ Verifier allows each component to be independently optimized and replaced.

## Limitations & Future Work
- **Verifier accuracy is only 81%**: Approximately 19% of judgments are inconsistent with humans, introducing noisy data.
- **Reliance on GPT-4o for synthesis**: The cost of closed-source APIs remain non-negligible (total cost of 175K trajectories estimated at ~$26K+).
- **Primary failure mode is grounding error**: The agent's grounding action during the refinement stage fails to match the natural language description.
- **Authentication limits**: A significant number of websites requiring authentication cannot be explored, leaving gaps in data coverage.
- Potential improvements: (a) Training open-source LLMs to replace GPT-4o for synthesis to reduce costs; (b) introducing self-correction mechanisms to handle grounding errors.

## Related Work & Insights
- **vs AgentTrek**: AgentTrek relies on tutorial knowledge to guide task generation, whereas Explorer adopts autonomous exploration, resulting in greater data diversity.
- **vs Mind2Web**: Mind2Web is a human-annotated static dataset, while Explorer offers automated synthesis with infinite scalability.
- **vs SeeAct**: SeeAct uses GPT-4 for inference without further training, while Explorer synthesizes data to fine-tune open-source models.

## Rating
- Novelty: ⭐⭐⭐⭐ Bottom-up exploration-based trajectory synthesis is a novel data generation paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering three benchmarks, in-domain evaluation, ablation studies, and scaling analyses.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is clearly described, and the data statistics are highly detailed.
- Value: ⭐⭐⭐⭐⭐ Provides a scalable solution to the training data bottleneck in web agents, releasing both the dataset and models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking](browsing_like_human_a_multimodal_web_agent_with_experiential_fast-and-slow_think.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Multimodal Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_multimodal_knowledge-induced_cognitive_reasoning_for_web.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[ICLR 2026\] Go-Browse: Training Web Agents with Structured Exploration](../../ICLR2026/llm_agent/go-browse_training_web_agents_with_structured_exploration.md)

</div>

<!-- RELATED:END -->
