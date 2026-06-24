---
title: >-
  [Paper Note] Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking
description: >-
  [ACL 2025][LLM Agent][Web Agent] This paper proposes the WebExperT framework, which simulates the human cognitive pattern of "fast and slow thinking" and continuously improves decision-making through an experiential learning mechanism that reflects on failures. It achieves outstanding performance under both supervised and unsupervised settings on the Mind2Web benchmark.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Web Agent"
  - "Fast-and-Slow Thinking"
  - "Experiential Learning"
  - "Multimodal"
  - "Web Navigation"
date: 2026-05-08
content_hash: 8dc94096758d45c2
---

# Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking

**Conference**: ACL 2025  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-long.697/)  
**Code**: None  
**Area**: LLM Agent / Web Navigation / Multimodal Reasoning  
**Keywords**: Web Agent, Fast-and-Slow Thinking, Experiential Learning, Multimodal, Web Navigation

## TL;DR

This paper proposes the WebExperT framework, which simulates the human cognitive pattern of "fast and slow thinking" and continuously improves decision-making through an experiential learning mechanism that reflects on failures. It achieves outstanding performance under both supervised and unsupervised settings on the Mind2Web benchmark.

## Background & Motivation

**Background**: Automated web navigation tasks require an agent to execute complex interactive operations on real websites based on natural language instructions (e.g., "book a flight from Beijing to Shanghai"). Recently, LLM- and multimodal model-based Web Agents have received widespread attention. Existing methods usually possess visual perception, planning, and memory capabilities.

**Limitations of Prior Work**: Despite technical advancements in existing Web Agents, their reasoning processes still deviate from human cognitive patterns. Specifically: (1) they lack systematic task decomposition strategies when facing complex tasks; (2) they fail to learn and adjust effectively after failures, repeatedly making the same mistakes; (3) they treat simple and complex operations uniformly, leading to low efficiency.

**Key Challenge**: Humans adaptively switch thinking modes when browsing webs based on task complexity—quickly executing familiar, simple operations (such as clicking a button) while thinking deeply face-to-face with complex decisions (such as multi-step form filling). Existing agents lack this adaptive capability.

**Goal**: To design a Web Agent framework closer to human cognition, with (1) the capability of adaptively switching between fast and slow thinking, and (2) the capability of continuous learning and experience accumulation from failures.

**Key Insight**: Inspired by Daniel Kahneman's "Thinking, Fast and Slow" theory, task planning is divided into fast thinking (System 1, handling routine operations) and slow thinking (System 2, processing complex decisions), and an experiential learning module is introduced to accumulate execution experience.

**Core Idea**: Decompose and execute web navigation tasks using a dual-system (fast/slow) thinking model, while continuously optimizing planning and decision-making by reflecting on failure experiences.

## Method

### Overall Architecture

The input to WebExperT consists of natural language user instructions and the visual screenshot of the current webpage, and the output is a sequence of interactive actions (clicking, typing, selecting, etc.). The overall framework consists of three core components: the Fast Thinking module, the Slow Thinking module, and the Experiential Learning module.

### Key Designs

1. **Fast Thinking Module (Fast Thinking / System 1)**:

    - **Function**: Fast processing of routine and simple webpage operations
    - **Mechanism**: Maintains an experience pool that stores successfully executed operation patterns. When facing a new webpage state, it first retrieves matching historical experience through similarity search. If a high-confidence match is found, it directly reuses the historical action strategy without deep reasoning, akin to human "muscle memory" for familiar actions.
    - **Design Motivation**: A large number of web operations are repetitive (e.g., "clicking the confirm button", "selecting a date"). Conducting deep reasoning on these operations wastes computational resources. The fast thinking module can significantly improve execution efficiency.

2. **Slow Thinking Module (Slow Thinking / System 2)**:

    - **Function**: Performs deep planning and reasoning for complex multi-step subtasks
    - **Mechanism**: Triggered when the fast thinking module cannot find a matching experience. It utilizes a multimodal LLM (such as GPT-4V) to perform deep analysis on the current web screenshot and user instructions, decomposing the complex task into a sequence of subgoals and generating detailed execution plans for each subgoal. The planning process considers both structured webpage information (DOM elements) and visual layouts.
    - **Design Motivation**: Complex interactions (e.g., booking processes spanning multiple pages) require global planning capabilities, as single-step greedy strategies easily fall into erroneous paths.

3. **Experiential Learning Module (Experiential Learning)**:

    - **Function**: Learns from execution results, with a particular focus on reflecting on failures
    - **Mechanism**: After each task execution, regardless of success or failure, the complete trajectory (state, action, outcome) is stored in the experience pool. For failed trajectories, an LLM is used to analyze the root causes of the failure and generate "lessons learned" tags. When encountering similar scenarios next time, the experience pool not only provides successful examples but also issues failure warnings, helping the Agent avoid repeating mistakes.
    - **Design Motivation**: The human ability to learn from failure is key to continuous improvement. Existing agents generally store only successful experiences, ignoring the vast value of failure experiences.

### Loss & Training

Under the supervised setting, WebExperT fine-tunes the multimodal model using annotated data from Mind2Web. Under the unsupervised setting, it generates training data through self-play. The experience pool continuously expands as the number of interactions increases.

## Key Experimental Results

### Main Results

| Test Set | Metric | WebExperT | MindAct | SeeAct | Gain |
|--------|------|-----------|---------|--------|------|
| Mind2Web-Cross-Task | Element Acc | Significantly Leading | Baseline | Baseline | ~5-8% |
| Mind2Web-Cross-Website | Element Acc | Significantly Leading | Baseline | Baseline | ~4-7% |
| Mind2Web-Cross-Domain | Element Acc | Best | Baseline | Baseline | ~3-6% |
| Supervised Overall | Step Success Rate | Best | - | - | Significant |
| Unsupervised Overall | Task Completion | Significantly Improved | - | - | Obvious |

### Ablation Study

| Configuration | Step Acc | Description |
|------|---------|------|
| Full WebExperT | Best | Full model |
| w/o Fast Thinking | Obvious Decline | Loses fast decision-making capability, lowering efficiency |
| w/o Slow Thinking | Significant Drop | Complex tasks cannot be effectively decomposed |
| w/o Experiential Learning | Moderate Drop | Repeatedly makes mistakes, unable to improve continuously |
| w/o Failure Reflection | Slight Drop | Demonstrates that failure experiences are indeed valuable |

### Key Findings
- The slow thinking module contributes the most, indicating that task decomposition and deep planning are core capabilities for web navigation.
- The fast thinking module brings a significant boost in efficiency—for repetitive tasks, reasoning speed is improved by about 2-3 times.
- Learning from failure (Failure Reflection) is more effective in cross-website and cross-domain scenarios, where mistakes are more prone to occur in new environments.
- Under the unsupervised setting, WebExperT's experience accumulation mechanism enables continuous improvement as the number of interactions grows.

## Highlights & Insights
- **Dual-System Thinking Framework** is an elegant paradigm for agent design. Combining Kahneman's cognitive theory with AI Agents provides both theoretical depth and practical effectiveness. This approach can be transferred to other agent tasks requiring adaptive decision-making.
- **Explicit Utilization of Failure Experiences** is another highlight of this work. Most agents only store successful experiences, neglecting the value of failure. Structuring failed trajectories as "lessons learned" and utilizing them for future decision-making is a highly reusable strategy.
- The design of the experience pool simulates "memory" and "growth" in the Agent, making it closer to real human behavior compared to stateless LLM calls.

## Limitations & Future Work
- Mind2Web is a static web benchmark, and dynamic changes of real websites (pop-ups, asynchronous loading, etc.) are not covered.
- Retrieval from the experience pool relies on similarity calculations of webpage states, which may be less effective in scenarios with major visual UI changes.
- The switching strategy between fast and slow thinking is relatively rule-based (based on retrieval hit rates); adaptive switching strategies could be learned in the future.
- Validation was not performed in real browser environments (such as WebArena).

## Related Work & Insights
- **vs MindAct**: MindAct uses a single reasoning pipeline to handle all operations, whereas WebExperT's dual-system design is more flexible in handling heterogeneous operations.
- **vs SeeAct**: SeeAct focuses on visual grounding, while WebExperT adds planning and learning dimensions on top of it.
- **vs Reflexion**: Reflexion also utilizes reflection to improve agents but focuses on general reasoning tasks; WebExperT integrates the reflection mechanism with the fast-and-slow thinking framework for web navigation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of fast-and-slow thinking and experiential learning is relatively novel in the Web Agent domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation on Mind2Web, with clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, and the framework diagram is intuitive.
- Value: ⭐⭐⭐⭐ Provides new insights into cognitive heuristic design for Web Agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [\[CVPR 2026\] iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception](../../CVPR2026/llm_agent/ishift_lightweight_slow-fast_gui_agent_with_adaptive_perception.md)
- [\[ICLR 2026\] FaSTA*: Fast-Slow Toolpath Agent with Subroutine Mining for Efficient Multi-turn Image Editing](../../ICLR2026/llm_agent/fasta_fast-slow_toolpath_agent_with_subroutine_mining_for_efficient_multi-turn_i.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Multimodal Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_multimodal_knowledge-induced_cognitive_reasoning_for_web.md)
- [\[ACL 2025\] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)

</div>

<!-- RELATED:END -->
