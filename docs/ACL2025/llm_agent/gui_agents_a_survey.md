---
title: >-
  [Paper Note] GUI Agents: A Survey
description: >-
  [ACL 2025 (Findings)][LLM Agent][GUI Agent] This paper presents a comprehensive survey of Graphical User Interface (GUI) agents based on large foundation models. It proposes a unified analytical framework covering four core capabilities—perception, reasoning, planning, and action—systematically reviews GUI Agent benchmarks, evaluation metrics, architectural designs, and training methods, and discusses key challenges and future directions.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "GUI Agent"
  - "automated interaction"
  - "benchmark evaluation"
  - "multi-platform operation"
date: 2026-05-08
content_hash: d6c50d1050ecd886
---

# GUI Agents: A Survey

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2412.13501](https://arxiv.org/abs/2412.13501)  
**Code**: None  
**Area**: Agent / Human-Computer Interaction  
**Keywords**: GUI Agent, automated interaction, LLM Agent, benchmark evaluation, multi-platform operation

## TL;DR

This paper presents a comprehensive survey of Graphical User Interface (GUI) agents based on large foundation models. It proposes a unified analytical framework covering four core capabilities—perception, reasoning, planning, and action—systematically reviews GUI Agent benchmarks, evaluation metrics, architectural designs, and training methods, and discusses key challenges and future directions.

## Background & Motivation

**Background**: GUI Agents refer to intelligent agents capable of autonomously interacting with digital systems' graphical user interfaces. They complete user-specified tasks by observing screenshots, understanding interface elements, and performing actions such as clicking, typing, and scrolling. With the emergence of multimodal large models like GPT-4V and Gemini, GUI Agents have evolved from early rule-based and manual feature-based solutions to LLM-based end-to-end systems, representing a qualitative leap in capability.

**Limitations of Prior Work**: The GUI Agent field is rapidly developing yet highly fragmented. Different platforms (Web, mobile, desktop) have distinct interaction paradigms, different works employ varied benchmarks and evaluation metrics, and new methods keep emerging with no unified comparison framework. It is difficult for researchers to quickly grasp the full picture or determine which techniques are genuinely effective.

**Key Challenge**: GUI Agents need to simultaneously possess four capabilities: visual perception (identifying interface elements), semantic understanding (comprehending task intentions and interface meanings), planning and reasoning (decomposing complex tasks into action sequences), and precise execution (performing the correct action at the correct location). However, prior works typically focus on only one or two aspects, lacking a comprehensive capability analysis perspective.

**Goal**: To provide a comprehensive, structured survey of the GUI Agent domain and establish a unified analytical framework to help researchers and practitioners quickly understand current progress, core technologies, and open issues.

**Key Insight**: Starting from the four core capabilities of GUI Agents (perception $\rightarrow$ reasoning $\rightarrow$ planning $\rightarrow$ action), the authors organize existing works within this unified framework, while horizontally covering benchmarks, training methods, and cross-platform differences.

**Core Idea**: A four-dimensional capability framework (Perception, Reasoning, Planning, Acting) is proposed to analyze various GUI Agent approaches in a unified style, systematically organizing benchmarks, architectures, and training strategies under this framework.

## Method

### Overall Architecture

The analytical framework of this paper categorizes GUI Agent capabilities into four levels, forming a complete pipeline from input to output:

1. **Perception**: Extract interface information from screenshots and accessibility trees.
2. **Reasoning**: Understand the current interface state, user intent, and task context.
3. **Planning**: Decompose complex tasks into executable action sequences.
4. **Acting**: Translate planned actions into concrete GUI interaction steps.

### Key Designs

1. **Perception Capability Classification**:

    - **Function**: Summarizes how GUI Agents "understand" the interface.
    - **Mechanism**: Perception is the foundation of GUI Agents, determining how much interface information they can acquire. This survey categorizes perception methods into three categories:
        - **Screenshot Understanding**: Directly uses screenshots as visual inputs, relying on Vision-Language Models (VLMs) to understand interface layouts and content. The advantage is that it does not depend on platform APIs, but it faces limitations in resolution and struggles to recognize small elements.
        - **DOM/Accessibility Tree Parsing**: Leverages structured information provided by platforms (such as HTML DOM, Android View Hierarchy) to obtain element locations and attributes. This approach is highly accurate but relies on platform support, showing poor generalization across different application environments.
        - **Hybrid Methods**: Utilizes both visual screenshots and structured information, such as the Set-of-Mark (labeling element indices on screenshots) approach. This balances accuracy with visual understanding.
    - Key Challenges: Recognition of small icons/dense interfaces, multi-resolution adaptation, and tracking dynamic interface changes.

2. **Planning Capability Analysis**:

    - **Function**: Analyzes how GUI Agents decompose complex tasks into executable steps.
    - **Mechanism**: This survey categorizes planning strategies into three types:
        - **Reactive**: Directly decides the next action based on the current screenshot at each step, without long-term planning. It is simple and efficient but prone to local optima.
        - **Pre-planning**: Generates a complete action plan prior to execution, which is then executed sequentially. This achieves global optimization, but plans are easily invalidated by interface changes.
        - **Adaptive**: Generates an initial plan and dynamically adjusts it during execution based on actual interface states. This combines the advantages of both and is currently the mainstream direction.
    - Related Technologies: Reasoning frameworks such as ReAct, Reflexion, and Tree-of-Thought are widely used in the planning modules of GUI Agents.

3. **Benchmarks and Evaluation**:

    - **Function**: Systematically structures the evaluation ecosystem of GUI Agents.
    - **Mechanism**: The survey categorizes major benchmarks by platform:
        - **Web**: MiniWoB++ (simple web operations), WebArena (real-world website interactions), Mind2Web (cross-website generalization).
        - **Mobile**: AndroidWorld, Mobile-Env, AITW (Android In The Wild).
        - **Desktop**: OSWorld (cross-operating systems), WindowsAgentArena.
        - **Cross-platform**: OmniACT (unified multi-platform evaluation).
    - Evaluation Metrics: Task success rate (core), step accuracy, efficiency (step count), and generalization (performance on unseen tasks/applications).
    - Key Findings: Significant discrepancies exist among current benchmarks. Even within the same benchmark, the experimental configurations of different papers are not unified, leading to poor comparability. The authors call for more standardized evaluation protocols.

### Loss & Training

This survey classifies training methods for GUI Agents into four categories:

- **Prompt Engineering**: Does not train the model; instead, it guides general LLMs/VLMs to perform GUI operations using carefully designed prompts and in-context examples. Low entry barrier, but performance upper bound is limited.
- **Supervised Fine-Tuning (SFT)**: Fine-tunes models with human operation trajectories. It requires high-quality labeled data, delivering strong performance but with expensive data collection costs.
- **Reinforcement Learning (RL)**: Trains the model using reward signals obtained from trial-and-error in the environment. It does not require annotations but suffers from unstable training and demands efficient simulator environments.
- **Hybrid Methods**: Primarily establishes foundational capabilities via SFT, followed by online optimization with RL. This achieves the best overall performance and represents an active research frontier.

## Key Experimental Results

### Main Results

#### Performance Comparison of Major Benchmarks Across Platforms

| Benchmark | Platform | Best Method | Success Rate | Description |
|------|------|---------|--------|------|
| MiniWoB++ | Web (Simple) | GPT-4V + SoM | ~95% | Simple tasks are almost saturated |
| WebArena | Web (Real) | Best Agent | ~35% | Real websites remain extremely challenging |
| Mind2Web | Web (Generalization) | Best Method | ~60% (elem acc) | Poor generalization across websites |
| AITW | Mobile | Best Method | ~70% | Mobile platform is relatively mature |
| OSWorld | Desktop | GPT-4V Agent | ~12% | Desktop platform is the hardest |

#### Comparison of Different Method Paradigms

| Method Paradigm | Representative Work | Advantages | Disadvantages |
|---------|---------|------|------|
| Prompt-only | GPT-4V + ReAct | Zero-shot generalization | Performance limits |
| SFT | CogAgent, UGround | Strong on specific tasks | Poor generalization, expensive data |
| RL-based | DigiRL, AppAgent | Continuous improvement | Unstable training |
| Hybrid (SFT+RL) | WebAgent, AgentQ | Best overall performance | High complexity |

### Key Findings

- **Huge gap between real and simplified environments**: The task success rate drops from nearly 95% on MiniWoB++ to ~35% on WebArena, and further plummets to ~12% on OSWorld. This demonstrates that GUI Agents are still far from ready for deployment in complex real-world environments.
- **Visual perception is a major bottleneck**: Many failure cases stem from the model's inability to correctly identify small buttons, dropdown menus, or dynamically loaded UI elements. Auxiliary labeling strategies like Set-of-Mark can significantly improve perception accuracy.
- **Insufficient long-horizon planning capabilities**: The success rate for complex tasks requiring 10+ steps is substantially lower than that for simple 3-5 step tasks. Error propagation is a core issue—minor errors at each step accumulate over time.
- **Cross-platform generalization remains an open problem**: Most methods are optimized for specific platforms, showing drastic performance drops on others. Unified GUI representation and action abstraction represent crucial future directions.
- **Data is a limiting factor**: High-quality human operation trajectories are difficult to collect, while synthetic data lacks sufficient quality and diversity. Automated data collection and trajectory filtering are vital engineering problems.

## Highlights & Insights

- **The four-dimensional capability framework (Perception-Reasoning-Planning-Acting)** provides a clear analytical perspective, helping to localize the contributions of different works and identify the overall bottlenecks of GUI Agents. This framework is also applicable to other types of autonomous agents (e.g., Embodied Agents, API Agents).
- **The systematic structuring of the benchmark evaluation ecosystem** is highly valuable—compiling major benchmarks, evaluation metrics, and current results per platform, making it an excellent reference for entering this domain.
- **The pedagogical classification of training paradigms** (Prompt $\rightarrow$ SFT $\rightarrow$ RL $\rightarrow$ Hybrid) clearly outlines the evolutionary trajectory and future directions of GUI Agent training methodologies.
- **The paper has been cited 18 times (as of April 2026)**, proving its modest but steady impact within the academic community.

## Limitations & Future Work

- **Relatively shallow technical depth**: As a survey, while it covers a broad scope, the depth remains limited; the concrete implementation details and failure modes of specific methods are not thoroughly dissected.
- **Lack of rigorous quantitative comparison tables**: Although results of various benchmarks are compiled, they are not compared under strictly controlled conditions, making it hard for readers to gauge genuine performance gaps between methods.
- **Insufficient discussion on multimodal perception**: The challenges of visual perception (such as OCR, layout understanding, and visual grounding) are under-explored, despite being one of the major current bottlenecks.
- **Under-addressed safety and reliability issues**: Safety and privacy risks arising from GUI Agents automatically operating user devices require far more attention.
- **Future Key Directions**:
    - Unified cross-platform GUI representation.
    - Self-correction and safety boundary mechanisms.
    - Efficient data collection and annotation pipelines.
    - Hybrid automation paradigms collaborating with humans.
    - Continuous online learning driven by RL.

## Related Work & Insights

- **vs Specific methods like WebAgent/AutoGPT**: This paper does not propose a new method but systematically summarizes existing literature, offering a birds-eye perspective highly suitable for domain introduction and research planning.
- **vs Other Agent Surveys (e.g., "A Survey on Large Language Model based Autonomous Agents")**: Those surveys focus on general LLM Agents, whereas this work specializes in the vertical field of GUI interactions, offering deeper coverage of GUI-specific visual perception, interface modeling, and adjacent issues.
- **vs Software Test Automation**: Traditional UI automation testing (such as Selenium) relies on deterministic scripts, whereas GUI Agents operate based on AI perception and reasoning, handling dynamic interface changes at the expense of reliability. The two can be complementary.

## Rating

- **Novelty**: ⭐⭐⭐ As a survey, there is no technical novelty, but the proposal of the four-dimensional capability framework carries structural value.
- **Experimental Thoroughness**: ⭐⭐⭐ Containing no original experiments, but the compilation of existing benchmarks and baseline results is comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear organization and rational taxonomy; 30 co-authors guarantee a broad, balanced coverage of the field.
- **Value**: ⭐⭐⭐⭐ Highly valuable as a reference for entering the GUI Agent field and choosing research directions, featuring high systematism.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)
- [\[CVPR 2025\] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](../../CVPR2025/llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)
- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)

</div>

<!-- RELATED:END -->
