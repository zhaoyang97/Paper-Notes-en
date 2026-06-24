---
title: >-
  [Paper Note] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use
description: >-
  [ACL 2025][LLM Agent][OS Agent] A comprehensive survey of Operating System Agents (OS Agents) based on Multimodal Large Language Models (MLLMs), systematically analyzing their fundamental concepts (environment/observation/action space), core capabilities (understanding/planning/grounding), construction methodology (foundation models + agent frameworks), and evaluation systems. It covers a categorical comparison of 30+ foundation models and 20+ Agent frameworks.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "OS Agent"
  - "GUI automation"
  - "multimodal LLM"
  - "agent framework"
  - "reinforcement learning"
date: 2026-05-08
content_hash: a5dc6c9b683d9514
---

# OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use

**Conference**: ACL 2025  
**arXiv**: [2508.04482](https://arxiv.org/abs/2508.04482)  
**Code**: [https://github.com/OS-Agent-Survey](https://github.com/OS-Agent-Survey) (maintained by open-source community)  
**Area**: LLM Agent  
**Keywords**: OS Agent, GUI automation, multimodal LLM, agent framework, reinforcement learning

## TL;DR
A comprehensive survey of Operating System Agents (OS Agents) based on Multimodal Large Language Models (MLLMs), systematically analyzing their fundamental concepts (environment/observation/action space), core capabilities (understanding/planning/grounding), construction methodology (foundation models + agent frameworks), and evaluation systems. It covers a categorical comparison of 30+ foundation models and 20+ Agent frameworks.

## Background & Motivation

**Background**: From Siri, Cortana to Alexa, virtual assistants have demonstrated the potential of AI to automatically operate computing devices. With breakthroughs in multimodal LLMs such as GPT-4o, Gemini, and Claude, products/research like Anthropic Computer Use, Apple Intelligence, and AutoGLM have emerged, driving the OS Agent field into a period of rapid growth.

**Limitations of Prior Work**: Research on OS Agents is scattered across various subfields such as Web navigation, mobile automation, and desktop operations, lacking a unified conceptual framework and taxonomy. Researchers face fragmented benchmarks, inconsistent evaluation protocols, and the dilemma of lacking a global perspective.

**Key Challenge**: An OS Agent needs to simultaneously possess GUI understanding (pixel-level vision + textual semantics), long-chain planning (multi-step task decomposition), and precise execution/grounding (coordinate-level clicking/input). The model capabilities required for these three aspects are vastly different. How to coordinate them within a unified framework is the core challenge. Moreover, engineering issues such as safety, privacy, and personalized adaptation require systematic discussion.

**Goal**: (1) To establish a unified conceptual system and taxonomy for OS Agents; (2) to systematically organize two technical routes: foundation model construction (architecture + pre-training + SFT + RL) and Agent framework design; (3) to summarize evaluation benchmarks and future directions (security/personalization/self-evolution).

**Key Insight**: Taking the "operating system" as a unified abstraction—whether PC, smartphone, or Web, interaction occurs through the environment and input/output interfaces provided by the OS, bringing disparate subfields into the same framework.

**Core Idea**: The first survey of MLLM-based Agents from the unified perspective of OS, providing a panoramic overview from basic concepts to construction methodologies and evaluation systems.

## Method

### Overall Architecture
The survey is structured into 5 major modules: (1) fundamental definitions—environment, observation space, action space + three core capabilities of understanding, planning, and grounding; (2) foundation model construction—architecture selection + pre-training/SFT/RL training strategies; (3) Agent framework—training-free assembly of perception/planning/memory/action modules; (4) survey of evaluation benchmarks; (5) challenges and future directions.

### Key Designs

1. **Three-Dimensional Capability System (Understanding / Planning / Grounding)**:

    - **Function**: Defines the three core capabilities that an OS Agent must possess.
    - **Mechanism**: **Understanding**—processes multimodal observations such as HTML and GUI screenshots to handle high-resolution, information-dense interfaces; **Planning**—decomposes complex tasks into sequences of sub-tasks and dynamically adjusts based on environmental feedback (e.g., ReAct, CoAT strategies); **Grounding**—maps natural language instructions to executable actions (coordinates, input values, etc.) and precisely localizes targets among many candidate elements.
    - **Design Motivation**: The three components form a complete "perception $\rightarrow$ decision $\rightarrow$ execution" closed loop, where each is indispensable and requires distinct model capabilities.

2. **Foundation Model Taxonomy**:

    - **Function**: Categorizes the four architectures and three training strategies for OS Agent foundation models.
    - **Mechanism**: **Architecture**—direct use of LLM (T5/LLaMA reading HTML), direct use of MLLM (LLaVA/Qwen-VL reading screenshots), hybrid MLLM (encoder + LLM combination), modified MLLM (adding high-resolution encoders, such as CogAgent's $1120 \times 1120$ visual encoder). **Training**—pre-training (GUI grounding + screen understanding + OCR), SFT (trajectory collection + action annotation), RL (learning error correction and planning from environmental rewards). AutoGLM represents a model utilizing all three training strategies.
    - **Design Motivation**: A systematic classification of 30+ models allows researchers to quickly locate where their methods stand in the technology stack.

3. **Perception-Planning-Memory-Action Architecture of Agent Frameworks**:

    - **Function**: Summarizes engineering paradigms for constructing OS Agents in a training-free manner.
    - **Mechanism**: **Perception**—text description / GUI screenshot / visual grounding / semantic grounding / dual-channel grounding; **Planning**—global planning (one-time decomposition) vs. iterative planning (step-by-step adjustment); **Memory**—experience accumulation through automatic exploration, historical experience enhancement, and management mechanisms; **Action**—input operations / navigation operations / extension operations (calling external tools). 20+ frameworks are categorized along these 4 dimensions (e.g., Agent S: GS+SG perception + global planning + experience enhancement with automatic exploration and management memory + input and navigation actions).
    - **Design Motivation**: The Agent framework represents a rapid iteration path of "not modifying the model, but modifying the engineering," and tabular classification facilitates model selection.

### Future Directions
- Safety and Privacy: Agent operations involve users' sensitive data (passwords, personal info), requiring action constraints and permission management.
- Personalization and Self-Evolution: Learning user habits/preferences to continuously improve during usage.

## Key Experimental Results

### Foundation Model Comparison (Survey Summary)

| Model | Architecture | Pre-training | SFT | RL | Key Characteristics |
|------|------|--------|-----|-----|----------|
| OS-Atlas | Existing MLLM | ✓ | ✓ | - | Cross-platform GUI grounding data synthesis |
| AutoGLM | Existing LLM | ✓ | ✓ | ✓ | Self-evolving online curriculum RL |
| CogAgent | Modified MLLM | ✓ | ✓ | - | High-resolution visual encoder ($1120 \times 1120$) |
| SeeClick | Existing MLLM | ✓ | ✓ | - | Text + coordinate bidirectional grounding pre-training |
| Ferret-UI | Existing MLLM | - | ✓ | - | Any-resolution segmentation + zooming strategy |

### Agent Framework Comparison

| Framework | Perception | Planning | Memory | Action |
|------|------|------|------|------|
| Agent S | Screenshot + Semantic Grounding | Global | Exploration + Experience + Management | Input + Navigation |
| OS-Copilot | Text Description | - | - | Input + Navigation + Extension |
| ClickAgent | Screenshot | Iterative | Exploration | Input + Navigation |
| OSCAR | Screenshot + Dual-Channel Grounding | Iterative | Exploration | Extension Action |

### Key Findings
- Pure LLMs (reading HTML/text only) can actually complete a large number of OS tasks, but visual information is crucial for reducing hallucination and improving generalization.
- High-resolution processing ($1120 \times 1120$ or any-resolution) is a key bottleneck in GUI understanding—standard $224 \times 224$ cannot recognize small icons and text.
- The introduction of RL (e.g., AutoGLM) represents a qualitative leap from "mimicking humans" to "autonomous exploration," but currently, only a few works adopt it.
- Cross-platform action space unification is an easily overlooked but key engineering issue (OS-Atlas found that lack of unification leads to SFT conflicts).

## Highlights & Insights
- "OS" as a unified abstraction is an elegant perspective—it brings the work of three disjointed communities (Web, Mobile, Desktop) into a unified framework, establishing a common conceptual language (the environment/observation/action triad).
- The dual-track taxonomy of Foundation Models vs. Agent Frameworks is clear and practical—the former modifies the model while the latter modifies the engineering structure, which holds directive value for researchers choosing their technical routes.
- The tabular feature vector classification (Table 1 and Table 2) serves as an excellent reference tool for quickly identifying technical gaps.

## Limitations & Future Work
- The version accepted by ACL 2025 is a 9-page condensed version; the compression causes some technical details (such as specific metric comparisons of various benchmarks) to lack depth.
- The survey is current as of December 2024, leaving the latest developments in 2025 (such as scale feedback from Claude 3.5 Computer Use) uncovered.
- Discussions on failure cases and safety incidents remain at the prospective level, lacking actual case studies.
- The evaluation benchmarks section only lists benchmarks without deeply analyzing their limitations and interrelationships.

## Related Work & Insights
- **vs. WebAgent (Gur et al.)**: An early Web navigation work trained from scratch using HTML-T5; this survey positions it as a representative of "LLM Architecture + Pre-training + SFT".
- **vs. Computer Use (Anthropic)**: The most influential industrial OS Agent product, cited in the survey as a milestone event marking "the dream of OS Agents approaching reality".
- **vs. Cradle**: An Agent framework extended to gaming environments, displaying the generalizability potential of the OS Agent concept.
- **Value to Researchers**: Quickly grasp the landscape of OS Agents, locate one's technical direction, and discover underexplored gaps.

## Rating
- Novelty: ⭐⭐⭐ The survey itself does not propose a new methodology, but the unified "OS" perspective and dual-track classification make valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐ As a survey paper, there are no original experiments, but the categorical comparison tables are detailed.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and figures/tables are rich; the 9-page condensed version still maintains excellent readability.
- Value: ⭐⭐⭐⭐ The first systematic survey in the OS Agent field, holding high reference value for beginners and topic selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)
- [\[ACL 2025\] GUI Agents: A Survey](gui_agents_a_survey.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)
- [\[CVPR 2026\] RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward](../../CVPR2026/llm_agent/retouchiq_mllm_agents_for_instruction-based_image_retouching_with_generalist_rew.md)

</div>

<!-- RELATED:END -->
