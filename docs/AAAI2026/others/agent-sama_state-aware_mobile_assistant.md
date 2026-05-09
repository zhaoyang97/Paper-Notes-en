---
title: >-
  [Paper Note] Agent-SAMA: State-Aware Mobile Assistant
description: >-
  [AAAI 2026][GUI Agent] This paper proposes Agent-SAMA, which for the first time introduces a finite state machine (FSM) into mobile GUI agents, modeling UI screens as states and user actions as transitions. Four specialized agents collaborate to achieve state-aware task planning, execution verification, and error recovery, improving success rate by up to 12% and recovery rate by 13.8% on cross-app benchmarks.
tags:
  - AAAI 2026
  - GUI Agent
  - Finite State Machine
  - Multi-Agent Collaboration
  - Error Recovery
  - Mobile Task Automation
date: 2026-05-08
content_hash: 96bd0638d0d6ed45
---

# Agent-SAMA: State-Aware Mobile Assistant

**Conference**: AAAI 2026
**arXiv**: [2505.23596v3](https://arxiv.org/abs/2505.23596v3)
**Code**: [Zenodo](https://doi.org/10.5281/zenodo.15430187)
**Area**: Other
**Keywords**: GUI Agent, Finite State Machine, Multi-Agent Collaboration, Error Recovery, Mobile Task Automation

## TL;DR
This paper proposes Agent-SAMA, which for the first time introduces a finite state machine (FSM) into mobile GUI agents, modeling UI screens as states and user actions as transitions. Four specialized agents collaborate to achieve state-aware task planning, execution verification, and error recovery, improving success rate by up to 12% and recovery rate by 13.8% on cross-app benchmarks.

## Background & Motivation
Mobile GUI agents leverage MLLMs to interpret UI screenshots and execute actions such as tapping and swiping, with prior work including AppAgent and the Mobile-Agent series. However, existing agents are fundamentally **reactive**—they determine the next action solely based on the current screen, lacking a structured representation of app navigation flow. This resembles a tourist navigating street by street, aware of visited locations but without a global understanding of the overall route. This leads to three critical limitations: (1) inability to understand execution context (i.e., the current stage within a task); (2) inability to detect whether action outcomes match expectations; and (3) absence of structured support for error recovery, making agents prone to repetitive failure loops.

## Core Problem
How to provide GUI agents with a structured representation of app navigation that enables them to track execution progress, anticipate action outcomes, and precisely roll back to stable states upon failure? This is particularly critical for long-horizon cross-app tasks, where action chains are lengthy, error probability is high, and reactive agents are fundamentally insufficient.

## Method

### Overall Architecture
Agent-SAMA is a multi-agent framework comprising four phases: Planning → Execution → Verification & Recovery → Knowledge Retention. The core innovation lies in modeling app interactions using an FSM $\mathcal{M} = (S, A, T, s_0, G)$: UI screens are represented as states $S$, user actions as $A$, and screen-to-screen transitions as the transition function $T$. The FSM is constructed incrementally in real time during execution.

### Key Designs
1. **Planner Agent + LLM-as-Judge Planning**: High-level tasks are decomposed into a sub-task sequence $\pi = [(g_1, r_1), ..., (g_k, r_k)]$, where each sub-task is accompanied by a rationale. The key innovation is generating five candidate plans and then applying LLM-as-judges to evaluate them across dimensions such as goal relevance, execution efficiency, and robustness, selecting the optimal plan to avoid suboptimal single-path planning.
2. **State Agent + Real-Time FSM Construction**: The core module of the execution phase. A Screen Parser extracts UI element coordinates and descriptions; the State Agent maps each screen to an FSM node containing three elements: current state description $d_i$, predicted next state $d_{i+1}$, and pre/post-conditions. To mitigate state explosion, a **State Beacon** mechanism is introduced—each state is assigned a concise semantic label (e.g., "Homepage of Walmart"), and newly encountered states are first matched against existing beacons, with matches reusing existing nodes. In cross-app tasks, each app maintains an independent FSM.
3. **Reflection Agent for Error Recovery**: Structured verification and recovery are enabled via the FSM. The FSM-predicted transition (including post-conditions) is compared against the actual screen, yielding one of three verdicts: Success / NoChange / Fail. Upon failure, the FSM is used to identify a previously validated stable state $s_j$, and a recovery plan is generated to roll back and retry. If recovery fails consecutively ($n=2$ times), the process escalates to the Planner for re-planning, preventing infinite recovery loops.
4. **Mentor Agent for Knowledge Retention**: Upon task completion, reusable knowledge $K$ (action sequences, guiding cues, and constructed FSMs) is extracted and stored in long-term memory. At the start of new tasks, relevant knowledge is retrieved as context (e.g., a shopping FSM for Walmart can be transferred to Amazon), improving planning efficiency and robustness.

### Loss & Training
No training is required; the entire framework is driven by prompt engineering over an MLLM (GPT-4o) with temperature set to 0 to reduce variability. The Screen Parser employs DBNet for OCR, GroundingDINO for icon localization, and Qwen-VL-Plus for icon description generation.

## Key Experimental Results

| Dataset | Metric | Agent-SAMA | Mobile-Agent-E+Evo | Gain |
|--------|------|-----------|-------------|------|
| Mobile-Eval-E | Success Rate | 84.0% | 72.0% | +12.0% |
| Mobile-Eval-E | Recovery Success | 71.88% | 67.34% | +4.53% |
| Mobile-Eval-E | Action Accuracy | 83.24% | 76.65% | +6.59% |
| Mobile-Eval-E | Satisfaction Score | 86.15% | 78.97% | +7.18% |
| SPA-Bench | Success Rate | 80.0% | 75.0% | +5.0% |
| SPA-Bench | Recovery Success | 66.67% | 52.86% | +13.81% |
| AndroidWorld | Success Rate | 63.7% | 53.4% | +10.3% |

### Ablation Study
- **Planning module has the greatest impact**: Removing the Planner causes SR to drop from 84% to 52% on Mobile-Eval-E, and from 80% to 45% on SPA-Bench.
- **Multi-plan selection is effective**: Single-path planning vs. 5-candidate + Judge selection yields an SR difference of approximately 12%.
- **Pre/Post-conditions**: Removing them reduces SR from 84% to 72% on Mobile-Eval-E, with a notable impact on verification and recovery quality.
- **Mentor knowledge retention**: Removing it reduces SR from 84% to 68% on Mobile-Eval-E, demonstrating the importance of cross-task knowledge transfer.
- All four components are complementary and mutually reinforcing; none is dispensable.

## Highlights & Insights
- **First application of FSM to GUI agents**: Modeling mobile app interactions as a state machine is a natural and effective abstraction, providing agents with structured memory and reasoning support.
- **State Beacon deduplication mechanism**: Concise and practical, it effectively mitigates the state explosion problem.
- **Elegant error recovery design**: The FSM provides stable anchor points, and a hierarchical recovery strategy (local rollback first, then global re-planning) demonstrably reduces execution errors (32 vs. 49 on Mobile-Eval-E).
- **Agent-SAMA remains competitive under weaker MLLMs**: When using Claude 3.5, it still outperforms the GPT-4o-based baseline, indicating that the framework's gains are independent of the underlying model.
- **Model-agnostic method**: The FSM layer can serve as a lightweight memory module pluggable into any existing GUI agent.

## Limitations & Future Work
- **State Beacon relies on LLM text matching**: This may introduce false matches or missed matches; replacing it with visual-semantic embeddings for more efficient matching is worth exploring.
- **Flat FSM may suffer state explosion in ultra-long-horizon tasks**: Tasks spanning 5+ apps and 20+ steps lack hierarchical abstraction.
- **Only three benchmarks are evaluated**: Scenarios involving dynamic content (e.g., ad pop-ups) and external interruptions are not assessed.
- **Performance varies across runs**: Differences exist between the results reported in the paper and reproduced baseline results; uncertainty remains despite averaging over 5 runs.
- **The effectiveness of cross-app FSM transfer in knowledge retention** has not been quantified in isolation.

## Related Work & Insights
- **vs. Mobile-Agent-E+Evo**: Both employ a multi-agent architecture and long-term memory, but Agent-SAMA adds FSM-based structured representation, yielding more precise recovery (recovery rate 4.5%–13.8% higher) and fewer execution errors.
- **vs. GUI-Xplore**: Both use graph-based modeling of app navigation, but GUI-Xplore constructs a static graph offline from videos for inference-time reasoning; Agent-SAMA builds the FSM online in real time for execution decision-making and recovery—an "online and practical" counterpart.
- **vs. AgentS2/V-Droid**: These are strong baselines on AndroidWorld (54.3%/59.5%); Agent-SAMA surpasses both at 63.7%, while placing greater emphasis on long-horizon cross-app tasks.
- **FSM as a general-purpose agent memory layer**: This paper demonstrates that an FSM can function as a "lightweight, model-agnostic memory layer"—a concept generalizable to web agents, desktop agents, and even embodied agents.
- **LLM-as-Judge for plan selection**: Generating multiple candidate plans and scoring them with a judge is more robust than single-pass generation; this design pattern is transferable to other agent settings such as code generation and research automation.
- **Formalization via pre/post-conditions**: Rooted in Design by Contract from classical software engineering, this approach regains practical value in the context of LLM-based agents.

## Rating
- Novelty: ⭐⭐⭐⭐ (The idea of modeling app interactions as FSMs is natural; the genuine contribution lies in the engineering instantiation and multi-agent collaboration design.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three benchmarks, ablation studies, and multi-MLLM comparisons are provided, though the effectiveness of cross-app knowledge transfer lacks quantitative analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Structure is clear, figures are intuitive, and the appendix includes complete prompts and case studies; some tables are slightly disorganized.)
- Value: ⭐⭐⭐⭐ (The FSM-as-agent-memory-layer idea is broadly applicable; code is open-sourced and can serve as a foundation for future research.)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

<!-- RELATED:END -->
