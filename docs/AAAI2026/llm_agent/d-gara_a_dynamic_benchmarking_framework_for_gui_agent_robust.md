---
title: >-
  [Paper Note] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies
description: >-
  [AAAI 2026][LLM Agent][GUI Agent] This paper proposes D-GARA, a dynamic robustness evaluation framework for Android GUI Agents. By injecting real-world anomalies—such as permission dialogs, low-battery warnings…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "robustness evaluation"
  - "dynamic benchmark"
  - "anomaly injection"
  - "Android"
  - "interruption handling"
date: 2026-05-08
content_hash: 56cc6601f47020e3
---

# D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies

**Conference**: AAAI 2026
**arXiv**: [2511.16590](https://arxiv.org/abs/2511.16590)  
**Code**: [sen0609/D-GARA](https://github.com/sen0609/D-GARA)  
**Area**: Agent / LLM
**Keywords**: GUI Agent, robustness evaluation, dynamic benchmark, anomaly injection, Android, interruption handling

## TL;DR

This paper proposes D-GARA, a dynamic robustness evaluation framework for Android GUI Agents. By injecting real-world anomalies—such as permission dialogs, low-battery warnings, and app crashes—during live interactions, D-GARA reveals that existing SOTA agents (including UI-TARS-72B and GPT-4o) suffer an average success rate drop of over 17.5%, with a maximum degradation of 33%, under interruption scenarios.

## Background & Motivation

**Background**: GUI Agents aim to operate graphical interfaces like humans, completing complex tasks through visual understanding, task planning, and action execution—a capability considered critical on the path toward AGI.

**Limitations of Prior Work**: Existing evaluation datasets (e.g., Android Control, Mind2Web) adopt a static screenshot + fixed action sequence paradigm. While agents perform well in these idealized settings, such benchmarks conceal their fragility in real-world dynamic environments.

**Key Challenge**: Real-world smartphone usage is permeated by interruptions—permission dialogs, low-battery alerts, app update prompts, and network switches—that can disrupt an agent's execution flow or cause it to deviate entirely from the intended goal.

**Limitations of Existing "Anomaly" Evaluations**: Prior work such as GUI-Robust introduces anomalous samples but remains within a static paradigm, inserting a single pop-up screenshot into an otherwise normal trajectory. The agent merely needs to dismiss the dialog and resume the original path—essentially a teacher-forcing setup that fails to simulate complex anomalies that genuinely alter the execution trajectory.

**Limitations of Dynamic Environments**: AndroidWorld provides a dynamic evaluation environment but features overly simplistic tasks (only 116 clean tasks), does not support custom tasks, and includes no anomalies.

**Goal**: A standardized framework is needed to comprehensively evaluate agent robustness against anomalies in dynamic, real-world environments. D-GARA is designed to fill this gap.

## Method

### Overall Architecture

D-GARA is a modular dynamic evaluation framework whose core procedure forms an "execution loop": capture screenshots and XML layouts from an Android device → determine whether to inject an anomaly → pass multimodal inputs to the agent → receive action commands (tap, type, etc.) → execute via ADB → update UI state → verify whether the goal is achieved. The framework comprises two core modules—the **anomaly triggering mechanism** and the **success validation mechanism**—which operate sequentially within each execution cycle.

### Key Designs

#### 1. Semantic Anomaly Triggering

- **Function**: Automatically determines whether to inject an anomaly based on the semantic content of the current UI state.
- **Mechanism**: A lightweight rule engine inspects textual content in XML files. Each rule defines a set of keywords and a matching threshold; when the XML satisfies the condition, the corresponding anomaly is triggered. For example, a navigation-page rule specifies keywords `["Drive", "Nearby", "Metro", "Mine"]` with a threshold of 0.75 (i.e., at least three of the four words must match), upon which a location-permission dialog is injected.
- **Design Motivation**: Anomaly injection is semantically grounded in the current UI context rather than randomly inserted, making interruptions more natural and closer to real-world occurrences.

#### 2. Two-Stage Anomaly Pipeline

- **Function**: Models anomalies as two distinct phases: pop-up presentation followed by state transition.
- **Mechanism**: In Stage 1, the agent faces a foreground dialog (e.g., a permission request) and makes a choice. In Stage 2, D-GARA executes the corresponding follow-up action via ADB commands based on the agent's decision—for instance, selecting "Allow" navigates to the system settings page, while selecting "Deny" causes the app to terminate.
- **Design Motivation**: Unlike static pop-ups that can simply be dismissed, different responses to real anomalies lead to entirely different execution paths. This two-stage design simulates scenarios where anomalies redirect the agent to unexpected interfaces, testing whether the agent can maintain goal awareness and autonomously recover.

#### 3. State-Based Success Validator

- **Function**: After each action, verifies whether the UI state satisfies declaratively defined goal conditions, rather than relying on an agent-reported "done" signal.
- **Mechanism**: Checks attribute values of specific elements in the XML. For example, the success condition for a video-liking task requires the `content-desc` attribute of the element with `resource-id=tv.app:id/like_button` to equal `"Liked"`. The same rule can be reused across tasks of the same type, substantially reducing annotation overhead.
- **Design Motivation**: Agent completion signals are unreliable—smaller models may complete a task without outputting "done," while larger models may falsely report "done" and terminate prematurely. Using the target UI state as the criterion decouples evaluation from intermediate step correctness, allowing agents to take detours, make errors, or backtrack, as long as the final state meets the goal.

#### 4. Externally Configured Anomaly Definitions

- **Function**: All triggering logic is declared via external YAML configuration files rather than hard-coded.
- **Mechanism**: Each rule prototype specifies a target Activity, trigger conditions, display content, and follow-up actions. Anomaly interface templates are designed in Android Studio and compiled as standalone APKs, with parameters populated by the configuration files at runtime.
- **Design Motivation**: Decoupling anomaly logic from the target application avoids inconsistencies in native system dialogs across different devices and Android versions, while enabling researchers to add new anomaly types by extending configuration files without modifying the framework's core code.

### Evaluation Metrics

- **Success Rate (SR)**: Proportion of tasks successfully completed.
- **Robust Success Rate (RSR)**: Among tasks that can be completed under baseline conditions, the proportion that are still completed under interruption conditions, defined as:

$$\text{RSR} = \frac{|\{i \mid SR_{\text{baseline}}^{(i)}=1 \land SR_{\text{interruption}}^{(i)}=1\}|}{|\{i \mid SR_{\text{baseline}}^{(i)}=1\}|}$$

RSR decouples interruption robustness from overall task capability, enabling fair comparison across agents with different baseline abilities.

## Key Experimental Results

### Main Results: Performance Comparison With and Without Interruptions

| Model | SR (no interruption) | SR (with interruption) | RSR |
|---|---|---|---|
| Gemini2.5-flash | 80.26% | 68.42% | 73.77% |
| GPT-4o | 69.08% | 60.53% | 66.67% |
| Qwen2.5-VL-7B | 69.08% | 46.05% | 53.33% |
| UI-TARS-1.5-72B | 50.66% | 39.47% | 48.05% |
| AgentCPM-GUI-8B | 59.87% | 26.97% | 39.56% |

All models exhibit an average SR drop exceeding 17.5% under interruptions. General-purpose large models (GPT-4o, Gemini) demonstrate significantly greater robustness than GUI-specialized agents (UI-TARS, AgentCPM), suggesting that GUI-specific training primarily enhances visual perception rather than planning ability, the latter being more dependent on the underlying foundation model.

### Ablation Study: Dual-Button vs. Single-Button Interaction Mode

| Model | Dual-Button RSR | Single-Button RSR |
|---|---|---|
| Qwen2.5-VL-7B | 96.15% | 41.27% |
| AgentCPM-GUI-8B | 82.35% | 9.30% |

In dual-button mode, agents can select "Close" to skip the interruption; single-button mode forces agents to navigate a complex path (e.g., completing an installation before returning to the task). AgentCPM's RSR plummets from 82% to 9% in single-button mode, revealing that current agents' interruption-handling strategy is fundamentally "dismiss the dialog," with no genuine capacity to navigate complex anomaly recovery paths.

### Ablation Study: Effect of Input Modality (Screenshot vs. Screenshot + XML)

| Model | Input Modality | SR (no interruption) | SR (with interruption) |
|---|---|---|---|
| AgentCPM-GUI-8B | Screenshot + XML | 59.87% | 26.97% |
| AgentCPM-GUI-8B | Screenshot only | 56.58% | 19.74% |
| Gemini2.5-flash | Screenshot + XML | 80.26% | 68.42% |
| Gemini2.5-flash | Screenshot only | 45.33% | 41.33% |

Gemini's SR drops from 80% to 45% (a 35% reduction) in vision-only mode, whereas the GUI-specialized AgentCPM shows a smaller decline. This indicates that general-purpose models are heavily dependent on XML coordinate information: agents know what action to perform but not where to execute it, revealing visual grounding as a current bottleneck.

## Key Findings

- **Perception Drift**: After an app crash and recovery, an agent's decisions can be misguided by stale actions in its historical prompt. For example, GPT-4o had entered a search keyword before a crash; after the app restarts and the search bar resets to recommended terms, GPT-4o mistakenly assumes the keyword is still present based on its history and directly taps "search," causing task failure. This highlights the need for agents to distinguish between "useful historical memory" and "outdated misleading information."
- **Significant Variation in Anomaly Difficulty**: Permission Control anomalies are the easiest (Gemini/GPT-4o achieve 100% RSR) as they can be resolved with common sense; UI-TARS performs relatively better on App Malfunction anomalies due to partial coverage in its specialized training.
- **Clear Scaling Effect**: Larger models demonstrate greater robustness, likely benefiting from stronger planning capabilities—most interruptions can in principle be mitigated through effective planning.

## Highlights & Insights

1. **Qualitative Shift from Static to Dynamic Evaluation**: Unlike static pop-up insertion, anomalies in D-GARA genuinely alter execution paths—a more fundamental test of agent decision-making that requires understanding the current state rather than relying on historical trajectories.
2. **Elegant RSR Metric Design**: Decoupling robustness from baseline capability prevents the confound of agents that "already perform poorly and thus appear unaffected by interruptions," enabling fairer cross-model comparisons.
3. **"Dismissing Dialogs" ≠ Robustness**: The dual-button/single-button contrast experiment elegantly demonstrates that high RSR may simply reflect a strategy of always selecting "Close"—genuine robustness requires handling complex recovery paths.
4. **Configuration-Driven Extensibility**: All anomaly logic is externalized as YAML configurations and standalone APK templates, allowing researchers to add new anomalies without modifying core code, lowering the barrier to community adoption.

## Limitations & Future Work

1. **Android-Only Scope**: The framework does not cover Web, iOS, or desktop GUI environments; cross-platform generalizability remains to be validated.
2. **Limited Application Coverage**: D-GARA-152 includes only 152 tasks across 8 applications, with non-uniform task distribution skewed toward high-frequency apps, potentially limiting representativeness.
3. **Diagnostic Without Remediation**: The framework is positioned as a diagnostic tool and proposes no training strategies or inference methods to improve agent robustness.
4. **Manual Success Validation Design**: XML attribute-based rules require manual extraction; the paper acknowledges that human review is needed when automated validation is insufficient, constraining scalability. Future work could explore LLM-assisted validation.
5. **Coverage of Anomaly Triggering Rules**: Keyword-based semantic matching may miss edge cases, and no systematic methodology is provided for setting rule thresholds.
6. **Compound Anomalies Not Addressed**: Real-world scenarios may involve multiple simultaneous or consecutive anomalies; the framework does not appear to deeply investigate such combinatorial effects.

## Related Work & Insights

- **GUI-Robust**: Injects pop-up/login page anomalies statically; agents only need to dismiss the dialog, constituting a shallow test. D-GARA injects anomalies that alter execution paths in a dynamic environment, representing substantially higher difficulty and realism.
- **AndroidWorld**: Provides a dynamic evaluation environment but with simple tasks and no anomalies; D-GARA adds the anomaly injection dimension on top of a dynamic environment.
- **UI-TARS / AgentCPM**: Represent large-scale and lightweight GUI agent paradigms, respectively; their low RSR on D-GARA indicates that current GUI-specific training insufficiently addresses robustness.
- **Insights**: (1) Training data should incorporate large-scale interruption scenarios to teach agents "interrupt–recover–resume" strategies; (2) Inference-time mechanisms for "state consistency checking" may be needed, enabling agents to compare current visual state against historical memory and discard outdated information before each action; (3) The Perception Drift finding suggests agents require a human-like "working memory refresh" capability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first GUI Agent robustness evaluation framework to combine dynamic anomaly injection with a live Android environment; the problem is clearly defined and timely.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 models, 5 anomaly categories, and multiple ablations (interaction mode, input modality); the Perception Drift case analysis is convincing.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain is complete, the static vs. dynamic contrast is clearly articulated, and figures complement the text effectively.
- **Value**: ⭐⭐⭐⭐⭐ Provides significant diagnostic value to the GUI Agent community; the open-source framework with configuration-driven design has the potential to become a standard evaluation tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](../../ACL2026/llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](probench_benchmarking_gui_agents_with_accurate_process_infor.md)

</div>

<!-- RELATED:END -->
