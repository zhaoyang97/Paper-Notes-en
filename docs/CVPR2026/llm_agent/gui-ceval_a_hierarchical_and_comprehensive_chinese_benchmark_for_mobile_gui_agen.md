---
title: >-
  [Paper Note] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents
description: >-
  [CVPR 2026][LLM Agent][GUI Agent] This paper proposes GUI-CEval, the first comprehensive benchmark for Chinese mobile GUI agents, covering 201 mainstream Chinese apps and 4 device types. It adopts a two-tier structure (foundation + application) to perform fine-grained diagnosis across five dimensions—perception, planning, reflection, execution, and evaluation. Experiments on 20 representative models reveal that current models exhibit significant deficiencies in reflection and self-evaluation.
tags:
  - CVPR 2026
  - LLM Agent
  - GUI Agent
  - Chinese Mobile Benchmark
  - Multimodal Evaluation
  - Hierarchical Diagnosis
  - Mobile Interaction
date: 2026-05-08
content_hash: 148e7708a1c802ac
---

# GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents

**Conference**: CVPR 2026
**arXiv**: [2603.15039](https://arxiv.org/abs/2603.15039)
**Code**: To be released
**Area**: LLM Agent
**Keywords**: GUI Agent, Chinese Mobile Benchmark, Multimodal Evaluation, Hierarchical Diagnosis, Mobile Interaction

## TL;DR
This paper proposes GUI-CEval, the first comprehensive benchmark for Chinese mobile GUI agents, covering 201 mainstream Chinese apps and 4 device types. It adopts a two-tier structure (foundation + application) to perform fine-grained diagnosis across five dimensions—perception, planning, reflection, execution, and evaluation. Experiments on 20 representative models reveal that current models exhibit significant deficiencies in reflection and self-evaluation.

## Background & Motivation
**Background**: The rapid development of MLLMs has given rise to mobile GUI agents capable of visual perception, cross-modal reasoning, and interactive control. Several benchmarks, including ScreenSpot, AndroidControl, and AndroidWorld, have driven progress in this area.

**Limitations of Prior Work**: (i) *Language bias*—the vast majority of benchmarks are English-centric and fail to reflect the linguistic and interaction characteristics of the Chinese mobile ecosystem; (ii) *Inconsistent scenarios*—data is drawn from heterogeneous platforms, lacking a focused evaluation targeting the mobile domain; (iii) *Task one-sidedness*—existing benchmarks either test only UI element localization or only offline agent success rates, without a unified end-to-end capability assessment; (iv) *Insufficient data authenticity*—automated collection and verification overlook real user intent.

**Key Challenge**: The absence of a unified, fine-grained, and diagnosable evaluation framework for Chinese mobile GUI agents makes it impossible to systematically identify weaknesses across the full pipeline from perception to execution.

**Goal**: To construct the first comprehensive benchmark for Chinese mobile GUI agents that simultaneously evaluates atomic capabilities and end-to-end application performance.

**Key Insight**: A hierarchical design (foundation + application) is adopted, defining five core dimensions along the complete agent workflow, with all data manually collected and verified on real devices.

**Core Idea**: Through a two-tier, five-dimension design, GUI-CEval enables comprehensive and diagnosable evaluation—from atomic skills to end-to-end execution—within a real Chinese mobile environment.

## Method

### Overall Architecture
GUI-CEval comprises 4,194 multimodal QA tasks and 4,028 agent application tasks, covering 201 mainstream Chinese apps across 4 real device types (smartphones, tablets, foldable screens, etc.). Evaluation is organized into two tiers:

- **Foundation Tier**: Atomic skills are assessed via multimodal QA, enabling fine-grained diagnosis.
- **Application Tier**: Three scenarios—GUI Grounding, Offline Agent, and Online Agent—are unified to evaluate end-to-end execution capability.

### Key Designs

1. **Five-Dimension Foundation Evaluation**:

    - **Perception**: Assesses recognition of the current app, page, operable controls, and on-screen text. Element localization is decomposed into appearance, spatial, and functional cues; Set-of-Marks (SoM) candidate boxes are used to reduce sensitivity to coordinate representation.
    - **Planning**: Given a high-level instruction and the current screenshot, the model predicts the correct next action. Decomposed into task planning (global goal understanding and decomposition) and action decision-making (selection from a feasible action space).
    - **Reflection**: Evaluates the agent's ability to self-inspect and correct errors. Includes short-horizon reflection (whether a single-step action is correct) and long-horizon reflection (identifying erroneous or redundant steps across an entire trajectory).
    - **Execution**: Encompasses GUI Grounding and actual operation execution within Offline/Online Agent scenarios.
    - **Evaluation**: Involves judging task completion, inferring execution intent, and reconstructing the correct step sequence from shuffled screenshots.

2. **Three-Level Application Tasks**:

    - **GUI Grounding**: Given a screenshot and a natural language instruction, the model selects the correct interaction location, measuring click accuracy.
    - **Offline Agent**: Interactions are replayed on static snapshots; the model iteratively predicts the next action and its parameters, with system noise eliminated to isolate planning and decision-making capabilities.
    - **Online Agent**: The agent is tested on real devices, exposed to real-world challenges such as pop-ups, advertisements, permission prompts, and network fluctuations.

3. **Data Collection & Quality Control**:

    - **Function**: All data are manually collected on real mobile devices, including single-image collection (screenshot + XML) and trajectory collection (10–20 instruction execution records per trajectory).
    - **Mechanism**: A three-stage quality control pipeline—manual cross-checking → automated quality verification (dual validation with both strong and weak models) → human evaluation (20% sampling validation).
    - **Design Motivation**: To ensure data authenticity and reproducibility, and to avoid template bias and data leakage introduced by automated collection.

## Key Experimental Results

### Main Results
Evaluation is conducted on 20 representative models (general MLLMs, GUI-specialized models, and multi-agent systems):

| Model | Perception | Planning | Reflection | Evaluation | Grounding | Offline | Online | Avg |
|-------|-----------|----------|-----------|-----------|-----------|---------|--------|-----|
| Qwen2.5-VL-72B | 82.28 | 66.68 | 21.01 | 40.09 | 88.10 | 70.30 | 26.94 | 61.41 |
| UI-TARS-72B-SFT | 70.28 | 45.49 | 10.97 | 41.08 | 90.10 | 79.40 | 33.33 | 56.22 |
| Qwen2.5-VL-32B | 75.25 | 63.57 | 19.24 | 49.24 | 88.70 | 70.00 | 31.87 | 55.46 |
| MIMO-VL-RL | 73.56 | 51.67 | 15.55 | 46.78 | 90.00 | 60.80 | 17.22 | 49.67 |
| GPT-4o | 37.55 | 26.06 | 13.60 | 35.72 | 35.10 | 25.50 | 0.83 | 27.69 |

### Ablation Study

| Dimension | Best Model | Best Score | Note |
|-----------|-----------|-----------|------|
| Reflection | Qwen2.5-VL-72B | 21.01 | Reflection is weak across all models; best score is only 21% |
| Online Agent | UI-TARS-72B-SFT | 33.33 | Overall online execution capability is low |
| Grounding | UI-TARS-72B-SFT | 90.10 | GUI localization capability is relatively strong |

### Key Findings
- **Reflection is the most critical bottleneck**: All models score noticeably low on the Reflection dimension (maximum 21%), indicating severe deficiencies in self-inspection and error correction.
- **Cross-advantage between GUI-specialized and general models**: UI-TARS demonstrates stronger performance on Grounding and Offline/Online Agent tasks, while general-purpose models (Qwen2.5-VL) outperform on perception, planning, and reflection.
- **Online Agent remains highly challenging**: Even the best-performing model achieves only approximately 33% on Online Agent tasks, demonstrating that real-world interference such as pop-ups and advertisements remains a formidable challenge.
- **Chinese vs. English GUI gap**: GPT-4o performs substantially lower on this benchmark than on English counterparts, highlighting the unique challenges posed by the Chinese mobile environment.

## Highlights & Insights
- The **two-tier, five-dimension design** is highly systematic—it provides a clear hierarchical decomposition of agent capabilities from atomic to end-to-end levels, enabling both holistic evaluation and targeted diagnosis, which confers greater diagnostic value than existing benchmarks.
- The design in which **foundation and application tasks share the same apps and pages** is elegant—it directly correlates localization capability with execution outcomes, enabling precise attribution of failure causes.
- The **introduction of the Reflection dimension** is forward-looking: reflection is a critical capability for agent autonomy, yet virtually all models perform poorly on it, pointing to a clear direction for future research.

## Limitations & Future Work
- Coverage is limited to the Chinese mobile domain; cross-lingual comparison is absent (a paired comparison of Chinese and English versions of the same app would be more compelling).
- Although 201 apps provide broad coverage, certain vertical domains (e.g., complex interactions in financial or medical apps) may still be underrepresented.
- Online Agent evaluation is constrained to commonly used functions that require no login; many critical real-world functions can only be assessed post-authentication.
- The dataset scale (8,222 instances) remains relatively small compared to large-scale English benchmarks.

## Related Work & Insights
- **vs. ScreenSpot/ScreenSpot Pro**: These benchmarks evaluate only GUI Grounding, whereas GUI-CEval covers the full perception-to-execution pipeline.
- **vs. AndroidWorld**: AndroidWorld contains only 116 online tasks and is English-only; GUI-CEval includes 4,028 application tasks with an exclusive focus on Chinese.
- **vs. MMBench-GUI**: MMBench-GUI is cross-platform but lacks online agent evaluation; GUI-CEval provides deeper coverage within the mobile domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First comprehensive benchmark for Chinese mobile GUI agents, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 20 models and 47 configurations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-defined task formulations.
- **Value**: ⭐⭐⭐⭐ — Strong contribution to advancing Chinese mobile agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](../../ICLR2026/llm_agent/m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

</div>

<!-- RELATED:END -->
