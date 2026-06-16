---
title: >-
  [Paper Note] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents
description: >-
  [CVPR 2026][LLM Agent][GUI Agent] This work proposes GUI-CEval, the first comprehensive benchmark for Chinese mobile GUI Agents. It covers 201 mainstream Chinese Apps across 4 device types, utilizing a "Foundation + Application" two-layer structure to conduct fine-grained diagnosis across five dimensions: perception, planning, reflection, execution, an
tags:
  - CVPR 2026
  - LLM Agent
  - GUI Agent
date: 2026-05-08
content_hash: d7fbe5e07b8cd8ac
---
# GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents

**Conference**: CVPR 2026  
**arXiv**: [2603.15039](https://arxiv.org/abs/2603.15039)  
**Code**: TBD  
**Area**: LLM Agent  
**Keywords**: GUI Agent, Chinese mobile benchmark, multimodal evaluation, hierarchical diagnosis, mobile interaction

## TL;DR
This work proposes GUI-CEval, the first comprehensive benchmark for Chinese mobile GUI Agents. It covers 201 mainstream Chinese Apps across 4 device types, utilizing a "Foundation + Application" two-layer structure to conduct fine-grained diagnosis across five dimensions: perception, planning, reflection, execution, and evaluation. Experiments on 20 representative models reveal that current models still exhibit significant weaknesses in reflection and self-evaluation.

## Background & Motivation
**Background**: The rapid development of MLLMs has birthed mobile GUI Agents with visual perception, cross-modal reasoning, and interaction control capabilities. Several benchmarks such as ScreenSpot, AndroidControl, and AndroidWorld have driven progress in this field.

**Limitations of Prior Work**: (i) Language bias—the vast majority of benchmarks are primarily in English, failing to reflect the linguistic and interactive characteristics of the Chinese mobile ecosystem; (ii) Inconsistent scenarios—data originates from disparate platforms, lacking focused evaluation specifically for mobile devices; (iii) One-sided tasks—existing benchmarks either only measure UI element localization or only measure offline Agent success rates, lacking a unified full-pipeline capability assessment; (iv) Insufficient data authenticity—automated collection and verification often overlook real user intent.

**Key Challenge**: There is a lack of a unified, fine-grained, and diagnostic evaluation framework for Chinese mobile GUI Agents, which prevents systematic identification of weak links across the entire pipeline from perception to execution.

**Goal**: To construct the first comprehensive benchmark for Chinese mobile GUI Agents that simultaneously evaluates atomic skills and end-to-end application capabilities.

**Key Insight**: A hierarchical design (Foundation + Application) is adopted to define five core dimensions along the complete Agent workflow, with all data manually collected and verified on real devices.

**Core Idea**: To achieve comprehensive and diagnostic evaluation from atomic skills to end-to-end execution in real Chinese mobile environments through a two-layer structure and a five-dimension design.

## Method

### Overall Architecture
GUI-CEval aims to address an issue often avoided by existing English benchmarks: where exactly does a GUI Agent fail in the pipeline from "understanding a Chinese interface" to "completing a task on a real device"? To this end, it defines five capability dimensions—perception, planning, reflection, execution, and evaluation—along the complete Agent workflow. These are distributed across two layers, allowing atomic skills and end-to-end execution to share the same set of real screenshots and Apps, thereby enabling precise failure attribution to specific capability dimensions. The Foundation layer uses 4,194 multimodal QAs to independently diagnose four types of **atomic skills** (perception, planning, reflection, evaluation). The Application layer uses 4,028 Agent tasks to unify three scenarios—GUI Grounding, Offline Agent, and Online Agent—to examine **full-link execution** (the fifth dimension, "execution," is assessed end-to-end at this layer). The entire dataset covers 201 mainstream Chinese Apps and 4 real device types (phones, tablets, foldable screens, etc.), all manually collected on real hardware.

### Key Designs

**1. Foundation Capability Layer: Decomposing atomic skills into independently diagnostic multimodal QAs**

Existing benchmarks either only measure element localization or only measure offline success rates, where a single aggregate score masks the specific weaknesses of a model. GUI-CEval defines five dimensions along the Agent workflow (perception, planning, reflection, execution, evaluation). Four atomic skills—perception, planning, reflection, and evaluation—are scored independently in the Foundation layer using multimodal QAs (the fifth dimension, "execution," is left for end-to-end assessment in the Application layer). Consequently, a capability radar chart can pinpoint specific shortcomings. **Perception** examines the model's recognition of the current App, page, operable controls, and on-screen text, further decomposing element localization into appearance, spatial, and functional cues. To prevent sensitivity to coordinate number formats, localization queries use Set-of-Marks (SoM) bounding boxes, requiring the model to select an ID rather than outputting coordinates. **Planning** requires the model to predict the next action given high-level instructions and the current screenshot, subdivided into task planning (understanding and decomposing global goals) and action decision (selecting from the action space). **Reflection** distinguishes Agent autonomy and is divided into short-term reflection (judging the correctness of the immediate single step) and long-range reflection (identifying errors or redundant steps within a trajectory). **Evaluation** assesses the model's ability as a judge: determining task completion, inferring execution intent, and reordering shuffled screenshot sequences into the correct steps.

**2. Three-level Application Tasks: Using scenarios with increasing noise to isolate "correct logic" from "real-device completion"**

Possessing atomic skills does not guarantee subsequent end-to-end task completion, as real-device phenomena like pop-up ads can conflate planning ability with environmental robustness. GUI-CEval presents three levels of application scenarios with increasing environmental noise to approximate real-world usage. **GUI Grounding** is the most basic, requiring the model to select the correct interaction location given a screenshot and a natural language instruction, measuring only clicking accuracy. **Offline Agent** involves replaying interactions within pre-recorded static snapshots, where the model iteratively predicts the next action and parameters—because snapshots are fixed and system noise is eliminated, this level isolates planning and decision-making capabilities. **Online Agent** deploys the Agent directly onto real devices, exposing it to pop-ups, ads, permission prompts, and network fluctuations, testing the model's comprehensive survival capability in a real Chinese mobile environment. These three levels share the same Apps and pages, allowing gaps such as "successful grounding vs. online failure" to be directly observed.

**3. Real-device Manual Collection + Three-stage Quality Control: Blocking template bias and leakage via reproducible real-world data**

While automated scripts are efficient for batch data acquisition, they introduce templated biases and risk overlapping with training corpora, distorting the benchmark. GUI-CEval ensures all data is manually collected on real mobile devices, including single-image collection (screenshots paired with XML layouts) and trajectory collection (logs of 10–20 step real instruction executions). Post-collection, data passes through three quality control stages: manual cross-checking of annotations, followed by automated verification—using a dual-verification system with one strong and one weak model (only tasks where the strong model answers correctly and the weak model fails are considered appropriately difficult and reliable), and finally, a manual review of a 20% sample of the full set. This "real-device collection + strong/weak model verification + sampling review" pipeline ensures data authenticity and reproducibility, making it harder for models to achieve high scores through memorization.

## Key Experimental Results

### Main Results
Evaluation was conducted on 20 representative models (general MLLMs + GUI-specific models + multi-Agent systems):

| Model | Perception | Planning | Reflection | Evaluation | Grounding | Offline | Online | Average |
|------|------|------|------|------|-----------|---------|--------|------|
| Qwen2.5-VL-72B | 82.28 | 66.68 | 21.01 | 40.09 | 88.10 | 70.30 | 26.94 | 61.41 |
| UI-TARS-72B-SFT | 70.28 | 45.49 | 10.97 | 41.08 | 90.10 | 79.40 | 33.33 | 56.22 |
| Qwen2.5-VL-32B | 75.25 | 63.57 | 19.24 | 49.24 | 88.70 | 70.00 | 31.87 | 55.46 |
| MIMO-VL-RL | 73.56 | 51.67 | 15.55 | 46.78 | 90.00 | 60.80 | 17.22 | 49.67 |
| GPT-4o | 37.55 | 26.06 | 13.60 | 35.72 | 35.10 | 25.50 | 0.83 | 27.69 |

### Key Findings

| Dimension | Best Model | Best Score | Description |
|------|---------|---------|------|
| Reflection | Qwen2.5-VL-72B | 21.01 | Reflection ability is weak across all models, peaking at only 21% |
| Online Agent | UI-TARS-72B-SFT | 33.33 | Online execution capability is generally low |
| Grounding | UI-TARS-72B-SFT | 90.10 | GUI localization capability is relatively strong |

### Key Findings
- **Reflection is the primary bottleneck**: Scores for all models in the Reflection dimension are significantly low (maxing out at 21%), indicating that current models are severely deficient in self-checking and error correction.
- **Specialized vs. General GUI Model Trade-offs**: UI-TARS performs stronger in Grounding and Offline/Online Agent tasks, whereas general models (Qwen2.5-VL) excel in perception, planning, and reflection.
- **Online Agent tasks pose a massive challenge**: Even the best model achieves only approximately 33% on Online Agent tasks, demonstrating that real-world interference like pop-ups and ads remains a severe challenge.
- **Gap between Chinese and English GUI performance**: GPT-4o's performance on this benchmark is significantly lower than its performance on English benchmarks, highlighting the unique challenges of the Chinese mobile ecosystem.

## Highlights & Insights
- The **two-layer, five-dimension design** is highly systematic—it decomposes Agent capabilities from atomic to end-to-end levels hierarchically, enabling both overall assessment and pinpointing of weaknesses, offering higher diagnostic value than existing benchmarks.
- The design where **Foundation and Application tasks share the same Apps and pages** is ingenious—it allows for direct correlation between localization capabilities and execution outcomes, enabling precise attribution of failure causes.
- The **introduction of the Reflection dimension** is forward-looking: reflection is a critical capability for Agent autonomy, yet almost all models remain weak in this area, providing a clear direction for future research.

## Limitations & Future Work
- Only covers Chinese mobile environments and lacks cross-lingual comparisons (comparisons between Chinese and English versions of the same App would be more persuasive).
- While spanning 201 Apps, coverage may still miss certain vertical domains (e.g., complex interactions in financial or medical Apps).
- Online Agent evaluation is restricted to common functions that "do not require login"; many critical functions in real scenarios require authentication for evaluation.
- The dataset size (8,222 entries) remains relatively small compared to large-scale English benchmarks.

## Related Work & Insights
- **vs ScreenSpot/ScreenSpot Pro**: These only evaluate GUI Grounding, whereas GUI-CEval covers the complete perception-to-execution pipeline.
- **vs AndroidWorld**: AndroidWorld contains only 116 online tasks and is in English; GUI-CEval provides 4,028 application tasks and focuses on Chinese.
- **vs MMBench-GUI**: While MMBench-GUI is cross-platform, it lacks online Agent evaluation; GUI-CEval provides a deeper dive into the mobile domain.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive benchmark for Chinese mobile GUI Agents, filling a significant gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 20 models and 47 configurations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-defined tasks.
- Value: ⭐⭐⭐⭐ Provides a substantial push for research into Chinese mobile Agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

| Paper | Venue | Highlights |
|-------|-------|------------|
| ScreenSpot | CVPR 2024 | Benchmark for GUI element localization. |
| AndroidWorld | ICML 2024 | Scalable environment for autonomous Android agents. |
| UI-TARS | arXiv 2024 | Advanced model for GUI interaction trained on large-scale data. |

</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] MMBench-GUI: A Unified Hierarchical Evaluation Framework for Multi-Platform GUI Agents](mmbench-gui_a_unified_hierarchical_evaluation_framework_for_multi-platform_gui_a.md)
- [\[CVPR 2026\] ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices](proactivemobile_a_comprehensive_benchmark_for_boosting_proactive_intelligence_on.md)
- [\[CVPR 2026\] OS-Oracle: A Comprehensive Framework for Cross-Platform GUI Critic Models](os-oracle_a_comprehensive_framework_for_cross-platform_gui_critic_models.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)

</div>

<!-- RELATED:END -->
