---
title: >-
  [Paper Note] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining
description: >-
  [ICML 2026][LLM Agent][GUI agent] Video2GUI utilizes a four-stage pipeline—"Metadata Coarse Filtering → Video Quality Fine Filtering → Gemini-3-Pro Task/Action Extraction → High-Resolution Three-Frame Precise Spatial Gro…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "GUI agent"
  - "video-to-trajectory"
  - "coarse-to-fine filtering"
  - "spatial grounding"
  - "WildGUI dataset"
date: 2026-05-08
content_hash: e2d3e87b6be35e80
---

# Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining

**Conference**: ICML 2026  
**arXiv**: [2605.14747](https://arxiv.org/abs/2605.14747)  
**Code**: Project Page <https://weiminxiong.github.io/Video2GUI/>  
**Area**: GUI Agent / Multimodal Pretraining / Data Synthesis  
**Keywords**: GUI agent, video-to-trajectory, coarse-to-fine filtering, spatial grounding, WildGUI dataset

## TL;DR
Video2GUI utilizes a four-stage pipeline—"Metadata Coarse Filtering → Video Quality Fine Filtering → Gemini-3-Pro Task/Action Extraction → High-Resolution Three-Frame Precise Spatial Grounding"—to refine 500 million YouTube video metadata into WildGUI (12.7M trajectories, 124.5M screenshots, 1500+ apps). It improves Qwen2.5-VL/Mimo-VL performance by 5–20% across multiple GUI grounding and agent benchmarks.

## Background & Motivation
**Background**: GUI agents (MLLMs capable of autonomously clicking, typing, and scrolling on web/desktop/mobile platforms to complete tasks) represent one of the most practical directions in the trend of agentic MLLMs. A prerequisite for a generalized GUI agent is large-scale, diverse, and accurately grounded interaction trajectory data that records the complete sequence of "interface state + user action + task intent."

**Limitations of Prior Work**: (i) Human-annotated datasets (MIND2WEB, AITW, AndroidControl) are limited in scale (thousands to tens of thousands) and cover only a few hundred apps, making generalization to new interfaces/tasks difficult; (ii) Simulation environments (MiniWoB++) allow large-scale sampling but are semantically poor and lack realism compared to actual UIs; (iii) Existing web video-based works (TongUI, VideoAgentTrek) rely on foreground/background detection or inverse dynamics, learning only low-level, short-horizon visual cues without understanding the task intent behind actions, and suffer from coordinate localization errors due to frame compression.

**Key Challenge**: The tension between "Data Scale $\propto$ Annotation Cost" and the requirement for "Task-level Understanding + Pixel-level Grounding." While internet videos are a natural goldmine, most are irrelevant to GUIs; even for GUI videos, converting them into "trajectories with coordinates" requires overcoming three obstacles: task segmentation, action recognition, and pixel localization.

**Goal**: (i) To filter high-quality GUI tutorials from 500 million videos at a controllable cost; (ii) To automatically parse videos into a triad of task-level instructions + step-level actions + high-resolution coordinates; (iii) To achieve consistent performance gains on multi-platform GUI benchmarks after pretraining with synthetic data.

**Key Insight**: The authors make two critical observations. First, video metadata (titles, descriptions, keywords) can filter out 95%+ of noise for almost zero cost, reducing 500 million to 20 million, followed by dimension-wise fine filtering using omnimodal models. Second, only a few screenshots in a video represent actual changes; by decoupling trajectory extraction (long-horizon reasoning by a strong VLM on compressed frames) from spatial grounding (pixel localization on three high-resolution original frames), one can achieve both "long-horizon understanding" and "pixel-level precision."

**Core Idea**: A three-layer architecture consisting of "Coarse-to-fine video filtering + High/low-level instruction decoupling + Decoupling task reasoning from spatial grounding" to transform video data into GUI agent training assets.

## Method
Video2GUI is a pipeline divided into three stages plus a two-stage training process.

### Overall Architecture
Input: YouTube metadata of 500 million videos. Output: WildGUI dataset, where each sample is $(u, e)$—$u$ being a high-level task instruction and $e=(u, a_1, o_1, \dots, a_n, o_n)$ being a step-level trajectory with high-resolution screenshots and precise coordinates. The three stages are: (A) Coarse-to-Fine Video Filtering → 4.16M high-quality tutorials (~300k hours); (B) Trajectory Extraction → Task-level instruction + timestamp-level action descriptions + reasoning; (C) Action Spatial Grounding → Using $\pm 0.5$s high-resolution original frames to precisely locate $b_t=g_\phi(o_{t-0.5s}, o_t, o_{t+0.5s}, \tau_t)$. Finally, WildGUI is used for continued pretraining on Qwen2.5-VL/Mimo-VL, followed by post-training fine-tuning on open-source datasets.

### Key Designs

1.  **Coarse-to-Fine Video Filtering**:
    - **Function**: Filters the vast majority of noise from 500 million videos at a hierarchical cost, preventing the waste of storage and compute on vlogs or news commentary.
    - **Mechanism**: The first layer, "Meta Info Filtering," uses only titles/descriptions/keywords. DeepSeek-V3 labels 10k samples for supervision, which are distilled into a Qwen2.5-7B + classification head to classify the full 500M metadata, yielding ~20M candidates. The second layer, "Video Quality Scoring," takes the first minute of each candidate video. An omnimodal model (Qwen2.5-Omni, distilled from Gemini 3 Pro scores on 200 hours) evaluates three dimensions: Topic Relevance, Instruction Clarity, and Screen Recording Quality, ultimately retaining 4.16M videos.
    - **Design Motivation**: Analyzing 500M videos directly is infeasible due to PB-level storage/compute requirements. Metadata filtering is nearly cost-free but only ensures topic relevance; ensuring tutorial quality requires looking at the video content itself. Two-stage distillation (DeepSeek-V3 → Qwen2.5-7B; Gemini 3 Pro → Qwen2.5-Omni) scales the judgment capability of strong models at an affordable cost.

2.  **Trajectory Extraction with Sliding Window and Historical Memory**:
    - **Function**: Uses Gemini-3-Pro to parse videos (up to an hour long) into multiple instruction-trajectory pairs, where each action includes accurate timestamps, types, parameters, and reasoning.
    - **Mechanism**: Long videos are divided into continuous segments $\{S_1, \dots, S_M\}$ of up to 4 minutes. When processing segment $j$, the current segment frames and the results from previous segments $D(S_{1:j-1})$ are provided as text context. This allows the model to maintain long-term memory across segments, identifying cross-segment tasks or dependencies. The model is also required to output a "low-level instruction" (visually anchored text description) for each action to facilitate subsequent grounding. Finally, each video yields $D(V)=\{(u^{(k)}, e^{(k)})\}_{k=1}^N$.
    - **Design Motivation**: Long videos exceed the VLM context window, and isolated segment processing breaks cross-task dependencies. Existing methods (TongUI, VideoAgentTrek) rely on low-level visual cues, capturing only short-term correlations without the "why" intent. Strong VLM + Historical Memory + Dual-level instruction output enables both "understanding and explanation" in one pass.

3.  **Three-Frame High-Resolution Spatial Grounding**:
    - **Function**: Maps "approximate timestamps" to "pixel-level targets," compensating for coordinate errors caused by video compression.
    - **Mechanism**: For each action at timestamp $t$, a triplet $O_t=\{o_{t-0.5s}, o_t, o_{t+0.5s}\}$ is extracted from the original video. Gemini-3-Pro uses the low-level instruction $\tau_t$ to determine if the target is localizable and predicts the bounding box or screen coordinates $b_t=g_\phi(o_{t-0.5s}, o_t, o_{t+0.5s}, \tau_t)$. The 0.5s offset roughly matches the average duration of a single GUI action, ensuring the three frames cover the full "pre/at/post" states.
    - **Design Motivation**: Frames used in trajectory extraction are compressed, with insufficient resolution for locating pixel-level UI elements (e.g., a "Click Shoes for men" button). However, grounding is a critical supervision signal for GUI agents and requires pixel precision. Decoupling allows the VLM to handle long-horizon reasoning first before performing local grounding on high-resolution frames, saving tokens while maintaining accuracy.

### Loss & Training
Training is divided into two stages: (i) Continued Pretraining (CPT): Large-scale pretraining on Qwen2.5-VL/Mimo-VL using WildGUI to absorb interaction patterns across multiple platforms and apps; (ii) Supervised Fine-Tuning (SFT): Task-level supervision using curated open-source GUI datasets (ScreenSpot-Pro, OSWorld-G training sets, etc.) to refine for specific downstream benchmarks. The objective is to treat WildGUI as a "universal prior" and open-source data as "task-specific polishing."

## Key Experimental Results

### Main Results
Comparison on ScreenSpot-Pro and OSWorld-G grounding benchmarks:

| Model | ScreenSpot-Pro Avg | OSWorld-G Avg |
| :--- | :--- | :--- |
| Gemini-2.5-Pro (Closed) | 11.4 | 45.2 |
| Seed1.5-VL (Closed) | 60.9 | 62.9 |
| Qwen3-VL-2B (Open Baseline) | 41.9 | 45.9 |
| Qwen3-VL-8B (Open Baseline) | 49.9 | 54.8 |
| Qwen3-VL-32B | 54.9 | 60.6 |
| GTA1-7B | 50.1 | 55.1 |
| UI-Venus-7B | 50.8 | 58.8 |
| GUI-Owl-7B | 54.9 | 55.9 |
| **Qwen2.5-VL-7B + WildGUI (Ours)** | **Significant +$\Delta$** | **Significant +$\Delta$** |

The baseline Qwen2.5-VL-7B scored only 26.8 on ScreenSpot-Pro, but after continued pretraining on WildGUI, it jumped to a level comparable to Qwen3-VL-32B and GUI-Owl-7B, validating that large-scale real video data can elevate a general VLM to the level of a GUI specialist.

Dataset comparison for WildGUI:

| Dataset | Platform Coverage | # Environments | # Traj/Instr | # Screenshots | Avg Steps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AITW | Mobile | 357 | 30k | 715k | 6.5 |
| AndroidControl | Mobile | 833 | 14.5k | 15k | 4.8 |
| GUI-World | Tri-platform | – | 12k | 83k | 6.7 |
| GUI-Net | Tri-platform | 280 | 1M | 1M | 4.7 |
| MONDAY | Mobile | – | 20k | 313k | 15.7 |
| GUI-360° | Desktop | 3 | 13.75k | 105k | 7.6 |
| **WildGUI** | **Tri-platform** | **1500+** | **12.7M** | **124.5M** | **9.7** |

WildGUI leads across the board in environmental diversity (1500+), scale (12.7M trajectories), and average trajectory length (9.7 steps), covering web, mobile, and desktop platforms simultaneously.

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Full pipeline | Best | Three stages + CPT + SFT |
| w/o Coarse Meta Filter | Exploding Costs | Processing 500M videos directly is infeasible |
| w/o Fine Quality Scoring | Lower Grounding Acc | Introduces low-quality tutorials with instruction-visual mismatches |
| w/o Window Memory | Missed Tasks | Shipped videos cannot identify cross-segment tasks |
| w/o 3-Frame High-Res | High Coord Bias | Positioning directly on compressed frames degrades accuracy |
| w/o CPT (SFT only) | Significant Lag | Validates WildGUI as a universal prior |

### Key Findings
- Data scale + real diversity is the bottleneck for GUI agent generalization: 13.75k high-quality human data points (GUI-360) are far less effective than 12.7M semi-automatically synthesized data points.
- The decoupling of "Task Understanding" and "Spatial Grounding" is key to quality: the former requires long-horizon reasoning (on compressed frames), while the latter requires pixel-level localization (on high-res originals). Forcing them together sacrifices one.
- Coarse-to-fine filtering is a viable path to refine 500M videos into 4M high-quality samples; relying solely on strong models to watch every video is computationally impractical. Distilling to lightweight 7B models is the engineering key.
- Cross-platform and cross-language coverage (inherent in YouTube) significantly enhances generalization to unseen interfaces/tasks compared to single-platform datasets.

## Highlights & Insights
- Successfully established an end-to-end reproducible pipeline for "Internet Video → Agent Training Data," clearly explaining the distillation path from strong to lightweight models. This "coarse-to-fine + distillation" strategy can be transferred to scenes like robot demonstration videos, autonomous driving dashcams, or converting educational slides into structured content.
- The decoupling design reveals a universal principle: when token budgets or resolution are limited by the context window, decomposing a target into "low-res long context" + "high-res short context" steps often outperforms simply stacking tokens. This is valuable for long video understanding, document parsing, and script generation.
- Splitting instructions into "High-level User Intent" + "Low-level Visual Anchoring" provides a "dual-track" paradigm for training GUI agents, offering significantly more information than simple click coordinates.

## Limitations & Future Work
- The pipeline relies heavily on closed-source teachers (Gemini-3-Pro, DeepSeek-V3), creating high barriers to entry and cost; the quality ceiling for distilled models is locked to the teachers.
- The "Three-frame $\pm 0.5$s" heuristic for spatial grounding assumes an average action duration of 0.5s, which may cause localization errors for longer actions like long-press, drag, or double-click.
- Data generation is fully automated and lacks an "Executor Closed-loop Verification" stage—there is no feedback on whether predicted actions can actually be reproduced in a real environment.
- Result reporting is detailed for grounding benchmarks, but online agent benchmark (e.g., OSWorld full tasks, WebArena) data is limited; the end-to-end execution improvement needs more evaluation.
- Copyright, bias, and data contamination (privacy) issues in 500M videos require more rigorous handling processes.

## Related Work & Insights
- **vs TongUI / VideoAgentTrek**: Those rely on foreground/background detection or inverse dynamics to extract trajectories from short videos, learning only low-level cues. Video2GUI uses strong VLMs + sliding windows for long-horizon task understanding with explicit reasoning.
- **vs Human Datasets**: AITW/MIND2WEB are limited to 10k-30k samples on single platforms. WildGUI provides true universality across 12.7M trajectories and 1500+ apps.
- **vs Synthetic Methods**: GUI-Net/GUI-World use LLMs to synthesize from HTML/screenshots, lacking the temporal dynamics and context of real videos. Video2GUI mines real use cases for a distribution closer to real-world scenarios.
- **Insight**: (i) Distilling large models to small models for large-scale filtering is a critical engineering pattern for data synthesis; (ii) The "low-res long context + local high-precision" decoupling framework is applicable to any multimodal task requiring both semantics and precision; (iii) Dual-track instructions (high/low level) significantly boost agent performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The data synthesis route is an engineering innovation. The design of coarse-to-fine distillation + trajectory/grounding decoupling is the first systematic presentation for GUI agent data generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison of 10+ models on ScreenSpot-Pro and OSWorld-G with detailed dataset comparisons; agent benchmark coverage could be expanded.
- Writing Quality: ⭐⭐⭐⭐ Pipeline diagrams are clear; Table 1 provides an excellent comparison; motivations and pain points are well-articulated.
- Value: ⭐⭐⭐⭐⭐ The release of 12.7M real multi-platform trajectories would be a rare and massive contribution to the open-source GUI agent community, potentially driving the ecosystem more than the method itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents](../../AAAI2026/llm_agent/tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)
- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[ACL 2026\] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](../../ACL2026/llm_agent/lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)
- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)

</div>

<!-- RELATED:END -->
