---
title: >-
  [Paper Note] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining
description: >-
  [ICML 2026][LLM Agent][GUI agent] Video2GUI employs a four-stage pipeline—coarse metadata filtering → fine-grained video quality filtering → Gemini-3-Pro for task/action extraction → high-resolution three-frame precise s…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "GUI agent"
  - "video-to-trajectory"
  - "coarse-to-fine filtering"
  - "spatial grounding"
  - "WildGUI dataset"
date: 2026-05-08
content_hash: 557e0f689e90f729
---

# Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining

**Conference**: ICML 2026  
**arXiv**: [2605.14747](https://arxiv.org/abs/2605.14747)  
**Code**: Project page <https://weiminxiong.github.io/Video2GUI/>  
**Area**: GUI Agent / Multimodal Pretraining / Data Synthesis  
**Keywords**: GUI agent, video-to-trajectory, coarse-to-fine filtering, spatial grounding, WildGUI dataset

## TL;DR
Video2GUI employs a four-stage pipeline—coarse metadata filtering → fine-grained video quality filtering → Gemini-3-Pro for task/action extraction → high-resolution three-frame precise spatial grounding—to distill 500 million YouTube video metadata entries into WildGUI (12.7M trajectories, 124.5M screenshots, 1500+ applications), boosting Qwen2.5-VL/Mimo-VL by 5–20% on multiple GUI grounding and agent benchmarks.

## Background & Motivation
**Background**: GUI agents (MLLMs capable of autonomously clicking, typing, and scrolling to complete tasks on web/desktop/mobile) represent one of the most practical directions in the trend toward MLLM agentization. A universal GUI agent requires large-scale, diverse, and precisely annotated interaction trajectory data to record the complete sequence of "interface state + user action + task intent."

**Limitations of Prior Work**: (i) Manually annotated datasets (MIND2WEB, AITW, AndroidControl) are limited in scale (thousands to tens of thousands), covering only hundreds of applications, making generalization to new interfaces/tasks difficult; (ii) Simulated environments (MiniWoB++) can be scaled up but lack semantic richness and differ significantly from real UIs; (iii) Existing web video-based approaches (TongUI, VideoAgentTrek) rely on foreground/background detection or inverse dynamics, learning only short-horizon low-level visual cues, failing to capture task intent behind actions, and suffer from coordinate errors due to frame compression.

**Key Challenge**: The tension between "data scale ∝ annotation cost" and "data quality requiring task-level understanding + pixel-level grounding." Internet videos are a natural goldmine, but the vast majority are unrelated to GUIs; even for GUI videos, converting them into "coordinate-annotated trajectories" requires overcoming three hurdles: task segmentation, action recognition, and pixel-level localization.

**Goal**: (i) At the scale of 500 million videos, filter out truly high-quality GUI tutorials at a controllable cost; (ii) Automatically parse videos into task-level instructions, step-level actions, and high-resolution coordinates; (iii) Use the synthesized data for pretraining to achieve consistent gains on multi-platform GUI benchmarks.

**Key Insight**: Two critical observations—first, video metadata (title, description, keywords) can filter out 95%+ noise at almost zero cost, reducing 500M to 20M, followed by dimension-wise fine filtering using omnimodal models; second, only a few moments in a video actually change, so decoupling trajectory extraction (using strong VLMs for long-range reasoning on compressed frames) from spatial grounding (using three high-res original frames for pixel localization) enables both "long-range understanding" and "pixel-level accuracy."

**Core Idea**: A three-layer architecture of "coarse-to-fine video filtering + decoupling high/low-level instructions + decoupling task reasoning and spatial grounding" to stream video data into valuable GUI agent training assets.

## Method

Video2GUI is a pipeline consisting of three main stages plus a two-stage training process.

### Overall Architecture
Input: Metadata from 500 million YouTube videos. Output: WildGUI dataset, where each sample is $(u, e)$, with $u$ as the high-level task instruction and $e=(u,a_1,o_1,\dots,a_n,o_n)$ as the step-level trajectory with high-resolution screenshots and precise coordinates. The three stages are: (A) Coarse-to-Fine Video Filtering → 4.16M high-quality tutorials (~300k hours); (B) Trajectory Extraction → task-level instruction + timestamped action descriptions + reasoning; (C) Action Spatial Grounding → precise localization using three high-res frames at $\pm 0.5$s: $b_t=g_\phi(o_{t-0.5s},o_t,o_{t+0.5s},\tau_t)$. Finally, WildGUI is used for continued pretraining on Qwen2.5-VL/Mimo-VL, followed by post-training fine-tuning on open-source datasets.

### Key Designs

1. **Coarse-to-Fine Video Filtering**:

    - **Function**: Filters out the vast majority of noise at the 500M scale in a cost-effective manner, avoiding waste of storage/compute on vlogs and news commentary.
    - **Mechanism**: The first layer, "Meta Info Filtering," uses only titles/descriptions/keywords. DeepSeek-V3 labels 10k samples for supervision, distilled into a Qwen2.5-7B + classification head, which classifies all 500M entries, yielding ~20M candidates. The second layer, "Video Quality Scoring," samples the first minute of each candidate video, and an omnimodal model (Qwen2.5-Omni distilled from Gemini 3 Pro, trained on 200 hours) scores on three dimensions: Topic Relevance (is it teaching GUI operations), Instruction Clarity (is the explanation clear), and Screen Recording Quality (is the recording clear and stable). Ultimately, 4.16M videos (~300k hours) are retained.
    - **Design Motivation**: Direct content analysis of 500M videos is petabyte-scale and infeasible; metadata filtering is nearly costless but only ensures "topic relevance," while "instructional quality" requires video inspection. Two-stage distillation (DeepSeek-V3 → Qwen2.5-7B; Gemini 3 Pro → Qwen2.5-Omni) scales strong model judgment affordably.

2. **Sliding Window + Historical Memory Trajectory Extraction**:

    - **Function**: Uses Gemini-3-Pro to parse videos (up to an hour long) into multiple instruction-trajectory pairs, each action annotated with accurate timestamp, action type, parameters, and reasoning.
    - **Mechanism**: Long videos are segmented into continuous chunks of no more than 4 minutes $\{S_1,\dots,S_M\}$. When processing segment $j$, the model receives current frames plus previous extraction results $D(S_{1:j-1})$ as textual context, enabling cross-segment memory and recognition of tasks spanning segments or dependent on prior actions. Each action is also required to output a "low-level instruction" (textual visual anchor) for subsequent grounding. Each video yields $D(V)=\{(u^{(k)},e^{(k)})\}_{k=1}^N$, supporting multiple independent task instances.
    - **Design Motivation**: Long videos exceed VLM context windows; single-segment processing breaks cross-task dependencies. Prior methods (TongUI uses foreground/background detection, VideoAgentTrek uses inverse dynamics) rely on low-level visual cues, capturing only short-term relations and lacking task intent. Using a strong VLM + historical memory + dual high/low-level instruction output achieves "understanding + explanation" in one step.

3. **Three-Frame High-Resolution Spatial Grounding**:

    - **Function**: Maps the "approximate moment" of an action in the video to a "pixel-level target," correcting coordinate errors from video compression.
    - **Mechanism**: For each action at timestamp $t$, extract the triplet $O_t=\{o_{t-0.5s},o_t,o_{t+0.5s}\}$ from the original video. Gemini-3-Pro, given the low-level instruction $\tau_t$, determines if the action target is localizable in each frame and predicts the bounding box or screen coordinates $b_t=g_\phi(o_{t-0.5s},o_t,o_{t+0.5s},\tau_t)$. The 0.5s offset roughly matches the average duration of a GUI action, ensuring the three frames cover the "pre/at/post" states.
    - **Design Motivation**: The trajectory extraction stage uses compressed frames, insufficient for pixel-level UI control localization (e.g., "click Shoes for men" button); yet grounding is the key supervisory signal for GUI agent training and requires pixel accuracy. Decoupling allows the VLM to perform long-range reasoning first, then use high-res originals for local grounding, saving tokens while maintaining precision.

### Loss & Training

Training is two-stage: (i) Continued Pretraining (CPT): WildGUI is used for large-scale continued pretraining on Qwen2.5-VL/Mimo-VL, enabling the model to absorb interaction patterns across platforms and applications; (ii) Supervised Fine-Tuning (SFT): Task-level supervision is performed on carefully selected open-source GUI datasets (ScreenSpot-Pro, OSWorld-G training set, etc.), fine-tuning for specific downstream benchmarks. The overall goal is to use WildGUI as a "general prior" and open-source data as "task-specific post-polishing."

## Key Experimental Results

### Main Results

Comparison on ScreenSpot-Pro and OSWorld-G GUI grounding benchmarks:

| Model | ScreenSpot-Pro Avg | OSWorld-G Avg |
|------|--------------------|---------------|
| Gemini-2.5-Pro (closed) | 11.4 | 45.2 |
| Seed1.5-VL (closed) | 60.9 | 62.9 |
| Qwen3-VL-2B (open baseline) | 41.9 | 45.9 |
| Qwen3-VL-8B (open baseline) | 49.9 | 54.8 |
| Qwen3-VL-32B | 54.9 | 60.6 |
| GTA1-7B | 50.1 | 55.1 |
| UI-Venus-7B | 50.8 | 58.8 |
| GUI-Owl-7B | 54.9 | 55.9 |
| **Qwen2.5-VL-7B + WildGUI** (Ours) | Significant +Δ | Significant +Δ |

The baseline Qwen2.5-VL-7B achieves only 26.8 on ScreenSpot-Pro, but after continued pretraining with WildGUI, it rises to the level of Qwen3-VL-32B and GUI-Owl-7B, demonstrating that large-scale real video data can elevate a general VLM to GUI-specialist performance.

WildGUI dataset itself is also compared with existing GUI datasets:

| Dataset | Platform Coverage | #Environments | #Trajectories/Instructions | #Screenshots | Avg Steps |
|--------|------------------|--------------|---------------------------|--------------|-----------|
| AITW | mobile | 357 | 30k | 715k | 6.5 |
| AndroidControl | mobile | 833 | 14.5k | 15k | 4.8 |
| GUI-World | three platforms | – | 12k | 83k | 6.7 |
| GUI-Net | three platforms | 280 | 1M | 1M | 4.7 |
| MONDAY | mobile | – | 20k | 313k | 15.7 |
| GUI-360° | desktop | 3 | 13.75k | 105k | 7.6 |
| **WildGUI** | three platforms | **1500+** | **12.7M** | **124.5M** | 9.7 |

WildGUI leads in environment diversity (1500+), scale (12.7M trajectories), and average trajectory length (9.7 steps), while covering web/mobile/desktop platforms.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full pipeline | Optimal | Three-stage pipeline + CPT + SFT |
| w/o Coarse Meta Filter | Training cost explodes | Directly processing 500M videos is infeasible |
| w/o Fine Quality Scoring | Grounding accuracy drops | Many low-quality tutorials included, instructions mismatch visuals |
| w/o Sliding Window Historical Memory | Cross-segment tasks missed | Long videos split lose cross-task context |
| w/o Three-Frame High-Res Grounding | Large coordinate error | Direct localization on compressed frames reduces pixel accuracy |
| w/o CPT (SFT only) | Significantly worse | Validates WildGUI's value as a general prior |

### Key Findings
- Data scale and real-world diversity are the bottlenecks for GUI agent generalization: 13.75k high-quality manual data (GUI-360) is far less effective than 12.7M semi-automatically synthesized data.
- Decoupling "task understanding" and "spatial grounding" is key to quality: the former requires long-range reasoning (strong VLM on compressed frames), the latter requires pixel-level localization (high-res originals); combining both sacrifices one or the other.
- Coarse-to-fine two-stage metadata/content filtering is a feasible path to distill 500M into 4M high-quality data; relying solely on strong models to process all videos is computationally impractical, and distilling to lightweight 7B models for staged processing is the engineering key.
- Cross-platform (web + mobile + desktop) and cross-language (YouTube's inherent multilingualism) coverage significantly improves generalization to unseen interfaces/tasks compared to single-platform datasets.

## Highlights & Insights
- The work delivers an end-to-end reproducible pipeline for "internet video → agent training data," with clear strong-to-lightweight model distillation at each step, serving as an exemplary methodology for data synthesis. The same "coarse-to-fine + strong model distillation" strategy can be transferred to domains like robot demonstration videos, autonomous driving dashcam data, and converting instructional videos to structured courseware.
- The decoupled design of "task understanding" and "pixel grounding" reveals a general principle: when token budget/resolution is limited by context window, decomposing the target into "low-res long context" + "high-res short context" often outperforms simply increasing tokens. This is instructive for long video understanding, long document parsing, and movie script generation.
- Splitting instructions into "high-level user intent" + "low-level visual anchoring" enables agents to follow both natural language and precise localization, establishing a "dual-track" paradigm for GUI agent training data, which is far richer than simply annotating click coordinates.

## Limitations & Future Work
- The pipeline heavily relies on closed-source Gemini-3-Pro and DeepSeek-V3 teachers, making reproduction costly and capping the judgment quality of distilled lightweight models.
- The "three-frame ±0.5s" heuristic for spatial grounding assumes an average action duration of 0.5s; for longer actions like long-press, drag, or double-click, localization may be inaccurate, and the paper does not discuss these in detail.
- All training data is generated automatically, lacking "executor closed-loop verification"—there is no feedback on whether predicted actions can actually be executed in real environments, so trajectories may look correct but fail in practice.
- Only grounding benchmark results are reported in detail; data on improvements in online agent benchmarks (e.g., OSWorld full tasks, WebArena) is limited, and more evaluation is needed for end-to-end agent execution capability.
- Issues of YouTube video copyright, bias, and data contamination (including personal information) require more rigorous handling in large-scale synthetic data.

## Related Work & Insights
- **vs TongUI / VideoAgentTrek**: These rely on foreground/background detection or inverse dynamics to extract trajectories from short videos, learning only short-range low-level visual cues; Video2GUI uses strong VLM + sliding window for long-range task-level understanding and explicitly outputs reasoning, achieving higher data quality and task coverage.
- **vs AITW / AndroidControl / MIND2WEB**: Manually annotated datasets are 10–30k in scale and single-platform, limiting generalization; WildGUI, with 12.7M trajectories and 1500+ applications across three platforms, achieves true universality.
- **vs GUI-Net / GUI-World**: Also automated, but synthesize data using LLMs on HTML/screenshots, lacking the action sequence and context inherent in real videos; Video2GUI directly mines real usage scenarios, yielding distributions closer to real-world practice.
- **Insights**: (i) Distilling large models into small models for large-scale filtering is a key engineering pattern for data synthesis; (ii) The decoupled framework of "long-context understanding + local high-precision processing" can be generalized to any multimodal task requiring both semantics and precision; (iii) Dual-track instructions ("high-level + low-level") greatly enhance downstream agent training.

## Rating
- Novelty: ⭐⭐⭐⭐ The data synthesis approach is an engineering innovation (no major new algorithms), but the overall design of coarse-to-fine multi-stage distillation + trajectory/grounding decoupling is the first systematic presentation in GUI agent data generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared 10+ models on ScreenSpot-Pro and OSWorld-G grounding benchmarks, with detailed dataset comparisons; agent benchmark coverage could be further expanded.
- Writing Quality: ⭐⭐⭐⭐ The pipeline diagram clearly explains the three stages, and Table 1 makes dataset comparison intuitive; motivation and pain points are well articulated, though some details (ablation, quantitative gains) are in the appendix.
- Value: ⭐⭐⭐⭐⭐ If the 12.7M real multi-platform trajectories are open-sourced as promised, it will be a rare large-scale public dataset for the GUI agent community, potentially driving open-source agent ecosystems even more than the method itself.

## Related Papers

- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](../../ICLR2026/llm_agent/m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)
- [\[CVPR 2025\] SpiritSight Agent: Advanced GUI Agent with One Look](../../CVPR2025/llm_agent/spiritsight_agent_advanced_gui_agent_with_one_look.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](../../CVPR2026/llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](../../CVPR2026/llm_agent/echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)

</div>

<!-- RELATED:END -->
