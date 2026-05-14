---
title: >-
  [Paper Note] TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents
description: >-
  [AAAI 2026][LLM Agent][GUI Agent] TongUI proposes a framework that automatically converts multimodal web tutorials (videos and illustrated articles) into GUI operation trajectories…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Trajectory Data"
  - "Multimodal Tutorials"
  - "VLM Fine-tuning"
  - "Cross-platform Generalization"
date: 2026-05-08
content_hash: 9044469f2daad3ab
---

# TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents

**Conference**: AAAI 2026
**arXiv**: [2504.12679](https://arxiv.org/abs/2504.12679)
**Code**: [https://github.com/TongUI-agent/TongUI-agent](https://github.com/TongUI-agent/TongUI-agent)
**Area**: GUI Agent / Multimodal VLM
**Keywords**: GUI Agent, Trajectory Data, Multimodal Tutorials, VLM Fine-tuning, Cross-platform Generalization

## TL;DR
TongUI proposes a framework that automatically converts multimodal web tutorials (videos and illustrated articles) into GUI operation trajectories, constructing the million-scale GUI-Net-1M dataset for fine-tuning Qwen2.5-VL. The resulting models surpass or approach state-of-the-art methods such as UI-TARS across multiple grounding and navigation benchmarks.

## Background & Motivation
**Background**: GUI agents are a trending research direction that leverages LLMs/VLMs to simulate human interaction with computers and mobile devices, executing operations such as clicking, form filling, and scrolling to complete tasks.

**Limitations of Prior Work**: Training high-quality GUI agents requires large-scale operation trajectory data. Existing data acquisition approaches either rely on costly human annotation (high quality but limited scale) or large-model synthesis (insufficient quality and diversity).

**Key Challenge**: An ideal GUI agent must generalize across diverse operating systems and applications, yet existing datasets suffer from bottlenecks in scale, platform coverage, and application diversity. Even the largest open-source dataset, AITW, contains only 715K trajectories and covers Android exclusively.

**Goal**: To obtain large-scale, high-quality, cross-platform GUI operation trajectory data at low cost.

**Key Insight**: The authors observe that the internet hosts an abundance of multimodal GUI tutorials (YouTube videos, WikiHow illustrated articles) that naturally contain step-by-step GUI operation guidance, requiring only a well-designed conversion pipeline to transform them into training data.

**Core Idea**: Automatically convert multimodal web tutorials into GUI operation trajectories to construct a million-scale, cross-platform training dataset.

## Method

### Overall Architecture
The TongUI framework consists of four stages: tutorial crawling → tutorial processing → trajectory generation → data filtering. The input is multimodal web tutorials (video or illustrated text), and the output is formatted trajectory data $(q, \{o_i, r_i, a_i\}_{i=1}^T)$, where $q$ is the task query, $o_i$ is the screenshot observation, $r_i$ is the reasoning thought, and $a_i$ is the executed action. Sequential decision-making is performed under the ReAct framework.

### Key Designs

1. **Tutorial Crawling**:

    - Function: Crawl multimodal GUI tutorials from platforms including YouTube, Bilibili, WikiHow, and Baidu Jingyan.
    - Mechanism: Seed task keywords (e.g., "Chrome change font size") are first generated via brainstorming, then expanded by an LLM, and used for retrieval in the format "application name + task content." Video platforms are queried via API for download; illustrated-text platforms are crawled by category tags.
    - Design Motivation: Covering four sources ensures diversity; YouTube/Bilibili supply video tutorials while WikiHow/Baidu Jingyan supply illustrated-text tutorials.

2. **Tutorial Processing**:

    - Function: Uniformly convert heterogeneous multimodal tutorials into a structured format of "text description + screenshot."
    - Mechanism: For text, video audio is transcribed using Whisper ASR while illustrated-text pages are parsed directly from HTML structure; an LLM then extracts the task $q$ and step-wise descriptions $\{h_1, \ldots, h_T\}$. For visuals, illustrated-text tutorials use embedded images directly, while video tutorials apply the MOG2 background subtraction algorithm to detect significantly changed frames as keyframes, segmented by audio timestamps.
    - Design Motivation: Data formats vary greatly across sources, necessitating a unified processing pipeline. GPT-4o-mini is used to filter out non-screenshot images (e.g., cartoons, natural scene photos).

3. **Trajectory Generation**:

    - Function: Generate standardized reasoning $r_i$ and action $a_i$ for each step.
    - Mechanism: A pretrained zero-shot GUI agent (e.g., UI-TARS) performs inference on each step's screenshot $o_i$ and description $h_i$ to produce formatted thoughts and actions. Notably, the step description $h_i$ rather than the overall task query $q$ is used as the input query. If generation fails for a particular step, the original trajectory is split into two independent sub-trajectories.
    - Design Motivation: Using $h_i$ instead of $q$ allows the agent to more accurately identify what the current step requires; the splitting strategy prevents error propagation.

4. **Multi-stage Data Filtering**:

    - Function: Incrementally clean the data, retaining 33% of trajectories as high-quality samples from the raw collection.
    - Mechanism: Three-stage filtering — (1) deduplication based on URL/video ID; (2) LLM-based judgment of whether the tutorial is relevant to a GUI task; (3) trajectory-level filtering that discards steps where the agent predicts wait/call_user, followed by Qwen2.5-VL-7B-based trajectory quality evaluation.

### Loss & Training
Two-stage training: the base VLM is first fine-tuned on GUI-Net-1M, then further supervised fine-tuned (SFT) on the training splits of evaluation benchmarks. LoRA is applied (rank=16, alpha=32, covering only 0.5% of parameters), with a context window of 8192 tokens, retaining at most the first 2 prior observations (3 images total), and limiting each image to 1350 visual tokens.

## Key Experimental Results

### Main Results — Grounding (ScreenSpot)

| Method | ScreenSpot Avg | ScreenSpot-V2 Avg | ScreenSpot-Pro Avg |
|--------|---------------|-------------------|-------------------|
| ShowUI-2B | 75.1 | - | 7.7 |
| UI-TARS-7B | 89.5 | 91.6 | 35.7 |
| Qwen2.5-VL-7B (baseline) | 78.6 | 84.0 | 12.5 |
| **TongUI-3B** | 83.6 | 85.5 | 18.0 |
| **TongUI-7B** | 86.0 | 88.7 | 24.7 |
| **TongUI-32B** | 88.5 | 92.1 | 33.1 |

### Ablation Study — Effect of Data Source Scaling

| Configuration | ScreenSpot Avg | Mind2Web Step SR |
|---------------|---------------|-----------------|
| No SFT | 56.5 | 1.7 |
| Refined data only | 68.0 | 40.7 |
| + WikiHow 50K | 75.8 | 44.4 |
| + Baidu 50K | 78.7 | 45.5 |
| + Video 50K | 79.6 | 46.0 |
| + All data | 83.6 | 49.5 |

### Key Findings
- TongUI-3B alone surpasses ShowUI-2B (75.1→83.6), underscoring the importance of data quality and scale.
- On AndroidControl, TongUI-3B (73.3/91.5) even outperforms UI-TARS-2B (68.9/89.3), demonstrating the generalization advantage conferred by multi-source data.
- The data source scaling ablation clearly shows consistent gains with each additional source category: public data only → +WikiHow → +Baidu Jingyan → +video, with ScreenSpot improving from 68→76→79→80→84.
- On online navigation (MiniWob), TongUI-3B (72.7) surpasses ShowUI-2B (71.5), indicating that offline tutorial data also improves generalization to online scenarios.
- A user study shows that the quality score of filtered data improves from 3.22 to 4.12, approaching the ShowUI dataset score of 4.26.

## Highlights & Insights
- **Data Source Innovation**: The work cleverly leverages existing internet tutorial resources rather than human annotation or model synthesis. This "standing on the shoulders of giants" paradigm is transferable to other agent tasks requiring large-scale labeled data (e.g., code agents, office automation).
- **Multimodal Tutorial Processing Pipeline**: Video (ASR → keyframe extraction) and illustrated-text (HTML parsing) sources are unified into the same trajectory format through a well-engineered processing pipeline.
- **Zero-shot Agent for Trajectory Annotation**: No real execution environment is required; an existing agent performs zero-shot inference on screenshots to obtain action annotations, eliminating simulator dependency.
- **Progressive Data Filtering**: Three-stage filtering retains 33% of the data, and user studies confirm a significant quality improvement after filtering.

## Limitations & Future Work
- Data is collected and used for one-time training with no continual learning capability, a limitation acknowledged by the authors.
- Trajectory generation depends on the zero-shot capability of UI-TARS, whose quality directly determines the data quality ceiling.
- Using step description $h_i$ as the query rather than the full task $q$ may result in generated thoughts that lack global planning capacity.
- A 33% data retention rate means 67% is discarded; the filtering strategy may be overly aggressive or insufficiently fine-grained.
- Only LoRA fine-tuning is employed; full-parameter fine-tuning or multi-round RL training may yield further performance gains.

## Related Work & Insights
- **vs. UI-TARS**: UI-TARS achieves stronger performance but its data is not open-sourced. TongUI is fully open-source (data + code + models), and already approaches or surpasses UI-TARS on certain benchmarks.
- **vs. AgentTrek**: Both acquire data from web tutorials, but AgentTrek contains only 10.4K trajectories and relies on simulator-based exploration, whereas TongUI operates at the million scale without any simulator.
- **vs. ShowUI**: ShowUI uses 137K training samples, yet TongUI with its million-scale data outperforms it on most benchmarks, validating the value of large-scale data.
- Inspiration: The "web tutorial → trajectory data" paradigm can be extended to other agent domains, such as training code agents from StackOverflow tutorials.

## Rating
- Novelty: ⭐⭐⭐⭐ — The data collection idea is creative, though no algorithmic innovation is introduced; the core contribution lies in data engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 offline benchmarks + 1 online benchmark + user study + ablation study.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed pipeline descriptions.
- Value: ⭐⭐⭐⭐⭐ — A fully open-source million-scale GUI dataset provides substantial value to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](history-aware_reasoning_for_gui_agents.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](../../CVPR2026/llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)

</div>

<!-- RELATED:END -->
