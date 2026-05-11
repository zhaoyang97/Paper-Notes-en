---
title: >-
  [Paper Note] LensWalk: Agentic Video Understanding by Planning How You See in Videos
description: >-
  [CVPR 2026][Video Understanding][Video Agent] This paper presents LensWalk, an agentic framework that enables an LLM reasoner to actively control the temporal scope and sampling density of video observations. Through a r…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Agent"
  - "Active Observation"
  - "Vision-Language Models"
  - "Long Video Understanding"
  - "Tool Calling"
date: 2026-05-08
content_hash: 1d64aab8fa48e023
---

# LensWalk: Agentic Video Understanding by Planning How You See in Videos

**Conference**: CVPR 2026
**arXiv**: [2603.24558](https://arxiv.org/abs/2603.24558)
**Code**: N/A
**Area**: Video Understanding
**Keywords**: Video Agent, Active Observation, Vision-Language Models, Long Video Understanding, Tool Calling

## TL;DR

This paper presents LensWalk, an agentic framework that enables an LLM reasoner to actively control the temporal scope and sampling density of video observations. Through a reason-plan-observe loop, LensWalk achieves adaptive video understanding without any fine-tuning, yielding plug-and-play performance gains exceeding 5% on long video benchmarks.

## Background & Motivation

Video understanding is a core task in computer vision, yet the densely temporal nature of video poses substantial challenges for automated analysis. Existing approaches suffer from a fundamental disconnect between reasoning and perception.

Three categories of limitations are prevalent: (1) single-pass forward methods uniformly sample videos into a fixed visual context, risking missed critical events or saturation by redundant information; (2) heuristic keyframe selection methods offer finer granularity but remain one-shot static sampling unable to adapt as intermediate hypotheses evolve; (3) retrieval-based agents can dynamically acquire information but operate over pre-processed static representations (e.g., ASR transcripts, clip-level captions), precluding on-demand generation of new observations from source video.

**Key Challenge**: A model's reasoning process should drive *what* and *how* it observes, yet existing pipelines decouple observation from reasoning—observations are completed before reasoning begins or are constrained to fixed preprocessing artifacts. **Key Insight**: This paper draws on human visual cognition strategies, wherein humans cope with information overload through purposeful information seeking, continuously alternating between macro scanning and fine-grained focusing while reflecting and verifying throughout. **Core Idea**: Allow the LLM reasoner to autonomously determine the temporal scope and sampling density of observations, transforming video understanding into an active reason-plan-observe loop.

## Method

### Overall Architecture

LensWalk models video understanding as a multi-round iterative process. In each round, the LLM reasoner ($M_r$) analyzes the current question and accumulated evidence, then formulates an action plan ($a_t$) specifying the observation tool, guiding sub-question, temporal range, and sampling density. This plan is executed by a VLM observer ($M_o$) to extract visual evidence from the video. The evidence is appended to the history, forming the input for the next round of reasoning. Additionally, the system maintains timestamp anchors and a global entity memory table to ensure cross-round consistency.

### Key Designs

1. **Multi-Granularity Observation Tool Suite**:
    - **Function**: Provides three complementary observation tools supporting video browsing at different granularities.
    - **Mechanism**: *Scan Search* discovers cues by scanning slices in parallel over a broad temporal range; *Segment Focus* performs high-density sampling within a single temporal segment to extract fine-grained details; *Stitched Verify* combines frames from multiple non-contiguous temporal segments into a single batch, enabling cross-segment comparison and causal verification.
    - **Design Motivation**: The three tools constitute a complete cognitive chain of "discover–focus–verify," covering the full spectrum from global cue search to local detail extraction to cross-segment integration.

2. **Reasoning-Scheduled Active Observation Mechanism**:
    - **Function**: Enables the reasoner to explicitly control the temporal range ($\mathcal{I}_t$) and sampling strategy at each step.
    - **Mechanism**: Each action $a_t = (o_t, q_t, \mathcal{I}_t, \rho_{o_t})$ encodes the tool selection, guiding question, spatiotemporal range, and tool-specific parameters, realizing an end-to-end mapping from reasoning state to observation plan.
    - **Design Motivation**: Embedding parameterized observation plans into the history allows the agent to track its own exploration progress and provides a basis for subsequent steps to target insufficiently explored regions.

3. **Evidence Anchoring and Entity Memory**:
    - **Function**: Ensures temporal localization accuracy and entity consistency throughout long multi-round reasoning.
    - **Mechanism**: Timestamp anchors insert inter-frame temporal markers during VLM observation, enabling the observer to return answers with precise temporal references; a global entity memory table is maintained independently of the reasoning history, recording entity attributes and appearance times.
    - **Design Motivation**: Avoids the overhead of redundantly recognizing the same entities across rounds, prevents entity reference confusion in long historical contexts, and provides temporal anchors for precise re-observation in subsequent steps.

### Loss & Training

- LensWalk is a training-free, plug-and-play framework requiring no model fine-tuning.
- The agent is limited to a maximum of 20 tool calls, one per round.
- Per-call frame budgets for Scan Search, Segment Focus, and Stitched Verify are 180, 32, and 128, respectively.
- The reasoner simultaneously serves as the updater of the entity memory table.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (Best Config) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| LVBench | Accuracy | 68.6% (o3 self) | 60.8% (MR.Video) | +7.8% |
| VideoMME Long | Accuracy (w/o sub) | 71.4% (o3 self) | 67.3% (DVD) | +4.1% |
| LongVideoBench | Accuracy | 70.6% (o3 self) | 68.6% (DVD) | +2.0% |
| MMVU (MC) | Accuracy | 80.9% (o3/GPT-4.1) | 78.9% (o3) | +2.0% |
| Video-MMMU | Overall | 78.33% (o3 self) | 75.44% (o3) | +2.89% |
| EgoSchema | Val | 77.2% (o3/Qwen2.5-VL-72B) | 76.6% (DVD) | +0.6% |

### Ablation Study

| Configuration | Key Metric (VideoMME Long) | Note |
|------|---------|------|
| Full LensWalk (o3/GPT-4.1) | 70.0% | Baseline |
| w/o Scan Search | 65.4% | −4.6%; cue localization is most critical |
| w/o Stitched Verify | 66.8% | −3.2%; cross-segment integration is important |
| w/o Segment Focus | 68.1% | −1.9%; fine-grained extraction contributes |
| w/o Timestamp Anchor | 69.4% | −0.6% |
| w/o Subject Memory | 69.7% | −0.3% |

### Key Findings

- When o3 serves as its own observer (reasoner and observer share the same model), performance is exceptional—+11.5% on LVBench and +6.7% on VideoMME Long—representing a "free lunch."
- The open-source reasoner Qwen3-235B-A22B is effective with weak observers (e.g., +4.3% for Qwen2.5-VL-7B) but offers limited benefit with strong observers (e.g., only +0.1% for GPT-4.1).
- The agent exhibits six behavioral patterns: direct querying, progressive zooming, range splitting, strategic reflection, integrative verification, and static repetition.
- The framework adaptively allocates the observation budget: simple questions are resolved quickly with few frames, while complex questions involve more observation rounds.

## Highlights & Insights

- The core design philosophy of incorporating "how to observe" into the reasoning loop is highly elegant, analogous to purposeful human visual search strategies.
- The training-free, plug-and-play nature allows the framework to directly enhance existing models, offering considerable engineering value.
- The emergent diversity of cognitive strategies (progressive zooming, strategic reflection, etc.) demonstrates the agent's autonomous reasoning capability.
- Token consumption is comparable to single-pass forward methods while substantially reducing peak token count per round, alleviating memory pressure from long contexts.

## Limitations & Future Work

- Framework effectiveness is highly dependent on the reasoner's cognitive capability—weaker reasoners may generate invalid observation plans.
- A small proportion of "static repetition" behavior (repeated observation of the same region) persists, indicating that the planning mechanism remains imperfect.
- Current observation tools are limited to the visual modality and do not exploit multimodal information such as audio or subtitles.
- The maximum of 20 tool calls may be insufficient in extreme long-video scenarios.

## Related Work & Insights

- **vs. Deep Video Discovery (DVD)**: DVD pre-generates captions for an entire video to support reasoning, consuming millions of tokens; LensWalk observes on demand, with token consumption approximating single-pass forward methods while achieving higher accuracy.
- **vs. MR.Video**: MR.Video relies on pre-processed clip retrieval, with fixed observation granularity and scope; LensWalk dynamically adjusts the temporal range and sampling density of observations.
- **vs. VideoAgent**: VideoAgent's tools operate solely on preprocessing artifacts; LensWalk directly schedules new observations from source video.
- **Insights**: The notion of "scalable visual cognition"—not merely scaling model size but enabling models to learn active observation—offers a compelling research direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reframes video understanding as an active observation scheduling problem; conceptually innovative and elegantly realized.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, multiple model combinations, detailed ablations, and behavioral analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative, clear exposition of principles, and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play framework that directly enhances existing strong models; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)

</div>

<!-- RELATED:END -->
