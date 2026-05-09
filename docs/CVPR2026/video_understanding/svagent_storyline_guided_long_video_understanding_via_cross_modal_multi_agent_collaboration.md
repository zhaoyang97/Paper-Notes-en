---
title: >-
  [Paper Note] SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration
description: >-
  [CVPR 2026][Video Understanding][long video QA] This paper proposes SVAgent, a storyline-guided cross-modal multi-agent framework for long video question answering. By progressively constructing narrative representations, employing DPP-based evidence selection, cross-modal consistency verification, and iterative refinement, SVAgent achieves performance gains of 5.5%–11.5% over baselines.
tags:
  - CVPR 2026
  - Video Understanding
  - long video QA
  - multi-agent
  - storyline
  - cross-modal reasoning
  - DPP
date: 2026-05-08
content_hash: 7f9bff50ee4613e8
---

# SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration

**Conference**: CVPR 2026
**arXiv**: [2604.05079](https://arxiv.org/abs/2604.05079)
**Code**: None
**Area**: Video Understanding
**Keywords**: long video QA, multi-agent, storyline, cross-modal reasoning, DPP

## TL;DR

This paper proposes SVAgent, a storyline-guided cross-modal multi-agent framework for long video question answering. By progressively constructing narrative representations, employing DPP-based evidence selection, cross-modal consistency verification, and iterative refinement, SVAgent achieves performance gains of 5.5%–11.5% over baselines.

## Background & Motivation

Video question answering (VideoQA) requires integrating spatial, temporal, and semantic information. Existing methods suffer from three limitations: (1) lack of explicit mechanisms to preserve global temporal structure; (2) absence of reliability guarantees in evidence retrieval; and (3) lack of explicit verification, making models prone to errors caused by insufficient or inconsistent evidence.

Humans naturally comprehend videos through coherent storylines rather than by locating relevant frames in isolation. SVAgent simulates this cognitive process by constructing a global temporal scaffold, upon which hypothesis-driven reasoning and cross-modal verification are performed.

## Method

### Overall Architecture

Six interacting agents form a closed-loop system: Storyline Agent → Hypothesis Agent + DPP evidence selection → Text/Visual Decision Agents → Meta-Decision Agent → Suggestion Agent (iterative refinement).

### Key Designs

1. **Storyline Agent**: Progressively constructs a narrative representation of the video conditioned on the query, compressing the video into a compact temporal abstraction that retains temporally and semantically relevant cues for reasoning. Supports incremental updates, revising incomplete narrative segments upon incorporating new frames.

2. **Hypothesis-Driven Reasoning + DPP Evidence Selection**: The Hypothesis Agent proposes answer hypotheses and identifies supporting or contradicting evidence. Determinantal Point Processes (DPP) are applied to select two frame sets $\mathcal{Y}_q$ and $\mathcal{Y}_e$ based on the query $\mathcal{Q}$ and evidence $\mathcal{E}$ respectively; the DPP kernel matrix ensures that selected frames are both diverse and relevant. A coarse verification step uses the intersection ratio $|\mathcal{Y}_q \cap \mathcal{Y}_e| / (|\mathcal{Y}_q| + |\mathcal{Y}_e|) > \alpha$: if the ratio exceeds threshold $\alpha$, the process proceeds to fine-grained cross-modal verification; otherwise, the Suggestion Agent proposes new frames for the next iteration.

3. **Cross-Modal Decision Verification**: The Text and Visual Decision Agents independently produce answers. The Meta-Decision Agent checks for consistency — if consistent, the answer is confirmed; if inconsistent, the Suggestion Agent proposes new frames for the next round of refinement.

### Loss & Training

No additional training is required. SVAgent operates as a zero-shot multi-agent collaboration built upon open-source Video MLLMs (e.g., Qwen2.5-VL). The maximum number of iterations $T$ controls computational overhead.

## Key Experimental Results

### Main Results

| Baseline → +SVAgent | LongVideoBench | MLVU | LVBench | VideoMME |
|---------------------|---------------|------|---------|---------|
| Qwen2.5-VL 3B → +SVAgent | 53.0→**59.7** | 53.6→**61.2** | 31.6→**38.5** | 52.8→**60.7** |
| Gain | +6.7 | +7.6 | +6.9 | +7.9 |

### Key Findings

- Consistent improvements of 5.5%–11.5% across four long video benchmarks.
- A small model (3B) augmented with SVAgent can approach or even surpass a large model (72B) under single-pass inference.
- Cross-modal consistency verification effectively identifies reasoning uncertainty.
- Storyline construction is critical for maintaining temporal coherence.
- The Storyline Agent supports incremental updates, revising incomplete narrative segments upon incorporating new frames.
- The Suggestion Agent leverages historical failure records to propose new frames for targeted exploration rather than random sampling.
- The Text and Visual Decision Agents independently produce answers; the Meta-Decision Agent checks consistency — confirming when consistent, and triggering refinement when not.

## Highlights & Insights

- Simulates human cognitive processes in video comprehension: construct storyline → form hypotheses → retrieve evidence → cross-validate → review as needed.
- DPP-based evidence selection simultaneously ensures diversity and relevance.
- The closed-loop iterative refinement mechanism adaptively determines when to terminate.

## Limitations & Future Work

- Multi-agent invocations increase inference latency and API costs.
- Storyline quality depends on the accuracy of frame captioning.
- May introduce unnecessary complexity for simple questions.
- The iteration limit $T$ requires balancing computational cost against answer quality.
- The framework operates as zero-shot multi-agent collaboration on open-source Video MLLMs (e.g., Qwen2.5-VL) without additional training.
- Extension to non-multiple-choice formats (e.g., open-ended QA) remains unexplored.
- The comparison between Qwen2.5-VL 72B single-pass inference and 3B+SVAgent demonstrates the value of agent collaboration.
- Frame captioning quality serves as an upstream bottleneck for storyline construction; stronger captioning models could be explored to further improve performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Storyline-driven multi-agent video reasoning
- **Technical Depth**: ⭐⭐⭐⭐ — Six-agent closed-loop design with strong systematicity
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Consistent validation across four benchmarks and model scales from 3B to 72B
- **Practical Value**: ⭐⭐⭐ — High inference overhead; best suited for high-quality application scenarios

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[CVPR 2026\] A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a4vl_multiagent_long_video_reasoning.md)
- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)

<!-- RELATED:END -->
