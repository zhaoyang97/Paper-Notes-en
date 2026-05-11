---
title: >-
  [Paper Note] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
description: >-
  [NeurIPS 2025][LLM Agent][video understanding] This paper proposes DVD (Deep Video Discovery), an agent that frames long-form video understanding as a multi-step information search problem. It first constructs a multi-gr…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "video understanding"
  - "agentic search"
  - "tool use"
  - "long-form video"
  - "multi-granular database"
  - "adaptive workflow"
date: 2026-05-08
content_hash: 81b3838145fe21c3
---

# Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2505.18079](https://arxiv.org/abs/2505.18079)
**Code**: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)
**Area**: LLM Agent / Video Understanding
**Keywords**: video understanding, agentic search, tool use, long-form video, multi-granular database, adaptive workflow

## TL;DR
This paper proposes DVD (Deep Video Discovery), an agent that frames long-form video understanding as a multi-step information search problem. It first constructs a multi-granular structured database from a long video (global summary + clip-level caption embeddings + frame-level pixels), then provides three search tools (Global Browse / Clip Search / Frame Inspect). A reasoning LLM autonomously orchestrates the search trajectory via an observe-reason-act loop. DVD achieves 74.2% on LVBench (surpassing the previous SOTA MR.Video by 13.4 pp), and 76.0% with subtitles.

## Background & Motivation
- Long videos (hour-scale) are extremely information-dense; even LLMs with million-token context windows cannot process them directly, and instruction-following and reasoning capabilities degrade as context grows.
- Prior video agents (VideoTree, VCA) adopt **fixed workflows**—e.g., tree search from root to leaf, or fixed predict-reflect-search-merge loops—and cannot adaptively select strategies for different queries.
- Core insight: Inspired by Deep Research/Deep Search, long-form video understanding is reframed as a **multi-step information search problem**, where the video is the environment to be explored, clips are information units, and the agent autonomously plans its search path.

## Method

### Stage 1: Multi-Granular Video Database Construction (Offline)

**Temporal Segmentation**: Long video $V$ is uniformly segmented into $N = \lceil\text{len}(V)/t\rceil$ non-overlapping clips at $t=5$ seconds each; each clip is decoded at 2 fps into a frame sequence.

**Three-Level Information Extraction**:
1. **Frame Level**: Raw decoded frames $f_i$ are stored by clip index for subsequent pixel-level detail analysis.
2. **Clip Level**: A VLM (GPT-4.1) generates a textual description $c_i$ for each clip, which is then encoded into a vector $e_i$ by a language embedding model for semantic retrieval.
3. **Global Level**: During the per-clip description generation process, a **progressive entity registry $S$** is maintained—whenever a new entity (person, object, etc.) appears, its name, appearance, identity, actions, and temporal span are recorded. The final $S_N$ constitutes the global entity index.

**Final Database**: $\mathcal{D} = \{S, \{f_i, c_i, e_i\}_{i=1}^{N}\}$—a structured representation supporting both text-based retrieval and pixel-level traceback.

### Stage 2: Agentic Search and Answer (ASA)

**Three Search Tools**:

| Tool | Granularity | Input | Output | Purpose |
|------|-------------|-------|--------|---------|
| Global Browse | Global | Database $D$ + user query $Q$ | Entity summary + event summary | Obtains global context; entity summary from the pre-built registry; event summary generated via uniformly sampled frames + VLM |
| Clip Search | Clip-level | Database $D$ + agent-synthesized query $\hat{Q}$ + top-$k$ | Top-$k$ relevant clips and descriptions | Retrieves relevant segments via cosine similarity in embedding space; agent may call iteratively to progressively refine queries |
| Frame Inspect | Frame-level | Database $D$ + agent-synthesized query $\hat{Q}$ + time range $[t_s, t_e]$ | VQA answer | Loads raw frames (up to 50) + VLM for open-ended VQA to obtain fine-grained visual information |

**Agent Design — Observe-Reason-Act Loop (ReAct-style)**:
- Action space $\mathcal{A}$ = {Global Browse, Clip Search, Frame Inspect, Answer}
- At each step: LLM reasons over history $H_i$ → selects action $A_i$ and parameters $P_i$ → obtains observation $O_i$ → updates history
- Termination: selects the Answer action or reaches maximum steps $N=15$
- **Key design decision**: No manual specification of tool usage patterns or search strategies; the LLM's reasoning capability fully governs orchestration.

### Implementation Details
- Database construction VLM: GPT-4.1 for LVBench; GPT-4.1-mini for other benchmarks (cost reduction)
- Reasoning model $M_{\text{reasoning}}$: OpenAI o3 (also used for VQA in Frame Inspect)
- Clip Search default top-$k=16$; LLM may adjust this
- Frames uniformly resized to 720p
- Subtitle-augmented variant: WhisperX used for ASR; transcripts guide segmentation and enrich descriptions

## Key Experimental Results

### LVBench (1,549 questions / 103 hour-scale videos)

| Method | Accuracy |
|--------|----------|
| GPT-4o | 48.9% |
| OpenAI o3 (256-frame direct input) | 57.1% |
| MR.Video (prev. SOTA agent) | 60.8% |
| **DVD (Ours)** | **74.2%** |
| DVD + subtitles | **76.0%** |

- Surpasses MR.Video by 13.4 pp, VCA by 32.9 pp, and base VLM o3 by 17.1 pp.

### Other Benchmarks
- LongVideoBench Long subset: 68.6% (surpasses prev. SOTA by 7.0 pp)
- Video MME Long: 67.3% (surpasses AdaRETAKE by 2.3 pp)
- EgoSchema: 76.6% (surpasses human-level performance ~76%)

### Ablation Study

**Impact of Model Selection (Table 4)**:
- Reasoning model is most critical: o3 → o4-mini drops 5.8 pp; o3 → GPT-4o drops 13.7 pp
- Database construction VLM: GPT-4.1 → 4.1-mini drops only 4.1 pp
- Frame Inspect VLM: o3 → 4.1-mini drops 3.7 pp

**Tool Ablation (Table 5)**:
- Removing Clip Search is most detrimental (−12.3 pp)—core retrieval capability
- Removing Frame Inspect drops 8.4 pp—fine-grained visual understanding dependency
- Removing Global Browse drops 2.9 pp—global context assistance

**Open-Source Model Compatibility (Table 6)**:
- DeepSeek-R1 as reasoning model: 68.5% (still surpasses all prior methods)
- Qwen3-32B: 57.3% (a 32B model already outperforms GPT-4o and direct o3 input)

**Adaptive vs. Fixed Workflow (Table 7)**:
- Fixed VideoAgent workflow averages 11.1 steps and achieves only 70.2%; DVD's adaptive approach averages 7.3 steps and achieves 74.2%—fewer steps, better performance.

## Agent Behavior Pattern Analysis (A Core Contribution)

The paper categorizes agent tool-calling behavior into five patterns and analyzes them:
1. **Global Browse Only**: A single global browse suffices for an answer; rare but yields very high accuracy.
2. **Simple Action**: Direct search → inspect → answer; most common (>50%), with high accuracy.
3. **Iterative Search**: Multiple alternating rounds of Clip Search and Frame Inspect; longer trajectories (~8 steps), slightly lower accuracy.
4. **Frame Inspect Trap**: 3+ consecutive Frame Inspect calls trapped in detail loops; accuracy drops significantly.
5. **Clip Search Trap**: 3+ consecutive Clip Search calls repeatedly retrieving similar information; the primary failure mode for o3.

**Two Key Findings**:
- **Duality of reasoning length**: Within the same model, longer reasoning trajectories tend to reflect greater uncertainty and lower accuracy; yet across models, those capable of deeper reasoning perform better.
- **Overconfidence leads to behavioral collapse**: GPT-4o applies the Simple Action pattern to 91.4% of queries (averaging only 4.6 steps), draws premature conclusions, and rarely explores alternative strategies—the root cause of its poor performance.

## Highlights & Insights
- **Autonomous search paradigm**: No predefined workflow; the LLM decides the search trajectory—closer to how humans analyze video.
- LVBench 74.2% represents a decisive lead (surpassing prev. SOTA by 13.4 pp).
- The multi-granular database design is elegant: text-retrievable and pixel-traceable, balancing efficiency and precision.
- Agent behavior pattern analysis provides practical insights for video agent design.
- Open-source model DeepSeek-R1 also achieves 68.5%, demonstrating the framework's generality.

## Limitations & Future Work
- Iterative reasoning introduces substantial computational overhead (multiple rounds of LLM + VLM calls).
- Heavy dependence on reasoning model capability—weaker models (e.g., GPT-4o) exhibit behavioral collapse.
- The tool set is fixed at three types, limiting extensibility (e.g., no dedicated OCR/ASR tools).
- Azure content filtering incorrectly flags some benchmark data, causing minor performance loss.

## Related Work & Insights
- **vs. VideoTree / VCA**: Fixed tree search strategy → DVD adaptive search; surpasses VCA by 32.9 pp on LVBench.
- **vs. MR.Video**: Previous best agent at 60.8% → DVD at 74.2%; the key difference is that DVD does not prescribe tool ordering.
- **vs. AdaRETAKE**: Visual token compression at 53.3% → DVD at 74.2%; agentic search substantially outperforms compression-based strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ Transfers the Deep Search paradigm to video understanding; agent autonomously orchestrates search workflows.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks + detailed ablations + behavior pattern analysis.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; behavior analysis is insightful.
- Value: ⭐⭐⭐⭐⭐ A new paradigm for long-form video understanding, with decisive performance leads and open-source reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](../../CVPR2026/llm_agent/haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](../../CVPR2026/llm_agent/haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[NeurIPS 2025\] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Understanding](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)
- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)

</div>

<!-- RELATED:END -->
