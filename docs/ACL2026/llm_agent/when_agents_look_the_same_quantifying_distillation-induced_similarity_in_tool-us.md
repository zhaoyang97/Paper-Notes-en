---
title: >-
  [Paper Note] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors
description: >-
  [ACL 2026][LLM Agent][model distillation] This paper proposes two complementary metrics, RPS and AGS, to quantify distillation-induced behavioral homogenization in LLM agents' tool-use behaviors. By distinguishing necessary from unnecessary behaviors, the framework reveals cross-family behavioral inheritance patterns across 18 models, finding that Kimi-K2 exhibits greater behavioral similarity to Claude Sonnet 4.5 than Anthropic's own models do.
tags:
  - ACL 2026
  - LLM Agent
  - model distillation
  - behavioral homogenization
  - tool use
  - agent evaluation
  - behavioral similarity
date: 2026-05-08
content_hash: f41fb24a94ac2616
---

# When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors

**Conference**: ACL 2026
**arXiv**: [2604.21255](https://arxiv.org/abs/2604.21255)
**Code**: [https://github.com/Syuchin/AgentEcho](https://github.com/Syuchin/AgentEcho)
**Area**: LLM Agent
**Keywords**: model distillation, behavioral homogenization, tool use, agent evaluation, behavioral similarity

## TL;DR
This paper proposes two complementary metrics, RPS and AGS, to quantify distillation-induced behavioral homogenization in LLM agents' tool-use behaviors. By distinguishing necessary from unnecessary behaviors, the framework reveals cross-family behavioral inheritance patterns across 18 models, finding that Kimi-K2 exhibits greater behavioral similarity to Claude Sonnet 4.5 than Anthropic's own models do.

## Background & Motivation

**Background**: The LLM agent landscape is experiencing a Cambrian explosion, with a growing number of high-performance agents emerging from diverse sources. Despite their varied origins, these models exhibit strikingly consistent behaviors in reasoning steps, tool-calling habits, and even failure modes, suggesting that many may be "distillation echoes" of a small number of dominant teacher models.

**Limitations of Prior Work**: Existing similarity metrics focus primarily on response-level similarity in static dialogues and fail to capture the dynamic nature of multi-step tool-use trajectories. More critically, they do not distinguish between *necessary behaviors* (actions required for task success) and *unnecessary behaviors* (actions reflecting a model's autonomous preferences), causing similarity scores to be inflated by the shared correct paths imposed by the task itself.

**Key Challenge**: Without distinguishing necessary from unnecessary behaviors, it is impossible to determine whether two models converge because only one correct path exists, or because one model is blindly imitating the other's habits — a fundamental obstacle to quantifying distillation effects.

**Goal**: To design a systematic framework that isolates unnecessary behavioral patterns and quantifies distillation-induced behavioral homogenization between agents along two dimensions: linguistic expression and tool operation.

**Key Insight**: The authors observe that many agents perform redundant tool calls (e.g., exhaustively trying all available tools even when the answer is obvious). These unnecessary behavioral choices serve as "behavioral fingerprints" for identifying whether a model has been distilled.

**Core Idea**: By decomposing agent trajectories into necessary and unnecessary behaviors, the framework captures behavioral inheritance signals along two dimensions using RPS (Response Pattern Similarity) for linguistic expression and AGS (Action Graph Similarity) for tool-operation patterns.

## Method

### Overall Architecture
Given a set of models and a collection of tool-use tasks, execution trajectories are collected for each model and analyzed for similarity along two orthogonal dimensions: RPS captures *how* models express responses verbally (verbal fingerprint), while AGS captures *how* models select and organize tool calls (behavioral fingerprint). Claude Sonnet 4.5 (thinking) serves as the reference oracle model against which all other models' behavioral similarity is computed.

### Key Designs

1. **Response Pattern Similarity (RPS)**:

    - **Function**: Quantifies the degree of linguistic similarity between two models' responses.
    - **Mechanism**: A two-stage pipeline is employed. First, Stage Annotation semantically aligns trajectories to five canonical stages (authentication, information retrieval, execution, verification, notification), resolving the issue of different models requiring different numbers of turns for the same task. Then, within shared stages, an LLM Judge scores responses on three dimensions — Style, Structure, and Alignment — on a 1–5 scale, and the mean Overall score is computed.
    - **Design Motivation**: Directly comparing full trajectories or aligning turn-by-turn risks matching unrelated content, yielding unreliable scores. Semantic-level stage alignment ensures that only functionally equivalent interaction segments are compared.

2. **Action Graph Similarity (AGS)**:

    - **Function**: Analyzes structured behavioral patterns from tool-call sequences.
    - **Mechanism**: Dialogue trajectories are constructed as directed graphs $G=(V, E_s, E_d)$, where nodes represent tool calls, $E_s$ denotes sequential order edges, and $E_d$ denotes dependency edges (where the output of one tool is consumed by another). Similarity is measured along three sub-dimensions: $S_{\text{node}}$ (optional-tool agreement rate, excluding mandatory tools that all successful models must call), $S_{\text{seq}}$ (sequential pattern similarity, computed as cosine similarity over a three-dimensional feature vector encoding write-then-verify rate, pre-write confirmation rate, and error retry rate), and $S_{\text{dep}}$ (dependency pattern similarity, computed as cosine similarity over output reuse rate, longest dependency chain length, and output fan-out rate).
    - **Design Motivation**: The core innovation lies in $S_{\text{node}}$, which identifies and excludes mandatory tools via the intersection operation $\mathcal{F}_t^{\text{mandatory}} = \bigcap_{M \in \mathcal{M}_t^*} \text{Tools}(M, t)$, computing agreement only over optional tools. This avoids score inflation caused by shared correctness (average inflation of 12.2pp).

3. **LLM-Based Dependency Edge Validation**:

    - **Function**: Accurately identifies output-to-input dependency relationships between tool calls.
    - **Mechanism**: Simple string matching produces numerous false positives (e.g., common dates or IDs appearing coincidentally). An LLM Judge therefore validates each candidate dependency edge semantically, determining whether a matched value genuinely originates from the source tool's output or was known a priori (e.g., from user input).
    - **Design Motivation**: Ensures the accuracy of the dependency graph and prevents noisy edges from distorting similarity computation.

### Loss & Training
This paper presents an evaluation framework and does not involve model training. In controlled distillation experiments, LoRA is used to fine-tune Qwen2.5-14B-Instruct on 200 Claude Sonnet 4.5 trajectories from τ-Bench, with DeepSeek R1 as a non-teacher control, in order to validate the directional detection capability of the proposed metrics.

## Key Experimental Results

### Main Results

| Model | AGS (%) | RPS Overall | $S_{\text{node}}$ (%) | $S_{\text{dep}}$ (%) |
|------|---------|-------------|----------------------|---------------------|
| Claude Opus 4.1 (thinking) | 83.0 | 3.85 | 81.0 | 93.7 |
| Kimi-K2 (thinking) | 82.7 | 3.65 | 82.6 | 94.7 |
| GPT-4.1 | 79.5 | 3.15 | 75.9 | 88.0 |
| GPT-5 | 76.1 | 2.70 | 71.3 | 87.7 |
| DeepSeek-R1 | 78.6 | 3.05 | 78.3 | 85.0 |
| GLM-4.6 | 80.3 | 3.42 | 80.4 | 88.7 |
| Qwen3-235B (thinking) | 75.9 | 2.40 | 68.1 | 92.4 |

### Ablation Study

| Configuration | AGS toward Teacher | AGS toward Control | Notes |
|------|-------------------|-------------------|------|
| Baseline (undistilled) | 0.59 | 0.64 | Original Qwen2.5-14B |
| Distilled | 0.72 (+0.13) | 0.59 (−0.05) | AGS shows directional signal |
| GED Baseline | 0.42 | 0.39 | Reference comparison |
| GED Distilled | 0.65 (+0.23) | 0.59 (+0.20) | GED cannot distinguish direction |

### Key Findings
- Within-family model pairs exhibit AGS scores 5.9pp higher than cross-family pairs, validating the metric's ability to capture behavioral inheritance.
- Kimi-K2 (thinking) surpasses Anthropic's own Opus 4.1 on both $S_{\text{node}}$ and $S_{\text{dep}}$, suggesting strong cross-family behavioral inheritance.
- The Pearson correlation between RPS and AGS is only 0.491, indicating that the two metrics capture independent behavioral dimensions.

## Highlights & Insights
- The distinction between mandatory and optional tools is an elegant design contribution. Excluding mandatory tools reduces $S_{\text{node}}$ by an average of 12.2pp, demonstrating that omitting this distinction substantially overestimates cross-model similarity. This principle generalizes naturally to other agent behavior analysis settings.
- The controlled distillation experiment's directional validation is particularly well-designed: AGS increases toward the teacher direction (+0.13) while decreasing toward the control direction (−0.05), whereas GED increases in both directions (+0.23/+0.20). This cleanly demonstrates that AGS distinguishes teacher-specific convergence from general capability improvement.
- Case analyses reveal that Kimi-K2 and Claude share an enthusiastic affirmative tone (e.g., "Excellent!", "Perfect!") and a preference for redundant verification (calling `find_user_id_by_email` before proceeding), while GPT-5 exhibits an entirely different style. These fine-grained behavioral fingerprints are highly compelling.

## Limitations & Future Work
- Results are reported relative to a single reference model, Claude Sonnet 4.5 (thinking); exhaustive pairwise comparison across all 18 models would require 153 comparisons, entailing substantial computational cost.
- Evaluation covers only three English-language customer service domains from τ-Bench and τ²-Bench; generalization to other domains, task types, and languages remains to be validated.
- RPS relies on a domain-specific stage taxonomy; extending the framework to non-tool-use paradigms such as code generation or multi-agent collaboration requires additional methodological work.

## Related Work & Insights
- **vs. RSE (Lee et al., 2025)**: RSE computes semantic similarity over model responses but does not distinguish necessary from unnecessary behaviors, making it unable to detect distillation directionality (similarity increases toward both teacher and control).
- **vs. GED (Graph Edit Distance)**: GED measures structural graph differences but similarly cannot distinguish behavioral necessity; after distillation, GED increases substantially toward both teacher and non-teacher directions, losing directional discriminative power.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The first framework to distinguish necessary from unnecessary behaviors for distillation detection in tool-use settings; the framing is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 18 models from 8 providers with rigorous controlled experiments, though limited to English customer service domains.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, vivid case analyses, and a complete argumentative chain from intuition to quantification.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ — Highly valuable for understanding behavioral homogenization in the current LLM ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)
- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)

</div>

<!-- RELATED:END -->
