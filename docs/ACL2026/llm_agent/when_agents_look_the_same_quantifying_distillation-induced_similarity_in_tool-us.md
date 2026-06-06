---
title: >-
  [Paper Note] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors
description: >-
  [ACL 2026][LLM Agent][Model Distillation] This paper proposes two complementary metrics, RPS and AGS, to quantify the homogenization of LLM Agent tool-use behaviors resulting from distillation. By distinguishing between…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Model Distillation"
  - "Behavioral Homogenization"
  - "Tool Use"
  - "Agent Evaluation"
  - "Behavioral Similarity"
date: 2026-05-08
content_hash: fe6c68ae6b48bf66
---

# When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors

**Conference**: ACL 2026  
**arXiv**: [2604.21255](https://arxiv.org/abs/2604.21255)  
**Code**: [https://github.com/Syuchin/AgentEcho](https://github.com/Syuchin/AgentEcho)  
**Area**: LLM Agent  
**Keywords**: Model Distillation, Behavioral Homogenization, Tool Use, Agent Evaluation, Behavioral Similarity

## TL;DR
This paper proposes two complementary metrics, RPS and AGS, to quantify the homogenization of LLM Agent tool-use behaviors resulting from distillation. By distinguishing between mandatory and non-mandatory behaviors across 18 models, it reveals cross-family behavior inheritance patterns, finding that the behavioral similarity between Kimi-K2 and Claude Sonnet 4.5 even exceeds that of Anthropic's own models.

## Background & Motivation

**Background**: The current LLM Agent landscape is undergoing a "Cambrian explosion" with the emergence of numerous high-performance agents. However, despite diverse origins, these models exhibit highly consistent behaviors in reasoning steps, tool-calling habits, and even failure modes, suggesting many may be "distillation echoes" of a few dominant teacher models.

**Limitations of Prior Work**: Existing similarity measures primarily focus on response-level similarity in static dialogues, failing to capture the dynamic nature of multi-step tool-use trajectories. More critically, these methods cannot distinguish between "mandatory behaviors" (actions required for task success) and "non-mandatory behaviors" (actions reflecting autonomous model preferences), leading to similarity scores inflated by shared correct paths required by the task itself.

**Key Challenge**: Without distinguishing mandatory from non-mandatory behaviors, it is impossible to determine whether two models converge because there is only one correct path or because one model is blindly imitating the habits of another—a fundamental obstacle to quantifying distillation impact.

**Goal**: Design a systematic framework to isolate non-mandatory behavior patterns and quantify distillation-induced behavioral homogenization between agents across verbal expression and tool operation dimensions.

**Key Insight**: Authors observe that many agents perform redundant tool calls (e.g., trying all available tools sequentially even when the answer is obvious); these non-mandatory behavioral choices serve as "behavioral fingerprints" for identifying distilled models.

**Core Idea**: By decomposing agent trajectories into mandatory and non-mandatory behaviors, the framework uses RPS (Response Pattern Similarity) and AGS (Action Graph Similarity) to capture signals of behavioral inheritance across different dimensions.

## Method

### Overall Architecture
Given a set of models and tool-use tasks, execution trajectories are collected for each model. Similarity analysis is then conducted across two orthogonal dimensions: RPS focuses on how models express responses verbally (verbal fingerprint), while AGS focuses on how they select and organize tool calls (behavioral fingerprint). Using Claude Sonnet 4.5 (thinking) as the reference "oracle" model, behavioral similarity is calculated for other models relative to it.

### Key Designs

1. **Response Pattern Similarity (RPS)**:

    - **Function**: Quantifies how similar two models are in their verbal expressions.
    - **Mechanism**: Employs a two-stage pipeline—first, Stage Annotation aligns trajectory semantics to five canonical stages (Authorization, Info Acquisition, Execution, Verification, Notification), addressing the varying turn counts across models for the same task. Then, on shared stages, an LLM Judge scores the similarity across three dimensions: Style, Structure, and Alignment (1-5 scale), using the mean Overall score.
    - **Design Motivation**: Direct comparison of full trajectories or turn-by-turn alignment often matches irrelevant content, leading to unreliable ratings. Semantic-level stage alignment ensures only functionally equivalent interaction segments are compared.

2. **Action Graph Similarity (AGS)**:

    - **Function**: Analyzes structured behavioral patterns from tool-calling sequences.
    - **Mechanism**: Constructs dialogue trajectories as directed graphs $G=(V, E_s, E_d)$, where nodes are tool calls, $E_s$ represents temporal sequence edges, and $E_d$ represents dependency edges (output of a previous tool used by a subsequent one). Similarity is measured via three sub-dimensions: $S_{\text{node}}$ (optional tool consistency, excluding mandatory tools), $S_{\text{seq}}$ (sequence pattern similarity using cosine similarity of 3D feature vectors: write-after-verify, write-before-confirm, and error-retry rates), and $S_{\text{dep}}$ (dependency pattern similarity using cosine similarity of output reuse, max dependency chain length, and output fan-out rates).
    - **Design Motivation**: The key innovation, $S_{\text{node}}$, identifies and excludes mandatory tools via an intersection operation $\mathcal{F}_t^{\text{mandatory}} = \bigcap_{M \in \mathcal{M}_t^*} \text{Tools}(M, t)$. Calculating consistency only on optional tools avoids score inflation caused by shared correctness (average inflation of 12.2pp).

3. **LLM Verification for Dependency Edges**:

    - **Function**: Accurately identifies output-input dependencies between tools.
    - **Mechanism**: Simple string matching generates many false positives (e.g., common dates or IDs repeating by chance). An LLM Judge validates the semantic validity of each candidate dependency edge, determining if the matched value truly originates from a source tool's output or is prior knowledge (e.g., user input).
    - **Design Motivation**: Ensures the accuracy of the dependency graph and prevents noisy edges from interfering with similarity calculations.

### Loss & Training
This is an evaluation framework and does not involve model training. In controlled distillation experiments, Qwen2.5-14B-Instruct was fine-tuned using LoRA on 200 Claude Sonnet 4.5 trajectories from τ-Bench. DeepSeek R1 was used as a non-teacher control group to verify the directional detection capability of the metrics.

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

| Configuration | AGS toward Teacher | AGS toward Control | Description |
|------|-------------------|-------------------|------|
| Baseline (Undistilled) | 0.59 | 0.64 | Original Qwen2.5-14B |
| Distilled | 0.72 (+0.13) | 0.59 (-0.05) | AGS shows directional signal |
| GED Baseline | 0.42 | 0.39 | Original comparison |
| GED Distilled | 0.65 (+0.23) | 0.59 (+0.20) | GED cannot distinguish direction |

### Key Findings
- Within-family model pairs show AGS scores 5.9pp higher than cross-family pairs, validating that the metrics capture behavioral inheritance.
- Kimi-K2 (thinking) exceeds Anthropic's own Opus 4.1 in both $S_{\text{node}}$ and $S_{\text{dep}}$, suggesting strong cross-family behavior inheritance.
- The Pearson correlation coefficient between RPS and AGS is only 0.491, indicating that the two metrics capture independent behavioral dimensions.

## Highlights & Insights
- Incorporating the distinction between mandatory and optional tools into distillation detection is a clever design. Excluding mandatory tools reduces $S_{\text{node}}$ by an average of 12.2pp, showing that failing to make this distinction severely overestimates cross-model similarity. This approach is generalizable to other agent behavior analyses.
- The directional validation in controlled distillation experiments is well-designed: AGS increased toward the teacher (+0.13) while decreasing toward the control (-0.05). In contrast, GED increased toward both (+0.23/+0.20), clearly proving that AGS distinguishes "specific teacher-oriented convergence" from "general capability improvement."
- Case studies reveal that Kimi-K2 and Claude share an "enthusiastic affirmative tone" (e.g., "Excellent!", "Perfect!") and redundant verification preferences (calling `find_user_id_by_email` before proceeding), whereas GPT-5 has a completely different style. These fine-grained behavioral fingerprints are highly convincing.

## Limitations & Future Work
- Results are reported using only Claude Sonnet 4.5 (thinking) as the reference model; a full pairwise comparison of all 18 models would require 153 comparisons, which is computationally expensive.
- Evaluation only covers three English customer service domains in τ-Bench and τ²-Bench; generalization to other domains, task types, and languages remains to be verified.
- RPS relies on domain-specific stage taxonomies. Extending it to non-tool-use paradigms like code generation or multi-agent collaboration requires further methodological work.

## Related Work & Insights
- **vs RSE (Lee et al., 2025)**: RSE calculates semantic similarity on model responses but does not distinguish between mandatory/non-mandatory behaviors, making it unable to detect distillation directionality (it increases toward both teacher and control).
- **vs GED (Graph Edit Distance)**: GED measures graph structure differences but similarly fails to distinguish the necessity of behaviors. After distillation, GED increases significantly toward both teacher and non-teacher models, losing directional discriminative power.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First framework to distinguish mandatory/non-mandatory behaviors for tool-use distillation detection; very unique entry point.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 18 models from 8 providers with rigorous controlled experiments, though limited to English customer service domains.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, vivid case studies, and a complete logical chain from intuition to quantification.
- **Value**: ⭐⭐⭐⭐⭐ Provides significant value for understanding behavioral homogenization within the current LLM ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](../../ICML2026/llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"](toolgrad_efficient_tool-use_dataset_generation_with_textual_gradients.md)

</div>

<!-- RELATED:END -->
