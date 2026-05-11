---
title: >-
  [Paper Note] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories
description: >-
  [ACL 2026][LLM Agent][Small VLM] This paper proposes SPECTRA, a supervision-free framework that enables small vision-language models (SVLMs) to discover effective tool-calling and visual reasoning behaviors through pure…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Small VLM"
  - "Tool Calling"
  - "Cold-Start Reinforcement Learning"
  - "Multi-Objective Reward"
  - "Agentic Trajectory Optimization"
date: 2026-05-08
content_hash: 7ab52d362efbe3e8
---

# Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories

**Conference**: ACL 2026
**arXiv**: [2604.17475](https://arxiv.org/abs/2604.17475)
**Code**: [GitHub](https://github.com/ab-iitd/spectra)
**Area**: Multimodal Agent / Visual Reasoning
**Keywords**: Small VLM, Tool Calling, Cold-Start Reinforcement Learning, Multi-Objective Reward, Agentic Trajectory Optimization

## TL;DR

This paper proposes SPECTRA, a supervision-free framework that enables small vision-language models (SVLMs) to discover effective tool-calling and visual reasoning behaviors through pure environment interaction, leveraging cold-start reinforcement learning (GRPO) and soft structured multi-turn rollout topological constraints. SPECTRA achieves up to 5% improvement in task accuracy and 9% improvement in tool efficiency across 4 multimodal benchmarks, and introduces the Tool Instrumental Utility (TIU) metric to quantify tool effectiveness without supervision.

## Background & Motivation

**Background**: Small vision-language models (SVLMs, e.g., Qwen2.5-VL-7B) are well-suited as agent controllers due to low latency and deployment cost, but lag behind larger models in long-horizon reasoning, fine-grained visual perception, and tool orchestration. Existing improvement approaches follow two paradigms: (1) trajectory fine-tuning—supervised fine-tuning on synthetic tool-calling data (e.g., MM-Traj from T3-Agent), yielding ~20% gains; (2) reinforcement learning—e.g., Tool-R1 optimizes tool-calling sampling efficiency via RL.

**Limitations of Prior Work**: (1) Trajectory fine-tuning relies on expensive synthetic supervision data (typically distilled from large models), limiting scalability and generalization; (2) existing methods optimize tool-calling reasoning without directly improving visual perception—tool use and visual understanding remain decoupled; (3) no metric exists to evaluate tool effectiveness without annotated trajectory labels—existing Tool Accuracy requires ground-truth trajectories.

**Key Challenge**: Teaching SVLMs effective multi-step tool calling requires high-quality supervised trajectories, yet obtaining such trajectories is itself costly and limits generalization. Can a model learn effective tool-use strategies from scratch (cold-start) through environment feedback alone?

**Goal**: (1) Design a supervision-free agent policy optimization method that bypasses dependence on supervised trajectories; (2) improve SVLM visual perception via structured rollout constraints; (3) propose a ground-truth-free metric for tool effectiveness evaluation.

**Key Insight**: The authors observe that the "visual blindness" of SVLMs can be alleviated by enforcing a structured tool-call → observation → perception sequence, anchoring reasoning to visual evidence obtained via tools rather than reasoning directly from raw images. This topological constraint can serve as a structural prior for RL.

**Core Idea**: Combine GRPO reinforcement learning with soft structured rollout topological constraints and multi-objective rewards (correctness + structural integrity + tool utility), enabling SVLMs to autonomously discover tool-driven visual reasoning strategies under cold-start conditions.

## Method

### Overall Architecture

SPECTRA uses an SVLM (Qwen2.5-VL) as the backbone, freezing the visual encoder and adapting the language decoder via LoRA. For each multimodal input $(I, q)$, $G$ structured rollout trajectories are sampled; group-relative advantages are computed from multi-objective rewards and used to optimize policy parameters via the GRPO objective. The action space consists of natural language tokens plus 4 tool primitives (Image Captioning, Object Detection, OCR, Visual Perception).

### Key Designs

1. **Soft Structured Multi-turn Rollout (SSMR)**:

    - **Function**: Forces the model to follow a "gather evidence before reasoning" topological sequence.
    - **Mechanism**: Optimal trajectories must follow the topological sequence $\tau = \langle reason \to tool \to obs \to percep \to reason \to ans \rangle$—first reasoning to select a tool, obtaining tool output (Observation), integrating it with visual features (Perception), reasoning again, then producing an answer. The constraint is "soft": deviations are not hard-blocked but are progressively penalized via a structural integrity reward $R_{struct} = \alpha \cdot \gamma^{\phi(\tau)}$ ($\alpha=2.0$, $\gamma=0.75$, $\phi(\tau)$ maps degree of deviation).
    - **Design Motivation**: Direct reasoning by SVLMs is prone to visual hallucination. Enforcing the tool–observation–perception sequence anchors reasoning to visual evidence provided by tools. Ablation experiments show removing the structural constraint leads to performance drops exceeding 5% on ScienceQA.

2. **Multi-Objective Agent Reward**:

    - **Function**: Simultaneously optimizes correctness, structure, and tool use.
    - **Mechanism**: Total reward $R_{total} = \lambda_1 R_{corr} + \lambda_2 R_{struct} + \lambda_3 R_{tool} + \lambda_4 R_{term}$, comprising four components: (a) task correctness $R_{corr} = C_1 \cdot \mathbb{1}(y_{pred} = y_{gt})$; (b) structural integrity $R_{struct}$—whether the trajectory conforms to the SSMR topology; (c) tool utility $R_{tool} = \mathbb{1}_{syntax} + \mathbb{1}_{success} + R_{div}$—whether tool calls are syntactically valid, successfully executed, and whether diverse tools are used ($R_{div}$ applies per-tool saturation cap $\kappa$ and global cap $\eta$ to prevent reward hacking); (d) termination signal $R_{term}$—ensures reasoning converges to a definitive answer. Final normalization: $R_{Total} = S \times R_{total} / N_{norm}$.
    - **Design Motivation**: Correctness-only rewards cause the model to shortcut (e.g., guessing without tool calls). Multi-objective rewards ensure the model learns not only to answer correctly but also to follow the correct process. In particular, the saturation design of $R_{div}$ prevents mode collapse (e.g., relying exclusively on OCR).

3. **Tool Instrumental Utility (TIU) Evaluation Metric**:

    - **Function**: Quantifies tool effectiveness without ground-truth trajectories.
    - **Mechanism**: $TIU = TER \times \frac{1+TTAC}{2} \times \tanh(TSS)$, composed of three components: (a) Tool Execution Reliability (TER)—successful execution rate of tool calls; (b) Task-Tool Alignment Coefficient (TTAC)—point-biserial correlation between tool use and task success, where positive values indicate tool use facilitates success; (c) Tool Selectivity Score (TSS)—KL divergence of the tool usage distribution from the uniform distribution, where high values indicate strategic rather than random tool selection. $\tanh$ provides a bounded mapping for TSS; $(1+TTAC)/2$ normalizes TTAC to $[0,1]$.
    - **Design Motivation**: Existing Tool Accuracy requires annotated correct tool sequences, which are unavailable in the unsupervised setting. TIU provides a comprehensive evaluation across three dimensions—reliability, alignment, and selectivity—without requiring any ground-truth trajectories.

### Loss & Training

GRPO objective: $\mathcal{J}_{SPECTRA}(\theta) = \mathbb{E}[\frac{1}{G}\sum_i \frac{1}{|\tau_i|}\sum_t \min(\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon_l, 1+\epsilon_h)\hat{A}_{i,t})] - \psi D_{KL}(\pi_\theta \| \pi_{\theta_{ref}})$. The VERL framework with vLLM engine is used; LoRA fine-tuning is applied to Qwen2.5-VL (3B/7B) with 1,000 training and 200 test samples per dataset.

## Key Experimental Results

### Main Results

**Benchmark Comparison (Accuracy %)**

| Model | AI2D | TQA | OK-VQA | ScienceQA | Avg. | MMMU-Pro (OOD) |
|-------|------|-----|--------|-----------|------|----------------|
| GPT-4o | 76.5 | 77.0 | 88.5 | 86.0 | 82.0 | 61.8 |
| Qwen2.5-VL [7B] (base) | 63.8 | 74.6 | 71.5 | 73.5 | 70.9 | 40.5 |
| VERL Baseline [7B] | 67.5 | 73.3 | 74.6 | 78.3 | 73.4 | 44.3 |
| **SPECTRA [7B]** | **71.1** | **77.5** | **79.6** | **83.1** | **77.8** | **46.7** |

**Tool Instrumental Utility (TIU, 7B variants)**

| Configuration | TER (%) | TTAC | TSS | TIU (%) |
|---------------|---------|------|-----|---------|
| Baseline Agent | 77.30 | -0.003 | 2.05 | 35.63 |
| **SPECTRA** | **88.69** | **0.009** | **2.98** | **44.66** |

### Ablation Study

**Leave-One-Out Reward Ablation (SPECTRA 7B)**

| Configuration | AI2D | TQA | OK-VQA | ScienceQA | Avg. |
|---------------|------|-----|--------|-----------|------|
| Full $R_{total}$ | 71.1 | 77.5 | 79.7 | 83.2 | 77.8 |
| w/o $R_{corr}$ | 68.5 | 78.5 | 80.5 | 77.5 | 76.2 |
| w/o $R_{struct}$ | 66.0 | 77.5 | 82.5 | 77.0 | 75.7 |
| w/o $R_{tool}$ | 74.5 | 74.0 | 79.5 | 78.0 | 76.5 |
| w/o $R_{term}$ | 72.0 | 75.5 | 77.5 | 78.0 | 75.7 |

### Key Findings

- SPECTRA 7B outperforms the strongest VERL baseline by an average of 4.4 percentage points and by 2.4 points on OOD (MMMU-Pro).
- TIU improves from 35.63% to 44.66%: TER increases by 11.4% (tool call success rate), and TTAC shifts from negative to positive (tool use transitions from "irrelevant" to "positively correlated" with task success).
- Trajectory analysis shows SPECTRA substantially increases correct Reasoning→Terminal paths (+48) and reduces recursive Tool_Call→Tool_Call loops (−103).
- Removing any reward component on ScienceQA causes drops exceeding 5%; the full multi-objective framework is most critical for complex reasoning.
- The 3B variant also shows consistent gains (60.3→63.9), demonstrating the method's effectiveness on smaller models.

## Highlights & Insights

- The concept of "cold-start RL" is highly valuable—the model can discover tool-use strategies without supervised trajectories, substantially reducing data costs. The key is that the structural prior (SSMR) provides sufficient inductive bias.
- The three-dimensional decomposition of TIU (reliability–alignment–selectivity) offers a reusable framework for unsupervised agent evaluation that can be directly transferred to other tool-calling scenarios.
- The saturation design of the reward diversity term $R_{div}$ is a practical technique—it encourages tool diversity while preventing reward hacking, and is more robust than simple count-based rewards.

## Limitations & Future Work

- Only 4 visual tools are integrated; general-purpose tools such as code execution and search engines are absent, limiting applicability to complex tasks requiring external knowledge.
- Intermediate reasoning steps occasionally exhibit hallucination (e.g., invoking non-existent tools), even when the final answer is correct.
- Training and evaluation are conducted exclusively in MCQ settings; performance on open-ended generation tasks remains unknown.
- The efficiency of cold-start learning depends on well-designed reward signals—new tasks require re-engineering of the reward function.

## Related Work & Insights

- **vs. T3-Agent**: T3-Agent fine-tunes on automatically generated supervised trajectories (MM-Traj); SPECTRA is fully supervision-free, reducing data costs—though T3-Agent may achieve a higher ceiling given sufficient supervised data.
- **vs. Tool-R1**: Tool-R1 applies RL to optimize tool calling but does not directly improve visual perception; SPECTRA anchors tool outputs to visual understanding via SSMR.
- **vs. RL4VLM**: RL4VLM uses environment rewards without structural priors; SPECTRA's topological constraint provides the critical inductive bias for cold-start learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Triple contributions of cold-start RL, soft topological constraints, and TIU metric; individual component techniques (GRPO, LoRA) are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 benchmarks + OOD + ablation + trajectory analysis + qualitative analysis with complete statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and mathematical derivations are complete, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐ Provides a practical framework for unsupervised agent training; the TIU metric can be applied independently.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)
- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)
- [\[ICLR 2026\] Towards Scalable Oversight via Partitioned Human Supervision](../../ICLR2026/llm_agent/towards_scalable_oversight_via_partitioned_human_supervision.md)
- [\[ACL 2026\] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)

</div>

<!-- RELATED:END -->
