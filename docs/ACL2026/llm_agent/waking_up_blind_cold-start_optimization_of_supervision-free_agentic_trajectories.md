---
title: >-
  [Paper Note] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories
description: >-
  [ACL 2026][LLM Agent][Small VLM] This paper proposes SPECTRA, a framework that requires no supervised trajectories. Through cold-start Reinforcement Learning (GRPO) and soft-structured multi-round rollout topological con…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Small VLM"
  - "Tool Calling"
  - "Cold-Start RL"
  - "Multi-Objective Reward"
  - "Agent Trajectory Optimization"
date: 2026-05-08
content_hash: d2ff721c7353bd33
---

# Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories

**Conference**: ACL 2026  
**arXiv**: [2604.17475](https://arxiv.org/abs/2604.17475)  
**Code**: [GitHub](https://github.com/ab-iitd/spectra)  
**Area**: Multimodal Agent / Visual Reasoning  
**Keywords**: Small VLM, Tool Calling, Cold-Start RL, Multi-Objective Reward, Agent Trajectory Optimization

## TL;DR

This paper proposes SPECTRA, a framework that requires no supervised trajectories. Through cold-start Reinforcement Learning (GRPO) and soft-structured multi-round rollout topological constraints, it enables Small Vision-Language Models (SVLMs) to autonomously discover effective tool-calling and visual reasoning behaviors through pure environmental interaction. It improves task accuracy by up to 5% and tool efficiency by 9% across four multimodal benchmarks, while introducing the Tool Instrumental Utility (TIU) metric to quantify tool effectiveness in unsupervised settings.

## Background & Motivation

**Background**: Small Vision-Language Models (SVLMs, such as Qwen2.5-VL-7B) are suitable as agent controllers due to low latency and deployment costs, but they lag behind large models in long-horizon reasoning, fine-grained visual perception, and tool orchestration. Existing improvement methods follow two paths: (1) Trajectory Fine-Tuning—supervised fine-tuning using synthetic tool-calling data (e.g., T3-Agent's MM-Traj), which can improve performance by ~20%; (2) Reinforcement Learning—e.g., Tool-R1, which optimizes the sampling efficiency of tool calls via RL.

**Limitations of Prior Work**: (1) Trajectory fine-tuning relies on expensive synthetic supervised data (usually distilled from large models), limiting scalability and generalization; (2) Existing methods for optimizing tool-calling reasoning do not directly improve visual perception—tool use and visual understanding remain decoupled; (3) There is a lack of metrics to evaluate tool effectiveness without annotated trajectory labels—existing Tool Accuracy depends on ground truth trajectories.

**Key Challenge**: Enabling SVLMs to learn effective multi-step tool calls requires high-quality supervised trajectories, but obtaining these trajectories is inherently expensive and limits generalization. Can a model discover effective tool-use strategies from scratch (cold-start) solely through environmental feedback?

**Goal**: (1) Design an unsupervised agent policy optimization method to bypass the dependency on supervised trajectories; (2) Improve visual perception in SVLMs through structured rollout constraints; (3) Propose tool effectiveness evaluation metrics that do not rely on ground truth.

**Key Insight**: It is observed that the "visual blind spots" of SVLMs can be mitigated by enforcing a structured sequence of tool-call-observation-perception. This allows the model to obtain visual evidence via tools first and then reason based on that evidence, rather than reasoning directly from the original image. This topological constraint serves as a structural prior for RL.

**Core Idea**: Use GRPO reinforcement learning + soft-structured rollout topological constraints + multi-objective rewards (correctness + structural integrity + tool utility) to enable SVLMs to self-discover tool-driven visual reasoning strategies under cold-start conditions.

## Method

### Overall Architecture

SPECTRA is based on SVLMs (Qwen2.5-VL), with the vision encoder frozen and the language decoder adapted using LoRA. For each multimodal input $(I, q)$, $G$ structured rollout trajectories are sampled. Group relative advantage is computed through multi-objective rewards, and policy parameters are optimized using the GRPO objective. The action space consists of natural language tokens and four tool primitives (Image Captioning, Object Detection, OCR, and Visual Perception).

### Key Designs

1.  **Soft-Structured Multi-Round Rollout (SSMR)**:
    - **Function**: Enforces a topological sequence of "obtain evidence before reasoning."
    - **Mechanism**: Optimal trajectories must follow the topological sequence $\tau = \langle reason \to tool \to obs \to percep \to reason \to ans \rangle$—reasoning to select a tool, obtaining the tool output (Observation), integrating the output with visual features (Perception), and providing an answer after further reasoning. This constraint is "soft"—deviations are not strictly prohibited but are progressively penalized via a structural integrity reward $R_{struct} = \alpha \cdot \gamma^{\phi(\tau)}$ ($\alpha=2.0$, $\gamma=0.75$, where $\phi(\tau)$ maps the degree of deviation).
    - **Design Motivation**: Direct reasoning by SVLMs is prone to visual hallucinations. Enforcing the tool-observation-perception sequence forces the model to anchor its reasoning in visual evidence provided by tools. Ablation studies show that removing structural constraints leads to a performance drop of over 5% on ScienceQA.

2.  **Multi-Objective Agent Reward**:
    - **Function**: Simultaneously optimizes correctness, structure, and tool usage.
    - **Mechanism**: The total reward is $R_{total} = \lambda_1 R_{corr} + \lambda_2 R_{struct} + \lambda_3 R_{tool} + \lambda_4 R_{term}$, comprising four components: (a) Task Correctness $R_{corr} = C_1 \cdot \mathbb{1}(y_{pred} = y_{gt})$; (b) Structural Integrity $R_{struct}$—whether the trajectory follows the SSMR topology; (c) Tool Utility $R_{tool} = \mathbb{1}_{syntax} + \mathbb{1}_{success} + R_{div}$—whether the tool call is syntactically valid, executes successfully, and uses diverse tools ($R_{div}$ has a per-tool saturation cap $\kappa$ and a global cap $\eta$ to prevent reward hacking); (d) Termination $R_{term}$—ensuring reasoning converges to a clear answer. Final normalization: $R_{Total} = S \times R_{total} / N_{norm}$.
    - **Design Motivation**: Using only correctness rewards leads to shortcuts (e.g., guessing without tools). Multi-objective rewards ensure the model learns "process correctness" alongside "result correctness." Specifically, the saturation design of $R_{div}$ prevents mode collapse (e.g., over-relying on OCR).

3.  **Tool Instrumental Utility (TIU) Metric**:
    - **Function**: Quantifies tool effectiveness in the absence of ground truth trajectories.
    - **Mechanism**: $TIU = TER \times \frac{1+TTAC}{2} \times \tanh(TSS)$, composed of three factors: (a) Tool Execution Reliability (TER)—the success rate of tool execution; (b) Task-Tool Alignment Coefficient (TTAC)—the point-biserial correlation between tool usage and task success, where positive values indicate tool use aids success; (c) Tool Selectivity Score (TSS)—the KL divergence between the tool usage distribution and a uniform distribution, where high values indicate strategic selection. $\tanh$ bounds TSS, and $(1+TTAC)/2$ normalizes TTAC to [0,1].
    - **Design Motivation**: Existing Tool Accuracy metrics require annotated gold tool sequences, which are unavailable in unsupervised settings. TIU evaluates from reliability, alignment, and selectivity dimensions without requiring ground truth trajectories.

### Loss & Training

GRPO objective: $$\mathcal{J}_{SPECTRA}(\theta) = \mathbb{E}[\frac{1}{G}\sum_i \frac{1}{|\tau_i|}\sum_t \min(\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon_l, 1+\epsilon_h)\hat{A}_{i,t})] - \psi D_{KL}(\pi_\theta \| \pi_{\theta_{ref}})$$. Training uses the VERL framework + vLLM engine, with LoRA fine-tuning on Qwen2.5-VL (3B/7B). Each dataset includes 1,000 training and 200 testing samples.

## Key Experimental Results

### Main Results

**Benchmark Comparison (Accuracy %)**

| Model | AI2D | TQA | OK-VQA | ScienceQA | Avg. | MMMU-Pro(OOD) |
|-------|------|-----|--------|-----------|------|---------------|
| GPT-4o | 76.5 | 77.0 | 88.5 | 86.0 | 82.0 | 61.8 |
| Qwen2.5-VL [7B] (base) | 63.8 | 74.6 | 71.5 | 73.5 | 70.9 | 40.5 |
| VERL Baseline [7B] | 67.5 | 73.3 | 74.6 | 78.3 | 73.4 | 44.3 |
| **SPECTRA [7B]** | **71.1** | **77.5** | **79.6** | **83.1** | **77.8** | **46.7** |

**Tool Instrumental Utility (TIU, 7B variants)**

| Configuration | TER(%) | TTAC | TSS | TIU(%) |
|---------------|--------|------|-----|--------|
| Baseline Agent | 77.30 | -0.003 | 2.05 | 35.63 |
| **SPECTRA** | **88.69** | **0.009** | **2.98** | **44.66** |

### Ablation Study

**Leave-one-out Reward Ablation (SPECTRA 7B)**

| Configuration | AI2D | TQA | OK-VQA | ScienceQA | Avg. |
|---------------|------|-----|--------|-----------|------|
| Full $R_{total}$ | 71.1 | 77.5 | 79.7 | 83.2 | 77.8 |
| w/o $R_{corr}$ | 68.5 | 78.5 | 80.5 | 77.5 | 76.2 |
| w/o $R_{struct}$ | 66.0 | 77.5 | 82.5 | 77.0 | 75.7 |
| w/o $R_{tool}$ | 74.5 | 74.0 | 79.5 | 78.0 | 76.5 |
| w/o $R_{term}$ | 72.0 | 75.5 | 77.5 | 78.0 | 75.7 |

### Key Findings

- SPECTRA 7B outperforms the strongest VERL baseline by an average of 4.4 percentage points and improves by 2.4 points on OOD (MMMU-Pro).
- TIU improves from 35.63% to 44.66%—TER increases by 11.4% (tool execution success), and TTAC shifts from negative to positive (tool use becomes positively correlated with success).
- Trajectory Analysis: SPECTRA significantly increases valid Reasoning→Terminal paths (+48) and reduces Tool_Call→Tool_Call recursive loops (-103).
- On ScienceQA, removing any reward component results in a >5% drop, indicating the multi-objective framework is critical for complex reasoning.
- The 3B variant shows consistent gains (60.3→63.9), proving the method's effectiveness for smaller models.

## Highlights & Insights

- The concept of "Cold-Start RL" is highly valuable—enabling models to discover tool-use strategies without supervised trajectories significantly reduces data costs. The key lies in the structural prior (SSMR) providing sufficient inductive bias.
- The three-dimensional decomposition of the TIU metric (Reliability-Alignment-Selectivity) provides a reusable framework for unsupervised agent evaluation that can be migrated to other tool-calling scenarios.
- The saturation design of the reward diversity term $R_{div}$ is a practical trick—it encourages tool variety while preventing reward hacking, making it more robust than simple count-based rewards.

## Limitations & Future Work

- Integration is limited to 4 visual tools, lacking general-purpose tools like code execution or search engines, which limits applicability to complex tasks requiring external knowledge.
- Despite correct final results, intermediate reasoning steps occasionally exhibit hallucinations (e.g., hallucinating non-existent tools).
- Training and evaluation are conducted only in MCQ scenarios; performance on open-ended generation tasks remains unknown.
- The efficiency of cold-start learning depends on well-designed rewards; new tasks may require re-designing reward signals.

## Related Work & Insights

- **vs T3-Agent**: T3-Agent uses automatically generated supervised trajectories (MM-Traj) for fine-tuning; SPECTRA is entirely unsupervised, reducing data costs, though T3-Agent may have a higher upper bound when sufficient supervised data is available.
- **vs Tool-R1**: Tool-R1 uses RL to optimize tool calls but does not directly improve visual perception; SPECTRA anchors tool outputs to visual understanding via SSMR.
- **vs RL4VLM**: RL4VLM uses environmental rewards but lacks structural priors; SPECTRA’s topological constraints provide a critical inductive bias for cold-start learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Significant contributions via Cold-Start RL + Soft Topological Constraints + TIU metric, though individual components (GRPO, LoRA) are established.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks + OOD + Ablations + Trajectory Analysis + Qualitative Analysis with complete statistical testing.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete mathematical derivations, though high notation density requires careful reading.
- Value: ⭐⭐⭐⭐ Provides a practical framework for unsupervised agent training; the TIU metric is independently useful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)
- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)
- [\[ICLR 2026\] Towards Scalable Oversight via Partitioned Human Supervision](../../ICLR2026/llm_agent/towards_scalable_oversight_via_partitioned_human_supervision.md)

</div>

<!-- RELATED:END -->
