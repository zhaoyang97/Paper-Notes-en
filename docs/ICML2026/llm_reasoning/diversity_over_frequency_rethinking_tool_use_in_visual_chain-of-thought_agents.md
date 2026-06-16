---
title: >-
  [Paper Note] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents
description: >-
  [ICML 2026][LLM Reasoning][RFT] In "tool-optional" visual Agent tasks such as 3D spatial reasoning, the authors discover that vanilla RFT causes tool call rates to collapse to near zero, while explicitly encouraging tool use yields only marginal gains. The true driver of performance enhancement is the exploration diversity of rollouts. By employing a
tags:
  - ICML 2026
  - LLM Reasoning
  - RFT
date: 2026-05-08
content_hash: 864d027a612a76f5
---
# Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents

**Conference**: ICML 2026  
**arXiv**: [2606.00096](https://arxiv.org/abs/2606.00096)  
**Code**: https://scaffolded-exploration.github.io  
**Area**: LLM Reasoning / Visual Agents / Reinforcement Learning  
**Keywords**: Visual Chain-of-Thought, Tool Use Collapse, Entropy Regularization, RFT, Exploration Diversity

## TL;DR
In "tool-optional" visual Agent tasks such as 3D spatial reasoning, the authors discover that vanilla RFT causes tool call rates to collapse to near zero, while explicitly encouraging tool use yields only marginal gains. The true driver of performance enhancement is the exploration diversity of rollouts. By employing adaptive entropy regularization, they push the 3DSRBench accuracy from 59.2% to 62.9% and reposition tools as "training-time scaffolding" rather than inference-time necessities.

## Background & Motivation

**Background**: Current visual Agents (DeepEyes, Mini-o3, PixelReasoner, Chain-of-Focus, etc.) string together visual tools like `<grounding>`, cropping, and segmentation into a visual chain-of-thought. They then use group-based RL, such as GRPO or DAPO, for fine-tuning (RFT), achieving significant results on "high-resolution visual search" benchmarks like V\*.

**Limitations of Prior Work**: Nearly all existing works focus on visual search scenarios where "tools are inevitably useful"—targets are small and must be zoomed in to be seen clearly. However, in broader visual reasoning tasks (3D spatial relations, medical VQA), whether and when to use tools is less clear. Existing methods exhibit strange behavior in these "tool-optional" settings: either not using tools leads to higher accuracy, or forcing tool use fails to improve accuracy.

**Key Challenge**: In tool-optional scenarios, an optimization asymmetry exists. Tool-based rollouts involve more interaction rounds, longer token sequences, and higher variance than tool-free ones. Even with token-level loss and over-turn masking, GRPO naturally favors tool-free paths. Yet, one cannot simply conclude that "tools are useless," as disabling tools entirely leads to a performance drop.

**Goal**: (1) Systematically investigate the overlooked regime of "tool-optional + complex visual reasoning"; (2) Identify the shared essence of failure in vanilla RFT and tool-encouraging RFT; (3) Provide an intervention that enables RL to truly learn in such settings.

**Key Insight**: The authors move beyond the 2D perspective of "tool frequency vs. accuracy" to quantify diversity in both text space (distinct-n-grams of the pre-grounding `<think>` span) and visual space (mean pairwise IoU of crop boxes for the same query + CLIP alignment with question keywords). They found that both vanilla and tool-encouraging paths monotonically collapse, whereas the initial ~20% tool calls provide diverse exploration histories.

**Core Idea**: Treat tools as "scaffolding" during training rather than necessities for inference. As long as rollouts explore with enough diversity early in training (either through tool use or active entropy regularization), tools can naturally fade out in later stages while the model remains strong.

## Method

### Overall Architecture
This paper aims to clarify how RFT should handle visual tools in "tool-optional" complex visual reasoning tasks. The baseline agent is Mini-o3 (Qwen2.5-VL-7B-Instruct + SFT + RFT, already capable of calling `<grounding>`), and the training data consists of 1.2k 3D spatial reasoning QA pairs from SpatialReasoner. The Agent follows a thought–action–observation cycle: each step generates a `<think>` reasoning, then either outputs a `<grounding>` with `(bbox_2d, source)` to trigger a zoom-in crop and append new observations to the history, or directly provides an `<answer>`. The paper follows a "diagnosis–intervention–causal validation" chain: diagnosing that diversity is the culprit using vanilla, tool-banned, and tool-encouraged RFT; providing adaptive entropy regularization as an intervention; and finally using tool-banned + entropy regularization to prove that the gains from entropy regularization must be realized through the visual exploration provided by tools.

### Key Designs

**1. Diversity Diagnosis Metrics: Decoupling "Exploration Breadth" from "Tool Frequency"**

The first step is identifying a measurable metric independent of "tool call rates." The authors decompose rollout diversity into text and visual axes. On the text side, they only count the `<think>` segment preceding `<grounding>`. They use an entropy probe to confirm that model uncertainty primarily resides in reasoning spans rather than bbox coordinates, calculating the ratio of distinct-$n$-grams for $n\in\{3,4,5,6\}$. On the visual side, they sample 50 rollouts for the same (image, question) and calculate: first, the mean pairwise IoU of all crop boxes (lower means wider coverage); second, the CLIP similarity between crop patches and question noun keywords (higher means more relevant exploration). These two visual metrics must be viewed together: low mIoU alone might mean "broad but off-track," while adding CLIP distinguishes "broad and relevant" from "concentrated and fixed." These metrics are crucial because they immediately challenge the intuition that "more tool calls equal more exploration"—tool-encouraging lines call tools ~3x more than vanilla, yet mIoU remains >0.55 and CLIP does not increase, shifting the focus from "tool frequency" to the true lack of "exploration diversity."

**2. Adaptive Entropy Regularization: A Proportional Feedback Knob to Prevent Early Rollout Collapse**

After diagnosis points to diversity, the intervention targets the root cause of "preventing collapse" without manipulating tool frequency. An entropy reward term is added to the GRPO objective:

$$\mathcal{J}_{\text{ent}}(\theta)=\mathcal{J}_{\text{GRPO}}(\theta)+\lambda_t\cdot\mathbb{E}_{q,\tau}[\bar{\mathcal{H}}(\tau)],$$

where $\bar{\mathcal{H}}(\tau)$ is the token-level average entropy of the entire rollout, and single-token entropy is $\mathcal{H}(\pi_\theta(\cdot\mid s_k))=-\sum_v \pi_\theta(v\mid s_k)\log\pi_\theta(v\mid s_k)$. The difficulty lies in the coefficient $\lambda$: fixing $\lambda$ is fragile—too small has no effect, while too large leads to mixed languages or repetition loops. Thus, the authors adopt proportional feedback control, adapting the coefficient based on the current batch entropy: $\lambda_t = K_p\,[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$, with a target entropy $\mathcal{H}_{\text{target}}=0.9$ and $K_p=0.03$. Pressure is applied only when the current entropy is below the target. This single-knob, tuning-free design yielded results: tool calls still dropped from ~20% to 3%, but 3DSRBench accuracy reached 62.9%, significantly higher than the baseline 59.2% and the forced-tool 59.9%.

**3. Tool-banned × Entropy Regularization: Isplaying "Entropy Gains" from "Tool Gains"**

The final component isoltes the confounding factor of whether entropy regularization is a universal remedy. The authors reuse the strict tool-banned protocol (masking `<grounding>` trigger tokens during rollout) and apply the same adaptive entropy regularization without tools. They also use over-turn masking throughout—removing rollouts that exceed the budget from advantage calculations rather than giving them a negative reward, preventing GRPO from implicitly punishing long rollouts and exacerbating tool-use collapse. The results show that tool-banned + entropy regularization only reaches 57.8%, lower than the vanilla tool-banned 58.1%, and well below the tool-enabled + entropy regularization 62.9%. This proves that entropy regularization gains are not magic—without the visual evidence diversity provided by tools early on, simply increasing entropy can be harmful. The "tools as scaffolding" argument is thus firmly established.

### Loss & Training
The base optimizer is DAPO (GRPO + clip-higher + dynamic sampling + token-level policy loss + over-turn masking), trained for $\le 100$ steps. Two types of comparison rewards are used: DeepEyes tool bonus $R_{\text{DE}}=\mathbb{I}[y=y^*]+\lambda_{\text{tool}}\mathbb{I}[y=y^*]\mathbb{I}[u(\tau)=1]$ and PixelReasoner curiosity reward $R_{\text{PR}}=\mathbb{I}[y=y^*]+\alpha\max(H-\mathrm{RaPR}(q),0)\mathbb{I}[u(\tau)=1]+\beta\,r_{\text{penalty}}(\tau)$, where $\mathrm{RaPR}(q)=\mathbb{E}_\tau[u(\tau)]$ is the tool call rate for that query and $r_{\text{penalty}}=\min(N-n_{\text{tool}}(\tau),0)$ is a tool count cap. Evaluations use Avg@8, with VLM-as-judge (Qwen2.5-VL-7B) extracting options.

## Key Experimental Results

### Main Results: 3DSRBench + CV-Bench-3D Overview

| Configuration | Tools? | 3DSRBench Acc | CV-Bench-3D Acc | Tool Use Rate (Initial → Saturation) |
|------|-------|---------------|-----------------|---------------------|
| Qwen2.5-VL 7B (Generalist) | – | 48.4 | 82.9 | – |
| Mini-o3 (Zero-shot) | Yes | 54.5 | 77.6 | – |
| SpatialReasoner (Specialist) | – | 60.3 | 80.3 | – |
| Vanilla RFT | Yes | 59.2 | 76.7 | ~20% → ~2% |
| Tool-banned | No | 58.1 | – | 0% → 0% |
| Tool-Encourage (DeepEyes) | Yes | 59.9 | 74.5 | ~20% → 100% |
| **Entropy-Regularized** | Yes | **62.9** | **78.8** | ~20% → ~3% |
| Tool-banned + Entropy | No | 57.8 | – | 0% → 0% |

### Ablation/Analysis: Exploration Diversity

| Method | Crop mIoU (↓) | CLIP (↑) | Visual Behavior |
|------|---------------|----------|----------|
| Vanilla RFT | 0.554 | 0.184 | Highly rigid |
| Tool-Encourage | 0.557 | 0.187 | Highly rigid (despite ~3× crops) |
| Entropy-Regularized | **0.494** | 0.184 | Active exploration with alignment |

VQA-RAD (Medical VQA, OpenThinkIMG heterogeneous toolset): Vanilla 46.34 → Tool-Encourage 47.23 → Entropy 48.78, showing identical trends.

### Key Findings
- "Tool frequency" and "accuracy" are almost orthogonal on 3DSRBench: accuracy rises steadily even as tool-use collapses from 20% to 2%; forcing 100% tool use only adds 0.7 percentage points.
- Diversity is the hidden variable: Text distinct-n-grams monotonically decline in both vanilla and tool-encourage, and crop box mIoU remains >0.55. Only entropy regularization keeps mIoU at 0.494 without dropping CLIP, leading to significant accuracy gains.
- The "scaffolding" hypothesis is supported by two counterpoints: tool-banned dropping to 58.1% shows early tools are indispensable; tool-banned + entropy dropping to 57.8% shows entropy gains require tools to be realized.
- On CV-Bench-3D, both vanilla RFT and tool-encourage perform worse than the Mini-o3 base (-0.9 / -3.1), while entropy regularization gains +1.2, suggesting that both forced tool use and collapse harm general visual understanding.

## Highlights & Insights
- The diagnosis framework clearly decoupling "tool rate" from "exploration diversity" is elegant: the combination of text distinct-n-gram + visual mIoU + CLIP can be directly applied to any visual Agent RL project for monitoring.
- The proportional feedback $\lambda_t = K_p[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$ for adaptive entropy regularization is a cost-effective engineering trick—single knob, no tuning, avoids parrot-like collapse, and integrates easily into GRPO/DAPO pipelines.
- The "tools as training-time scaffolding" framing upends the prevailing "train agents to use more tools" mindset: inference-time tool call rates aren't the optimization goal; what matters is the breadth of the "experience map" traversed during training. This is insightful for all tool-augmented RL—a tool bonus in rewards might inadvertently suppress diversity.
- The experimental design of tool-banned + entropy regularization is the ultimate ablation, cleanly isolating "entropy gain" from "tool gain" interference.

## Limitations & Future Work
- The primary evaluation is on 1.2k SpatialReasoner training samples + 3DSRBench; cross-task generalization relies on VQA-RAD/CV-Bench-3D, and whether "scaffolding" is harmful in "tool-essential" settings like V\* visual search hasn't been systematically verified.
- Training stopped at 100 steps due to DAPO dynamic sampling engineering limits (degenerate groups causing wall-clock explosions). Whether entropy regularization maintains diversity over longer periods is unknown.
- The base agent is locked to Mini-o3 (Qwen2.5-VL-7B). Whether the "initial 20% tool use" and entropy gains hold for weaker bases (that haven't learned tools) or larger models remains to be tested.
- Target entropy $\mathcal{H}_{\text{target}}=0.9$ and $K_p=0.03$ are empirical values. Theoretically, they should correlate with model size and vocabulary distribution, but a guideline is missing.

## Related Work & Insights
- **vs. DeepEyes / Mini-o3 / PixelReasoner / Chain-of-Focus**: These methods train Agents to use zoom/crop for visual search, verified on "tool-essential" benchmarks like V\*. This paper moves to "tool-optional" settings like 3D spatial reasoning, revealing that their common reward designs collapse or saturate, and offers a cross-scenario intervention (entropy regularization).
- **vs. PixelReasoner curiosity reward / DeepEyes tool bonus**: These explicit tool-encouraging rewards are shown to increase tool rates but not accuracy, while causing diversity to collapse. This paper successfully replaces them with "task-agnostic" exploration pressure (entropy regularization).
- **vs. SpatialReasoner**: The specialist approach reaches 60.3% by explicitly injecting 3D coordinates into reasoning. This work hits 62.9% without specialized spatial supervision, suggesting exploration control in RL dynamics can replace domain-specific inductive biases.

## Rating
- Novelty: ⭐⭐⭐⭐ The "tool use collapse" phenomenon is systematically named and quantified for the first time, and repositioning tools as scaffolding is a genuine cognitive update for visual agent RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five-method comparisons + dual-axis diversity diagnosis + causal tool-banned/entropy split are complete. VQA-RAD and CV-Bench-3D provide evidence across tasks/tools; however, scale is limited to 100 steps and cross-base model validation is missing.
- Writing Quality: ⭐⭐⭐⭐ The early vs. late training comparison and "scaffolding" metaphor are intuitive; the narrative chain from 3.1 to 3.5 is clear; math density is manageable.
- Value: ⭐⭐⭐⭐ Two takeaways for tool-augmented RL: (1) add diversity metrics to training monitoring; (2) rethink tool bonuses that might suppress diversity. The diagnosis and entropy schemes are ready for zero-cost deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)

</div>

<!-- RELATED:END -->
