---
title: >-
  [Paper Note] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents
description: >-
  [ICML 2026][LLM Reasoning][Visual Chain-of-Thought] In "tool-optional" visual agent tasks such as 3D spatial reasoning, the authors find that vanilla RFT causes tool invocation rates to collapse to near zero…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Visual Chain-of-Thought"
  - "Tool-Use Collapse"
  - "Entropy Regularization"
  - "RFT"
  - "Exploration Diversity"
date: 2026-05-08
content_hash: 72f9a1093ac0d91e
---

# Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents

**Conference**: ICML 2026  
**arXiv**: [2606.00096](https://arxiv.org/abs/2606.00096)  
**Code**: https://scaffolded-exploration.github.io  
**Area**: LLM Reasoning / Visual Agents / Reinforcement Learning  
**Keywords**: Visual Chain-of-Thought, Tool-Use Collapse, Entropy Regularization, RFT, Exploration Diversity

## TL;DR
In "tool-optional" visual agent tasks such as 3D spatial reasoning, the authors find that vanilla RFT causes tool invocation rates to collapse to near zero, while explicitly encouraging tool use yields only marginal gains. The true driver of performance is the exploration diversity of rollouts. By employing adaptive entropy regularization, the authors improve 3DSRBench accuracy from 59.2% to 62.9% and reposition tools as "training-time scaffolding" rather than inference-time necessities.

## Background & Motivation

**Background**: Current visual agents (e.g., DeepEyes, Mini-o3, PixelReasoner, Chain-of-Focus) sequence visual tools like `<grounding>`, cropping, or segmentation into a visual chain-of-thought. These are fine-tuned using group-based RL (RFT) like GRPO or DAPO, showing significant results on "high-resolution visual search" benchmarks such as V\*.

**Limitations of Prior Work**: Existing research primarily focuses on visual search scenarios where "tools are inevitably useful"—situations where targets are small and must be zoomed in on. However, in broader visual reasoning tasks (e.g., 3D spatial relations, medical VQA), it is often unclear if or when tools should be used. Existing methods exhibit erratic behavior in these "tool-optional" settings: either accuracy increases when ignoring tools, or forcing tool use fails to improve performance.

**Key Challenge**: A fundamental optimization asymmetry exists in tool-optional scenarios. Tool-based rollouts involve more interaction rounds, longer token sequences, and higher variance compared to tool-free paths. Even with token-level loss and over-turn masking, GRPO naturally favors tool-free paths. Yet, tools cannot be deemed useless, as disabling them entirely leads to performance degradation.

**Goal**: (1) Systematically investigate the overlooked regime of "tool-optional + complex visual reasoning"; (2) Identify the essential failure common to both vanilla RFT and tool-encouraging RFT; (3) Provide an intervention that allows RL to truly learn in such settings.

**Key Insight**: The authors move beyond the two-dimensional "tool frequency vs. accuracy" perspective to quantify rollout diversity in two spaces: textual (distinct-n-grams of the pre-grounding `<think>` span) and visual (pairwise mIoU of crop boxes and CLIP alignment with question keywords). They find that diversity collapses in both vanilla and tool-encouraged setups, whereas the initial ~20% of tool invocations provide the diverse exploration history necessary for learning.

**Core Idea**: Treat tools as "training-time scaffolding" rather than inference-time necessities. By ensuring rolls explore diversely during early training (maintaining diversity through tools or active entropy regularization), tools naturally fade in later stages while model capability continues to improve.

## Method

### Overall Architecture
The baseline agent uses Mini-o3 (Qwen2.5-VL-7B-Instruct + SFT + RFT, capable of invoking `<grounding>`), trained on 1.2k 3D spatial reasoning QA pairs from SpatialReasoner. The agent follows a thought–action–observation cycle: generating `<think>` reasoning at each step, then either outputting a `<grounding>` tag with `(bbox_2d, source)` to trigger a zoom-in crop and append observations to the history, or directly providing an `<answer>`. The pipeline follows a "diagnosis then intervention" approach: Sections 3.1–3.3 use four experimental groups—vanilla RFT, tool-banned, tool-encouraging (DeepEyes tool bonus and PixelReasoner curiosity reward)—to diagnose that "tool frequency is decoupled from accuracy, and diversity is the true driver." Section 3.4 introduces adaptive entropy regularization, and Section 3.5 uses tool-banned + entropy regularization to verify that the gains depend on visual exploration provided by tools.

### Key Designs

1.  **Diversity Diagnosis = Textual distinct-n-gram + Visual mIoU/CLIP**:
    - **Function**: Decomposes rollout diversity into two measurable axes to quantitatively prove that "tool frequency $\neq$ exploration breadth."
    - **Mechanism**: On the text side, it counts distinct-$n$-gram ratios ($n\in\{3,4,5,6\}$) for the `<think>` span preceding `<grounding>`. On the visual side, it samples 50 rollouts for the same (image, question) and calculates the mean pairwise IoU (low indicates wide coverage) and CLIP similarity between crop patches and question keywords (high indicates semantic relevance).
    - **Design Motivation**: Tool-encouraging methods use tools ~3x more than vanilla RFT, yet their mIoU remains $>0.55$ (similar to vanilla) and CLIP scores do not improve. This contradicts the intuition that more tool use equals more exploration, identifying "exploration diversity" as the missing component.

2.  **Adaptive Entropy Regularization + Proportional Feedback**:
    - **Function**: Adds an entropy reward term to the GRPO objective, automatically adjusted by the current batch entropy, serving as the sole knob for exploration control.
    - **Mechanism**: The objective is modified to $\mathcal{J}_{\text{ent}}(\theta)=\mathcal{J}_{\text{GRPO}}(\theta)+\lambda_t\cdot\mathbb{E}_{q,\tau}[\bar{\mathcal{H}}(\tau)]$, where $\bar{\mathcal{H}}(\tau)$ is the token-level average entropy of the rollout. To avoid the instability of a fixed $\lambda$, a proportional feedback mechanism is used: $\lambda_t = K_p\,[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$, with $\mathcal{H}_{\text{target}}=0.9$ and $K_p=0.03$. Pressure is applied only when current entropy falls below the target.
    - **Design Motivation**: Directly addresses the root cause: premature rollout collapse. The adaptive rule avoids manual tuning. Consequently, even as tool use drops from ~20% to 3%, validation accuracy on 3DSRBench peaks at 62.9% (vs. 59.2% baseline and 59.9% forced tool use).

3.  **Tool-banned Ablation × Causal Splitting of Entropy Regularization**:
    - **Function**: Separates the "general benefits of entropy regularization" from the "visual scaffolding provided by tools."
    - **Mechanism**: Replicates the strict tool-banned protocol (blocking `<grounding>` during rollout) and adds the same adaptive entropy regularization. It uses over-turn masking to exclude rollouts exceeding the budget from advantage calculations, preventing GRPO from implicitly punishing long rollouts.
    - **Design Motivation**: Tool-banned + entropy regularization only reaches 57.8%, lower than vanilla tool-banned (58.1%) and significantly lower than tool-enabled + entropy (62.9%). This falsifies the idea that entropy is a universal fix; the gains must stem from the visual evidence diversity provided by tools early in training.

### Loss & Training
The base optimizer is DAPO (GRPO + clip-higher + dynamic sampling + token-level policy loss + over-turn masking), trained for $\leq 100$ steps. Two types of comparison rewards are used: DeepEyes tool bonus $R_{\text{DE}}=\mathbb{I}[y=y^*]+\lambda_{\text{tool}}\mathbb{I}[y=y^*]\mathbb{I}[u(\tau)=1]$ and PixelReasoner curiosity reward $R_{\text{PR}}=\mathbb{I}[y=y^*]+\alpha\max(H-\mathrm{RaPR}(q),0)\mathbb{I}[u(\tau)=1]+\beta\,r_{\text{penalty}}(\tau)$, where $\mathrm{RaPR}(q)=\mathbb{E}_\tau[u(\tau)]$ and $r_{\text{penalty}}=\min(N-n_{\text{tool}}(\tau),0)$. Evaluations use Avg@8 with VLM-as-judge (Qwen2.5-VL-7B) for answer extraction.

## Key Experimental Results

### Main Results: 3DSRBench + CV-Bench-3D Overview

| Configuration | Tools? | 3DSRBench Acc | CV-Bench-3D Acc | Tool Use (Init → Saturation) |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL 7B (generalist) | – | 48.4 | 82.9 | – |
| Mini-o3 (zero-shot) | yes | 54.5 | 77.6 | – |
| SpatialReasoner (specialist) | – | 60.3 | 80.3 | – |
| Vanilla RFT | yes | 59.2 | 76.7 | ~20% → ~2% |
| Tool-banned | no | 58.1 | – | 0% → 0% |
| Tool-Encourage (DeepEyes) | yes | 59.9 | 74.5 | ~20% → 100% |
| **Ours (Entropy-Regularized)** | yes | **62.9** | **78.8** | ~20% → ~3% |
| Tool-banned + Entropy | no | 57.8 | – | 0% → 0% |

### Ablation Study: Exploration Diversity

| Method | crop mIoU (↓) | CLIP (↑) | Visual Behavior |
| :--- | :--- | :--- | :--- |
| Vanilla RFT | 0.554 | 0.184 | Highly fossilized |
| Tool-Encourage | 0.557 | 0.187 | Highly fossilized (despite ~3× crops) |
| **Ours (Entropy-Regularized)** | **0.494** | 0.184 | Active exploration / semantic alignment maintained |

VQA-RAD (Medical VQA, OpenThinkIMG heterogeneous toolset) shows an identical trend: Vanilla 46.34 → Tool-Encourage 47.23 → Entropy 48.78.

### Key Findings
- "Tool frequency" and "Accuracy" are nearly orthogonal on 3DSRBench: accuracy increases as tool use collapses to 2%, and forcing 100% tool use only yields a 0.7% gain.
- Diversity is the hidden independent variable: both vanilla and tool-encouraged methods show declining text distinct-n-grams and crop mIoU $>0.55$. Only entropy regularization reduces mIoU to 0.494 while maintaining CLIP scores, leading to superior accuracy.
- The "scaffolding" hypothesis is supported by reciprocal evidence: tool-banned drops to 58.1% (proving early tools are needed), and tool-banned + entropy drops further to 57.8%, proving entropy gains require tools.
- On CV-Bench-3D, vanilla and tool-encouraged RFT perform worse than Mini-o3 base (-0.9 / -3.1), whereas entropy regularization gains +1.2, suggesting extreme collapse or forced use damages general visual understanding.

## Highlights & Insights
- The diagnostic framework decoupling "tool frequency" from "exploration diversity" is elegant. Using textual distinct-n-gram + visual mIoU/CLIP can be applied to monitor any Agent RL project with visual tools.
- The "proportional feedback $\lambda_t = K_p[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$" adaptive entropy regularization is a cost-effective engineering trick—single knob, no tuning required, and avoids repetition collapse.
- The "tools as training-time scaffolding" framing challenges the "train agents to use tools more" paradigm. Tool frequency is not the optimization goal; the breadth of the "experience map" explored during training is. This suggests that explicit tool bonuses in rewards may inadvertently suppress diversity.

## Limitations & Future Work
- The study focuses on 1.2k SpatialReasoner samples. Cross-task generalization relies on VQA-RAD and CV-Bench-3D. Whether "scaffolding" is detrimental in "tool-essential" settings like visual search (V\*) remains unverified.
- Training is limited to 100 steps due to DAPO dynamic sampling constraints; it is unclear if entropy regularization holds or if new collapse modes emerge over longer horizons.
- The findings are tied to Mini-o3 (Qwen2.5-VL-7B). Results for weaker models (that haven't learned tools) or significantly larger models need validation.
- The target entropy $\mathcal{H}_{\text{target}}=0.9$ and $K_p=0.03$ are empirical. Theoretical guidelines based on model size or vocabulary distribution are missing.

## Related Work & Insights
- **Comparison with DeepEyes/Mini-o3/PixelReasoner**: These methods focus on "tool-essential" visual search. This paper moves to "tool-optional" 3D reasoning, revealing that their reward designs collapse or saturate in such scenarios.
- **Comparison with Curiosity/Tool Bonus**: Explicit tool rewards increase frequency but not diversity or accuracy. This work replaces "task-specific" signals with "task-agnostic" entropy pressure.
- **Comparison with SpatialReasoner**: While specialist routes rely on explicit 3D coordinate injection (60.3%), this work achieves 62.9% without specialized spatial supervision by regulating training dynamics.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic quantification of "tool-use collapse" and repositioning tools as training scaffolding is a significant cognitive update for visual agent RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong comparison across five methods, dual-axis diversity diagnosis, and causal ablation of tool-banned × entropy.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative chain from "diagnosis" to "intervention." The "scaffolding" metaphor is highly intuitive.
- Value: ⭐⭐⭐⭐ High practical value for Agent RL practitioners: monitor diversity, and be cautious of tool bonuses that suppress it.

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
## Related Papers

- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)

</div>

<!-- RELATED:END -->
