---
title: >-
  [Paper Note] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents
description: >-
  [ICML 2026][LLM Reasoning][RFT] In "tool-optional" visual agent tasks such as 3D spatial reasoning, authors found that vanilla RFT causes tool calling rates to collapse to near zero, while explicitly encouraging tool use yields only marginal gains. The true driver of performance is the exploration diversity of rollouts. By employing adaptive entropy
tags:
  - ICML 2026
  - LLM Reasoning
  - RFT
date: 2026-05-08
content_hash: a7f0727ca6ddcdfd
---
# Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents

**Conference**: ICML 2026  
**arXiv**: [2606.00096](https://arxiv.org/abs/2606.00096)  
**Code**: https://scaffolded-exploration.github.io  
**Area**: LLM Reasoning / Visual Agent / Reinforcement Learning  
**Keywords**: Visual Chain-of-Thought, Tool Use Collapse, Entropy Regularization, RFT, Exploration Diversity

## TL;DR
In "tool-optional" visual agent tasks such as 3D spatial reasoning, authors found that vanilla RFT causes tool calling rates to collapse to near zero, while explicitly encouraging tool use yields only marginal gains. The true driver of performance is the exploration diversity of rollouts. By employing adaptive entropy regularization, 3DSRBench accuracy is improved from 59.2% to 62.9%, repositioning tools as "training-time scaffolding" rather than inference-time necessities.

## Background & Motivation

**Background**: Current visual agents (DeepEyes, Mini-o3, PixelReasoner, Chain-of-Focus, etc.) sequence visual tools like `<grounding>`, cropping, and segmentation into a visual chain-of-thought. These agents are fine-tuned using group-based RL (RFT) like GRPO/DAPO, achieving significant results on benchmarks like V* that require high-resolution visual search.

**Limitations of Prior Work**: Existing research almost exclusively focuses on visual search scenarios where "tools are essential"—targets are small and must be zoomed in to be seen. However, in broader visual reasoning tasks (3D spatial relationships, medical VQA), the necessity and timing of tool use are less clear. Existing methods exhibit strange behavior in these "tool-optional" settings: either accuracy increases despite not using tools, or accuracy fails to increase even when tool use is forced.

**Key Challenge**: An asymmetric optimization exists in tool-optional scenarios. Tool-based rollouts involve more interaction turns, longer token counts, and higher variance. Even with token-level losses and over-turn masking, GRPO naturally favors tool-free paths. Yet, one cannot simply conclude that "tools are useless," as disabling tools entirely leads to performance degradation.

**Goal**: (1) Systematically investigate the neglected regime of "optional tools + complex visual reasoning"; (2) Identify the essential failure common to vanilla RFT and tool-encouraging RFT; (3) Provide an intervention to enable RL to learn effectively in this setting.

**Key Insight**: Moving beyond the 2D perspective of "tool frequency vs. accuracy," the authors quantify rollout diversity in both text space (distinct-n-grams of the pre-grounding `<think>` span) and visual space (pairwise mIoU of crop boxes for the same query + CLIP alignment with question keywords). They found that both vanilla and tool-encouraging paths monotonically collapse, while the initial ~20% tool usage provided diverse exploration history.

**Core Idea**: Treat tools as "training-time scaffolding" rather than inference-time requirements. By ensuring sufficiently diverse exploration early in training (through tools or active entropy regularization), tools can naturally fade in later stages while the model remains strong.

## Method

### Overall Architecture
The study investigates how RFT should handle visual tools in complex "tool-optional" visual reasoning tasks. The base agent is Mini-o3 (Qwen2.5-VL-7B-Instruct + SFT + RFT, capable of calling `<grounding>`), trained on 1.2k 3D spatial reasoning QA pairs from SpatialReasoner. The agent follows a thought–action–observation loop: generating a `<think>` reasoning step, followed by either a `<grounding>` tag with `(bbox_2d, source)` for a zoom-in crop (appended to history) or a final `<answer>`. The research follows a "diagnosis–intervention–causal verification" chain: diagnosing the decoupling of tool frequency and accuracy via comparisons, intervening with adaptive entropy regularization, and causally proving that tools provide the necessary visual exploration for entropy gains.

### Key Designs

**1. Diversity Metrics: Decoupling "Exploration Breadth" from "Tool Frequency"**

The first step is identifying a measurable variable separate from "tool calling rate." Rollout diversity is split into text and visual axes. On the text side, only the `<think>` segment before `<grounding>` is analyzed. Since uncertainty primarily resides in reasoning spans rather than bbox coordinates, the ratio of distinct-$n$-grams for $n\in\{3,4,5,6\}$ is calculated for reasoning text. On the visual side, 50 rollouts are sampled per (image, question). Analysis includes the mean pairwise IoU of all crop boxes (lower indicates broader coverage) and CLIP similarity between crop patches and question keywords (higher indicates relevancy). These metrics revealed that tool-encouraging RFT, despite doubling tool usage, failed to improve mIoU or CLIP alignment compared to vanilla RFT, shifting the focus to "exploration diversity."

**2. Adaptive Entropy Regularization: Preventing Premature Rollout Collapse**

Targeting the root cause of collapse, an entropy reward term is added to the GRPO objective:

$$\mathcal{J}_{\text{ent}}(\theta)=\mathcal{J}_{\text{GRPO}}(\theta)+\lambda_t\cdot\mathbb{E}_{q,\tau}[\bar{\mathcal{H}}(\tau)],$$

where $\bar{\mathcal{H}}(\tau)$ is the token-level average entropy of the rollout. To solve the sensitivity of $\lambda$, a proportional feedback control is used: $\lambda_t = K_p\,[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$, with $\mathcal{H}_{\text{target}}=0.9$ and $K_p=0.03$. This design exerts pressure only when current entropy is below the target. This single-knob design increased 3DSRBench accuracy to 62.9%, significantly higher than the 59.2% baseline and 59.9% forced tool use, while tool usage still naturally decreased from ~20% to 3%.

**3. Tool-banned × Entropy Regularization: Isolating "Entropy Gain" vs. "Tool Gain"**

To verify if tools are necessary for entropy regularization to work, a tool-banned protocol was used ($<grounding>$ tokens masked during rollout) while applying entropy regularization. Tool-banned + entropy regularization only reached 57.8%, lower than the vanilla tool-banned score of 58.1% and far below the tool-enabled 62.9%. This confirms that entropy regularization requires the visual diversity provided by early tool usage to be effective, supporting the "tools as scaffolding" hypothesis.

### Loss & Training
The base optimizer is DAPO (GRPO + clip-higher + dynamic sampling + token-level policy loss + over-turn masking), trained for $\le 100$ steps. Two reward types were compared: DeepEyes tool bonus $R_{\text{DE}}=\mathbb{I}[y=y^*]+\lambda_{\text{tool}}\mathbb{I}[y=y^*]\mathbb{I}[u(\tau)=1]$ and PixelReasoner curiosity reward $R_{\text{PR}}=\mathbb{I}[y=y^*]+\alpha\max(H-\mathrm{RaPR}(q),0)\mathbb{I}[u(\tau)=1]+\beta\,r_{\text{penalty}}(\tau)$. Evaluation uses Avg@8 with Qwen2.5-VL-7B as a judge.

## Key Experimental Results

### Main Results: 3DSRBench + CV-Bench-3D Overview

| Config | Tool? | 3DSRBench Acc | CV-Bench-3D Acc | Tool Usage (Init→Sat.) |
|------|-------|---------------|-----------------|---------------------|
| Qwen2.5-VL 7B (Generalist) | – | 48.4 | 82.9 | – |
| Mini-o3 (Zero-shot) | Yes | 54.5 | 77.6 | – |
| SpatialReasoner (Specialist) | – | 60.3 | 80.3 | – |
| Vanilla RFT | Yes | 59.2 | 76.7 | ~20% → ~2% |
| Tool-banned | No | 58.1 | – | 0% → 0% |
| Tool-Encourage (DeepEyes) | Yes | 59.9 | 74.5 | ~20% → 100% |
| **Ours (Entropy-Reg.)** | Yes | **62.9** | **78.8** | ~20% → ~3% |
| Tool-banned + Entropy | No | 57.8 | – | 0% → 0% |

### Ablation Study: Exploration Diversity

| Method | Crop mIoU (↓) | CLIP (↑) | Visual Behavior |
|------|---------------|----------|----------|
| Vanilla RFT | 0.554 | 0.184 | Highly Stereotyped |
| Tool-Encourage | 0.557 | 0.187 | Highly Stereotyped (despite ~3× crops) |
| **Entropy-Regularized** | **0.494** | 0.184 | Active Exploration, maintained alignment |

On VQA-RAD (Medical VQA): Vanilla 46.34 → Tool-Encourage 47.23 → Entropy 48.78, showing consistent trends.

### Key Findings
- "Tool frequency" and "accuracy" are nearly orthogonal on 3DSRBench: tool use collapsed from 20% to 2% while accuracy grew; forcing 100% usage only yielded a 0.7 point gain.
- Diversity is the hidden variable: both vanilla and tool-encouraging RFT saw text distinct-n-grams drop, while crop mIoU stayed $>0.55$. Only entropy regularization reduced mIoU to 0.494 while maintaining CLIP alignment.
- The "Scaffolding" hypothesis is supported by: (1) Tool-banned accuracy falling to 58.1%, showing early tool use is essential; (2) Tool-banned + entropy falling to 57.8%, showing entropy gains require tools.
- On CV-Bench-3D, both vanilla RFT and tool-encouragement performed worse than the base Mini-o3, while entropy regularization improved it, suggesting extremes harm general visual understanding.

## Highlights & Insights
- The diagnosis framework decoupling "tool usage" and "exploration diversity" (text distinct-n-gram + visual mIoU + CLIP) is highly effective for monitoring agent RL.
- Adaptive entropy regularization via proportional feedback is an inexpensive engineering trick that avoids manual tuning and prevents repetition collapse.
- The "tools as training-time scaffolding" framing challenges the "training agents = training tool usage" paradigm. It suggests the "experience map" covered during training rollouts is more important than inference-time tool frequency.
- The tool-banned + entropy regularization ablation experiment provides a clean causal separation between the benefits of tools and the benefits of entropy.

## Limitations & Future Work
- The study primarily focused on 1.2k SpatialReasoner samples; whether "scaffolding" is harmful in tasks where tools are strictly required (like V*) remains unverified.
- Training was limited to 100 steps due to DAPO wall-clock time constraints; long-term entropy stability and potential new collapse modes are unknown.
- The base model was limited to Qwen2.5-VL-7B; performance with weaker or larger base models requires further validation.
- The entropy target $\mathcal{H}_{\text{target}}=0.9$ and $K_p=0.03$ are empirical; a theoretical guide for these values relative to model size is missing.

## Related Work & Insights
- **Comparison with DeepEyes/Mini-o3/PixelReasoner**: These methods focus on visual search in "tool-essential" benchmarks. This work reveals that their reward designs collapse or saturate in "tool-optional" tasks and provides a cross-scenario intervention.
- **Comparison with SpatialReasoner**: While the specialist approach uses explicit 3D coordinate supervision for 60.3%, this RL approach reaches 62.9% without specialized spatial supervision, suggesting exploration control can replace domain inductive biases.

## Rating
- Novelty: ⭐⭐⭐⭐ The "tool use collapse" phenomenon is systematically named and quantified; "tools as scaffolding" is a significant cognitive update for visual agent RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad comparisons, dual-axis diversity diagnosis, and causal ablations are provided across multiple tasks, though constrained by training steps and base model variety.
- Writing Quality: ⭐⭐⭐⭐ Clear metaphors like "scaffolding" and a logical narrative chain from diagnosis to causal verification.
- Value: ⭐⭐⭐⭐ Practical takeaways for tool-augmented RL: monitor diversity instead of frequency and be cautious of task-agnostic tool bonuses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)

</div>

<!-- RELATED:END -->
