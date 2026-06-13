---
title: >-
  [Paper Note] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning
description: >-
  [ICML 2026][Multimodal VLM][Visual Reasoning] Addressing the counter-intuitive phenomenon where "explicit visual grounding hampers CoT reasoning…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Reasoning"
  - "CoT"
  - "Reinforcement Learning"
  - "GRPO"
  - "Consistency Reward"
date: 2026-05-08
content_hash: 2ca4d2f23883e37f
---

# iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.31096](https://arxiv.org/abs/2605.31096)  
**Code**: https://visual-ai.github.io/ivgr/ (Project Page)  
**Area**: Multimodal VLM  
**Keywords**: Visual Reasoning, CoT, Reinforcement Learning, GRPO, Consistency Reward

## TL;DR
Addressing the counter-intuitive phenomenon where "explicit visual grounding hampers CoT reasoning," the authors propose iVGR—a dual-stream GRPO training framework. It allows textual CoT and grounded CoT (with boxes) to rollout simultaneously, using a consistency reward to "internalize" the visual localization capabilities of high-quality grounded trajectories into pure textual CoT. This enables the model to reap the benefits of grounded reasoning during inference without actually outputting coordinates.

## Background & Motivation

**Background**: In high-resolution fine-grained VQA, standard textual CoT often misses small objects. Consequently, the community has branched into two visually grounded CoT approaches: **crop-tool-based streams** like DeepEyes/PixelReasoner, which call cropping tools during inference to view local regions; and **explicit box-based streams** like TreeVGR/GRIT, which force the model to intersperse bounding box coordinates within the CoT. Both types are trained via RL (e.g., GRPO) and claim to enhance fine-grained perception.

**Limitations of Prior Work**: The authors conducted a set of counter-intuitive experiments: using models specifically trained for grounded CoT, such as DeepEyes-7B and TreeVGR-7B, and only modifying the inference prompt to run standard textual CoT. Results across eight benchmarks (V*, HR4K, HR8K, MME-RW-Lite, etc.) showed that textual CoT actually yielded higher average scores (DeepEyes 75.1 vs. 74.1, TreeVGR 75.7 vs. 74.7). Further analysis by IoU buckets revealed that the crop-tool-based stream only outperformed textual CoT during high-quality localization ($\mathrm{IoU}>0.5$), while the box-based stream was outperformed by textual CoT at all IoU intervals.

**Key Challenge**: Explicit grounding during inference simultaneously undertakes two tasks—"accurate localization" and "final answering"—which compete for token budget and attention. When localization is suboptimal, incorrect coordinates or crops pollute the final answer, leading to more harm than good.

**Goal**: Decouple "leveraging grounding supervision during training" from "mandatory grounding output during inference." The aim is to extract visual priors from grounded CoT during training while allowing the model to run only pure textual CoT during inference, utilizing localization capabilities "silently."

**Key Insight**: Since textual CoT already performs better at inference, the training direction should be "ensuring the textual CoT still knows the target's location during generation," rather than "forcing it to write coordinates explicitly." This is equivalent to giving the textual rollout an additional reward in RL training based on whether it "looks at the same place" as a grounded "teacher."

**Core Idea**: Roll out two parallel streams—a grounded stream (forced boxes) and a textual stream (pure text)—within the GRPO framework. An LLM-scored **consistency reward** is used to align the textual stream with high-quality grounded streams, internalizing visual localization into pure textual CoT.

## Method

### Overall Architecture
iVGR performs GRPO post-training on Qwen2.5-VL / Qwen3-VL. For each query $q$, the policy $\pi_\theta$ samples $N$ rollouts using two different system prompts, resulting in a grounded set $\mathcal{O}^b$ and a textual set $\mathcal{O}^t$. Both sets independently calculate rewards and group-wise normalized advantages $\mathcal{A}^b, \mathcal{A}^t$ to update the policy. The critical coupling point is the **consistency reward** $R_{\text{consistency}}$ added to the textual stream reward. Its "teacher" is a high-quality trajectory selected from the grounded stream within the same batch (refined by a cross-step Rollout Archive to preserve historical best teachers), thereby embedding localization capability into the textual stream. At inference, only the textual stream is executed, with an optional tool-assisted test-time scaling workflow to fuse multi-view information.

### Key Designs

1.  **Dual-stream GRPO Training**:
    - **Function**: Enables a single policy to learn two reasoning paradigms simultaneously and complete knowledge transfer via a shared backbone.
    - **Mechanism**: The grounded stream follows the format constraints of TreeVGR `<think>...</think><answer>...</answer>`, with rewards $R^b_i = R_{\text{format}} + R_{\text{acc}} + R_{\text{box}}$, where $R_{\text{box}}=\tfrac{1}{2}\big(\tfrac{1}{|\mathcal{B}_{\text{gt}}|}\sum_{b}\mathrm{MaxIoU}(b,\mathcal{B}_{\text{pred}})+\tfrac{1}{|\mathcal{B}_{\text{pred}}|}\sum_{\hat{b}}\mathrm{MaxIoU}(\hat{b},\mathcal{B}_{\text{gt}})\big)$ is a bidirectional IoU match considering both recall and precision. The textual stream removes box supervision, with rewards $R^t_i = R_{\text{format}} + R_{\text{acc}} + R_{\text{consistency}}$. Both streams use group normalization for advantages and are updated via GRPO.
    - **Design Motivation**: Training only textual CoT lacks visual supervision; training only grounded CoT makes localization a "must-do," interfering with answering during inference. Parallel dual-streams allow the same policy to "train localization while training coordinate-free reasoning," using consistency rewards for cross-stream transfer.

2.  **Consistency Reward + Rollout Archive**:
    - **Function**: Selects grounded trajectories that "accurately target objects" as teachers to reward the textual stream for learning the same visual focus.
    - **Mechanism**: Reference screening is performed first—a grounded rollout must satisfy $R_{\text{format}}=1$, $R_{\text{acc}}=1$, and $R_{\text{box}}>\tau$ to qualify as a teacher. To counter quality fluctuations within a batch caused by policy updates, a per-query Rollout Archive is maintained, updated via $\mathcal{Z}_{\text{archive}}^{(q)}\leftarrow\arg\max_{z\in\{\mathcal{Z}_{\text{archive}}^{(q)}, o_{\text{best}}^{b}\}} R_{\text{box}}(z)$ to retain the historical best. An external LLM (Qwen2.5-72B) acts as a judge, scoring across four tiers: $\alpha=1.0$ (complete consistency) / $0.7$ (unidirectional deviation) / $0.3$ (both missing and extra) / $0.0$ (direct contradiction). Final reward is $R_{\text{consistency}}=\alpha$ (0 if archive is empty).
    - **Design Motivation**: IoU cannot be used directly as cross-stream supervision because textual streams do not output coordinates. The consistency reward cleverly converts "where to look" into "what is described," allowing the LLM judge to transform visual alignment into semantic alignment, which is fully differentiable (in the GRPO sense) for natural language outputs. The archive addresses the non-stationarity of reinforcement learning where "teacher quality progresses from low to high," preventing early models from being locked by incorrect supervision.

3.  **Tool-Assisted Test-Time Scaling**:
    - **Function**: Recovers the model's box outputs for multi-view fusion while retaining the "textual CoT by default" at inference.
    - **Mechanism**: The model first generates a grounded CoT with boxes to extract all predicted bounding boxes. A crop tool then generates local views, while a **minimal bounding rectangle (union crop)** covering all boxes is constructed to preserve relative spatial relationships. Finally, the original image, local crops, and the union crop are combined as multi-view inputs for standard textual CoT answering.
    - **Design Motivation**: Dual-stream training naturally preserves the model's capability to output boxes when prompted. When details are crucial, grounding acts as a free visual router. Using the union crop to maintain spatial context elegantly absorbs the advantages of tool-based streams (DeepEyes/PixelReasoner) without replacing the core mechanism.

### Loss & Training
Both streams independently perform group normalization and calculate PPO surrogate losses under GRPO, which are then summed for backpropagation. $\tau$ controls the threshold for grounded teacher selection, and $N$ is the number of rollouts per stream per query. The archive persists across steps to ensure the stability of the consistency reward.

## Key Experimental Results

### Main Results
iVGR-Qwen2.5-VL-7B is compared against open-source general MLLMs and various grounded reasoning methods across fine-grained VQA (V*, HR4K, HR8K, MME-RW-Lite) and general VQA tasks.

| Model | Tools | V* | HR4K | HR8K | MME-RW-Lite | POPE | RWQA | CV-2D | CV-3D |
|-------|-------|----|------|------|-------------|------|------|-------|-------|
| Qwen2.5-VL-7B Baseline | ✗ | 78.5 | 69.0 | 65.1 | 44.5 | 86.3 | 68.1 | 75.7 | 73.6 |
| DeepEyes-7B | ✓ | 82.7 | 75.1 | 72.6 | 53.2 | 87.7 | 69.4 | 75.0 | 77.3 |
| TreeVGR-7B | ✗ | 83.8 | 77.1 | 73.1 | 54.9 | 87.3 | 67.3 | 76.6 | 77.2 |
| Thyme-7B | ✓ | 82.2 | 77.0 | 72.0 | 55.2 | 86.8 | 70.2 | 78.0 | 75.1 |
| **iVGR-Qwen2.5-VL-7B** | ✗ | **86.4** | **78.3** | **75.5** | **55.6** | **88.9** | 68.6 | **78.4** | — |

iVGR outperforms grounded models of similar size on most fine-grained tasks. Without external tools, pure textual CoT inference improves V* from 78.5 to 86.4.

### Counter-intuitive Comparisons
One of the most convincing supportive experiments: evaluating existing grounded models by switching prompts to run textual CoT.

| Model | CoT Mode | V* | HR4K | HR8K | MME-RW-Lite | Avg |
|-------|----------|----|------|------|-------------|-----|
| DeepEyes-7B | grounded (G, w/ crop) | 82.7 | 75.1 | 72.6 | 53.2 | 74.1 |
| DeepEyes-7B | textual (T) | 81.7 | 74.9 | 73.1 | 53.5 | **75.1** |
| TreeVGR-7B | grounded (G, w/ box) | 83.8 | 77.1 | 73.1 | 54.9 | 74.7 |
| TreeVGR-7B | textual (T) | 84.3 | 76.9 | 74.7 | 54.7 | **75.7** |

The textual mode average is higher, directly falsifying the implicit assumption that "explicit grounding is mandatory during inference," which serves as the experimental cornerstone for iVGR.

### Key Findings
- Grounded CoT by IoU bucket: DeepEyes only outperforms textual CoT when $\mathrm{IoU}>0.5$; TreeVGR is outperformed at all intervals. This indicates that "explicit coordinates" are not the source of gain; **"visual priors learned during training" are**.
- iVGR-Qwen2.5-VL-7B matches or exceeds tool-based models like DeepEyesV2-7B / Mini-o3-7B / Thyme-7B without tools, proving the efficiency of internalization (no tool calls, fewer tokens).
- Gains in general VQA (POPE/CV-Bench) suggest that the "visual focus" learned via consistency rewards does not compromise general reasoning.

## Highlights & Insights
- The "counter-intuitive analysis" of this work is robust: it overturns consensus through controlled experiments (grounded CoT < textual CoT at inference), explains why using IoU buckets, and then derives the method. This scientific narrative can be applied to other designs that "seem useful but have negative side effects."
- The consistency reward converts visual alignment into semantic alignment judged by an LLM, bypassing the issue of textual streams being unable to calculate IoU. This trick is applicable to any RL distillation scenario where the "teacher has structured output but the student has natural language," such as aligning textual planners with grounded code planners.
- Rollout Archive is a practical pattern for handling non-stationary teachers in GRPO training, allowing "teacher quality to improve along with the policy" without explicit warmup or two-stage training.

## Limitations & Future Work
- Consistency rewards depend on an external 72B LLM as a judge, introducing training costs and potential judge bias. The degradation when using smaller judge models is not quantified.
- Dual-stream rollout doubles training compute; its feasibility for larger scales (32B+) remains to be verified.
- Alignment of "where it looked/what it thought" occurs only at the natural language level, lacking direct visual attention supervision. When objects are tiny but descriptions are similar (e.g., different ROIs in medical imaging), the judge might give inflated scores.
- The disclosed improvement from tool-assisted test-time scaling is limited. Deciding when to activate tool streams still relies on empirical scheduling rather than a learned policy.

## Related Work & Insights
- **vs. DeepEyes / PixelReasoner / Mini-o3**: These use crop-tool-based streams. iVGR defaults to no-tool inference but retains tool capability as an optional test-time scaling, offering lower engineering costs and decoupled training/inference.
- **vs. TreeVGR / GRIT**: These force box insertion into CoT. iVGR uses grounding rewards during training but does not mandate coordinates at inference, treating them as "auxiliary training tasks" to offload cognitive burden.
- **vs. DeepSeek-R1 / GRPO series**: iVGR is a multi-stream extension of GRPO, introducing "cross-stream consistency rewards" and "cross-step teacher archiving," serving as a template for "Multi-task / Multi-modal GRPO."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "internalization over explicit output" perspective is unique among grounded reasoning work, backed by a disruptive experimental setup.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive across eight benchmarks with multiple backbones and IoU bucket comparisons, though lacking robustness ablation for the consistency judge model.
- Writing Quality: ⭐⭐⭐⭐⭐ A coherent flow from "counter-intuitive observation → hypothesis → method → verification."
- Value: ⭐⭐⭐⭐⭐ Dual-stream GRPO + consistency reward is a transferable RL template with implications for agent training and multimodal reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](../../CVPR2026/multimodal_vlm/deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)
- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[ACL 2026\] Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention](../../ACL2026/multimodal_vlm/towards_visually_grounded_multimodal_summarization_via_cross-modal_transformer_a.md)

</div>

<!-- RELATED:END -->
