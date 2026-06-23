---
title: >-
  [Paper Note] Self-Prophetic Decoding to Unlock Visual Search in LVLMs
description: >-
  [ICML 2026][Multimodal VLM][LVLM] SeProD pairs a post-trained LVLM optimized for visual search with its non-finetuned pre-trained version. The pre-trained model acts as a "prophet," generating single-step draft prefixes at each turn, while the post-trained model selectively accepts these prefixes based on a probability threshold. This approach restores
tags:
  - ICML 2026
  - Multimodal VLM
  - LVLM
date: 2026-05-08
content_hash: 9538f06fb40f1b6d
---
# Self-Prophetic Decoding to Unlock Visual Search in LVLMs

**Conference**: ICML 2026  
**arXiv**: [2605.28741](https://arxiv.org/abs/2605.28741)  
**Code**: Not yet released  
**Area**: Multimodal VLM  
**Keywords**: Visual Search, LVLM, Prophetic Decoding, Speculative Decoding, Multi-step Reasoning  

## TL;DR
SeProD pairs a post-trained LVLM optimized for visual search with its non-finetuned pre-trained version. The pre-trained model acts as a "prophet," generating single-step draft prefixes at each turn, while the post-trained model selectively accepts these prefixes based on a probability threshold. This approach restores single-step foundational capabilities and maintains multi-step reasoning coherence without additional training or extra computational overhead.

## Background & Motivation
**Background**: Equipping LVLMs with "think-and-look" visual search capabilities currently follows two paths. One is external tool enhancement (e.g., SEAL, DyFo, ZoomEye), which externalizes operations like cropping, zooming, and localization to visual experts via function calling. The other is intrinsic capability extension (e.g., Pixel Reasoner, DeepEyes, Mini-o3), which directly performs visual search post-training on base models, allowing them to initiate zoom-in and grounding within a single forward pass.

**Limitations of Prior Work**: The external tool path uses rigid interfaces, breaking continuous multi-step reasoning into independent tool calls and losing context. The intrinsic extension path appears more elegant, but this paper finds specific costs of post-training in models like Mini-o3: single-step grounding accuracy dropped by 49.3%, OCR by 2.3%, spatial understanding by 10.9%, and counting by 3.0%. Furthermore, as multi-step trajectories lengthen, early errors propagate; removing irrelevant steps from the context actually improved VisualProbe-test splits by 5.66%, 2.24%, and 5.66%, respectively.

**Key Challenge**: Visual search post-training relies on limited data and largely uses RL with rewards at the end of trajectories, lacking intermediate supervision. The optimization goal favors "task completion," causing independent capabilities like grounding, counting, and OCR to interfere or be forgotten. Conversely, without post-training, models lack cross-step planning and search initiation capabilities—resulting in strong single-step but weak multi-step performance.

**Goal**: To recombine the "strong single-step capabilities preserved in the pre-trained version" with the "multi-step search framework acquired by the post-trained version" without re-training or increasing the inference budget, allowing them to mutually calibrate at each step.

**Key Insight**: The authors noted that the post-trained model and its pre-trained base share the same vocabulary and largely similar output distributions. This alignment allows borrowing the speculative decoding paradigm from LLMs: letting a lightweight "draft model" guess while the target model accepts tokens based on probability. The difference here is that the two models act not as "accelerator + primary" but as "single-step expert + multi-step planner."

**Core Idea**: Use the pre-trained LVLM as a "prophet" to continuously generate single-step draft prefixes for the post-trained LVLM. The latter only accepts prefixes where the joint probability exceeds a threshold, effectively grafting single-step expertise back into multi-step reasoning.

## Method

### Overall Architecture
SeProD couples two models: the post-trained LVLM, called the **search model**, handles multi-turn search trajectories; the non-finetuned pre-trained version of the same base, called the **prophet model**, is invoked independently at each turn to generate single-step drafts. In turn $i$, the search model maintains the full history $H_{i-1}=\{(I,Q),(I_1,C_1),\dots,(I_{i-1},C_{i-1})\}$ and outputs in one of two modes: **grounding mode**, producing a reasoning fragment $R_i$ and a candidate region $G_i$ (zoomed to get $I_i$), or **answering mode**, producing $R_i$ and the final answer $A_i$. Each search turn triggers a prophet call, where the prophet processes $I_i$ and a mode-specific query $Q^p$ to output a draft $O_i$ of length $L_d$ in one forward pass. This draft is then filtered and "absorbed" by the search model using a probability threshold. The entire loop operates only during inference and is plug-and-play for any intrinsic-extended LVLM.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["High-res Image I + Question Q"] --> B["Search Model (Post-trained) Turn i<br/>Reads History H(i−1), Decides Mode, Generates C_i"]
    B -->|Grounding Mode| C["Output Region G_i → Crop/Zoom to get I_i"]
    B -->|Answering Mode| C2["Enter Answer Generation (I_i = I(i−1))"]
    C --> P
    C2 --> P
    P["Search–Prophet Pairing and Single-step Focusing<br/>Prophet only sees cropped I_i, ignores text C_i"] --> Q2["Grounding Verification and Answer Drafting Prefixes<br/>Switch Query Q^p by Mode, Auto-regressive Draft O_i"]
    Q2 -->|Grounding: Query Q^g| G["Verify if ROI contains target<br/>Draft guides next turn R(i+1)"]
    Q2 -->|Answering: Query Q| H["Draft Answer O_i for current turn A_i"]
    G --> ACC["Probability Threshold Prophetic Acceptance<br/>s_j = p_s^α · p_p^(1−α), reject beyond first s_j < τ"]
    H --> ACC
    ACC -->|Grounding: Accepted prefix → R(i+1), back to next turn| B
    ACC -->|Answering: On-the-fly acceptance for A_i| Z["Final Answer"]
```

### Key Designs

**1. Search–Prophet Pairing and Single-step Focusing: Separating Planning and Expertise**

While post-training embeds the multi-step search framework into the search model, it degrades single-step capabilities like grounding and OCR. The pre-trained base retains these. SeProD divides the labor: the search model manages the global trajectory and cross-step context, deciding "where to look and what to ask," while the prophet model focuses solely on the current crop $I_i$ for an "expert" single-step judgment. Crucially, the prophet **does not receive** the search model's text output $C_i$ to avoid being biased by the search model’s reasoning, thereby preserving its independent single-step capability.

The authors found that feeding the prophet's output back as a text prompt is equivalent to inferior tool-calling—prefixes either have no effect or disrupt reasoning coherence (see failure cases in Appendix Fig. 8). Letting the prophet see images alone while the search model defines the query allows "task-relevant focus" and "independent single-step ability" to be transmitted separately.

**2. Grounding Verification and Answer Drafting Prefixes: Mode-Aware Drafting**

The search model outputs in two modes, prompting the prophet's query $Q^p$ to switch accordingly. The prophet generates a draft $O_i$ of length $L_d$ according to $p_p(O_i\mid I_i, Q^p)=\prod_j p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})$. These drafts serve different purposes in the trajectory:

- **Grounding mode → Verification Query $Q^g$**: The prophet judges if the target area exists within the current crop $I_i$, outputting true/false. If true, it provides details as draft $O_i$, which, once accepted, rewrites the reasoning fragment $R_{i+1}$ for the **next turn** regarding "where to look." If false, it prompts the search model to re-localize.
- **Answering mode → Original Query $Q$**: The prophet directly drafts the answer $O_i$ as a prefix for the search model's **current turn** final answer $A_i$. $A_i$ is not pre-generated but produced on-the-fly during the acceptance process, saving a full answer decoding cycle.

By splitting "guiding the next search step" and "correcting the final answer" into two categories at different points in the trajectory, the prophet's single-step capability is applied precisely.

**3. Probability Threshold Prophetic Acceptance: Absorbing High-Likelihood Tokens**

Prophet drafts are not forced as external inputs but treated as token prefixes that the search model can selectively accept. Accepted portions enter the KV cache as if generated by the search model itself. For each token $o_{i,j}$ in $O_i$, a geometric mean consistency score is calculated:

$$s_j = p_s(o_{i,j}\mid H_i,o_{i,<j})^{\alpha} \cdot p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})^{1-\alpha}$$

where $\alpha$ starts at 0.5 and adjusts automatically based on the token's normalized rank in the search model's logits (higher rank $\rightarrow$ larger $\alpha$, favoring the search model's distribution). Tokens are rejected starting from the first position where $s_j < \tau$, after which sampling reverts to $p_s(x_j\mid H_i,x_{<j})$. All $s_j$ can be computed in parallel in one forward pass (since draft tokens are pre-prepared), making the overhead equivalent to a single standard decoding step.

This joint probability gate ensures that only tokens already within the search model's high-likelihood region are absorbed—leveraging the prophet's knowledge without causing a "personality shift" in reasoning. Figure 2(c) confirms the output distribution curves remain nearly identical.

### Loss & Training
SeProD is entirely training-free and introduces no trainable parameters. Two hyperparameters exist: the consistency threshold $\tau$ controls acceptance strictness, and the balance factor $\alpha$ is adaptive online. The prophet model defaults to the same base as the search model (e.g., Qwen-2.5-VL-3B), though smaller related bases can be used to reduce costs.

## Key Experimental Results

### Main Results
Evaluated on 4 high-resolution visual search benchmarks (12 splits) using Pixel Reasoner and DeepEyes as search backbones, with a 3B prophet.

| Benchmark / Split | Pixel Reasoner | + SeProD | DeepEyes | + SeProD |
|-------------------|---------------:|---------:|---------:|---------:|
| VisualProbe-Hard | 28.7 | 30.2 (+1.5) | 38.4 | 41.9 (+3.5) |
| VisualProbe-Medium | 29.0 | 30.4 (+1.4) | 30.5 | 32.3 (+1.8) |
| VisualProbe-Easy | 58.7 | 61.7 (+3.0) | 61.2 | 64.7 (+3.5) |
| V* Bench Overall | 86.9 | 88.5 (+1.6) | 89.0 | 91.1 (+2.1) |
| HR-Bench 4K Overall | 72.6 | 73.6 (+1.0) | 73.0 | 73.8 (+0.8) |
| HR-Bench 8K Overall | 64.3 | 65.1 (+0.8) | 69.9 | 71.9 (+2.0) |

Improvement was observed across all 12/12 splits, with more significant gains in scenarios where complexity and spatial/cross-instance perception are critical (e.g., +3.5 points on VisualProbe-Hard for DeepEyes). SeProD also shows consistent minor gains on general VQA, with no additional latency due to prefix evaluation parallelization.

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| Search only (baseline) | Single-step grounding dropped 49.3% | Baseline for capability degradation post-training |
| Prophet as text prompt | Interrupted reasoning, unstable gains | Failure cases in Appendix Fig. 8 |
| Removing irrelevant context| VisualProbe-test splits +5.56/2.24/5.66% | Validates long-context interference |
| Prob. Threshold (SeProD) | Dist. nearly identical to search (Fig. 2(c)) | Preserves multi-step coherence |

### Key Findings
- Feeding prophet output as a text prompt provides almost no stable gain; "prefix acceptance" in the probability domain is required to reintegrate single-step capabilities into multi-step trajectories.
- Gains correlate positively with search difficulty—tasks like VisualProbe-Hard and HR-Bench 8K, requiring long trajectories and strong spatial perception, saw the largest gains (+2~+3.5), indicating SeProD compensates for lost fine-grained skills.
- A 3B prophet is sufficient; since prefix evaluation is parallelized, the paper reports "no additional computational overhead" in terms of latency.

## Highlights & Insights
- Transitioning the "draft + acceptance" paradigm of speculative decoding from LLM acceleration to VLM quality is a novel semantic re-use of technology. Parallel acceptance becomes an interface for "cross-modal capability transfer."
- Using the pre-trained version of the same base as the prophet is a brilliant engineering decision—distributions are naturally aligned, leading to higher acceptance rates.
- The authors elevate the "interface design" to a core contribution, explicitly differentiating between token-level probability interfaces and text-prompt interfaces. This suggests that LVLM collaborations should avoid text-only interfaces in favor of token-level coupling.

## Limitations & Future Work
- The method depends on the search and prophet sharing the same base, making it inapplicable to black-box commercial LVLMs (e.g., GPT-4o).
- Only validated on visual search (thinking-with-images); generalization to multimodal reasoning without explicit zoom-in/grounding (e.g., complex math charts) is unknown.
- The threshold $\tau$ is globally fixed; no automated tuning scheme is provided, and optimal values may vary by backbone or benchmark.
- While prophet inference is parallelized, VRAM usage nearly doubles, which might be costly for deploying 70B-class search backbones.

## Related Work & Insights
- **vs SEAL / DyFo / ZoomEye (External Tools)**: Tools use text/function interfaces that break multi-step reasoning; SeProD uses probability interfaces to couple models in the same decoding loop, preserving context.
- **vs DeepEyes / Mini-o3 / Pixel Reasoner (Intrinsic Post-training)**: These works use RL to teach self-initiated search at the cost of single-step degradation; SeProD fixes this post-hoc by re-pairing them with their pre-trained bases.
- **vs Speculative Decoding (Leviathan et al., 2023)**: Speculative decoding uses smaller models to accelerate larger ones for "equivalent output with less compute." SeProD uses the same parallel mechanism for "enhancing multi-step logic with single-step experts," shifting from unbiased sampling to probability thresholds.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-purposing speculative decoding for quality repair is novel; same-base pairing is a key insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Consistent gains across 12 splits with diagnostic experiments on degradation and context interference.
- Writing Quality: ⭐⭐⭐⭐ Clearly explains why text interfaces fail and probability ones work; Fig. 2 is very persuasive.
- Value: ⭐⭐⭐⭐ Training-free and plug-and-play; offers nearly zero-cost enhancement for existing visual search models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[ICML 2025\] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration](../../ICML2025/multimodal_vlm/towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](../../ICLR2026/multimodal_vlm/self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ICML 2026\] LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation](language_bias_in_lvlms_from_in-depth_analysis_to_simple_and_effective_mitigation.md)
- [\[CVPR 2026\] Revisiting Visual Corruptions in LVLMs: A Shape-Texture Perspective on Model Failures](../../CVPR2026/multimodal_vlm/revisiting_visual_corruptions_in_lvlms_a_shape-texture_perspective_on_model_fail.md)

</div>

<!-- RELATED:END -->
