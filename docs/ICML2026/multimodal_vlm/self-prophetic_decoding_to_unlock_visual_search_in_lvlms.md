---
title: >-
  [Paper Note] Self-Prophetic Decoding to Unlock Visual Search in LVLMs
description: >-
  [ICML 2026][Multimodal VLM][LVLM] SeProD pairs a visual search post-trained LVLM with its un-finetuned pre-trained version, treating the pre-trained model as a "prophet" that generates single-step draft prefixes. The post-trained model selectively accepts these prefixes based on probability thresholds, preserving both single-step fundamental capabiliti
tags:
  - ICML 2026
  - Multimodal VLM
  - LVLM
date: 2026-05-08
content_hash: fbebc9d0a9a228da
---
# Self-Prophetic Decoding to Unlock Visual Search in LVLMs

**Conference**: ICML 2026  
**arXiv**: [2605.28741](https://arxiv.org/abs/2605.28741)  
**Code**: Not yet released  
**Area**: Multimodal VLM  
**Keywords**: Visual Search, LVLM, Prophetic Decoding, Speculative Decoding, Multi-step Reasoning  

## TL;DR
SeProD pairs a visual search post-trained LVLM with its un-finetuned pre-trained version, treating the pre-trained model as a "prophet" that generates single-step draft prefixes. The post-trained model selectively accepts these prefixes based on probability thresholds, preserving both single-step fundamental capabilities and multi-step reasoning coherence without additional training or extra computation.

## Background & Motivation
**Background**: There are currently two paths for equipping LVLMs with "think-as-you-look" visual search capabilities. One is external tool augmentation (SEAL, DyFo, ZoomEye, etc.), which offloads operations like cropping, zooming, and localization to vision experts via function calls. The other is intrinsic capability expansion (Pixel Reasoner, DeepEyes, Mini-o3, etc.), which performs visual search post-training directly on the base model, enabling it to initiate zoom-in and grounding in a single forward pass.

**Limitations of Prior Work**: External tool interfaces are rigid, breaking continuous multi-step reasoning into multiple independent tool calls, which results in context loss. While intrinsic expansion paths appear more elegant, evaluations on models like Mini-o3 reveal specific costs of post-training: grounding single-step accuracy drops by 49.3%, OCR by 2.3%, spatial understanding by 10.9%, and counting by 3.0%. Furthermore, as multi-step trajectories lengthen, early errors propagate; removing irrelevant steps from the context actually improves performance on VisualProbe-test splits by 5.66%/2.24%/5.66%.

**Key Challenge**: Visual search post-training data is limited and primarily relies on RL rewards at the end of trajectories, lacking intermediate supervision. The optimization target biases toward "task completion," causing independent capabilities like grounding, counting, and OCR to interfere with each other or be forgotten. Conversely, without post-training, the model lacks cross-step planning and search initiation capabilities—strong in single steps but weak in multi-step sequences.

**Goal**: To reintegrate the "strong single-step capabilities" preserved in the pre-trained version with the "multi-step search skeleton" gained in the post-trained version, allowing them to mutually calibrate at each step without further training or increasing the inference budget.

**Key Insight**: The authors observe that the post-trained model and its pre-trained base share the same vocabulary and largely similar output distributions. This alignment is sufficient to borrow the speculative decoding paradigm from LLMs—letting a lightweight "draft model" guess while a target model accepts based on probability. The difference here is that the two models function not as "accelerator + main model," but as "single-step expert + multi-step planner."

**Core Idea**: Use the pre-trained LVLM as a "prophet" to continuously generate single-step draft prefixes for the post-trained LVLM. The post-trained model only accepts prefixes where the joint probability exceeds a threshold, thereby grafting single-step capabilities back into multi-step reasoning.

## Method

### Overall Architecture
SeProD couples two models: the post-trained LVLM is the search model, responsible for running multi-round search trajectories; the version of the same base without visual search post-training is the prophet model, independently invoked each round to generate single-step drafts. In round $i$, the search model maintains the full history $H_{i-1}=\{(I,Q),(I_1,C_1),\dots,(I_{i-1},C_{i-1})\}$ and outputs in one of two modes: grounding mode, producing reasoning segment $R_i$ plus a candidate region $G_i$ (zoomed to obtain $I_i$), or answering mode, producing $R_i$ plus the final answer $A_i$. Each search round triggers a prophet call; the prophet views $I_i$ and a mode-specific query $Q^p$, outputting a draft $O_i$ of length $L_d$. This draft is then filtered and "absorbed" by the search model as a prefix for its subsequent tokens. This loop is used only during inference and is plug-and-play for any intrinsic-extended LVLM.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["High-res image I + Question Q"] --> B["Search Model (Post-trained) Round i<br/>Reads history H(i−1), decides mode, generates C_i"]
    B -->|grounding mode| C["Outputs region G_i → Crop/Zoom to get I_i"]
    B -->|answering mode| C2["Enter answer generation (I_i = I(i−1))"]
    C --> P
    C2 --> P
    P["Search–Prophet Pairing & Single-step Focus<br/>Prophet views crop I_i, ignores text C_i"] --> Q2["Grounding Verification & Answer Drafting<br/>Switch query Q^p by mode, autoregress draft O_i"]
    Q2 -->|grounding: query Q^g| G["Verify if ROI contains target<br/>Draft guides next round R(i+1)"]
    Q2 -->|answering: query Q| H["Draft answer O_i for current round A_i"]
    G --> ACC["Probabilistic Threshold Acceptance<br/>s_j = p_s^α · p_p^(1−α), fallback to p_s at first s_j < τ"]
    H --> ACC
    ACC -->|grounding: accept prefix → R(i+1), return| B
    ACC -->|answering: accept & generate A_i| Z["Final Answer"]
```

### Key Designs

**1. Search–Prophet Pairing and Single-step Focus: Separating Planning from Expertise**

Post-training embeds the multi-step search skeleton into the search model but causes degradation in single-step grounding and OCR; the pre-trained base retains these capabilities. SeProD delegates global trajectory task and "where to look" decisions to the search model, while the prophet model focuses solely on the current crop $I_i$ for an "expert" single-step judgment. Crucially, the prophet **does not receive** the search model's text output $C_i$ to avoid being biased by its reasoning, thereby preserving independent single-step expertise.

The authors found that injecting prophet output as a text prompt (akin to tool calling) either fails to impact or breaks reasoning coherence (Appendix Fig. 8). Maintaining independent vision for the prophet allows task-relevant focus and single-step capability to be transmitted through separate channels.

**2. Grounding Verification and Answer Drafting: Mode-aware Drafting**

The search model's output mode dictates the prophet's query $Q^p$. The prophet generates a draft $O_i$ of length $L_d$ according to $p_p(O_i\mid I_i, Q^p)=\prod_j p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})$. Drafts serve different roles:

- **Grounding Mode → Verification Query $Q^g$**: The prophet determines if the target region exists in $I_i$ (true/false). If true, it provides regional details as draft $O_i$, which, once accepted, rewrites the reasoning segment $R_{i+1}$ for the search model’s **next round**. If false, it prompts the search model to re-localize.
- **Answering Mode → Original Query $Q$**: The prophet drafts an answer $O_i$ as a prefix for the search model’s **current round** final answer $A_i$. $A_i$ is produced on-the-fly during the acceptance process, saving one full answer decoding pass.

**3. Probabilistic Threshold Acceptance: Selective Prefix Absorption**

The prophet's draft is not treated as hard external input but as a prefix the search model selectively accepts. Accepted tokens enter the KV cache as if generated by the search model. For each token $o_{i,j}$ in $O_i$, a geometric mean consistency score is calculated:

$$s_j = p_s(o_{i,j}\mid H_i,o_{i,<j})^{\alpha} \cdot p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})^{1-\alpha}$$

where $\alpha$ starts at 0.5 and automatically adjusts based on the token's normalized rank in the search model's logits (higher rank increases $\alpha$, favoring the search model's distribution). Tokens are rejected starting from the first position where $s_j < \tau$, after which sampling reverts to $p_s(x_j\mid H_i,x_{<j})$. Since draft tokens are pre-calculated, all $s_j$ are computed in one parallel forward pass, incurring no additional latency.

This method ensures only tokens residing in the search model's high-likelihood region are absorbed, utilizing prophet knowledge without causing a "personality shift" in multi-step reasoning.

### Loss & Training
SeProD is entirely training-free and introduces no trainable parameters. It uses two hyperparameters: the consistency threshold $\tau$ and an adaptive balance factor $\alpha$. The prophet model defaults to the same base as the search model (e.g., Qwen-2.5-VL-3B).

## Key Experimental Results

### Main Results
Testing on 4 high-resolution visual search benchmarks (12 splits) using Pixel Reasoner and DeepEyes as search backbones with a 3B prophet.

| Benchmark / Split | Pixel Reasoner | + SeProD | DeepEyes | + SeProD |
|-------------------|---------------:|---------:|---------:|---------:|
| VisualProbe-Hard | 28.7 | 30.2 (+1.5) | 38.4 | 41.9 (+3.5) |
| VisualProbe-Medium | 29.0 | 30.4 (+1.4) | 30.5 | 32.3 (+1.8) |
| VisualProbe-Easy | 58.7 | 61.7 (+3.0) | 61.2 | 64.7 (+3.5) |
| V* Bench Overall | 86.9 | 88.5 (+1.6) | 89.0 | 91.1 (+2.1) |
| HR-Bench 4K Overall | 72.6 | 73.6 (+1.0) | 73.0 | 73.8 (+0.8) |
| HR-Bench 8K Overall | 64.3 | 65.1 (+0.8) | 69.9 | 71.9 (+2.0) |

Improvement was observed across all 12 splits. The gains are more significant in scenarios requiring high spatial/instance awareness (e.g., +3.5 on VisualProbe-Hard).

### Ablation Study

| Configuration | Key Observation | Explanation |
|------|---------|------|
| Search only (baseline) | Single-step grounding drops 49.3% | Baseline capability degradation from post-training |
| Prophet as text prompt | Reasoning interrupted, unstable gains | Failed cases in Appendix Fig. 8 |
| Removing irrelevant context | VisualProbe-test splits +5.56/2.24/5.66% | Confirms long-context interference |
| Probabilistic acceptance | Distribution aligns with original search | Maintains multi-step coherence |

### Key Findings
- Injecting prophet outputs as text prompts yields inconsistent gains; a token-level probabilistic interface is necessary to integrate single-step expertise into multi-step trajectories.
- Gains correlate with search difficulty—longer trajectories and stronger spatial sensing tasks (VisualProbe-Hard, HR-Bench 8K) show the largest improvements (+2 to +3.5 points).
- A 3B prophet is sufficient; prefix evaluation is parallelized, resulting in "zero additional computational overhead" relative to standard decoding.

## Highlights & Insights
- SeProD repurposes the "draft + accept" paradigm of speculative decoding from inference acceleration to inference quality. The parallel acceptance mechanism originally intended for speed becomes an interface for "inter-model capability transfer."
- Pairing the post-trained model with its own pre-trained version is a clever engineering insight—the distributions are naturally close, leading to high acceptance rates.
- The shift from text-level to token-level interfaces is a significant contribution, suggesting that LVLM ensembles should favor probabilistic coupling over text-based interaction.

## Limitations & Future Work
- The method requires the search and prophet models to share the same base, making it inapplicable to closed-source LVLMs like GPT-4o.
- Verification is limited to visual search; generalization to multimodal reasoning without explicit zooming (e.g., complex math charts) remains unknown.
- The threshold $\tau$ is fixed; no automated tuning scheme is provided.
- While inference is parallel, memory usage nearly doubles, which increases deployment costs for large (e.g., 70B) backbones.

## Related Work & Insights
- **vs SEAL / DyFo / ZoomEye (External Tools)**: These use text/function interfaces that break multi-step reasoning; SeProD uses a token-level interface to preserve cross-step context.
- **vs DeepEyes / Mini-o3 / Pixel Reasoner (Intrinsic Expansion)**: These models learn to search via RL but lose fine-grained single-step skills; SeProD fixes this post-hoc by pairing them with their pre-trained bases.
- **vs Speculative Decoding (Leviathan et al., 2023)**: Conventional speculative decoding aims for "equivalent output with less computation." SeProD targets "enhanced multi-step reasoning via single-step experts," changing the acceptance criteria from unbiased sampling to probability thresholds.

## Rating
- Novelty: ⭐⭐⭐⭐ Repurposing speculative decoding for capability repair is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Consistent improvements across 12 splits and diagnostic experiments supportive of core claims.
- Writing Quality: ⭐⭐⭐⭐ Clear explanation of why probabilistic interfaces outperform text interfaces.
- Value: ⭐⭐⭐⭐ Training-free and plug-and-play, high industrial utility for existing search-capable models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](../../ICLR2026/multimodal_vlm/self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ICML 2025\] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration](../../ICML2025/multimodal_vlm/towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICML 2026\] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)

</div>

<!-- RELATED:END -->
