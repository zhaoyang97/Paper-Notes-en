---
title: >-
  [Paper Note] Self-Prophetic Decoding to Unlock Visual Search in LVLMs
description: >-
  [ICML 2026][Multimodal VLM][Visual Search] SeProD pairs a post-trained visual-search LVLM with its original un-finetuned pre-trained version. By treating the pre-trained model as a "prophet" to generate single-step draft…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Search"
  - "LVLM"
  - "Prophetic Decoding"
  - "Speculative Decoding"
  - "Multi-step Reasoning"
date: 2026-05-08
content_hash: 9bcd3ccbd6a5b692
---

# Self-Prophetic Decoding to Unlock Visual Search in LVLMs

**Conference**: ICML 2026  
**arXiv**: [2605.28741](https://arxiv.org/abs/2605.28741)  
**Code**: Not yet released  
**Area**: Multimodal VLM  
**Keywords**: Visual Search, LVLM, Prophetic Decoding, Speculative Decoding, Multi-step Reasoning  

## TL;DR
SeProD pairs a post-trained visual-search LVLM with its original un-finetuned pre-trained version. By treating the pre-trained model as a "prophet" to generate single-step draft prefixes, which are then selectively accepted by the post-trained model via a probability threshold, it preserves both fundamental single-step capabilities and multi-step reasoning coherence without additional training or extra computational overhead.

## Background & Motivation
**Background**: There are currently two paths to equip LVLMs with "thinking-while-looking" visual search capabilities. One is external tool augmentation (SEAL, DyFo, ZoomEye, etc.), where operations like cropping, zooming, and localization are outsourced to visual experts via function calls. The other is intrinsic capability extension (Pixel Reasoner, DeepEyes, Mini-o3, etc.), which performs visual search post-training directly on the base model, allowing it to initiate zoom-in and grounding operations within a single forward pass.

**Limitations of Prior Work**: The external tool path uses rigid interfaces, breaking continuous multi-step reasoning into multiple independent tool calls, which loses context. The intrinsic extension path appears more elegant, but the paper identifies specific costs of post-training in models like Mini-o3: single-step grounding accuracy dropped by 49.3%, OCR by 2.3%, spatial understanding by 10.9%, and counting by 3.0%. Furthermore, as multi-step trajectories lengthen, early errors propagate; experimental results showed that removing irrelevant steps from the context improved VisualProbe-test scores across three splits by 5.66%/2.24%/5.66%.

**Key Challenge**: Visual search post-training uses limited data and relies heavily on RL rewards at the end of trajectories, lacking intermediate supervision signals. Optimization targets favor "task completion," causing independent capabilities like grounding, counting, and OCR to interfere with one another or be forgotten. Conversely, without post-training, the model lacks the ability for cross-step planning and initiating searches—resulting in strong single-step but weak multi-step performance.

**Goal**: Without retraining or increasing the推理 budget, re-integrate the "strong single-step capabilities" retained by the pre-trained version with the "multi-step search skeleton" acquired by the post-trained version, allowing them to mutually calibrate at each step.

**Key Insight**: The authors observe that the post-trained model and its pre-trained base share the same vocabulary and largely similar output distributions. This alignment is sufficient to borrow the paradigm of LLM speculative decoding—letting a lightweight "draft model" guess first, followed by the target model accepting based on probability. The difference here is that the two models function as a "single-step expert + multi-step planner" rather than "accelerator + main model."

**Core Idea**: Use the pre-trained LVLM as a "prophet" to continuously generate single-step draft prefixes for the post-trained LVLM. The post-trained model only accepts prefixes where the joint probability exceeds a threshold, thereby grafting single-step expertise back into multi-step reasoning.

## Method

### Overall Architecture
SeProD couples a pair of models: the post-trained LVLM is the **search model**, responsible for multi-round search trajectories; the un-finetuned version of the same base is the **prophet model**, which is invoked independently each round to generate single-step drafts. In round $i$, the search model maintains the full history $H_{i-1}=\{(I,Q),(I_1,C_1),\dots,(I_{i-1},C_{i-1})\}$ and outputs in one of two modes: **grounding mode**, producing a reasoning segment $R_i$ plus a candidate region $G_i$ (zoomed to $I_i$); or **answering mode**, producing $R_i$ plus the final answer $A_i$. Every search round triggers a prophet call; the prophet views $I_i$ and a mode-specific query $Q^p$, generating a draft $O_i$ of length $L_d$. This draft is filtered and "absorbed" by the search model as a prefix for its subsequent tokens based on a probability threshold. The entire loop is training-free and plug-and-play for any intrinsic-extended LVLM.

### Key Designs

1.  **Search-Prophet Pairing and Single-step Focus**:
    *   **Function**: Decouples the multi-step search skeleton from fundamental single-step capabilities—the search model manages the global trajectory and cross-step context, while the prophet focuses on "expert-level" judgment on the current crop $I_i$.
    *   **Mechanism**: The prophet does not receive the search model's textual output $C_i$ to avoid being biased by search traces. The query $Q^p$ switches based on the search mode—using a grounding verification query $Q^g$ ("Is the target region in this crop, and where?") or drafting an answer from the original $Q$. The prophet generates autoregressively: $p_p(O_i\mid I_i, Q^p)=\prod_j p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})$.
    *   **Design Motivation**: The authors found that feeding prophet outputs as text prompts acts like a tool-calling interface, causing prefixes to either fail or break coherence. Allowing the prophet to look at the image independently while the search model decides "what to ask" separates task relevance from independent single-step expertise.

2.  **Probabilistic Threshold Prophetic Acceptance**:
    *   **Function**: Treats the prophet-generated $O_i$ as acceptable token prefixes rather than external input; accepted tokens enter the KV cache as if generated by the search model.
    *   **Mechanism**: For each token $o_{i,j}$ in $O_i$, a geometric mean consistency score is calculated: $s_j = p_s(o_{i,j}\mid H_i,o_{i,<j})^{\alpha} \cdot p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})^{1-\alpha}$. $\alpha$ is initialized at 0.5 and adaptively adjusted based on the token's normalized rank in the search model's logits (higher rank increases $\alpha$, favoring the search distribution). Tokens are rejected starting from the first position where $s_j < \tau$. Consistency scores are computed in parallel in one forward pass.
    *   **Design Motivation**: Unlike direct output overwriting, probability-based acceptance ensures only tokens within the search model's high-likelihood region are absorbed, preserving the native distribution and preventing "personality shifts" in multi-step reasoning.

3.  **Grounding Verification and Answer Drafting Prefixes**:
    *   **Function**: Maps two types of prophet outputs to different positions in the search trajectory—grounding results as prefixes for the next reasoning segment $R_{i+1}$, and answer drafts as prefixes for the current answer $A_i$.
    *   **Mechanism**: In grounding mode, the prophet responds with true/false and region details. If accepted, this prefix rewrites the search model's reasoning on "where to look next." In answering mode, the prophet drafts an answer, and the search model generates $A_i$ while simultaneously accepting the draft, saving decoding steps.
    *   **Design Motivation**: Separating "next-step guidance" from "final answer correction" allows the prophet's single-step expertise to act at the appropriate temporal locations within the search trajectory.

### Loss & Training
SeProD is entirely training-free and introduces no trainable parameters. Two hyperparameters are used: the consistency threshold $\tau$ and the adaptive balance factor $\alpha$. The prophet model defaults to the same base as the search model (e.g., Qwen-2.5-VL-3B), though smaller homologous bases are also supported to reduce overhead.

## Key Experimental Results

### Main Results
SeProD was evaluated using Pixel Reasoner and DeepEyes as search backbones across 12 splits of 4 high-resolution visual search benchmarks, with a 3B prophet.

| Benchmark / Split | Pixel Reasoner | + SeProD | DeepEyes | + SeProD |
|-------------------|---------------:|---------:|---------:|---------:|
| VisualProbe-Hard | 28.7 | 30.2 (+1.5) | 38.4 | 41.9 (+3.5) |
| VisualProbe-Medium| 29.0 | 30.4 (+1.4) | 30.5 | 32.3 (+1.8) |
| VisualProbe-Easy | 58.7 | 61.7 (+3.0) | 61.2 | 64.7 (+3.5) |
| V* Bench Overall | 86.9 | 88.5 (+1.6) | 89.0 | 91.1 (+2.1) |
| HR-Bench 4K Overall| 72.6 | 73.6 (+1.0) | 73.0 | 73.8 (+0.8) |
| HR-Bench 8K Overall| 64.3 | 65.1 (+0.8) | 69.9 | 71.9 (+2.0) |

Gains were observed across all 12 splits. Improvements were most significant in scenarios with higher difficulty or where spatial/instance-level perception is critical (e.g., +3.5 points for DeepEyes on VisualProbe-Hard). Small but consistent gains were also noted in general VQA.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Search only (baseline) | Single-step grounding dropped 49.3% | Baseline capability degradation from post-training. |
| Prophet as text prompt | Reasoning interrupted, unstable gains | Failed cases discussed in Appendix Fig. 8. |
| Removing irrelevant context | VisualProbe-test splits rose 2.24-5.66% | Validates long-context interference. |
| Probabilistic threshold (SeProD) | Output distribution matches search model (Fig. 2c) | Preserves multi-step coherence. |

### Key Findings
- Feeding prophet output as a text prompt provides unstable gains; "prefix acceptance" in the probability domain is essential to return single-step expertise to the multi-step trajectory.
- Gains correlate positively with search difficulty—maximal gains (+2 to +3.5) occur in long-trajectory, spatially intensive scenarios like VisualProbe-Hard and HR-Bench 8K.
- With a 3B prophet and parallel prefix evaluation, SeProD introduces no perceived extra computational latency.

## Highlights & Insights
- Transitioning the "draft+accept" paradigm of speculative decoding from LLM acceleration to VLM quality is a novel semantic reinterpretation of existing technology.
- Pairing a post-trained model with its own pre-trained version is a brilliant engineering decision—the distributions are naturally aligned, leading to high acceptance rates.
- Elevating "interface design" as a core contribution—the authors demonstrate why token-level probability interfaces outperform text-prompt interfaces, suggesting token-level coupling is a more robust direction for LVLM collaboration.

## Limitations & Future Work
- The method depends on the search and prophet models sharing the same base, making it inapplicable to black-box proprietary LVLMs (e.g., GPT-4o).
- Validated only on "thinking-with-images" visual search; generalizability to multimodal reasoning without explicit zoom/grounding (e.g., complex math charts) is unknown.
- The threshold $\tau$ is globally fixed; no automatic tuning scheme is provided.
- While inference is parallel, VRAM usage nearly doubles, which is non-trivial for 70B-class search backbones.

## Related Work & Insights
- **vs SEAL / DyFo / ZoomEye (External Tools)**: Tools use text/function interfaces that segment multi-step reasoning; SeProD uses a token-level probability interface to couple models within the same decoding loop, preserving context.
- **vs DeepEyes / Mini-o3 / Pixel Reasoner (Intrinsic Extensions)**: These teach models to initiate search via RL at the cost of single-step degradation; SeProD repairs this degradation post-hoc by pairing them with pre-trained bases.
- **vs Speculative Decoding (Leviathan et al., 2023)**: Traditionally used for speed with "equivalent output + less compute"; SeProD uses the same mechanism for "multi-step enhancement via single-step experts," shifting the criterion to a probability threshold.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Repurposing speculative decoding for quality repair in VLM reasoning is a fresh perspective; same-base pairing is a key insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Consistent improvements across 4 benchmarks and 12 splits, supported by diagnostic experiments on degradation and context interference.
- **Writing Quality**: ⭐⭐⭐⭐ The transition from "why text fails" to "why probability works" is clear, and Fig. 2 is highly persuasive.
- **Value**: ⭐⭐⭐⭐ Training-free and plug-and-play; offers near-zero-cost integration for existing visual search models with high industry applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](../../ICLR2026/multimodal_vlm/self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[NeurIPS 2025\] Visual Structures Help Visual Reasoning: Addressing the Binding Problem in LVLMs](../../NeurIPS2025/multimodal_vlm/visual_structures_helps_visual_reasoning_addressing_the_binding_problem_in_vlms.md)
- [\[ICML 2026\] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)

</div>

<!-- RELATED:END -->
