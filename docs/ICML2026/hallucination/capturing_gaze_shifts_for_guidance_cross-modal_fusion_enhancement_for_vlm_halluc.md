---
title: >-
  [Paper Note] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation
description: >-
  [ICML 2026][Hallucination Detection][VLM Hallucination Mitigation] The GIFT method is proposed, which constructs a visual saliency map by tracking positive changes in visual attention ("gaze shifts") when a VLM understan…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "VLM Hallucination Mitigation"
  - "Attention Shift"
  - "Cross-Modal Fusion"
  - "Visual Saliency"
  - "Inference-time Intervention"
date: 2026-05-08
content_hash: c9b07c021f7cac81
---

# Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation

**Conference**: ICML 2026  
**arXiv**: [2510.22067](https://arxiv.org/abs/2510.22067)  
**Code**: https://github.com/amazon-science/GIFT  
**Area**: Hallucination Detection  
**Keywords**: VLM Hallucination Mitigation, Attention Shift, Cross-Modal Fusion, Visual Saliency, Inference-time Intervention  

## TL;DR
The GIFT method is proposed, which constructs a visual saliency map by tracking positive changes in visual attention ("gaze shifts") when a VLM understands user queries, and enhances both visual and query token attention during the decoding phase to maintain cross-modal fusion balance. It achieves up to a 20.7% improvement on CHAIR with only a 1.13× latency increase.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) have made significant progress in tasks such as visual question answering and image description, but they remain prone to hallucinations—generating content that cannot be supported by text or visual input. This poses a serious threat in high-risk areas such as medicine, autonomous driving, and robotics.

**Limitations of Prior Work**: Research has found that hallucinations primarily stem from VLMs' over-reliance on language priors while ignoring visual inputs. Existing inference-time mitigation methods are mainly divided into three categories: contrastive decoding (requires generating contrastive distributions, high computational cost), visual input modification (requires additional forward passes), and attention guidance (e.g., VAF, which amplifies visual token attention proportional to attention scores). However, these methods have two key flaws: first, they ignore the **visual attention sink** problem, where attention is continuously allocated to task-irrelevant visual regions; second, they only enhance visual attention without adjusting query token attention, leading to **cross-modal fusion imbalance**.

**Key Challenge**: Simply amplifying visual token attention simultaneously intensifies attention on incorrect regions and weakens the model's understanding of user queries, failing to resolve the three problems of attention sinks, insufficient visual contribution, and cross-modal fusion imbalance concurrently.

**Goal**: Design an inference-time method capable of (1) precisely locating task-relevant visual regions and filtering attention sink noise; (2) simultaneously enhancing visual and query token attention to maintain cross-modal fusion balance.

**Key Insight**: Inspired by human vision—humans dynamically shift their "gaze" to capture relevant visual information while reading a question. When a VLM processes information-rich words in a query (nouns, verbs, adjectives, etc.), visual attention undergoes positive changes. Tracking this "gaze shift" can naturally filter out attention sink noise (as attention change in irrelevant regions is very small).

**Core Idea**: Construct a pre-computed saliency map by tracking positive changes in visual attention during query processing, and use this map to simultaneously guide the enhancement of visual and query token attention during decoding.

## Method

### Overall Architecture
GIFT consists of two stages: (1) the pre-filling stage, which tracks positive changes in visual attention ("gaze shifts") while the model processes the user query to construct a visual saliency map; (2) the decoding stage, which utilizes the saliency map to enhance both visual and query token attention in key cross-modal fusion layers. The input is a standard VLM triplet (system instruction $X_S$, visual tokens $X_V$, query tokens $X_T$), and the output is the hallucination-mitigated generated text.

### Key Designs

1.  **Gaze Shift Tracking for Saliency Map Construction**:
    - **Function**: Constructs a task-relevant visual saliency map during the pre-filling stage while naturally mitigating the attention sink problem.
    - **Mechanism**: First, information-rich words are extracted from the query using spaCy POS tagging (NOUN/VERB/ADJ/ADV/NUM/PROPN), corresponding to token set $X_{Tr}$. Then, in each layer, the top-50% attention heads $\hat{\mathcal{H}}^l_{TrV}$ with the largest cumulative positive attention changes are selected. The positive change in visual attention is calculated as $\Delta \mathbf{A}^l_{h,i,j} = \max(\mathbf{A}^l_{h,i,j} - \mathbf{A}^l_{h,i-1,j}, 0)$. The saliency map $\hat{\mathcal{S}^l}$ is obtained by averaging across selected heads and informative tokens, followed by min-max normalization. Finally, the layer with the largest total positive change is selected as the final saliency map.
    - **Design Motivation**: Tracking only positive changes rather than average attention is effective because irrelevant regions produce almost no change in attention, thereby naturally filtering attention sink noise. Experimental verification shows that the normalized saliency score of the "shift" method is 11.92, which is 2.2 times that of the "static" method (5.40).

2.  **Saliency-Guided Visual Attention Enhancement**:
    - **Function**: Enhances the attention weights of task-relevant visual tokens during the decoding phase.
    - **Mechanism**: In the selected enhancement layers $\mathcal{L}$, for the top-50% visual attention heads $\mathcal{H}^l_{OV}$, the attention is multiplied by a saliency guidance factor $\hat{\bm{A}}^l_{h,-1,j} = \bm{A}^l_{h,-1,j} \cdot \exp(\alpha \hat{\mathcal{S}}_j)$, where $\alpha$ is a scaling factor. The saliency map is truncated at 3 standard deviations before normalization to avoid over-focusing.
    - **Design Motivation**: Using a pre-computed saliency map instead of current-step attention scores provides a global visual saliency view based on the complete query context, avoiding attention sink noise in step-by-step attention scores.

3.  **Cross-Modal Fusion Balance**:
    - **Function**: Proportionally boosts query token attention while enhancing visual attention to prevent cross-modal fusion imbalance.
    - **Mechanism**: The visual attention enhancement ratio $r^l = \sum_{h,j} \hat{\bm{A}}^l_{h,-1,j} / \bm{A}^l_{h,-1,j}$ is calculated. Then, the query token attention in query attention heads $\mathcal{H}^l_{OT}$ is multiplied by $\beta r^l$ (default $\beta$ is 1.0), and finally, the entire attention matrix is normalized. Enhancement layers are selected by analyzing the ratio of output tokens' attention to visual and query tokens—layers where the two show consistent trends are chosen.
    - **Design Motivation**: Enhancing only visual attention weakens query understanding, causing the model to focus on the correct region but misunderstand the question. Scalable query attention maintains the original cross-modal balance.

## Key Experimental Results

### Main Results

| Model | Method | CHAIR $C_s$↓ | CHAIR $C_i$↓ | POPE F1↑ | POPE Acc↑ | MMHal Hal↓ | MMHal Score↑ |
|-------|--------|-------------|-------------|----------|----------|------------|-------------|
| LLaVA-1.5 7B | Greedy | 50.2 | 15.4 | 82.4 | 79.5 | 65.2 | 2.22 |
| LLaVA-1.5 7B | VAF | 49.6 | 14.3 | 81.0 | 77.2 | 66.3 | 2.16 |
| LLaVA-1.5 7B | VCD | 52.2 | 16.3 | 80.9 | 77.7 | 60.5 | 2.37 |
| LLaVA-1.5 7B | **GIFT** | **39.8** | **10.6** | **83.8** | **81.9** | **57.3** | **2.48** |
| Qwen2-VL 7B | Greedy | 24.8 | 9.1 | 86.0 | 86.5 | 32.7 | 3.53 |
| Qwen2-VL 7B | **GIFT** | **21.2** | **7.7** | **86.8** | **86.9** | **27.5** | **3.58** |
| Qwen3-VL 8B | Greedy | 51.4 | 10.6 | 88.9 | 88.5 | 28.3 | 4.80 |
| Qwen3-VL 8B | **GIFT** | **49.4** | **9.3** | **89.1** | **88.7** | **26.4** | **4.84** |

### Ablation Study

| Model | Configuration | MMHal Hal↓ | MMHal Score↑ | POPE F1↑ | POPE Acc↑ |
|-------|---------------|------------|-------------|----------|----------|
| LLaVA-1.5 7B | Inc. V. (Visual Only) | 60.8 | 2.36 | 82.3 | 79.3 |
| LLaVA-1.5 7B | Cal. V. (Distribution Calib.) | 61.5 | 2.32 | 82.4 | 79.5 |
| LLaVA-1.5 7B | **GIFT (Full)** | **57.3** | **2.48** | **83.8** | **81.9** |
| Qwen2-VL 7B | Inc. V. | 35.2 | 3.41 | 85.3 | 86.0 |
| Qwen2-VL 7B | Cal. V. | 31.9 | 3.56 | 85.8 | 86.4 |
| Qwen2-VL 7B | **GIFT (Full)** | **27.5** | **3.58** | **86.8** | **86.9** |

### Key Findings
- Both components (visual attention enhancement + cross-modal fusion balance) are indispensable; the full GIFT improves performance by up to 25.4% compared to visual-only enhancement.
- GIFT's performance on general vision-language benchmarks (MME, SEED) is comparable to greedy decoding, while several baseline methods show performance degradation.
- GIFT's latency is only 1.13× that of greedy decoding, much lower than VCD (1.99×) and VAR (11.10×).
- Enhancement layer selection is robust; multiple mid-range configurations yield good results.
- LLM-based informative word extraction is more accurate than POS tagging (MMHal hallucination rate 52.6% vs. 57.3%) but at a higher computational cost.

## Highlights & Insights
- The "gaze shift" analogy is very clever: using the **positive change** of attention instead of absolute values to build a saliency map essentially replaces static signals with differential signals, naturally filtering attention sink noise without extra modules.
- The design of cross-modal fusion balance is worth emulating: enhancing only visual attention disrupts the visual-textual attention ratio. Synchronous scaling of query attention is a simple but often overlooked critical step. This "synchronous scaling" idea can be transferred to any attention intervention method requiring multimodal balance.
- The choice of saliency map extraction layers is an intrinsic model property rather than data-dependent: peak layers remain consistent across different datasets and random seeds, suggesting that VLM visual information integration occurs at fixed network depths.

## Limitations & Future Work
- When the query is irrelevant to the image or most of the query content is non-visual, gaze shifts might produce inaccurate saliency maps.
- POS tagging is not precise enough for extracting informative words; LLM-based methods are better but increase computational overhead. A lightweight classifier might be a compromise.
- Validated only on LLaVA-1.5 and Qwen series, not covering other architectures like InternVL.
- Applicability to multi-turn dialogue scenarios has not been explored; the quality of saliency maps under more complex query contexts needs verification.

## Related Work & Insights
- **VAF** (ClearSight): Amplifies visual token attention proportional to current-step attention scores but ignores attention sinks and cross-modal balance.
- **VAR**: Identifies visual sink tokens and redistributes attention but does not solve the issue of insufficient visual contribution and has a latency of up to 11.10×.
- **VCD**: Compares output distributions of original and perturbed visual inputs; requires generating contrastive outputs with a latency of 1.99×.
- The attention sink phenomenon is found in LLMs, ViTs, and VLMs. The gaze shift mechanism in this paper provides a new strategy to circumvent it.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/hallucination/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[ICML 2026\] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling](tag_tangential_amplifying_guidance_for_hallucination-resistant_sampling.md)
- [\[CVPR 2026\] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](../../CVPR2026/hallucination/mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)
- [\[ICML 2026\] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)
- [\[AAAI 2026\] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](../../AAAI2026/hallucination/pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)

</div>

<!-- RELATED:END -->
