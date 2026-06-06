---
title: >-
  [Paper Note] CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Reasoning] Inspired by Baddeley's Working Memory Theory, CSMR transforms "when to introduce visual evidence into reasoning" into a dynamic decision process. The LLM maintains the re…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Reasoning"
  - "Working Memory Theory"
  - "Dynamic Visual Evidence Acquisition"
  - "Perception-Reasoning Decoupling"
  - "Zero-Shot"
date: 2026-05-08
content_hash: 797c30be8a459ddd
---

# CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.28160](https://arxiv.org/abs/2605.28160)  
**Code**: https://github.com/YangZhang2511/CSMR  
**Area**: Multimodal VLM / Multimodal Reasoning / Tool-use  
**Keywords**: Multimodal Reasoning, Working Memory Theory, Dynamic Visual Evidence Acquisition, Perception-Reasoning Decoupling, Zero-Shot  

## TL;DR
Inspired by Baddeley's Working Memory Theory, CSMR transforms "when to introduce visual evidence into reasoning" into a dynamic decision process. The LLM maintains the reasoning state and calls an independent perception module (VLM) to fetch visual evidence on demand until sufficient evidence is obtained. This addresses the flaws of two existing paradigms (pre-reasoning textualization losing details and unified VL space being contaminated by language priors), outperforming baselines zero-shot on multiple multimodal reasoning benchmarks.

## Background & Motivation

**Background**: There are two major paradigms in multimodal reasoning: (a) pre-reasoning visual-to-text (e.g., DDCoT, converting images to captions before reasoning), and (b) unified vision-language space (e.g., CCoT, ICoT, AIMCoT, where VLMs perform end-to-end reasoning).

**Limitations of Prior Work**: (a) Static textualization occurs before reasoning without foresight into later required details; coarse-grained captions irreversibly lose fine-grained information. (b) In the unified paradigm, visual representations are contaminated by language priors—substantial evidence (Section 4.2 of the paper) shows that self-attention systematically assigns higher attention to text tokens (approximately $2.5\times$), which is further amplified by soft-max, resulting in long-term suppression of visual tokens.

**Key Challenge**: The timing of visual evidence introduction determines reasoning quality. One-time early introduction misses details, while continuous unified processing yields to language dominance. An "on-demand evidence acquisition" mechanism is required to judge whether to look at the image, where to look, and whether enough has been seen based on the current reasoning state.

**Goal**: (1) Analyze the issue of language prior dominance in unified paradigms; (2) Enable the LLM to maintain reasoning states and dynamically schedule visual evidence acquisition; (3) Decouple the perception-reasoning structure to avoid visual representation contamination; (4) Outperform baselines in zero-shot settings.

**Key Insight**: Drawing from Baddeley's Working Memory Theory—comprising a central executive that schedules the visuospatial sketchpad and the phonological loop. The LLM acts as the central executive maintaining the reasoning state, while an independent VLM serves as the visuospatial sketchpad, returning textualized visual evidence on demand.

**Core Idea**: A decoupled structure consisting of a Cognitive Reasoning Core (CRC, LLM) and a Perception-Verification Pipeline (PVP, VLM). The CRC determines when and what to check, while the PVP independently inspects the original image to answer queries. Visual evidence is iteratively acquired, driven by the reasoning state, until it is sufficient to terminate.

## Method

### Overall Architecture

The CRC (LLM) maintains a reasoning state (containing the original question + a list of acquired visual evidence). At each step:
1. It decides whether more visual evidence is needed.
2. If so, it generates a targeted visual query (e.g., "What color is at the bottom right of the image?").
3. It calls the PVP (an independent VLM focusing on the original image) which returns a textualized answer.
4. It integrates the evidence into the state and updates the reasoning.
5. If the reasoning is sufficient, it outputs the final answer directly.

The PVP does not participate in reasoning and only performs question answering; its visual representations are not influenced by the language context of the CRC.

### Key Designs

1. **Decoupling of Perception and Reasoning Architecture (PVP independent of CRC's language context)**:

    - **Function**: Prevents visual representations from being contaminated by language priors during the reasoning process.
    - **Mechanism**: The PVP is an independent VLM instance that only receives the original image + visual query (without the CRC's reasoning context) for each call. After the CRC receives the textualized answer, it integrates it into the reasoning without allowing the query context to flow back into subsequent PVP calls.
    - **Design Motivation**: The unified paradigm allows LLM reasoning to gradually dominate visual representations (Figure 2 in the paper shows a $2.5\times$ attention bias). Decoupling allows the PVP to view the image "freshly" each time, unoccupied by language contamination along the reasoning path.

2. **Reasoning-State-Driven Dynamic Visual Querying**:

    - **Function**: The CRC determines when to call the PVP and what query to generate based on the current reasoning state.
    - **Mechanism**: Instead of pre-planning all queries, the CRC generates them incrementally based on the current reasoning state. If the returned evidence is insufficient, the PVP can be called repeatedly; queries can range from coarse to fine (e.g., asking about "main content" before "detailed regions").
    - **Design Motivation**: Pre-reasoning textualization is a one-shot process that misses details. Dynamic acquisition allows "reasoning to guide perception"—fetching only the evidence truly needed for reasoning to avoid interference from irrelevant information.

3. **Confidence-Based Early Termination**:

    - **Function**: Stops directly when the reasoning state is sufficient to support an answer, avoiding unnecessary queries.
    - **Mechanism**: At each step, the CRC evaluates whether the current evidence is sufficient to derive an answer. If yes, it generates the final answer; otherwise, it continues querying. This is similar to entropy-based early stopping in PathCTM.
    - **Design Motivation**: Different cases vary in difficulty—simple questions may need only 1-2 queries, while complex ones require multiple rounds. A fixed number of queries is wasteful and insufficient for long-chain reasoning; adaptive termination balances accuracy and efficiency.

### Quantitative Evidence of Attention Bias (Figure 2)

Mean attention across 35 layers was measured on the ScienceQA subset using Qwen3-VL-8B:
- Average text token attention is $2.5\times$ higher than visual token attention.
- The proportion of visual tokens is further compressed after soft-max.
- Consistent phenomena were observed on LLaVA-1.6-7B, proving this is a systemic issue of the VLM paradigm rather than a single model quirk.

## Key Experimental Results

### Main Results (Multiple Benchmarks, Zero-Shot)

| Benchmark | Pre-reason (DDCoT) | Unified (CCoT) | Unified (AIMCoT) | **CSMR** |
|------|------------|----------|--------|--------|
| ScienceQA | 72.4 | 75.8 | 77.3 | **80.6** |
| A-OKVQA | 56.7 | 58.9 | 60.4 | **63.8** |
| MMStar | 39.5 | 41.2 | 42.8 | **45.7** |
| MMBench-Reasoning | 52.1 | 54.6 | 56.0 | **59.3** |
| RealWorldQA | 45.3 | 47.8 | 49.2 | **52.6** |

CSMR consistently leads by 3-4 points across 5 benchmarks, showing significant advantages particularly in tasks requiring fine-grained visual verification (ScienceQA, A-OKVQA).

### Ablation Study

| Configuration | ScienceQA |
|------|---------|
| Full CSMR (**Ours**) | 80.6 |
| − Early Termination (Fixed query count) | 78.4 |
| − Perception Decoupling (PVP accepts CRC context) | 76.9 |
| − Dynamic Querying (One-time planning of all queries) | 75.2 |
| Revert to pre-reason DDCoT | 72.4 |

All three modules contribute positively; perception decoupling contributes the most ($-3.7$), proving the actual existence of language contamination.

### Early Termination Efficiency

| Difficulty | Avg. Query Count | Accuracy |
|------|------------|------|
| Easy | 1.4 | 87% |
| Medium | 2.7 | 79% |
| Hard | 4.2 | 64% |

The model identifies difficulty and adaptively allocates the number of queries.

### Key Findings
- **Perception-Reasoning Decoupling is critical**: It contributes the most in ablation studies ($-3.7$), confirming the language contamination issue in unified paradigms.
- **Dynamic querying far outperforms pre-planning**: Planning all queries at once leads to a 5.4 point drop.
- **Early termination saves queries without losing accuracy**: Fixed query counts lead to a 2.2 point drop, proving the effectiveness of adaptive termination.
- **Cross-architecture versatility**: CRC can be replaced with GPT-4 / Claude / Qwen-LLM, and PVP can be replaced with LLaVA / Qwen-VL, offering flexible combinations.

## Highlights & Insights
- **Engineering inspired by Cognitive Science**: Baddeley's Working Memory Theory provides a clear division of roles (central executive vs. visuospatial sketchpad) for LLM-VLM collaboration, rather than arbitrary architectural design.
- **"Perception-Reasoning Decoupling" is a true paradigm innovation**: Previous assumptions favored unification; Ours proves that decoupling + dynamic calling is superior, challenging the default "end-to-end is better" assumption.
- **Quantitative evidence of $2.5\times$ attention bias**: Transformed the "feeling" of language prior contamination into a metric, providing a benchmark for future research.
- **Training-free + Modular**: Both CRC and PVP can be independently replaced or upgraded, making it friendly for industrial deployment; new generations of LLMs/VLMs can be directly swapped in.

## Limitations & Future Work
- Multiple rounds of queries accumulate tokens; context grows rapidly in long-chain reasoning—query summarization or graph-based structures could be considered.
- PVP textualization of visual evidence may still lose information—returning structured outputs (coordinates, bounding boxes) instead of free text could be beneficial.
- The CRC's decision on when to call the PVP is a zero-shot prompted behavior; a learnable scheduling strategy might be more stable.
- Total latency = LLM reasoning + multiple VLM calls, which is unfriendly for latency-sensitive scenarios.
- Complex collaboration between CRC and PVP (e.g., PVP actively suggesting focus points) has not been explored.

## Related Work & Insights
- **vs. DDCoT (pre-reasoning textualization)**: That method performs a one-time captioning; CSMR performs dynamic multiple checks.
- **vs. CCoT / AIMCoT (unified VL reasoning)**: Those suffer from language contamination; CSMR solves this via decoupling.
- **vs. ReAct / Toolformer**: Those treat tools as external APIs; CSMR treats the VLM as a "perception tool," sharing the same logic but focusing on visual perception.
- **vs. PathCTM**: PathCTM uses multi-scale reasoning + early stopping; CSMR uses tool calling + early stopping; both share the principle of "evidence acquisition on demand," but PathCTM is internal multi-scale while CSMR uses an external VLM.
- **Insight**: Re-examine "unified end-to-end"—tasks requiring different capabilities like perception-reasoning / retrieval-generation / computation-verification can all consider a decoupling + scheduling mode.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Perception-reasoning decoupling + dynamic visual querying is a true paradigm-level innovation; the cognitive science mapping provides a theoretical foundation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 5 benchmarks + detailed ablation + quantitative attention bias; lacks comparison with ReAct-style tool-use baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Comparison of the three paradigms is clear (Figure 1), with Figure 2 providing decisive evidence of attention bias.
- **Value**: ⭐⭐⭐⭐ Training-free + modular + multi-benchmark SOTA; applicable to all tasks requiring fine-grained visual verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VisionPulse: Dynamic Visual Sparsification in Multimodal Reasoning](visionpulse_dynamic_visual_sparsity_for_efficient_multimodal_reasoning.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[CVPR 2026\] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](../../CVPR2026/multimodal_vlm/adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)
- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](../../ICLR2026/multimodal_vlm/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
