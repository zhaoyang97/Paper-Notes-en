---
title: >-
  [Paper Note] Mitigating Object Hallucinations via Sentence-Level Early Intervention
description: >-
  [ICCV 2025][Multimodal VLM][MLLM hallucination] This paper proposes SENTINEL, a framework grounded in the key observation that hallucinations emerge early in generation and propagate forward. By combining in-domain candidate bootstrapping with dual-detector cross-validation to construct sentence-level preference data, and employing Context-aware DPO (C-DPO) for early intervention, SENTINEL reduces hallucinations on Object HalBench by 92% while preserving general capabilities.
tags:
  - ICCV 2025
  - Multimodal VLM
  - MLLM hallucination
  - preference learning
  - early intervention
  - DPO
  - object detection verification
date: 2026-05-08
content_hash: 7095e5e66ee0a60a
---

# Mitigating Object Hallucinations via Sentence-Level Early Intervention

**Conference**: ICCV 2025
**arXiv**: [2507.12455](https://arxiv.org/abs/2507.12455)
**Code**: [GitHub](https://github.com/pspdada/SENTINEL)
**Area**: Multimodal VLM
**Keywords**: MLLM hallucination, preference learning, early intervention, DPO, object detection verification

## TL;DR
This paper proposes SENTINEL, a framework grounded in the key observation that hallucinations emerge early in generation and propagate forward. By combining in-domain candidate bootstrapping with dual-detector cross-validation to construct sentence-level preference data, and employing Context-aware DPO (C-DPO) for early intervention, SENTINEL reduces hallucinations on Object HalBench by 92% while preserving general capabilities.

## Background & Motivation

1. **Background**: Multimodal large language models (MLLMs) have achieved remarkable progress in cross-modal understanding, yet hallucination—generating content inconsistent with the visual input—remains a central challenge.
2. **Limitations of Prior Work**: Existing mitigation approaches suffer from three categories of problems: (1) decoding strategies (VCD, OPERA, DoLa) introduce additional inference overhead and latency; (2) preference alignment methods rely on large proprietary models (e.g., GPT) or human annotation, incurring high costs; (3) output rewriting methods introduce distributional mismatch between training data and the model's original outputs.
3. **Key Challenge**: Hallucination severity increases with the length of generated text—earlier sentences contain fewer hallucinations, while subsequent sentences exhibit progressively more. Early intervention is critical, yet existing methods do not explicitly exploit this temporal propagation property.
4. **Goal**: To achieve efficient early intervention against MLLM hallucinations without relying on external large models or introducing distributional shift.
5. **Key Insight**: Perform multiple sampling rounds from the model itself → cross-validate with object detectors → label each sentence as hallucinated or non-hallucinated → construct in-domain preference pairs → train with DPO.
6. **Core Idea**: Leverage the model's own in-distribution sampled outputs; apply dual-detector cross-validation to assign sentence-level hallucination labels; perform preference learning intervention at the position where hallucinations first appear.

## Method

### Overall Architecture
SENTINEL proceeds in six steps: (1) sample multiple in-domain candidates conditioned on an image, prompt, and context $c$; (2) extract all mentioned objects from each generated sentence; (3) cross-validate object existence using two open-vocabulary detectors; (4) categorize sentences as hallucinated or non-hallucinated; (5) append verified non-hallucinated sentences to the context to guide subsequent outputs; (6) fine-tune the model with the C-DPO loss.

### Key Designs

1. **In-domain Candidate Bootstrapping**:
    - **Function**: Obtain positive and negative samples for preference learning from the model's own sampled outputs, without external models.
    - **Mechanism**: Perform $n$ sampling-decoding passes on the current model, stopping after each sentence is generated. Nouns are extracted using SceneGraphParser and then cross-validated by two open-vocabulary detectors—GroundingDINO and YOLO World. If both confirm absence → hallucination; both confirm presence → factual; conflicting → uncertain (discarded to reduce detector bias).
    - **Design Motivation**: Positive and negative samples originate from the same model distribution, preserving stylistic consistency and linguistic structure. Dual-detector cross-validation is more reliable than a single detector, as confirmed by ablation studies.

2. **Context-aware Preference Data Construction**:
    - **Function**: Build preference pairs enriched with contextual information, supporting iterative bootstrapping.
    - **Mechanism**: Positive samples are divided into **context-consistent positives** $y_w^+$ (objects mentioned in the context) and **context-independent positives** $y_w^-$ (objects not mentioned in the context). Only $y_w^+$ is used as the positive sample, as its stronger contextual association enhances the model's context coherence. **Iterative Context Bootstrapping (ICB)** is adopted: each round appends $y_w^+$ to the context as $c_{i+1} = c_i + y_w^+$, and sampling and preference pair construction continue under the updated context, ensuring data coverage over progressively complex contexts.
    - **Design Motivation**: Context-consistent positives carry richer contextual signals, helping the model maintain coherence and prioritize salient content. ICB ensures the preference data is representative across diverse contexts, enhancing generalization.

3. **Context-aware DPO (C-DPO)**:
    - **Function**: Conduct sentence-level preference learning focused on the position where hallucinations first appear.
    - **Mechanism**: Standard DPO is modified into a context-aware variant by conditioning on context $c$ as an additional input. The model is trained to maximize the probability of generating context-consistent positive samples $y_w^+$ and minimize the probability of hallucinated negatives $y_l$. By focusing on the sentence where hallucinations first appear, the approach implements early intervention to prevent subsequent propagation.
    - **Design Motivation**: Observations confirm that eliminating hallucinated objects in the second sentence leads to a substantial reduction in hallucination probability in the third sentence. Intervening at the first occurrence is the most efficient strategy.

### Loss & Training
The C-DPO loss shares the same structure as standard DPO but is additionally conditioned on context $c$. Data construction is iterative via ICB, and training requires only a single epoch. No external model rewriting or human annotation is needed.

## Key Experimental Results

### Main Results

| Model | Method | Object HalBench Resp.↓ | AMBER Hal.↓ | VQAv2↑ | ScienceQA↑ | MM-Vet↑ |
|--------|------|------|----------|------|------|------|
| LLaVA-1.5-7B | baseline | 52.7 | 35.5 | 78.5 | 66.8 | 31.0 |
| LLaVA-1.5-7B | OPERA | 45.3 | 28.5 | 78.2 | 68.2 | 30.3 |
| LLaVA-1.5-7B | HA-DPO | 37.0 | — | — | — | — |
| LLaVA-1.5-7B | **SENTINEL** | **~4.2** | **~12.4** | **78.5** | **improved** | **improved** |

SENTINEL reduces hallucinations on Object HalBench by approximately 92% and on AMBER by approximately 65%, while maintaining or improving performance on VQAv2 and ScienceQA.

### Ablation Study

| Configuration | Object HalBench | Notes |
|------|---------|------|
| Full SENTINEL | Best | Complete model |
| Single detector (GroundingDINO) | Sub-optimal | Cross-validation is more reliable |
| $y_w^-$ as positive sample | Degraded | Context-independent samples hurt generalization |
| w/o ICB | Degraded | Single context limits generalization |
| w/o Early Intervention | Degraded | Not focusing on first occurrence reduces effectiveness |

### Key Findings
- Hallucination probability is significantly higher in later sentences than in earlier ones, validating the necessity of early intervention.
- Eliminating hallucinated objects in the second sentence reduces hallucination probability in the third sentence by more than 50%.
- Context-consistent positives $y_w^+$ substantially outperform context-independent positives $y_w^-$.
- SENTINEL is model-agnostic and can be applied to different MLLMs.

## Highlights & Insights
- The quantitative analysis of "hallucinations increasing with text length" is concise and compelling; the positional distribution figure intuitively demonstrates the necessity of early intervention.
- The entirely in-domain data construction pipeline requires no external models, no rewriting, and no human annotation, resulting in extremely low cost.
- The dual-detector cross-validation design is simple yet effective, and more robust than using a single detector.
- ICB is an elegant form of data augmentation, constructing preference pairs under progressively complex contexts.

## Limitations & Future Work
- Object detectors may miss rare objects or produce incorrect judgments in complex scenes.
- The method addresses only object hallucinations and does not handle attribute or relational hallucinations.
- SceneGraphParser may incompletely extract objects from complex descriptions.
- The iterative sampling process incurs non-trivial computational overhead, requiring multiple sampling and detection rounds per image.

## Related Work & Insights
- **vs. VCD/OPERA/DoLa**: These decoding strategies increase inference overhead; SENTINEL does not modify the inference procedure.
- **vs. HA-DPO**: Reliance on external model rewriting introduces distributional shift.
- **vs. EFUF**: Also a preference learning approach, but does not focus on early intervention and exhibits limited generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation of early hallucination propagation and the sentence-level intervention strategy are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of hallucination benchmarks and general capability benchmarks, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from motivation analysis to method design is clear.
- Value: ⭐⭐⭐⭐⭐ A low-cost, model-agnostic hallucination mitigation solution that preserves general capabilities; highly practical.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](../../NeurIPS2025/multimodal_vlm/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](../../NeurIPS2025/multimodal_vlm/when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)

<!-- RELATED:END -->
