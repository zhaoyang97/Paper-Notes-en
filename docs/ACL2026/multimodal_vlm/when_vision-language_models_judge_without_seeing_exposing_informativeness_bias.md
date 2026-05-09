---
title: >-
  [Paper Note] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias
description: >-
  [ACL 2026][Multimodal VLM][VLM-as-a-Judge] This paper exposes a severe *informativeness bias* in VLM-as-a-Judge systems—judges tend to favor more detailed and elaborate responses even when such responses contradict the image content. The proposed BIRCH paradigm first calibrates candidate answers against the image before comparison, reducing bias by up to 17% and improving performance by up to 9.8%.
tags:
  - ACL 2026
  - Multimodal VLM
  - VLM-as-a-Judge
  - informativeness bias
  - visual grounding
  - evaluation reliability
  - multimodal evaluation
date: 2026-05-08
content_hash: 69d4abcb3ecf83b0
---

# When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias

**Conference**: ACL 2026
**arXiv**: [2604.17768](https://arxiv.org/abs/2604.17768)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: VLM-as-a-Judge, informativeness bias, visual grounding, evaluation reliability, multimodal evaluation

## TL;DR

This paper exposes a severe *informativeness bias* in VLM-as-a-Judge systems—judges tend to favor more detailed and elaborate responses even when such responses contradict the image content. The proposed BIRCH paradigm first calibrates candidate answers against the image before comparison, reducing bias by up to 17% and improving performance by up to 9.8%.

## Background & Motivation

**Background**: VLM-as-a-Judge—using vision-language models as automatic evaluators—has become the dominant approach for assessing VLM output quality. Following the paradigm of LLM-as-a-Judge, a powerful VLM scores or ranks candidate responses as a cost-effective alternative to human evaluation.

**Limitations of Prior Work**: The authors identify a concerning phenomenon: VLM judges pay insufficient attention to the image when making decisions. They exhibit a systematic preference for more informative and descriptive responses, even when such responses contradict the actual image content. More strikingly, even when a judge can identify that a response is inconsistent with the image, it may still select that response simply because it appears richer in content.

**Key Challenge**: VLM judges face an implicit trade-off between *informativeness* and *correctness*. Existing evaluation paradigms conflate these two dimensions, causing the judge's attention to shift away from visual grounding toward surface-level textual quality.

**Goal**: (1) Systematically quantify the severity of informativeness bias in VLM-as-a-Judge systems; (2) design a new evaluation paradigm that redirects the judge's focus from informativeness to image-grounded correctness.

**Key Insight**: The authors propose decomposing the judgment process into two stages—first calibrating candidate responses to resolve image inconsistencies (thereby eliminating the confounding effect of informativeness differences), then comparing the calibrated versions. This ensures the judge addresses *who is more correct* rather than *who says more*.

**Core Idea**: By introducing a *truthful anchor*—a corrected version of each candidate response that is aligned with the image—the judge can compare correctness under informativeness-balanced conditions.

## Method

### Overall Architecture

BIRCH (**B**alanced **I**nformativeness and Co**R**rectness with a Truthful An**CH**or) is a two-stage evaluation paradigm. Given an image and two candidate responses, it produces a judgment of which response is better. In Stage 1, for each candidate response, a VLM corrects inconsistencies with the image to produce a *truthful anchor* version. In Stage 2, the VLM judge compares how much each original response deviates from its corresponding anchor; greater deviation indicates weaker image consistency.

### Key Designs

1. **Systematic Definition and Quantification of Informativeness Bias**:

    - **Function**: Establish quantitative metrics for measuring the degree of informativeness bias in VLM judges.
    - **Mechanism**: Construct controlled experiments by pairing a correct but concise response with an informative but erroneous one, and observe the judge's preference. Bias rate is defined as the proportion of cases in which the judge selects the incorrect yet detailed response. The pervasiveness of this bias is quantified through systematic experiments across multiple benchmarks and VLMs.
    - **Design Motivation**: Quantifying the severity of the problem is a prerequisite for evaluating the effectiveness of any solution. No prior work has systematically studied this specific bias in VLM judges.

2. **Truthful Anchor Generation**:

    - **Function**: Generate an image-aligned corrected version for each candidate response.
    - **Mechanism**: Given the image and a candidate response, a VLM is prompted to verify whether each described element is consistent with the image, replacing inconsistent parts with correct descriptions while preserving the overall structure and informativeness of the response. This corrected version serves as the *truthful anchor*—it retains the original response's informativeness and writing style while resolving contradictions with the image.
    - **Design Motivation**: Directly prompting a judge to focus on correctness is insufficient because informativeness bias operates implicitly. Explicit correction disentangles correctness differences from informativeness differences.

3. **Anchor-Based Fair Comparison**:

    - **Function**: Compare the correctness of candidate responses after eliminating the confounding effect of informativeness.
    - **Mechanism**: Rather than directly comparing two original responses, the judge compares the degree of deviation between each response and its truthful anchor. If response A requires more corrections to align with the image, A exhibits weaker image consistency. The judge only needs to assess *which response requires fewer modifications*, thereby circumventing informativeness bias.
    - **Design Motivation**: This reframes the evaluation criterion from *which response looks better* to *which response is more consistent with the image*, addressing the root cause of the bias.

## Key Experimental Results

### Main Results

| Benchmark / Judge Model | Original Bias Rate | Bias Rate after BIRCH | Bias Reduction | Accuracy Gain |
|------------------------|-------------------|----------------------|---------------|--------------|
| GPT-4V as Judge | Baseline | Reduced | −17% | +9.8% |
| Gemini as Judge | Baseline | Reduced | −14% | +7.2% |
| LLaVA as Judge | Baseline | Reduced | −11% | +5.6% |
| Multi-benchmark Average | High bias | Significantly reduced | −12∼17% | +5∼9.8% |

### Ablation Study

| Configuration | Bias Rate | Accuracy | Note |
|--------------|-----------|----------|------|
| BIRCH (full) | Lowest | Highest | Both correction and comparison stages active |
| Correction only, no anchor comparison | Moderate | Moderate | Demonstrates that the comparison strategy also matters |
| Direct prompting to "focus on correctness" | Still high | Limited gain | Demonstrates that simple prompting cannot eliminate implicit bias |
| Different VLMs as the corrector | Minimal variation | Stable | Method is robust to the choice of correction model |

### Key Findings

- Informativeness bias is pervasive across all tested VLMs, including the strongest models such as GPT-4V.
- Even when judges are explicitly instructed to "ignore informativeness and focus on correctness," bias remains pronounced—indicating a deep-seated model tendency rather than an instruction-following failure.
- Both stages of BIRCH contribute: the correction step eliminates content discrepancies, while the comparison step avoids residual informativeness interference.
- Informativeness bias is more severe in scenarios with complex image descriptions, where BIRCH also yields greater gains.

## Highlights & Insights

- **Problem identification is itself a significant contribution**: Informativeness bias is a previously overlooked yet consequential issue—if automatic evaluation is unreliable, model selection and training decisions based on it may be systematically misguided.
- **The "correct-then-compare" paradigm is elegantly designed**: Rather than making the judge "smarter," BIRCH eliminates the source of bias through preprocessing. This strategy of *modifying the input rather than the model* is broadly applicable to other evaluation bias problems.
- The approach is transferable to analogous biases in LLM-as-a-Judge settings—for instance, LLM judges may similarly favor longer or more formatted responses.

## Limitations & Future Work

- The correction stage itself relies on the VLM's visual understanding capability; if the corrector's visual comprehension is flawed, it may introduce new errors.
- The two-stage pipeline increases inference cost, as each judgment requires an additional correction call.
- The current work focuses on informativeness bias as a single bias type; VLM judges may exhibit other forms of bias, such as position bias or length bias.
- Future work could explore training dedicated debiased judges that internalize the BIRCH paradigm directly into the model.

## Related Work & Insights

- **vs. LLM-as-a-Judge bias research**: Prior work primarily addresses position bias and verbosity bias in LLM judges. This paper is the first to systematically study the informativeness bias unique to VLM judges, with a more precise problem formulation.
- **vs. direct scoring methods**: Direct VLM scoring approaches are equally susceptible to informativeness bias; the BIRCH correction strategy is applicable to scoring settings as well.
- **vs. human evaluation**: BIRCH narrows the gap between automatic and human evaluation, though human assessment remains indispensable for highly subjective evaluation dimensions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to reveal and systematically quantify informativeness bias in VLM judges; the problem formulation is both novel and significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive experiments across multiple models and benchmarks with thorough ablation validation.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated and experimental design is logically rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Significant implications for the VLM automatic evaluation community; both the identified bias problem and the proposed solution carry broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](../../AAAI2026/multimodal_vlm/multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)

</div>

<!-- RELATED:END -->
