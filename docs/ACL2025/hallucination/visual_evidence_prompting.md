---
title: >-
  [Paper Note] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models
description: >-
  [ACL 2025 (Long Paper)][Hallucination Detection][LVLM Hallucination] This paper proposes Visual Evidence Prompting (VEP), which uses the outputs of small vision expert models (such as object detectors and scene graph generators) as textualized "visual evidence" input for LVLMs. This training-free approach significantly reduces hallucinations across 11 LVLMs—improving LLaVA-1.5 by 7.2% and Claude 3 by 12.1% on the POPE benchmark.
tags:
  - "ACL 2025 (Long Paper)"
  - "Hallucination Detection"
  - "LVLM Hallucination"
  - "Visual Evidence Prompting"
  - "Small Model assisting Large Model"
  - "Object Detection"
  - "Scene Graph Generation"
date: 2026-05-08
content_hash: f3ea2e0c08e297e6
---

# Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models

**Conference**: ACL 2025 (Long Paper)  
**Code**: Unreleased  
**Area**: Hallucination Detection  
**Keywords**: LVLM Hallucination, Visual Evidence Prompting, Small Model assisting Large Model, Object Detection, Scene Graph Generation  

## TL;DR
This paper proposes Visual Evidence Prompting (VEP), which uses the outputs of small vision expert models (such as object detectors and scene graph generators) as textualized "visual evidence" input for LVLMs. This training-free approach significantly reduces hallucinations across 11 LVLMs—improving LLaVA-1.5 by 7.2% and Claude 3 by 12.1% on the POPE benchmark.

## Background & Motivation

**Core Problem**: What is the root cause of hallucinations (generating non-existent objects, relations, or attributes in images) in LVLMs?

**Attribution Analysis Findings**: Through in-depth attention attribution analysis, the authors reveal that hallucinations primarily stem from a **lack of fine-grained visual discrimination capability** rather than language bias. Specific evidence includes: (1) When hallucinations occur, the model's incorrect activation in semantically or visually similar regions accounts for up to 58.5% (e.g., misidentifying a baseball bat as a ball); (2) The CLIPScore of hallucinated objects is higher relative to the image, showing that semantic similarity is the root cause of confusion; (3) The internal confidence of visual tokens for hallucinated objects is anomalously higher than that for correct objects, indicating that the model "errs confidently."

**Limitations of Prior Work**: Existing hallucination mitigation methods either require retraining (such as LRV instruction tuning, which poses risks of catastrophic forgetting) or require modifying model architectures (such as VCD contrastive decoding or VHR attention head enhancement). These methods have limited applicability and are difficult to deploy on closed-source API models (e.g., GPT-4V, Claude, Gemini).

**Design Motivation**: Since hallucinations originate from insufficient fine-grained visual perception, utilizing small vision expert models adept at fine-grained recognition to "supplement" visual information—and injecting it into LVLMs in pure text format as prefixes—can mitigate hallucinations without modifying the models.

## Method

### Overall Architecture
The structured outputs of "small vision expert models" are converted into natural language descriptions, which serve as context prefixes and are inputted into the LVLM along with the original question. This is analogous to a human carefully identifying key elements in an image before answering visual questions. The entire process is completely training-free and model-free, making it applicable to any LVLM including API services.

### Key Designs

1. **Visual Evidence Extraction**

    - **Object Detector** (e.g., DINO): Outputs the categories and counts of detected objects in the image, formatted as text: "3 dogs, 1 cat, 2 chairs".
    - **Scene Graph Generator** (e.g., SGG models): Outputs <subject, relationship, object> triplets, formatted as: "man on surfboard, man has hair, dog near table".
    - **Complementary evidence**: The detector addresses the "what objects exist" problem (object hallucination), while the SGG addresses the "what relationships exist between objects" problem (relationship hallucination).

2. **Minimalist Prompt Construction**

    - Template: "You can see {evidence} in the image. {question}"
    - The visual evidence is directly prepended as context, requiring no complex prompt engineering.
    - The minimalist design ensures cross-model generalization—the exact same template is effective across all 11 evaluation LVLMs.

3. **Attribution Verification Mechanism**

    - Visualization via attention attribution maps verifies that after adding VEP, incorrect activations on hallucinated regions are significantly suppressed, while activations on correct regions are enhanced.
    - Quantitative analysis: VEP increases the attention weights of visual tokens for correct objects by approximately 15-20%.

### Loss & Training
- Fully training-free and plug-and-play.
- Only requires running a small detector or SGG model as an extra step (inference overhead of ~50ms/image).
- Applicable to both open-source and closed-source API models.

## Experimental Results

### Main Results: Hallucination Evaluation on 11 LVLMs

| Model | POPE Acc | +VEP | AMBER CHAIR↓ | +VEP | RPE Acc | +VEP |
|------|---------|------|-------------|------|---------|------|
| LLaVA-1.5-7B | 80.23 | **87.43** (+7.2) | 8.07 | **6.78** (-1.3) | 61.92 | **68.00** (+6.1) |
| LLaVA-1.6-7B | 84.93 | **89.43** (+4.5) | 8.59 | **7.73** (-0.9) | 70.20 | **70.46** (+0.3) |
| MiniGPT-4-v2 | 75.33 | **83.17** (+7.8) | 8.67 | **8.39** (-0.3) | 60.75 | **68.38** (+7.6) |
| GPT-4V (API) | 82.21 | **86.41** (+4.2) | 6.97 | **6.76** (-0.2) | 75.56 | **76.05** (+0.5) |
| Claude 3 (API) | 75.40 | **87.50** (+12.1) | 5.34 | **5.00** (-0.3) | 69.57 | **70.57** (+1.0) |
| Gemini 1.5 Pro | 82.43 | **87.32** (+4.9) | 8.70 | **7.63** (-1.1) | 69.06 | **71.13** (+2.1) |

### Ablation Study

| Ablation Dimension | Conclusion |
|----------|------|
| Object detection evidence only | Contributes most to POPE (object hallucination), accounting for 60-70% of the total improvement |
| Scene graph evidence only | Contributes most to RPE (relationship hallucination), accounting for 50-60% of the total improvement |
| Combined detection + scene graph | Yields the best performance across all benchmarks, proving the mutual complementarity of both evidence types |
| Ground truth annotations as evidence | Achieves a higher upper bound (POPE +10-15%), indicating that the headroom for improvement depends on the quality of the small expert models |
| Impact on general VQA | Maintained or slightly improved performance on general benchmarks like MMBench/SEED, with no side effects |

### Key Findings
- Claude 3 achieved the most significant improvement (POPE +12.1%), likely because Claude has a relatively weaker vision encoder but stronger language comprehension, which allows VEP to precisely complement its visual weak spots.
- Controllable inference speed impact: token/sec slightly dropped from 28.86 to 23.96 (approx. 17%) due to the extended input prompt.
- The newly proposed RPE (Relation Prediction Evaluation) dataset fills the gap in relationship hallucination evaluation.
- When detectors generate false positives, the LVLM exhibits a certain level of error-correction capability, preventing it from blindly accepting all visual evidence.

## Highlights & Insights
- **Analysis-driven Design**: Pinpoints the root cause of hallucinations first via attribution analysis (58.5% of incorrect activations arise from semantically similar areas) and then prescribes a targeted solution.
- **Minimalist and Efficient**: Significantly mitigates hallucinations through simple text concatenation with no retraining or model parameter modifications, reducing engineering deployment barriers.
- **Cross-model Generalization**: Highly effective across 11 different LVLMs, covering both open-source and closed-source API models.
- **Symbolic Bridge**: Small vision experts instruct the LVLM to look more accurately using symbolic output, serving as an elegant paradigm for weak-to-strong model collaboration.

## Limitations & Future Work
- Heavy reliance on the quality of external small models—misses or false positives from detectors can introduce new error sources.
- Limited label space of object detectors (e.g., COCO's 80 classes) restricts effective visual evidence generation for open-world objects.
- Increases inference latitude by roughly 17% (due to longer prompts), requiring tradeoffs in latency-sensitive applications.
- Automatical quality evaluation and filtering mechanisms for visual evidence remain unexplored.
- Synergy and integration effects with internal methods such as VHR (attention head enhancement) have not been verified.

## Related Work & Insights
- **vs VCD (Contrastive Decoding)**: VCD corrects errors during decoding at the output level, while VEP supplements visual information at the input level; the two are orthogonal and combinable.
- **vs LRV (Instruction Tuning)**: LRV requires training and faces risks of catastrophic forgetting, whereas VEP is entirely training-free.
- **vs VHR (Visual Head Reconstruction)**: VHR internally enhances visual attention heads, while VEP externally enriches visual information; hypothetically, combining both could yield a "1+1 > 2" effect.
- **vs Woodpecker (Post-processing Correction)**: Post-processing demands additional API calls and introduces new sources of hallucination, while VEP addresses this in a single pass at the input side.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple method but profound insights; the analysis-driven design is compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models evaluated across 5 benchmarks, with exceptionally detailed analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain spanning from analysis to methodology to verification.
- Value: ⭐⭐⭐⭐ Highly practical plug-and-play hallucination mitigation method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](../../ICCV2025/hallucination/only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[CVPR 2026\] VES-RFT: Rewarding Visual Evidence Sensitivity to Mitigate Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/ves-rft_rewarding_visual_evidence_sensitivity_to_mitigate_hallucinations_in_larg.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](../../ICML2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)

</div>

<!-- RELATED:END -->
