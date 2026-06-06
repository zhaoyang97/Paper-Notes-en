---
title: >-
  [Paper Note] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias
description: >-
  [ACL 2026][LLM Evaluation][VLM-as-a-Judge] Reveals that VLM-as-a-Judge systems exhibit a severe "informativeness bias"—judges tend to favor more detailed and rich responses even when they contradict image content. The pa…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "VLM-as-a-Judge"
  - "informativeness bias"
  - "image anchoring"
  - "evaluation reliability"
  - "multimodal evaluation"
date: 2026-05-08
content_hash: d43ed9243ca311bc
---

# When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias

**Conference**: ACL 2026  
**arXiv**: [2604.17768](https://arxiv.org/abs/2604.17768)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: VLM-as-a-Judge, informativeness bias, image anchoring, evaluation reliability, multimodal evaluation

## TL;DR

Reveals that VLM-as-a-Judge systems exhibit a severe "informativeness bias"—judges tend to favor more detailed and rich responses even when they contradict image content. The paper proposes the BIRCH paradigm to reduce bias by up to 17% and improve performance by up to 9.8% by correcting candidate answers before comparison.

## Background & Motivation

**Background**: VLM-as-a-Judge (using vision-language models as automatic evaluators) has become a mainstream method for assessing the output quality of VLMs. It draws on the LLM-as-a-Judge approach, where a powerful VLM scores or ranks multiple candidate responses to replace expensive human evaluation.

**Limitations of Prior Work**: The authors' analysis reveals a concerning issue—VLM judges often pay insufficient attention to images when making decisions. They tend to blindly prefer more informative and detailed descriptions, even when the content contradicts the actual image. Surprisingly, even if a judge can identify an inconsistency in a response, it might still select that response because it "appears richer."

**Key Challenge**: VLM judges face an implicit trade-off—informativeness vs. correctness. Existing evaluation paradigms conflate these two dimensions, causing the judge's focus to shift from visual grounding to surface-level text quality.

**Goal**: (1) Systematically quantify the severity of informativeness bias in VLM-as-a-Judge; (2) Design a new evaluation paradigm that shifts the judge's focus from informativeness to image-based correctness.

**Key Insight**: The authors propose splitting the evaluation process into two steps: first, correct the content in candidate answers that is inconsistent with the image (eliminating interference from informativeness differences), and then perform comparisons based on the corrected versions. This way, the judge only needs to focus on "who is more correct" rather than "who says more."

**Core Idea**: By introducing a "Truthful Anchor"—generating a corrected version aligned with the image—the judge can then compare correctness under balanced informativeness conditions.

## Method

### Overall Architecture

BIRCH (Balanced Informativeness and CoRrectness with a Truthful AnCHor) is a two-stage evaluation paradigm. The input consists of an image and two candidate answers, and the output is a judgment of which answer is better. Stage 1: For each candidate answer, a VLM corrects inconsistencies based on the image content to generate a "Truthful Anchor" version; Stage 2: The VLM judge compares the deviation of the original answers from their corrected versions. A larger deviation indicates the original answer was less consistent with the image.

### Key Designs

1.  **Systematic Definition and Quantification of Informativeness Bias**:

    *   **Function**: Establish quantitative metrics to measure the degree of informativeness bias in VLM judges.
    *   **Mechanism**: Construct comparative experiments—pairing a correct but concise answer with an informative but erroneous answer to observe the judge's selection. Bias degree is defined as the rate at which the judge selects the incorrect but detailed answer. The prevalence of bias is quantified through systematic experiments across multiple benchmarks and VLMs.
    *   **Design Motivation**: The effectiveness of solutions can only be evaluated after quantifying the severity of the problem. No prior work has systematically studied this specific bias in VLM judges.

2.  **Truthful Anchor Generation**:

    *   **Function**: Generate a corrected version of each candidate answer that is aligned with the image content.
    *   **Mechanism**: Given an image and a candidate answer, the VLM is prompted to check if each description in the answer is consistent with the image. Inconsistent parts are replaced with correct descriptions while maintaining the overall structure and informativeness. This corrected version serves as the "Truthful Anchor"—it preserves the original informativeness and style but fixes contradictions.
    *   **Design Motivation**: Forcing the judge to focus directly on correctness is difficult (because informativeness bias is implicit); explicit correction separates correctness differences from informativeness differences.

3.  **Anchor-Based Fair Comparison**:

    *   **Function**: Compare the correctness of candidate answers while eliminating informativeness interference.
    *   **Mechanism**: Instead of directly comparing two original answers, the judge evaluates the degree of deviation between each answer and its truthful anchor. If Answer A requires more correction to align with the image, it implies Answer A has poorer image consistency. The judge only needs to evaluate "which answer required fewer modifications," thereby bypassing informativeness bias.
    *   **Design Motivation**: This shifts the evaluation criterion from "which answer looks better" to "which answer is more consistent with the image," fundamentally addressing the source of the bias.

## Key Experimental Results

### Main Results

| Benchmark / Judge Model | Original Bias Rate | Bias Rate with BIRCH | Bias Reduction | Accuracy Gain |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4V Judge | Baseline | Lower | -17% | +9.8% |
| Gemini Judge | Baseline | Lower | -14% | +7.2% |
| LLaVA Judge | Baseline | Lower | -11% | +5.6% |
| Average across Benchmarks | High Bias | Significantly Lower | -12~17% | +5~9.8% |

### Ablation Study

| Configuration | Bias Rate | Accuracy | Description |
| :--- | :--- | :--- | :--- |
| BIRCH Full Scheme | Lowest | Highest | Includes both correction and comparison steps |
| Correction only, no comparison | Medium | Medium | Proves the comparison strategy is also important |
| Direct prompt "focus on correctness" | Still High | Limited Gain | Proves simple prompting cannot eliminate implicit bias |
| Different VLMs as correctors | Minor difference | Stable | Method is not sensitive to the choice of corrector model |

### Key Findings

*   Informativeness bias is prevalent across all tested VLMs, even the strongest models (e.g., GPT-4V) are affected.
*   Even when explicitly instructed to "ignore informativeness and focus on correctness," bias remains significant—indicating this is a deep-seated model tendency rather than an instruction-following issue.
*   Both steps of BIRCH contribute: the correction step eliminates content bias, while the comparison step avoids residual informativeness interference.
*   In scenarios with more complex image descriptions, informativeness bias becomes more severe, and the gains from BIRCH are larger.

## Highlights & Insights

*   **Problem discovery is a major contribution**: Informativeness bias is a previously overlooked but far-reaching problem—if automatic evaluation is unreliable, model selection and training based on it may be misled.
*   **The "correct-then-compare" paradigm design is ingenious**: It does not try to make the judge "smarter" but eliminates the source of bias through preprocessing. This "change the input, not the model" approach can be applied to other evaluation bias problems.
*   The logic can be transferred to similar bias scenarios in LLM-as-a-Judge, such as judges favoring longer or specifically formatted responses.

## Limitations & Future Work

*   The correction step itself relies on the VLM's visual understanding—if the corrector's own visual understanding is flawed, it may introduce new biases.
*   The two-step process increases inference costs (requiring additional correction calls for each judgment), sacrificing efficiency.
*   Currently focuses primarily on "informativeness bias"; VLM judges may harbor other biases (e.g., position bias, length bias).
*   Future work could explore training specialized "de-biased" judges that internalize the BIRCH methodology within the model.

## Related Work & Insights

*   **vs LLM-as-a-Judge bias research**: Previous work mainly focused on position bias and verbosity bias in LLM judges; this paper is the first to systematically study informativeness bias specific to VLM judges with a precise problem definition.
*   **vs Direct scoring methods**: Methods that ask VLMs to assign scores are equally affected by informativeness bias; BIRCH's correction logic is applicable to scoring scenarios.
*   **vs Human evaluation**: BIRCH narrows the gap between automatic and human evaluation, though human evaluation remains irreplaceable for highly subjective dimensions.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First to reveal and systematically quantify informativeness bias in VLM judges; the problem definition is novel and important.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive experiments across multiple models and benchmarks with sufficient ablation verification.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation and logically rigorous experimental design.
*   Value: ⭐⭐⭐⭐⭐ Significant impact on the field of VLM automatic evaluation; both the identified bias and the proposed solution have broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)
- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)

</div>

<!-- RELATED:END -->
