---
title: >-
  [Paper Note] Core: Robust Factual Precision with Informative Sub-Claim Identification
description: >-
  [ACL 2025][LLM Safety][Factual precision evaluation] This paper proposes the Core framework, which achieves robust factual precision evaluation by identifying and filtering informative sub-claims, addressing the issue of inaccurate evaluation in existing methods caused by the dilution effect of uninformative claims.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Factual precision evaluation"
  - "sub-claim decomposition"
  - "robust evaluation"
  - "informativeness filtering"
  - "factual verification"
date: 2026-05-08
content_hash: 18763aacbd9cadb2
---

# Core: Robust Factual Precision with Informative Sub-Claim Identification

**Conference**: ACL 2025  
**Area**: LLM Safety  
**Keywords**: Factual precision evaluation, sub-claim decomposition, robust evaluation, informativeness filtering, factual verification

## TL;DR
This paper proposes the Core framework, which achieves robust factual precision evaluation by identifying and filtering informative sub-claims, addressing the issue of inaccurate evaluation in existing methods caused by the dilution effect of uninformative claims.

## Background & Motivation

**Background**: Evaluating the factual accuracy of LLM-generated text is a core task in current NLP. The mainstream evaluation paradigm is "decompose-then-verify": it first decomposes the generated text into atomic claims, then verifies the factual correctness of each claim one by one, and finally calculates the ratio of correct claims as the factual precision score. FActScore is a representative work of this paradigm.

**Limitations of Prior Work**: Existing "decompose-then-verify" methods suffer from an overlooked yet serious issue—the decomposed claims contain a large number of uninformative claims, such as "this is an important issue" or "the field has developed rapidly in recent years". Although such claims do not count as factual errors (they are generally correct generalized statements), they do not contain meaningful factual information either. Their presence dilutes the factual precision score; a model can obtain a falsely inflated precision score simply by generating a large amount of uninformative but "correct" fluff.

**Key Challenge**: Factual precision should measure "how much of the generated meaningful information is correct," but existing methods actually measure "how many of all generated claims are correct." The gap between the two is precisely caused by uninformative claims. This makes it difficult to distinguish between a model that merely generates empty, correct statements and a model that generates rich, accurate information based on scores alone.

**Goal**: (1) Define and quantify the impact of uninformative claims on factual precision evaluation; (2) design an effective identification method for informative sub-claims; (3) build a robust factual precision evaluation framework, Core.

**Key Insight**: The authors define "informativeness" as whether a claim contains verifiable concrete facts (such as specific times, locations, numbers, and causal relationships) and train a specialized classifier to distinguish informative from uninformative claims.

**Core Idea**: In factual precision calculation, uninformative sub-claims are first filtered out. Factual verification and precision calculation are performed only on informative sub-claims, thereby eliminating the dilution effect of uninformative claims and obtaining a more robust factual precision score.

## Method

### Overall Architecture
The evaluation process of Core is: Input generated text $\rightarrow$ claim decomposition $\rightarrow$ informativeness filtering $\rightarrow$ factual verification $\rightarrow$ calculate precision score. Compared with traditional methods, the key novel step is informativeness filtering, which acts as a quality gate between decomposition and verification, ensuring that only meaningful claims enter the verification phase.

### Key Designs

1. **Informative Claim Classifier**:

    - **Function**: Determine whether each sub-claim contains meaningful, verifiable factual information.
    - **Mechanism**: Define informativeness as a combination of five elements—whether it contains specific named entities, whether it contains quantitative information (numbers, dates, etc.), whether it contains verifiable relations (causality, attributes, etc.), whether it is a specific rather than generalized statement, and whether it has clear truth conditions (can be judged as true or false). A binary classifier is fine-tuned on DeBERTa-v3 using 3,000 manually annotated claim-level data points. The classifier simultaneously outputs scores for the five elements, making classification results interpretable.
    - **Design Motivation**: Informativeness is a prerequisite for factual precision evaluation—only claims containing specific facts are worth verifying, whereas verifying empty, generalized claims is meaningless. The five-element definition transforms "informativeness" from a vague concept into an operational judgment standard.

2. **Hierarchical Claim Decomposition Strategy**:

    - **Function**: Improve existing decomposition methods to achieve a better balance between claim granularity and quantity.
    - **Mechanism**: Adopt a two-step decomposition strategy—the first step is coarse-grained decomposition, grouping text into paragraph-level claims; the second step is fine-grained decomposition, further breaking down each paragraph-level claim into atomic claims. For the second step, a "decomposition necessity" judgment is introduced—if a paragraph-level claim is already atomic (containing only one fact), it will not be further decomposed. This avoids fragmentation and information loss caused by over-decomposition. Decomposition is completed using a few-shot prompted LLM.
    - **Design Motivation**: Over-decomposition produces more uninformative fragments (e.g., "He was born", "in a city"), increasing noise. Hierarchical decomposition reduces fragmentation while maintaining atomicity.

3. **Weighted Precision Calculation**:

    - **Function**: Consider the variation in information density of claims in the final precision calculation.
    - **Mechanism**: While traditional precision is defined as $P = \frac{\text{正确声明数}}{\text{总声明数}}$, the improvement of Core is $P_{core} = \frac{\sum_{i \in I} w_i \cdot v_i}{\sum_{i \in I} w_i}$, where $I$ is the set of informative claims, $w_i$ is the information density weight of claim $i$ (the weighted sum of the five-element scores), and $v_i \in \{0, 1\}$ is the verification result. This ensures that claims containing richer, more specific information carry more weight in the precision calculation.
    - **Design Motivation**: Not all informative claims are equally important—"Einstein published relativity in 1905" is more informative than "Einstein was a physicist," hence it should receive a higher weight in precision evaluation.

### Loss & Training
The informativeness classifier is trained using binary cross-entropy loss, with an additional multi-label auxiliary loss for the five elements. Training data is obtained through crowdsourced annotation, where each claim is independently annotated by 3 annotators with informativeness and five-element labels, and the final label is determined by majority vote.

## Key Experimental Results

### Main Results

| Evaluation Method | Biography Gen (ρ) | News Summarization (ρ) | QA Answering (ρ) | Overall (ρ) | Robustness to Uninformative Claims |
|---------|-----------|-----------|---------|--------|-------------------|
| FActScore | 0.68 | 0.61 | 0.64 | 0.64 | Low |
| SAFE | 0.71 | 0.65 | 0.67 | 0.68 | Medium |
| Core (Ours) | 0.82 | 0.76 | 0.78 | 0.79 | High |
| Core w/o Weighting | 0.78 | 0.72 | 0.75 | 0.75 | High |

### Ablation Study

| Configuration | Correlation (ρ) | Dilution Effect  | Description |
|------|----------|----------|------|
| Full Core | 0.79 | 0.03 | Full model, minimum dilution effect |
| w/o Informativeness Filtering | 0.64 | 0.21 | Degrades to FActScore, severe dilution |
| w/o Weighted Precision | 0.75 | 0.03 | No weighting but filtered, still effective |
| w/o Hierarchical Decomposition | 0.76 | 0.07 | Direct atomic decomposition, increased fragments |
| Threshold 0.3 (Looser) | 0.73 | 0.10 | Insufficient filtering |
| Threshold 0.7 (Stricter) | 0.77 | 0.02 | Filters out too many claims |

### Key Findings
- Informativeness filtering is the core of improving robustness—the correlation increases from 0.64 to 0.79, and the dilution effect decreases from 0.21 to 0.03.
- Weighted precision contributes an additional 4 correlation points on top of filtering, indicating that variations in information density indeed impact evaluation quality.
- In scenarios with a high proportion of uninformative claims (e.g., long-text generation), Core's advantage over FActScore is more pronounced.
- The optimal filtering threshold is around 0.5, balancing information retention and noise removal.

## Highlights & Insights
- Identifying and quantifying the "dilution effect of uninformative claims" is a core contribution—this is a blind spot in existing factual evaluation methods and explains why some models score high on FActScore but behave poorly in actual quality.
- The five-element definition of informativeness operationalizes a vague concept, providing a reusable standard for informativeness judgments.
- The method can serve as a plug-and-play plugin for any "decompose-then-verify" evaluation framework, unrestricted to specific verification methods.

## Limitations & Future Work
- The judgment of informativeness remains subjective, with inter-annotator agreement around 0.75, indicating many boundary cases.
- Current evaluation mainly focuses on English text; informativeness judgment in multilingual scenarios may expose cultural differences.
- The weight design for weighted precision is relatively simple; more complex information value quantification schemes are worth exploring.
- Future work can combine informativeness evaluation and recall evaluation to construct a more comprehensive factual quality evaluation system.

## Related Work & Insights
- **vs FActScore**: FActScore treats all claims equally, while Core distinguishes differences in the importance of informative claims, showing a clear advantage in high-noise scenarios.
- **vs SAFE (Google)**: SAFE verifies claims through search engines but does not filter uninformative claims; Core's filtering mechanism can be integrated into SAFE.
- **vs ClaimDecomp**: ClaimDecomp focuses on decomposition quality, whereas Core focuses on post-decomposition claim selection, making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Identifies the dilution effect issue and proposes a systematic solution
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task verification, detailed ablation, and comparison with mainstream methods
- Writing Quality: ⭐⭐⭐⭐⭐ Precise problem definition and clear analytical logic
- Value: ⭐⭐⭐⭐⭐ High value for key improvements in the LLM factual evaluation field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective Extraction and Evaluation of Factual Claims](towards_effective_extraction_and_evaluation_of_factual_claims.md)
- [\[ACL 2025\] Mamba Knockout for Unraveling Factual Information Flow](mamba_knockout_for_unraveling_factual_information_flow.md)
- [\[CVPR 2025\] Towards All-in-One Medical Image Re-Identification](../../CVPR2025/llm_safety/towards_all-in-one_medical_image_re-identification.md)
- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[ACL 2025\] Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations](faithful_and_robust_llm-driven_theorem_proving_for_nli_explanations.md)

</div>

<!-- RELATED:END -->
