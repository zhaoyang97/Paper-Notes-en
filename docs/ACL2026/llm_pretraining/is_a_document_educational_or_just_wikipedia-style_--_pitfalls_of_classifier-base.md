---
title: >-
  [Paper Note] Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering
description: >-
  [ACL2026][LLM Pretraining][Pre-training corpora] This paper discovers that Classifier-based Quality Filtering (CQF) mistakenly identifies "Wikipedia-style writing" as having higher "educational value." Simple rephrasing…
tags:
  - "ACL2026"
  - "LLM Pretraining"
  - "Pre-training corpora"
  - "quality filtering"
  - "educational classifier"
  - "data bias"
  - "Wikipedia-style rephrasing"
date: 2026-05-08
content_hash: a75537bd29cd4d0c
---

# Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering

**Conference**: ACL2026  
**arXiv**: [2605.23721](https://arxiv.org/abs/2605.23721)  
**Code**: https://github.com/mklimasz/cqf-pitfalls  
**Area**: LLM Pre-training / Data Quality Filtering  
**Keywords**: Pre-training corpora, quality filtering, educational classifier, data bias, Wikipedia-style rephrasing  

## TL;DR
This paper discovers that Classifier-based Quality Filtering (CQF) mistakenly identifies "Wikipedia-style writing" as having higher "educational value." Simple rephrasing allows low-quality web pages to bypass pre-training data filtering thresholds, causing decision flips for approximately 7% of samples in FineWeb-Edu.

## Background & Motivation
**Background**: Modern LLM pre-training corpus construction increasingly relies on quality filtering. Beyond heuristic rules like language identification, deduplication, and character ratios, datasets such as FineWeb-Edu, DCLM, and Nemotron-CC have begun using Classifier-based Quality Filtering (CQF), where small classifiers assign "educational value" scores to web pages to decide their inclusion in the pre-training corpus.

**Limitations of Prior Work**: CQF appears to be a scalable quality judge, but it essentially mimics the scoring preferences of an LLM teacher. If the teacher mistakes writing style, formatting habits, or domain distribution for educational value, the student classifier inherits these biases.

**Key Challenge**: The scale of pre-training corpora is massive, making manual verification nearly impossible. The more automated filtering is relied upon, the more filters must identify actual content quality rather than superficial styles like "resembling a textbook/encyclopedia." Otherwise, low-quality content can bypass filters simply by changing its writing style.

**Goal**: The authors aim to test a specific question: Do CQF models judge whether text has educational value, or do they prefer Wikipedia-like organizational forms and linguistic styles?

**Key Insight**: The paper designs a direct intervention experiment: keeping original facts basically unchanged while rewriting web pages into a Wikipedia style, then comparing CQF scores and filtering decisions before and after rewriting.

**Core Idea**: If merely changing the style significantly increases CQF scores, then "educational classifiers" possess stylistic shortcuts and cannot be directly equated to true data quality judgments.

## Method

### Overall Architecture
The paper does not propose a new training algorithm but performs a diagnostic experiment on CQF. The process consists of two steps: first, randomly sampling web pages from FineWeb and using Qwen2.5-72B-Instruct to rewrite them into Wikipedia-style text (requiring facts, entities, dates, and approximate token counts to be preserved); second, inputting both original and rewritten texts into multiple CQF models to compare score changes, threshold flips, and domain distribution biases.

The authors perform two additional analyses. First, they use an Nvidia domain classifier to categorize text into 26 domains to check if score offsets are consistent across different fields. Second, three human annotators score 100 documents based on the original FineWeb-Edu educational prompt to determine if the bias originates from the student classifier or the LLM teacher labeling itself.

### Key Designs
1. **Wikipedia-style rephrasing intervention**:
	- **Function**: Construct counterfactual samples that change presentation style while keeping factual content constant.
	- **Mechanism**: Use Qwen2.5-72B-Instruct to rewrite web pages into a format similar to Wikipedia entries, without adding new facts, while maintaining dates, locations, and entities, and keeping token counts within 10% of the original.
	- **Design Motivation**: This intervention isolates the impact of "content itself" from "encyclopedic writing style." A significant score increase indicates that CQF relies at least partially on stylistic shortcuts.

2. **Multi-model CQF comparison**:
	- **Function**: Ensure conclusions are not limited to a single FineWeb-Edu classifier.
	- **Mechanism**: Compare three CQF models: FineWeb-Edu, NemoCurator Mixtral, and NemoCurator Nemotron. All are based on BERT-sized embedding models like Snowflake-Arctic-Embed-M, suitable for large-scale filtering.
	- **Design Motivation**: If multiple CQF models exhibit similar biases, the issue likely stems from the "educational filtering paradigm" or teacher labeling preferences rather than the failure of a specific model.

3. **Domain and human annotation diagnosis**:
	- **Function**: Determine if biases exist across domains and whether they originate from LLM teacher labeling.
	- **Mechanism**: In domain analysis, 20,000 samples per domain are compared. In human analysis, 100 documents with large score differences are chosen, each scored by 3 annotators using the same educational rubric.
	- **Design Motivation**: If scores increase across all domains after rewriting, the stylistic bias is not an outlier for specific domains. If human scores are lower than LLM teacher scores, the upstream supervision for CQF is inherently optimistic.

### Loss & Training
This paper does not train new models but focuses on evaluation protocols. CQF scores typically range from 0 to 5, with common thresholds at 3 or 4. The Wikipedia rephrasing prompt emphasizes "changing form, not adding facts," while the educational scoring prompt follows the FineWeb-Edu 5-point additive standard. Human annotation uses the same prompt to compare against LLM teacher judgments.

## Key Experimental Results

### Main Results
| CQF Model | Original Avg Score | Wikipedia-style Avg Score | Score Change | Interpretation |
|-----------|--------------------|---------------------------|--------------|----------------|
| FineWeb-Edu | 1.19 | 1.49 | +0.30 | Most robust among the three, but false positives at high thresholds remain evident |
| NemoCurator Mixtral | 1.17 | 1.60 | +0.43 | Largest score increase due to stylistic rewriting |
| NemoCurator Nemotron | 1.18 | 1.59 | +0.41 | Similar to Mixtral, clearly prefers encyclopedic expression |

### Ablation Study
| Analysis Item | Data / Setting | Key Result | Meaning |
|---------------|----------------|------------|---------|
| Threshold 3 False Positives | Samples with original score $\le 2$ filtered by threshold 3 after rewriting | $>7\%$ for NemoCurator Mixtral, $\approx 5\%$ for Nemotron, $\approx 6\%$ for FineWeb-Edu | Low-quality content can bypass common filtering thresholds via stylistic rewriting |
| Threshold 4 False Positives | Stricter threshold 4 | $\approx 1\%$ still fails to be filtered in FineWeb-Edu | Increasing thresholds cannot completely eliminate stylistic vulnerabilities |
| Domain Sensitivity | 26 domains, 20k samples each | Rewritten text avg scores higher than original in all domains | Stylistic bias is consistent across domains |
| Human Educational Annotation | 100 documents, 3 annotators | Humans average 0.77 points lower than Llama 3.1 70B teacher | Bias likely stems from teacher labeling itself, not just student classifiers |

### Key Findings
- Wikipedia-style rewriting has a systematic score-lifting effect on all three CQF models, indicating "educational" scores are mixed with strong stylistic preferences.
- FineWeb-Edu shows the smallest average score difference but still admits $\approx 6\%$ of originally low-scoring samples during threshold filtering; this warns that average robustness does not equal decision robustness.
- Domain analysis shows CQF prefers certain domains. If downstream models need data from low-preference domains, fixed thresholds might systematically weaken domain coverage.
- Human annotations are $\approx 0.77$ points lower than the LLM teacher, suggesting that "overly optimistic teacher labels" may be the upstream source of CQF bias.

## Highlights & Insights
- The experimental design is simple but incisive: by rewriting style without changing factual content, decision flips provide strong evidence that classifiers learn surface shortcuts.
- It links data poisoning risks with data filtering bias. Malicious content does not necessarily need complex attacks; packaging it in an "encyclopedic" form can increase its probability of entering the pre-training corpus.
- Implications for pre-training corpus construction: quality filtering should not rely solely on a single CQF score but should combine content consistency, source reputation, domain quotas, and adversarial stylistic perturbation tests.
- This work reminds evaluators that LLM-as-teacher labeling prompts require auditing; student model biases might just be amplified versions of teacher biases.
- A deeper insight is that filters need to remain stable against "fidelity-preserving rewriting" rather than automatically treating standardized formats, header hierarchies, and encyclopedic tones as evidence of high quality.

## Limitations & Future Work
- Authors acknowledge that automatic rewriting introduces noise, so exact percentages are approximations of the problem scale rather than precise estimates of bypass rates on the real internet.
- Experiments primarily examine Wikipedia-style rewriting without systematically exploring other styles, such as textbook, Q&A, academic abstracts, or malicious SEO text.
- The paper does not use the falsely admitted data to actually pre-train downstream models, thus cannot quantify the final impact of this CQF vulnerability on model capability, safety, or bias.
- Human annotation is limited to 100 documents, sufficient to suggest teacher bias but insufficient to establish a complete human quality standard.

## Related Work & Insights
- **vs FineWeb-Edu**: FineWeb-Edu distills LLM labels into lightweight CQF models for large-scale educational filtering; this paper points out it may mistake Wikipedia-style writing for high educational value.
- **vs Nemotron-CC / Nemotron-CLIMB**: The Nemotron series emphasizes data quality and domain mixing; the domain sensitivity results in this paper suggest CQF scores should be audited alongside domain ratios.
- **vs Traditional Heuristic Filtering**: Rules like lang-id, deduplication, and perplexity are transparent but coarse; CQF is more flexible but opaque. A combination of both with adversarial robustness checks might be needed.
- **Inspiration for Future Research**: A "style invariance" test set could be constructed, requiring quality filters to remain stable under fidelity-preserving rewriting while remaining sensitive to actual content quality changes.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Problem approach is very clear, intervention design is persuasive; method is more of a diagnostic experiment than a complex algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three CQF models, 100k samples, 26 domains, and human annotation; lacks verification of downstream pre-training impact.
- Writing Quality: ⭐⭐⭐⭐☆ Concise and direct with clear conclusions; some key percentages are in the text narration and could be more fully tabulated.
- Value: ⭐⭐⭐⭐⭐ Significant for pre-training data filtering, data poisoning defense, and LLM teacher annotation auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[AAAI 2026\] Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](../../AAAI2026/llm_pretraining/perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](../../ICML2026/llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)

</div>

<!-- RELATED:END -->
