---
title: >-
  [Paper Note] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web
description: >-
  [NeurIPS 2025][Social Computing][image-text verification] AVerImaTeC introduces the first image-text fact-checking dataset with complete evidence annotation — 1,297 real-world image-text claims…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "image-text verification"
  - "evidence retrieval"
  - "QA reasoning"
  - "multimodal verification"
  - "dataset"
date: 2026-05-08
content_hash: 96cabc5d998fdff5
---

# AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web

**Conference**: NeurIPS 2025
**arXiv**: [2505.17978](https://arxiv.org/abs/2505.17978)  
**Code**: [https://github.com/abril4416/AVerImaTeC](https://github.com/abril4416/AVerImaTeC)  
**Area**: Fact-Checking / Multimodal
**Keywords**: image-text verification, evidence retrieval, QA reasoning, multimodal verification, dataset

## TL;DR
AVerImaTeC introduces the first image-text fact-checking dataset with complete evidence annotation — 1,297 real-world image-text claims, a 5-stage annotation pipeline (extraction → QA reasoning → sufficiency check → iterative refinement → second check), and temporally constrained evidence (to prevent temporal leakage). The baseline system achieves 82% accuracy with ground-truth evidence, but drops to 15–25% under automatic evidence retrieval, revealing the substantial challenges of image-text verification.

## Background & Motivation

**Background**: Text-based fact-checking has established benchmarks such as FEVER and AVeriTeC, yet image-text fact-checking lacks a benchmark with evidence annotation. Existing multimodal datasets either rely on synthetic claims or provide no evidence reasoning paths.

**Limitations of Prior Work**: (a) Datasets without evidence annotation can only train verdict classifiers (whether something is false) rather than reasoning systems (why it is false); (b) synthetic claims do not reflect the dissemination patterns of real misinformation; (c) temporal leakage — using evidence published after the fact-check article constitutes an unfair advantage.

**Key Challenge**: Effective fact-checking requires a reasoning path (QA chain → verdict), yet annotating such paths is extremely costly — each claim requires multiple rounds of verification and at least three annotators.

**Goal**: Construct an image-text fact-checking dataset with complete reasoning paths, advancing the field from verdict classification to evidence-based reasoning.

**Key Insight**: A 5-stage annotation pipeline ensures the sufficiency and consistency of reasoning paths — Phases 3 and 5 employ independent annotators who judge solely from QA pairs (without access to the original content), verifying whether the reasoning path is self-contained.

**Core Idea**: Real image-text claims + QA reasoning path annotation + dual sufficiency review + temporally constrained evidence + MLLM baseline system = the first complete benchmark for image-text fact-checking.

## Method

### Overall Architecture
**Annotation Pipeline**: Phase 1 extracts and normalizes atomic image-text claims → Phase 2 generates QA pairs (questions may reference images) → Phase 3 sufficiency check (third-party annotators judge from QA pairs alone) → Phase 4 iterative refinement if inconsistencies arise → Phase 5 second sufficiency check. **Baseline System**: Generator → Answerer → Verdict predictor → Explainer pipeline.

### Key Designs

1. **QA Reasoning Path Annotation**:

    - Function: Annotates the verification reasoning process for each claim.
    - Mechanism: Annotators formulate questions required for verification (62.5% involve images; 53.9% require image search) and locate answers. Question–answer pairs form a reasoning chain that converges into a final verdict.
    - Design Motivation: QA pairs make the fact-checking process interpretable — revealing not only that a claim is false but also why it is false.

2. **Dual Sufficiency Review Mechanism**:

    - Function: Ensures that QA pairs are independently sufficient to determine the veracity of a claim.
    - Mechanism: Independent annotators in Phases 3 and 5 view only the QA pairs (not the original claim or image) and attempt to render a verdict. Agreement indicates that the QA pairs are self-contained. If Phase 3 reveals disagreement, the annotation enters Phase 4 for iterative refinement.
    - Design Motivation: Prevents annotators from omitting critical information in QA pairs — the sufficiency check eliminates approximately 15% of cases requiring re-annotation.

3. **Temporally Constrained Evidence**:

    - Function: Ensures that evidence predates the publication of the claim, preventing temporal leakage.
    - Mechanism: Only web pages published before the claim's publication date are permitted as evidence. For the 5% of claims lacking a precise date, the date is estimated.
    - Design Motivation: Using the fact-check article itself as "evidence" constitutes cheating — a real-world system cannot have prior knowledge of the verification outcome.

### Loss & Training
- This work is a dataset annotation effort; no model training is involved.
- Baselines: MLLM pipeline using Gemini/Qwen/LLaVA as verdict predictors.
- Evaluation: accuracy (verdict) + ROUGE-1 (explanation quality) + LLM-based reference scoring.

## Key Experimental Results

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Number of claims | 1,297 (793 train, 152 dev, 352 test) |
| Proportion of image-related questions | 62.5% |
| Questions requiring image search | 53.9% |
| Inter-annotator κ | 0.742 |
| QA pair overlap (recall) | 74.7% |
| QA pair overlap (precision) | 67.2% |
| Label distribution | ~78% refuted (test set skewed) |

### Baseline Performance

| Condition | Verdict Accuracy | Explanation ROUGE-1 |
|-----------|-----------------|---------------------|
| Ground-truth evidence | **82%** (Gemini) | 0.50 |
| Automatic retrieval evidence | **15–25%** | Low |
| Parallel QG (best) | — | 0.39–0.43 |
| Dynamic QG | — | 0.27–0.33 |

### Key Findings
- 82% accuracy with ground-truth evidence — demonstrating that MLLMs can comprehend and utilize evidence effectively.
- Accuracy drops to 15–25% under automatic retrieval — **evidence retrieval is the bottleneck** (recall rate 15–25%), not reasoning.
- Parallel question generation outperforms dynamic generation — simpler strategies prove more stable.
- MLLMs exhibit ~30% reverse image search bias — a tendency to over-rely on image search.
- Label skew (78% refuted) necessitates rebalancing in future work.

## Highlights & Insights
- **Dual sufficiency review is central to quality assurance**: Other QA datasets omit this step, leaving QA pairs susceptible to missing critical information.
- **Evidence retrieval is the core bottleneck**: Model reasoning capacity is adequate, but the system fails to locate correct evidence — this redefines the primary challenge in image-text fact-checking.
- **Temporal constraints** make evaluation more realistic, preventing the "look up the answer first, then justify" form of leakage.

## Limitations & Future Work
- Small scale (1,297 claims) — an inherent limitation of manual annotation.
- Severe label skew (78% refuted) — likely an artifact of fact-checking article selection.
- 5% of claims lack precise dates, introducing error into temporal constraints.
- ROUGE-1 is inadequate for evaluating explanation quality — low correlation with ground-truth references.
- Pre-trained MLLMs may have been exposed to fact-check articles — potential data contamination.

## Related Work & Insights
- **vs. AVeriTeC**: A text-only fact-checking dataset; AVerImaTeC extends the paradigm to image-text multimodality.
- **vs. FEVER**: Based on synthetic Wikipedia claims; AVerImaTeC employs real-world misinformation in circulation.
- **vs. Snopes et al.**: Fact-checking websites lack standardized QA reasoning path annotation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First image-text fact-checking dataset with complete reasoning paths.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines + sufficiency analysis + error analysis.
- Writing Quality: ⭐⭐⭐⭐ Annotation pipeline described in detail.
- Value: ⭐⭐⭐⭐⭐ Establishes the evaluation standard for image-text fact-checking.

### Additional Method Notes
- **Human cost of the 5-stage annotation**: Each claim requires approximately 45 minutes of annotation (across all 5 stages); 1,297 claims amount to roughly 975 person-hours — the price of high-quality data.
- **Sufficiency check rejection rate**: Approximately 15% of QA pairs are judged insufficient at Phase 3 and returned to Phase 4 for refinement — this quality gate ensures the self-containedness of reasoning paths.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](precise_information_control_in_long-form_text_generation.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](../../ACL2026/social_computing/claimdb_a_fact_verification_benchmark_over_large_structured_data.md)
- [\[ACL 2026\] FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes](../../ACL2026/social_computing/figsim_a_dataset_for_fine-grained_suicide_severity_and_figurative_language_in_su.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](../../ACL2026/social_computing/persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)

</div>

<!-- RELATED:END -->
