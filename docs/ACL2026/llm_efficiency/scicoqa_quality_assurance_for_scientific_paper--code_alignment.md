---
title: >-
  [Paper Note] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment
description: >-
  [ACL 2026][LLM Efficiency][Paper-Code Alignment] SciCoQA is the first benchmark for detecting discrepancies between scientific papers and their code implementations, revealing that the strongest models detect only 46.7% of real discrepancies.
tags:
  - ACL 2026
  - LLM Efficiency
  - Paper-Code Discrepancy Detection
  - Scientific Reproducibility
  - Cross-Modal Verification
  - LLM Evaluation
content_hash: 99b40fded80dd8f1
---

# SciCoQA: Quality Assurance for Scientific Paper–Code Alignment

**Conference**: ACL 2026
**arXiv**: [2601.12910](https://arxiv.org/abs/2601.12910)
**Code**: [https://github.com/ukplab/scicoqa](https://github.com/ukplab/scicoqa)
**Area**: Scientific Reproducibility / Paper-Code Consistency Verification
**Keywords**: Paper-Code Discrepancy Detection, Scientific Reproducibility, Cross-Modal Verification, LLM Evaluation, Quality Assurance

## TL;DR
SciCoQA is the first benchmark for detecting semantic discrepancies between scientific papers and their code implementations, containing 635 discrepancy instances (92 real + 543 synthetic). Evaluation of 22 LLMs reveals the strongest model detects only 46.7% of real discrepancies, uncovering a critical capability gap in automated scientific quality assurance.

## Method

### Key Designs

1. **Strict Discrepancy Definition with Three-Type Classification**: Difference (code logic differs from paper), Paper Omission (code contains undescribed components), Code Omission (paper-described steps missing in code). Explicitly excludes bugs, CLI-resolvable hyperparameter differences, and standard engineering practices.

2. **Six-Category Impact Taxonomy**: Algorithm, Model, Loss, Evaluation, Data, Training.

3. **Synthetic Data Generation Pipeline**: Extends dataset from CS/AI to physics, statistics, and quantitative biology. Real-synthetic detection rate correlation reaches $r = 0.94$.

## Key Experimental Results

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| GPT-5 | 88.0 | 51.2 | 64.7 |
| Gemini 2.5 Pro | 94.6 | 41.1 | 57.3 |

- Recall is the core bottleneck: models find mostly correct matches but miss too many
- Paper Omission is hardest to detect; longer inputs consistently degrade performance
- Data contamination significantly affects results: detection rates lower on 2025 papers

## Highlights & Insights
- Fills a critical gap by formalizing paper-code consistency verification as a benchmarkable NLP task
- "High precision, low recall" insight: in verification scenarios, missed discrepancies provide false security

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](../../ICLR2026/llm_efficiency/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)
- [\[NeurIPS 2025\] SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](../../NeurIPS2025/llm_efficiency/sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)
- [\[ACL 2026\] SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)

<!-- RELATED:END -->
