---
title: >-
  [Paper Note] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling
description: >-
  [ICLR 2026][LLM Evaluation][LLM ensembling] This paper proposes SAFE (Stable And Fast LLM Ensembling), which selectively ensembles multiple heterogeneous-tokenizer LLMs at the token level via a Generate-Verify-Ensemble l…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "LLM ensembling"
  - "tokenization mismatch"
  - "OOV-like token"
  - "speculative ensembling"
  - "probability distribution alignment"
date: 2026-05-08
content_hash: 9280b605b11a251a
---

# When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling

**Conference**: ICLR 2026
**arXiv**: [2510.15346](https://arxiv.org/abs/2510.15346)  
**Code**: [https://github.com/yoon6503/SAFE](https://github.com/yoon6503/SAFE)  
**Area**: LLM Evaluation
**Keywords**: LLM ensembling, tokenization mismatch, OOV-like token, speculative ensembling, probability distribution alignment

## TL;DR
This paper proposes SAFE (Stable And Fast LLM Ensembling), which selectively ensembles multiple heterogeneous-tokenizer LLMs at the token level via a Generate-Verify-Ensemble loop. SAFE addresses OOV-like contamination caused by tokenization mismatch in long-sequence generation, achieving performance gains by ensembling on fewer than 1% of tokens—improving UniTE from 59.6% to 77.4% on MATH500.

## Background & Motivation

**Background**: Probability-level LLM ensembling (aggregating next-token probability distributions) is an effective approach for leveraging complementary strengths across models. Existing methods such as UniTE and GaC perform well on short-answer tasks (multiple choice, direct answer) but degrade or collapse in long-sequence generation (CoT reasoning).

**Limitations of Prior Work**: **(1) OOV-like token problem**—when the token selected during ensembling does not conform to the tokenization scheme of a participating model, that model is forced to predict over an illegal prefix, corrupting its probability distribution and generating erroneous tokens (e.g., "Sofia" may be split as "So"+"fia" by one model but treated as a single token by another; "So" then becomes an OOV-like token for the latter, causing garbled output such as "Ã"). Such errors accumulate and amplify over long sequences. **(2) Efficiency problem**—ensembling at every token requires cross-vocabulary alignment, with overhead scaling linearly with sequence length.

**Key Challenge**: UniTE ensembles at every token; under CoT generation, accuracy on MATH500 drops catastrophically from 72.4% to 59.6%, and to 43.4% on EXAONE+Qwen2.5. Ensembling can perform worse than a single model.

**Goal**: In long-sequence generation, **when** (i.e., at which token positions) should ensembling be performed to achieve both stability and efficiency?

**Key Insight**: Two key factors determine where ensembling should occur: (i) whether tokenization boundaries are aligned (to avoid OOV-like contamination), and (ii) whether models share sufficient consensus in their probability distributions (to skip unnecessary ensembling). Inspired by speculative decoding, a drafter–verifier architecture is adopted to reduce the number of autoregressive forward passes.

**Core Idea**: Not every token requires ensembling; ensembling only at positions that are tokenization-safe and where model disagreement exists simultaneously improves both stability and efficiency.

## Method

### Overall Architecture
SAFE designates one model as the drafter (the strongest model, responsible for generation) and the remaining models as verifiers. The process follows a three-step loop: (1) **Generate**—the drafter generates a lookahead sequence of $n$ tokens; (2) **Verify**—verifiers check each token in a single forward pass to determine whether ensembling is needed (OOV-like check + consensus check); (3) **Ensemble**—ensembling is performed only at verified token positions, with optional probability sharpening. The drafter then resumes generation from the ensembled token.

### Key Designs

1. **OOV-like Token Detection**:

    - Function: Determine whether a given drafter token would place any verifier model in an illegal tokenization state.
    - Mechanism: For each verifier $LLM_v$, the drafter's sequence $\mathbf{t}_{<i+n}$ is re-tokenized using the verifier's tokenizer. If the drafter token $t_j$ falls within a token boundary of $LLM_v$ (i.e., $\text{Decode}(\mathbf{t}_{<j+1})$ does not align with any tokenization boundary of $LLM_v$), then $t_j$ is flagged as OOV-like, and ensembling at position $t_{j+1}$ is **skipped**.
    - Design Motivation: This prevents OOV-like contamination at the root rather than correcting errors post hoc, by proactively avoiding ensembling at unsafe positions.

2. **Ensemble Distribution Verification (Consensus Detection)**:

    - Function: Determine whether the drafter's token is already the argmax of the ensemble distribution; if so, skip ensembling to improve efficiency.
    - Mechanism: Two sufficient conditions (either triggers a skip):
        - **Unanimous consensus**: All verifiers' argmax tokens agree with the drafter's token.
        - **Average probability > 0.5**: The average probability of the token across all models exceeds 0.5.
    - Design Motivation: Avoid the overhead of constructing ensemble distributions unnecessarily. The paper proves that both conditions guarantee the skipped token is indeed the argmax of the ensemble distribution, incurring no accuracy loss.

3. **Probability Sharpening**:

    - Function: When the ensemble distribution is overly smooth (max < 0.5), concentrate probability mass on the most likely token.
    - Mechanism: Two strategies—(a) **Heuristic suffix merging**: redistribute probabilities of subword token variants to their common prefix (e.g., merge probabilities of "Sofia" and "SofiaŢ" into "So"), applied only to drafter tokens with probability > $\lambda$; (b) **Geometric mean instead of arithmetic mean**: penalize tokens that receive low probability from any model, concentrating mass on tokens consistently supported by all models.
    - Design Motivation: Heterogeneous tokenization scatters probability mass for the same word across multiple subword tokens; direct arithmetic averaging yields overly smooth distributions.

4. **KV Cache Management**:

    - Function: Maintain KV cache consistency after token replacement during ensembling.
    - Mechanism: After each ensembling step, the KV caches of all models are updated to align with the ensembled output.
    - Design Motivation: Prior methods abandoned KV caching due to difficulties handling cache inconsistencies introduced by ensembling. SAFE is the first to implement complete KV cache management across heterogeneous tokenizers.

### Loss & Training
SAFE is a training-free inference-time method involving no loss functions. It is a plug-and-play framework that can be directly integrated with existing ensembling methods (UniTE, GaC, etc.).

## Key Experimental Results

### Main Results
CoT ensembling with three heterogeneous-tokenizer models (Internlm3-8B + Qwen2.5-7B + EXAONE3.5-7.8B):

| Method | MMLU-redux | MATH500 | GSM8K | BBH | ARC-C | Avg |
|--------|-----------|---------|-------|-----|-------|-----|
| Best Single Model | 76.89 | 74.8 | 91.81 | 82.26 | 90.44 | 82.87 |
| UniTE (per-token ensemble) | 73.39 | 59.6 | 75.06 | 79.58 | 87.97 | 75.12 |
| UniTE + SAFE (2 models) | **77.81** | **77.4** | **92.04** | **82.97** | 90.78 | **84.20** |
| UniTE + SAFE (3 models) | 77.60 | **79.0** | 92.04 | 82.77 | **91.55** | **84.59** |
| GaC | 77.00 | 74.2 | 91.28 | 82.34 | 90.61 | 83.09 |
| GaC + SAFE | 77.11 | 76.0 | 91.36 | 82.34 | 91.13 | 83.59 |

### Ablation Study

| Component | MATH500 | Notes |
|-----------|---------|-------|
| UniTE (no SAFE) | 59.6 | Per-token ensembling, severe OOV contamination |
| + OOV-like detection | ~74 | OOV check alone yields substantial recovery |
| + Consensus skipping | ~76 | Reducing unnecessary ensembling further improves |
| + Probability sharpening | 77.4 | Full SAFE |
| E/T ratio (MATH500) | 3.82% | Ensembling on <4% of tokens |
| E/T ratio (general domain) | ~15% | General domain requires more ensembling |

### Key Findings
- **SAFE recovers UniTE on MATH500 from 59.6% to 79.0%** (3-model), surpassing the best single model by 4.2%.
- **Mathematical tasks require almost no ensembling**: E/T ratio is only 4.85%, as equations and structured expressions exhibit low variability, leading to high inter-model consensus.
- **General-domain tasks require more ensembling**: E/T ratio is approximately 15%, as greater linguistic variability leads to more model disagreement.
- **2-model ensembling often outperforms 3-model ensembling**: selecting the top-2 models when model rankings are known is more effective.
- **Latency approaches that of a single model**: SAFE's inference speed is nearly equivalent to single-model inference, owing to the speculative strategy, selective ensembling, and KV cache management.
- **Geometric mean sharpening is unstable across datasets**; heuristic prefix merging is more robust.

## Highlights & Insights
- **Precise problem formulation**: The concept of OOV-like tokens captures the core challenge of cross-tokenizer ensembling. Prior methods entirely overlooked this issue; the absence of failures on short-answer tasks was coincidental, as short answers rarely expose tokenization mismatches.
- **"When to ensemble" matters more than "how to ensemble"**: SAFE's core contribution lies not in a novel ensembling algorithm but in determining where to ensemble. The finding that ensembling on fewer than 1% of tokens outperforms full-token ensembling fundamentally reframes the problem.
- **Elegant transfer of speculative decoding**: The drafter–verifier architecture transfers speculative decoding from "accelerating a single model" to "accelerating multi-model ensembling," reducing forward pass counts while naturally resolving asynchronous generation.
- **Strong practical utility**: The method is plug-and-play, training-free, supports heterogeneous tokenizers, and includes complete KV cache management, enabling direct deployment.

## Limitations & Future Work
- **Only models at the 7B–8B scale are evaluated**: Appendix includes 32B experiments, but main results are limited to small models; effectiveness at 70B+ is unknown.
- **Drafter selection requires prior model ranking**: One must know which model is strongest to serve as the drafter.
- **Fixed lookahead length $n$**: $n=5$ may not be optimal in all settings; adaptive adjustment could be beneficial.
- **Probability sharpening strategies are relatively coarse**: The heuristic prefix merging may behave differently for certain languages (e.g., Chinese).
- **Future directions**: Exploring adaptive $n$ and adaptive drafter selection; extending SAFE to larger ensembles (>3 models); investigating applications in RLHF/alignment settings (e.g., ensembling aligned and unaligned models).

## Related Work & Insights
- **vs. UniTE**: UniTE ensembles at every token, causing catastrophic degradation under CoT. SAFE ensembles on fewer than 20% of tokens and surpasses both UniTE and single models.
- **vs. GaC**: GaC ensembles when the primary model's probability falls below 0.5, incidentally avoiding some OOV issues but without a systematic solution. SAFE further improves upon GaC.
- **vs. Speculative Decoding**: Speculative decoding requires drafter and target to share the same tokenizer. SAFE is the first to extend this paradigm to heterogeneous tokenizer settings.
- **vs. Post-inference ensembling (MoA, etc.)**: These methods aggregate complete responses, bypassing token-level issues but requiring multiple full inference passes. SAFE operates at the token level with efficiency approaching that of a single model.

## Rating
- Novelty: ⭐⭐⭐⭐ The identification of OOV-like tokens and the systematic solution constitute a significant contribution, though the overall approach (selective ensembling + speculative decoding) is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, multiple model combinations (2/3 models), efficiency analysis, ablation studies, and 32B scaling experiments provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated (Figures 1/2 are highly intuitive) and algorithm descriptions are rigorous, though some notation is slightly complex.
- Value: ⭐⭐⭐⭐ Addresses a critical barrier to practical deployment of probability-level LLM ensembling, making cross-heterogeneous-tokenizer ensembling genuinely viable.

## Background & Motivation
1. **Limitations of Prior Work**: Probability-level ensembling fails in long-sequence generation—tokenization scheme mismatches produce OOV-like tokens that corrupt probability distributions, and per-token ensembling is inefficient.
2. **Core Idea**: Ensemble only at positions where verifiers disagree, avoiding OOV contamination and redundant operations at consensus positions.

## Method
- **Generate-Verify-Ensemble loop**: The drafter generates $n$ lookahead tokens → verifiers identify ensembling points → selective ensembling is performed.
- **OOV-like verification**: Checks whether token boundaries align with the tokenization schemes of all verifiers.
- **Ensemble distribution verification**: Skip when unanimous consensus OR average probability > 0.5 holds; apply probability sharpening when the distribution is overly smooth.

## Key Experimental Results

| Task | UniTE | UniTE+SAFE | Best Baseline |
|------|-------|-----------|---------------|
| MATH500 | 59.6% | **77.6%** | 72.4% |
| ARC-C | 87.97% | **90.78%** | 90.27% |

## Key Findings
- Selective ensembling is more stable and efficient than full-token ensembling.
- OOV-like tokens are the root cause of failure in cross-model ensembling.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic solution to cross-tokenizer ensembling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on MATH500 and ARC-C.
- Value: ⭐⭐⭐⭐ A practical solution for multi-LLM collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](../../ACL2026/llm_evaluation/cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[ICLR 2026\] Enabling Fine-Grained Operating Points for Black-Box LLMs](enabling_fine-grained_operating_points_for_black-box_llms.md)
- [\[ICLR 2026\] When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](../../ACL2026/llm_evaluation/multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)

</div>

<!-- RELATED:END -->
