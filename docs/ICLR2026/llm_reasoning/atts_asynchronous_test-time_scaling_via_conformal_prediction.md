---
title: >-
  [Paper Note] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction
description: >-
  [ICLR 2026][LLM Reasoning][test-time scaling] This paper proposes ATTS, an asynchronous test-time scaling framework based on conformal prediction that eliminates synchronization overhead by reformulating rejection sampli…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "test-time scaling"
  - "speculative decoding"
  - "conformal prediction"
  - "asynchronous inference"
  - "rejection sampling"
date: 2026-05-08
content_hash: 9535c18d7a604cca
---

# ATTS: Asynchronous Test-Time Scaling via Conformal Prediction

**Conference**: ICLR 2026
**arXiv**: [2509.15148](https://arxiv.org/abs/2509.15148)
**Code**: [https://github.com/menik1126/Asynchronous-Test-Time-Scaling](https://github.com/menik1126/Asynchronous-Test-Time-Scaling)
**Area**: LLM Reasoning
**Keywords**: test-time scaling, speculative decoding, conformal prediction, asynchronous inference, rejection sampling

## TL;DR
This paper proposes ATTS, an asynchronous test-time scaling framework based on conformal prediction that eliminates synchronization overhead by reformulating rejection sampling as a hypothesis testing procedure. On mathematical reasoning benchmarks such as MATH and AIME, ATTS achieves up to 56.7× speedup and 4.14× throughput improvement without accuracy loss. A 1.5B/70B draft/target model combination reaches the AIME performance level of o3-mini (high).

## Background & Motivation
**Background**: Test-time scaling (increasing computational budget at inference time) substantially enhances LLM reasoning capabilities through sequential scaling (longer reasoning chains) and parallel scaling (more samples). Speculative decoding — where a small model generates candidates verified by a large model — is a natural approach to accelerating inference.

**Limitations of Prior Work**: When speculative decoding meets test-time scaling, two bottlenecks emerge: (1) **Memory bottleneck** — KV cache explosion under high-concurrency sampling leads to GPU out-of-memory errors; (2) **Synchronization overhead** — rejection sampling requires global ranking or softmax normalization over all candidates, causing synchronization wait times that grow exponentially with the number of sampling rounds.

**Key Challenge**: Efficient test-time scaling requires scaling along both parallel and sequential dimensions simultaneously, but global synchronization for ranking and normalization makes asynchronous execution infeasible — all candidates must wait for each other to complete before ranking can proceed.

**Goal**: How can the synchronization bottleneck of rejection sampling in test-time scaling be eliminated while preserving statistical guarantees?

**Key Insight**: Conformal prediction is introduced to construct prediction sets, replacing normalized softmax scores with p-values for ordinal classification, enabling each candidate to be independently accepted or rejected without waiting for global rankings.

**Core Idea**: Replace global ranking with conformal prediction p-values to realize asynchronous rejection sampling, thereby eliminating the synchronization bottleneck in test-time scaling.

## Method

### Overall Architecture
ATTS is a three-stage pipeline: (1) the draft model generates $m$ candidate reasoning chains in parallel; (2) a conformal p-value is used to asynchronously determine whether each candidate falls within the prediction set (accept/reject), without waiting for all candidates; (3) accepted candidates are continued by the target model. Sequential scaling is achieved by increasing the number of rounds; parallel scaling is achieved by increasing the number of candidates.

### Key Designs

1. **Asynchronous Arithmetic Intensity Analysis**:

    - Function: Defines the asynchronous arithmetic intensity $r = T_c / (T_m + T_s) \approx T_c / T_s$ to quantify performance bottlenecks.
    - Mechanism: Traditional arithmetic intensity considers only computation and memory access, but in test-time scaling the synchronization overhead $T_s$ far exceeds memory access time $T_m$ and becomes the true bottleneck. As the number of samples grows, $r$ decreases, indicating that synchronization is the dominant bottleneck.
    - Design Motivation: Provides theoretical motivation and a quantitative tool for asynchronous system design.

2. **Ordinal Classification via Conformal Prediction**:

    - Function: Transforms the global ranking problem into independent hypothesis tests.
    - Mechanism: For each candidate, a conformal p-value $p_\xi^k$ is computed based on the non-normalized conformity score $s_\xi^k = -\ell(X_\xi, \hat{Y}_\xi^k)$; comparing this value against a threshold $\alpha$ determines acceptance or rejection. Crucially, p-values do not require global normalization — the current score is compared only against historical scores in the calibration set.
    - Design Motivation: Avoids the synchronization requirements of softmax normalization and global ranking, enabling each candidate to be evaluated independently and asynchronously.
    - Statistical Guarantee: Both marginal and conditional coverage guarantees are provided — $\mathbb{P}(y \in C_\alpha(Y)) \geq 1 - \alpha$.

3. **Online Calibration + Budget Prediction**:

    - Function: Dynamically maintains the calibration set at test time in the absence of held-out data.
    - Mechanism: A memory bank stores historically sampled scores and is continuously updated as testing proceeds. The rejection rate is precisely controlled by $\alpha$ — the prediction set size equals the predefined budget $B$, preventing GPU out-of-memory errors.
    - Design Motivation: Test-time scaling does not have a reserved calibration set, necessitating online accumulation.

### Loss & Training
No training is required (training-free, lossless). ATTS operates entirely at inference time and does not modify model weights.

## Key Experimental Results

### Main Results (Across Different Draft–Target Model Families)

| Dataset | Draft Model | Target Model | Accuracy | Mar Speedup | Con Speedup |
|---------|------------|-------------|----------|-------------|-------------|
| MATH100 | Qwen2.5-7B-Inst | QwQ-32B | 96.0% (=TM) | **7.19×** | 5.35× |
| AIME24 | Qwen2.5-7B-Inst | QwQ-32B | 46.7% | **5.71×** | 10.10× |
| AIME25 | Qwen2.5-7B-Inst | QwQ-32B | 40.0% | **14.50×** | 12.82× |
| AMC23 | Qwen2.5-7B-Inst | QwQ-32B | 76.0% | **10.42×** | 8.20× |

### Large-Scale Scaling Results

| Configuration | Description |
|---------------|-------------|
| Up to 56.7× speedup | Under test-time scaling settings |
| 4.14× throughput improvement | With simultaneous sequential and parallel scaling |
| 1.5B/70B draft/target | Reaches o3-mini (high) AIME performance level |
| Rejection rate accurately controlled | Highly consistent with the predefined $\alpha$ |

### Key Findings
- **Cross-family draft–target combinations are effective**: Even when the draft and target models come from different model families (Qwen → QwQ, Llama → QwQ), ATTS still provides substantial speedup.
- Results marked in red indicate "lossless acceleration" — post-acceleration accuracy equals or exceeds the target model baseline.
- The asynchronous approach shows a clear advantage when the number of samples is large — synchronization overhead grows exponentially, whereas asynchronous overhead remains constant.
- Conditional coverage (per-instance guarantee) is generally more conservative but more reliable than marginal coverage; the appropriate choice involves a trade-off depending on the scenario.

## Highlights & Insights
- **Innovative application of conformal prediction to LLM inference acceleration**: Introducing conformal prediction from statistics into speculative decoding, replacing global ranking with hypothesis testing, represents an elegant integration of theory and engineering.
- **Asynchronous arithmetic intensity metric**: Provides a new quantitative tool for characterizing test-time scaling bottlenecks, which can guide system design decisions.
- **Engineering practicality of "lossless acceleration"**: Training-free, model-agnostic, and statistically guaranteed, ATTS is directly deployable.

## Limitations & Future Work
- Online calibration requires accumulation of sufficient historical scores; accuracy during the cold-start phase may be limited.
- Accuracy under certain draft–target combinations falls below the target model baseline (particularly with weaker draft models), indicating that draft model quality remains important.
- Evaluation is conducted exclusively on mathematical reasoning tasks; applicability to open-ended generation tasks is unknown.
- Deploying both draft and target models simultaneously incurs additional GPU resource requirements.

## Related Work & Insights
- **vs. Standard Speculative Decoding**: ATTS extends speculative decoding from the token level to the chain level (entire reasoning chains) and addresses the synchronization bottleneck unique to test-time scaling settings.
- **vs. Best-of-N (BoN)**: BoN requires all $N$ reasoning chains to be fully generated before selection, whereas ATTS enables asynchronous incremental filtering, substantially reducing latency.
- **vs. TPT Early-Stopping Methods**: Early stopping risks pruning correct reasoning paths; ATTS uses conformal guarantees to ensure high-quality candidates are not discarded.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of conformal prediction and asynchronous test-time scaling is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, diverse draft–target combinations, and dual evaluation of speedup and accuracy.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear system analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a practical framework for efficient deployment of test-time scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](../../ACL2026/llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)

</div>

<!-- RELATED:END -->
