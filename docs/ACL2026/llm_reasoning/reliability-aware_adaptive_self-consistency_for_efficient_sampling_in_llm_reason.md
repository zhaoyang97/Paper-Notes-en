---
title: >-
  [Paper Note] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning
description: >-
  [ACL2026][LLM Reasoning][Self-Consistency] ReASC transforms adaptive self-consistency from "counting answer votes" into "judging whether reliable evidence is sufficient" by using response-level confidence to weight Beta…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Self-Consistency"
  - "Reasoning Sampling"
  - "Confidence Estimation"
  - "Adaptive Stopping"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: d3a6dfa4668cc875
---

# Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning

**Conference**: ACL2026  
**arXiv**: [2601.02970](https://arxiv.org/abs/2601.02970)  
**Code**: Not yet public  
**Area**: llm_reasoning  
**Keywords**: Self-Consistency, Reasoning Sampling, Confidence Estimation, Adaptive Stopping, Inference Efficiency

## TL;DR
ReASC transforms adaptive self-consistency from "counting answer votes" into "judging whether reliable evidence is sufficient" by using response-level confidence to weight Beta accumulation. It significantly reduces multi-sample inference costs on GSM8K, MATH500, Omni-Math, and GPQA-Diamond while maintaining near-original accuracy.

## Background & Motivation
**Background**: Self-consistency (SC) significantly improves the reliability of LLMs in mathematical and complex reasoning tasks by sampling multiple reasoning paths and performing majority voting. However, it typically uses a fixed sampling budget of $k$ outputs, spending the same budget on both simple and difficult problems.

**Limitations of Prior Work**: Existing methods like Adaptive Consistency and Early-Stopping Self-Consistency dynamically stop based on observed answers, but their core evidence remains answer counts or consistency within a window. This assumes every response contains the same amount of information, ignoring that some reasoning trajectories are inherently more reliable while others are low-confidence noise.

**Key Challenge**: The fundamental decision during inference should be "whether the current evidence is sufficient to support a reliable answer," rather than "how many times an answer appeared." If early high-confidence responses already provide strong evidence, continued sampling wastes computation; if low-confidence answers appear frequently, pure counting might lead to premature or incorrect aggregation.

**Goal**: The authors aim to design a training-free framework that works solely during inference. It uses the model's own confidence signals to determine if a single sample is sufficient and allows high-confidence responses to contribute more evidence when multiple samples are needed.

**Key Insight**: The paper interprets response-level confidence as "evidence strength" and adopts Bottom 10% Group Confidence to capture the most unstable, low-confidence segments within a reasoning chain. this signal distinguishes between correct and incorrect answers more effectively than average self-certainty.

**Core Idea**: A confidence gating mechanism first handles samples where the "single response is already reliable enough." For remaining samples, confidence-weighted Beta posterior updates are performed to achieve a decision reliability close to standard self-consistency with fewer samples.

## Method
ReASC is a pure inference-stage method that does not modify model parameters. It splits the reasoning process for each question into two stages: the first stage uses single-response confidence for early stopping, and the second stage continues sampling for questions that fail the gate, treating each response's confidence as a soft count in a Beta update. Unlike ASC/ESC, ReASC's stopping criterion considers not only answer frequency but also response reliability.

### Overall Architecture
Given a problem, the model first generates one reasoning response and calculates the Bottom 10% Group Confidence from the token probability distribution. If this confidence exceeds a calibrated threshold, ReASC accepts the answer directly. Otherwise, it enters Stage 2 to sample additional responses. Each response is categorized by its answer, and its confidence adds weighted evidence to that category. The system continuously calculates the Beta posterior probability of the leading answer maintaining its advantage over the runner-up until it exceeds a stopping threshold or reaches the maximum budget.

### Key Designs
1. **Bottom 10% Group Confidence as a Reliability Signal**:

    - **Function**: Estimates whether a reasoning trajectory is reliable during generation.
    - **Mechanism**: The token sequence is partitioned into sliding windows (groups). Token-level self-certainty is calculated for each group, and the average of the lowest 10% of groups is used as the response-level confidence. Compared to a global average, it focuses on the weakest, most error-prone local segments of the reasoning chain.
    - **Design Motivation**: Erroneous reasoning is often not low-confidence throughout but contains local uncertainties at critical steps. Aggregating tail low-confidence scores exposes this risk better than mean values.

2. **Single-sample Gating Decision**:

    - **Function**: Avoids redundant sampling for simple cases that are already reliable.
    - **Mechanism**: After the first response, the confidence $S(y)$ is calculated. If $S(y) \geq \tau_{gate}$, the answer is accepted. In the offline setting, a labeled calibration set estimates the mean of correct samples and the threshold for the target accuracy. in the online setting without labels, a two-component GMM fits the confidence distribution, approximating the high-confidence component as the distribution of correct answers.
    - **Design Motivation**: Many problems are reliable at pass@1; multi-sample voting is wasteful for these cases. Gating turns "the need for self-consistency" into an instance-level judgment.

3. **Confidence-weighted Beta Evidence Accumulation**:

    - **Function**: Allows reliable answers to drive stopping faster when multiple samples are required.
    - **Mechanism**: In ASC, counts of the top and runner-up answers form a $Beta(v_1+1, v_2+1)$. ReASC normalizes each response's confidence to $z(y)$ and uses $\max(1, \exp(\lambda z(y)))$ as a soft count increment. It then calculates $1-I_{1/2}(\alpha, \beta)$, stopping when the probability of the leading answer maintaining its advantage exceeds $C_{threshold}=0.95$.
    - **Design Motivation**: Frequency represents quantity of evidence, while confidence reflects quality. Weighted updates allow high-confidence consistent answers to reach posterior certainty faster while retaining the robust Beta framework of ASC.

### Loss & Training
ReASC does not involve model training; it only requires inference-time confidence calculation and threshold calibration. Experiments use LLaMA-3.2-3B, Qwen-2.5-3B/7B, and Gemma-3-4B/27B. Offline calibration uses 128 held-out samples with a target accuracy $p_{target}=0.9$. Online calibration fits a GMM from the confidence distribution of the first responses in the test set. Stage 2 uses $C_{threshold}=0.95$, $\lambda=0.7$, and aligns maximum budgets with SC $k=16$.

## Key Experimental Results

### Main Results
The main results demonstrate that ReASC generally maintains accuracy close to SC/ASC across various models and datasets while significantly reducing TFLOPs.

| Model / Dataset | Method | Acc ↑ | TFLOPs ↓ | Acc/TF ↑ | Relative cost vs SC |
|-------------|------|-------|----------|----------|-----------------|
| Gemma-3-4B / GSM8K | SC | 92.12 | 32.67 | 2.82 | - |
| Gemma-3-4B / GSM8K | ASC | 92.12 | 12.26 | 7.52 | -62.5% |
| Gemma-3-4B / GSM8K | ReASC offline | 92.04 | 9.45 | 9.74 | -71.1% |
| Qwen-2.5-7B / MATH500 | SC | 80.6 | 71.59 | 1.13 | - |
| Qwen-2.5-7B / MATH500 | ASC | 80.8 | 37.25 | 2.17 | -48.0% |
| Qwen-2.5-7B / MATH500 | ReASC offline | 81.2 | 29.26 | 2.78 | -59.1% |
| Gemma-3-27B / GSM8K | SC | 97.04 | 166.93 | 0.58 | - |
| Gemma-3-27B / GSM8K | ReASC offline | 96.89 | 29.36 | 3.30 | -82.4% |

### Ablation Study
Stage 1 analysis shows that a large number of problems can be reliably solved by a single sample, with accepted samples typically exceeding 90% accuracy.

| Model | Dataset | Calibration | Stage 1 Accept % | Accepted Acc |
|------|--------|----------|------------------|----------------|
| LLaMA-3.2-3B | GSM8K | Offline | 48.98 | 91.33 |
| Gemma-3-4B | GSM8K | Offline | 51.18 | 97.78 |
| Qwen-2.5-7B | GSM8K | Offline | 59.59 | 97.58 |
| Gemma-3-27B | GSM8K | Offline | 60.58 | 98.62 |
| Qwen-2.5-7B | MATH500 | Online | 31.8 | 93.08 |
| Gemma-3-27B | MATH500 | Online | 36.2 | 97.31 |

Stage 2 ablation indicates that cost savings are not solely from Phase 1; even excluding accepted Stage 1 samples, Stage 2 is more efficient than count-based ASC.

| Model / Dataset | Method | Acc ↑ | TFLOPs ↓ | Note |
|-------------|------|-------|----------|------|
| LLaMA-3.2-3B / GSM8K | ASC | 83.85 | 6.27 | Count-based stopping |
| LLaMA-3.2-3B / GSM8K | ReASC Stage2 only | 84.38 | 5.33 | Weighted Beta reduces sampling |
| LLaMA-3.2-3B / GSM8K | ReASC | 83.85 | 4.38 | Stage 1 further reduces cost |
| Qwen2.5-7B / MATH500 | ASC | 80.80 | 37.25 | Count-based stopping |
| Qwen2.5-7B / MATH500 | ReASC Stage2 only | 81.20 | 34.05 | Weighted accumulation is more efficient |
| Qwen2.5-7B / MATH500 | ReASC | 81.20 | 29.26 | Twond stages are complementary |

### Key Findings
- ReASC's advantages hold across different model scales (3B to 27B); stronger models tend to have higher Stage 1 acceptance rates.
- Online calibration works effectively without labels, providing a better accuracy-cost trade-off than SC/ASC on Omni-Math and GPQA-Diamond.
- Bottom 10% Group Confidence yields an AUROC of 0.860, surpassing the 0.823 of average group confidence, confirming that low-confidence local segments are better at distinguishing reasoning quality.
- Accuracy in Qwen2.5-7B monotonically increases from 20.00% (lowest 20% confidence bin) to 93.27% (highest 20%), supporting the "high confidence is more reliable" hypothesis.
- ReASC online also achieves the highest Acc/TF on StrategyQA, Last Letter Concatenation, and NQ-Open, suggesting it generalizes beyond math tasks.

## Highlights & Insights
- The paper interprets self-consistency sampling as evidence accumulation. This perspective naturally explains why counting votes is insufficient: two high-reliability responses should carry more weight than two low-reliability ones.
- Stage 1 is a highly practical design. In many deployment scenarios, simple requests dominate; determining if pass@1 is reliable can avoid massive amounts of redundant sampling.
- Using Bottom 10% Group Confidence is an intuitive choice because reasoning errors are often triggered by a few vulnerable steps. Focusing on low-confidence segments matches the failure modes of chain-of-thought reasoning better than global averages.
- The method requires no new training and no external verifiers, making it easy to integrate into existing self-consistency services. If the server provides token logprobs, the overhead is negligible.

## Limitations & Future Work
- ReASC relies on the assumption that a model's self-confidence correlates with correctness. While experiments support this trend, confidence might be distorted by systematic overconfidence, strong hallucinations, or OOD tasks.
- Bottom 10% Group Confidence requires access to token-level probability distributions, which some closed-source APIs or high-throughput frameworks might not provide consistently.
- Online calibration uses GMM fitting on test set confidence; threshold estimation may become unstable if the distribution is not clearly bimodal.
- The paper focus on computation and latency but does not deeply explore the synergy with verifiers, process reward models (PRMs), or tree-search reasoning.

## Related Work & Insights
- **vs Self-Consistency**: SC uses fixed sampling ($k=16$), whereas ReASC stops dynamically based on evidence sufficiency, leading to much lower costs at similar accuracy levels.
- **vs ASC / ESC**: While ASC and ESC rely on counts or window consistency, ReASC incorporates confidence as a soft count weighting within the same Beta framework to model evidence quality.
- **vs verifier / reranker**: Verifiers usually require additional models or training data. ReASC leverages endogenous model confidence, making it lighter to deploy but more sensitive to the quality of confidence calibration.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using confidence as evidence weight for adaptive self-consistency is clear and effective; the statistical framework builds solidly on ASC.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, datasets, calibration modes, and extended tasks with sufficient evidence.
- Writing Quality: ⭐⭐⭐⭐☆ The method is clearly described, and formulas support the empirical claims.
- Value: ⭐⭐⭐⭐⭐ Highly practical for LLM services needing multi-sample reasoning with cost constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[ACL 2026\] SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](../../NeurIPS2025/llm_reasoning/sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)

</div>

<!-- RELATED:END -->
