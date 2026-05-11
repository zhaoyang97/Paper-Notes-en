---
title: >-
  [Paper Note] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty
description: >-
  [ICLR 2026][Image Generation][Unsupervised conformal inference] This paper proposes an unsupervised conformal inference framework (BB-UCP) that achieves distribution-free finite-sample coverage guarantees for LLM generat…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Unsupervised conformal inference"
  - "bootstrapping"
  - "LLM hallucination detection"
  - "Gram matrix"
  - "conformal alignment"
date: 2026-05-08
content_hash: 58220dd2298f5558
---

# Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty

**Conference**: ICLR 2026
**arXiv**: [2509.23002](https://arxiv.org/abs/2509.23002)
**Code**: None
**Area**: Image Generation
**Keywords**: Unsupervised conformal inference, bootstrapping, LLM hallucination detection, Gram matrix, conformal alignment

## TL;DR
This paper proposes an unsupervised conformal inference framework (BB-UCP) that achieves distribution-free finite-sample coverage guarantees for LLM generation under label-free, API-compatible conditions, via Gram matrix interaction energy scoring, batch bootstrap calibration, and conformal alignment, effectively detecting and filtering hallucinated outputs.

## Background & Motivation

### State of the Field

**Background**: Uncertainty quantification (UQ) for LLMs is critical for trustworthy AI. In black-box API settings, gradients, logits, and hidden states are inaccessible, requiring decisions based solely on sampled outputs. Existing methods include semantic entropy, self-consistency, and embedding-geometry-based approaches.

**Limitations of Prior Work**: (1) Conformal prediction (CP) provides distribution-free finite-sample guarantees, but generative tasks break the classical supervised setup — text prompts are not quantifiable covariates; (2) Full-UCP requires recomputation for each candidate (computationally intensive), while Split-UCP is data-inefficient (high splitting cost); (3) Existing methods fail to align low-cost signals with high-cost quality metrics through calibration.

**Key Challenge**: A label-free, API-compatible, theoretically grounded test-time filtering mechanism is needed, yet existing CP methods are ill-suited for unsupervised generative settings.

**Goal**: How to provide theoretically guaranteed quality control for LLM generation under a label-free, black-box-API-only setting?

**Key Insight**: Leveraging the geometric signal of the Gram matrix of response embeddings as a consistency score, and designing batched, bootstrap-enhanced unsupervised conformal prediction, with conformal alignment to calibrate cheap signals against quality predicates.

**Core Idea**: Gram matrix interaction energy quantifies response typicality; bootstrap calibration stabilizes thresholds; conformal alignment binds geometric signals to factuality objectives.

## Method

### Overall Architecture
A three-layer framework: (1) Gram matrix atypicality scoring — measuring the alignment of each response with others in the same group; (2) BB-UCP — batch bootstrap calibration for exact quantile estimation; (3) Conformal alignment — calibrating a single strictness parameter $\tau$ so that a user-defined predicate holds on unseen batches with probability $\geq 1-\alpha$.

### Key Designs

1. **Atypicality Scoring**:

    - Interaction energy $e(i;G) = \|G_{:,i}\|_2 = \|Vv_i\|_2$; under unit-norm embeddings, $e(i;G) = (\sum_j \cos^2\theta_{ij})^{1/2}$
    - Bounds: $1 \leq e(i;G) \leq \sqrt{n}$ (Theorem 2.1)
    - Atypicality score $\Phi(i;G) = 1 - {e(i;G)}/{B_E}$; higher values indicate greater novelty/atypicality

2. **Batch-Bootstrapped UCP (BB-UCP)**:

    - B-UCP: within-batch leave-one-out residuals $R_{j,i} = \phi(Y_{j,i}; \mathcal{B}_{j,-i})$, aggregated across batches and thresholded at an adjusted quantile
    - BB-UCP: applies bootstrapping over per-batch residuals on top of B-UCP to stabilize empirical quantiles and reduce the influence of outlier batches
    - Coverage guarantees (Theorems 3.1–3.2): under batch exchangeability, $\Pr\{Y_{n+1} \in C_n\} \geq 1-\alpha$

3. **Conformal Alignment**:

    - Parameterized strictness $\tau \in [0,1]$; filtered retention set $\hat{J}_j(\tau) = \{i: Q_{j,i} > \tau\}$
    - Predicate $\mathcal{P}_j^{\text{CVAR}}(\tau)$ evaluates the quality gap between the retained and discarded sets based on CVaR difference
    - Calibrated $\hat{\tau}$ is the $K = \lceil(1-\alpha)(J+1)\rceil$-th order statistic of minimum passing strictness over historical batches
    - Alignment guarantee (Theorem 3.3): $\Pr\{\mathcal{P}_{J+1}(\hat{\tau}) = 1\} \geq 1-\alpha$

### Evaluation Metrics
Factuality severity $\text{FS}(a) = 1 - \max_{r \in \mathcal{R}_q} \text{BERTScoreF1}(\text{head}(a), r)$, evaluated only on the answer head (first sentence or Final field), truncated to 16 tokens.

### Loss & Training
The model is trained end-to-end with an objective combining task loss and regularization terms.

## Key Experimental Results

### Experiment 1: Single-Query Calibration (S-UCP vs. BB-UCP)
- BB-UCP yields smaller interval lengths across all datasets and all $\alpha$ levels
- BB-UCP coverage is more conservative (above target), yet produces tighter thresholds

### Experiment 2: Cross-Query Calibration
- LOQO empirical coverage is close to the $1-\alpha$ target across all datasets
- $\Delta\text{FS} > 0$ holds across all datasets and all $\alpha$ levels (retained sets exhibit better factuality)
- $\Delta\text{FS} \approx 0.209$ on NQ-Open (largest improvement)

### Experiment 3: Conformal Alignment (CVaR-gap)
- Reductions in factuality severity are consistently positive across all datasets and risk levels
- NQ-Open and NQ-Open-Vend exhibit the largest median improvements

### Key Findings
- The bootstrap mechanism effectively stabilizes threshold estimation, producing tighter intervals while maintaining conservative coverage
- Gram matrix geometric signals are highly correlated with factuality without requiring logits or gradients
- In low-pool/low-diversity settings (NQ-Open), coverage is slightly below target, yet factuality gains remain significant
- Conformal alignment effectively bridges cheap geometric signals and expensive factuality metrics

## Highlights & Insights
- **Fully label-free and API-compatible**: requires no token probabilities, gradients, or annotated data — only sampled outputs and embeddings
- **Seamless integration of theory and practice**: distribution-free finite-sample guarantees (without model-specific assumptions) combined with practical hallucination detection
- **Elegant three-layer progressive design**: scoring → calibration → alignment, each layer addressing one core problem
- **Generality of conformal alignment**: adaptable to any non-decreasing right-continuous predicate, enabling broad applicability

## Limitations & Future Work
- The batch exchangeability assumption may be violated in real-world deployment
- Lightweight embedding models (MiniLM) may lose subtle semantic distinctions
- Gram matrix discriminability degrades in low-diversity generation scenarios (e.g., simple factual questions)
- Applying this framework to hallucination detection in multimodal LLMs is a promising direction

## Related Work & Insights
- **vs. Semantic Entropy**: semantic entropy requires defining equivalence classes; the proposed method directly leverages embedding geometry
- **vs. Conformal Risk Control**: this work extends to fully unsupervised batch settings
- **vs. QRM/URM**: those methods require preference data for training; the proposed method requires no annotation whatsoever

## Supplementary Details
- Datasets: ASQA (ambiguous), NQ-Open (single-hop factual), HotpotQA (multi-hop compositional), AmbigQA (aliases and answer sets)
- Embedding model: all-MiniLM-L6-v2 (lightweight sentence encoder)
- BERTScore uses roberta-large with baseline rescaling
- Ablations include decoding entropy stress tests and vendor/model switching
- Supports multi-vendor APIs including OpenAI, Together, and Gemini
- Bootstrap resampling incurs low computational overhead, preserves exchangeability, and is directly deployable

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of unsupervised conformal prediction, Gram matrix geometry, and conformal alignment is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ Four QA datasets, three experimental dimensions, two ablation analyses
- Writing Quality: ⭐⭐⭐⭐ Theoretically rigorous but notation-heavy; readers are expected to have a CP background
- Value: ⭐⭐⭐⭐⭐ Provides a practically needed quality control tool for LLM deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICLR 2026\] From Parameters to Behaviors: Unsupervised Compression of the Policy Space](from_parameters_to_behaviors_unsupervised_compression_of_the_policy_space.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](glass_flows_reward_alignment_diffusion.md)
- [\[ICLR 2026\] Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations](translate_policy_to_language_flow_matching_generated_rewards_for_llm_explanation.md)

</div>

<!-- RELATED:END -->
