---
title: >-
  [Paper Note] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models
description: >-
  [NeurIPS 2025][LLM Pretraining][LLM alignment] This paper proposes the Residual Alignment Model (RAM), which formalizes the LLM alignment process as importance sampling and decomposes a large model into a frozen Proposal…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "LLM alignment"
  - "importance sampling"
  - "residual alignment"
  - "modular alignment"
  - "token-level decoding"
  - "parameter-efficient"
date: 2026-05-08
content_hash: 91b177883dc934f0
---

# Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.19700](https://arxiv.org/abs/2505.19700)
**Code**: To be confirmed
**Area**: LLM Pretraining
**Keywords**: LLM alignment, importance sampling, residual alignment, modular alignment, token-level decoding, parameter-efficient

## TL;DR

This paper proposes the Residual Alignment Model (RAM), which formalizes the LLM alignment process as importance sampling and decomposes a large model into a frozen Proposal Module and a trainable lightweight Residual Aligner. Using fewer than 1/8 of the parameters, RAM achieves alignment performance comparable to or exceeding full-parameter SFT/DPO, while also resolving the first-token latency problem.

## Background & Motivation

LLM alignment is a critical step in ensuring that model outputs conform to domain requirements and human values. Conventional approaches (SFT, RLHF, DPO) require fine-tuning the entire large model, which introduces the following challenges:

**Resource intensity**: Training 8B+ models demands substantial GPU resources and incurs high costs.

**Deployment fragmentation**: Different domains require separate model deployments, preventing traffic sharing.

**Insufficient flexibility**: Rapid adaptation to diverse alignment requirements is difficult.

Existing modular approaches (e.g., Aligner) decouple alignment by training adapters to learn "correction residuals," but suffer from:
- **First-token latency**: The upstream model must generate a complete response before correction can be applied.
- **OOD risk**: The Aligner is conditioned on a reference response $\mathbf{y}'$; at inference time, this reference comes from the Proposal rather than the true distribution, introducing out-of-distribution problems.

## Method

### Overall Architecture

RAM decomposes the target aligned distribution into a linear combination of two modules:

$$P_{\text{Aligned}}(\mathbf{y}|\mathbf{x}) \propto P_{\text{ProposalModule}}(\mathbf{y}|\mathbf{x}) \cdot P_{\text{ResidualAligner}}(\mathbf{y}|\mathbf{x})$$

The **Proposal Module** (large model, frozen) provides the base distribution, while the **Residual Aligner** (small model, trainable) serves as an estimator of importance weights to compensate for the alignment gap.

### Importance Sampling Decomposition

Assuming the pretrained model $P_M(\mathbf{y}|\mathbf{x})$ estimates a general distribution $P_\mathcal{D}$, the alignment objective is to approximate a biased subset distribution $P_\mathcal{S}$. Leveraging importance sampling:

$$P_\mathcal{S}(\mathbf{y}|\mathbf{x}) = P_M(\mathbf{y}|\mathbf{x}) \cdot \frac{P_\mathcal{S}(\mathbf{y}|\mathbf{x})}{P_M(\mathbf{y}|\mathbf{x})}$$

An autoregressive model $Q_\theta$ is introduced to estimate the importance weights, yielding the normalized RAM formulation:

$$P_\theta(\mathbf{y}|\mathbf{x}) = \frac{P_M(\mathbf{y}|\mathbf{x}) \cdot Q_\theta(\mathbf{y}|\mathbf{x})}{Z_\theta(\mathbf{x})}$$

where $Z_\theta(\mathbf{x}) = \sum_{\mathbf{y}} P_M(\mathbf{y}|\mathbf{x}) Q_\theta(\mathbf{y}|\mathbf{x})$.

### Sentence-Level Training Strategy

Starting from the SFT objective, the final loss function is derived via Jensen's inequality and the method of Lagrange multipliers:

$$\mathcal{L}_{\text{SFT}}(P_\theta) = -\mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \mathcal{S}}[\log Q_\theta(\mathbf{y}|\mathbf{x})] + \alpha \mathbb{E}_{\mathbf{x} \sim \mathcal{S}, \mathbf{y} \sim P_M}[\log Q_\theta(\mathbf{y}|\mathbf{x})]$$

- The first term maximizes the likelihood of the Residual Aligner on the target data.
- The second term controls the influence of the Proposal Module distribution ($\alpha \in [0,1]$).

A key advantage is that $P_M$ remains frozen throughout training, requiring only a one-time generation of sampled data.

### Token-Level Decoding: Proposing-Aligning-Reducing (PAR) Sampling

The central innovation lies in transforming sequence-level importance sampling into a token-by-token autoregressive process:

$$P_\theta(y_l | y_{<l}, \mathbf{x}) = \frac{P_M(y_l | y_{<l}, \mathbf{x}) \cdot Q_\theta(y_l | y_{<l}, \mathbf{x})}{Z_\theta(y_{<l}, \mathbf{x})}$$

The PAR three-step pipeline:
1. **Propose**: Sample $n$ candidate tokens from the Proposal Module via nucleus sampling.
2. **Align**: Compute importance weights $w(y_l^i) = \frac{Q_\theta(y_l^i | y_{<l}, \mathbf{x})}{Z_\theta(y_{<l}, \mathbf{x})}$ for each candidate using the Residual Aligner.
3. **Reduce**: Normalize the weights and perform categorical sampling to select the final token.

In practice, sparse Softmax is employed: logits for tokens not sampled by the Proposal are set to $-\infty$, making the operation equivalent to standard Softmax followed by sampling.

### Variance Control

- **Training**: $Q_\theta$ learns to compensate for the discrepancy between $P_M$ and $P_\mathcal{S}$, smoothing extreme weights.
- **Inference**: Top-P region sampling combined with self-normalized importance sampling.

### KL Divergence Safeguard

When $D_{KL}(P_M \| Q_\theta) > 0.1$, decoding falls back directly to $P_M$ to prevent degeneration of the Residual Aligner.

## Key Experimental Results

### Main Results: Instruction Following and Domain Adaptation (SFT)

| Strategy | UltraChat LC% | TL;DR LC% |
|------|--------------|-----------|
| **Llama3.1-8B Family** | | |
| W.Up 8B (baseline) | 5.06 | 60.71 |
| SFT 1B (small model standalone) | 1.77 | 37.18 |
| W.Up 8B + Aligner 1B | 2.34 | 53.85 |
| **W.Up 8B + R.A. 1B** | **6.46** | **65.11** |
| SFT 8B (full-parameter fine-tuning) | 6.81 | 64.12 |
| **SFT 8B + R.A. 1B** | **7.32** | **66.11** |
| **Qwen2.5-14B Family** | | |
| W.Up 14B | 10.42 | 53.11 |
| W.Up 14B + Aligner 3B | 8.08 | 53.85 |
| **W.Up 14B + R.A. 3B** | **12.32** | **57.76** |
| SFT 14B | 12.87 | 58.64 |
| **SFT 14B + R.A. 3B** | **12.88** | **64.91** |

A 1B Residual Aligner paired with an 8B Proposal can match or surpass 8B full-parameter SFT.

### Preference Optimization Experiments (Anthropic-HH)

| Strategy | Helpfulness GPT4-LC% | Harmlessness GPT4-LC% |
|------|---------------------|----------------------|
| SFT 8B | 58.59 | 65.31 |
| DPO 8B | 68.03 | 73.06 |
| DPO 8B + Aligner 1B | 55.31 | 70.12 |
| **DPO 8B + R.A. 1B** | **72.22** | **79.89** |
| DPO 14B | 74.53 | 71.41 |
| **DPO 14B + R.A. 3B** | **75.39** | **74.76** |

Even when the DPO model already achieves a win rate exceeding 70%, the Residual Aligner yields an additional 5–9% improvement. The Aligner approach, by contrast, substantially underperforms the baseline due to OOD issues.

### Ablation Study

**Effect of Residual Aligner size**:
- Performance increases with model size from 0.5B to 8B, but the margin is modest (average 2.4% for Llama3, 2.1% for Qwen2.5).
- Small models capture most of the benefit, offering an excellent cost-performance ratio.

**Effect of hyperparameter $\alpha$**:
- Performance varies minimally across the range 1e-5 to 0.1 (CV of only 1.67%–2.17%).
- No fine-grained hyperparameter tuning is required.

**Training efficiency**:
- SFT setting: **4×** improvement in efficiency over full-parameter fine-tuning.
- DPO setting: **13.33×** improvement in efficiency.

### Key Findings

1. The primary advantage of RAM lies not in large absolute performance gains, but in approaching full-parameter performance with as few as 1/8 of the parameters.
2. The Aligner severely underperforms RAM on preference optimization tasks due to OOD issues.
3. Small Residual Aligners (1B–3B) offer the best cost-effectiveness.
4. Multiple Residual Aligners can share a single Proposal Module, enabling cross-domain traffic sharing.

## Highlights & Insights

1. **Elegant theoretical framework**: Formalizing alignment as importance sampling leads to a natural modular decomposition, achieving strong unity between theory and practice.
2. **Complete elimination of first-token latency**: PAR sampling converts sequence-level resampling into a token-by-token operation, yielding latency comparable to standard autoregressive decoding.
3. **Full decoupling at training time**: The Proposal Module requires only a one-time data synthesis step and is entirely excluded from subsequent training.
4. **Strong engineering value**: The deployment paradigm of one large model paired with multiple small Aligners substantially reduces the cost of multi-domain alignment.
5. **Generality**: The same framework applies to SFT, DPO, and domain adaptation settings.

## Limitations & Future Work

1. **Same-family requirement**: The Proposal Module and Residual Aligner must share a vocabulary (i.e., belong to the same model family), limiting compositional flexibility.
2. **KL divergence threshold**: The hard threshold of 0.1 lacks theoretical justification and may be overly conservative or aggressive in certain scenarios.
3. **Evaluation scope**: The evaluation relies primarily on the AlpacaEval 2 framework, lacking more diverse assessment dimensions.
4. **Diminishing returns for larger Residual Aligners**: The marginal gain from increasing Aligner parameter count is small, suggesting an information bottleneck inherent to the framework.
5. **RLHF not explored**: Validation is limited to SFT and DPO; online RLHF settings are not addressed.

## Related Work & Insights

- **Relationship with Residual EBM (Deng et al. 2020)**: RAM shares the same formulation as the Residual EBM, but achieves token-level decoding through autoregressive factorization.
- **Distinction from Controlled Decoding**: Controlled Decoding learns a prefix scorer, whereas RAM learns a full importance weight estimator.
- **Fundamental distinction from Aligner**: RAM directly models $P(\mathbf{y}|\mathbf{x})$ rather than $P(\mathbf{y}|\mathbf{y}', \mathbf{x})$, avoiding OOD issues.
- **Implications for MoE alignment**: The multi-Aligner shared-Proposal paradigm of RAM can be viewed as a lightweight Mixture-of-Experts alignment scheme.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The importance sampling framework is theoretically elegant, and the PAR decoding strategy is innovative.
- **Practicality**: ⭐⭐⭐⭐⭐ — Training efficiency gains are substantial, and the deployment paradigm offers significant engineering value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers two model families × three tasks × multiple baselines, though evaluation dimensions are somewhat narrow.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, though notation is dense.
- **Recommended Reading**: ⭐⭐⭐⭐ — An important contribution to LLM alignment; essential reading for researchers working on modular alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](ricl_temporal_credit.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)

</div>

<!-- RELATED:END -->
