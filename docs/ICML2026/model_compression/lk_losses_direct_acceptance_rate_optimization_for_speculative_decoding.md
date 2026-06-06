---
title: >-
  [Paper Note] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding
description: >-
  [ICML 2026][Model Compression][Speculative Decoding] This paper argues that using KL divergence as a proxy for acceptance rate in speculative decoding training is suboptimal. For small-capacity draft models…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Speculative Decoding"
  - "Acceptance Rate Optimization"
  - "KL vs TV Divergence"
  - "Draft Model Training"
  - "EAGLE/MEDUSA"
date: 2026-05-08
content_hash: bd48c31deca13cc2
---

# LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding

**Conference**: ICML 2026  
**arXiv**: [2602.23881](https://arxiv.org/abs/2602.23881)  
**Code**: https://huggingface.co/nebius/lk-speculators  
**Area**: Model Compression / Speculative Decoding / LLM Inference Acceleration  
**Keywords**: Speculative Decoding, Acceptance Rate Optimization, KL vs TV Divergence, Draft Model Training, EAGLE/MEDUSA

## TL;DR
This paper argues that using KL divergence as a proxy for acceptance rate in speculative decoding training is suboptimal. For small-capacity draft models, KL minimization does not imply acceptance rate maximization. The authors propose LK losses (directly maximizing negative log acceptance rate combined with a KL trust-region hybrid) as a plug-in replacement, achieving a consistent 8-10% improvement in average acceptance length across 4 draft architectures and 6 target models (8B-685B).

## Background & Motivation

**Background**: LLM inference is constrained by memory bandwidth, resulting in low utilization during autoregressive single-token decoding. Speculative decoding utilizes a small draft model $q$ to propose $K$ tokens, which are verified in parallel by a target model $p$. The acceptance probability per token is $\beta = \min(1, p/q)$, and the sequence is truncated at the first rejection. Existing architectures include MEDUSA (parallel heads), EAGLE/EAGLE-3 (autoregressive heads with feature fusion), and MTP (native draft modules in DeepSeek-V3).

**Limitations of Prior Work**: Current draft models are trained using KL divergence (or equivalent Cross-Entropy) as the objective, treating KL as a proxy for acceptance rate. While $q = p$ achieves both KL = 0 and an acceptance rate of 1 at the global optimum, draft models typically possess only 1-5% of the target's capacity. At suboptimal points, KL minimization provides no guarantee for maximizing the acceptance rate.

**Key Challenge**: The actual objective is the acceptance rate (mathematically equivalent to $1 - \text{TV}(p, q)$, where TV is Total Variation distance). However, training utilizes KL, and the behaviors of these two metrics differ significantly at suboptimal points: forward KL is mode-covering (spreading mass, leading to suboptimal acceptance), while reverse KL is mode-seeking (collapsing to dominant modes). Neither perfectly maximizes distributional overlap.

**Goal**: Replace KL with a loss function targeting the acceptance rate directly. Requirements include: (1) applicability to native draft modules trained from scratch, (2) zero additional computational overhead, and (3) universality across architectures and scales.

**Key Insight**: Although DistillSpec identified TV distance as the precise mathematical counterpart to acceptance rate, it performed poorly on pre-trained LMs due to weak TV gradients. This work finds that this limitation primarily applies to pre-trained scenarios; for **randomly initialized native draft modules**, direct TV-style acceptance rate losses are highly effective.

**Core Idea**: LK losses consist of (a) "LK-direct," which maximizes the negative log acceptance rate (similar to maximum likelihood), and (b) "LK-hybrid," which utilizes a gradual switch from KL to LK (acting as a trust-region method to maintain stability early on).

## Method

### Overall Architecture

The acceptance rate is defined as $\alpha = \sum_x \min(q(x), p(x)) = 1 - \text{TV}(p, q)$.

Two variants of LK losses are introduced:
- **LK-direct**: $\mathcal{L}_{\text{LK}} = -\log \alpha = -\log \sum_x \min(q(x), p(x))$
- **LK-hybrid**: $\mathcal{L}_{\text{hybrid}}(t) = (1 - w(t)) \cdot \text{KL}(p \| q) + w(t) \cdot \mathcal{L}_{\text{LK}}$, where $w(t)$ increases from 0 to 1.

These function as drop-in replacements for the loss function without requiring architectural changes.

### Key Designs

1. **LK-direct: Direct Negative Log Acceptance Rate Optimization**:
    - **Function**: Aligns the training objective with the inference-time acceptance rate (one-to-one mapping with TV).
    - **Mechanism**: The gradient of $\mathcal{L}_{\text{LK}} = -\log \alpha$ with respect to draft logits $z_q$ differs from KL. While the KL gradient $\nabla_{z_q} \text{KL}(p \| q) = q - p$ applies pressure across all tokens, the LK gradient concentrates on regions of distributional overlap. Similar to TV's mass-focused nature, it encourages the draft model to concentrate its capacity on high-probability tokens of the target model.
    - **Design Motivation**: In capacity-limited scenarios, the choice between "covering the entire support" (KL requirement) and "aligning with high-probability regions" (TV/LK preference) is critical. In speculative decoding, the latter directly improves the acceptance rate.

2. **LK-hybrid: KL → LK Gradual Switch (Trust-Region Approach)**:
    - **Function**: Uses KL to provide robust global gradient signals during early training, then switches to LK for fine-tuning the acceptance rate.
    - **Mechanism**: $\mathcal{L}_{\text{hybrid}}(t) = (1-w(t)) \text{KL}(p\|q) + w(t) \mathcal{L}_{\text{LK}}$, where $w(t)$ follows a schedule (e.g., linear or sigmoid) from 0 to 1. This balances a stable surrogate with the true objective.
    - **Design Motivation**: At random initialization, the KL gradient $\|q - p\| \sim \mathcal{O}(1/\sqrt{k})$ provides a strong signal. Conversely, the LK gradient can be near zero if distributions are entirely disjoint ($\min(q, p) \to 0$). The hybrid approach uses KL to push the distributions into overlap before LK refines the output.

3. **Gradient Structure Analysis**:
    - **Function**: Mathematically explains why LK is superior at suboptimal points.
    - **Mechanism**: The forward KL gradient $\nabla_{z_q} \text{KL} = q - p$ penalizes the draft model across all tokens (mass-covering). The LK gradient (derived in the appendix) only contributes for tokens where $q < p$. It specifically updates tokens that are "underestimated by the draft but high-probability for the target."
    - **Design Motivation**: This selective updating allows limited capacity to be "invested" in high-impact tokens, which is precisely what the acceptance rate requires.

## Key Experimental Results

### Main Results: 4 Draft Architectures × 6 Target Models

| Draft Architecture | Target Model | KL Acceptance Length $\tau$ | **LK Acceptance Length $\tau$** | Gain |
|--------|------|------|------|------|
| MEDUSA-3 head | Llama-3-8B | 2.31 | 2.48 | +7.4% |
| EAGLE-3 | Llama-3-70B | 3.12 | 3.45 | +10.6% |
| EAGLE-3 | Qwen3-235B-A22B | 2.85 | 3.16 | +10.9% |
| EAGLE-3 | DeepSeek-V3 (685B) | 2.94 | 3.21 | +9.2% |
| MTP module | DeepSeek-V3 | 2.78 | 3.02 | +8.6% |
| Standalone Qwen-1.5B → Llama-3-70B | – | 2.46 | 2.68 | +8.9% |

An 8-10% improvement in average acceptance length is consistent across all architectures and scales. The advantage becomes more pronounced as the draft length $K$ increases.

### Results by Domain

| Domain | KL $\tau$ | LK $\tau$ | Gain |
|------|------|------|------|
| General (MT-Bench) | 2.85 | 3.10 | +8.8% |
| Code (HumanEval) | 3.12 | 3.43 | +9.9% |
| Math (GSM8K) | 2.78 | 3.05 | +9.7% |

The gains from LK are larger in Code and Math domains, where token distributions are more skewed and the benefits of TV-style focus are amplified.

### LK-direct vs LK-hybrid Comparison

| Training Phase | KL | LK-direct | LK-hybrid |
|--------|-----|---------|----------|
| Early (random init) | Stable convergence ($\tau=2.10$) | Slow start ($\tau=1.85$) | Stable ($\tau=2.12$) |
| Late (well-trained) | $\tau=2.85$ | $\tau=3.10$ | $\tau=3.13$ |

LK-direct converges slowly in the early stages due to weak gradient signals. LK-hybrid achieves the best results by combining early stability with late-stage refinement.

### Key Findings
- **KL is indeed suboptimal for small draft models**: The 8-10% acceptance rate improvement is stable across architectures, scales, and domains.
- **Longer $K$ benefits more**: As seq length increases, small improvements in per-step acceptance rate compound, leading to larger gaps between LK and KL.
- **Trust-region hybrid is the robust engineering choice**: LK-hybrid solves the "cold start" problem of direct TV-style optimization.
- **Plug-and-play practical utility**: Implementing the loss function requires minimal code changes and introduces no overhead.

## Highlights & Insights
- **Identifying Objective-Proxy Mismatch**: The observation that KL is a poor proxy for acceptance rate in capacity-constrained scenarios highlights a general issue in knowledge distillation (KD) that could apply beyond speculative decoding.
- **Trust-Region Philosophy**: The LK-hybrid design acts as a general curriculum template for scenarios where the true objective has sparse gradients but a stable surrogate exists.
- **Theoretical Foundations**: The work provides more than empirical results; the analysis of the gradient structure explains "why" selective investment in tokens is superior to uniform pressure.
- **Validation at 685B Scale**: Significant improvements on DeepSeek-V3 demonstrate that the method scales to frontier-class models.

## Limitations & Future Work
- The $w(t)$ switching schedule is currently handcrafted; an adaptive mechanism based on current KL/LK ratios might be more robust.
- The effectiveness of LK on established pre-trained standalone draft models remains to be fully verified, given prior reports of TV performing poorly in such settings.
- While acceptance rate improves, the precise end-to-end wall-clock speedup depends on the specific verifier overhead.
- Future work could explore more aggressive objectives, such as directly maximizing end-to-end throughput.

## Related Work & Insights
- **vs DistillSpec**: While DistillSpec explored TV on pre-trained external drafts with mixed results, this work demonstrates stability and success on native, randomly initialized draft modules.
- **vs MEDUSA / EAGLE / MTP**: These existing frameworks use KL. Switching to LK provides an immediate, architecture-agnostic performance boost.
- **Insight**: In any KD scenario, KL should not be the default assumption; the choice of loss should align with the mathematical metric used for downstream evaluation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While the concept of TV optimization exists (e.g., DistillSpec), the clarification of the "native vs. pre-trained" setting and the introduction of LK-hybrid are significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 4 architectures, 6 models, and 3 domains, supported by theoretical gradient analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear pedagogical examples (e.g., Gaussian toy models) and precise mathematical analysis.
- **Value**: ⭐⭐⭐⭐⭐ Speculative decoding is a primary LLM acceleration method; an 8-10% acceptance rate boost with zero overhead is a high-value contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)
- [\[ACL 2026\] SSSD: Simply-Scalable Speculative Decoding](../../ACL2026/model_compression/sssd_simply-scalable_speculative_decoding.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)

</div>

<!-- RELATED:END -->
