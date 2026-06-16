---
title: >-
  [Paper Note] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding
description: >-
  [ICML 2026][Model Compression][EAGLE/MEDUSA] This paper argues that using KL divergence as a proxy for acceptance rate in speculative decoding training is sub-optimal—minimizing KL for small-capacity draft models does not imply maximizing the acceptance rate. The authors propose LK losses (direct maximization of the negative log acceptance rate + a trust-region h
tags:
  - ICML 2026
  - Model Compression
  - EAGLE/MEDUSA
date: 2026-05-08
content_hash: 85fe1302e7a1da49
---
# LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding

**Conference**: ICML 2026  
**arXiv**: [2602.23881](https://arxiv.org/abs/2602.23881)  
**Code**: https://huggingface.co/nebius/lk-speculators  
**Area**: Model Compression / Speculative Decoding / LLM Inference Acceleration  
**Keywords**: Speculative Decoding, Acceptance Rate Optimization, KL vs TV Divergence, Draft Model Training, EAGLE/MEDUSA

## TL;DR
This paper argues that using KL divergence as a proxy for acceptance rate in speculative decoding training is sub-optimal—minimizing KL for small-capacity draft models does not imply maximizing the acceptance rate. The authors propose LK losses (direct maximization of the negative log acceptance rate + a trust-region hybrid with KL) as a plug-in replacement. Across 4 draft architectures and 6 target models (8B-685B), it consistently improves the average acceptance length by 8-10%.

## Background & Motivation

**Background**: LLM inference is limited by memory bandwidth, and autoregressive single-token decoding suffers from low utilization. Speculative decoding allows a small draft model $q$ to propose $K$ tokens, which the target model $p$ validates in parallel. The acceptance probability for each token is $\beta = \min(1, p/q)$, and the first rejected token discards all subsequent ones. Existing architectures include MEDUSA (parallel heads), EAGLE/EAGLE-3 (autoregressive heads + feature fusion), and MTP (DeepSeek-V3 native draft modules).

**Limitations of Prior Work**: Training for all these draft models uses KL divergence (or equivalent Cross-Entropy) as the objective, treating KL as a proxy for the acceptance rate. Theoretically, when $q = p$, KL = 0 and the acceptance rate = 1, making the global optimum consistent. However, draft model capacity is typically only 1-5% of the target model, meaning the global optimum is never reached. At sub-optimal points, minimizing KL **provides no guarantee** for maximizing the acceptance rate.

**Key Challenge**: The direct optimization goal is the acceptance rate (mathematically equivalent to $1 - \text{TV}(p, q)$, where TV is Total Variation distance). However, training uses KL. The behaviors of these two diverge significantly at sub-optimal points—forward KL is mode-covering (spreading mass over the support, leading to sub-optimal acceptance), while reverse KL is mode-seeking (collapsing to the dominant mode). Neither maximizes distributional overlap.

**Goal**: Replace KL with a loss targeting the acceptance rate directly. Requirements: (1) Applicable to native training of draft modules from scratch rather than just external pre-trained speculators; (2) Simple implementation with zero computational overhead; (3) Generalizability across architectures and scales.

**Key Insight**: DistillSpec previously identified TV distance as the exact mathematical counterpart to the acceptance rate but found it performed poorly on pre-trained LMs due to weak TV gradients (mass concentrated tokens). This paper discovers that this limitation applies primarily to pre-trained scenarios; for **randomly initialized native draft modules**, direct TV-style acceptance rate losses are effective.

**Core Idea**: LK losses — (a) Directly maximize the negative log acceptance rate ("LK-direct", similar to maximum likelihood); (b) A progressive transition from KL to LK ("LK-hybrid", a trust-region method that uses KL for stability initially and switches to LK later).

## Method

### Overall Architecture

In speculative decoding, the acceptance probability for each token is $\beta = \min(1, p/q)$. The expected acceptance rate for the entire proposal is exactly $\alpha = \sum_x \min(q(x), p(x)) = 1 - \text{TV}(p, q)$. This implies that the quantity to be maximized is the total overlap between the draft distribution $q$ and target distribution $p$ (equivalent to minimizing TV distance), rather than KL divergence. Based on this, LK losses replace the training objective with the acceptance rate itself: LK-direct minimizes the negative log acceptance rate $-\log\alpha$ (its gradient is equivalent to TV optimization with $1/\alpha$ adaptive scaling), while LK-hybrid adaptively mixes KL and TV terms based on the current acceptance rate $\alpha$, transitioning smoothly from KL-dominance in early training to TV-dominance later. Neither variant requires architectural changes or extra computation; they only replace a single line in the loss function and are naturally compatible with pipelines using vocabulary truncation (like EAGLE-3). Consequently, they can be plugged into MEDUSA, EAGLE-3, MTP, or any other existing training framework.

### Key Designs

**1. LK-direct: Aligning Loss Directly with Acceptance Rate instead of a Proxy**

The pain point is capacity: draft models have only 1-5% of the target's capacity and never reach $q=p$. KL minimization only equals acceptance rate maximization at the global optimum. LK-direct defines the loss as the negative logarithm of the acceptance rate, treating $\alpha$ as the marginal probability of "proposing a token and it being accepted":

$$\mathcal{L}_{\text{LK}}^{\alpha} = -\log \alpha = -\log \sum_x \min(p(x), q(x))$$

The relationship with TV is clean: it can be proven that $\nabla_{z_q}\mathcal{L}_{\text{LK}}^{\alpha} = \tfrac{1}{\alpha}\,\nabla_{z_q}\text{TV}(p, q)$. This acts as TV optimization with adaptive scaling. The $1/\alpha$ factor automatically amplifies gradients when the acceptance rate is low ($\alpha \to 0$), compensating for the near-zero gradients of pure TV at initialization. Intuitively, while KL forward is mode-covering, TV/LK maximizes the overlap of two distributions, concentrating finite capacity on target high-probability areas.

**2. LK-hybrid: Adaptive KL and TV Mixing to Solve Random Initialization Cold Starts**

Pure TV or LK-direct has a weakness at initialization: when $q$ is spread across a large vocabulary ($V > 10^5$), overlap with $p$ is minimal, and TV gradients are near zero. LK-hybrid addresses this by weighting the two terms:

$$\mathcal{L}_{\text{LK}}^{\lambda}(p, q) = \lambda \cdot \text{KL}(p\|q) + (1-\lambda)\cdot \text{TV}(p, q)$$

Instead of a fixed weight, the paper uses an adaptive schedule based on the **current acceptance rate $\alpha$**: $\lambda = \exp(-\eta \cdot \text{sg}[\alpha])$. When the acceptance rate is low (early training), $\lambda \to 1$, allowing KL to lead the distributions toward overlap with smooth gradients. As acceptance increases, $\lambda$ decays, switching to TV-dominance for direct refinement. This is interpreted as a trust-region approach: $\min_q \text{TV}(p,q)\ \text{s.t.}\ \text{KL}(p\|q) \le \delta$, where the adaptive schedule implicitly controls the threshold $\delta$.

**3. Gradient Structure Analysis: Why KL is a Proxy and Pure TV Fails at Initialization**

The design motivation stems from gradients. The three losses have distinct gradient structures regarding draft logits $z_q$. The KL forward gradient $\nabla_{z_q}\text{KL}(p\|q) = q - p$ applies pressure based on the difference between prediction and target with a magnitude of $\mathcal{O}(1/\sqrt{k})$, which is strong and smooth regardless of alignment. TV gradients $\nabla_{z_q}\text{TV} = \tfrac{1}{2}q\odot(s - \mathbb{E}_q[s])$ (where $s_i = \text{sign}(q_i - p_i)$) carry only sign information and are near zero during random initialization. This explains why LK-direct (which uses $1/\alpha$ to amplify TV gradients) and LK-hybrid (which uses KL to bridge the cold start) are necessary for native draft modules trained from scratch.

**4. Compatibility with Vocabulary Truncation: No Need for "Proxies for Proxies"**

EAGLE-3 truncates the draft LM head vocabulary to a high-frequency subset to reduce latency. This is problematic for KL because $q_i = 0$ for truncated tokens while $p_i > 0$, causing KL to diverge. Standard practice involves re-normalizing the target distribution $p$, creating a "proxy for a proxy." LK losses handle this naturally: since $\alpha = \sum_x \min(p, q)$, truncated tokens contribute $\min(p_i, 0) = 0$ and do not affect the acceptance rate. Thus, LK optimizes the original target distribution without secondary approximations.

## Key Experimental Results

### Main Results: 4 Draft Architectures × 6 Target Models

| Draft Architecture | Target Model | KL Training $\tau$ | **LK Training $\tau$** | Gain |
|:---|:---|:---|:---|:---|
| MEDUSA-3 head | Llama-3-8B | 2.31 | 2.48 | +7.4% |
| EAGLE-3 | Llama-3-70B | 3.12 | 3.45 | +10.6% |
| EAGLE-3 | Qwen3-235B-A22B | 2.85 | 3.16 | +10.9% |
| EAGLE-3 | DeepSeek-V3 (685B) | 2.94 | 3.21 | +9.2% |
| MTP module | DeepSeek-V3 | 2.78 | 3.02 | +8.6% |
| Standalone Qwen-1.5B → Llama-3-70B | – | 2.46 | 2.68 | +8.9% |

The 8-10% improvement in average acceptance length is consistent across all architectures and scales. The advantage is more pronounced at larger values of $K$.

### Domain Distribution

| Domain | KL $\tau$ | LK $\tau$ | Gain |
|:---|:---|:---|:---|
| General (MT-Bench) | 2.85 | 3.10 | +8.8% |
| Code (HumanEval) | 3.12 | 3.43 | +9.9% |
| Math (GSM8K) | 2.78 | 3.05 | +9.7% |

The gains are larger in Code/Math domains where token distributions are more skewed.

### Ablation Study: LK-direct vs LK-hybrid

| Training Phase | KL | LK-direct | LK-hybrid |
|:---|:---|:---|:---|
| Early (random init) | $\tau=2.10$ Stable | $\tau=1.85$ Slow | $\tau=2.12$ Stable |
| Late (well-trained) | $\tau=2.85$ | $\tau=3.10$ | $\tau=3.13$ |

LK-direct converges slowly early on; LK-hybrid achieves both early stability and late refinement.

### Key Findings
- **KL is indeed sub-optimal for small-capacity drafts**: The 8-10% gain is stable across scales and domains.
- **Larger $K$ benefits more**: Acceptance length $\tau$ widens the gap for $K \geq 4$ as sequential acceptance probabilities compound.
- **Trust-region hybrid is the robust choice**: It solves the cold-start problem of direct optimization.
- **Plug-and-play**: Requires changing only one line of code with no extra overhead.

## Highlights & Insights
- **Identified target-proxy misalignment**: KL's mass-covering property wastes limited capacity in small models. This perspective is applicable to any Knowledge Distillation (KD) task.
- **Trust-region design philosophy**: Using a stable surrogate before switching to the true objective is a general template for scenarios with sparse target gradients.
- **Theoretical support via gradient analysis**: The paper quantifies why KL acts as a smooth proxy and why TV requires specialized handling.
- **Verification at 685B scale**: Improvements on frontier models like DeepSeek-V3 prove the method's scalability.

## Limitations & Future Work
- The $w(t)$ switch schedule is manual; adaptive versions based on relative losses might be more robust.
- Effectiveness on pre-trained standalone drafts (where TV was previously reported as weak) requires further validation.
- End-to-end inference speedup was not reported in detail; this depends on the actual cost of verifier steps.
- Future work could explore maximizing end-to-end throughput directly.

## Related Work & Insights
- **vs DistillSpec**: DistillSpec explored TV in pre-trained external drafts with mixed results; this paper demonstrates stability in native draft modules trained from scratch.
- **vs MEDUSA/EAGLE/MTP**: These use KL; this paper provides a simple upgrade path for all existing speculative decoding systems.
- **Insight**: In KD, KL is not always optimal. One must consider which mathematical quantity corresponds to the evaluation metric.

## Rating
- Novelty: ⭐⭐⭐⭐ LK-direct exists conceptually, but the native vs. pre-trained clarification and LK-hybrid are new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive results across 4 architectures, 6 models, and 3 domains.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations and intuitive toy examples.
- Value: ⭐⭐⭐⭐⭐ Speculative decoding is critical for LLM acceleration; this is a plug-and-play gain.

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
