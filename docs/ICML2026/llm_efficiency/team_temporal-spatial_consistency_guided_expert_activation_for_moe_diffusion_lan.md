---
title: >-
  [Paper Note] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration
description: >-
  [ICML2026][LLM Efficiency][Diffusion Language Models] TEAM addresses the inherent mismatch in MoE Diffusion Language Models (dLLM) where "a large number of experts are activated while only few tokens are accepted." By ut…
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "Diffusion Language Models"
  - "MoE"
  - "Expert Activation"
  - "Temporal-Spatial Consistency"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 113461337abfa517
---

# TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration

**Conference**: ICML2026  
**arXiv**: [2602.08404](https://arxiv.org/abs/2602.08404)  
**Code**: https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM  
**Area**: LLM Efficiency / Diffusion Language Models / MoE  
**Keywords**: Diffusion Language Models, MoE, Expert Activation, Temporal-Spatial Consistency, Inference Acceleration

## TL;DR
TEAM addresses the inherent mismatch in MoE Diffusion Language Models (dLLM) where "a large number of experts are activated while only few tokens are accepted." By utilizing temporal and spatial consistency during in-block decoding, TEAM designs differentiated expert activation and decoding strategies for decoded, hot, and cold tokens. It achieves up to 2.2× speedup on SDAR 30B-A3B with near-zero precision loss.

## Background & Motivation

**Background**: Diffusion Language Models (dLLM) utilize bidirectional attention to "denoise" an entire response simultaneously, naturally supporting parallel decoding. Recent works like SDAR and LLaDA 2.0 utilize autoregressive (AR) initialization and replace FFNs with a Mixture-of-Experts (MoE) to bring dLLM efficiency and accuracy closer to mainstream AR LLMs. These are regarded as a next-generation paradigm that inherits AR priors while breaking the serial bottleneck of AR.

**Limitations of Prior Work**: Directly grafting MoE onto dLLM results in "reversed degradation" of inference efficiency. In every denoising iteration of a block, all tokens (including those already accepted) undergo a parallel forward pass under bidirectional attention, where each token independently routes its top-$k$ experts. However, only a few tokens with confidence exceeding a threshold $\tau$ are truly unmasked. Consequently, the "number of activated experts is far greater than the number of tokens accepted per step." In SDAR, each token routes to 8 experts, but the ratio of activated experts to accepted tokens is significantly higher than 8, negating the sparse activation advantage of MoE and causing it to degrade toward a dense model.

**Key Challenge**: There is a structural mismatch between the block-parallelism of dLLM and the sparse activation of MoE. Block-wise parallelism implies that the compute load often fails to reach the GPU's compute–bandwidth equilibrium, making the decoding memory-bound. In this state, every additional expert activation requires extra parameter movement, slowing down the process. Therefore, the true direction for optimization is "activating fewer experts + accepting more tokens," though vanilla routing fails to utilize the internal structure of dLLM blocks.

**Goal**: To reduce inference latency without retraining the MoE dLLM by **activating fewer experts** while **accepting more tokens** per forward pass. The problem is decomposed into three token categories: (a) decoded tokens that still participate in the forward pass; (b) hot tokens that are about to be accepted; (c) cold tokens that will not be accepted in the short term.

**Key Insight**: Measurements on SDAR 30B-A3B reveal two layers of consistency in block-wise decoding. **Temporal Consistency**: The hidden states of accepted tokens tend to stabilize after the subsequent forward pass (consistent with dKV-Cache findings), yet vanilla routing still triggers expert activation for them at every step. **Spatial Consistency**: Routing for mask tokens is highly concentrated—a few experts handle the decoding of almost all mask tokens; furthermore, the acceptance sequence is approximately autoregressive, with newly accepted tokens spatially clustered near already decoded tokens. Together, these imply that tokens at different positions can be handled with differentiated strategies.

**Core Idea**: Differentiate the management of "decoded, hot, and cold" tokens—decoded tokens use KV cache and no longer activate experts; hot tokens undergo proactive speculative exploration to accept multiple tokens at once; cold tokens are forced to reuse the set of experts already activated by hot/decoded tokens. This eliminates redundant activations and reinvests the "saved compute" into positions more likely to succeed.

## Method

### Overall Architecture
The input is a masked sequence divided into $B$ blocks, each containing $L$ tokens. Inter-block causal attention with KV cache is used alongside intra-block bidirectional attention for multi-step denoising, where tokens with confidence $c_k > \tau$ are unmasked at each step. TEAM categorizes intra-block tokens into three types before each forward pass: (i) **decoded** (unmasked and processed by at least one subsequent forward pass); (ii) **hot mask**—tokens satisfying "current confidence $c_k > \tau_h$" or "distance to any decoded position $< L_h$," defined as $y_i^{k\text{-}hot}=\{y_i^k \mid (c_k > \tau_h)\ \text{or}\ (\forall j,\ |k-j|<L_h)\}$; (iii) **cold mask**—the remaining mask tokens. These three types follow DCD, SEH, and LAC paths respectively, which are merged into a single forward pass. The framework is plug-and-play, requiring no weight changes or retraining.

### Key Designs

1.  **DCD — Delayed Caching for Decoded Tokens**:
    *   **Function**: Tokens that have been accepted and passed through one additional forward pass no longer participate in subsequent forward passes; their KV pairs and corresponding expert activations are retrieved from the cache.
    *   **Mechanism**: Once a dLLM enters the block-diffusion paradigm, intra-block parallel scale is relatively small and far from compute-bound, so vanilla models do not require fine-grained KV cache. However, under MoE, "recomputing decoded tokens" triggers many expert activations that rarely overlap with mask tokens, causing activation explosion. Delaying caching by one step (caching after the forward pass following acceptance) matches the observed stability of hidden states.
    *   **Design Motivation**: Unlike dKV-Cache which requires periodic global refreshes, this design leverages AR initialization and near-autoregressive acceptance to remain **refresh-free** without precision loss (Table 3). This fundamentally cuts the expert activation overhead for decoded tokens.

2.  **SEH — Speculative Exploration for Hot Tokens**:
    *   **Function**: Proactively performs top-$k$ multi-branch speculation for "hot tokens" likely to be accepted next, verifying multiple decoding candidates in a single forward pass to increase the number of accepted tokens per step.
    *   **Mechanism**: Hot tokens are identified by $c_k > \tau_h$ or proximity to decoded tokens ($< L_h$). Top-$k$ confidence candidates are accepted for these tokens, constructing multi-branch block units for parallel validation. Tokens across branches are highly similar, introducing almost no new expert activations. Since MoE inference is memory-bound, increasing the arithmetic intensity of each expert utilizes idle compute resources.
    *   **Design Motivation**: For dense models, multi-branching immediately makes the process compute-bound, requiring multiple GPUs. In MoE, compute is naturally distributed across experts; on a single GPU, "adding branches without adding experts" is nearly free. This reinvests the budget saved by DCD into TPF (accepted tokens per forward), achieving 1.5–1.7× TPF.

3.  **LAC — Limited Activation for Cold Tokens**:
    *   **Function**: Cold mask tokens are not allowed independent routing and are forced to perform second-round routing within the "essential expert set $E_A$" defined by newly decoded and hot tokens.
    *   **Mechanism**: A two-round routing algorithm is used. The first round performs regular top-$k$ routing for newly accepted tokens $D_a$ and hot tokens $H$ to get weights $W_1$, forming the union $E_A = \text{top-}k(W_1)$. The second round runs $W_2 = Router(C, E_A)$ for cold tokens $C$, forcing them to select within $E_A$. The final weights are $W = Concat(W_1, W_2)$. Due to spatial consistency (concentrated mask token routing), this reuse is nearly harmless to cold tokens but eliminates their unique expert activations.
    *   **Design Motivation**: Activating experts exclusively for cold tokens that will not be accepted soon is wasteful. "Limiting activation instead of canceling" minimizes expert counts on the mask side without affecting future unexpected acceptance.

### Loss & Training
TEAM is entirely training-free with no gradient updates. It introduces three inference hyperparameters: hot confidence threshold $\tau_h$, hot spatial distance $L_h$, and top-$k$ for speculative branches. The paper uses $\tau_h \in [0.4, 0.8]$ and $L_h \in [2, 6]$.

## Key Experimental Results

### Main Results
The backbone is SDAR 30B-A3B (8 experts per token). Evaluation is conducted on HumanEval, MBPP, GSM8K, and Math-500. Metrics include task scores, **APF** (experts activated per forward), **TPF** (tokens accepted per forward), and **APT = APF / TPF**.

| Benchmark | Method | Score | APF↓ | TPF↑ | APT↓ | Speedup |
|---|---|---|---|---|---|---|
| HumanEval | Vanilla | 79.27 | 53.34 | 2.91 | 18.33 | 1× |
| HumanEval | TEAM | 79.88 (+0.61) | 34.48 (-35%) | 5.07 (1.74×) | 6.80 (-63%) | **2.20×** |
| MBPP | Vanilla | 65.76 | 49.59 | 2.74 | 18.10 | 1× |
| MBPP | TEAM | 65.76 (+0.00) | 30.92 (-38%) | 4.56 (1.66×) | 6.78 (-63%) | **2.08×** |
| GSM8K | Vanilla | 90.60 | 59.11 | 3.16 | 18.71 | 1× |
| GSM8K | TEAM | 90.30 (-0.30) | 36.20 (-39%) | 4.79 (1.52×) | 7.56 (-60%) | **1.83×** |
| Math-500 | Vanilla | 76.00 | 57.90 | 3.74 | 15.48 | 1× |
| Math-500 | TEAM | 75.40 (-0.60) | 36.31 (-37%) | 5.57 (1.49×) | 6.52 (-58%) | **1.64×** |
| **Avg** | TEAM | 77.84 (-0.07) | 34.48 (-37%) | 5.00 (1.59×) | 6.92 (-61%) | **1.94×** |

On average, accuracy drops only by 0.07, APT is reduced by 61%, and end-to-end speedup is 1.94×.

### Ablation Study
| Configuration | Avg Score | Avg APF | Description |
|---|---|---|---|
| Full TEAM (DCD+SEH+LAC) | 77.84 | 34.48 | Full model, 1.94× |
| Refresh-free DCD only | ≈ Vanilla | ↓ to ~26.5 (HumanEval) | DCD alone yields 1.58× (HumanEval), 1.32–1.55× across tasks |
| Refresh-8 DCD | -1.22 (HumanEval) | 29.61 | Periodic refresh actually results in lower precision |
| Refresh-4 DCD | -0 / -1.20 | 32.67 | Refreshing too frequently drops speedup from 1.58× to 1.38× |
| hot hyperparameter $(\tau_h, L_h)$ scan | 70.05 → 73.68 | 21.33 → 23.35 | $\tau_h=0.6, L_h=4$ is the optimal trade-off |

### Key Findings
- **DCD alone can achieve 1.3–1.6× speedup**, and Refresh-free is both faster and more accurate than Refresh-4/8. This confirms that for block-diffusion with AR-init, "delayed caching without refresh" is a safe strategy.
- **Hot hyperparameters are sensitive to precision but insensitive to APF**: Increasing $\tau_h$ from 0.4 to 0.8 only reduces APF from 23.35 to 21.33, while precision fluctuates. This suggests SEH gains primarily from "whether speculation is triggered," as branch counts have near-zero activation cost.
- **Speedup is inversely proportional to task length**: HumanEval/MBPP (short code generation) see up to 2.2×, while GSM8K/Math-500 (long reasoning chains) drop to 1.6–1.8×. Longer outputs have more blocks and naturally higher in-block acceptance rates, diluting TEAM's relative gains.

## Highlights & Insights
- **The "MoE × dLLM efficiency degradation" is a significant, overlooked issue**: The authors are the first to introduce the APF/TPF/APT metrics to reveal that sparse activation is negated by block parallelism, providing a clear motivation.
- **The paradigm of differentiated treatment for three token types** is highly transferable. Using "position/state" as routing variables rather than relying solely on learned router weights can extend to any scenario with non-uniform token importance, such as long-context AR inference or vision MoE.
- **"Reinvesting saved activation budget into speculative branches"** is the essence of SEH. While most acceleration works only perform subtraction (caching/pruning), TEAM uses subtraction (DCD/LAC) to free memory bandwidth and addition (SEH) to convert it into TPF.
- **LAC's two-round routing is almost a "free lunch"**: By leveraging spatial consistency, cold tokens are constrained to essential expert sets with negligible precision loss.

## Limitations & Future Work
- Only verified on SDAR 30B-A3B. Whether it holds for LLaDA 2.0 or future 100B+ MoE dLLMs requires more evidence.
- Speedup is lower for long reasoning (Math-500), and there are slight score drops (0.3–0.6). The "hot/cold division" might be too coarse for complex reasoning.
- DCD's refresh-free nature depends on AR initialization priors. If future dLLMs use purely non-autoregressive training (e.g., global attention without KV cache), these safety boundaries may need re-evaluation.
- Multi-branch validation in batched serving may compete for expert capacity with other requests; gains in high-throughput cloud scenarios may be lower than in single-stream settings.

## Related Work & Insights
- **vs dKV-Cache (Ma et al., 2025a)**: dKV-Cache targets global bidirectional dLLMs and requires global refreshes. TEAM proves refreshes are unnecessary in block-diffusion + AR-init settings and reveals that MoE caching gains come from expert suppression rather than attention savings.
- **vs dInfer (Ma et al., 2025b)**: dInfer focuses on expert-parallel deployment for cloud-scale MoE dLLMs; TEAM targets single-GPU/low-latency scenarios, focusing on reducing expert counts per forward pass.
- **vs speculative decoding (Chen 2023 / Leviathan 2023)**: Traditional AR speculation requires a draft model. SEH moves speculation inside the dLLM block without extra models, leveraging MoE compute distributability to make speculation "nearly free."

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically characterizes MoE dLLM activation mismatch; differentiated strategy combination is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 benchmarks with APF/TPF/APT metrics and single-component ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation figures (Fig 1/2) and well-structured categorization.
- Value: ⭐⭐⭐⭐ Provides ~2× speedup and is plug-and-play for MoE dLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](../../ACL2026/llm_efficiency/alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)
- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](../../ICLR2026/llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](../../ACL2026/llm_efficiency/breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)

</div>

<!-- RELATED:END -->
