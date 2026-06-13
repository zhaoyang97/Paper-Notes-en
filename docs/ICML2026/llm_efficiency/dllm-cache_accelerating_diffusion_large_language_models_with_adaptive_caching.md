---
title: >-
  [Paper Note] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching
description: >-
  [ICML 2026][LLM/NLP][Diffusion Large Language Models] Addressing the issue where Diffusion Large Language Models (dLLMs) suffer from extremely slow inference due to bidirectional attention being unable to reuse KV cache…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Diffusion Large Language Models"
  - "Inference Acceleration"
  - "Adaptive Caching"
  - "V-verify"
  - "LLaDA / Dream"
date: 2026-05-08
content_hash: ec607d06e06b6ec0
---

# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching

**Conference**: ICML 2026  
**arXiv**: [2506.06295](https://arxiv.org/abs/2506.06295)  
**Code**: https://github.com/maomaocun/dLLM-cache (Available)  
**Area**: LLM Efficiency  
**Keywords**: Diffusion Large Language Models, Inference Acceleration, Adaptive Caching, V-verify, LLaDA / Dream

## TL;DR
Addressing the issue where Diffusion Large Language Models (dLLMs) suffer from extremely slow inference due to bidirectional attention being unable to reuse KV cache, this paper proposes the training-free dLLM-Cache. It employs long-interval caching for static prompts and short-interval refreshing for dynamic responses, combined with a strategy that selects the top 25% most "dynamic" tokens for local recomputation based on Value cosine similarity. This achieves up to a 9.1× FLOPs speedup on LLaDA 8B / Dream 7B with negligible performance degradation.

## Background & Motivation

**Background**: Autoregressive LLMs (ARMs) have long dominated text generation, naturally supporting Key-Value caching through causal attention, which reduces generation complexity from $O(N^3)$ to $O(N^2)$. Recently, diffusion language models (dLLMs) like LLaDA and Dream have utilized masking, bidirectional attention, and multi-step denoising paradigms for text generation, effectively bypassing the "reversal curse" and achieving performance comparable to Llama 3 8B.

**Limitations of Prior Work**: Actual dLLM inference is prohibitively slow—generating a sequence of length $N$ requires $N$ denoising iterations. Each step necessitates a full recomputation of bidirectional attention across all tokens, resulting in $O(N^3)$ complexity. Furthermore, because bidirectional masks are non-monotonic, **traditional KV cache cannot be directly applied**. Even on an RTX 4090, LLaDA 8B achieves only 7.3 TPS on GSM8K, significantly lower than the 47.7 TPS of a similarly sized Llama 3 8B.

**Key Challenge**: Bidirectional attention is both a performance advantage (global context visibility) and an efficiency bottleneck (lack of causal pruning due to mutual dependencies). Naive "refresh every K steps" uniform strategies either suffer from significant performance drops or offer limited computational savings.

**Goal**: To identify a caching strategy that accurately characterizes the computational redundancy of dLLMs without retraining, bringing dLLM inference speeds closer to those of ARMs.

**Key Insight**: The authors analyzed heatmaps of cosine similarity for Key/Value/AttnOut/FFNOut between adjacent denoising steps, observing two strong signals: (1) The **prompt region** remains almost entirely consistent across all steps (similarity near 1) because the input is static; (2) The **response region** shows high overall similarity, but **only a few tokens undergo sudden changes** at certain steps. These changes in Value are highly correlated with changes in downstream AttnOut/FFNOut. Thus, prompts and responses should be treated differently, and the response region must further distinguish between "stable tokens" and "active tokens."

**Core Idea**: A three-stage adaptive caching mechanism: long-interval prompt caching, short-interval full response refreshing, and Value-similarity-guided local updates to eliminate both types of redundancy in dLLMs.

## Method

### Overall Architecture
dLLM-Cache is a training-free plugin for the dLLM inference forward pass. For each Transformer layer $l$, it maintains two caches: the **Prompt Cache** $\mathcal{C}_p$ storing $\mathbf{K}^{(l)}, \mathbf{V}^{(l)}, \mathbf{AttnOut}^{(l)}, \mathbf{FFNOut}^{(l)}$ for the prompt segment; and the **Response Cache** $\mathcal{C}_r$ storing the same four sets of features for the response segment. Three hyperparameters control the refresh rhythm: the prompt refresh interval $K_p$ (typically 50–100), the response full refresh interval $K_r$ (typically 5–10), and the adaptive update ratio $\rho \in [0,1]$ (fixed at 0.25).

Inference process: At step 0 ($k=K$), a full forward pass is performed, and prompt/response features are written into their respective caches. Subsequently, as $k$ decrements from $K-1$ to 1, three actions are taken at each step and layer: (1) If $k \equiv 0 \pmod{K_p}$, the prompt segment is recomputed to refresh $\mathcal{C}_p$; otherwise, it's read from the cache. (2) If $k \equiv 0 \pmod{K_r}$, the entire response segment is recomputed to refresh $\mathcal{C}_r$; otherwise, the "partial update" branch (V-verify) is executed. (3) The layer proceeds with the current features.

### Key Designs

1.  **Long-Interval Prompt Cache**:
    *   **Function**: Caches the intermediate features of the prompt segment—which are otherwise recomputed at every step despite remaining almost unchanged—and recomputes them only every $K_p$ steps.
    *   **Mechanism**: dLLM training uses per-token independent random masking, meaning prompt tokens remain **constant inputs** across all denoising steps. Empirical tests show that cosine similarities for K/V/AttnOut/FFNOut in the prompt segment are near 1. Thus, $\mathbf{K}_p^{(l)}, \mathbf{V}_p^{(l)}, \mathbf{AttnOut}_p^{(l)}, \mathbf{FFNOut}_p^{(l)}$ are cached and recomputed only when $k \equiv 0 \pmod{K_p}$. Notably, this caches not just KV but also Attention and FFN outputs, allowing the prompt segment to **entirely bypass the Transformer layer**.
    *   **Design Motivation**: dLLMs often handle long prompt + short response scenarios; repeated recomputation of the prompt is the primary source of overhead. This also distinguishes dLLM-Cache from concurrent works like dKV-Cache/Fast-dLLM, which only cache KV while still recomputing FFN.

2.  **V-verify-guided Adaptive Response Update**:
    *   **Function**: Between full response refreshes, only the $\lfloor \rho |\mathbf{y}^{(k)}| \rfloor$ most "violently changing" tokens in the response are recomputed; the rest are read from the cache.
    *   **Mechanism**: A lightweight projection calculates the new $\mathbf{V}_r^{\text{new}}$ for all response tokens in the current layer. For each token $j$, the cosine similarity $s_j = \frac{(\mathbf{v}_{r,j}^{(l)})^\top \tilde{\mathbf{v}}_{r,j}^{(l)}}{\|\mathbf{v}_{r,j}^{(l)}\| \|\tilde{\mathbf{v}}_{r,j}^{(l)}\|}$ is computed relative to the cached $\tilde{\mathbf{v}}_{r,j}^{(l)}$. The proportion $\rho$ of tokens with the lowest $s_j$ are deemed "active," for which $\mathbf{K}, \mathbf{AttnOut}, \mathbf{FFNOut}$ are fully recomputed and scattered back into the cache. Since $\mathbf{V}_r$ is already computed for the check, the entire $\mathbf{V}_r$ cache is updated.
    *   **Design Motivation**: Empirical results in Figure 2 demonstrate a strong correlation between changes in V (and K) and downstream AttnOut/FFNOut. Therefore, **using the early and cheap V can predict whether downstream expensive features need updating**, avoiding the paradox of "recomputing everything just to decide whether to update."

3.  **Differentiated Scheduling of Prompt + Response ($K_p \gg K_r$, $\rho \approx 0.25$)**:
    *   **Function**: Manages the two types of redundancy—static prompts versus slowly and non-uniformly evolving responses—using distinct refresh frequencies.
    *   **Mechanism**: Defaults are set to $K_p = 50\text{–}100$ (infrequent prompt refreshes), $K_r = 5\text{–}10$ (frequent but local response refreshes), and $\rho = 0.25$ (updating only 1/4 of tokens). This scheduling introduces only three hyperparameters that rarely require tuning across tasks or models. The additional memory cost is $T \times d \times 4 \times L$, which for LLaDA 8B is only +1 GB (~5% VRAM).
    *   **Design Motivation**: Ablation studies show that uniform caching ($K_p = 1, \rho = 0$ or solely increasing $K_r$) either saves little computation or causes significant performance drops. Differentiating the two token types allows for 5×–9× acceleration on GSM8K without performance loss.

### Loss & Training
Entirely training-free. It does not modify model weights or require cache-aware fine-tuning; it is simply integrated into the inference forward pass of LLaDA / Dream.

## Key Experimental Results

### Main Results
Evaluated on LLaDA 8B Base/Instruct and Dream 7B Base/Instruct across 8 benchmarks (GSM8K / GPQA / Math / MMLU-pro / MMLU / BBH / MBPP / HumanEval) using a single RTX 4090 with $\rho = 0.25$.

| Model | Task | TPS (baseline → +Cache) | FLOPs Gain | Score Change |
| :--- | :--- | :--- | :--- | :--- |
| LLaDA Base | GSM8K | 7.32 → 23.19 | 5.02× | 69.06 → 70.66 (+1.60) |
| LLaDA Instruct | GPQA | 5.33 → 28.01 | **8.08×** | 29.01 → 29.01 (0) |
| LLaDA Instruct | BBH | 6.18 → 27.55 | 6.16× | 51.49 → 51.98 (+0.49) |
| Dream Base | GSM8K | 6.36 → 32.44 | 6.90× | 76.95 → 76.95 (0) |
| Dream Base | GPQA | 5.80 → 30.95 | 7.15× | 33.92 → 34.15 (+0.23) |
| Dream Instruct | MMLU | 8.45 → 38.01 | 6.10× | 73.40 → 73.42 (+0.02) |

Comparison with Llama 3 8B on GSM8K: LLaDA Base with 256 steps originally yields 7.37 TPS / 69.06%. With dLLM-Cache, it reaches 20.64 TPS / 70.66% (at a 5% VRAM cost). Combined with SlowFast Sampling, it reaches 49.86 TPS / 67.17%, matching Llama 3 8B's throughput (47.73 TPS) while outperforming it in accuracy by 18.12 percentage points.

Comparison with concurrent works (LLaDA Instruct + Dream Base baseline):

| Task | dKV-Cache | Fast-dLLM | **dLLM-Cache** |
| :--- | :--- | :--- | :--- |
| GPQA (Dream Base) | 1.74× / 32.83 | 3.83× / 31.31 | **5.33× / 34.15** |
| MMLU (LLaDA Inst) | 1.42× / 60.87 | 2.03× / 61.43 | **2.10× / 62.82** |
| HumanEval (LLaDA Inst) | 1.36× / 37.20 | 2.03× / 36.59 | **4.24× / 39.02** |

### Ablation Study

| Configuration | Key Findings |
| :--- | :--- |
| Token selection: V-verify vs K-verify vs random (scanning $\rho$) | Both similarity strategies significantly outperform random; V-verify achieves the best score/FLOPs trade-off at $\rho \approx 0.25$. |
| $K_p$ scan ($K_r=1, \rho=0$) | Increasing $K_p$ from 1 to 100 drastically reduces FLOPs while accuracy remains nearly constant → prompts can be refreshed very sparsely. |
| $K_r$ scan, $K_p=1, \rho=0$ (Uniform) vs $K_p=50, \rho=0.25$ (Ours) | Uniform caching accuracy drops sharply as $K_r$ increases; our configuration maintains high scores at lower FLOPs. |
| 256-step baseline reduced to 32 steps | TPS improves to 53.55 but GSM8K plunges to 22.25%. dLLM-Cache (256 steps) yields 20.64 TPS / 70.66%, showing acceleration should not come from simple step reduction. |
| TPS vs. $\rho$ curve | TPS initially drops when $\rho > 0$ (fixed overhead from kernel launch/scatter) then decreases slowly as $\rho$ rises. |
| Storage Overhead | Caching 4 feature types per layer: $T \cdot d \cdot 4 \cdot L$. Measured at +1 GB (5%) for LLaDA 8B. |

### Key Findings
- The primary contribution stems from the differentiated scheduling of "long-interval prompt + short-interval local response updates." Removing either prompt caching or local response updates leads to significant performance loss or insufficient acceleration.
- The effectiveness of V-verify is built on a non-trivial empirical observation: the cosine similarity of response token Value is strongly correlated with its downstream AttnOut/FFNOut similarity. This allows using cheap Vs to predict expensive features—the cornerstone of the local update strategy.
- $\rho \approx 0.25$ is a universal "sweet spot": smaller values see gains consumed by fixed GPU kernel overheads, while larger values compute too much.
- dLLM-Cache is **orthogonal** to "step reduction/parallel decoding" methods like SlowFast Sampling. Their combination allows dLLM to match Llama 3 8B throughput while maintaining higher accuracy.

## Highlights & Insights
- **Correct Decomposition of dLLM Inference Redundancy**: Prompt redundancy across steps vs. dynamic sparse token redundancy in responses. This dual decomposition is more granular than ARM's KV cache and fits the bidirectional nature of dLLMs better—this is the paper's cleanest insight.
- **Using Value as a Proxy for Downstream Change**: This avoids the infinite loop of "needing to calculate everything to decide if it needs calculation." The principle can be transferred to any "iterative refinement" model (e.g., multi-step image diffusion), provided an early, cheap signal correlates with late-stage features.
- **Caching Beyond KV**: By including AttnOut and FFNOut, the method essentially "bypasses the entire Transformer layer." This is the fundamental reason it outperforms dKV-Cache and Fast-dLLM. The insight: In bidirectional dLLMs where FFN is heavy, **the FFN is the component truly worth skipping**, not just attention.

## Limitations & Future Work
- Experiments focused on LLaDA 8B and Dream 7B. Generalization to other dLLM variants (e.g., multimodal MaskedDiff) is unverified. While $K_p/K_r$ are robust, optimal values may require minor scanning per model.
- TPS decreases when $\rho$ is very small—indicating that V-verify + scatter implementation is hindered by fixed costs like GPU kernel launches and memory movement. Future work could use fused selective-recomputation at the operator level to smooth these costs as $\rho \to 0$.
- The prompt-response boundary is statically defined, needing extension for dynamic lengths, streaming prompt modifications, or Chain-of-Thought scenarios. V-verify token selection is layer-local; synchronizing active token sets across layers might further reduce computation.

## Related Work & Insights
- **vs. dKV-Cache (Ma et al., 2026)**: Both add caching to dLLMs, but dKV-Cache only reuses KV after decoding. dLLM-Cache uses prompt/response bifurcation, V-verify local updates, and caches up to the FFN output level, yielding 5.33× vs 1.74× on GPQA/Dream Base.
- **vs. Fast-dLLM (Wu et al., 2026)**: Fast-dLLM uses block-level approximate KV cache + confidence-aware parallel decoding (a "step-reduction + coarse caching" route). dLLM-Cache follows a "step-retention + fine-grained adaptive caching" route, resulting in more stable quality.
- **vs. SlowFast Sampling (Wei et al., 2026)**: A step-reduction approach orthogonal to this work. Combining them allows LLaDA 8B to reach 49.86 TPS while maintaining 67.17% on GSM8K—likely the current Pareto optimal for dLLM acceleration.
- **vs. ARM KV-Cache (Pope et al., 2023)**: ARMs use casual masks for lossless KV caching. This work adapts the idea of "reusing computed features" to bidirectional attention, translating "reusing historical tokens" into "reusing the same token from the previous denoising step."

## Rating
- Novelty: ⭐⭐⭐⭐ First to propose training-free, fine-grained adaptive caching for dLLM bidirectional attention. V-verify is a simple yet powerful proxy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 2 dLLM families × 8 benchmarks, compares with 2 concurrent works, provides cross-paradigm comparison with Llama 3 8B, and ablates all hyperparameters.
- Writing Quality: ⭐⭐⭐⭐ The redundancy analysis in Figures 1/2 forms a strong backbone; the narrative is concise, though some figure descriptions depend heavily on the originals.
- Value: ⭐⭐⭐⭐⭐ Pulls dLLM inference speed to near-ARM levels for the first time, addressing the biggest bottleneck for production deployment. Its orthogonality with parallel decoding makes it highly valuable for engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](spa-cache_singular_proxies_for_adaptive_caching_in_diffusion_language_models.md)
- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](../../ICLR2026/llm_nlp/d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)
- [\[ICML 2026\] Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference.md)
- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)

</div>

<!-- RELATED:END -->
