---
title: >-
  [Paper Note] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models
description: >-
  [ICML 2026][LLM/NLP][Diffusion LM] SPA-Cache shifts the determination of which tokens require updating in Diffusion Language Models (DLMs) from the original $d=4096$ dimensional Value space to a singular subspace of $r=1…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Diffusion LM"
  - "KV Cache"
  - "Singular Value Decomposition"
  - "Adaptive Budget"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: f8a962c657855c74
---

# SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.02544](https://arxiv.org/abs/2602.02544)  
**Code**: https://github.com/wenhao728/spa-cache (Available)  
**Area**: LLM Efficiency / Diffusion Language Models / Inference Acceleration  
**Keywords**: Diffusion LM, KV Cache, Singular Value Decomposition, Adaptive Budget, Inference Acceleration

## TL;DR
SPA-Cache shifts the determination of which tokens require updating in Diffusion Language Models (DLMs) from the original $d=4096$ dimensional Value space to a singular subspace of $r=128$. By dynamically allocating update budgets across layers, it achieves a $6.4\times$ throughput increase for LLaDA-8B on GSM8K and $8\times$ on MBPP without accuracy degradation, reaching a total speedup of $28\times$ when integrated with parallel decoding.

## Background & Motivation
**Background**: Diffusion Language Models (DLMs) such as LLaDA-8B and Dream-7B replace the left-to-right generation paradigm of AR models with bidirectional attention and arbitrary-order decoding, demonstrating competitiveness in multimodal tasks, reasoning, and addressing the "reversal curse." however, DLMs require a full forward pass of the sequence length $N$ at each decoding step, resulting in $O(T \cdot N^2)$ complexity, which is highly inefficient compared to the KV-Cache in AR models.

**Limitations of Prior Work**: Standard KV-Cache is inapplicable due to non-fixed decoding orders. Subsequent works follow two paths: (i) dKV-Cache, d2Cache, and Fast-dLLM adopt window heuristics, assuming only hidden states near recently decoded tokens need updating, which lacks a theoretical basis; (ii) dLLM-Cache monitors Value state drift to identify "drift tokens" at arbitrary positions, but calculating projections and similarities in $d$-dimensional space at each step incurs high overhead. Furthermore, these methods use a uniform update ratio $\rho$ across all layers, distributing a fixed budget equally.

**Key Challenge**: There is a trade-off between identification overhead and sparse gains—reducing the update ratio $\rho$ decreases attention/FFN computation, but the $d$-dimensional similarity calculation itself consumes most of the savings. Additionally, as observed in Figure 2, the proportion of drift tokens varies significantly across layers: shallow layers perform embedding transformations, deep layers tend to stabilize, and middle layers are peaks for drift. Using a uniform $\rho=25\%$ wastes budget in shallow/deep layers, while $\rho=20\%$ misses tokens in high-variance middle layers.

**Goal**: Simultaneously optimize the identification of "which tokens to update" and the "allocation of update budgets per layer."

**Key Insight**: A formal analysis of DLM hidden state evolution proves that the cosine similarity of Value states provides an upper bound for the drift in attention and FFN outputs (Theorem 3.1/3.2). Moreover, truncated SVD of $W$ preserves the similarity structure within an $r \ll d$ subspace (Theorem 3.4). This implies that similarity does not need to be calculated on full-dimensional Values; the first $r$ singular directions are sufficient.

**Core Idea**: Use "singular proxies" constructed via truncated SVD of $W$ to replace full-dimensional Values for identifying drift tokens, and apply a piecewise Gaussian function to adaptively allocate the update budget $\rho(l)$ per layer.

## Method

### Overall Architecture
The SPA-Cache workflow for a single Transformer block consists of three phases (Algorithm 1):

- **Phase 1: Update Identification**: Input hidden states $H \in \mathbb{R}^{N \times d}$ are projected via a low-dimensional $f_\text{proxy}: \mathbb{R}^d \to \mathbb{R}^r$ to obtain proxy identifiers. These are compared with cached proxies from the previous step using cosine similarity. The Top-$k$ ($k = N\rho$) least similar indices $\mathcal{I}$ are selected based on the current layer's budget $\rho(l)$.
- **Phase 2: Partial Cached Attention**: $Q_\mathcal{I}, K_\mathcal{I}, V_\mathcal{I}$ are computed only for tokens in $\mathcal{I}$ and written back to the $K^c, V^c$ caches via scatter operations. A sparse query performs attention against the full KV cache to produce sparse outputs.
- **Phase 3: FFN and Output Update**: Sparse attention outputs pass through the FFN and are scattered back to the output cache $H^c$. Unselected tokens directly reuse their cached states.

**Mechanism**: The computation for attention and FFN in each layer is reduced from $O(N)$ to $O(k) = O(N \rho)$, with $\rho$ adaptively adjusted per layer.

### Key Designs

1. **Theoretical Foundation: Why Value is a Valid Drift Indicator**:
    - **Function**: Theoretically explains why Value states, rather than Query, Key, or attention outputs, should be used to determine token updates.
    - **Core Idea**: Theorem 3.1 proves that the cosine dissimilarity of attention outputs is bounded by Value dissimilarity: $1 - \mathcal{S}_\cos(h_i^t, h_i^{t+1}) \le C \cdot (1 - \mathcal{S}_\cos(v_i^t, v_i^{t+1})) + \epsilon$. Theorem 3.2 further proves that FFN output differences are bounded by input similarity: $\|f_\text{FFN}(h_1) - f_\text{FFN}(h_2)\|_2 \le C \cdot \sqrt{1 - \mathcal{S}_\cos(h_1, h_2)} + \epsilon$. Together, these imply: Stable Value $\to$ Stable Attention $\to$ Stable FFN.
    - **Design Motivation**: While dLLM-Cache used Value based on empirical observation, this work explains why other components are less suitable. Empirical results (Table 1) confirm this: Value is the only indicator maintaining 78.59% accuracy and 164.88 TPS; attention outputs drop accuracy to 73.92% due to anisotropy in deeper layers projecting different tokens into a narrow cone.

2. **Singular Proxies: Reducing Identification from $d^2$ to $rd$**:
    - **Function**: Replaces full-dimensional Value projection and cosine calculation, reducing identification overhead from $O(d^3) + O(d)$ to $O(rd^2) + O(r)$.
    - **Core Idea**: SVD is applied to the Value projection matrix $W \in \mathbb{R}^{d \times d}$: $W \approx U \Lambda V^\top$. A truncated projection $W_r = \Lambda_r V_r^\top \in \mathbb{R}^{r \times d}$ is constructed using the first $r$ singular vectors, with the proxy defined as $f_\text{proxy}(h_i) = W_r h_i$. Theorem 3.4 proves the similarity error after truncation is bounded by $2(\lambda_{r+1}/\lambda_r)^2$, independent of specific inputs.
    - **Design Motivation**: For LLaDA-8B where $d=4096$, full-dimensional similarity offsets sparse gains. Using $r=128$ (a 32$\times$ reduction), TPS increases from 164.88 to 179.43 with negligible accuracy loss (78.59 to 78.23, Table 5). $r=128$ is identified as the optimal balance point.

3. **Piecewise Gaussian Adaptive Budget**:
    - **Function**: Transforms the update ratio $\rho(l)$ from a constant to a curve following the drift distribution, maintaining the average budget while allocating more resources to high-variance middle layers.
    - **Core Idea**: Layer-wise budget is parameterized using a piecewise Gaussian function centered at peak layer $l_p$: $\rho(l) = \rho_p \exp\!\left(\ln(\rho_1/\rho_p) \cdot ((l-l_p)/(l_p-1))^2\right)$ for $l \le l_p$, with a symmetric branch for $l > l_p$. $\rho_p$ is the peak ratio (default 25%), while $\rho_1, \rho_L$ are ratios for the first and last layers.
    - **Design Motivation**: Figure 2 shows that the proportion of drift tokens follows an asymmetric bell shape—stable at shallow (embedding) and deep (integration) layers, and peaking at middle (transformation) layers. Allocation improves throughput while maintaining accuracy (Table 4: TPS 179 $\to$ 189).

### Loss & Training
**Training-free**: SPA-Cache is an inference-time plug-in. It does not modify DLM weights but adds proxy projections and Top-$k$ selection. All hyperparameters ($r=128$, $\rho_p=0.25$, and Gaussian parameters $l_p, \rho_1, \rho_L$) are configured once per model.

## Key Experimental Results

**Setting**: Evaluated on LLaDA-8B-Instruct and Dream-v0-Instruct-7B across 7 benchmarks (GSM8K, MATH500, GPQA, BBH, MMLU-pro, MBPP, HumanEval) against vanilla decoding, dLLM-Cache, and Fast-dLLM. Experiments were conducted on a single NVIDIA B200.

### Main Results (LLaDA-8B-Instruct)

| Benchmark | Baseline TPS | dLLM-Cache TPS | Fast-dLLM TPS | SPA-Cache TPS | Speedup | Accuracy (SPA vs Baseline) |
|-----------|-------------:|---------------:|--------------:|--------------:|-------:|-----------------------:|
| GSM8K | 29.67 | 68.62 ($2.3\times$) | 93.86 ($3.2\times$) | **190.73** | $6.4\times$ | 78.24 / 78.62 |
| MATH500 | 33.35 | 74.26 ($2.2\times$) | 85.94 ($2.6\times$) | **172.19** | $5.2\times$ | 33.44 / 33.18 |
| MMLU-pro | 20.68 | 52.71 ($2.5\times$) | 81.25 ($3.9\times$) | **124.06** | $6.0\times$ | 36.30 / 37.08 |
| MBPP | 5.75 | 8.38 ($1.5\times$) | 12.49 ($2.2\times$) | **46.12** | $8.0\times$ | 39.00 / 39.20 |
| HumanEval | 37.48 | 40.29 ($1.1\times$) | 81.90 ($2.2\times$) | **132.91** | $3.5\times$ | 42.07 / 42.07 |

Accuracy remains nearly identical to the baseline, while throughput is typically 2-5$\times$ that of dLLM-Cache and 1.5-3.7$\times$ that of Fast-dLLM.

### Combined with Parallel Decoding (Table 3, LLaDA-8B)

| Benchmark | Baseline | Fast-dLLM Parallel | SPA-Cache + Parallel | Total Speedup |
|-----------|---------:|---------------:|-----------------:|-------:|
| GSM8K | 29.67 | 176.45 ($5.9\times$) | **276.39** | $9.3\times$ |
| BBH | 24.85 | 301.33 ($12.1\times$) | **693.96** | $\mathbf{27.9\times}$ |
| MMLU-pro | 20.68 | 86.40 ($4.2\times$) | **224.97** | $10.9\times$ |
| MBPP | 5.75 | 50.11 ($8.7\times$) | **143.25** | $24.9\times$ |

SPA-Cache is orthogonal to parallel decoding; combining them achieves nearly $28\times$ total speedup on BBH, outperforming the dual cache in Fast-dLLM.

### Ablation Study (LLaDA-8B, GSM8K, Table 4-5)

| Configuration | Peak $\rho_p$ | Avg $\bar\rho$ | TPS | Accuracy |
|------|-------------:|----------------:|-----:|-----:|
| Baseline (No cache) | 100% | 100% | 29.01 | 78.62 |
| Full-dim Value | 25% | 25% | 164.88 | 78.59 |
| + Singular-128 (Proxy) | 25% | 25% | 179.43 | 78.23 |
| + Adaptive Budget | 25% | 16% | **189.13** | 78.24 |
| Uniform 16% (No adaptive) | 16% | 16% | 190.06 | 75.65 |

### Key Findings
- **Singular proxies contribute speedup without loss**: Moving from full Value to Singular-128 increases TPS by 9% with only a 0.36 accuracy drop, validating the similarity-preserving bound in Theorem 3.4.
- **Adaptive budget is a "free lunch"**: Reducing the average budget from 25% to 16% increases TPS by 5% while keeping accuracy stable. In contrast, a uniform 16% budget across all layers causes a 2.59-point drop.
- **Maximum speedup on MBPP ($8\times$)**: Long sequences in MBPP allow for the greatest absolute gains from sparse computation; HumanEval's shorter sequences yield a relatively smaller $3.5\times$ speedup.
- **dLLM-Cache overhead**: On Dream-7B for HumanEval, dLLM-Cache actually slows down the baseline ($0.8\times$), confirming that full-dimensional identification overhead can negate sparse gains.

## Highlights & Insights
- **From empirical heuristics to bounded theory**: While dLLM-Cache chose Value ad-hoc, this work uses theorems to establish Value similarity as an upper bound for attention/FFN drift, and SVD as a bounded error proxy.
- **Novel application of SVD on weights**: SVD is traditionally used for weight compression; here it constructs "proxy identifiers" that retain singular directions for similarity calculation without modifying the weights themselves.
- **Parameterized layer heterogeneity**: Using a piecewise Gaussian function instead of per-layer scalars provides a inductive bias ("high middle, low ends") with only 4 hyperparameters, avoiding overfitting or extensive search.
- **Orthogonality with parallel decoding**: Caching optimizes per-step computation while parallel decoding optimizes the number of tokens per step. Combining them yields a $28\times$ compound speedup.

## Limitations & Future Work
- **Model Scale**: Validation is limited to 8B-scale models; performance on larger models (30B+) with different singular spectrum decays remains to be seen.
- **Hyperparameter Tuning**: The 4 Gaussian parameters are empirically set. The paper provides values for LLaDA and Dream but lacks an automated selection scheme.
- **Optimal $r$**: The study does not explore whether different tasks (e.g., code vs. reasoning) require different proxy dimensions $r$.
- **Training Acceleration**: SPA-Cache is restricted to inference. Training costs for DLMs remain an open challenge.

## Related Work & Insights
- **vs dLLM-Cache**: SPA-Cache refines the Value similarity approach with lower computational costs and finer-grained budget allocation, theoretically grounding the use of Value.
- **vs Fast-dLLM**: Fast-dLLM uses windows and parallel decoding; SPA-Cache uses global token selection. They are orthogonal and yield superior results when combined.
- **vs dKV-Cache / d2Cache**: These methods rely on proximity heuristics and lack global dependency modeling. SPA-Cache captures long-range drifts via similarity-based Top-$k$ selection.

## Rating
- Novelty: ⭐⭐⭐⭐ Using weight SVD for identification proxies is an innovative angle; the adaptive piecewise Gaussian budget is a clever inductive bias.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 benchmarks across multiple baselines; lacks validation on very large-scale models.
- Writing Quality: ⭐⭐⭐⭐ Clear "Theory -> Algorithm -> Evidence" structure.
- Value: ⭐⭐⭐⭐ Significant throughput gains ($8\times$ to $28\times$) for a training-free, low-engineering-barrier method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](../../ICLR2026/llm_nlp/d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [\[ICML 2026\] Rethinking LLM Ensembling from the Perspective of Mixture Models](rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)
- [\[ICML 2026\] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)
- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](../../ICLR2026/llm_nlp/stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)

</div>

<!-- RELATED:END -->
