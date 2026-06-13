---
title: >-
  [Paper Note] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts
description: >-
  [ICML 2026][LLM Efficiency][Mixture-of-Experts] ProbMoE reformulates top-$k$ routing as "probabilistic inference over cardinality-constrained subset distributions." It uses the SIMPLE estimator to sample from the exact-$…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Probabilistic Routing"
  - "Subset Sampling"
  - "SIMPLE Gradient Estimator"
  - "Dynamic Expert Allocation"
date: 2026-05-08
content_hash: f3cf128b524a582b
---

# ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2606.01509](https://arxiv.org/abs/2606.01509)  
**Code**: https://github.com/HengHugoZhao/ProbMoE.git (Available)  
**Area**: LLM Efficiency / MoE Routing  
**Keywords**: Mixture-of-Experts, Probabilistic Routing, Subset Sampling, SIMPLE Gradient Estimator, Dynamic Expert Allocation

## TL;DR
ProbMoE reformulates top-$k$ routing as "probabilistic inference over cardinality-constrained subset distributions." It uses the SIMPLE estimator to sample from the exact-$k$ subset distribution during the forward pass and employs analytically computed expert marginal probabilities $m_j=\partial \log Z_k/\partial \log p_j$ as a differentiable proxy for discrete selection during the backward pass. This approach significantly improves performance on GSM/Law/Translation tasks for OLMoE and Qwen1.5-MoE while notably enhancing expert utilization. It also naturally extends to a Dynamic-$k$ variant that adaptively activates expert counts based on token difficulty.

## Background & Motivation
**Background**: Sparse MoE achieves scaling with "parameter counts far exceeding activation FLOPs" by activating only $k$ experts per token (e.g., Switch Transformer, GLaM, DeepSeek-MoE). The core component is a softmax router combined with a top-$k$ selector.

**Limitations of Prior Work**: The top-$k$ operator is discrete and piecewise constant, resulting in zero gradients almost everywhere with respect to router logits. Standard training treats the selection $S_{\text{top-}k}$ as a fixed constant in the forward pass and only backpropagates gradients through the softmax weights $\pi_j$ of the selected experts (omitting the "discrete-selection path" in Eq. 2). Consequently, the router receives no learning signal regarding "unselected alternative subsets," leading to increasingly sharp routing distributions, repeated reinforcement of a few experts, expert collapse, and training instability.

**Key Challenge**: The router essentially needs to learn a *discrete combinatorial object* ($k$-subset selection). However, existing methods either use heuristic noise/reshuffling to approximate stochasticity (Shazeer et al.) or use dense STE (DenseMixer) to compute gradients over all experts. These approaches are "patches on a deterministic top-$k$" and fail to explicitly model the "distribution over $k$-subsets," preventing systematic exploration of alternative subsets.

**Goal**: (i) Rewrite the router training objective as the expected loss under a *subset distribution* $\mathcal{J}(\theta)=\mathbb{E}_{S\sim\mathbb{P}_r(\cdot\mid|S|=k)}[\mathcal{L}(y_S(x;r))]$; (ii) Provide gradients that reflect the entire subset distribution while maintaining sparse activation of only $k$ experts per step; (iii) Naturally extend the framework to dynamic-$k$ ($k\in[k_{\min},k_{\max}]$).

**Key Insight**: The authors observe that the SIMPLE estimator (Ahmed et al. 2023) can compute exact normalization for a Bernoulli product distribution conditioned on "exactly $k$ selections" in $\mathcal{O}(Nk)$ time, providing analytical conditional marginal probabilities $m_j$ for each variable. By treating each expert selection as an independent Bernoulli $p_i=\sigma(r_i)$ conditioned on $|S|=k$, routing becomes a probabilistic layer with exact normalization and analytical marginals.

**Core Idea**: Replace "top-$k$ & softmax-only gradient path" with *"sampling from a $k$-cardinality subset distribution + using conditional marginals as a backward proxy"*. This transforms routing into truly differentiable discrete probabilistic inference. The same normalization constant can be summed to $Z^*=\sum_{k=k_{\min}}^{k_{\max}} Z_k$ to derive a range-constrained dynamic-$k$ version.

## Method

### Overall Architecture
Consider an MoE layer with $N$ experts and token hidden state $x\in\mathbb{R}^d$. The router outputs logits $r=\mathrm{Router}_\theta(x)\in\mathbb{R}^N$ and softmax weights $\pi_i=\exp(r_i)/\sum_j\exp(r_j)$. Given subset $S$, MoE output is $y_S(x;r)=\sum_{j\in S}\pi_j f_j(x)$.

ProbMoE models routing as a two-stage hierarchical distribution: independent Bernoulli $p_i=\sigma(r_i)$ for each expert → conditoned on cardinality constraints (exact-$k$ or range $[k_{\min},k_{\max}]$) to obtain the subset distribution $\mathbb{P}_r(S\mid \cdot)$. Pipeline:

1. **Forward**: Compute $p_i$ from logits → compute exact normalization $Z_k$ via SIMPLE in $\mathcal{O}(Nk)$ → sample a $k$-hot mask $z\in\{0,1\}^N$ → execute only the $k$ selected experts $f_j(x)$.
2. **Marginal Calculation**: Use dynamic programming to compute conditional marginals for each expert $m_j=\mathbb{P}_r(j\in S\mid |S|=k)=\partial \log Z_k/\partial \log p_j$, which is analytical and differentiable.
3. **Routing Weight Assembly**: Combine the sampled mask, marginals, and softmax via STE: $w=(\operatorname{stopgrad}(z-m)+m)\odot\pi$. In the forward pass, $w=z\odot\pi$ (remains sparse); in the backward pass, gradients flow through both $m$ and $\pi$.
4. **Inference**: Select the MAP subset (top-$k$ of $m$ for exact-$k$; joint selection of $k$ and $S$ within $[k_{\min},k_{\max}]$ for dynamic-$k$). Inference cost remains equal to standard MoE.

### Key Designs

1. **Cardinality-constrained Subset Distribution + SIMPLE Normalization**:
    - **Function**: Replaces the "deterministic top-$k$ operator" in the router output layer with a "$k$-cardinality Bernoulli subset distribution" and provides an exact normalization constant $Z_k=\sum_{|S|=k}\prod_{j\in S}p_j\prod_{j\notin S}(1-p_j)$ computable in $\mathcal{O}(Nk)$ ($\mathcal{O}(\log N\log k)$ via vectorization).
    - **Mechanism**: Independent Bernoulli $p_i=\sigma(r_i)$ form an unconstrained product measure; conditioning on $|S|=k$ yields $\mathbb{P}_r(S\mid|S|=k)=Z_k^{-1}\prod_{j\in S}p_j\prod_{j\notin S}(1-p_j)$. SIMPLE uses 1D convolution-style DP to recursively compute $Z_k$ by considering the inclusion of the $i$-th expert, avoiding explicit enumeration of $\binom{N}{k}$ subsets. Generalizing to range constraints requires replacing a single $Z_k$ with $Z^*=\sum_{k=k_{\min}}^{k_{\max}} Z_k$ (Theorem 5.1, complexity remains $\mathcal{O}(Nk_{\max})$).
    - **Design Motivation**: Previous continuous relaxations like Gumbel-Softmax or Concrete are biased or high-variance and cannot explicitly represent hard constraints like "exactly $k$." Subset enumeration is combinatorial. DP normalization via SIMPLE makes "exact probabilistic inference over combinatorial spaces" feasible in MoE routers for the first time, serving as the foundation for exact marginals, sampling, and dynamic $k$.

2. **Marginal-Embedded Routing Weights + Straight-Through Backward**:
    - **Function**: Maintains sparse $k$-expert evaluation in the forward pass while ensuring router backward gradients reflect the *entire* subset distribution's dependence on each logit, rather than just the softmax weights of selected experts.
    - **Mechanism**: Uses conditional marginals $m_j=\partial \log Z_k/\partial \log p_j$ as differentiable "summaries" of discrete choices, constructing routing weights via STE: $w=(\operatorname{stopgrad}(z-m)+m)\odot\pi$. Forward $w_i=z_i\pi_i$ (sparsity preserved), while the backward gradient decomposes as $\partial \mathcal{L}/\partial r_i=\sum_j \langle \partial \mathcal{L}/\partial y, f_j(x)\rangle (m_j \partial \pi_j/\partial r_i + \pi_j \partial m_j/\partial r_i)$. The second term is the new "marginal path," transmitting information about "what if an alternative subset was chosen" back to the router. Synthetic experiments in Appendix F show this estimator has lower variance than DenseMixer's dense STE.
    - **Design Motivation**: Ablations (Fig. 2) demonstrate that only "Sample (Forward Stochastic) + Marginal (Backward Analytical)" achieves 50.24% EM (OLMoE/GSM), while "Sample + Dense STE" drops to 46.6% with high variance, and "Top-$k$ + Marginal" underperforms ProbMoE. This suggests *forward probabilistic sampling must pair with marginal gradients based on the same distribution*; otherwise, inconsistency leads to performance degradation.

3. **Range-constrained Dynamic-$k$ Routing**:
    - **Function**: Naturally extends exact-$k$ to allow $|S|$ to be chosen freely within $[k_{\min},k_{\max}]$, enabling the router to adaptively allocate expert counts based on token difficulty.
    - **Mechanism**: The conditional distribution is $\mathbb{P}_r(S\mid k_{\min}\le|S|\le k_{\max})=Z^{*-1}\prod_{j\in S}p_j\prod_{j\notin S}(1-p_j)$. Since $Z^*=\sum_{k=k_{\min}}^{k_{\max}} Z_k$, one can first sample $k$ from the cardinality marginal $\mathbb{P}_r(|S|=k\mid\cdot)=Z_k/Z^*$ and then sample the subset using exact-$k$, performing *joint inference of $k$ and $S$*. The backward pass replaces $m_j$ with range-constrained marginals $m_j^*=\partial \log Z^*/\partial \log p_j$ using the same STE routing weight. MAP selection is used at inference.
    - **Design Motivation**: Previous dynamic methods like DA-MoE/DynMoE/AdaMOE rely on heuristics like thresholds or null experts, lacking global normalization and rigorous differentiable training. Range constraints maintain the closure of the probabilistic framework, making dynamic-$k$ an almost "free" extension of exact-$k$. Table 2 shows Dynamic-$k$ on OLMoE/Qwen achieves comparable or higher EM while activating only 75–84% of experts. Fig. 5/6 show the router allocates more experts to rare/ambiguous tokens (e.g., punctuation, suffixes like `ons`, `:`, `?`) and fewer to common numbers/nouns, demonstrating true computation-on-demand.

### Loss & Training
The training objective is $\mathcal{J}(\theta)=\mathbb{E}_{S\sim\mathbb{P}_r(\cdot\mid|S|=k)}[\mathcal{L}(y_S(x;r))]$. ProbMoE uses $\nabla_\theta \mathcal{L}(y(x;r))$ (based on Eq. 7 routing weights) to approximate the expected gradient. Although $p_i=\sigma(r_i)$ and softmax $\pi$ derive from the same router logits, they serve distinct roles in subset sampling and weighting without conflict. Experiments follow the same data/split/evaluation protocol as DenseMixer (Yao et al. 2026), replacing only the routing module for fair comparison. On Qwen, ProbMoE is applied to routed experts, while shared experts remain unchanged.

## Key Experimental Results

### Main Results
Two MoE backbones: OLMoE-1B-7B (16 layers × 64 experts / 8 active) and Qwen1.5-MoE-A2.7B (24 layers × 60 routed + 4 shared / 4 active). Tasks include math reasoning (GSM8K, EM), law understanding, machine translation, summarization (LLM-as-judge), code generation (MBPP), and general knowledge (MMLU).

| Backbone | Method | GSM | Law | Translation | MBPP | Summary | MMLU |
|---|---|---|---|---|---|---|---|
| OLMoE (k=8) | Conventional | 45.94 | 25.00 | 27.56 | 23.20 | 33.70 | 54.04 |
| OLMoE (k=8) | DenseMixer | 47.00 | 27.90 | 30.32 | **24.40** | 37.50 | 53.95 |
| OLMoE (k=8) | **ProbMoE** | **50.19** | **29.00** | **31.63** | 22.80 | **39.29** | 53.69 |
| Qwen (k=4) | Conventional | 53.30 | 29.50 | 30.00 | 32.80 | 39.00 | 61.03 |
| Qwen (k=4) | DenseMixer | **54.97** | 30.75 | 33.75 | 34.00 | 41.00 | 61.03 |
| Qwen (k=4) | SparseMixer | 1.30 | 3.40 | 3.50 | 0.00 | 2.10 | – |
| Qwen (k=4) | ReMoE | 46.30 | 25.50 | 16.99 | 33.00 | 25.80 | – |
| Qwen (k=4) | **ProbMoE** | 53.29 | **34.40** | **39.23** | **35.00** | **44.40** | **61.05** |

ProbMoE ranks 1st in 4 out of 6 tasks on OLMoE (+2.2~+5.5 gains in GSM/Law/Translation/Summary) and 1st in 4 tasks on Qwen (Law +3.65, Translation +5.48, Summary +3.4). It is the only method to consistently outperform DenseMixer without requiring dense expert computation during training.

### Ablation Study

| Config (OLMoE/GSM, 3 seeds) | Forward | Backward | EM (%) | Variance σ |
|---|---|---|---|---|
| **ProbMoE** | Sample (k-subset) | Marginal | **50.24** | **0.09** |
| DenseMixer | Top-$k$ | Dense STE | ~47 | Med |
| Sample + Dense STE | Sample | Dense STE | 46.6 | 0.37 |
| Top-$k$ + Marginal | Top-$k$ | Marginal | < ProbMoE| – |

| Setting | Dataset | $\Delta$EM vs Exact-$k$ | Avg. Expert Usage |
|---|---|---|---|
| Dynamic-$k$ (OLMoE) | GSM | −1.82 | 80.00% |
| Dynamic-$k$ (OLMoE) | Law | −0.04 | 84.50% |
| Dynamic-$k$ (OLMoE) | Translation | +0.36 | 82.00% |
| Dynamic-$k$ (Qwen1.5) | GSM | −4.29 | 75.00% |
| Dynamic-$k$ (Qwen1.5) | Law | **+2.70** | 75.00% |
| Dynamic-$k$ (Qwen1.5) | Translation | **+3.22** | 75.00% |

### Key Findings
- **Forward-Backward Consistency is Essential**: The strongest performance comes from the pairing of "Forward Probabilistic Sampling + Backward Analytical Marginals." Mismatching them (e.g., Sample + Dense STE) drops EM by 4 points and quadruples variance, implying ProbMoE's gains stem from structural consistency rather than stochasticity alone.
- **Enhanced Expert Utilization**: On Qwen/Translation, ProbMoE requires more experts to accumulate 99% probability mass (Fig. 3), exhibiting lower top-4 mass and higher normalized entropy (Fig. 4). This indicates more diverse routing and better expert specialization, consistent with findings that broader expert participation mitigates collapse.
- **Train/Inference Cardinality Mismatch**: Table 3 show that models trained with Exact-$k$ ($k=8$) only select ~5 experts under dynamic-$k$ MAP inference ($k\in[4,8]$), suggesting they learn sharp distributions. ProbMoE, by explicitly modeling cardinality, achieves higher EM (44.50 vs 38.59) under dynamic inference.
- **Semantic Interpretability of Adaptation**: Dynamic-$k$ assigns more experts to punctuation/suffixes/context-sensitive symbols (`:`, `?`, `ons`) and fewer to numbers or concrete nouns (Fig. 6). The overall usage (Law > Translation > GSM) aligns with task complexity (Fig. 5).
- **Failure Cases**: SparseMixer fails on the Qwen backbone (GSM 1.30, MBPP 0.00), suggesting dynamic sparse-gradient routing may be unstable for large models. ReMoE's ReLU-based routing also lags significantly, validating the necessity of an explicit $k$-subset distribution.

## Highlights & Insights
- **"Router gradient failures are a modeling issue, not just an estimation issue"**: ProbMoE shifts the focus from "how to estimate gradients" to "what object to model." By modeling the parameters of a $k$-subset distribution rather than just softmax weights, the gradient inherently captures the utility of alternative subsets.
- **First Implementation of SIMPLE in MoE**: Porting cardinality-constrained Bernoulli DP normalization from combinatorial learning to MoE routers creates a "probabilistic layer + analytical marginal" framework that can be reused in any scenario requiring hard constraints with sparse forward passes (e.g., active learning, sparse attention).
- **Free Dynamic-$k$**: Converting exact-$k$ to dynamic-$k$ is as simple as summing $Z_k$ terms. This theoretical elegance translates directly to implementation simplicity.
- **Transferable Trick**: The STE routing weight (Eq. 7) is a general "Sparse Forward, Distribution-Aware Backward" pattern that is more stable than plain STE and could be applied to Mixture-of-Tokens or Mixture-of-Depths.

## Limitations & Future Work
- **Ours acknowledges**: Current experiments focus on SFT; pre-training at scale has not been verified. System-level gains (kernel acceleration) for dynamic-$k$ are not yet fully realized.
- **MMLU Saturation**: Improvements on MMLU/MMLU-Stem are minimal (< 0.5), suggesting ProbMoE's benefits are primarily for generative/reasoning tasks rather than knowledge retrieval.
- **GSM Dynamic Regression**: EM drops in dynamic-$k$ for GSM tasks (-1.82 for OLMoE, -4.29 for Qwen). Math reasoning may benefit more from fixed, stable expert sets.
- **Hyperparameter Sensitivity**: The $[k_{\min}, k_{\max}]$ range is only briefly discussed in Appendix C; the impact of per-layer tuning requires further study.
- **Computational Constant**: While $\mathcal{O}(Nk)$ is efficient, the DP is serial across tokens unless vectorized. Training wall-clock comparisons were not provided.
- **Future Directions**: Scaling to pre-training; learning $k_{\min}/k_{\max}$ parameters; using ProbMoE as a diagnostic tool for expert pruning/merging.

## Related Work & Insights
- **vs DenseMixer (Yao 2026)**: DenseMixer uses top-$k$ forward but dense STE backward (requiring dense expert computation during training). Ours maintains sparse experts throughout by using a "dense" signal from the subset distribution to the router. Ours outperforms DenseMixer on multi-task OLMoE with lower training cost.
- **vs SparseMixer (Liu 2023) / ReMoE (Wang 2025)**: These methods collapsed on the larger Qwen backbone, highlighting the stability of "discrete selection + distribution-level differentiability."
- **vs Gumbel-Softmax / Concrete**: ProbMoE avoids high variance and cannot-be-exactly-$k$ issues by using exact normalization through DP.
- **vs DA-MoE / DynMoE / AdaMOE**: These dynamic methods lack probabilistic normalization. ProbMoE Dynamic-$k$ maintains a closed-form probabilistic framework for rigorous optimization.
- **vs DeepSeek-MoE (Dai 2024)**: While DeepSeek-MoE improves architecture (shared experts), ProbMoE is an orthogonal training-side improvement verified to be compatible with shared-expert designs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to formalize MoE routing as probabilistic inference over cardinality-constrained subset distributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid multi-task evidence on two backbones, though lacking pre-training scale and wall-clock benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear derivation and intuition. Fig. 1 provides an excellent visual comparison of routing strategies.
- **Value**: ⭐⭐⭐⭐⭐ Provides a theoretically grounded, additive routing component with high utility for both performance and inference efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)
- [\[ICML 2026\] Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)

</div>

<!-- RELATED:END -->
