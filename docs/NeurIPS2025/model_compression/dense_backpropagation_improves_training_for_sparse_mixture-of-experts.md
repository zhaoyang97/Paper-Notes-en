---
title: >-
  [Paper Note] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
description: >-
  [NeurIPS 2025][Model Compression][Mixture-of-Experts] This paper proposes Default MoE, a method that maintains exponential moving averages (EMA) of inactive expert outputs as surrogate signals, enabling dense gradient updates for the MoE router without significant computational overhead, thereby improving sparse MoE training performance.
tags:
  - NeurIPS 2025
  - Model Compression
  - Mixture-of-Experts
  - sparse routing
  - dense gradient
  - EMA
  - TopK routing
date: 2026-05-08
content_hash: 808634d91c8ff16c
---

# Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

**Conference**: NeurIPS 2025
**arXiv**: [2504.12463](https://arxiv.org/abs/2504.12463)
**Code**: Available (training code open-sourced)
**Area**: Model Compression
**Keywords**: Mixture-of-Experts, sparse routing, dense gradient, EMA, TopK routing

## TL;DR
This paper proposes Default MoE, a method that maintains exponential moving averages (EMA) of inactive expert outputs as surrogate signals, enabling dense gradient updates for the MoE router without significant computational overhead, thereby improving sparse MoE training performance.

## Background & Motivation

**State of the Field**: Sparse MoE architectures have been widely adopted in models such as DeepSeek-V2/V3, Mixtral, and DBRX. TopK routing activates only $K$ experts per token, enabling large parameter scaling without increasing computational cost.

**Limitations of Prior Work**: TopK routing restricts the router to receiving gradient feedback only from activated experts—non-activated experts contribute zero gradient to the router. This prevents the router from obtaining a "global view" for optimal routing decisions, slowing learning and potentially leading to suboptimal convergence.

**Root Cause**: Providing the router with complete (dense) gradients requires activating all experts (i.e., reverting to a dense model), which negates the sparse computational advantage of MoE. There is a fundamental tension between dense gradients and sparse computation.

**Paper Goals**: Enable approximate dense gradient updates for the router while preserving the efficiency of sparse forward passes.

**Starting Point**: Straight-through estimators can bypass the non-differentiable TopK operation to provide dense gradients, but require the outputs of all experts. The key insight is that the outputs of non-activated experts can be approximated by their historical mean—a quantity already computed for free during standard forward passes.

**Core Idea**: Maintain a "default output vector" per expert via EMA, substituting these vectors for non-activated expert outputs during backpropagation to achieve dense router gradients with $O(1)$ additional memory.

## Method

### Overall Architecture
Standard MoE layer: input token $x$ → router computes $\pi = \text{Softmax}(Wx)$ → TopK selects active set $\mathcal{A}$ → output $y = \sum_{i \in \mathcal{A}} \pi_i E_i(x)$. Default MoE modifies the backward path: non-activated experts are replaced by default vectors $\hat{E}_i$ when computing gradients.

### Key Designs

1. **Dense Gradient via Default Vectors**:

    - **Function**: Ensures the router gradient term $\frac{\partial y}{\partial \pi_i}$ is nonzero for all experts, not only activated ones.
    - Standard TopK gradient: $\frac{\partial y}{\partial \pi_i} = E_i(x)$ (if $i \in \mathcal{A}$) or $0$ (if $i \notin \mathcal{A}$)
    - Default MoE gradient: $\frac{\partial y}{\partial \pi_i} = E_i(x)$ (if $i \in \mathcal{A}$) or $\hat{E}_i$ (if $i \notin \mathcal{A}$)
    - Gradient error analysis: Standard TopK error $\epsilon_{\text{TopK}} \propto \sum_{i' \notin \mathcal{A}} E_{i'}(x)$; Default MoE error $\epsilon_{\text{default}} \propto \sum_{i \notin \mathcal{A}} (E_i(x) - \mathbb{E}[E_i(x)])$, with zero expectation.
    - **Design Motivation**: Default vectors are unbiased estimates of missing expert outputs; zero expected error implies that the dense gradient is corrected perfectly on average.

2. **EMA Update of Default Vectors**:

    - **Function**: Tracks the historical mean output of each expert via exponential moving average.
    - Update rule: $\hat{E}_i^{(t)} = \beta \hat{E}_i^{(t-1)} + (1 - \beta) \overline{E_i(x)}$
    - Here $\overline{E_i(x)}$ is the mean output of expert $i$ over all tokens in the current batch that activated it—these outputs are already computed during standard forward passes at no additional cost.
    - Forward pass formula: $y = \sum_{i=1}^N \pi_i \cdot \begin{cases} E_i(x) & \text{if } i \in \text{TopK}(\pi) \\ \hat{E}_i^{(t)} & \text{otherwise} \end{cases}$
    - **Design Motivation**: EMA tracks changes in expert parameters during training more effectively than a simple running mean. Each expert requires only one additional vector of size `hidden_dim` ($O(1)$ extra memory).

3. **Router-Logit-Weighted EMA Update**:

    - **Function**: Weights the EMA update by the router's softmax probabilities, automatically adapting to different sparsity configurations.
    - **Design Motivation**: The optimal $\beta$ varies across sparsity levels (e.g., 1/8 vs. 1/32). With logit weighting, the choice of $\beta$ becomes insensitive, and all configurations converge to similarly strong performance.

### Loss & Training
- All models use 1.96B total parameters and are trained for 160B tokens.
- FineWeb-Edu and FineWeb datasets with the Llama3 tokenizer.
- Globally reduced load balancing loss is adopted throughout.

## Key Experimental Results

### Main Results (1.96B parameters, 160B tokens)

| Benchmark | TopK (8c1) | Default (8c1) | Gain |
|-----------|-----------|--------------|------|
| BoolQ | 58.5 | **62.0** | +6.1% |
| Lambada | 38.6 | **41.2** | +6.6% |
| SocialIQA | 39.7 | **41.0** | +3.2% |
| ARC | 45.7 | **47.4** | +3.7% |
| HellaSwag | 40.4 | **41.2** | +2.0% |
| **Average** | 46.9 | **47.9** | **+2.1%** |

The Improvement Score (percentage improvement relative to a random baseline) increases by an average of 5.0%.

### Ablation Study

| Dimension | Key Findings |
|---------|---------|
| MoE configuration (8c1/8c2/32c1/32c2/32c4) | Default MoE outperforms TopK across all 5 sparse configurations |
| Learning rate (5e-4 ~ 9e-4) | Default MoE consistently outperforms and tolerates larger learning rates (9e-4) |
| Model scale (557M ~ 7.33B) | Advantage is maintained as model scale increases |
| Convergence speed | Tokens required to reach perplexity 12.18 reduced by 9% |

### Key Findings
- Default MoE improves training stability: it tolerates larger learning rates (9e-4 vs. 7e-4); TopK suffers loss spikes at high learning rates, whereas Default MoE does not.
- Higher sparsity (e.g., 32c1 = 1/32) requires a longer warm-up period for Default MoE, but it ultimately surpasses TopK.
- Default vectors are more diverse in shallow layers (where router uncertainty is high) and converge toward similarity in deeper layers (where the router is confident).
- Throughput is virtually unaffected: only a 0.18% decrease on the 7.33B model; extra memory accounts for 0.03% of MoE parameters.
- Existing routing improvements (SparseMixer, ReMoE, Loss-Free Balancing) all fail to surpass a well-tuned TopK baseline under globally reduced auxiliary loss, while Default MoE succeeds.

## Highlights & Insights
- **Free lunch**: The EMA update reuses expert output means already computed during forward passes, incurring negligible overhead (0.03% memory + 0.18% throughput reduction) while delivering consistent performance gains.
- **Mathematical elegance**: Default vectors are unbiased estimates of missing expert outputs; the theoretical guarantee of zero expected gradient error is particularly clean.
- **Experimental methodology worth noting**: (1) Long training at 160B tokens ensures results do not reflect spurious convergence differences; (2) globally reduced auxiliary loss is used as the baseline, ensuring fair comparison—improvements reported in many prior works vanish under this baseline.

## Limitations & Future Work
- Validation is limited to the ~2B total parameter scale; experiments at 10B+ parameters are absent.
- The EMA mechanism assumes that the distribution of expert outputs changes smoothly—this assumption may not hold in very early training or under abrupt learning rate changes.
- Default vectors are input-agnostic constants; more refined approximations (e.g., input-conditioned default vectors) may yield further improvements.
- Effects on fine-tuning or downstream tasks are not evaluated; only pretraining benchmarks are assessed.
- No direct comparison with DeepSeek's auxiliary-loss-free scheme within the same codebase.

## Related Work & Insights
- **vs. SparseMixer**: SparseMixer estimates true router gradients via linear approximation, but the introduced noise causes it to lag behind TopK early in training, and it ultimately does not surpass Default MoE.
- **vs. ReMoE**: ReMoE attempts to learn continuous routing weights but is highly unstable; it fails to converge in the authors' experimental setting.
- **vs. Loss-Free Balancing (DeepSeek)**: Under globally reduced auxiliary loss, it underperforms a well-tuned TopK baseline.
- The Default MoE approach may serve as a reference for MoE training in DeepSeek-V3—any MoE model using TopK routing can incorporate EMA default vectors at near-zero cost.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of approximating missing expert outputs with EMA is simple and elegant, with clear theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Long 160B-token training, comprehensive ablations (5 MoE configurations, learning rates, model scales), and detailed efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The narrative framing via gradient error analysis is logically coherent and well-illustrated.
- **Value**: ⭐⭐⭐⭐ Practical improvements to MoE training, open-sourced code, and near-zero overhead make it highly adoptable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[NeurIPS 2025\] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](reordering_patches_improves_vision_models.md)
- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)

<!-- RELATED:END -->
