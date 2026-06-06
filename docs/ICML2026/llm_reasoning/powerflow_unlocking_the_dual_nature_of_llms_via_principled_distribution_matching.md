---
title: >-
  [Paper Note] PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching
description: >-
  [ICML 2026][LLM Reasoning][RLIF] This paper reformulates unsupervised LLM fine-tuning as a problem of "matching the $\alpha$-power distribution of the base model." It utilizes the Trajectory-Balance objective of GFlowNet…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "RLIF"
  - "GFlowNet"
  - "$\\alpha$-power distribution"
  - "Length bias"
  - "Creativity"
date: 2026-05-08
content_hash: 14751df0cb7c0de7
---

# PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching

**Conference**: ICML 2026  
**arXiv**: [2603.18363](https://arxiv.org/abs/2603.18363)  
**Code**: https://github.com/Chenruishuo/PowerFlow (Available)  
**Area**: LLM Reasoning / Unsupervised Fine-tuning / Distribution Matching  
**Keywords**: RLIF, GFlowNet, $\alpha$-power distribution, Length bias, Creativity

## TL;DR
This paper reformulates unsupervised LLM fine-tuning as a problem of "matching the $\alpha$-power distribution of the base model." It utilizes the Trajectory-Balance objective of GFlowNet as an amortized sampler and eliminates structural length bias in autoregressive generation through length-aware LA-TB reparameterization. A single knob $\alpha$ controls the direction: $\alpha > 1$ sharpens the distribution to elicit reasoning (comparable to or exceeding supervised GRPO), while $\alpha < 1$ smooths the distribution to release the suppressed creativity of aligned models, simultaneously improving both quality and diversity on the Pareto frontier.

## Background & Motivation
**Background**: Current methods for tapping into LLM potential primarily follow two approaches: RLVR (e.g., DeepSeek-R1, GRPO), which uses verifiable rewards for post-training, and RLIF (e.g., Intuitor, EMPO, TTRL), which uses internal signals (self-consistency, token entropy, majority voting) as intrinsic rewards to elicit reasoning without external labels.

**Limitations of Prior Work**: RLIF rewards are often heuristic combinations lacking a unified theoretical objective. This leads to frequent pathological behaviors during training: length collapse or explosion (reported in Intuitor and majority voting), mode collapse, overconfidence, and majority-voting reward hacking. Researchers often resort to post-hoc "patches" without explaining why rewards should be set in a specific way.

**Key Challenge**: Recent work attributes the gains of RL post-training to "distribution sharpening"—reconcentrating probability mass onto high-quality paths already known to the base model. In essence, RLIF implicitly sharpens the distribution, but existing methods lack a clear "target shape," causing all biases (including length bias) in the rewards to be amplified. Furthermore, for aligned models, excessive sharpening can stifle creativity—another manifestation of typicality bias.

**Goal**: To find a **principled target distribution** where sharpening and smoothing can be governed by a controllable parameter, and to design a training algorithm that directly optimizes this target without being poisoned by length bias.

**Key Insight**: The authors select the $\alpha$-power (escort) distribution as the target: $p_\alpha(y|q) \propto p_{\text{base}}(y|q)^\alpha$. This distribution is classic in statistical mechanics; its key property is **monotonic transformation**, which changes entropy while strictly preserving the relative probability rankings and multimodal structure of the base model. $\alpha > 1$ pushes mass toward high-probability modes (reasoning), and $\alpha < 1$ pushes mass toward the long tail (creativity), corresponding precisely to the "dual nature" of LLMs.

**Core Idea**: Use GFlowNet to amortize the "$\alpha$-power distribution matching" into an on-policy training objective. The standard prompt-level partition function $Z_\phi(q)$ is reparameterized into a token-level $(Z'_\phi(q))^{|y|}$ to keep gradients scale-invariant across trajectories of different lengths, thereby eliminating length bias.

## Method

### Overall Architecture
Input: An unlabeled prompt dataset $\mathcal{D}$, a fixed base model $p_{\text{base}}$, and a user-specified $\alpha$. Output: A fine-tuned policy $\pi_\theta$ whose distribution approximates a length-normalized version of $p_{\text{base}}^\alpha$. The training pipeline consists of three stages: (1) Formulating unsupervised fine-tuning as reverse KL minimization with the $\alpha$-power distribution as the target; (2) Using GFlowNet's Trajectory-Balance (TB) objective as a variational proxy to amortize the intractable partition function into a learnable module $Z_\phi$; (3) Using LA-TB reparameterization to eliminate length bias, adding a format penalty $\psi(y)$ to ensure instruction-following. Inference is performed via standard single-pass decoding without additional overhead, superior to methods like PowerSampling that require MCMC.

### Key Designs

1. **$\alpha$-power Target + Bi-directional Knob**:
    - **Function**: Simultaneously control "eliciting reasoning" and "releasing creativity" using a single scalar $\alpha$.
    - **Mechanism**: The target distribution is defined as $p_\alpha(y|q) = p_{\text{base}}(y|q)^\alpha / Z(q,\alpha)$. Since power transformation is monotonic, mode rankings and multimodal structures are preserved. This is a key difference from RLHF/GRPO methods based on external rewards, which might "drift" out of the base model's support set. For $\alpha > 1$, based on the "verification-generation asymmetry" hypothesis (verification is easier than generation), sharpening pushes mass onto hidden correct paths. For $\alpha < 1$, for models already over-sharpened by RLHF, flattening cancels typicality bias and restores buried long-tail creative paths.
    - **Design Motivation**: To move away from the "heuristic patchwork" of RLIF. Instead of separate rewards for reasoning and creativity, a single framework handles both with theoretical guarantees. The paper also proves (Theorem F.1) that empirical majority-voting RLIF is the limit of $\alpha$-power as $\alpha \to \infty$, positioning PowerFlow as a generalized formulation.

2. **GFlowNet as an Amortized Variational Sampler**:
    - **Function**: Convert "matching an unnormalized target $\tilde{p}_{\text{target}}$" into an RL-style on-policy objective, circumventing the intractability of calculating $Z(q)$.
    - **Mechanism**: Standard reverse KL is $\mathbb{D}_{\text{KL}}(\pi_\theta \| p_{\text{target}}) = \mathbb{E}_{y\sim\pi_\theta}[\log \pi_\theta(y|q) / \tilde{p}_{\text{target}}(y|q)] + \log Z(q)$. Zimmermann et al. (2023) proved that the GFlowNet TB loss is a variational proxy for this KL. Since LLM autoregressive generation is naturally a tree-structured DAG and the backward policy is $P_B \equiv 1$, the TB loss simplifies to $\mathcal{L}_{\text{TB}} = (\log Z_\phi(q) + \sum_t \log \pi_\theta(y_t|y_{<t},q) - \log \tilde{p}_{\text{target}}(y|q))^2$. The gradient of this loss equals $2\nabla_\theta \mathbb{D}_{\text{KL}}(P_F \| p_{\text{target}})$.
    - **Design Motivation**: Unlike the PPO/GRPO paradigm (policy-gradient + KL-penalty), GFlowNet provides an optimization framework that precisely matches any unnormalized density without a reward model. $Z_\phi$ amortizes the partition function estimation, significantly reducing gradient variance.

3. **Length-Aware TB Reparameterization (LA-TB)**:
    - **Function**: Eliminate training instability caused by the near-linear correlation between autoregressive log-probs and sequence lengths (e.g., short sequence collapse during sharpening, repetitive token explosion during smoothing).
    - **Mechanism**: Rewrite the prompt-level partition function in a length-aware form $Z_\phi(q,y) = (Z'_\phi(q))^{|y|}$ and divide the log-mismatch by $|y|$, yielding $\mathcal{L}_{\text{LA-TB}} = (\log Z'_\phi(q) + \tfrac{1}{|y|}\log(\pi_\theta(y|q)/\tilde{p}_{\text{target}}(y|q)))^2$. The convergence point is $\pi^*(y|q) \propto \tilde{p}_{\text{target}}(y|q) \cdot e^{-\lambda_q |y|}$. The paper provides two guarantees: (i) Prop 3.2: LA-TB is an I-projection of $\tilde{p}_{\text{target}}$ under a given expected length constraint; (ii) Prop 3.3: The global KL distortion is an $O(|\lambda_q|^3)$ second-order small quantity.
    - **Design Motivation**: Figure 3 shows that trajectory-level TB/RL leads to immediate length collapse, while token-level averaging is exploited by models using repetitive tokens. LA-TB eliminates length bias without destroying semantic structure—empirical tests on Qwen2.5-Math-1.5B showed a pair-wise inversion rate of only 0.09, meaning 91% of the $\alpha$-power ranking was preserved.

### Loss & Training
The final objective (Equation 10):
$$\mathcal{L}_{\text{PowerFlow}} = w \cdot (\log Z'_\phi(q) + \tfrac{1}{|y|}\log\pi_\theta(y|q) - \alpha[\tfrac{1}{|y|}\log p_{\text{base}}(y|q) + \psi(y)])^2$$
Where $w$ is the detached clipped IS ratio. Reasoning tasks use $\alpha=4$ (base) or $\alpha=2$ (instruct), while creative tasks use $\alpha=0.5$. Training data: 18k NuminaMath-CoT queries for reasoning and 300 prompts for creativity. The recipe follows EMPO for fair comparison.

## Key Experimental Results

### Main Results
Comparison with RLIF baselines and supervised GRPO across various model sizes and benchmarks (avg@16, %):

| Model | Method | MATH500 | AIME25 | AMC23 | Average |
|-------|--------|---------|--------|-------|---------|
| Qwen2.5-1.5B | Intuitor | 47.4 | 0.8 | 22.3 | 18.95 |
| Qwen2.5-1.5B | **PowerFlow** | **49.3** | **1.5** | **23.8** | **19.85** |
| Qwen2.5-1.5B | GRPO (sup) | 45.4 | 0.4 | 21.9 | 18.13 |
| Qwen2.5-Math-1.5B | EMPO | 69.9 | 4.6 | 46.2 | 32.45 |
| Qwen2.5-Math-1.5B | **PowerFlow** | **70.9** | **10.0** | **53.3** | **34.30** |
| Qwen2.5-Math-1.5B | GRPO (sup) | 71.4 | 6.7 | 49.5 | 32.75 |
| Qwen2.5-Math-7B | EMPO | 79.3 | 12.3 | 60.2 | 40.88 |
| Qwen2.5-Math-7B | **PowerFlow** | 78.1 | **14.4** | **63.4** | **42.17** |
| Qwen2.5-Math-7B | GRPO (sup) | 78.4 | 12.9 | 63.4 | 42.38 |

PowerFlow **outperforms supervised GRPO** on Qwen2.5-1.5B, Qwen2.5-Math-1.5B, and Llama-3.2-3B-Instruct (gap > 1σ).

### Ablation Study
Comparison of four length-bias elimination strategies (Figure 3):
- **TB-traj / RL-traj**: Immediate length collapse as the model picks short paths.
- **TB-token / RL-token**: Initial gain followed by collapse; models exploit repetitive tokens.
- **LA-TB / PowerFlow**: Sustainably stable and ascending performance.

Creativity Tasks (Figure 5): PowerFlow ($\alpha=0.5$) is the only method to **simultaneously improve quality and semantic diversity**. High-temp increases diversity but sacrifices quality.

### Key Findings
- **LA-TB is a fundamental solution to length bias**: Trajectory-level matching fails quickly; token-average heuristics are eventually hacked. Only making the partition function length-aware ensures stability.
- **PowerFlow preserves higher path diversity**: On AIME24/25, PowerFlow diversity score (4.05) is higher than GRPO (3.93) and EMPO (3.80). Unsupervised sharpening is less prone to mode collapse than supervised RL.
- **Aligned models prefer lower $\alpha$**: Using $\alpha=2$ for instruct models is better than $\alpha=4$, suggesting that the alignment process has already partially sharpened the distribution.

## Highlights & Insights
- **Unified Theory**: Viewing all RLIF rewards as approximations of the same $\alpha$-power target is elegant. Majority voting is the extreme $\alpha \to \infty$ case, while token entropy/self-consistency are different approximations.
- **Length Bias Engineering**: The insight that length bias stems from near-linear log-probs and can be cured by $(Z'_\phi)^{|y|}$ is powerful. This trick is applicable to any GFlowNet $\times$ Autoregressive generation scenario.
- **Universal Mechanism**: Explains why sharpening helps reasoning and why alignment kills creativity within the same framework. $\alpha$ places these phenomena on a continuous spectrum.
- **Unsupervised vs. Supervised**: Outperforming supervised GRPO signals that RL post-training gains come largely from "distribution reshaping" rather than "knowledge injection."

## Limitations & Future Work
- **Manual $\alpha$**: $\alpha$ values are currently hand-tuned (4 for base, 2 for instruct, 0.5 for creative). Optimal $\alpha$ likely depends on intrinsic entropy and could be scheduled automatically in the future.
- **Target Distortion**: The LA-TB target is technically a length-tilted version of $\alpha$-power. While divergence is an $O(\lambda_q^2)$ small quantity, it might be insufficient for tasks with extreme length disparities.
- **Fairness**: Baselines were not retrained under a unified recipe due to compute constraints.
- **Risk**: $\alpha < 1$ could potentially resurrect unsafe long-tail behaviors removed by RLHF, necessitating safety guardrails.

## Related Work & Insights
- **Comparison with Intuitor / EMPO / TTRL**: These are RLIF methods with heuristic internal rewards. PowerFlow provides a rigorous optimization target.
- **Comparison with PowerSampling**: Uses the same target distribution but relies on MCMC at inference time, which is costly. PowerFlow amortizes the cost into training.
- **Comparison with GRPO**: GRPO requires verifiable rewards; PowerFlow is completely unsupervised and achieves higher diversity.
- **Comparison with Standard GFlowNet**: Standard GFlowNet suffers from length bias in LLMs; LA-TB is a key engineering contribution making GFlowNet viable for LLM training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High. Clean unification of RLIF via $\alpha$-power matching and LA-TB.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Strong coverage across models and benchmarks, though lacks exploration of adaptive $\alpha$.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exceptional. Clear narrative progression from RLIF challenges to the PowerFlow solution.
- **Value**: ⭐⭐⭐⭐⭐ Proves "shape engineering" is competitive with "knowledge injection," offering an easy-to-reuse framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)
- [\[NeurIPS 2025\] Note 1: Is CoT a Hallucination? A Data Distribution Perspective](../../NeurIPS2025/llm_reasoning/is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)
- [\[ICML 2026\] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](are_tools_always_beneficial_learning_to_invoke_tools_adaptively_for_dual-mode_mu.md)
- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)

</div>

<!-- RELATED:END -->
