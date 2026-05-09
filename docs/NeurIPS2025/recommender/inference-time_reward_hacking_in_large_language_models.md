---
title: >-
  [Paper Note] Inference-Time Reward Hacking in Large Language Models
description: >-
  [NeurIPS 2025][Recommender Systems][reward hacking] This paper mathematically proves that inference-time alignment methods (e.g., BoN) inevitably exhibit reward hacking (true reward first increases then decreases) when optimizing a proxy reward. It proposes Best-of-Poisson (BoP) sampling to approximate the optimal KL-reward trade-off distribution, and designs the HedgeTune algorithm to locate the optimal inference-time parameter via one-dimensional root-finding, effectively mitigating reward hacking in both mathematical reasoning and human preference settings.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - reward hacking
  - inference-time alignment
  - Best-of-N
  - winner's curse
  - hedging
date: 2026-05-08
content_hash: 94bf686d4a12604c
---

# Inference-Time Reward Hacking in Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.19248](https://arxiv.org/abs/2506.19248)
**Code**: None
**Area**: Recommender Systems
**Keywords**: reward hacking, inference-time alignment, Best-of-N, winner's curse, hedging

## TL;DR
This paper mathematically proves that inference-time alignment methods (e.g., BoN) inevitably exhibit reward hacking (true reward first increases then decreases) when optimizing a proxy reward. It proposes Best-of-Poisson (BoP) sampling to approximate the optimal KL-reward trade-off distribution, and designs the HedgeTune algorithm to locate the optimal inference-time parameter via one-dimensional root-finding, effectively mitigating reward hacking in both mathematical reasoning and human preference settings.

## Background & Motivation
**State of the Field**: The core paradigm of current LLM alignment methods (RLHF, DPO, BoN, etc.) is to maximize a reward function while minimizing KL divergence from a reference model. Among these, Best-of-N (BoN) is widely adopted for its simplicity and efficiency—generating N candidate responses and selecting the one with the highest reward.

**Limitations of Prior Work**: All proxy reward models are imperfect; they cannot precisely capture complex objectives such as correctness, helpfulness, and safety. Optimizing a biased proxy reward leads to reward hacking, where true performance first improves and then degrades.

**Root Cause**: Methods such as BoN are inherently susceptible to the winner's curse—as the number of candidates N increases, the selected response tends to be one whose true quality is overestimated by the proxy reward, resulting in over-optimization.

**Paper Goals**: To characterize the inevitability of inference-time reward hacking and provide practical mitigation mechanisms.

**Starting Point**: Drawing on the winner's curse from information theory and auction theory, the paper reformulates the parameter tuning problem for inference-time alignment as a one-dimensional root-finding problem.

**Core Idea**: Poisson-randomized sampling approximates the optimal tilted distribution via a single parameter $\mu$; HedgeTune then identifies the hacking threshold to achieve optimal "hedging" against the proxy reward.

## Method

### Overall Architecture
The central optimization objective studied in this paper is the standard KL-constrained reward maximization:
$$\pi^{\star} = \arg\max_{\pi} \mathbb{E}_{\pi}[r_p(X)] - \frac{1}{\lambda} D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

The theoretically optimal solution is an exponential tilt of the reference distribution, but direct sampling is intractable in practice (requiring enumeration of all possible continuations). Inference-time approximation methods are therefore necessary.

Pipeline:
1. Sample N candidate responses from the reference model $\pi_{\text{ref}}$
2. Score candidates with a proxy reward model
3. Select one output via a selection mechanism (BoN / SBoN / BoP)
4. Calibrate the selection mechanism's parameter with HedgeTune to avoid over-optimization

### Key Designs
1. **Formal Definition of Reward Hacking (Definition 1)**: Defines the hacking threshold $\theta^{\dagger}$—beyond which the true reward begins to decline. Theorem 1 proves that under TP2 (total positivity of order 2) and the monotone likelihood ratio condition, the true reward function with respect to the inference-time parameter is either monotone or unimodal (having exactly one maximum), thereby establishing the inevitability of reward hacking.

2. **Best-of-Poisson (BoP) Sampling (Algorithm 3)**: The core innovation—replacing the fixed sample count N in BoN with a Poisson-distributed random variable $n' \sim \text{Poisson}(\mu)$, setting $n = n' + 1$ to guarantee at least one sample. The BoP density is:
    $q_{\mu}(x) = (\mu x + 1) e^{\mu(x-1)}, \quad x \in [0,1]$
   Key advantage: BoP approximates the optimal tilted distribution with a single parameter $\mu$, achieving a KL gap of only $O(10^{-4})$ (under the uniform proxy reward assumption). This means BoP can serve as an inference-time approximation to the RLHF optimal policy without requiring model fine-tuning for each $\lambda$.

3. **HedgeTune Algorithm (Algorithm 4)**: Aims to identify the hacking threshold $\theta^{\dagger}$ at which the marginal gain in true reward is zero.

    - For each prompt, map proxy reward scores to empirical quantiles $u \in [0,1]$
    - Construct the residual function $R(\theta) = \mathbb{E}_{u \sim p_\theta}[r_t(u) \cdot \psi(u, \theta)]$
    - Solve $\bar{R}(\theta^{\star}) = 0$ via bisection or Newton's method
    - For BoN: find the optimal N; for SBoN: find the optimal $\lambda$; for BoP: find the optimal $\mu$

### Loss & Training
- HedgeTune does not require access to the LLM's internal distribution; only proxy reward and true reward scoring data are needed
- Requires one-time calibration, applicable to verifiable reward settings (mathematical reasoning, program synthesis) or LLM-as-a-judge scenarios
- The proxy reward model is trained with standard binary cross-entropy loss on preference pairs

## Key Experimental Results

### Main Results I: Verifiable Reward Setting
Using the PPE dataset (responses generated by GPT-4o-mini / Claude Haiku 3), scored by three reward models:

| Dataset | Reward Model | BoN Optimal N | BoP Optimal μ | HedgeTune Peak Recovery |
|---------|-------------|--------------|--------------|------------------------|
| MMLU Pro | InternLM-2 1.8B | ~8 | ~7 | ✓ Successful |
| MATH | Llama-3-Offset-Bias 8B | ~16 | ~14 | ✓ Successful |
| GPQA | Skywork-Llama-3.1 8B | ~32 | ~30 | ✓ Successful |

Key finding: Even with the RewardBench rank-12 Skywork 8B reward model, BoN still exhibits hacking on GPQA (accuracy declines when N is too large). HedgeTune successfully recovers the optimal operating point across all settings.

### Main Results II: Human Preference Setting
Using Pythia 1.4B reference model + AlpacaFarm + AlpacaRM gold-standard reward:

| Proxy RM Training Data Size | Label Noise | BoN Hacking Threshold N† | SBoN Optimal λ† | BoP Hacking Threshold μ† |
|---------------------------|------------|------------------------|----------------|------------------------|
| 10k | 0% | ~16 | ~2.5 | ~14 |
| 20k | 0% | ~64 | ~4.0 | ~60 |
| 46k | 25% | ~8 | ~1.5 | ~7 |
| 80k | 25% | ~32 | ~3.0 | ~28 |

Key finding: Smaller proxy RM training data or higher label noise leads to a lower hacking threshold (earlier degradation). SBoN can achieve peak true reward without hacking through temperature $\lambda$.

### Ablation Study

| Method | # Parameters | Approximates Optimal Distribution | KL Gap | Hacking Mitigation |
|--------|-------------|----------------------------------|--------|-------------------|
| BoN | 1 (N) | No | N/A | Requires HedgeTune |
| SBoN | 2 (N, λ) | No (but more flexible) | N/A | λ=0 falls back to reference |
| BoP | 1 (μ) | Yes (gap < 8×10⁻⁴) | O(10⁻⁴) | Requires HedgeTune |
| Optimal tilted distribution | 1 (λ) | Yes (theoretically optimal) | 0 | Not directly sampleable |

### Key Findings
- BoP achieves nearly identical KL-reward trade-off to the optimal tilted distribution with a single parameter; the KL gap is consistently < 8×10⁻⁴
- The "rise-then-fall" pattern of reward hacking is an intrinsic property of the MLR density family (including BoN and BoP)
- HedgeTune incurs minimal computational overhead (only one-dimensional root-finding) and can directly reuse existing sampling data
- SBoN can fully avoid hacking in certain settings with a fixed $\lambda$ (returning the best achievable reward when the threshold is unreachable)

## Highlights & Insights
- Elegantly connects the winner's curse from auction theory to LLM alignment, offering both theoretical novelty and practical guidance
- The design of BoP is highly elegant: Poisson randomization introduces exponential structure that naturally approximates the optimal tilted distribution
- Theorem 1 is broadly applicable—it holds for any inference-time method satisfying TP2, not limited to BoN
- HedgeTune is practically convenient: it requires no access to LLM internal parameters, only black-box scoring data

## Limitations & Future Work
- HedgeTune requires access to true rewards (or a strong judge), limiting applicability in open-ended tasks where verification is infeasible
- Theoretical analysis relies on the uniform proxy reward assumption (though the loss from CDF transformation is small, discrete cases require additional treatment in the appendix)
- Reward hacking in multi-turn dialogue or sequential decision-making settings is not discussed
- Poisson randomization in BoP introduces variance in sample count, which may affect latency predictability

## Related Work & Insights
- **vs. Gao et al. (2023) scaling law**: Gao et al. empirically observed reward hacking in BoN; this paper provides the first rigorous mathematical proof of its inevitability (Theorem 1)
- **vs. SBoN (Mayrink Verdun et al. 2025)**: SBoN introduces a temperature parameter for flexibility but requires tuning two parameters; BoP achieves comparable performance with a single parameter
- **vs. RLHF/DPO**: The proposed method operates entirely at inference time without model fine-tuning, and can be viewed as a training-free alternative to RLHF
- **vs. Huang et al. (2025) coverage analysis**: They prove that BoN inevitably hacks for large N; this paper provides concrete mitigation strategies

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of the winner's curse perspective, BoP, and HedgeTune demonstrates high innovation with solid theoretical contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both verifiable reward and human preference settings with multiple reward models and datasets, though experiments with larger-scale LLMs are absent
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with rigorous theoretical derivations and informative, aesthetically pleasing figures
- Value: ⭐⭐⭐⭐⭐ Directly relevant to the safe deployment of inference-time alignment methods; BoP and HedgeTune are plug-and-play

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](mmpb_its_time_for_multi-modal_personalization.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)

<!-- RELATED:END -->
