---
title: >-
  [Paper Note] Swap-guided Preference Learning for Personalized RLHF (SPL)
description: >-
  [ICLR 2026][LLM Alignment][personalized reward model] This paper addresses posterior collapse in Variational Preference Learning (VPL) by proposing SPL…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "personalized reward model"
  - "posterior collapse"
  - "latent variable preference learning"
  - "swap-guided regularization"
  - "preference diversity"
date: 2026-05-08
content_hash: def1df506b818ef2
---

# Swap-guided Preference Learning for Personalized RLHF (SPL)

**Conference**: ICLR 2026  
**arXiv**: [2603.12595](https://arxiv.org/abs/2603.12595)  
**Code**: [https://github.com/cobang0111/SPL](https://github.com/cobang0111/SPL)  
**Area**: LLM Alignment / Personalized Alignment  
**Keywords**: personalized reward model, posterior collapse, latent variable preference learning, swap-guided regularization, preference diversity

## TL;DR
This paper addresses posterior collapse in Variational Preference Learning (VPL) by proposing SPL, which introduces swap-guided base regularization (forcing latent variables to encode user preferences rather than being ignored), a Preferential-IAF decomposition of swap-reversible and swap-invariant signals, and adaptive latent variable modulation. On Llama-3.1-8B, SPL achieves 63.71% accuracy and 97.10% active units, whereas VPL collapses to 57.14% accuracy and 0% active units.

## Background & Motivation

**Background**: Unified reward models assume consistent preferences across all users, yet real-world user preferences exhibit significant diversity. VPL models user-specific preferences via latent variables.

**Limitations of Prior Work**: Under sparse data combined with a strong decoder, VPL's latent variables suffer from posterior collapse — the latent variable is entirely ignored and the model degenerates into a single reward model.

**Core Idea**: Swap-guided regularization forces the latent variables to remain informative, while IAF decomposes user-specific signals.

## Method

### Overall Architecture
SPL extends the Variational Preference Learning (VPL) framework: user preference data $\mathbb{D}_h$ → encoder maps to latent variable $z_0$ → P-IAF transforms it into a richer $z_K$ → reward decoder outputs personalized reward $r_\phi(x,y,z_K)$. The key innovation is leveraging swaps (exchanging chosen/rejected order) to construct virtual opposing users that guide the encoding process.

### Key Designs
1. **Swap-guided Base Regularization**: For each user $h$, a virtual opposing user $h_{swap}$ is constructed by swapping all chosen/rejected preference pairs. The encoder is constrained to satisfy:

    - Mean sign flip: $\mu \approx -\mu_{swap}$ (reversed preference direction → reversed latent direction)
    - Log-variance invariance: $\ell \approx \ell_{swap}$ (uncertainty is unaffected by preference direction)
    - Guidance loss: $\mathcal{L}_{guide} = \mathbb{E}_h[\frac{1}{2}(1+\cos(\mu, \mu_{swap})) + \eta \frac{1}{2}(1-\cos(\ell, \ell_{swap}))]$
2. **Preferential-IAF (P-IAF)**: The IAF context vector is decomposed into a swap-reversible component $c_d = \frac{1}{2}(c - c_{swap})$ and a swap-invariant component $c_s = \frac{1}{2}(c + c_{swap})$. $c_d$ is fed exclusively into the shift function $\mu_k$ (controlling preference direction), while $c_s$ is fed exclusively into the scale function $\sigma_k$ (controlling uncertainty), reducing cross-coupling. After $K$ transformation steps, a multimodal $z_K$ is obtained.
3. **Adaptive Latent Variable Modulation**: A FiLM-style feature modulation mechanism dynamically adjusts the contribution weight of $z_K$ to reward prediction based on signal strength — amplifying strong preference signals and attenuating uncertain ones.

### Loss & Training
$\mathcal{L}(\phi, \psi) = -\text{ELBO} + \lambda \mathcal{L}_{guide}$
- ELBO = expected preference likelihood $- \beta \cdot D_{KL}[q_\psi(z_K|\mathbb{D}_h) || p(z_K)]$
- $D_{KL}$ is computed efficiently via the Jacobian determinant of IAF
- $\mathcal{L}_{guide}$ imposes swap-mirror constraints on the base distribution $z_0$

## Key Experimental Results

### Posterior Collapse Diagnosis

| Model | Method | Accuracy | Active Units↑ | Collapse Status |
|------|------|--------|------------|---------|
| Llama-3.2-3B | VPL | 62.37% | 88.22% | Mild collapse |
| Llama-3.2-3B | **SPL** | **63.28%** | **93.07%** | Healthy |
| Llama-3.1-8B | VPL | 57.14% | **0.00%** | **Complete collapse!** |
| Llama-3.1-8B | **SPL** | **63.71%** | **97.10%** | Healthy |

### Cross-dataset Performance

| Dataset | VPL KL Stability | SPL KL Stability | SPL Accuracy Gain |
|--------|------------|------------|-------------|
| Pets (easy) | No collapse | No collapse | +0.5% |
| UF-P-2 (medium) | Partial collapse | No collapse | +2.1% |
| UF-P-4 (complex) | Complete collapse | No collapse | +6.6% |

### Key Findings
- **VPL completely collapses on the 8B model**: active units drop from 88% to 0%; a stronger decoder bypasses the latent variable entirely
- **SPL eliminates collapse**: active units remain at 97.10% even on the 8B model with complex data
- **Collapse correlates with model capacity**: larger decoders are more prone to bypassing $z$, making swap guidance more necessary
- **SPL is robust to $\beta$**: VPL is highly sensitive to the KL weight $\beta$, while SPL remains stable over a wide range
- **P-IAF decomposition is effective**: ablations show that removing the $c_d/c_s$ decomposition reduces user specialization

## Highlights & Insights
- **First report of posterior collapse in preference learning**: while well-known in the VAE literature, this phenomenon had not been identified in preference modeling
- Swap guidance is an elegant solution to posterior collapse — it leverages the natural symmetry of preference pairs (swapping chosen/rejected) to constrain the latent space structure
- The swap-reversible/invariant decomposition in P-IAF carries clear physical meaning in gradient space — directional preference signals are decoupled from uncertainty signals
- Adaptive modulation avoids overfitting caused by "forcing $z$ to be used" — automatically falling back when the $z$ signal is weak

## Limitations & Future Work
- Evaluation scenarios are limited; the personalization effect within actual RLHF training has not been verified (only preference prediction accuracy is assessed)
- User preference types are defined using predefined categories (helpfulness, honesty, etc.), with no handling of continuous preference spectra
- The selection of P-IAF step count $K$ and $\lambda$ relies on manual tuning, lacking an adaptive mechanism
- No direct comparison with other personalized RLHF methods (e.g., multi-reward model aggregation) is provided

## Related Work & Insights
- **vs VPL (Poddar et al., 2024)**: SPL addresses the core deficiency of VPL (posterior collapse), making personalized reward models viable for larger models
- **vs VAE posterior collapse literature**: collapse in preference learning has a distinctive cause — the decoder already obtains sufficient information from (prompt, response) pairs and has no need for $z$
- **vs multi-reward aggregation**: SPL represents user preferences in a continuous latent space, offering greater flexibility than discrete multi-reward approaches
- **Insight**: The central challenge of personalized alignment is not "how to model diverse preferences" but "how to ensure preference information is encoded into the latent variable"

## Rating
- Novelty: ⭐⭐⭐⭐ — swap-guided regularization and the P-IAF decomposition are conceptually novel
- Experimental Thoroughness: ⭐⭐⭐ — validated across multiple models, but application scenarios are limited
- Writing Quality: ⭐⭐⭐⭐ — the narrative arc from problem analysis (collapse → swap observation → method design) is clear and coherent
- Value: ⭐⭐⭐⭐ — provides a practical, theoretically grounded solution for personalized alignment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

</div>

<!-- RELATED:END -->
