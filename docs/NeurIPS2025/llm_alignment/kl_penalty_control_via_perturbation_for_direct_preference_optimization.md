---
title: >-
  [Paper Note] KL Penalty Control via Perturbation for Direct Preference Optimization
description: >-
  [NeurIPS 2025][LLM Alignment][DPO] This paper proposes ε-DPO, which achieves instance-level adaptive KL penalty control by monitoring the monotonicity of logits—used as preference model outputs—under small perturbations…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "DPO"
  - "KL penalty"
  - "preference optimization"
  - "instance-level adaptation"
  - "direct alignment"
  - "RLHF"
date: 2026-05-08
content_hash: cd7033a375f5c1f9
---

# KL Penalty Control via Perturbation for Direct Preference Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2502.13177](https://arxiv.org/abs/2502.13177)  
**Code**: [GitHub](https://github.com/oddqueue/e-dpo)  
**Area**: LLM Alignment
**Keywords**: DPO, KL penalty, preference optimization, instance-level adaptation, direct alignment, RLHF

## TL;DR

This paper proposes ε-DPO, which achieves instance-level adaptive KL penalty control by monitoring the monotonicity of logits—used as preference model outputs—under small perturbations of $\beta$ during training. The method incurs no additional computational overhead and significantly outperforms DPO and most direct alignment algorithms, achieving a 46.4% LC win rate on AlpacaEval 2 (vs. 40.3% for DPO).

## Background & Motivation

DPO reformulates the policy optimization problem in RLHF as preference modeling, eliminating the need to train a separate reward model. However, DPO relies on a critical assumption—that the KL penalty coefficient $\beta$ and the reference policy $\pi_{\text{ref}}$ remain fixed throughout training—which leads to suboptimal outcomes:

**Role of $\beta$**: $\beta$ controls the degree to which the policy deviates from the reference model (analogous to the Lagrange multiplier in a trust-region constraint). Applying a uniform $\beta$ across all instances is theoretically unjustified.

**Limitations of β-DPO**: Although β-DPO claims to adaptively select $\beta$ based on preference pair quality, it operates only at the batch level and depends on mini-batch size.

**Limitations of TR-DPO**: TR-DPO periodically updates the reference policy to prevent over-optimization, but these updates are non-adaptive and may introduce unnecessary KL divergence.

**Core gap**: Instance-level adaptive KL penalty control—requiring neither batch-level statistics nor additional model forward passes—remains unexplored.

## Method

### Overall Architecture

The core mechanism of ε-DPO proceeds as follows:

1. The policy trained by DPO is interpreted as a preference model (binary classifier), where $\beta$ simultaneously serves as an inverse temperature parameter.
2. A small perturbation $\varepsilon$ is applied to the current $\beta$, and the monotonicity of the logit (log-likelihood ratio of chosen vs. rejected responses) is observed.
3. The direction of monotonicity determines whether $\beta$ should be increased or decreased.

### Key Designs

**DPO as a Preference Model**:

The policy learned by DPO can be expressed as a preference probability:

$$\mathbb{P}_{\theta,\beta}(y^w \succ y^l | x) = \sigma\Big(\beta \big(z_\theta(x, y^w, y^l) - \gamma(x, y^w, y^l)\big)\Big)$$

where the logit is:

$$z_\theta(x, y^w, y^l) = \log \frac{\pi_\theta(y^w|x)}{\pi_\theta(y^l|x)}$$

and the adaptive margin is:

$$\gamma(x, y^w, y^l) = \log \frac{\pi_{\text{ref}}(y^w|x)}{\pi_{\text{ref}}(y^l|x)}$$

Here, $\beta$ simultaneously functions as the KL penalty coefficient and the inverse temperature of the preference classifier.

**Perturbation of $\beta$ and Logit Monotonicity**:

The perturbed values of $\beta$ are defined as:

$$\beta_\varepsilon^- = \frac{\beta}{1+\varepsilon}, \quad \beta_\varepsilon^+ = \frac{\beta}{1-\varepsilon}$$

If **logit monotonically increases** when $\beta$ is decreased:

$$z_{\theta(\beta_\varepsilon^-)} > z_{\theta(\beta)} > z_{\theta(\beta_\varepsilon^+)}$$

this indicates that reducing the KL penalty (decreasing $\beta$) improves preference confidence—the current $\beta$ is **over-regularizing**.

If **logit monotonically decreases**:

$$z_{\theta(\beta_\varepsilon^-)} < z_{\theta(\beta)} < z_{\theta(\beta_\varepsilon^+)}$$

this indicates that increasing the KL penalty improves preference confidence—the current $\beta$ is **under-regularizing**.

**Efficient Estimation of Perturbed Policies**:

A key challenge is that $\theta(\beta)$ is not directly accessible (training a separate model for each $\beta$ is infeasible). Drawing on the result of Liu et al. (Proposition 1), the perturbed policies are approximated via a linear combination of the current policy and reference policy logits:

$$\pi_{\theta(\beta_\varepsilon^-)}(y_{1:n}|x) \approx \prod_{i=1}^n \text{Softmax}\big((1+\varepsilon) f_\theta - \varepsilon f_{\text{ref}}\big)_{y_i}$$

$$\pi_{\theta(\beta_\varepsilon^+)}(y_{1:n}|x) \approx \prod_{i=1}^n \text{Softmax}\big((1-\varepsilon) f_\theta + \varepsilon f_{\text{ref}}\big)_{y_i}$$

**Zero Additional Computation**: Since DPO already requires computing $f_\theta$ and $f_{\text{ref}}$, ε-DPO only performs scalar-vector multiplications and additions on existing logits, requiring no additional model forward passes.

### Loss & Training

$$\mathcal{L}_{\text{DPO}}(x, y^w, y^l; \theta, \tilde{\beta}) = -\log \sigma\Big(\tilde{\beta} \big(z_\theta - \gamma\big)\Big)$$

The instance-level $\tilde{\beta}$ is determined by:

$$\tilde{\beta}(x, y^w, y^l; \theta) = \begin{cases} \beta_\varepsilon^- & \text{if logit is monotonically increasing (over-regularized)} \\ \beta_\varepsilon^+ & \text{if logit is monotonically decreasing (under-regularized)} \\ \beta & \text{otherwise} \end{cases}$$

After each update step, the baseline $\beta$ is updated to the mean of $\tilde{\beta}$ values within the current batch.

## Key Experimental Results

### Main Results

**UltraFeedback — Instruct Setting** (SimPO standard setup):

| Method | Mistral-7B AlpacaEval2 LC | Arena-Hard | Llama-3-8B AlpacaEval2 LC | Arena-Hard |
|--------|--------------------------|-----------|--------------------------|-----------|
| SFT | 17.1% | 12.6% | 26.0% | 22.3% |
| DPO | 26.8% | 16.3% | 40.3% | 32.6% |
| IPO | 20.3% | 16.2% | 35.6% | 30.5% |
| KTO | 24.5% | 17.9% | 33.1% | 26.4% |
| SimPO | 32.1% | 21.0% | 44.7% | 33.8% |
| **ε-DPO** | **35.6%** | 17.2% | **46.4%** | **36.7%** |

ε-DPO surpasses all direct alignment algorithms on AlpacaEval 2 LC, including SimPO which modifies the loss function.

**Comparison with Other KL Relaxation Methods** (Llama-3-Instruct):

| Method | AlpacaEval 2 LC | AlpacaEval 2 WR | Arena-Hard |
|--------|----------------|----------------|-----------|
| DPO | 40.3% | 37.9% | 32.6% |
| β-DPO | 43.4% | 38.2% | - |
| TR-DPO_τ | 42.8% | 47.2% | 32.4% |
| TR-DPO_α | 43.5% | 46.8% | 34.7% |
| **ε-DPO** | **46.4%** | 44.9% | **36.7%** |

ε-DPO comprehensively outperforms both β-DPO and TR-DPO.

**Qwen2.5-7B-Instruct (No Hyperparameter Search; Optimal Hyperparameters Transferred Directly from Llama-3)**:

| Method | AlpacaEval 2 LC | Arena-Hard | MT-Bench |
|--------|----------------|-----------|---------|
| DPO | 41.6% | 66.8% | 8.9 |
| SimPO | 32.4% | 60.2% | 8.8 |
| **ε-DPO** | **42.5%** | **67.5%** | **9.1** |

SimPO degrades significantly without hyperparameter search (32.4% vs. DPO's 41.6%), whereas ε-DPO consistently outperforms DPO.

### Ablation Study

**Separate Ablation of $\varepsilon_c$ and $\varepsilon_s$** (Anthropic-HH, $\beta=0.05$):

| εc \ εs | 0.005 | 0.01 | 0.02 |
|---------|-------|------|------|
| 0.005 | 76.4% | 76.7% | 76.4% |
| **0.01** | 78.4% | **79.2%** | 77.4% |
| 0.02 | 74.9% | 74.2% | 74.6% |

$\varepsilon_c$ (neighborhood size for monotonicity checking) has a greater impact on performance; excessively large $\varepsilon$ leads to inaccurate policy estimation and incorrect decisions.

**Computational Overhead Analysis**:

| Metric | Mistral-Instruct | Llama-3-Instruct |
|--------|-----------------|-----------------|
| Additional time per step | 0.0008 sec | 0.0006 sec |
| Additional time per epoch | 0.38 sec | 0.30 sec |
| Relative overhead | 0.0064% | 0.0045% |

The additional computational cost is negligible (approximately $\frac{2v}{N}$, as the vocabulary size is far smaller than the number of parameters).

### Key Findings

1. **ε-DPO and β-DPO assign $\beta$ in opposite directions**: β-DPO assigns higher $\beta$ (conservative updates) to preference pairs with large implicit reward margins, whereas ε-DPO assigns higher $\beta$ to pairs with low confidence (high perplexity). Analysis shows that implicit reward margin does not reliably reflect preference pair quality.

2. **Efficient KL trade-off**: On the Pareto frontier for Anthropic-HH, ε-DPO achieves higher performance under the same KL budget, whereas TR-DPO tends to introduce excessive KL divergence.

3. **Convergence behavior of $\varepsilon$**: Perturbation estimates are unstable in early training (when the current policy is far from optimal); after approximately 0.2 epochs, the upper bound of $\varepsilon$ stabilizes (around 0.008), consistent with the best hyperparameter setting.

4. **Training dynamics**: Larger $\varepsilon$ leads to faster growth in both KL divergence and performance, but with greater instability in early training.

## Highlights & Insights

1. **Elegant theoretical perspective**: Interpreting $\beta$ simultaneously as a KL coefficient and a preference classifier inverse temperature, and using temperature perturbation to detect regularization adequacy, is a natural and intuitive formulation.
2. **Truly instance-level control**: Unlike β-DPO (batch statistics) or TR-DPO (periodic updates), ε-DPO determines $\tilde{\beta}$ independently for each preference pair.
3. **Zero additional cost**: The method reuses existing logits without requiring additional forward passes, making it nearly non-intrusive to existing training pipelines.
4. **Identifying the primary bottleneck of DPO**: Fixed KL penalty is the primary limiting factor of DPO performance, rather than the form of the loss function—a more fundamental diagnosis than modifying the loss (e.g., IPO, SimPO).
5. **Sensitivity to ambiguous samples**: ε-DPO can identify preference pairs with noisy annotations and adjust accordingly, a capability absent in β-DPO.

## Limitations & Future Work

1. **Requires a reference policy**: Like DPO, the method requires storing an additional reference model (mitigable by pre-computing logits), making it less lightweight than reference-free methods such as SimPO and ORPO.
2. **Choice of $\varepsilon$**: Although experiments suggest $\varepsilon \approx 0.01$ is broadly effective, there is no theoretical guarantee of optimality across all settings.
3. **Early training instability**: When the current policy is still far from optimal, perturbation estimates may be inaccurate.
4. **Limited evaluation scope**: Validation is primarily conducted on general chatbot benchmarks; tasks such as code generation and long-form text generation are not evaluated.
5. **Approximation error**: The linear logit combination in Proposition 1 is an approximation of the true perturbed policy; larger $\varepsilon$ may introduce non-negligible bias.

## Related Work & Insights

- **DPO (Rafailov et al., 2024)**: The canonical direct alignment method, with fixed KL penalty as its core limitation.
- **β-DPO (Wu et al.)**: Batch-level $\beta$ control based on implicit reward margin; cannot achieve instance-level adaptation.
- **TR-DPO (Gorbatenko et al.)**: Periodically updates the reference policy; non-adaptive and KL-inefficient.
- **SimPO (Meng et al., 2024)**: Replaces KL penalty with a fixed margin; sensitive to hyperparameters.
- **Liu et al.**: Estimates policy distributions under different $\beta$ values via importance sampling with the reference policy—the theoretical foundation of ε-DPO.
- **Insight**: The remaining headroom for improving DPO-family methods may lie more in regulating training dynamics (e.g., adaptive regularization) than in redesigning the loss function; instance-level control represents a more fundamental direction than batch-level or periodic approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Deriving instance-level KL control from preference model temperature perturbation is a clever and original perspective.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — The theoretical derivation is complete, flowing coherently from the RLHF optimal policy through logit monotonicity to the practical algorithm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on UltraFeedback and Anthropic-HH with multiple baselines and rich training dynamics analysis.
- **Value**: ⭐⭐⭐⭐⭐ — A near-zero-cost improvement that can directly replace standard DPO training.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, though the dense notation requires careful reading.
- **Overall**: ⭐⭐⭐⭐⭐ (9/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)

</div>

<!-- RELATED:END -->
