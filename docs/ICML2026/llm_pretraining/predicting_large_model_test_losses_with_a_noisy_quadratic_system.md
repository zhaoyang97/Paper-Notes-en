---
title: >-
  [Paper Note] Predicting Large Model Test Losses with a Noisy Quadratic System
description: >-
  [ICML 2026][LLM Pretraining][Noisy Quadratic System] This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that frames LLM test loss as $L(N, B, K)$ (model size / batch size / update steps). It is the first to explicitly model batch size within a scaling law, improving extrapolation capabilities on Pythia + OWT2 from Chinchilla's ~20× compute range to ~4000×.
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Noisy Quadratic System"
  - "Chinchilla"
  - "scaling law"
  - "batch size modeling"
  - "extrapolation prediction"
date: 2026-05-08
content_hash: 8f56fe811917e1c3
---

# Predicting Large Model Test Losses with a Noisy Quadratic System

**Conference**: ICML 2026  
**arXiv**: [2605.09154](https://arxiv.org/abs/2605.09154)  
**Code**: GitHub release promised in the paper  
**Area**: LLM Pre-training / Scaling Laws / Training Dynamics  
**Keywords**: Noisy Quadratic System, Chinchilla, scaling law, batch size modeling, extrapolation prediction

## TL;DR
This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that frames LLM test loss as $L(N, B, K)$ (model size / batch size / update steps). It is the first to explicitly model batch size within a scaling law, improving extrapolation capabilities on Pythia + OWT2 from Chinchilla's ~20× compute range to ~4000×.

## Background & Motivation
**Background**: Chinchilla models LLM test loss as a simple power law $L(N, D)$, used to select the optimal ratio of $N$ and $D$ under fixed compute $C \approx 6ND$. However, as model scales increase, researchers find the need to model more variables (batch size, learning rate, weight decay), which Chinchilla's functional form struggles to accommodate.

**Limitations of Prior Work**: (1) Pure functional fitting like Chinchilla fails significantly when extrapolating beyond 50× the compute of the holdout data; (2) Loss-model-free approaches (e.g., Bergsma's power law fitting for optimal token budget, or $\mu$P making optimal lr scale-invariant) rely on human insight into patterns, lack clear interaction between rules, and are too removed from loss prediction for rigorous evaluation; (3) There is no principled way to incorporate batch size into loss models.

**Key Challenge**: To achieve precise loss prediction across multiple pre-training variables ($N, B, K, D, lr, wd, \dots$), purely phenomenological power laws lack mechanistic guidance, leading to a curse of dimensionality during expansion. Conversely, rigorous theoretical training dynamics (NQM, linear regression scaling) only provide asymptotic expressions across multiple phases, making them unsuitable as direct prediction tools.

**Goal**: Construct a loss model that is (1) as lightweight and easy to use as Chinchilla, (2) naturally extendable to multiple pre-training variables, and (3) capable of massive extrapolation prediction under strict train/holdout separation.

**Key Insight**: Unify three classic theoretical threads from training dynamics—linear regression scaling (providing $\mathcal{E}_{\mathrm{appx}}$ / $\mathcal{E}_{\mathrm{bias}}$ terms), the Noisy Quadratic Model (NQM, characterizing variance caused by batch size), and the equivalence of LayerNorm to dynamically adjusted lr—into a single stochastic optimization model. Instead of closed-form asymptotics, numerical computation is used to evaluate the entire trajectory.

**Core Idea**: Build a mechanistic loss model using a "triad": Projected SGD on a quadratic surface + power-law noise + LayerNorm-equivalent lr scheduling. This expresses LLM test loss as a closed-form numerical integral with 7+1 hyperparameters, upgrading Chinchilla's "$N, D$ fitting" to full trajectory simulation of "$N, B, K$".

## Method

### Overall Architecture
NQS conceptualizes LLM training as a physical process: running noisy projected SGD on an infinite-dimensional quadratic loss surface $\mathcal{Q}^{\mathrm{NQS}}(w) = \mathcal{E}_{\mathrm{irr}} + \tfrac{1}{2}\langle w-w^*, H(w-w^*)\rangle$. By sorting the eigenvectors of the Hessian $H$ by eigenvalue magnitude, the model updates only in the first $N$ directions (corresponding to finite trainable parameters), injecting mini-batch noise with variance proportional to $1/B$ each step for $K$ steps. The dynamics are driven by $N, B, K$, resulting in a numerically computable closed-form loss $L_\theta(N, B, K)$. Here, $\theta$ contains power indices $p, q, r$, scale coefficients $P, Q, R$, learning rate $\gamma$, and irreducible error $\mathcal{E}_{\mathrm{irr}}$. Since $\gamma$ can be absorbed by other parameters, the vanilla NQS has **7 degrees of freedom**, plus an 8th parameter $s = \mathbb{E}[\|w^{(0)}\|^2]$ for LayerNorm. Its elegance lies in being "mechanistic but tractable"—retaining interpretable structures while implicitly embedding complex asymptotic phases into numerical parameters.

### Key Designs

**1. Three Power-law Spectra: Enabling a single system to cover multiple asymptotic phases**

The inability of pure functional fitting like Chinchilla to scale stems from a lack of mechanistic structure to describe training events. NQS parameterizes the system using three independent power laws: Assumption 4.1 uses $\mathbb{E}[\lambda_n (\langle v_n, w^{(0)} - w^*\rangle)^2] = P/n^p$ for initial bias distribution; Assumption 4.2 uses $\lambda_n = Q/n^q$ for Hessian spectral decay; Assumption 4.3 uses $\xi_n^{(k)} \sim \mathcal{N}(0, R/(n^r B))$ for mini-batch noise. This simplifies linear-regression scaling models (using fixed projection instead of random $P$) and relaxes NQM's batch noise assumption to $r \neq q$. Crucially, while mini-batch noise theoretically causes training to pass through various asymptotic phases (Paquette 2025), NQS allows $p, q, r$ to "interpolate" to the correct phase automatically, reproducing Chinchilla’s $L \sim N^{-(p-1)} + D^{-(p/q - 1/q)}$ without manual segmentation.

**2. Projected SGD + Euler-Maclaurin Integration: Turning unpredictable theory into second-level predictors**

Training dynamics models like NQM usually provide only asymptotic bounds. NQS breaks through by writing the update rule as $w^{(k)} = w^{(k-1)} - \gamma \mathrm{Proj}_{\mathbb{W}_N}(Hw^{(k-1)} - Hw^*) + \gamma \sum_{n=1}^N \xi_n^{(k)} v_n$. Updating only in the first $N$ directions directly reflects the finite nature of parameters; the remaining unlearned dimensions constitute latent error, providing the $\mathcal{E}_{\mathrm{appx}} \sim P/N^{p-1}$ term found in Chinchilla. The expected loss after $K$ steps has a closed expression: the summation over $K$ is a geometric series, and the summation over $N$ is approximated as an integral via the Euler-Maclaurin formula, reducing cost to $\mathcal{O}(1)$. Evaluating any $(N, B, K)$ takes less than a second, and fitting $\theta$ takes ~5 minutes.

**3. LayerNorm-Equivalent Learning Rate $\gamma_k \propto 1/\|w^{(k)}\|^2$: The final piece for small batch sizes**

Vanilla NQS fits large batches well but shows systematic bias for small batches. To support compound resource allocation (choosing $B$ under time/memory constraints), the model must predict loss in "non-critical batch size" regions. Inspired by van Laarhoven, LayerNorm is equivalent to letting the effective lr vary inversely with weight norm $\gamma_k \propto 1/\|w^{(k)}\|^2$, an effect most pronounced with large noise in small batches. NQS explicitly models this by introducing the 8th parameter $s = \mathbb{E}[\|w^{(0)}\|^2]$ and substituting $s$ using the approximation $\|w^{(k)}\|^2 \approx \mathbb{E}[\|w^{(k)}\|^2]$. This allows NQS to cover small batch regions accurately.

### Loss & Training
Inference of $\theta = (P, Q, R, p, q, r, \gamma, \mathcal{E}_{\mathrm{irr}})$ follows four steps: collect training data $\{(N_i, B_i, K_i, l_i)\}$, fit using a log-space Huber/MSE objective $\mathcal{L}_\theta = \tfrac{1}{m}\sum_i (\log L_\theta(N_i, B_i, K_i) - \log l_i)^2$ with a gradient-based optimizer and multiple initializations, and finally determine $s$ via grid search on small batch data rather than joint optimization due to numerical stability.

## Key Experimental Results

### Main Results
Extrapolation prediction performance compared to Chinchilla method 3 across Pythia + OpenWebText2 + LM1B:

| Data | Evaluation Metric | Compute Gap | Chinchilla Holdout Huber ×10⁻⁵ | NQS Holdout Huber ×10⁻⁵ |
|------|------|------|------|------|
| Pythia + OWT2 | IsoFLOPs | 1024× | 9.0 | **2.5** |
| Pythia + OWT2 | B-K Plane | 1024× | 9.8 | **5.6** |
| Pythia + OWT2 | IsoFLOPs | 64× | 5.6 | **2.6** |
| Llama + LM1B | IsoFLOPs | 6× | 3.7 | **2.9** |
| Llama + LM1B | B-K Plane | 6× | 8.7 | **8.2** |

NQS outperforms Chinchilla on both IsoFLOPs (varying $N$) and B-K Plane (varying $B, K$) holdouts, with the gap widening as extrapolation distance increases.

### Ablation Study
The paper performs three ablations: necessity of LayerNorm correction, complexity fairness, and extrapolation robustness.

| Configuration | Key Effect | Note |
|------|------|------|
| Vanilla NQS (no LN correction) | Good large batch fit | Large bias in small batch training |
| NQS + LN correction ($\gamma \propto 1/\|w\|^2$) | Significant small batch improvement | Validates the necessity of Design 3 |
| Chinchilla on train | Huber ~1.0 | Good in-distribution fitting |
| Chinchilla on x20 holdout | Still acceptable | Extrapolation boundary ~20× |
| Chinchilla on x100+ holdout | Sharp deterioration | Functional form insufficient for extrapolation |
| NQS on x4000 holdout | Remains stable | Mechanistic form provides strong extrapolation |

### Key Findings
- NQS has higher training loss than Chinchilla (higher complexity) but significantly lower holdout loss, proving the mechanistic structure prevents overfitting. Complexity arises from the functional form reflecting real dynamics, not parameter count.
- LayerNorm correction is indispensable for small batch training, suggesting that scaling laws should not ignore the impact of normalization on effective learning rates.
- NQS enables compound resource allocation: by overlaying time/memory/data constraints onto IsoFLOP planes, the selected $(N, B, K)^*$ is near ground truth optimal, moving scaling laws toward production application.
- Extrapolation only begins to collapse at a 4000× compute gap, two orders of magnitude higher than Chinchilla's ~20× limit. This allows 100 PetaFLOP training data to predict 400,000 PetaFLOP model losses.

## Highlights & Insights
- Framing loss prediction as a better alternative to heuristic-based laws is a significant contribution, allowing the field to be rigorously quantified rather than stacking heuristics.
- Using numerical integration instead of asymptotic closed-forms is a powerful mechanistic modeling trick: it retains theoretical structure while discarding unrealistic limits for finite configurations.
- The three-parameter power law spectrum ($p, q, r$) allows NQS to implicitly cover multiple asymptotic phases, providing a unified approach to phase transitions in theoretical scaling studies.
- Extensions similar to LayerNorm correction could handle lr and batch schedules; NQS could serve as a "scaling law sandbox" for task-specific optimizer design.

## Limitations & Future Work
- The influence of the lr parameter $\gamma_0$ is currently larger in NQS than in real LLMs, indicating that lr modeling is not yet precise enough to predict lr × batch size interactions.
- Grid searching $s$ separately from $\theta$ is a numerical hack; a unified optimization would be ideal.
- Inferred $\theta$ values cannot be directly interpreted as physical Hessian spectra or noise intensities; they remain fitting parameters.
- Experiments are limited to Pythia/Llama and standard Adam; robustness to SGD, AdamW, or Adafactor remains unknown.
- The 7+1 degrees of freedom require more data points for stable fitting compared to Chinchilla's 5.

## Related Work & Insights
- **vs Chinchilla Method 3 (Hoffmann/Besiroglu)**: Chinchilla is a phenomenological $L(N, D)$ power law that collapses after 20× extrapolation; NQS is a mechanistic $L(N, B, K)$ model stable up to 4000×.
- **vs Noisy Quadratic Model (Zhang et al. 2019)**: NQM only captures estimation error; NQS adds projection to $N$ dimensions and $\mathcal{E}_{\mathrm{appx}}$ terms to correct unphysical behavior during $N$ increases.
- **vs Linear Regression Scaling (Bordelon et al.)**: Prior work provides only asymptotic expressions; NQS extends this to the finite regime via numerical integration and explicit mini-batch noise.
- **vs Bergsma et al. on optimal batch fitting**: These are loss-model-free heuristics; NQS provides a unified framework characterizing both loss and optimal configuration under compound constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First scaling law to integrate batch size into a mechanistic loss model with 100x better extrapolation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across model families and extrapolation curves.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from Chinchilla limitations to mechanistic construction and validation.
- Value: ⭐⭐⭐⭐⭐ Highly practical for industrial-scale pre-training planning, potentially saving significant sweep costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Language Model Developers Should Report Train-Test Overlap](../../ICML2025/llm_pretraining/language_model_developers_should_report_train-test_overlap.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training](../../ICLR2026/llm_pretraining/revisiting_the_scaling_properties_of_downstream_metrics_in_large_language_model_.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ICLR 2026\] TNT: Improving Chunkwise Training for Test-Time Memorization](../../ICLR2026/llm_pretraining/tnt_improving_chunkwise_training_for_test-time_memorization.md)

</div>

<!-- RELATED:END -->
