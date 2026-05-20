---
title: >-
  [Paper Note] Predicting Large Model Test Losses with a Noisy Quadratic System
description: >-
  [ICML 2026][LLM Pretraining][Noisy Quadratic System] This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that models LLM test loss as $L(N, B, K)$ (model size / batch size / update steps)…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Noisy Quadratic System"
  - "Chinchilla"
  - "scaling law"
  - "batch size modeling"
  - "extrapolation prediction"
date: 2026-05-08
content_hash: aef1b8fca4cff313
---

# Predicting Large Model Test Losses with a Noisy Quadratic System

**Conference**: ICML 2026  
**arXiv**: [2605.09154](https://arxiv.org/abs/2605.09154)  
**Code**: Paper promises GitHub release  
**Area**: LLM Pretraining / scaling law / training dynamics  
**Keywords**: Noisy Quadratic System, Chinchilla, scaling law, batch size modeling, extrapolation prediction

## TL;DR
This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that models LLM test loss as $L(N, B, K)$ (model size / batch size / update steps), explicitly modeling batch size in scaling law for the first time. On Pythia + OWT2, it improves extrapolation prediction capability from Chinchilla’s ~20× compute to ~4000× compute.

## Background & Motivation
**Background**: Chinchilla models LLM test loss as a simple power law $L(N, D)$, used to select the optimal $N, D$ ratio under fixed compute $C \approx 6ND$. However, as model scale increases, researchers find it necessary to model more variables (batch size, learning rate, weight decay), and Chinchilla’s functional form is difficult to extend.

**Limitations of Prior Work**: (1) Pure functional fitting methods like Chinchilla fail when extrapolating to >50× compute; (2) Loss-model-free approaches (e.g., Bergsma et al. fitting power law for optimal token budget, $\mu$P making optimal lr scale-invariant) rely on human insight into patterns, have unclear rule interactions, and are too far from loss prediction for rigorous evaluation; (3) There is no principled way to incorporate batch size into loss models.

**Key Challenge**: For accurate loss prediction across multiple pretraining variables ($N, B, K, D, lr, wd, \dots$), pure phenomenological power laws lack mechanistic guidance and suffer from the curse of dimensionality when extended; strict theoretical training dynamics (NQM, linear regression scaling) only provide asymptotic expressions in multiple phases, making them unsuitable as prediction tools.

**Goal**: Construct a loss model that is (1) as lightweight and easy to use as Chinchilla, (2) naturally extensible to multiple pretraining variables, and (3) capable of strong extrapolation prediction under strict train/holdout separation.

**Key Insight**: Unify three classic theoretical lines from training dynamics—linear regression scaling (providing $\mathcal{E}_{\mathrm{appx}}$ / $\mathcal{E}_{\mathrm{bias}}$ terms), Noisy Quadratic Model (NQM, capturing batch size-induced variance), and LayerNorm as dynamic lr adjustment—into a stochastic optimization model. Abandon closed-form asymptotics in favor of numerical computation to directly evaluate the full trajectory.

**Core Idea**: Construct a mechanistic loss model using “projected SGD on quadratic + power-law noise + LayerNorm-equivalent lr scheduling,” expressing LLM test loss as a closed-form numerical integration over 7+1 hyperparameters, upgrading Chinchilla’s “$N, D$ fitting” to “$N, B, K$ full trajectory simulation.”

## Method
The elegance of NQS lies in its balance of “mechanistic but tractable”—retaining the interpretability of quadratic optimization, encoding all asymptotic phases implicitly into numerical parameters, and using LayerNorm-equivalent lr to adapt to small batch sizes.

### Overall Architecture
NQS assumes LLM training is equivalent to running projected SGD on an infinite-dimensional quadratic loss $\mathcal{Q}^{\mathrm{NQS}}(w) = \mathcal{E}_{\mathrm{irr}} + \tfrac{1}{2}\langle w-w^*, H(w-w^*)\rangle$: eigenvectors of $H$ are sorted by eigenvalue, only the first $N$ dimensions are updated (corresponding to trainable model parameters), noise variance is proportional to $1/B$ (mini-batch noise), and $N, B, K$ determine the dynamics. The final closed-form $L_\theta(N, B, K)$ contains 7 hyperparameters (power exponents $p, q, r$ + scale coefficients + learning rate + irreducible error + noise strength), with LayerNorm introducing an 8th parameter $s = \mathbb{E}[\|w^{(0)}\|^2]$.

### Key Designs

1. **Quadratic + Power-Law Spectrum + Mini-Batch Noise (3 Core Assumptions):**

    - **Function**: Assumption 4.1 $\mathbb{E}[\lambda_n (\langle v_n, w^{(0)} - w^*\rangle)^2] = P/n^p$ models the initial bias spectrum; Assumption 4.2 $\lambda_n = Q/n^q$ models the Hessian spectrum; Assumption 4.3 $\xi_n^{(k)} \sim \mathcal{N}(0, R/(n^r B))$ models the mini-batch noise spectrum. Three independent power law parameterizations cover the entire system.
    - **Mechanism**: Simplifies Bordelon et al.’s linear regression scaling model (fixed projection instead of random $P$), generalizes NQM’s batch noise assumption (allowing $r \neq q$). This makes the model compatible with Chinchilla’s $L \sim N^{-(p-1)} + D^{-(p/q - 1/q)}$ asymptotics, while naturally adapting to different phases.
    - **Design Motivation**: Theoretically, mini-batch noise induces multiple asymptotic phases with different functional forms (Paquette 2025). NQS avoids case-by-case derivation, instead using three power law exponents to let the system “interpolate” to the correct phase, avoiding manual segmentation.

2. **Projected SGD on Finite Subspace + Closed-Form Computable Expression:**

    - **Function**: Update rule $w^{(k)} = w^{(k-1)} - \gamma \mathrm{Proj}_{\mathbb{W}_N}(Hw^{(k-1)} - Hw^*) + \gamma \sum_{n=1}^N \xi_n^{(k)} v_n$, updating and injecting noise only in the first $N$ eigen-directions. The expected loss after $K$ steps has a closed-form expression, with the sum over $N$ approximated by integration via the Euler-Maclaurin formula (cost $\mathcal{O}(1)$), and the geometric series over $K$ summed explicitly.
    - **Mechanism**: The “projection to first $N$ dimensions” term directly corresponds to the fact that model parameters are finite—the remaining dimensions are latent and untrained, corresponding to Chinchilla’s $\mathcal{E}_{\mathrm{appx}} \sim P/N^{p-1}$. Integrating over $N$ instead of summing enables near-instant evaluation (<1s), and training the entire $\theta$ takes only ~5 minutes.
    - **Design Motivation**: Theoretical NQM-like models typically provide only asymptotic bounds and cannot be used for direct prediction; NQS, by using “numerical integration instead of explicit formulas,” allows the system to handle arbitrary $N, B, K$ without being limited by asymptotic phase boundaries.

3. **LayerNorm Adjustment: Dynamic Learning Rate $\gamma_k \propto 1/\|w^{(k)}\|^2$**:

    - **Function**: Inspired by van Laarhoven, treats LayerNorm as equivalent to an effective learning rate scheduled by weight norm. Introduces an 8th parameter $s = \mathbb{E}[\|w^{(0)}\|^2]$, and approximates $\|w^{(k)}\|^2 \approx \mathbb{E}[\|w^{(k)}\|^2]$ using $s$, enabling NQS to accurately model small batch training.
    - **Mechanism**: Empirically, vanilla NQS fits large batch well but has large bias for small batch; LayerNorm’s effect is most significant when noise is large (small batch), so it is explicitly modeled. Typical $s$ value is $s = N \times 0.02^2$ (standard init), but the authors also recommend grid search on small batch data subsets.
    - **Design Motivation**: To enable the model to predict loss in the “non-critical batch size” regime and thus support compound resource allocation (e.g., selecting $B$ under time + memory constraints), LayerNorm correction is essential.

### Loss & Training
The process for inferring $\theta = (P, Q, R, p, q, r, \gamma, \mathcal{E}_{\mathrm{irr}})$: (1) Collect training data $\{(N_i, B_i, K_i, l_i)\}$; (2) Fit $\mathcal{L}_\theta = \tfrac{1}{m}\sum_i (\log L_\theta(N_i, B_i, K_i) - \log l_i)^2$; (3) Use a gradient-based optimizer with multi-initialization in parallel to search the loss surface; (4) Use large batch data to determine $\theta$, then grid search for $s$ on small batch data.

## Key Experimental Results

### Main Results
On Pythia + OpenWebText2 + LM1B, the extrapolation prediction capability of Chinchilla method 3 is compared:

| Data | Evaluation Dimension | Compute Gap | Chinchilla Holdout Huber ×10⁻⁵ | NQS Holdout Huber ×10⁻⁵ |
|------|------|------|------|------|
| Pythia + OWT2 | IsoFLOPs | 1024× | 9.0 | **2.5** |
| Pythia + OWT2 | B-K Plane | 1024× | 9.8 | **5.6** |
| Pythia + OWT2 | IsoFLOPs | 64× | 5.6 | **2.6** |
| Llama + LM1B | IsoFLOPs | 6× | 3.7 | **2.9** |
| Llama + LM1B | B-K Plane | 6× | 8.7 | **8.2** |

NQS outperforms Chinchilla on both IsoFLOPs (varying $N$) and B-K Plane (varying $B, K$) holdouts, with the gap widening as extrapolation distance increases.

### Ablation Study
The paper conducts ablations on the necessity of LayerNorm correction, fairness of complexity, and extrapolation robustness:

| Configuration | Key Effect | Notes |
|------|------|------|
| Vanilla NQS (no LN correction) | Fits large batch well | Large bias for small batch training |
| NQS + LN correction ($\gamma \propto 1/\|w\|^2$) | Significant improvement for small batch | Validates necessity of insight 3.3 |
| Chinchilla on train | Huber ~1.0 | Good in-distribution fit |
| Chinchilla on x20 holdout | Still acceptable | Extrapolation limit ~20× |
| Chinchilla on x100+ holdout | Rapid deterioration | Functional form insufficient for extrapolation |
| NQS on x4000 holdout | Still stable | Mechanistic form enables strong extrapolation |

### Key Findings
- NQS has higher train loss than Chinchilla (greater complexity), but significantly lower holdout loss, indicating that mechanistic structure effectively prevents overfitting—complexity arises not from parameter count, but from whether the functional form reflects true dynamics.
- LayerNorm correction is indispensable for small batch training, providing the insight that scaling law research must not ignore the effect of normalization layers on effective learning rate.
- NQS can be directly used for compound resource allocation: overlaying time/memory/data constraints on the IsoFLOPs plane, the $(N, B, K)^*$ selected by NQS is almost always close to the ground truth optimum—this pushes scaling law from “research” to “product” application.
- Extrapolation only begins to fail at a 4000× compute gap, two orders of magnitude higher than Chinchilla’s ~20× limit, which is highly significant for practical pretraining planning—100 PetaFLOP training data can be used to predict the loss of a 400,000 PetaFLOP model.

## Highlights & Insights
- Framing “loss prediction as a better alternative to heuristic-based laws” is crucial: the authors reposition scaling law methodology as “loss model fitting + holdout evaluation,” enabling strict quantitative assessment and avoiding ever-more complex heuristic stacking.
- Using numerical integration instead of asymptotic closed-form is a powerful trick for mechanistic modeling: it retains the structure provided by theoretical derivation but abandons simplifications valid only in the limit, enabling precise prediction in finite practical settings—transferable to other theoretical-empirical gaps beyond scaling.
- The three-parameter power law spectrum ($p, q, r$) allows NQS to implicitly cover multiple asymptotic phases identified in Paquette et al., avoiding case analysis—this “covering phases via parameter space” approach is highly insightful for the field.
- Extension mechanisms similar to LayerNorm correction can handle lr schedule, batch schedule, etc.; the discussion suggests NQS can serve as a “scaling law sandbox” for task-specific optimizer design, with great potential.

## Limitations & Future Work
- The effect of the lr parameter $\gamma_0$ is larger in NQS than in real LLMs, indicating that lr modeling is still imprecise; currently, interactions between lr × batch / lr × model size cannot be predicted.
- $s$ must be grid searched separately rather than jointly optimized with $\theta$, which the authors admit is a numerical hack; ideally, it should be unified.
- The $\theta$ inferred by NQS cannot be directly interpreted as the physical Hessian spectrum or noise strength; they remain fitting parameters—there is still a gap between mechanistic and interpretable.
- Experiments only cover Pythia / Llama families and the standard Adam optimizer; robustness to SGD, AdamW, Adafactor, and other optimizers is unknown.
- 7+1 degrees of freedom are more than Chinchilla’s 5; although there is no overfitting on holdout, more training points are needed for stable fitting, and the paper does not provide a recommended minimum data point count.

## Related Work & Insights
- **vs Chinchilla Method 3 (Hoffmann/Besiroglu)**: Chinchilla is a purely phenomenological power law $L(N, D)$, which collapses after 20× extrapolation; NQS is a mechanistic model $L(N, B, K)$, stable up to 4000× extrapolation, and explicitly models batch size for the first time.
- **vs Noisy Quadratic Model (Zhang et al. 2019)**: NQM only models estimation error (bias + variance), and naively increasing $N$ can increase loss; NQS adds projection to the first $N$ dimensions and $\mathcal{E}_{\mathrm{appx}}$-like terms, correcting this unphysical behavior.
- **vs Linear Regression Scaling (Bordelon et al.)**: Those only provide asymptotic expressions; NQS extends to the finite regime via numerical integration and explicitly incorporates mini-batch noise.
- **vs Bergsma et al. optimal batch fitting**: Those are loss-model-free heuristic rules; NQS provides a unified framework modeling both loss and optimal configuration, and can handle compound resource constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First scaling law to incorporate batch size into a mechanistic loss model, boosting extrapolation capability by two orders of magnitude.
- Experimental Thoroughness: ⭐⭐⭐⭐ Pythia + OWT2 + Llama + LM1B + compound resource cases all covered, with detailed extrapolation curves.
- Writing Quality: ⭐⭐⭐⭐ Flows from Chinchilla’s limitations → three theoretical inspirations → mechanistic construction → ablation in one coherent narrative.
- Value: ⭐⭐⭐⭐⭐ Directly serves industrial-scale pretraining planning, greatly reducing the cost of expensive scaling sweeps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2025\] Language Model Developers Should Report Train-Test Overlap](../../ICML2025/llm_pretraining/language_model_developers_should_report_train-test_overlap.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)

</div>

<!-- RELATED:END -->
