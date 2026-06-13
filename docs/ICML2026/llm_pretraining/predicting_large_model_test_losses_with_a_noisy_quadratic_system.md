---
title: >-
  [Paper Note] Predicting Large Model Test Losses with a Noisy Quadratic System
description: >-
  [ICML 2026][LLM Pretraining][Noisy Quadratic System] This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that characterizes LLM test loss as $L(N, B…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Noisy Quadratic System"
  - "Chinchilla"
  - "Scaling Law"
  - "Batch Size Modeling"
  - "Extrapolation Prediction"
date: 2026-05-08
content_hash: cab6a9200c707cb5
---

# Predicting Large Model Test Losses with a Noisy Quadratic System

**Conference**: ICML 2026  
**arXiv**: [2605.09154](https://arxiv.org/abs/2605.09154)  
**Code**: GitHub release promised in the paper  
**Area**: LLM Pre-training / Scaling Law / Training Dynamics  
**Keywords**: Noisy Quadratic System, Chinchilla, Scaling Law, Batch Size Modeling, Extrapolation Prediction

## TL;DR
This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that characterizes LLM test loss as $L(N, B, K)$ (model size / batch size / update steps). For the first time, batch size is explicitly modeled within a scaling law framework, improving extrapolation capability from Chinchilla's ~20× compute to ~4000× compute on Pythia + OWT2.

## Background & Motivation
**Background**: Chinchilla models LLM test loss as a simple power law $L(N, D)$ to select the optimal $N, D$ ratio under a fixed compute budget $C \approx 6ND$. However, as model scales increase, researchers find it necessary to model more variables (batch size, learning rate, weight decay), which the functional form of Chinchilla fails to extend efficiently.

**Limitations of Prior Work**: (1) Pure functional fitting like Chinchilla fails significantly when extrapolating more than 50× beyond the holdout compute. (2) Loss-model-free approaches (e.g., Bergsma et al. fitting power laws for optimal token budgets, or $\mu$P paths making optimal lr scale-invariant) rely on human insight for patterns; interactions between rules are unclear, and they are too disconnected from loss prediction for rigorous evaluation. (3) There is no principled way to incorporate batch size into a loss model.

**Key Challenge**: To perform precise loss prediction across multiple pre-training variables ($N, B, K, D, lr, wd, \dots$), pure phenomenological power laws lack mechanistic guidance and suffer from the curse of dimensionality during expansion. Conversely, rigorous theoretical training dynamics (NQM, linear regression scaling) only offer asymptotic expressions across multiple phases, making them unsuitable for direct use as prediction tools.

**Goal**: To construct a loss model that is (1) as lightweight and easy to use as Chinchilla, (2) naturally extensible to multiple pre-training variables, and (3) capable of significant extrapolation under strict train/holdout separation.

**Key Insight**: Three classic theoretical threads in training dynamics—linear regression scaling (providing $\mathcal{E}_{\mathrm{appx}}$ / $\mathcal{E}_{\mathrm{bias}}$ terms), Noisy Quadratic Model (NQM, characterizing variance terms caused by batch size), and the equivalence of LayerNorm to dynamic lr adjustment—are unified into a stochastic optimization model. The model abandons closed-form asymptotics in favor of numerical computation to evaluate the entire trajectory directly.

**Core Idea**: A mechanistic loss model is constructed using a "trinity" of projected SGD on quadratics + power-law noise + LayerNorm-equivalent lr scheduling. This expresses LLM test loss as a closed-form numerical integral with 7+1 hyperparameters, upgrading Chinchilla’s "$N, D$ fitting" to an "$N, B, K$ full-trajectory simulation."

## Method
The elegance of NQS lies in its "mechanistic but tractable" balance—retaining the interpretability of quadratic optimization while implicitly encoding all asymptotic phases into numerical parameters and adapting to small batches via LayerNorm-equivalent lr.

### Overall Architecture
NQS assumes LLM training is equivalent to running projected SGD on an infinite-dimensional quadratic loss $\mathcal{Q}^{\mathrm{NQS}}(w) = \mathcal{E}_{\mathrm{irr}} + \tfrac{1}{2}\langle w-w^*, H(w-w^*)\rangle$. The eigenvectors of $H$ are sorted by eigenvalue in descending order, and updates occur only in the first $N$ dimensions (corresponding to trainable parameters). Noise variance is proportional to $1/B$ (mini-batch noise), while $N, B, K$ determine the dynamics. The final closed-form expression $L_\theta(N, B, K)$ includes 7 hyperparameters (power indices $p, q, r$ + scale coefficients + learning rate + irreducible error + noise strength). LayerNorm introduces an 8th parameter $s = \mathbb{E}[\|w^{(0)}\|^2]$.

### Key Designs

1.  **Quadratic + Power-Law Spectrum + Mini-Batch Noise (3 Core Assumptions)**:
    - **Function**: Assumption 4.1 $\mathbb{E}[\lambda_n (\langle v_n, w^{(0)} - w^*\rangle)^2] = P/n^p$ characterizes the initial bias spectrum; Assumption 4.2 $\lambda_n = Q/n^q$ characterizes the Hessian spectrum; Assumption 4.3 $\xi_n^{(k)} \sim \mathcal{N}(0, R/(n^r B))$ characterizes the mini-batch noise spectrum. These three sets of independent power laws parameterize the system.
    - **Mechanism**: Simplifies the linear regression scaling model of Bordelon et al. (replacing random $P$ with fixed projection) and generalizes NQM's batch noise assumption (allowing $r \neq q$). This keeps the model compatible with the Chinchilla asymptotic form $L \sim N^{-(p-1)} + D^{-(p/q - 1/q)}$ while naturally adapting to different phases.
    - **Design Motivation**: Theoretically, mini-batch noise generates multiple asymptotic phases with different functional forms (Paquette 2025). Instead of case-by-case derivation, NQS uses three power-law indices to let the system "interpolate" to the correct phase, avoiding manual segmentation.

2.  **Projected SGD on finite subspace + Closed computable expression**:
    - **Function**: Update rule $w^{(k)} = w^{(k-1)} - \gamma \mathrm{Proj}_{\mathbb{W}_N}(Hw^{(k-1)} - Hw^*) + \gamma \sum_{n=1}^N \xi_n^{(k)} v_n$, updating and injecting noise only in the first $N$ eigen-directions. The expected loss after $K$ steps has a closed form, where the summation over $N$ is approximated as an integral via the Euler-Maclaurin formula (cost $\mathcal{O}(1)$), and the geometric series over $K$ is summed explicitly.
    - **Mechanism**: The term "projected to the first $N$ dimensions" directly corresponds to the fact that model parameters are finite—remaining dimensions are latent and untrained, corresponding to $\mathcal{E}_{\mathrm{appx}} \sim P/N^{p-1}$ in Chinchilla. Evaluating via integration rather than summation makes evaluation near-instantaneous (< 1 second), and training the entire $\theta$ takes only ~5 minutes.
    - **Design Motivation**: Theoretical NQM-type models typically provide asymptotic bounds rather than direct predictions. NQS uses "numerical integration instead of explicit formulas" to allow the system to handle arbitrary $N, B, K$ without being restricted by asymptotic phase boundaries.

3.  **LayerNorm Adjustment: Dynamic learning rate $\gamma_k \propto 1/\|w^{(k)}\|^2$**:
    - **Function**: Inspired by van Laarhoven, LayerNorm is treated as an effective learning rate schedule that varies with weight norm. An 8th parameter $s = \mathbb{E}[\|w^{(0)}\|^2]$ is introduced, and $\|w^{(k)}\|^2 \approx \mathbb{E}[\|w^{(k)}\|^2]$ is approximated using $s$, allowing NQS to correctly characterize small-batch training.
    - **Mechanism**: Empirical findings show vanilla NQS fits large batches well but deviates for small batches. LayerNorm significantly impacts training when noise is high (small batches), so it is explicitly modeled. While $s = N \times 0.02^2$ is a standard initialization, the authors suggest grid search on a small batch subset.
    - **Design Motivation**: To predict loss within "non-critical batch size" ranges for compound resource allocation (e.g., selecting $B$ under time + memory constraints), LayerNorm correction is essential.

### Loss & Training
The process for inferring $\theta = (P, Q, R, p, q, r, \gamma, \mathcal{E}_{\mathrm{irr}})$: (1) Collect training data $\{(N_i, B_i, K_i, l_i)\}$; (2) Fit $\mathcal{L}_\theta = \tfrac{1}{m}\sum_i (\log L_\theta(N_i, B_i, K_i) - \log l_i)^2$; (3) Search the loss surface using a gradient-based optimizer with multiple parallel initializations; (4) Use large-batch data to fix $\theta$, then grid search $s$ using small-batch data.

## Key Experimental Results

### Main Results
Extrapolation prediction capability compared to Chinchilla method 3 across Pythia + OpenWebText2 + LM1B:

| Data | Evaluation Dimension | Compute Gap | Chinchilla Holdout Huber ×10⁻⁵ | NQS Holdout Huber ×10⁻⁵ |
|------|------|------|------|------|
| Pythia + OWT2 | IsoFLOPs | 1024× | 9.0 | **2.5** |
| Pythia + OWT2 | B-K Plane | 1024× | 9.8 | **5.6** |
| Pythia + OWT2 | IsoFLOPs | 64× | 5.6 | **2.6** |
| Llama + LM1B | IsoFLOPs | 6× | 3.7 | **2.9** |
| Llama + LM1B | B-K Plane | 6× | 8.7 | **8.2** |

NQS outperforms Chinchilla across both IsoFLOPs (varying $N$) and B-K Plane (varying $B, K$) holdouts, with the gap widening as the extrapolation distance increases.

### Ablation Study
The paper conducts three ablations on LayerNorm correction necessity, complexity fairness, and extrapolation robustness:

| Configuration | Key Effect | Description |
|------|------|------|
| Vanilla NQS (no LN correction) | Good large-batch fit | Diverges for small-batch training |
| NQS + LN correction ($\gamma \propto 1/\|w\|^2$) | Significant small-batch improvement | Validates necessity of inspiration 3.3 |
| Chinchilla on train | Huber ~1.0 | Excellent in-distribution fit |
| Chinchilla on x20 holdout | Still acceptable | Extrapolation bound ~20× |
| Chinchilla on x100+ holdout | Deteriorates sharply | Functional form insufficient for extrapolation |
| NQS on x4000 holdout | Remains stable | Mechanistic form offers strong extrapolation |

### Key Findings
- NQS shows higher training loss than Chinchilla (due to complexity) but significantly lower holdout loss, indicating that the mechanistic structure effectively prevents overfitting. Complexity arises not from the number of parameters, but from whether the functional form reflects real dynamics.
- LayerNorm correction is indispensable for small-batch training, suggesting that scaling laws should not ignore the impact of normalization layers on effective learning rates.
- NQS can be used directly for compound resource allocation: by overlaying time / memory / data constraints on the IsoFLOPs plane, $(N, B, K)^*$ selected by NQS is consistently close to the ground truth optimal, moving scaling laws from research to practical product application.
- Breakdowns only begin to occur at a 4000× compute gap, two orders of magnitude higher than Chinchilla’s ~20× limit. This is significant for pre-training planning: loss for a 400,000 PetaFLOP model can be predicted using only 100 PetaFLOP of training data.

## Highlights & Insights
- The framing of "loss prediction as a better alternative to heuristic-based laws" is crucial: the authors reposition scaling law methodology as "loss model fitting + holdout evaluation," allowing the field to be rigorously quantified and avoiding the accumulation of increasingly complex heuristics.
- Using numerical integration instead of asymptotic closed-forms is a powerful trick for mechanistic modeling: it retains the structure provided by theoretical derivations while abandoning simplifications that only hold in the limit, allowing for precise predictions in finite real-world configurations.
- The three-parameterization of the power-law spectrum ($p, q, r$) allows NQS to implicitly cover multiple asymptotic phases identified in theories like Paquette's, avoiding case-by-case analysis. This "using parameter space to cover phases" approach is highly insightful for the community.
- Expansion mechanisms similar to LayerNorm correction could handle lr schedules and batch schedules; discussion in the paper suggests NQS could serve as a "scaling law sandbox" for designing task-specific optimizers.

## Limitations & Future Work
- The impact of the lr parameter $\gamma_0$ in NQS is currently larger than in real LLMs, indicating that lr modeling is not yet precise enough to predict lr × batch / lr × model size interactions accurately.
- $s$ must be grid-searched independently rather than jointly optimized with $\theta$, which the authors admit is a numerical hack; ideally, it should be unified.
- Parameters $\theta$ inferred by NQS cannot be directly interpreted as physical Hessian spectra or noise intensities; they remain fitting parameters—a gap still exists between mechanistic and interpretable.
- Experiments only cover Pythia / Llama families + standard Adam; robustness to other optimizers like SGD, AdamW, or Adafactor is unknown.
- With 7+1 degrees of freedom compared to Chinchilla's 5, NQS does not overfit holdouts but requires more training points for stable fitting; the paper does not recommend a minimum number of data points.

## Related Work & Insights
- **vs Chinchilla Method 3 (Hoffmann/Besiroglu)**: Chinchilla is a pure phenomenological power law for $L(N, D)$ that collapses after 20× extrapolation; NQS is a mechanistic model for $L(N, B, K)$ that remains stable at 4000× and explicitly models batch size for the first time.
- **vs Noisy Quadratic Model (Zhang et al. 2019)**: NQM only characterizes estimation error (bias + variance), where increasing $N$ naively increases loss; NQS corrects this unphysical behavior by adding projection to the first $N$ dimensions and $\mathcal{E}_{\mathrm{appx}}$ terms.
- **vs Linear Regression Scaling (Bordelon et al.)**: That line of work provides only asymptotic expressions; NQS extends to the finite regime via numerical integration and explicitly adds mini-batch noise.
- **vs Optimal Batch Fitting (Bergsma et al.)**: Those are loss-model-free heuristic regularities; NQS provides a unified framework characterizing both loss and optimal configuration while handling compound resource constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First scaling law to incorporate batch size into a mechanistic loss model, pushing extrapolation capability by two orders of magnitude.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered Pythia + OWT2 + Llama + LM1B + compound resource cases with detailed extrapolation curves.
- Writing Quality: ⭐⭐⭐⭐ Seamless progression from Chinchilla pain points to theoretical inspirations, mechanistic construction, and ablation.
- Value: ⭐⭐⭐⭐⭐ Directly serves industrial-grade pre-training planning, potentially reducing the costs of expensive scaling sweeps significantly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)
- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)

</div>

<!-- RELATED:END -->
