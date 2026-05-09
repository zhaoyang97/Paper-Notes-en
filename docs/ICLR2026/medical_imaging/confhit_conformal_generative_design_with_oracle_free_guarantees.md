---
title: >-
  [Paper Note] ConfHit: Conformal Generative Design with Oracle Free Guarantees
description: >-
  [ICLR 2026][Medical Imaging][conformal prediction] This paper proposes ConfHit, a framework that employs density-ratio-weighted conformal permutation p-values to perform **certification** (determining whether a generated batch contains a hit) and **design** (pruning the candidate set while preserving statistical guarantees). Without requiring an experimental oracle and under distributional shift, ConfHit provides finite-sample $1-\alpha$ coverage guarantees for generative molecular design.
tags:
  - ICLR 2026
  - Medical Imaging
  - conformal prediction
  - generative design
  - drug discovery
  - density ratio
  - statistical guarantee
date: 2026-05-08
content_hash: 80d56357f7858738
---

# ConfHit: Conformal Generative Design with Oracle Free Guarantees

**Conference**: ICLR 2026
**arXiv**: [2603.07371](https://arxiv.org/abs/2603.07371)
**Code**: None
**Area**: AI for Science / Statistical Machine Learning
**Keywords**: conformal prediction, generative design, drug discovery, density ratio, statistical guarantee

## TL;DR

This paper proposes ConfHit, a framework that employs density-ratio-weighted conformal permutation p-values to perform **certification** (determining whether a generated batch contains a hit) and **design** (pruning the candidate set while preserving statistical guarantees). Without requiring an experimental oracle and under distributional shift, ConfHit provides finite-sample $1-\alpha$ coverage guarantees for generative molecular design.

## Background & Motivation

**Background**: Deep generative models (VAEs, diffusion models, autoregressive Transformers) have demonstrated strong performance in molecular discovery. However, practical deployment requires guarantees that generated molecules satisfy target properties—something that can only be verified through costly wet-lab or in vivo experiments. Conformal Prediction (CP) offers a model-agnostic statistical guarantee framework and has recently been extended to generative tasks (Quach et al., 2023; Shahrokhi et al., 2025).

**Limitations of Prior Work**: (a) **Oracle access requirement**—existing CP-based generative methods require experimental validation (synthesis + testing) of newly generated samples, which is prohibitively expensive and impractical in drug discovery; (b) **Distributional shift**—the distribution $Q$ of generated samples may differ from the distribution $P$ of historically labeled data, violating the exchangeability assumption; (c) **Budget constraints**—under a limited generation budget, it is not always possible to guarantee the presence of a valid molecule, and one must honestly declare "insufficient confidence" rather than blindly claiming success.

**Key Challenge**: The fundamental challenge is providing statistical guarantees without validating new samples while simultaneously handling distributional shift—a core difficulty for classical CP frameworks.

**Goal**: Two central problems: (i) **Certification**—given a generated batch, can one guarantee with probability $1-\alpha$ that it contains at least one hit? (ii) **Design**—can the candidate set be pruned to a minimal subset while preserving the guarantee?

**Key Insight**: Exploit weighted exchangeability (corrected via density ratios) between "inactive" samples in the historically labeled data (with known $Y_i$) and generated samples, eliminating the need for an oracle.

**Core Idea**: Density-ratio-weighted permutation p-values + nested testing = oracle-free finite-sample guarantees.

## Method

### Overall Architecture

**Inputs**: Historical labeled data $\mathcal{D}_{\text{calib}}=\{(X_i,Y_i)\}_{i=1}^n$ (where $Y_i \in \{0,1\}$ are known property labels), generated samples $\{X_{n+j}\}_{j=1}^N$, and confidence level $\alpha$.

**ConfHit workflow**: (1) Estimate density ratios $w(x) = dQ/dP(x)$; (2) Construct weighted conformal p-values $p_k$ for each nested subset $\{X_{n+j}\}_{j=1}^k$; (3) Nested testing—find the minimal $\hat{N} = \inf\{k: p_k \leq \alpha\}$, outputting a pruned candidate set or declaring "insufficient confidence."

### Key Designs

1. **Weighted Conformal P-values (Certification Problem)**:
    - **Function**: Quantify the statistical confidence that a generated batch contains a hit.
    - **Mechanism**: Exploit weighted exchangeability between inactive labeled samples $\{X_i: Y_i=0\}$ and generated samples. A randomized p-value is computed over $B$ random permutations:
      $$p_N^{\text{rand}} = \frac{\sum_{b=0}^B \bar{w}(\pi^{(b)};\bm{X}) \mathbb{1}\{V(\pi_0;\bm{X}) \leq V(\pi^{(b)};\bm{X})\}}{\sum_{b=0}^B \bar{w}(\pi^{(b)};\bm{X})}$$
      where $\bar{w}(\pi;\bm{X}) = \prod_{j=1}^k w(X_{\pi(n+j)})$ is the joint likelihood ratio.
    - **Design Motivation**: Classical CP requires exchangeability, which distributional shift violates. Density-ratio weighting restores weighted exchangeability (Tibshirani et al., 2019) and is extended here to the multi-test-sample setting.
    - **Theorem 3.1**: $\Pr(p_N^{\text{rand}} \leq t \mid \max_{j} Y_{n+j}=0) \leq t$, holding in finite samples and model-agnostically.

2. **Nested Testing (Design Problem)**:
    - **Function**: Identify the minimal candidate set $\hat{\mathcal{C}} = \{X_{n+j}\}_{j=1}^{\hat{N}}$ while maintaining the $1-\alpha$ guarantee.
    - **Mechanism**: For each $k=1,\ldots,N$, construct hypothesis $H_k: Y_{n+j}=0, \forall j \leq k$. Monotonize the p-values as $p_k = \max_{k' \geq k} \tilde{p}_{k'}$, and set $\hat{N} = \inf\{k: p_k \leq \alpha\}$.
    - **Theorem 3.4**: The nested hypothesis structure combined with monotone p-values yields $\Pr(\max_{j \leq \hat{N}} Y_{n+j}=0) \leq \alpha$ without requiring multiple testing corrections.
    - **Design Motivation**: The key property of nested hypotheses—$H_k$ true implies $H_\ell$ ($\ell \leq k$) true—makes the stopping rule naturally avoid multiple testing issues.

3. **Robustness Framework for Density Ratio Estimation**:
    - **Function**: Ensure that estimation errors do not invalidate the guarantees.
    - **Theorem 3.5**: Quantifies the inflation of coverage due to estimation error, depending on the weighted error near the critical region of the p-value.
    - Three diagnostic tools: **(1) Balance check**—after weighting, the mean of calibration data should approximate that of generated data; **(2) Synthetic shift validation**—artificially introduce shift within labeled data to verify p-value uniformity; **(3) Sensitivity analysis**—perturb estimated weights to assess conclusion stability.

### Loss & Training

Four choices of scoring function $V$: (i) Max-pooling $V = \max_j \hat{\mu}(x_{n+j})$, (ii) Sum-of-prediction $V = \sum_j \hat{\mu}(x_{n+j})$, (iii) Rank-sum $V = \sum_j R_{n+j}$, (iv) Likelihood ratio $V = \sum_j \log(\hat{\mu}(x_{n+j})/(1-\hat{\mu}(x_{n+j})))$. The choice of scoring function affects test power but not error rate control.

## Key Experimental Results

### Main Results

**Task 1: Constrained Molecular Optimization (CMO-DRD2)**, 2 generative models:

| Model | $\alpha$ | Empirical Error Rate | Avg. Candidate Set Size | Certification Rate |
|-------|---------|---------------------|------------------------|--------------------|
| Hgraph2graph | 0.05 | 0.023 | 3.2 | 89% |
| Hgraph2graph | 0.10 | 0.056 | 2.1 | 94% |
| SELF-EdiT | 0.05 | 0.034 | 2.8 | 91% |
| SELF-EdiT | 0.10 | 0.068 | 1.7 | 96% |

**Task 2: Structure-Based Drug Design (SBDD)**, 3 generative models:

| Model | $\alpha$ | Empirical Error Rate | Avg. Candidate Set Size |
|-------|---------|---------------------|------------------------|
| TargetDiff | 0.10 | ≤0.10 | Significantly < N |
| DecompDiff | 0.10 | ≤0.10 | Significantly < N |
| MolCRAFT | 0.10 | ≤0.10 | Significantly < N |

All models across all $\alpha$ levels consistently satisfy the coverage guarantee (empirical error rate ≤ nominal $\alpha$).

### Ablation Study

| Ablation | Effect |
|----------|--------|
| Remove density ratio correction | Error rate exceeds $\alpha$ (guarantee violated) |
| Different scoring functions | Max-pooling and likelihood ratio yield better power; control holds throughout |
| Reduced calibration data size | P-value variance increases but guarantee remains valid |
| Estimated vs. true density ratio | Guarantee approximately holds when estimation error is controlled (Theorem 3.5) |

### Key Findings

1. **Consistent validity across 5 generative models × 2 tasks**, confirming model-agnosticism.
2. **Significant candidate set reduction**: Substantially smaller than the original $N$ candidates, reducing experimental costs.
3. **Density ratio correction is essential**: Removing it causes the error rate to exceed $\alpha$, invalidating the guarantee.
4. **Honest abstention**: When the generator is weak or the budget is insufficient, ConfHit outputs $\hat{N}=0$ ("insufficient confidence") rather than issuing a spurious guarantee.
5. **Robustness diagnostics are effective**: Balance checks and sensitivity analyses reliably identify the quality of density ratio estimates.

## Highlights & Insights

- **First oracle-free statistical guarantee framework for generative models**: By exploiting the exchangeability structure of historical data, the framework bypasses the oracle requirement and is genuinely applicable to resource-constrained drug discovery settings.
- **Nested hypotheses eliminate multiple testing corrections**: Statistically elegant—the nested structure of the test sequence enables a simple stopping rule to control the overall error rate.
- **Clean separation of certification and design problems**: The transparent problem decomposition retains informational value even when certification fails (indicating an intrinsically difficult task).
- **Balance between theory and practice**: The robustness analysis in Theorem 3.5 combined with three diagnostic tools yields quantifiable reliability for real-world deployment.

## Limitations & Future Work

- Experiments rely solely on computational oracles (DRD2 model, AutoDock Vina) for validation; no real wet-lab experiments are conducted.
- Density ratio estimation remains challenging in high-dimensional molecular spaces, and estimation quality directly affects test power.
- Only single-property guarantees are addressed; simultaneous multi-property guarantees (e.g., activity + selectivity + toxicity) are an important direction for extension.
- The covariate shift assumption $dQ/dP(x,y)=w(x)$ requires that properties are fully determined by structure, which may be overly strong in certain settings.

## Related Work & Insights

- **vs. Quach et al. (2023)**: A classical conformal generative method that requires oracle validation of new samples. ConfHit circumvents this by leveraging label information from historical data.
- **vs. CoDrug (Laghuvarapu et al., 2023)**: Focuses on conformal prediction intervals for property prediction. ConfHit addresses the generative design problem—guaranteeing that a candidate set contains a hit—which constitutes a different problem formulation.
- **vs. Conformal Selection (Jin & Candès, 2023b)**: Controls the false positive rate. ConfHit controls the "no-hit probability"—a distinct error type.
- **Insights**: Conformal inference offers a natural extension from prediction to generation; density ratio estimation and distributional shift handling are the central technical challenges.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Novel problem formulation (oracle-free generative guarantees); strong originality in the theoretical framework (nested testing + multi-test-sample p-values).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation across 5 models × 2 tasks × multiple $\alpha$ levels with thorough robustness diagnostics; real wet-lab experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous and clear theoretical derivations; problem motivation and methodological logic are tightly connected.
- **Value**: ⭐⭐⭐⭐⭐ Directly impacts deployment decisions in generative drug discovery, representing a paradigm shift from "try and see" to "guaranteed."

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/medical_imaging/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[AAAI 2026\] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification](../../AAAI2026/medical_imaging/provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[CVPR 2026\] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](../../CVPR2026/medical_imaging/eda_arbitrary_noise_diffusion_design_space.md)

<!-- RELATED:END -->
