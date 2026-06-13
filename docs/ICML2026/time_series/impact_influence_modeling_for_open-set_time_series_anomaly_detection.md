---
title: >-
  [Paper Note] IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection
description: >-
  [ICML 2026][Time Series][Influence Functions] IMPACT treats the "influence function" as both a searchlight and a scalpel—first training an initial model with a multi-channel deviation loss to calculate the influence scor…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Influence Functions"
  - "Pseudo-Anomaly Generation"
  - "Label Flipping"
  - "Contamination Correction"
  - "Open-Set Time Series Detection"
date: 2026-05-08
content_hash: 86d216976a15148a
---

# IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection

**Conference**: ICML 2026  
**arXiv**: [2603.29183](https://arxiv.org/abs/2603.29183)  
**Code**: https://github.com/mala-lab/IMPACT  
**Area**: Time Series Anomaly Detection / Open-Set Anomaly Detection  
**Keywords**: Influence Functions, Pseudo-Anomaly Generation, Label Flipping, Contamination Correction, Open-Set Time Series Detection  

## TL;DR
IMPACT treats the "influence function" as both a searchlight and a scalpel—first training an initial model with a multi-channel deviation loss to calculate the influence score of each training sample on the validation risk. Under theoretical guarantees of risk reduction, it flips high-influence contaminated unlabeled samples into labeled anomalies and perturbs "boundary normal samples" with minimal risk contribution along the gradient direction to create "unseen pseudo-anomalies." Finally, a dual-head network learns both seen and unseen anomalies, consistently outperforming over ten unsupervised and open-set baselines across 8 real-world time series benchmarks.

## Background & Motivation

**Background**: Time Series Anomaly Detection (TSAD) has long been dominated by unsupervised methods—such as reconstruction, one-class SVM, self-supervised prediction, and diffusion models—assuming a pure normal training set. Recently, Open-Set Anomaly Detection (OSAD) has gained traction, allowing for a small number of labeled seen anomaly classes to identify both "seen + unseen" anomalies. Representative methods include DRA, AHL, DPDL, MOSAD, and InvAD.

**Limitations of Prior Work**: While OSAD works reasonably well for images, it faces hurdles in time series. First is contamination: unknown anomalies are almost certainly mixed into the unlabeled training subset, and existing methods treat them as normal, leading to contaminated supervision signals. Second is pseudo-anomaly generation: common image augmentations like rotation, Cutout, CutPaste, and Mixup break temporal sequentiality—horizontally flipping an ECG segment violates cardiac physiology, and moving averages on short windows fail to eliminate long-term seasonality. Consequently, decision boundaries are biased by both types of noise.

**Key Challenge**: To simultaneously "clean the training set" and "generate representative pseudo-anomalies" without knowing which unlabeled samples are contaminants or what unseen anomalies look like, while provably ensuring both steps reduce test risk rather than introducing new biases.

**Goal**: Decomposition into three sub-problems: (i) designing a loss suitable for multi-channel time series that integrates with influence functions; (ii) automatically identifying contaminated samples and "boundary normal samples with lowest risk contribution" via influence scores; (iii) proving that both "label flipping" and "feature perturbation along influence directions" operations reduce test risk.

**Key Insight**: The authors revisit the influence function of Koh & Liang $\mathcal{I}_L(\bm z_i,\bm z_t)=-\nabla_\theta L(\bm z_t,\hat\theta)^\top H_{\hat\theta}^{-1}\nabla_\theta L(\bm z_i,\hat\theta)$. It not only indicates the marginal contribution of a sample to a prediction but also serves as a "steering wheel" for two modification operations: risk changes for label flipping ($\bm z_i\mapsto\bm z_{i\mathbf 1}$) and feature perturbation ($\bm w_i\mapsto\bm w_i+\zeta_i$) can be expressed in closed form using its second-order derivatives.

**Core Idea**: Use the influence function to drive both "contamination correction" and "pseudo-anomaly generation." Samples with $\mathcal{I}_L(\bm z_i)>0$ are flipped to the anomaly class, while boundary samples with the smallest absolute values in $\mathcal{I}_L(\bm z_i)<0$ are perturbed along the $\nabla_\varphi\nabla_{\theta_h}L$ direction to generate unseen anomalies, unified within a risk reduction framework.

## Method

The IMPACT pipeline involves three stages: Stage I trains an initial model with multi-channel deviation loss and calculates influence scores for each training sample to delineate the contaminated set $\mathcal{D}_{con}$, reference normal set $\mathcal{D}_{ref}$, and remaining clean set $\mathcal{D}_{clean}$. Stage II performs "label flipping + feature perturbation" guided by influence scores to construct the flipped set $\mathcal{D}_{con}'$ and perturbed feature set $\mathcal{W}_{per}'$. Stage III adds an unseen anomaly learning head and performs joint training with $L_{seen}+\lambda L_{unseen}$. Inference uses the maximum cross-channel anomaly score plus the feature deviation from the reference normal centroid.

### Overall Architecture
Input is a time series set $\mathcal{D}=\mathcal{D}_n\cup\mathcal{D}_a$, where each sample $\bm x_i\in\mathbb{R}^{D\times L}$ ($D$ channels, $L$ length). The model consists of two parts: a feature extractor $\bm\varphi_i=\phi(\bm x_i,\theta_\phi)$ (multivariate time series encoder) + an anomaly score head $h(\bm\varphi_i,\theta_h)\in\mathbb{R}^r$ (outputting $r$-channel anomaly scores). The training objective starts with multi-channel deviation loss, followed by influence-based secondary sampling, and ends with an added unseen anomaly head $h'(\cdot,\theta_{h'})$.

### Key Designs

1. **Multi-channel Deviation Loss (Training Pillar of TIS)**:
    - **Function**: Forces the model to push $r$-channel anomaly scores of normal samples toward the mean $\bm\mu_r$ of an isotropic Gaussian prior $\mathcal{N}(\bm\mu,\bm\Sigma)$, while pushing anomaly samples away by at least $a$, measured via Mahalanobis distance $\mathit{dev}(\bm x_i)=\sqrt{(f(\bm x_i,\theta)-\bm\mu_r)^\top\bm\Sigma_r^{-1}(f(\bm x_i,\theta)-\bm\mu_r)}$.
    - **Mechanism**: The loss is defined as $L(\bm z_i,\theta)=\tfrac{1}{r}\sum_{j=1}^r[(1-y_i)\mathit{dev}(\bm x_i)_j+y_i\max(0,a-\mathit{dev}(\bm x_i)_j)]$. Multi-channeling allows "evaluating anomalies from multiple angles"—all angles should align with the Gaussian prior for normal samples, while any single angle deviating beyond $a$ marks an anomaly. Theorem 1 proves this is equivalent to minimizing the entropy of the latent distribution $\mathcal{H}(S)=\tfrac{r}{2}(1+\log(2\pi\sigma^2))\propto\log\sigma^2$.
    - **Design Motivation**: Influence functions require the loss to be twice-differentiable with a reversible Hessian. Traditional deviation loss is single-channel and loses variance information; the multi-channel version expands expressiveness and provides a more stable gradient structure for $H_{\hat\theta}^{-1}\nabla_\theta L$ calculations. The entropy minimization interpretation provides an information-theoretic basis for "compressing normals + pushing anomalies."

2. **Influence-Driven Dual Operations (Core of RADG)**:
    - **Function**: Uses the same influence scores $\mathcal{I}_L(\bm z_i)=\sum_{\bm z_t\in\mathcal{V}}\mathcal{I}_L(\bm z_i,\bm z_t)$ for both contamination correction and unseen anomaly generation.
    - **Mechanism**: (a) Label Flipping—For samples in $\mathcal{D}_{con}=\{\bm z_i\in\mathcal{D}_n\mid\mathcal{I}_L(\bm z_i)>0\}$, it is proven that $\nabla_\theta L(\bm z_{i\mathbf 1},\theta)-\nabla_\theta L(\bm z_i,\theta)=-2\nabla_\theta L(\bm z_i,\theta)$, thus the influence of label flipping is $\mathcal{I}_{L\mathbf 1}(\bm z_i,\bm z_t)=-2\mathcal{I}_L(\bm z_i,\bm z_t)$. Theorem 2 shows the test risk change $\approx -\tfrac{2}{N\cdot|\mathcal{D}_{con}|}\sum_{\bm z_i\in\mathcal{D}_{con}}\mathcal{I}_L(\bm z_i)<0$, implying flipping necessarily reduces risk. (b) Feature Perturbation—For the top-$k$ boundary samples in $\mathcal{D}_{per}=\{\bm z_i\in(\mathcal{D}_n\cap\mathcal{D}_{clean})\mid\mathcal{I}_L(\bm z_i)<0\}$ with the smallest absolute values, perturbation $\bm\varphi_{i\zeta_i}=\bm\varphi_i+\alpha\mathcal{I}_{per}(\bm w_i)^\top$ is performed along direction $\mathcal{I}_{per}(\bm w_i)=-\nabla_{\theta_h}L(\mathcal{V},\hat\theta_h)^\top H_{\hat\theta_h}^{-1}[\nabla_\varphi\nabla_{\theta_h}L(\bm w_i,\hat\theta_h)]$. Theorem 3 proves the perturbed features follow a new distribution with a positive lower-bound distance from the original, and Theorem 4 gives the risk change $\approx -\tfrac{\alpha}{N\cdot|\mathcal{W}_{per}|}\sum\|\mathcal{I}_{per}(\bm w_i)\|_2^2<0$.
    - **Design Motivation**: Traditional contamination correction only deletes samples (e.g., GammaGMM, ExCeeD), wasting signals. Label flipping converts "harmful samples" into "valuable labeled anomalies," yielding double gains. For generation, instead of image-style heuristics, IMPACT moves in the feature space along the "risk maximization" direction, ensuring new samples deviate from known patterns (unseen-ness) while directing toward risk reduction (usefulness)—transforming data augmentation from manual heuristics into provable optimization.

3. **Dual-Head Training + Dual-Component Inference**:
    - **Function**: Separately learns seen and unseen anomaly classes, fusing the maximum cross-channel anomaly score and feature deviation during inference.
    - **Mechanism**: The training loss is $L_{re}=L_{seen}+\lambda L_{unseen}$, where $L_{seen}=\sum_{\bm z_i\in\mathcal{D}_s}L(\bm z_i,\theta)$, with $\mathcal{D}_s=\mathcal{D}_{con}'\cup\mathcal{D}_{ref}\cup\mathcal{D}_{clean}$ covering original labels + flipped anomalies + clean normals. $L_{unseen}=\sum_{\bm z_i\in\mathcal{D}_h}L(\bm z_i,\theta)+\sum_{\bm w_{i\zeta_i\mathbf 1}\in\mathcal{W}_{per}'}L(\bm w_{i\zeta_i\mathbf 1},\theta_{h'})$, where $\mathcal{D}_h$ contains the most helpful normal samples, and perturbed features are fed to the independent unseen head $h'$. Inference score $s=s_m+s_f$, where $s_m=\max_{l<r}(h(\bm\varphi_i,\theta_h)+h'(\bm\varphi_i,\theta_{h'}))_l$ and $s_f=\|\bm\varphi_i-\tfrac{1}{|\mathcal{D}_{ref}|}\sum_{\bm x_j\in\mathcal{D}_{ref}}\bm\varphi_j\|^2$ captures subtle deviations.
    - **Design Motivation**: The independent unseen head $h'$ prevents pseudo-anomaly gradients from contaminating the backbone representation. The max cross-channel score $s_m$ aligns with "multi-angle evaluation"—an anomaly is flagged if any channel lights up. The feature deviation $s_f$ serves as a safety net to detect subtle distribution shifts that $h$ or $h'$ might miss.

### Loss & Training
Two-stage training: first training an initial model $\hat\theta$ with the full set $\mathcal{D}$, then approximating the Hessian inverse using LiSSA-like methods to calculate $\mathcal{I}_L(\bm z_i)$ for set partitioning, and finally retraining with $L_{re}$. Parameters $\alpha$ controls perturbation strength, $\lambda$ balances losses, and $k$ controls the number of flipped and generated samples. All theorems require the loss to be twice-differentiable and the Hessian to be invertible in the vicinity of $\hat\theta$, which the standard multi-channel deviation loss satisfies.

## Key Experimental Results

### Main Results
Compared against 7 unsupervised methods (TCN-AE, THOC, TranAD, DCdetector, GPT4TS, COUTA, DADA) and multiple open-set methods (DevNet, DRA, AHL, DPDL, MOSAD, InvAD, WSAD-DT) on 8 real-world time series benchmarks (UCR, ASD, PSM, SMD, CT, SAD, PTBXL, TUSZ). Metrics: AUC (%) ± standard deviation.

| Setting / Dataset | Ours (IMPACT) | Prev. SOTA | Gain Trend |
|--------|------|----------|------|
| Open-set Avg AUC (8 datasets) | Significantly Highest | DRA / AHL / DPDL etc. | Consistently superior, optimal on most datasets |
| Unsupervised Contrast (UCR / TUSZ) | — | GPT4TS 54.60 / 66.31 | Higher OSAD performance justifies its necessity |
| Contamination Rate (0%–10%) | Most Robust | Most baselines drop sharply | IMPACT remains nearly flat |
| Ratio of Seen Anomaly Classes | Most Robust | Baselines sensitive to seen ratio | Validates unseen anomaly head effectiveness |

> Note: In Table 1 of the original paper, the best results are bolded and second-best underlined. IMPACT achieved bold or underlined results in most columns.

### Ablation Study
| Configuration | Key Metric Change | Description |
|------|---------|------|
| Full IMPACT | Baseline AUC | TIS + RADG + Dual-head |
| w/o Label Flipping (Retain contamination) | Decrease, worse with higher contamination | Validates correction gains from Theorem 2 |
| w/o Feature Perturbation (No unseen samples) | Significant decrease on unseen classes | Validates generation gains from Theorem 4 |
| Using CutAddPaste / COE (manual) instead of perturbation | Decrease | Heuristic augmentation is inferior to influence-guided perturbation |
| Single-channel Deviation Loss ($r=1$) | Decrease, unstable Hessian values | Multi-channel is a prerequisite for influence function stability |
| w/o Unseen Head $h'$ (shared $h$) | Decrease, seen classes slightly harmed | Decoupled heads avoid pseudo-sample backbone contamination |
| w/o Feature Deviation $s_f$ (only $s_m$) | Decrease on subtle anomaly datasets | Necessity of the safety net term |

### Key Findings
- Label flipping + feature perturbation yields a true "1+1>2" effect—each alone outperforms baselines, but the combination reaches the final performance, confirming the components address orthogonal issues (label contamination vs. insufficient expressiveness).
- In robustness tests from 0% to 10% contamination, almost all baselines (including strong OSAD methods) show monotonic AUC decline, while IMPACT remains nearly flat—directly reflecting the realization of Theorem 2.
- As the ratio of seen anomaly classes drops from 25% to 0% (approaching purely unseen), IMPACT shows the smallest decline, indicating that the unseen head effectively learns boundaries orthogonal to seen classes.
- Sensitivity analysis for feature perturbation strength $\alpha$ shows an inverted U-shaped curve; too small is ineffective, while too large pushes pseudo-samples out of the manifold.

## Highlights & Insights
- Upgrading "influence functions" from a "post-hoc diagnostic tool" to a "training steering wheel"—the same $\mathcal{I}_L$ drives both label flipping and feature perturbation. Utilizing second-order information in this dual manner is a rare application in self-supervised/weakly-supervised scenarios.
- Label flipping = converting negative signals into supervisory signals. It essentially replaces active learning queries with statistical influence, an idea extensible to any weakly supervised task where an unlabeled set contains target class samples (e.g., OOD, PU learning).
- Perturbation along $\mathcal{I}_{per}(\bm w_i)$ reverses adversarial perturbation logic: adversarial samples move toward "misclassification," while IMPACT moves toward "risk increment" to simulate unknown distributions—moving provably outside the known distribution without detaching completely from the manifold.
- The triple equivalence of multi-channel deviation loss, isotropic Gaussian priors, and entropy minimization is elegant, unifying geometric (pushing to center), statistical (minimum entropy), and information-theoretic (lowest uncertainty) perspectives.

## Limitations & Future Work
- Calculating Hessian inverses and LiSSA approximations remains heavy for large-scale time series; the paper does not provide runtime for million-scale sequences. Memory-saving second-order approximations like K-FAC or Arnoldi could be explored.
- Inherent convexity assumptions of influence functions near $\hat\theta$ might have large errors on deep Transformer backbones; the impact of these estimation errors on label flipping decisions was not discussed.
- The threshold $\mathcal{I}_L(\bm z_i)>0$ for "contamination" is a hard decision; soft label flipping (weighting by $|\mathcal{I}_L|$) could be introduced for boundary samples.
- Validation is limited to time series classification/segment-level anomalies; reference set $\mathcal{V}$ update strategies need re-designing for long-term point-level or streaming scenarios.

## Related Work & Insights
- **vs. DRA / AHL / DPDL (OSAD)**: These also use limited labels + pseudo-anomalies, but their pseudo-anomalies rely on manual augmentation. IMPACT upgrades this to provable influence-guided generation and adds contamination correction.
- **vs. CutAddPaste / DADA / COE (TS Augmentation)**: These use various temporal transforms that remain heuristic. IMPACT perturbs in the feature space along the risk direction, bypassing the difficulty of preserving semantics in raw time series.
- **vs. GammaGMM / ExCeeD (Contamination Estimation)**: These estimate contamination or calibrate scores only at inference. IMPACT flips labels during training, removing the source of contamination directly.
- **vs. Koh & Liang 2017 (Influence Function Origin)**: The original work uses influence for data valuation/interpretation. IMPACT uses it for closed-form training operations with risk reduction proofs—transforming a tool into a learning objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first OSAD framework to use influence functions for both contamination correction and pseudo-anomaly generation, with four theorems formalizing all training actions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 real datasets + multiple baselines + ablations on contamination/seen ratios/hyperparameters. Lacks a runtime comparison for truly massive long sequences.
- Writing Quality: ⭐⭐⭐⭐ Strong closed-loop of Motivation-Theory-Algorithm-Experiment; each theorem is validated experimentally. Formulas are dense but consistent.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretically grounded and reproducible (open-sourced) new baseline for open-set TS detection. The influence function + risk reduction paradigm is generalizable to PU/weakly-supervised tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection](anomseer_reinforcing_multimodal_llms_to_reason_for_time-series_anomaly_detection.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[ACL 2026\] Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](../../ACL2026/time_series/time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[ICML 2026\] QuITE: Query-based Irregular Time Series Embedding](quite_query-based_irregular_time_series_embedding.md)

</div>

<!-- RELATED:END -->
