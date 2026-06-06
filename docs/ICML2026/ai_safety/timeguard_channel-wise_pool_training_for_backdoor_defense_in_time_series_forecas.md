---
title: >-
  [Paper Note] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting
description: >-
  [ICML 2026][AI Safety][Time Series Forecasting] TimeGuard reframes backdoor defense in multivariate time series forecasting (TSF) from "discarding the entire window" to a reliable "channel-wise + time-step" pool training…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Time Series Forecasting"
  - "Backdoor Attacks"
  - "Training-phase Defense"
  - "Channel-wise Training"
  - "Trustworthy Machine Learning"
date: 2026-05-08
content_hash: 0df614adb151f882
---

# TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2605.22365](https://arxiv.org/abs/2605.22365)  
**Code**: https://github.com/qducnguyen/TimeGuard  
**Area**: AI Security / Backdoor Defense / Time Series Forecasting  
**Keywords**: Time Series Forecasting, Backdoor Attacks, Training-phase Defense, Channel-wise Training, Trustworthy Machine Learning

## TL;DR
TimeGuard reframes backdoor defense in multivariate time series forecasting (TSF) from "discarding the entire window" to a reliable "channel-wise + time-step" pool training mechanism. It initializes a high-purity pool using the intersection of Reverse-Consistency Filtering (RCF) and Neighborhood Diversity Filtering (NDF), then progressively expands it via Distance-Regularized Loss Selection (DRLS). Without relying on any clean data, it improves $\text{MAE}_{\text{P}}$ against SOTA attacks like BackTime to 1.96 times that of the strongest baseline, PDB.

## Background & Motivation

**Background**: Time Series Forecasting (TSF) is widely deployed in critical scenarios such as transportation, meteorology, and power grids. However, attacks like BackTime (Lin 2024) have demonstrated that TSF models are vulnerable to backdoor injections—attackers only need to pollute **a small subset of channels and a short time window** in the training sequences to hijack predictions at inference time using trigger patterns. Corresponding defenses are nearly non-existent: mature backdoor defenses primarily target classification (Spectral, ABL, Fine-pruning, NAD, STRIP, etc.) and have never been systematically evaluated for TSF.

**Limitations of Prior Work**: The authors first systematically benchmarked 13 representative classification-side defenses on TSF across four stages: pre-training, in-training, post-training, and inference. The results were poor: sample-level filtering (Spectral / TED / TED++) had FDER hovering around 0.5, indicating failure; pure loss-separation methods (ABL / ESTI) averaged 0.497 FDER, as poisoned sample losses became indistinguishable from clean ones within a few epochs; and inference-time detection (STRIP / TeCo / IBD-PSC) showed AUROC near 0.55 while increasing latency from 2 seconds to 200+ seconds. The only "marginal" methods, Fine-pruning, NAD, and PDB, require a trusted clean subset, which is expensive in TSF scenarios.

**Key Challenge**: The failure is attributed to two TSF-specific properties. (1) **Data entanglement**: Multivariate sequences have both channel structures and temporal dependencies. Attacks often affect only a few channels, but classification defenses make "whole-window" decisions, leading to *channel-level signal dilution* as clean channels dilute the signal. (2) **Task-formulation shift**: TSF involves continuous-value regression on overlapping windows. Poisoned windows can quickly minimize loss by fitting, leading to *training-loss degeneration* where the loss distribution collapses. These properties undermine both "sample-level filtering" and "loss-based differentiation."

**Goal**: To design a reliable training-phase sample selection mechanism that is sensitive to channel granularity and does not rely on a single loss signal or clean subsets.

**Key Insight**: Two observations are used. First, TSF backdoors define a **unidirectional** dependency (history → future); they do not protect the reverse (future → history). Training a backcaster can leverage this "directional asymmetry" to amplify differences between clean and poisoned samples. Second, using NTK-style kernel regression, the authors prove a backdoor success bound (Theorem 4.1), concluding that **successful TSF backdoors necessitate poisoned input windows "clustering" in the feature space**, meaning poisoned samples will exhibit **abnormally small neighborhood distances**.

**Core Idea**: Transforming training into "channel-wise pool training" at both **channel and time** granularities. A high-purity pool is initialized using RCF and NDF. Then, DRLS (Distance-Regularized Loss Selection) is used for progressive expansion, adding a distance regularizer to loss-based screening to prevent highly correlated poisoned windows from being re-absorbed in later stages.

## Method

### Overall Architecture
Let the training set be $\mathcal{D}=\{(\mathbf{X}_{t,h}, \mathbf{X}_{t,f})\}$, where each sample consists of a history window of size $L_{\text{in}} \times C$ and a future window of size $L_{\text{out}} \times C$. TimeGuard does **not** treat $(\mathbf{X}_{t,h},\mathbf{X}_{t,f})$ as an indivisible unit. Instead, it maintains an independent reliable pool $\mathcal{D}^{(c)}_{\text{rel}}$ for each channel $c$ and introduces a binary mask $m_{t,c} \in \{0,1\}$ to decide if the "channel $c$ at time $t$" participates in training. The predictor $f_\theta$ is trained with a masked loss:

$$\mathcal{L}_{\text{def}}(\theta;m) = \frac{1}{\sum_{t,c} m_{t,c}} \sum_{t,c} m_{t,c} \, \ell(f_\theta^{(c)}(\mathbf{X}_{t,h}), \mathbf{x}_{t,f}^{(c)})$$

The pipeline consists of two stages: **Stage I ($T_1=10$ epochs)** initializes a conservative but high-precision pool via RCF ∩ NDF; **Stage II ($T_2=90$ epochs)** uses DRLS to linearly expand the pool proportion $\gamma$ from $\alpha$ to $\beta=0.5$. No clean reference subset is needed, and the predictor architecture remains unchanged.

### Key Designs

1. **Channel-wise Pool Training**:
    - **Function**: Refines "sample-level" binary classification into a grid-level mask $m_{t,c}$ for "time $t \times$ channel $c$."
    - **Mechanism**: Maintains $\mathcal{D}^{(c)}_{\text{rel}}$ for each channel $c$. Filtering criteria (RCF / NDF / DRLS) are applied independently within each channel. The masked MAE/MSE loss is accumulated only on selected grid points.
    - **Design Motivation**: In ablation studies, reverting to sample-level filtering ("w/o Channel-wise") caused FDER to drop from 0.868 to 0.478. This is because attacks like BackTime often have $\eta_S = 0.3$. Sample-level discarding either removes clean channels (hurting $\text{MAE}_{\text{C}}$) or retains poisoned ones (hurting $\text{MAE}_{\text{P}}$), making channel granularity the correct alignment for defense.

2. **Stage I: RCF ∩ NDF Initialization**:
    - **Function**: Constructs a **conservative but high-precision** initial pool to prevent reinforcement of poisoned samples early in training.
    - **Mechanism**: (a) **RCF (Reverse-Consistency Filtering)**: Trains a backcaster $b_\phi$ to predict the reversed history window from the reversed future window. Loss is $\mathcal{L}_{\text{rcf}}(\mathbf{x}_t) = \ell(b_\phi(\text{Flip}(\mathbf{X}_{t,f})), \text{Flip}(\mathbf{x}_{t,h}))$. Samples below the $\alpha$-quantile of reverse loss enter $\mathcal{D}_{\text{RCF}}$. (b) **NDF (Neighborhood Diversity Filtering)**: Defines distance between windows as $d_\omega(\mathbf{x}_i,\mathbf{x}_j)=1-r_\omega(\mathbf{x}_i,\mathbf{x}_j)$ using Gaussian-weighted Pearson correlation (weight $\omega_\tau$ emphasizes the history-future boundary). Average distance to $K$ neighbors $S(\mathbf{x}_i)$ is calculated, and the top-$\alpha$ samples enter $\mathcal{D}_{\text{NDF}}$. The final pool is $\mathcal{D}_{\text{rel}} = \mathcal{D}_{\text{RCF}} \cap \mathcal{D}_{\text{NDF}}$.
    - **Design Motivation**: Theorem 4.1 provides a kernel regression upper bound $\|\hat y(\mathbf{x})-T(\mathbf{x})\|_2 \le \frac{N_{\text{bg}} M \varepsilon}{N_p \exp(-\gamma \sigma_p^2(\mathbf{x}))} + L_T \sigma_p(\mathbf{x})$. For an attack to succeed, poisoned input windows must cluster ($\sigma_p$ must be small), hence they naturally fall into the "abnormally small neighborhood distance" end targeted by NDF. RCF and NDF use independent loss and geometric signals, acting as uncorrelated lie detectors to boost pool purity.

3. **Stage II: DRLS (Distance-Regularized Loss Selection)**:
    - **Function**: Addresses the degradation where poisoned samples are re-absorbed due to collapsed loss signals; gradually expands the pool from $\alpha=0.2$ to $\beta=0.5$ to ensure clean accuracy.
    - **Mechanism**: Neighborhood distance is calculated using $\mathcal{D}_{\text{unrel}}$ (the unreliable subset) as reference neighbors to sharpen the signal as $\mathcal{D}_{\text{unrel}}$ becomes enriched with poisoned samples. Filtering involves two steps: (i) Selecting top $100\pi\gamma\%$ high-distance samples as candidates $\mathcal{D}_{\text{NDF}}^{\text{cand}}$; (ii) Selecting the $\gamma|\mathcal{D}|$ samples with the lowest loss from the candidates to form $\mathcal{D}_{\text{DRLS}}$.
    - **Design Motivation**: Removing DRLS ("w/o DRLS") reduced FDER to 0.607, confirming that loss signals alone are insufficient due to task-formulation shift. Pre-filtering with neighborhood diversity provides a necessary safety constraint.

### Loss & Training
Predictor $f_\theta$ is optimized with Adam (Stage I: 10 epochs, Stage II: 90 epochs). The backcaster $b_\phi$ is trained for 10 epochs. Hyperparameters: $\alpha=0.2$, $\beta=0.5$, $\pi \in [1.25, 1.5]$, $K \in [20, 32]$. All components operate independently per channel.

## Key Experimental Results

**Data**: PEMS03, Weather, ETTm1; **Models**: SimpleTM, FEDformer, TimesNet; **Attacks**: Random, FreqBack-TSF, BackTime; **Metrics**: $\text{MAE}_{\text{C}}$ ↓, $\text{MAE}_{\text{P}}$ ↑, FDER ↑.

### Main Results

Comparison on PEMS03 (average of three models):

| Method | $\text{MAE}_{\text{C}}$ ↓ (Random / BackTime) | $\text{MAE}_{\text{P}}$ ↑ (Random / BackTime) | FDER ↑ (Random / BackTime) |
|----------|-------|-------|-------|
| No Defense | 17.634 / 17.607 | 17.772 / 14.201 | – / – |
| Fine-pruning (requires clean data) | 19.020 / 18.686 | 31.643 / 19.736 | 0.633 / 0.623 |
| PDB (Prev. SOTA, requires clean data) | 18.630 / 18.967 | 54.690 / 22.397 | 0.693 / 0.639 |
| **TimeGuard (Ours, no clean data)** | **17.928 / 18.048** | **104.677 / 39.303** | **0.868 / 0.808** |

Overall: 1.96x gain in $\text{MAE}_{\text{P}}$ and 6.09% reduction in $\text{MAE}_{\text{C}}$ relative to PDB. On Weather, $\text{MAE}_{\text{C}}$ was 3.02% lower than No Defense, suggesting a regularization effect from distance filtering.

### Ablation Study (PEMS03 Average)

| Configuration | $\text{MAE}_{\text{C}}$ ↓ (Random / BackTime) | $\text{MAE}_{\text{P}}$ ↑ (Random / BackTime) | FDER ↑ (Random / BackTime) |
|------|-------|-------|-------|
| **Full TimeGuard** | 17.93 / 18.05 | **104.68** / **39.30** | **0.868** / **0.808** |
| w/o Channel-wise | 18.32 / 19.07 | 16.15 / 14.93 | 0.478 / 0.507 |
| w/o DRLS | 19.75 / 20.08 | 76.44 / 22.92 | 0.607 / 0.586 |

### Key Findings
- **Channel granularity is paramount**: Removing channel-wise training caused FDER to drop to ~0.49, proving signal dilution is the primary cause of prior defense failures.
- **DRLS is more critical than Stage I filtering**: Removing NDF/RCF individually only dropped FDER by 1–2%, but removing DRLS dropped it to 0.61. Managing long-term loss degradation is harder than initial pool purity.
- **Efficiency**: Training time is 1.58x vanilla, comparable to PDB and much faster than ESTI. **Zero overhead** at inference.
- **Adaptive Resistance**: An adaptive attack designed to minimize neighborhood distance actually performed worse ($\text{MAE}_{\text{P}}$ 15.34 vs 14.20), as disrupting the clustering also disrupts the attack mechanism per Theorem 4.1.

## Highlights & Insights
- The **"channel-wise + time-step" training paradigm** is the most transferable contribution. It is applicable to any scenario where attacks affect a subset of dimensions but losses aggregate across all (e.g., multi-task or multimodal defenses).
- **Asymmetric backcaster checking** is elegant: it translates the attacker's failure to ensure reverse consistency into a computable loss signal with minimal overhead.
- **NTK bound → geometric prior**: Theorem 4.1 bridges theoretical requirements for attack success with a geometric detection signal ("abnormally small neighborhood distance"), providing a principled path for defense.
- **Relative neighbor reference**: DRLS uses $\mathcal{D}_{\text{unrel}}$ as a reference. As the pool expands, the signal from the increasingly poisoned $\mathcal{D}_{\text{unrel}}$ becomes sharper, creating a self-strengthening loop.

## Limitations & Future Work
- **Limitations**: Training overhead remains (1.58x); applicable only to training-phase (not black-box deployment); inference-time detection in TSF remains an open problem.
- **Observations**: Theorem 4.1 relies on kernel approximations; effectiveness on long-horizon (720+) forecasting is not fully explored; fixed Gaussian weights might need tuning for different sampling rates.
- **Future Work**: Lightweight frequency-domain reconstruction to replace the backcaster; introducing cross-channel coordination mechanisms in DRLS; extending Theorem 4.1 to non-linear Transformer-based predictors.

## Related Work & Insights
- **vs BackTime**: TimeGuard directly counters BackTime's sparse channel poisoning by operating at the same granularity.
- **vs PDB**: Unlike PDB, which views backdoors as "capacity surplus," TimeGuard looks at the data pool and requires no clean subset, yielding a 1.96x $\text{MAE}_{\text{P}}$ gain.
- **vs ABL / ESTI**: These rely on early-loss separation, which fails in TSF as poisoned losses quickly collapse. TimeGuard uses distance-regularized criteria to overcome this.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic eval of TSF backdoor defense + first no-clean-data in-training method + channel-time granularity.
- Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × 3 predictors × 3 attacks + 13 baselines + adaptive/LLM transfer experiments.
- Writing: ⭐⭐⭐⭐ Clear logical flow, though some DRLS threshold definitions are densely formatted.
- Value: ⭐⭐⭐⭐⭐ High engineering value for deploying open-weight time series foundation models securely.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)
- [\[ICML 2026\] Training-Free Coverless Multi-Image Steganography with Access Control](training-free_coverless_multi-image_steganography_with_access_control.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)

</div>

<!-- RELATED:END -->
