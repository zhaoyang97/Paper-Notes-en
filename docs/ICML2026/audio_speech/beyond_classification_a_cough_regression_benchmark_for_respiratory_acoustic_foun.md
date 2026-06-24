---
title: >-
  [Paper Note] Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models
description: >-
  [ICML2026][Audio & Speech][Cough Acoustics] Existing respiratory acoustic foundation models (FMs) have been evaluated almost exclusively on cough **classification**. This paper presents the first systematic evaluation of FMs on **continuous regression** tasks (passively estimating age, BMI, and disease probability from cough audio). Using a multi-model multi-target benchmark protocol consisting of 5 FMs × 6 targets × 3 datasets with frozen encoders and three types of regressi…
tags:
  - "ICML2026"
  - "Audio & Speech"
  - "Cough Acoustics"
  - "Respiratory Foundation Models"
  - "Regression Benchmark"
  - "Linear Probe"
  - "Cross-dataset Generalization"
date: 2026-05-08
content_hash: eaeee03f9255f56c
---

# Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models

**Conference**: ICML2026  
**arXiv**: [2606.15436](https://arxiv.org/abs/2606.15436)  
**Code**: TBD  
**Area**: Audio & Speech / Respiratory Acoustics / Foundation Model Evaluation  
**Keywords**: Cough Acoustics, Respiratory Foundation Models, Regression Benchmark, Linear Probe, Cross-dataset Generalization  

## TL;DR
Existing respiratory acoustic foundation models (FMs) have been evaluated almost exclusively on cough **classification**. This paper presents the first systematic evaluation of FMs on **continuous regression** tasks (passively estimating age, BMI, and disease probability from cough audio). Using a multi-model multi-target benchmark protocol consisting of 5 FMs × 6 targets × 3 datasets with frozen encoders and three types of regression heads, the study reveals findings obscured by classification-based evaluations, including the "data scale × head capacity" tradeoff, the advantages of generative pre-training, and strongly asymmetric cross-dataset transfer.

## Background & Motivation

**Background**: Cough sounds encode physiological information far beyond binary "healthy/sick" labels—spectral-temporal features reflect airway geometry, respiratory muscle strength, and mucosal viscosity, all of which **quantitatively co-vary** with age, body composition, and disease severity. In low- and middle-income countries (LMIC) where birth records, scales, and imaging are often unavailable, passive estimation of age/BMI via cough provides a feasible triage alternative. Meanwhile, foundation models pre-trained on large-scale unlabeled audio (OPERA, HeAR, M2D+Resp) enable efficient transfer via linear probes, significantly reducing the annotation burden for clinical audio AI.

**Limitations of Prior Work**: While these FMs have been densely benchmarked on **classification**, their **regression capabilities remain largely uncharacterized**. Although the original HeAR paper evaluated age/BMI regression, it used a single model with a fixed linear probe and device-based splitting, lacking multi-model comparisons, non-linear heads, and cross-dataset generalization tests. The regression tasks for OPERA (lung capacity estimation) are derived from deep breathing and vowels rather than cough, and M2D+Resp has never been evaluated on cough regression.

**Key Challenge**: Classification metrics can **mask** what truly matters in regression tasks—a model that predicts the population mean for all samples might appear "effective" in certain classification settings but lacks predictive power for individual patients. Honest regression evaluation must consistently compare against a "Mean Absolute Deviation (MAD) baseline"; only when MAE < MAD is a signal beyond the population mean captured.

**Goal**: To provide a unified, comparable, subject-disjoint cough regression benchmark that answers four neglected questions: Which regression head is optimal at different data scales? Is generative or contrastive pre-training better suited for regression? Can models transfer across datasets, and is the direction symmetric? Which FMs are more label-efficient in low-data regimes?

**Key Insight**: By **freezing** all FMs as feature extractors and varying only the regression heads (linear / MLP-small / wide MLP), the variables are converged into three dimensions: "pre-training paradigm × head capacity × data scale," allowing the contribution of each factor to be isolated cleanly.

**Core Idea**: Use a regression benchmark of "multi-model × multi-target × multi-dataset + consistent MAD baseline" to expose differences in pre-training paradigms, capacity-scale tradeoffs, and transfer asymmetries that are invisible in classification evaluations.

## Method

### Overall Architecture
The benchmark takes raw audio from three cough datasets, resampled to 16 kHz mono and padded/clipped to 2 seconds. These are fed to 5 **frozen** respiratory/health audio encoders to extract embeddings (reused for all heads and evaluations). The embeddings are then fed to three regression probes (Linear / MLP-small / wide MLP) to produce MAE under three evaluation regimes: within-dataset regression (6 targets), regression head comparison (90 model×task×head combinations), and cross-dataset transfer (6 age transfer directions). All results are reported alongside the baseline MAD, with "signal strength" quantified by the $\text{best MAE}/\text{MAD}$ ratio. This is an evaluation pipeline rather than a new model, focusing on protocol design and controlled comparison.

### Key Designs

**1. Regression Protocol with Consistent MAD Comparison: Distinguishing "Seeming Efficacy" from "True Signal"**

Regression tasks often deceive when a model collapsing to the population mean yields a seemingly decent MAE. This study calculates the **Mean Absolute Deviation (MAD)** (the MAE of a naive mean-prediction baseline) for every label distribution. All model MAEs are reported alongside MAD, defining a signal strength ratio $\text{best MAE}/\text{MAD}$—only values significantly $<0.90$ are considered to have learned "patient-level signals above random." This design exposes a critical reality: in the Zambia clinical cohort (CIDRZ), the best/MAD for four targets ranges between 0.92–0.99 (with age within 1% of the baseline), indicating that FM embeddings contain **no usable individual-level signals** for that cohort. The gaps between models for clinical scores (X-ray abnormality, TB probability) are near zero (TB ≤ 0.004, X-ray ≤ 0.012 MAE), pointing to a shared representation ceiling rather than true recovery of clinical variance.

**2. Three Regression Heads × Data Scale: Decoupling the "Capacity-Scale" Tradeoff**

To clarify whether performance bottlenecks stem from pre-trained representations or mismatched probe capacity, three heads are compared: **Linear** ($d_{\text{feat}}\to1$), **MLP-small** ($d_{\text{feat}}\to256\to1$, with ReLU and 0.3 dropout, designed to be embedding-dimension agnostic), and **Wide MLP** ($d_{\text{feat}}\to d_{\text{feat}}\to1$, which creates ~15M hidden parameters for M2D+Resp's 3840-dim embedding). Findings show MLP-small wins in 23 out of 30 model×task combinations, outperforming linear probes by up to 0.38 yr (HeAR / Coswara). Conversely, the Wide MLP overfits on the small CIDRZ set ($N_{\text{train}}=669$, parameter-to-sample ratio ~22000:1) but recovers on the larger CoughVID set ($N_{\text{train}}=3050$), where Opera-GT achieves the best result of 9.53 yr. This establishes a **data scale × head capacity tradeoff**: use MLP-small for small data and Wide MLP only for large datasets. All heads use Adam (lr $=10^{-4}$, L2 $=10^{-5}$, batch 64) with a 0.97 decay per epoch and early stopping based on validation MAE.

**3. Controlled Comparison of Pre-training Paradigms: Generative vs. Contrastive**

Three encoders from the OPERA family with similar architectures but different objectives are compared: Opera-CT (Contrastive Transformer), Opera-CE (Contrastive CNN), and Opera-GT (Generative MAE), all pre-trained on the same 136K respiratory segments. Results show Opera-GT **outperforms** Opera-CT in age regression across all three datasets (CIDRZ 10.49 vs 10.52, Coswara 10.16 vs 10.25, CoughVID 9.62 vs 9.79 yr). While the 0.03 yr difference in CIDRZ is within one std, the 0.09/0.17 yr gaps in Coswara/CoughVID exceed seed variance, indicating the trend is driven by larger datasets. This extends the finding from the original OPERA paper—"generative pre-training favors regression"—from **respiratory sounds** to **cough**.

**4. Asymmetric Cross-dataset Transfer + Low-data Regime: Generalization Directions and Efficiency**

Testing across six age transfer directions (train on one, test on another without adaptation) reveals **strong asymmetry**. Transfer is successful only when CIDRZ is the target: CoughVID→CIDRZ shows a negative gap (−0.17 yr), and Coswara→CIDRZ is stagnant (+0.03 yr), with Opera-CE leading, suggesting large-scale crowdsourced data can substitute for scarce clinical training data. The inverse fails: CIDRZ→Coswara results in a +2.43 yr (+26.6%) degradation. In the low-data regime, HeAR and M2D+Resp approach full-scale performance at $N=50$ (HeAR is only 0.02 yr away from its $N=669$ performance), whereas OPERA models exhibit high variance at $N=50$ ($\text{std} \pm 0.22 \text{ yr}$) and only stabilize at $N=400$. This indicates that **pre-training data diversity determines low-data regression performance**.

### Loss & Training
All regression heads use MSE loss + Adam (lr $10^{-4}$, weight decay $10^{-5}$, batch 64), with LR $\times 0.97$ per epoch, maximum 64 epochs, and early stopping (patience 10) based on validation MAE. CIDRZ/Coswara use a 64/16/20% subject-disjoint split, while CoughVID uses the official UUID-level split (train 3050 / val 1019 / test 2789). Results are averaged over 5 seeds.

## Key Experimental Results

### Main Results

Intra-dataset age regression MAE (MLP-small, mean of 5 seeds, unit: yr; MAD is the baseline; lower best/MAD is better):

| Task | MAD | Opera-CT | Opera-GT | HeAR | M2D+Resp | best/MAD |
|------|-----|----------|----------|------|----------|----------|
| CIDRZ Age | 10.35 | 10.52 | 10.49 | 10.29† | 10.40 | 0.99 |
| Coswara Age | 11.31 | 10.25 | 10.16 | **9.12** | 9.58 | 0.81 |
| CoughVID Age | 10.29 | 9.79 | 9.62 | 9.61 | 9.79 | 0.93 |
| CIDRZ BMI (kg/m²) | 3.74 | 3.60 | 3.67 | 3.60† | 3.63 | 0.96 |
| CIDRZ TB (prob) | 0.205 | 0.189 | 0.190 | 0.188† | 0.192 | 0.92 |

> † HeAR results on CIDRZ may be affected by pre-training contamination (CIDRZ might be in HeAR's training data) and are excluded from headline claims. HeAR leads on Coswara age with 9.12 yr MAE (best/MAD = 0.81, the only task with a clear signal).

### Cross-dataset Age Transfer (MLP-small, best model per row; Gap = Cross-domain − Intra-domain MAE)

| Train → Test | Best Model | Cross-domain | Intra-domain | Gap |
|------|------|------|------|------|
| CoughVID → CIDRZ | Opera-CE | 10.34 | 10.51 | **−0.17** |
| Coswara → CIDRZ | Opera-CE | 10.54 | 10.51 | +0.03 |
| Coswara → CoughVID | Opera-CT | 10.42 | 9.79 | +0.63 |
| CoughVID → Coswara | HeAR | 10.05 | 9.12 | +0.94 |
| CIDRZ → CoughVID | HeAR | 10.54 | 9.61 | +0.94 |
| CIDRZ → Coswara | HeAR | 11.55 | 9.12 | +2.43 |

### Key Findings
- **CIDRZ clinical cohort hits the random floor**: Best/MAD for four targets ranges from 0.92–0.99, with age within 1% of the baseline, showing FM embeddings lack usable individual-level signals for this cohort.
- **Capacity-scale tradeoff serves as a deployment guide**: MLP-small wins in 23/30 cases; Wide MLP overfits on $N_{\text{train}}=669$ but excels at $N_{\text{train}}=3050$.
- **One-way transfer**: Crowdsourced/web data transfers to small clinical populations (even yielding negative gaps), but the reverse fails (up to +26.6% degradation).
- **Pre-training diversity dictates low-data efficiency**: HeAR / M2D+Resp reach near-peak performance at $N=50$, while OPERA requires $N=400$.

## Highlights & Insights
- The **best/MAD ratio** provides a "reality check" for regression, exposing that all models on CIDRZ are merely oscillating around the mean—a negative conclusion that would be hidden by MAE values alone.
- Isolating "pre-training paradigm / head capacity / data scale" using frozen encoders provides a reusable evaluation framework applicable to other clinical acoustic tasks.
- The "data leakage self-audit"—labeling potential HeAR–CIDRZ contamination and excluding it from headlines—is a commendable practice that should be adopted in all foundation model benchmarks.
- The finding that "generative pre-training excels in regression" is extended from respiratory sounds to cough, suggesting that generative models (MAE) should be prioritized for passive health estimation.

## Limitations & Future Work
- **Weak positive signals**: Only Coswara age (best/MAD 0.81) shows a clear signal; the benchmark serves more as an honest disclosure that "FMs underperform in cough regression" rather than a performance breakthrough.
- **Clinical scores $\neq$ clinical endpoints**: X-ray abnormality and TB probability are continuous derived scores, not binary microbiological diagnoses, limiting their clinical utility.
- **Small dataset sizes**: CIDRZ ($N=1049$) results in very small training sets after subject-disjoint splitting, making Wide MLP overfitting inevitable and limiting some conclusions to the scale of the data.
- **Potential pre-training contamination**: While HeAR–CIDRZ overlap was noted, its efficiency at $N=50$ might still be influenced by contamination, requiring further verification on cleaner hold-out sets.
- Transfer and paradigm comparisons were primarily conducted for **age**, with less focus on BMI or disease probability.

## Related Work & Insights
- **vs. HeAR (Baur 2024)**: HeAR evaluated cough age/BMI regression but with a single model and fixed probe; this work expands to multi-model, multi-head, and cross-dataset settings with MAD baselines.
- **vs. OPERA (Zhang 2024)**: OPERA tasks used breathing/vowels; this work focuses on cough and validates the "generative pre-training is better for regression" hypothesis in the cough domain.
- **vs. M2D+Resp (Niizumi 2025)**: Previously unevaluated for cough regression, this model is shown to be as efficient as HeAR in low-data regimes ($N=50$).

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-model multi-target cough regression benchmark; addresses a grounded problem, though it is an evaluation rather than a new methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of 5 FMs, 6 targets, 3 datasets, 3 head types, cross-domain, and low-data regimes with 5-seed variance.
- Writing Quality: ⭐⭐⭐⭐ Honest conclusions, clear comparisons, and proactive addressing of contamination; well-measured presentation of predominantly negative results.
- Value: ⭐⭐⭐⭐ Provides credible, controlled evidence and deployment guidelines for FM selection in passive health estimation for LMICs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RespiraMFM: A Multimodal Foundation Model for Respiratory Disease Recognition via Contrastive Audio-Language Alignment](../../ACL2026/audio_speech/respiramfm_a_multimodal_foundation_model_with_contrastive_audio-language_alignme.md)
- [\[ICML 2026\] Attend to Anything: Foundation Model for Unified Human Attention Modeling](attend_to_anything_foundation_model_for_unified_human_attention_modeling.md)
- [\[CVPR 2026\] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](../../CVPR2026/audio_speech/babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)
- [\[ICLR 2026\] YuE: Scaling Open Foundation Models for Long-Form Music Generation](../../ICLR2026/audio_speech/yue_scaling_open_foundation_models_for_long-form_music_generation.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)

</div>

<!-- RELATED:END -->
