---
title: >-
  [Paper Note] Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding
description: >-
  [ICLR2026][Self-Supervised Learning][Test-time adaptation] BTTA-DG compresses the moment-to-moment prediction sequence of each EEG trial into a Dirichlet parameter vector. It utilizes a GMM fitted on historical trials as the likelihood and the deep model output as the prior to perform a gradient-free Bayesian posterior calibration. It achieves SOTA and real-time performance (15.7 ms/trial) in cross-subject/cross-session transfer for motor imagery BCI.
tags:
  - "ICLR2026"
  - "Self-Supervised Learning"
  - "Test-time adaptation"
  - "EEG motor imagery"
  - "Dirichlet distribution"
  - "Gaussian mixture"
  - "Bayesian calibration"
date: 2026-05-08
content_hash: 71704178d2ab51cb
---

# Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=VDg6Pv4S3v](https://openreview.net/forum?id=VDg6Pv4S3v)  
**Code**: To be confirmed  
**Area**: Test-time adaptation / Self-supervised / EEG BCI  
**Keywords**: Test-time adaptation, EEG motor imagery, Dirichlet distribution, Gaussian mixture, Bayesian calibration

## TL;DR
BTTA-DG compresses the moment-to-moment prediction sequence of each EEG trial into a Dirichlet parameter vector. It utilizes a GMM fitted on historical trials as the likelihood and the deep model output as the prior to perform a gradient-free Bayesian posterior calibration. It achieves SOTA and real-time performance (15.7 ms/trial) in cross-subject/cross-session transfer for motor imagery BCI.

## Background & Motivation
**Background**: EEG-based Motor Imagery (MI) brain-computer interfaces (BCIs) control peripherals by decoding sensorimotor rhythms. While large-scale EEG pre-training models have significantly advanced representation learning, practical deployment is hindered by the non-stationarity of EEG signals—distribution shifts occur when models are applied to different subjects or sessions (cross-subject/cross-session shift), making them nearly unusable without fine-tuning.

**Limitations of Prior Work**: Test-Time Adaptation (TTA) is an appropriate direction for adjusting models online using unlabeled data during inference. However, existing EEG-TTA methods fall into two categories with distinct drawbacks: ① **Gradient-based methods** (such as Tent's entropy minimization, pseudo-labeling, consistency regularization, T-TIME, and OTTA) require backpropagation to update parameters, which is computationally expensive. In online EEG scenarios where the batch size is often 1, a single noisy trial can generate misleading gradients, overriding pre-trained structures and causing catastrophic forgetting. ② **Data alignment methods** (such as BN statistic recalculation and Euclidean Alignment) are fast as they do not update parameters but only perform shallow statistical alignment, failing to capture how deep "temporal prediction embeddings" deform in a new domain.

**Key Challenge**: The goal is to achieve computational efficiency and avoid destroying pre-trained weights (preventing catastrophic forgetting) while modeling deep distribution changes with a theoretical foundation—a combination neither gradient-based nor alignment-based methods fully achieve.

**Key Insight**: The authors observe that **domain shift is more reliably reflected in the "distribution of the model's moment-to-moment prediction sequence" rather than in any single prediction**. An EEG trial processed by an encoder outputs a sequence of time-varying class probability vectors $X=[x_1,\dots,x_T]$. Instead of focusing on the time-averaged point estimate, it is more effective to model the "concentration level" of the entire probability trajectory.

**Core Idea**: Use the Dirichlet distribution (a "distribution of distributions") to project the temporal probability trajectory into a low-dimensional parameter vector $\alpha$. Use a GMM to model the density of historical $\alpha$ values and fuse it with the model prior via Bayesian inference. The entire adaptation process is **gradient-free**, calibrating only the output without modifying weights.

## Method

### Overall Architecture
BTTA-DG addresses online, single-trial cross-subject TTA: the source domain consists of labeled EEG data from multiple subjects, while the target domain consists of unlabeled trials arriving sequentially from a new subject. The goal is to adapt the pre-trained model $f_\theta$ without touching source data or target labels. The pipeline consists of a "lightweight backbone for feature extraction → projecting temporal prediction trajectories into Dirichlet parameters → Bayesian posterior calibration using a GMM of historical parameters."

Specifically: A lightweight **SincAdaptNet** serves as the backbone, utilizing learnable Sinc band-pass filters to extract MI-related mu/beta/gamma frequency bands. The encoder outputs a moment-to-moment class probability trajectory $X\in\mathbb{R}^{|L|\times T}$ for each trial, where the time average provides the model prior $f_{cls}(X)=\frac1T\sum_j x_j$. Then, **Dirichlet feature projection** uses Maximum Likelihood Estimation (MLE) to compress the trajectory $X$ into $\alpha\in\mathbb{R}^{|L|}_+$, where each component $\alpha_i$ represents the "evidence concentration" for class $i$, and the total scale $\alpha_0$ reflects overall uncertainty. Finally, **GMM-driven Bayesian inference** stores $\alpha$ values of historical high-confidence trials in a memory bank categorized by predicted class. A GMM is fitted for each class to serve as the likelihood $p_{GMM}(\alpha\mid y)$, which is multiplied by the model prior $p_\theta(y)$ and normalized to obtain the calibrated posterior. The argmax is taken as the final prediction, and the current $\alpha$ is written back to the memory bank based on confidence/entropy thresholds.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Online EEG trial<br/>(batch=1)"] --> B["SincAdaptNet backbone<br/>Learnable Sinc band-pass filter"]
    B --> C["Moment-to-moment prediction trajectory X<br/>+ Time average yields prior pθ(y)"]
    C --> D["Dirichlet feature projection<br/>MLE compresses X into α"]
    D --> E["GMM-driven Bayesian inference<br/>Historical α fits GMM as likelihood"]
    C -->|Prior pθ(y)| E
    E -->|Calibrated posterior argmax| F["Final prediction ŷ_cal"]
    E -->|Write-back based on confidence/entropy| G["Memory bank update<br/>+ EM re-fits GMM"]
    G -.Historical α.-> E
```

### Key Designs

**1. SincAdaptNet Backbone: Replacing free convolutional kernels with interpretable band-pass filters to provide clean features for subsequent projection**

**Function**: In online EEG single-trial adaptation, the batch size is often 1. Standard convolutional backbones rely on batch statistics and learn black-box kernels that mix frequency bands. SincAdaptNet uses a structure of `Spat-Conv → Sinc-Conv → IncCh-Conv → Cls-Conv` and inserts LayerNorm (rather than BatchNorm) after temporal filtering and channel expansion to eliminate instability issues of batch statistics under single-sample conditions. The core is Sinc-Conv: inspired by SincNet, it learns only the low cutoff frequency $f_{low}$ and bandwidth $f_{band}$ ($f_{high}=f_{low}+f_{band}$) for each filter, generating a windowed sinc band-pass kernel. This separates MI-related mu (8–13 Hz), beta (13–30 Hz), and gamma (>30 Hz) rhythms with minimal parameters. The encoder output is mapped to a normalized probability space via softmax, yielding instantaneous class probability vectors $x_j\in\Delta^{|L|-1}$ at each time point.

**2. Dirichlet Feature Projection: Compressing a temporal prediction trajectory into an interpretable concentration vector**

**Mechanism**: Shallow alignment fails to capture how deep prediction distributions shift in a new domain, while single predictions are too noisy. The solution is to introduce the Dirichlet distribution—a "distribution over categorical distributions"—into EEG-TTA. Assuming $T$ probability vectors $x_j$ from a trial are i.i.d. samples from $\mathrm{Dir}(\alpha)$, a projection $\mathcal{P}:\mathbb{R}^{|L|\times T}\to\mathbb{R}^{|L|}_+$ is used to estimate $\hat\alpha_{MLE}=\arg\max_\alpha\sum_{j=1}^T\log D(x_j;\alpha)$ via MLE, where:

$$D(x_j;\alpha)=\frac{\Gamma(\alpha_0)}{\prod_{i=1}^{|L|}\Gamma(\alpha_i)}\prod_{i=1}^{|L|}x_{ij}^{\alpha_i-1}.$$

$\hat\alpha_{MLE}$ is solved efficiently via Minka’s fixed-point iteration: $\alpha_i^{new}=\psi^{-1}\!\big(\psi(\alpha_0^{old})+\frac1T\sum_{j=1}^T\log x_{ij}\big)$ ($\psi$ is the digamma function). This low-dimensional vector is more informative than a class label: $\alpha_i$ encodes evidence concentration for class $i$, and $\alpha_0$ encodes uncertainty/variance. It explicitly captures **distribution-level shifts** caused by domain variations.

**3. GMM-Driven Bayesian Inference: Using the density of historical parameters as the likelihood for gradient-free posterior calibration**

**Mechanism**: To adapt without modifying weights, a memory bank $\mathcal{M}_y$ organized by predicted labels stores $\alpha$ values of high-confidence trials. For each class, a GMM performs non-parametric density estimation to obtain the class-conditional likelihood:

$$p_{GMM}(\alpha\mid y)=\sum_{k=1}^K\pi_{y,k}\,\mathcal{N}(\alpha;\mu_{y,k},\Sigma_{y,k}).$$

For a current trial, the GMM likelihood and deep model prior $p_\theta(y)=f_\theta(s_i)$ are fused using Bayes' theorem:

$$p_{cal}(y\mid\hat\alpha_{MLE})=\frac{p_{GMM}(\hat\alpha_{MLE}\mid y)\,p_\theta(y)}{\sum_{y'=1}^{|L|}p_{GMM}(\hat\alpha_{MLE}\mid y')\,p_\theta(y')},$$

with the final prediction being $\hat y_{cal}=\arg\max_y p_{cal}(y\mid\hat\alpha_{MLE})$. The current $\hat\alpha$ is saved to the memory bank based on a confidence threshold $\tau_{conf}$ and entropy threshold $\tau_{ent}$. Standard EM is used to re-fit the GMM on the corresponding bank after each insertion. Since the Dirichlet feature dimension is extremely low, the EM overhead is minimal. This process **completely bypasses gradient optimization**, avoiding catastrophic forgetting and high computational costs.

### Loss & Training
The SincAdaptNet classifier is pre-trained normally on the source domain. During the test phase, **no gradient updates are performed**. Adaptation is realized entirely through Dirichlet projection (MLE fixed-point iteration), GMM (EM fitting), and Bayesian posterior fusion. Pre-processing involves 1–48 Hz band-pass filtering and Euclidean Alignment (EA). Key hyperparameters include the number of GMM components $K$, and thresholds $\tau_{conf}$ and $\tau_{ent}$.

## Key Experimental Results

### Main Results
Cross-subject Leave-One-Subject-Out (LOSO) was performed on three MOABB MI datasets (BNCI2014001/002, BNCI2015001) and SHU MI, with 10 independent runs for each method.

| Dataset (Cross-subject LOSO) | Metric | BTTA-DG | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| BNCI2014001 | Acc(%) | **78.70** | OTTA 77.58 | +1.12 |
| BNCI2014002 | Acc(%) | **80.29** | OTTA 78.29 | +2.00 |
| BNCI2015001 | Acc(%) | **77.92** | OTTA 76.20 | +1.72 |
| BNCI2014001 Cross-session | Acc(%) | **86.50** | OTTA 83.91 | +2.59 |

Notably, several gradient-based baselines (Tent, PL) suffered a **performance drop** after TTA because noisy trials in batch size 1 induced misleading gradient updates to BN layers, causing catastrophic forgetting. BTTA-DG avoids this by freezing the network.

**Computational Efficiency**: BTTA-DG averages 15.7 ms/trial and 141.6 MFLOPs for inference, which is 17.8% faster than T-TIME and 24.2% faster than OTTA.

### Ablation Study
Breakdown of components (average across datasets):

| Configuration | 2014001 Cross-session | 2014002 | Description |
|------|------|------|------|
| SincAdaptNet (Source Only) | 80.62 | 76.40 | Backbone baseline |
| BTTA-DG w/o EA | 81.88 | 77.55 | Removes EA; retains Dirichlet+GMM; still improves |
| SincAdaptNet + EA | 82.33 | 78.05 | Standard domain alignment |
| + EA + GMM | 82.47 | 78.25 | GMM on mean probability; marginal gain |
| + EA + Dirichlet | 84.04 | 78.88 | Dirichlet projection is the main driver |
| **BTTA-DG (Full)** | **86.50** | **80.29** | All modules combined; absolute gain of 2–6% |

### Key Findings
- **Dirichlet projection is the core source of gain**: Applying GMM directly to mean probabilities (+EA+GMM) was far less effective than using the Dirichlet projection (+EA+Dirichlet), emphasizing the value of modeling the temporal trajectory distribution.
- **Hyperparameter insensitivity**: Accuracy remained stable for $K\in[2,12]$ and various confidence/entropy thresholds.
- **Robustness to class imbalance**: When the test set ratio was shifted to 1:0.25, while overall accuracy decreased smoothly, minority class accuracy actually increased, indicating specialization towards rare events.
- **Physiological interpretability**: Learnable Sinc filters concentrated energy at mu (11.2 Hz), beta (30.5 Hz), and gamma (55.3 Hz).

## Highlights & Insights
- The **paradigm shift from "adapting point estimates" to "adapting distribution representations"** is a major contribution. Upgrading each trial from a single class prediction to a Dirichlet concentration vector captures more information about domain shift.
- **Reframing adaptation as density estimation and Bayesian fusion** rather than an optimization problem effectively avoids catastrophic forgetting and backpropagation overhead.
- The **Sinc band-pass backbone** provides spectral interpretability with minimal parameters, a reusable trick for medical/neural signal scenarios.

## Limitations & Future Work
- **Dependence on roughly balanced EEG data**: Density estimation on the memory bank may struggle with extreme class imbalance or strong non-stationarity.
- **Limitations of GMM assumptions**: Whether GMM is sufficient for highly multimodal or heavy-tailed real distributions of Dirichlet parameters remains to be explored.
- **Integration with large pre-trained models**: Applying BTTA-DG to large-scale EEG pre-trained models to check for additive gains is a future direction.

## Related Work & Insights
- **vs. Gradient-based TTA (Tent / T-TIME / OTTA)**: While these methods update parameters through backpropagation and are prone to noise and forgetting at batch=1, BTTA-DG freezes the network and uses gradient-free calibration, proving to be 17–24% faster and more performant.
- **vs. Data alignment methods (BN-adapt / Euclidean Alignment)**: These perform only shallow statistical alignment. BTTA-DG explicitly models distribution drift of deep trajectories via Dirichlet projection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Dirichlet distributions to EEG-TTA and using distribution parameters for gradient-free Bayesian calibration is a novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across four datasets and cross-subject/session scenarios, though comparison with large pre-trained models is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of logic from motivation to theory, supported by effective visualizations.
- Value: ⭐⭐⭐⭐ Real-time, anti-forgetting, and interpretable; highly relevant for practical BCI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment](architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment.md)
- [\[ICLR 2026\] NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering](neo_no-optimization_test-time_adaptation_through_latent_re-centering.md)
- [\[CVPR 2026\] Energy Waveify and Redistribution for Test-Time Adaptation: A Control System Perspective](../../CVPR2026/self_supervised/energy_waveify_and_redistribution_for_test-time_adaptation_a_control_system_pers.md)
- [\[ICLR 2026\] ZeroSiam: An Efficient Asymmetry for Test-Time Entropy Optimization without Collapse](zerosiam_an_efficient_asymmetry_for_test-time_entropy_optimization_without_colla.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
