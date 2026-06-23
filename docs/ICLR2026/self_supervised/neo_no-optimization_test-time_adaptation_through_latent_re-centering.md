---
title: >-
  [Paper Note] NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering
description: >-
  [ICLR 2026][Self-Supervised Learning][Test-Time Adaptation] NEO discovers that input distribution shifts cause a global translation in penultimate embeddings shared across samples and classes. By re-centering test features to the origin using a single global centroid vector, it outperforms seven mainstream TTA methods with **zero optimization, zero hyperparameters, and near-zer
tags:
  - ICLR 2026
  - Self-Supervised Learning
  - Test-Time Adaptation
  - Neural Collapse
  - Latent Re-Centering
  - Distribution Shift
  - Vision Transformer
  - Edge Inference
date: 2026-05-08
content_hash: 9543e901dc5facb0
---
# NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mVlIKLiizr](https://openreview.net/forum?id=mVlIKLiizr)  
**Code**: Public (The paper states it is open-source, replacing `nn.Linear` with a custom NEO layer)  
**Area**: Test-Time Adaptation / Self-Supervised Representation / Distribution Shift Robustness  
**Keywords**: Test-Time Adaptation, Neural Collapse, Latent Re-Centering, Distribution Shift, Vision Transformer, Edge Inference  

## TL;DR
NEO discovers that input distribution shifts cause a global translation in penultimate embeddings shared across samples and classes. By re-centering test features to the origin using a single global centroid vector, it outperforms seven mainstream TTA methods with **zero optimization, zero hyperparameters, and near-zero additional overhead**.

## Background & Motivation
**Background**: Test-Time Adaptation (TTA) allows models to self-adjust using unlabeled test data during the deployment phase to combat covariate shifts (corruptions like snow, fog, motion blur) between training and deployment. Mainstream approaches like TENT/SAR/EATA update BatchNorm affine parameters by minimizing prediction entropy, CoTTA performs continuous adaptation, and T3A uses prototypes to correct classifiers.

**Limitations of Prior Work**: Most existing methods rely on backpropagation or multiple forward passes, incurring three major costs: ① Significant increase in memory and latency (unaffordable for edge devices); ② Extreme sensitivity to hyperparameters, where poor tuning leads to catastrophic forgetting or zero accuracy; ③ Many methods assume the presence of BN layers or require large batches/numerous samples to calculate stable statistics. For scenarios like "on-vehicle traffic light recognition" requiring local, real-time, low-power processing, existing TTA is largely impractical.

**Key Challenge**: The effectiveness of TTA is almost always traded for "more computation, more samples, or more tuning." There is a **fundamental tension between accuracy gains and deployment feasibility**.

**Goal**: Develop a fully TTA method (no access to source data) that is **optimization-free, hyperparameter-free, and lacks significant overhead**, while remaining robust under single-sample, single-class, and continuous shifts.

**Key Insight**: **[Geometric Insight + Global Re-Centering]** The authors find that distribution shift in latent space manifests primarily as a **global translation shared across samples and classes**, concentrated in a few high-amplitude dimensions. Leveraging Neural Collapse theory, it can be proven that a model trained to convergence has a source data global centroid exactly at the origin ($\mu_G=0_d$). Therefore, "aligning the global centroid of corrupted samples back to the source" is equivalent to "re-centering test features to the origin." Adaptation thus only requires online estimation of a global centroid vector $\tilde\mu_G$ and subtracting it from features, requiring no gradients.

## Method

### Overall Architecture
NEO decomposes the model into an encoder $h$ and a linear classification head $\theta$ ($f=\theta\circ h$). It modifies no weights but inserts a "re-centering" operation between $h$ and $\theta$: it maintains an online global mean $\tilde\mu_G$ of corrupted embeddings and feeds $h(\tilde x)-\tilde\mu_G$ to the classification head for each test feature. Implementation requires only replacing a ViT `nn.Linear` with a custom layer; the storage cost is a single vector.

```mermaid
flowchart LR
    A[Corrupted Input x̃] --> B[Encoder h]
    B --> C["embedding h(x̃)"]
    C --> D["Online update global centroid μ̃_G<br/>(Cumulative average)"]
    C --> E[Re-center h(x̃) − μ̃_G]
    D --> E
    E --> F[Original linear head θ]
    F --> G[Prediction y]
```

### Key Designs

**1. Geometric Characterization of Distribution Shift: Shift as a globally shared translation on a few dimensions.** This is the empirical starting point. The authors decompose each sample as $h(\tilde x)=h(x)+\Delta_G+\Delta_c+\delta$, where $\Delta_G=\tilde\mu_G-\mu_G$ is the global centroid shift, $\Delta_c$ is the class shift after removing global shift, and $\delta$ is the sample residual. Statistics on 50,000 ImageNet-C samples show that for corruptions like contrast, the maximum difference for 95% of samples falls into fewer than 20/768 dimensions; across all corruptions, 80% of samples have maximum difference dimensions fewer than 50. Crucially, alignment experiments show that global alignment alone increases the cosine similarity between source/corrupted embeddings from −0.44 to 0.51 and reduces the L2 norm difference from 4.33 to 3.65 (the largest contribution), while subsequent class alignment yields minimal cosine gains and **increases the norm difference**. This indicates the "globally shared translation" is the primary component of shift, making class-wise refinement counterproductive.

**2. Grounding "Global Alignment" as "Origin Centering" via Neural Collapse.** This solves the problem of unknown source centroids in fully TTA. While $\Delta_G$ requires source data, Proposition 4.2 proves that under assumptions of neural collapse, cross-entropy, unconstrained features, and class balance, the source global centroid $\mu_G=0_d$. Thus, $\Delta_G=\tilde\mu_G-\mu_G=\tilde\mu_G$—**the global centroid of corrupted data itself is the shift to be subtracted**, requiring no source data. Empirically (Figure 3b, last row), the cosine similarity and norm difference obtained by subtracting $\tilde\mu_G$ from $h(\tilde x)$ are nearly identical to subtracting the true $\Delta_G$, validating this equivalence. Proposition 4.1 further explains that under neural collapse, linear classification is determined by $\cos(w_c,h(x))$ ($\arg\max_c \|w_c\|\,\|h(x)\|\cos(w_c,h(x))$); thus, restoring cosine similarity restores accuracy, while norm differences explain changes in confidence/calibration.

**3. Online Cumulative Mean + Continual Variant.** These allow the method to remain optimization-free while handling evolving shifts. The core NEO algorithm (Algorithm 1) is minimal: initialize $\tilde\mu_G\leftarrow 0_d$, and for each batch, update the centroid using a cumulative average $\tilde\mu_G\leftarrow\frac{i-1}{i}\tilde\mu_G+\frac{1}{i}\mathrm{Avg}(h(B))$, then output $\theta(h(B)-\tilde\mu_G\mathbf{1}_b^T)$. The process involves only averaging, addition, scalar multiplication, and one vector broadcast, with no gradients or matrix decompositions. Because it estimates a single global quantity from all samples, the estimation is highly robust (compared to T3A, which is unreliable due to few samples per class), batch-size independent, pseudo-label independent, and free of catastrophic forgetting. For time-varying shifts, **NEO-Continual** replaces the cumulative average with an Exponential Moving Average (EMA) $\tilde\mu_G\leftarrow(1-\alpha)\tilde\mu_G+\alpha\,\mathrm{Avg}(h(B))$, introducing only one semantically clear hyperparameter $\alpha$ to track drifting feature means.

## Key Experimental Results

### Main Results
Accuracy (%) for ViT-Base on 15 ImageNet-C corruptions using 512 samples for adaptation (selected + summary):

| Method | Contrast | Fog | Glass | ImageNet-C Avg | CIFAR-10-C | ImageNet-R | ImageNet-S |
|------|---------|-----|-------|-----------------|-----------|-----------|-----------|
| No Adapt | 32.6 | 65.8 | 35.3 | 55.6 | 80.4 | 59.2 | 45.4 |
| TENT | 36.9 | 62.4 | 36.8 | 56.3 | 81.3 | 59.4 | 45.7 |
| FOA | 54.5 | 70.7 | 36.8 | 58.4 | 80.9 | 60.2 | 46.3 |
| Surgeon | 31.7 | 63.0 | 36.8 | 56.1 | **82.7** | 60.2 | 47.0 |
| **NEO** | **58.2** | **71.2** | **37.9** | **59.2** | 82.4 | **60.3** | **47.2** |

- NEO achieves the highest accuracy in 12 out of 15 corruptions and ranks second in the remaining 3 (surpassed only by the much more expensive Surgeon); Average Gain is 3.6%, nearly doubling on Contrast (32.6→58.2).
- **NEO never degrades accuracy on any corruption**, contrasting with other TTA methods that often collapse due to poor tuning.
- It ranks first across ImageNet-C / ImageNet-R / ImageNet-S and beats 6 out of 7 methods on CIFAR-10-C.

### Ablation Study
Impact of different alignment methods on the quality of source/corrupted embedding alignment (ImageNet-C Severity 5 Gaussian Noise, ViT-Base, average of 50,000 samples):

| Alignment Type | Cosine Similarity w/ $h(x)$ ↑ | L2 Norm Difference ↓ |
|---------|----------------------|------------|
| $h(\tilde x)$ (No Alignment) | −0.44 | 4.33 |
| $-\Delta_G$ (Global Alignment) | 0.51 | 3.65 |
| $-\Delta_G-\Delta_c$ (Add Class Alignment) | 0.64 | 5.47 |
| $-\Delta_G-\Delta_c-\delta$ (Ideal Full Alignment) | 1.00 | 0.00 |
| $-\tilde\mu_G$ (**Origin Centering, calculatable**) | 0.49 | **3.64** |

Key Signal: Origin centering (the NEO scheme calculatable under fully TTA) nearly matches global alignment in cosine similarity and achieves the lowest norm difference, validating the $\Delta_G=\tilde\mu_G$ theory. Class alignment improves cosine similarity but worsens norm difference, justifying "global-only" correction.

### Key Findings
- **Sample Efficiency**: Adaptation with just 1 sample improves ImageNet-C accuracy by 1.5%; returns diminish quickly after one batch (64 samples).
- **Cross-Class Generalization**: Adapting with data from only 1 class improves the accuracy of the other 999 classes by over 3%.
- **Better Calibration**: ECE is maintained or improved in 9 out of 12 settings; ECE on ViT-S/ImageNet-C is lower than all baseline TTA methods.
- **Edge Efficiency**: On Jetson Orin Nano, NEO is the only method that does not increase inference time or peak memory. The abstract reports a 63% reduction in relative inference time and 9% reduction in memory compared to baselines.
- **Continual Adaptation**: NEO-Continual outperforms CoTTA on 15 randomly ordered corruptions, second only to the resource-intensive Surgeon.
- **Centroid Transferability**: High cosine similarity of $\tilde\mu_G$ between noise-type and blur-type corruptions suggests centroids can be reused across domains.

## Highlights & Insights
- **Reducing TTA from an "Optimization Problem" to "Vector Subtraction"**: While the field focuses on backprop-free iterative optimizers, NEO provides an analytical, optimization-free solution backed by Neural Collapse theory.
- **Elegance of "Origin Centering"**: The Neural Collapse conclusion that $\mu_G=0$ makes "source centroid alignment" actionable without source data, representing the most elegant leap in the paper.
- **Low Barrier for Engineering**: One line of code to replace `nn.Linear`, storage of a single vector, inherently batch-size independent, no forgetting, and no pseudo-label dependency—highly edge-friendly.
- **Rigorous Diagnostics**: Decomposition experiments on dimensions, alignment stages, and centroid similarity heatmaps thoroughly explain *why* global re-centering is sufficient.

## Limitations & Future Work
- **Narrow Architecture Coverage**: Validated only on ViT (ViT-S/B/L); it is unknown if "global shared translation" holds for CNNs, ConvNeXt, or hybrid architectures.
- **Penultimate Layer Focus**: Insights are limited to penultimate-layer activations; whether this applies to other layers or representations remains to be seen.
- **Strong Theoretical Assumptions**: $\mu_G=0_d$ depends on neural collapse, class balance, and unconstrained features; the equivalence may weaken in models with severe class imbalance or insufficient training.
- **Boundary of Single Global Shift**: NEO assumes test samples come from the same distribution. Rapid drift is managed by NEO-Continual's EMA, but $\alpha$ still requires manual setting, and it only corrects "translation" rather than complex covariance deformations.

## Related Work & Insights
- **Entropy-Minimization TTA** (TENT/SAR/EATA) contrasts with NEO: the former updates BN parameters via backprop and is sensitive to hyperparameters, whereas NEO fixes all weights.
- **Prototype/Output Analytical TTA** (T3A, LAME, FOA, Surgeon) are optimization-light, but T3A suffers from insufficient per-class samples and FOA requires source statistics. NEO avoids these vulnerabilities using a single global quantity.
- **Neural Collapse** (Papyan 2020, Súkeník 2023) was previously used for domain generalization and OOD detection; this work is the first to introduce it to TTA to argue that "global re-centering is enough," offering a new paradigm of "guiding deployment adaptation with convergence geometry."
- Insight: Many distribution shift problems might not require re-optimization but rather a **geometric correction in representation space**—a valuable paradigm for low-power, privacy-sensitive edge learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Turning the NC $\mu_G=0$ into actionable origin centering to reduce TTA to vector subtraction is a novel and theoretically grounded perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4 datasets × 3 ViTs × 7 baselines, covering accuracy, calibration, single-sample, continuous, and edge devices. Solid diagnostics, though limited to ViTs and the penultimate layer.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression from geometric observation to theoretical propositions to a minimal algorithm. Figure 3 is particularly effective.
- **Value**: ⭐⭐⭐⭐⭐ Zero hyperparameters, zero optimization, near-zero overhead, and no degradation. Deployable with one line of code; high practical value for edge/real-time systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ZeroSiam: An Efficient Asymmetry for Test-Time Entropy Optimization without Collapse](zerosiam_an_efficient_asymmetry_for_test-time_entropy_optimization_without_colla.md)
- [\[ICLR 2026\] Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment](architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [\[CVPR 2026\] Energy Waveify and Redistribution for Test-Time Adaptation: A Control System Perspective](../../CVPR2026/self_supervised/energy_waveify_and_redistribution_for_test-time_adaptation_a_control_system_pers.md)
- [\[ICLR 2026\] Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding](bayesian_test-time_adaptation_via_dirichlet_feature_projection_and_gmm-driven_in.md)

</div>

<!-- RELATED:END -->
