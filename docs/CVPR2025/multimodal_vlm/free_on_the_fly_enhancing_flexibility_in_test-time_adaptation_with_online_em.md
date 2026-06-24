---
title: >-
  [Paper Note] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM
description: >-
  [CVPR 2025][Multimodal VLM][Test-Time Adaptation] FreeTTA proposes a training-free and storage-free test-time adaptation method that explicitly models the target domain distribution via an online EM algorithm. By leveraging CLIP zero-shot predictions as priors to iteratively estimate the Gaussian distribution parameters of each class, it consistently outperforms existing TTA methods across 15 datasets.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Test-Time Adaptation"
  - "Vision-Language Models"
  - "Online EM Algorithm"
  - "Gaussian Mixture Model"
  - "Training-free"
date: 2026-05-08
content_hash: 901416067e108413
---

# Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM

**Conference**: CVPR 2025  
**arXiv**: [2507.06973](https://arxiv.org/abs/2507.06973)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Test-Time Adaptation, Vision-Language Models, Online EM Algorithm, Gaussian Mixture Model, Training-free

## TL;DR

FreeTTA proposes a training-free and storage-free test-time adaptation method that explicitly models the target domain distribution via an online EM algorithm. By leveraging CLIP zero-shot predictions as priors to iteratively estimate the Gaussian distribution parameters of each class, it consistently outperforms existing TTA methods across 15 datasets.

## Background & Motivation

**Background**: Vision-Language Models (such as CLIP) have achieved strong zero-shot generalization capabilities through large-scale image-text contrastive pre-training. However, in practical deployment, performance drops significantly when test data exhibits a domain shift from the training data. The Test-Time Adaptation (TTA) paradigm was introduced to address this, aiming to adjust models online using unlabeled testing data.

**Limitations of Prior Work**: Current TTA methods suffer from three main limitations: (1) **Lack of target distribution modeling**—methods like TPT process samples independently or only utilize relationships among very few samples (e.g., TDA), failing to exploit the intrinsic relationships among test samples; (2) **Usability limitations**—prompt tuning methods modify model parameters, and TDA stores historical test features, making them infeasible in API-access and privacy-sensitive scenarios; (3) **Efficiency and stability issues**—TPT/DiffTPT require multiple data augmentations and gradient backpropagation per sample to optimize prompts, inducing huge computational overhead, while entropy minimization may lead to overfitting and overconfidence.

**Key Challenge**: An ideal TTA method should simultaneously satisfy three properties: explicit modeling of the target distribution, usability without additional assumptions, and training-free efficiency. However, existing methods satisfy at most one or two of these properties, failing to achieve all three simultaneously.

**Goal**: How to design a TTA method that simultaneously meets the following criteria: (1) explicitly models the target domain distribution to exploit inter-sample relationships; (2) does not require access to or storage of training data or historical test data; (3) requires no gradient backpropagation, maintaining high efficiency and stability?

**Key Insight**: It is assumed that the test features for each class follow a Gaussian distribution, making the overall test data a Gaussian Mixture Model (GMM). By employing CLIP text embeddings as initial class means, an online EM algorithm sequentially processes each test sample—computing posterior probabilities in the E-step and updating distribution parameters in the M-step—thereby gradually approximating the true target domain distribution without storing historical data.

**Core Idea**: Initialize the Gaussian Mixture Model parameters using CLIP text embeddings, sequentially update the distribution parameters per sample via an online EM algorithm, and integrate CLIP zero-shot confidence weighting to achieve training-free online domain adaptation.

## Method

### Overall Architecture

Given a test sample $x_t$, a frozen CLIP image encoder is used to extract features, while the text encoder generates class feature vectors using prompt templates. The online EM algorithm initializes the mean vectors with text features and the shared covariance matrix with the identity matrix. For each incoming sample, the E-step calculates the posterior probability of each class, and the M-step updates the class means and covariance using the current prediction. The final predicted logits are a weighted combination of the CLIP zero-shot predictions and the GDA probabilities.

### Key Designs

1. **Parameter Initialization and Gaussian Discriminant Analysis (GDA)**:

    - **Function**: To provide reasonable initial parameter estimates for online EM
    - **Mechanism**: Use the CLIP text encoder to generate text features $g(t_y)$ for each class $y$ as the initial mean $\mu_y$, while the shared covariance matrix $\Sigma$ is initialized to the identity matrix $I$. Under this assumption, classification is converted into comparing the Mahalanobis distance of a test sample to each class mean. The posterior probability is calculated as $P(y|x_t) \propto \exp\left(-\frac{1}{2}(x_t - \mu_y)^\top \Sigma^{-1}(x_t - \mu_y)\right)$
    - **Design Motivation**: CLIP text embeddings and visual embeddings occupy the same shared space. Utilizing text features as the initial class means is both reasonable and cost-free. Initializing with an identity matrix is simple and unbiased, and subsequent EM iterations will progressively approximate the true distribution.

2. **Online EM Algorithm (Online EM)**:

    - **Function**: Sequentially update GMM parameters online to adapt to the target domain distribution
    - **Mechanism**: For the $t$-th test sample:
        - **E-step**: Compute the posterior probability of the sample belonging to each class: $\gamma_{y,t} = \frac{\pi_y \cdot \mathcal{N}(x_t|\mu_y, \Sigma)}{\sum_j \pi_j \cdot \mathcal{N}(x_t|\mu_j, \Sigma)}$
        - **M-step**: Update the prior $\pi'_y = \frac{N_y + \gamma_{y,t}}{n_t}$, the mean $\mu'_y = \frac{N_y \cdot \mu_y + \gamma_{y,t} \cdot x_t}{N_y + \gamma_{y,t}}$, and the covariance $\Sigma' = \frac{(n_t-1)\Sigma + \sum_y \gamma_{y,t}(x_t - \mu'_y)(x_t - \mu'_y)^\top}{n_t - 1}$
        - The key lies in: only the current sample and accumulated statistics (sample count per class $N_y$, current mean $\mu_y$, covariance $\Sigma$) are required, without needing to store any historical samples.
    - **Design Motivation**: Traditional EM requires concurrent access to all data for batch updates, which is unsuitable for the online setting. Implementing online EM with incremental update formulas allows each sample to be "assigned" to different classes weighted by posterior probabilities, achieving streaming adaptation.

3. **VLM Prior Integration and Confidence Weighting**:

    - **Function**: Leverage the confidence of CLIP zero-shot predictions to control the influence of each sample on parameter updates.
    - **Mechanism**: Compute the self-information entropy of the CLIP zero-shot prediction $H(x_t) = -\sum_y P_{\text{CLIP}}(z_y=1|x_t) \log P_{\text{CLIP}}(z_y=1|x_t)$. Define a weighting function $w(h) = e^{-\beta h}$, where high-confidence (low-entropy) samples receive large weights, and low-confidence (high-entropy) samples receive small weights. Multiply $\gamma_{y,t}$ in the EM update formula by $w(H(x_t))$. The final logits are a linear combination of CLIP zero-shot logits and GDA logits: $\text{logits}_y = FT_y^\top + \alpha(w_y^\top F + b_y)$
    - **Design Motivation**: At the beginning of TTA, parameters are unstable, and highly uncertain samples may introduce noise, leading to parameter drift. Weighting by the original CLIP prediction confidence is equivalent to trusting the CLIP prior more in the early stages, and gradually increasing the contribution of EM as the distribution estimation becomes more accurate over time.

### Loss & Training

FreeTTA is entirely training-free—executing no gradient backpropagation and optimizing no parameters. All "updates" are incremental computations of statistical utilities (online updates of means and covariances), incurring minimal computational overhead.

## Key Experimental Results

### Main Results (Cross-Domain, ViT-B/16)

| 方法 | T.D. | Avail. | T.F. | AIR | CAL | DTD | FLWR | SUN | UCF | AVG |
|------|------|--------|------|-----|-----|-----|------|-----|-----|-----|
| CLIP zero-shot | - | - | - | 23.22 | 93.55 | 45.04 | 66.99 | 65.63 | 65.16 | 64.59 |
| TPT | ✗ | ✔ | ✗ | 24.78 | 94.16 | 47.75 | 68.98 | 65.50 | 68.04 | 65.10 |
| TDA | ✗ | ✗ | ✔ | 23.91 | 94.24 | 47.40 | 71.42 | 67.62 | 70.66 | 67.53 |
| **FreeTTA** | ✔ | ✔ | ✔ | **26.93** | **95.81** | **50.80** | **74.15** | **69.06** | **72.87** | **69.80** |

### Ablation Study

| Configuration | ImageNet | AVG(cross-domain) | Description |
|------|----------|-------------------|------|
| CLIP zero-shot | 68.34 | 64.59 | Baseline |
| + GDA (w/o EM) | 68.83 | 65.90 | Static Gaussian Classification |
| + Online EM | 69.18 | 67.32 | Add Online Update |
| + Confidence Weighting | 69.48 | 68.61 | Add Entropy Weighting |
| + Final Combination | **69.84** | **69.80** | Full Model |

### Key Findings
- FreeTTA is the only method that simultaneously satisfies all three characteristics: "target distribution modeling, usability, and training-free operation."
- Compared to CLIP zero-shot, the average improvement is around 5 percentage points (64.59 → 69.80), and exceeds 20 percentage points on specific datasets (e.g., EuroSAT).
- The contribution of online EM (+1.42) is greater than that of static GDA (+1.31), demonstrating that online updates indeed progressively approximate the true distribution.
- Confidence weighting brings a stable improvement (+1.29), validating the necessity of suppressing low-confidence samples.
- FreeTTA does not require gradient backpropagation, with an inference speed close to the original CLIP, whereas TPT requires ~35 forward passes per sample.

## Highlights & Insights
- **The integration of online EM and VLM priors** is the most elegant design of this work: initializing means with text embeddings solves the "unlabeled" challenge, online updating solves the "storage-free" requirement, and entropy weighting resolves the "instability" issue—three challenges are gracefully solved within a unified framework.
- The strategy of **using statistical models to replace neural network optimization** is highly inspiring: complex TTA problems are modeled as simple GMM parameter estimation, avoiding the instability and computational overhead of gradient optimization.
- **Rethinking TTA from a data distribution perspective** rather than model adaptation provides a brand-new paradigm—future work could explore more complex distribution assumptions (such as the von Mises-Fisher distribution, which is more suitable for normalized CLIP features).

## Limitations & Future Work
- The assumption of Gaussian distributions is overly simplified: class-conditional distributions in the CLIP feature space are not necessarily Gaussian, especially when class overlap is large.
- The shared covariance matrix assumption limits expressiveness: different classes may have vastly different distribution structures/shapes.
- Online updates are sensitive to the sequence of incoming samples; skewed initial sample distributions may bias subsequent parameter estimation.
- Hyperparameters $\alpha$ (mixture weight for CLIP and GDA logits) and $\beta$ (entropy weighting strength) require tuning.
- The online update of the covariance matrix becomes computationally complex when the number of classes is large (e.g., 1000 classes in ImageNet).

## Related Work & Insights
- **vs TPT/DiffTPT**: These methods optimize prompts through gradient backpropagation, which is computationally expensive and unstable. FreeTTA requires no training, achieving fast speed and stability.
- **vs TDA**: TDA caches historical test samples for KNN classification, requiring data storage without distribution modeling. FreeTTA only maintains statistical utilities and does not store any samples.
- **vs PromptAlign**: PromptAlign requires access to source-domain data to align distributions, whereas FreeTTA does not require any auxiliary source data.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of online EM into VLM TTA is highly novel, though GMM+EM itself is a classic method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 datasets, two settings (cross-domain and OOD), multiple backbones, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations and clean framework diagrams, though some notations are somewhat redundant.
- Value: ⭐⭐⭐⭐ The method is simple and efficient, but its broader impact depends on the practical demand for TTA during deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[CVPR 2026\] Condensed Test-Time Adaptation of VLMs for Action Recognition](../../CVPR2026/multimodal_vlm/condensed_test-time_adaptation_of_vlms_for_action_recognition.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[ICLR 2026\] Flatness-Guided Test-Time Adaptation for Vision-Language Models](../../ICLR2026/multimodal_vlm/flatness_guided_test-time_adaptation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
