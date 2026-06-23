---
title: >-
  [Paper Note] Exposing Mixture and Annotating Confusion for Active Universal Test-Time Adaptation
description: >-
  [ICLR 2026][Others][Universal Test-Time Adaptation] This paper proposes a new paradigm of Active Universal Test-Time Adaptation (AUTTA) and introduces the EMAC method to incorporate sparse human annotations during testing. It decouples domain and class shifts using SVD + GMM to expose samples in the "mixed region," employs a reward-driven strategy to select representati
tags:
  - ICLR 2026
  - Others
  - Universal Test-Time Adaptation
  - Active Learning
date: 2026-05-08
content_hash: d5c0d6a8d962adfb
---
# Exposing Mixture and Annotating Confusion for Active Universal Test-Time Adaptation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=89gxHJkCXk](https://openreview.net/forum?id=89gxHJkCXk)  
**Code**: To be confirmed  
**Area**: Test-Time Adaptation / Active Learning / Transfer Learning  
**Keywords**: Universal Test-Time Adaptation, Active Learning, Dual Shift, Gaussian Mixture Model, Open-Set Recognition  

## TL;DR
This paper proposes a new paradigm of Active Universal Test-Time Adaptation (AUTTA) and introduces the EMAC method to incorporate sparse human annotations during testing. It decouples domain and class shifts using SVD + GMM to expose samples in the "mixed region," employs a reward-driven strategy to select representative samples for annotation, and utilizes a clustering contrastive loss to balance annotations and pseudo-labels, achieving SOTA performance under dual shifts.

## Background & Motivation
**Background**: Universal Test-Time Adaptation (UTTA) aims to handle both **domain shift** (style/corruption changes) and **class shift** (emergence of unseen classes) in an open-world setting where source data is unavailable and data arrives in a stream. The goal is to accurately classify known classes while rejecting new classes. Existing UTTA methods (e.g., OPTTT, TTAC) rely primarily on pseudo-labels and heuristic rules.

**Limitations of Prior Work**: When domain and class shifts **occur simultaneously**, the error rate of pseudo-labels rises sharply, causing self-training to collapse. While introducing active learning for human intervention is intuitive, analysis shows (Fig. 1) that traditional active learning based on entropy/uncertainty tends to select samples with **strong class shifts**, which yield low annotation gains and introduce training bias.

**Key Challenge**: High-value samples reside in the **"mixed region"** where domain and class shifts overlap. Here, pseudo-label error rates are highest (exp. shows ~70-80% in mixed regions vs ~12-15% in pure shift regions), yet existing active learning methods fail to cover this area, leading to wasted annotation budgets.

**Goal**: Introduce sparse human annotations into the UTTA framework and design a mechanism to precisely locate and annotate mixed-region samples to maximize the utility of the limited budget.

**Core Idea**: **[Expose Mixture + Annotate Confusion]** Explicitly expose samples in the dual-shift mixed region to form a candidate pool, select the most informative representatives using reward signals, and integrate scarce annotations with pseudo-labels using contrastive clustering.

## Method

### Overall Architecture
EMAC (Exposing Mixture and Annotating Confusion) is a coarse-to-fine two-stage annotation and one-step optimization pipeline. **Step 1 (Coarse Filtering)** uses SVD to decouple classifier parameters into "known/unknown" subspaces and fits a GMM on unknown energy to expose candidates in the mixed region. **Step 2 (Fine Selection)** employs a reward-driven Max-Min Entropy strategy to select representative samples that contribute most to distinguishing new/old classes for oracle annotation. **Step 3 (Optimization)** uses a clustering contrastive loss to joint-optimize annotations and high-confidence pseudo-labels, mitigating decision boundary blurring caused by annotation scarcity.

```mermaid
flowchart LR
    A[Target batch features z] --> B[SVD decoupled classifier<br/>Known/Unknown Subspace]
    B --> C[GMM modeling unknown energy<br/>Bimodal distribution]
    C --> D[Expose confusion candidates Xmix<br/>+ Old-class region Xold / New-class region Xnew]
    D --> E[Reward-driven selection<br/>Max-Min Entropy + EMA]
    E --> F[Oracle labels representative samples → DLT]
    F --> G[Clustering contrastive loss<br/>Balancing labels & pseudo-labels]
    G --> H[Test-time model update]
```

### Key Designs

**1. Exposing Mixed Candidates: SVD Decoupling + GMM Bimodal Modeling**
Since source data is unavailable, source knowledge is extracted from classifier weights $W_{cls} \in \mathbb{R}^{C \times D}$. Singular Value Decomposition performs orthogonal decomposition $W_{cls} = F_{known}\Sigma F_{unknown}^\top$ to separate "known" basis vectors $F_{known}$ from "unknown" ones $F_{unknown}$. Target features $z$ are projected onto these subspaces as $z_{known}$ and $z_{unknown}$, satisfying $\|z_{known}\|_2^2 + \|z_{unknown}\|_2^2 = 1$. A key observation is that the distribution of unknown energy $\|z_{unknown}\|_2^2$ is **bimodal**: low means for old classes and high means for new classes. A GMM fits this: $p(z_{unknown}) = \pi\mathcal{N}(\mu_{old}, \sigma_{old}^2) + (1-\pi)\mathcal{N}(\mu_{new}, \sigma_{new}^2)$. Samples are partitioned into $X_{old}$ (below $\mu_{old}$), $X_{new}$ (above $\mu_{new}$), and $X_{mix}$ (between). Annotating $X_{mix}$ avoids wasting budget on fringe areas, as the error rate in $X_{mix}$ (~78-83%) is much higher than in $X_{old}/X_{new}$ (~10-14%).

**2. Reward-Driven Selection: Picking Representatives via Marginal Info Gain**
To avoid distribution bias in the candidate pool, a Max-Min Entropy objective $L_{MME} = \Gamma_{old} + \Gamma_{new}$ quantifies marginal information gain. $\Gamma_{old} = \sum_i H(f(X_{old}, \theta_t)) - H(f(X_{old}, \theta_{t-1}))$ measures the confidence boost (entropy reduction) for old classes, while $\Gamma_{new} = \sum_i H(f(X_{new}, \theta_{t-1})) - H(f(X_{new}, \theta_t))$ measures improved rejection of new classes. Weighted average rewards $R'_{old}, R'_{new}$ (balanced by $\omega_{old}, \omega_{new}$) are smoothed via EMA $R_{old} = \alpha R'_{old,t} + (1-\alpha)R'_{old,t-1}$. If $R_{old} > R_{new}$, the **highest-entropy** old-class sample is chosen; otherwise, the **lowest-entropy** new-class sample is chosen.

**3. Balancing Labels & Pseudo-labels: Clustering Contrastive Optimization**
Clustering contrastive loss $L_c$ uses high-confidence pseudo-label features $F_p$ and annotated features $F_a$ to compute class prototypes $p_c = \frac{1}{|F_p||F_a|}(\sum_{u_i \in F_p} u_i + \sum_{v_i \in F_a} v_i)$. The loss is defined as $L_c = \sum_{i \in I_{old}} \frac{-1}{|Q(i)|}\sum_{p \in Q(i)} \log\frac{\exp(s_{ip})}{S(i)}$, where $N(i) = I_{new} \cup I_{old}$ and $s_{ij}$ is cosine similarity. This objective minimizes intra-class distance by pulling samples toward prototypes and maximizes inter-class distance by pushing new-class samples away. The total objective is $L = L_{MME} + L_c$.

## Key Experimental Results

### Main Results (DomainNet, higher AH is better)

| Category | Method | Avg. AH | GPU(s) |
|------|------|---------|--------|
| TTA | TEST | 43.7 | 392 |
| TTA | TENT | 46.1 | 479 |
| TTA | SHOT | 46.0 | 564 |
| UTTA | TTAC | 46.7 | 651 |
| UTTA | OPTTT | 49.8 | 697 |
| ATTA | SimATTA | 47.1 | 747 |
| ATTA | EATTA | 47.7 | 738 |
| ATTA | BiTTA | 47.2 | 687 |
| **AUTTA** | **Ours (EMAC)** | **52.2** | 735 |
| **AUTTA** | **EMAC\*** | **53.1** | 779 |

EMAC outperforms the strongest baseline OPTTT by 2.4 points. On VisDA-C, EMAC achieves AH scores of 79.8/77.4/72.3 for NOISE/MNIST/SVHN shifts, leading significantly over OPTTT (77.8/75.2/69.2).

### Active Learning Comparison (using TENT framework, B=Budget)

| Method | DomainNet Avg. | VisDA-C |
|------|----------------|---------|
| Random (B=1000) | 46.8 | 66.7 |
| Entropy (B=1000) | 46.3 | 65.4 |
| Coreset (B=1000) | 49.4 | 68.7 |
| BADGE (B=1000) | 49.5 | 68.2 |
| SimATTA (B=1000) | 47.1 | 67.9 |
| **EMAC (B=800)** | **50.8** | **73.1** |
| **EMAC (B=1000)** | **52.2** | **76.5** |
| **EMAC (B=1500)** | **53.1** | **78.2** |

EMAC with a budget of 800 outperforms other methods using 1000, demonstrating high annotation efficiency.

### Ablation Study (EMSC=Mix Exposure / SC=Selection / BTPO=Bal. Opt.)

| EMSC | SC | BTPO | DomainNet AH | VisDA-C AH |
|------|-----|------|--------------|------------|
| ✓ | - | - | 43.7 | 65.0 |
| - | ✓ | - | 47.1 | 69.4 |
| - | - | ✓ | 47.9 | 71.2 |
| ✓ | ✓ | - | 49.1 | 71.7 |
| - | ✓ | ✓ | 50.8 | 73.1 |
| ✓ | ✓ | ✓ | **52.2** | **76.5** |

## Key Findings
- **Mixed Region Effectiveness**: The error rate of $X_{mix}$ via GMM (~78-83%) is significantly higher than $X_{old}/X_{new}$ (~10-14%), proving that annotating samples in the mixed region provides the highest gain.
- **Complementarity**: All three modules are required for optimal performance, with BTPO (Balanced Optimization) contributing most individually.
- **Robustness to Small Batches**: Simple GMM is unstable for batch size $\le 8$. Using a sliding window and EMA smoothing improved batch=1 AH from 16.3 to 47.3.

## Highlights & Insights
- **Shift from Uncertainty to Confusion**: The core insight is that "uncertainty $\neq$ annotation value" under dual shifts. High-value samples reside in the class-overlap mixed region, which traditional entropy sampling tends to skip.
- **Source-free Localization via SVD**: Decoupling the classifier space into known/unknown bases without source data is a lightweight yet clever engineering design.
- **Theoretically Grounded Selection**: Linking Max-Min Entropy rewards to information gain/generalization error bounds provides more rigor than purely heuristic scoring.

## Limitations & Future Work
- **GMM Bimodal Assumption**: Relies on $\|z_{unknown}\|^2$ exhibiting clean bimodality. In cases of extreme class ratios, the distribution may collapse into a unimodal peak, degrading GMM accuracy.
- **Labeling Latency**: As a human-in-the-loop approach, applicability in real-time streaming where oracle feedback is delayed or costly remains a challenge.
- **Task Scope**: Evaluation is restricted to image classification on DA benchmarks; scalability to complex tasks like detection or segmentation is unexplored.
- **Hyperparameter Sensitivity**: Parameters like $\omega_{old}, \omega_{new}, \alpha$ require tuning, which is difficult in label-free test environments.

## Related Work & Insights
- **UTTA / Open-set TTA**: OPTTT and TTAC model class shifts/distribution alignment; this work introduces active labels to break the pseudo-label ceiling.
- **Active TTA (ATTA)**: SimATTA, EATTA, and BiTTA use confidence scores, which are unreliable under dual shifts. EMAC addresses this failure point.
- **Active Learning**: Traditional AL (Coreset, BADGE) ignores class shifts. This paper reveals their failure modes in open-world dual-shift scenarios.
- **Inspiration**: The coarse-to-fine "expose then select" paradigm is applicable to other budget-constrained open-world tasks like Continual Learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Proposes the AUTTA paradigm. The insight on the "mixed region" combined with SVD/GMM localization is fresh, though components have some combinational elements.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on DA datasets with various shifts. Includes budget, ablation, and visualization studies. Limited to classification.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation (Fig. 1) is clear and compelling. Progressive methodological explanation is well-structured.
- **Value**: ⭐⭐⭐⭐ Significant for open-world scenarios requiring human intervention (e.g., autonomous driving). High annotation efficiency is practically valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](prior-free_tabular_test-time_adaptation.md)
- [\[ICLR 2026\] PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference](priorguide_test-time_prior_adaptation_for_simulation-based_inference.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](../../CVPR2025/others/effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](../../ICML2026/others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)

</div>

<!-- RELATED:END -->
