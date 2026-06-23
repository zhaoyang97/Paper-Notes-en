---
title: >-
  [Paper Note] PRISM: Progressive Robust Learning for Open-World Continual Category Discovery
description: >-
  [ICLR 2026][Self-Supervised Learning][Paper Note] PRISM proposes "Open-World Continual Category Discovery" (OW-CCD), a more realistic setting where data streams contain both new categories and domain shifts. By utilizing a "High-frequency Categorical Shunting + Sparse Assignment Matching + Invariant Knowledge Transfer" toolkit, it consistently achieves new CCD SOTA on
tags:
  - ICLR 2026
  - Self-Supervised Learning
date: 2026-05-08
content_hash: 005383f84d4e8126
---
# PRISM: Progressive Robust Learning for Open-World Continual Category Discovery

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5JwUWsewWH](https://openreview.net/forum?id=5JwUWsewWH)  
**Area**: Self-Supervised / Category Discovery  
**Keywords**: Continual Category Discovery, Open-world, Domain Shift, High-frequency Spectrum, Optimal Transport

## TL;DR
PRISM proposes "Open-World Continual Category Discovery" (OW-CCD), a more realistic setting where data streams contain both new categories and domain shifts. By utilizing a "High-frequency Categorical Shunting + Sparse Assignment Matching + Invariant Knowledge Transfer" toolkit, it consistently achieves new CCD SOTA on SSB-C and DomainNet (with a 15.1% gain on the clean domain of CUB-C).

## Background & Motivation
**Background**: Category Discovery aims to automatically discover new concepts in unlabeled data using knowledge from labeled known classes. From NCD (all unlabeled are new classes) to GCD (unlabeled is a mix of known and unknown), and then to CCD (Continual Category Discovery, incorporating continual learning), the task becomes increasingly realistic—the model must continuously discover new classes in an incoming unlabeled data stream while not forgetting old ones.

**Limitations of Prior Work**: Existing CCD methods almost all default to an implicit assumption—**the data at each stage comes from a single fixed domain, and the entire stream is a stationary distribution**. This rarely holds in open environments: an online platform continuously receives images of animals from different cameras, users, and styles/lighting, where domains change as rare species emerge. Once domain shift exists, existing CCD methods fail to maintain recognition of known classes and struggle to reliably discover new ones.

**Key Challenge**: Directly applying Domain Adaptation (DA) to align distributions is ineffective. Traditional DA assumes overlapping label spaces between source and target, whereas the target stream here contains many unknown new classes; **naive alignment causes negative transfer, erasing new class signals** and suppressing discovery. Moreover, most DA focuses on aligning known classes and provides little guidance on how to explore unknown label spaces.

**Key Insight**: The authors leverage **spectral analysis** and observe a useful phenomenon—**high-frequency components** of images carry more domain-invariant global semantics (structure, shape), while **low-frequency components** encode domain-related style details (tone, texture). Thus, the determination of "known vs. unknown" can be grounded: high-frequency features are more robust to domain shifts and are better suited for separating known classes from unknown ones.

**Core Idea**: Replace "align then discover" with **divide and conquer**—first separate the data stream into known and unknown subsets using high-frequency information. The known subset is assigned reliable pseudo-labels using sparse optimal transport, while the unknown subset transfers semantic knowledge from known classes by "maintaining category relationship rankings across domains," enabling stable discovery of new classes under domain shift.

## Method

### Overall Architecture
PRISM addresses the following: **an unlabeled data stream with domain shifts arrives stage-by-stage, containing both old known classes and new unknown classes, and the model must identify known classes and discover new ones online**. The overall paradigm consists of "one base pre-training + T rounds of online discovery." In the base stage, a feature extractor $f$ and classifier head $g$ are pre-trained on labeled data. In each online discovery round, the new unlabeled stream $D^u_t$ is processed by three serial modules: **HCS (High-frequency Categorical Shunting)** splits the stream into "known-like" and "unknown-like" subsets; the known subset is passed to **SAM (Sparse Assignment Matching)** to obtain pseudo-labels via proximal optimal transport; the unknown subset is passed to **IKT (Invariant Knowledge Transfer)** to maintain relationship rankings with known class prototypes under cross-domain perturbations, followed by Affinity Propagation for clustering. Finally, "known pseudo-labels + new clusters" are merged to incrementally update the model and expand the online classifier.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Unlabeled Stream D^u_t<br/>(Old Known + New Unknown + Domain Shift)"] --> B["HCS (High-frequency Categorical Shunting)<br/>Extract HF Features → Density Scoring → GMM Bipartition"]
    B -->|π(x)≥0.5 Known Subset| C["SAM (Sparse Assignment Matching)<br/>Proximal OT for Reliable Pseudo-labels"]
    B -->|π(x)<0.5 Unknown Subset| D["IKT (Invariant Knowledge Transfer)<br/>Cross-domain Ranking Consistency → AP Clustering"]
    C --> E["Merge Pseudo-labels + New Clusters<br/>Incremental CE Update + Expand Classifier"]
    D --> E
    E --> F["Output: Recognize Old + Discover New"]
```

### Key Designs

**1. HCS: Separating Known/Unknown via Domain-Invariant High-Frequency Information**

This step addresses the difficulty of identifying known vs. unknown samples under domain shift. Instead of direct alignment, frequency domain decomposition is performed: the 2D Discrete Fourier Transform $F(x_i)$ of input $x_i$ is masked by a square mask $M$ with radius $r$ to separate low-frequency $F_l = M \odot F(x_i)$ and high-frequency $F_h = (I-M) \odot F(x_i)$. The inverse transform yields the high-frequency image $x_i^h$. A density score is defined as:

$$S(x) = \nu\!\left(\max_c \frac{f(x^h)\cdot e_c}{\|f(x^h)\|\,\|e_c\|}\right),$$

where $e_c$ is the prototype of known class $c$ from the previous stage, and $\nu(\cdot)$ normalizes the score to $[0,1]$ via min-max. The intuition is that a higher $S(x)$ indicates the high-frequency representation is closer to a known prototype. $S(x)$ follows a **bimodal distribution**, which is fitted using a two-component Gaussian Mixture Model $P(x)=\pi(x)\,N(x|\mu_{kno},\sigma^2_{kno})+(1-\pi(x))\,N(x|\mu_{unk},\sigma^2_{unk})$. The posterior $\pi(x)$ is estimated via EM, and the stream is split into $D^u_{t,kno}$ and $D^u_{t,unk}$ based on $\pi(x)\ge 0.5$.

**2. SAM: Reliable Pseudo-labels for Known Classes via Sparse Optimal Transport**

The known subset $x_{kno}$ and the previous prototypes share a semantic space. Optimal Transport (OT) can align samples to prototypes to mitigate domain differences. To avoid the blurry transport plans caused by standard entropy regularization, SAM replaces it with an $\ell_2$ proximal term:

$$\min_{\gamma\in\Delta}\ \sum_{i}\sum_{j}\Big[\gamma_{ij}C_{ij}+\tfrac{\varepsilon}{2}\big(\gamma_{ij}-\gamma^{(l)}_{ij}\big)^2\Big],$$

where cost $C_{ij}=-\log\big(g(f(x_{i,kno}))_j\big)$ and the proximal term $\tfrac{\varepsilon}{2}\sum(\gamma_{ij}-\gamma^{(l)}_{ij})^2$ suppresses oscillations and encourages sparse, stable solutions. The dual problem is solved efficiently for high-quality pseudo-labels.

**3. IKT: Robust Discovery via Cross-Domain Category Relationship Ranking Consistency**

Category discovery relies on transferring knowledge from known classes to unknown ones via semantic correlation. IKT ensures that only **domain-invariant category relationships** are preserved. For an unknown sample, style-transferred versions $\hat x^t_{i,unk}$ are created by perturbing low-frequency statistics $\mu(F_l),\sigma(F_l)$ while keeping high-frequency content. The cosine similarity between samples and known prototypes $e^{t-1}_c$ is mapped to Plackett-Luce (PL) model parameters $\kappa_{i,c}=\exp(\cos(z,e_c))$. The ranking is enforced to be cross-domain consistent via a KL divergence loss on the factorized PL likelihood:

$$L_{rank}=\frac{1}{N_{t,unk}}\sum_i \ell_{KL}\big(P(\cdot|\kappa_i),P(\cdot|\hat\kappa_i)\big).$$

This ensures the model preserves the global relative ranking of unknown samples against known prototypes, facilitating stable knowledge transfer.

### Loss & Training
During the online stage, SAM pseudo-labels are used for known samples, and Affinity Propagation (non-parametric clustering) is used for unknown samples to expand the classifier. The model is updated incrementally via cross-entropy without replaying historical data. The total objective is $L_{total}=L_{ce}+\lambda_1 L_{rank}$. The backbone is DINO pre-trained ViT-B/16, fine-tuning only the last transformer block for 30 epochs per stage, with $\lambda_1=1$, $T=3$, cross-frequency ratio $r=0.3$, and proximal strength $\varepsilon=0.5$.

## Key Experimental Results

### Main Results
Evaluation is performed on SSB-C (semantic shift benchmark with 9 corruptions) and DomainNet (6 domains). The primary metric is continual clustering accuracy (cACC), split by All/Old/New.

| Benchmark / Task | Metric(All) | PRISM | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| CUB-C Clean Domain | cACC | 49.3 | 34.2 (VB-CGCD) | +15.1 |
| CUB-C Corrupted Domain | cACC | 44.0 | 31.7 (VB-CGCD) | +12.3 |
| DomainNet Real→Painting (Real) | cACC | 60.9 | 57.3 (VB-CGCD) | +3.6 |
| DomainNet Real→Painting (Painting) | cACC | 39.2 | 32.4 (VB-CGCD) | +6.8 |

PRISM significantly outperforms existing CCD baselines (G&M, Happy, PA-CGCD, DEAN, PromptCCD, VB-CGCD) and re-implemented GCD methods across all tasks.

### Ablation Study
Component-level ablation (Real→Painting, reporting All):

| HCS | SAM | IKT | Real(All) | Painting(All) | Description |
|------|------|------|------|------|------|
| ✗ | ✗ | ✗ | 54.6 | 28.7 | Baseline, fails under domain shift |
| ✓ | ✓ | ✗ | 58.1 | 35.0 | Improved known class recognition |
| ✓ | ✗ | ✓ | 56.9 | 33.2 | Improved new class discovery |
| ✓ | ✓ | ✓ | 60.9 | 39.2 | Full model is best |

Shunting strategy comparison: HCS (60.9/39.2) outperforms original image (55.0/29.6), entropy (54.4/29.9), and energy (55.8/30.6).

### Key Findings
- HCS+SAM primarily improves **known class** accuracy, while IKT enhances **new class** discovery (New metric from 49.9 to 55.1).
- The choice of features for shunting is critical: high-frequency information filters style noise better than original images or entropy/energy.
- Higher relative gains are observed in domains with larger shifts (e.g., Painting/Sketch), validating PRISM's robustness.

## Highlights & Insights
- **Divide and Conquer via Spectrum**: High-frequency information is used both to judge known/unknown (HCS) and to preserve semantic structure during style perturbation (IKT).
- **Category Relationships as Ranking Invariants**: Instead of aligning absolute features, IKT maintains the relative ranking of unknown samples against known prototypes, bypassing negative transfer and combinatorial complexity.
- **Sparse OT vs. Entropy Regularization**: Using an $\ell_2$ proximal term provides sparser and more reliable pseudo-labels compared to the Sinkhorn algorithm.

## Limitations & Future Work
- The method relies on the empirical assumption that "high-frequency = domain-invariant." Its performance in scenarios where domain differences exist in high frequencies (e.g., medical modalities) remains unexplored.
- The pipeline is serial and complex (DFT, GMM-EM, proximal OT, PL ranking, AP clustering); efficiency comparisons with lightweight baselines are missing.
- Hyperparameters like $r$ and $\varepsilon$ are fixed; sensitivity to $T \gg 3$ requires further validation.

## Related Work & Insights
- **vs. GCD / SimGCD**: These assume a single stationary distribution. PRISM addresses OW-CCD with streaming data and explicit domain-invariance.
- **vs. CCD Baselines**: Previous methods assume a single-domain stream; PRISM explicitly incorporates domain shift robustness.
- **vs. Domain Adaptation / OSDA**: While DA targets overlapping labels and can cause negative transfer for new classes, PRISM preserves cross-domain structure without suppressing new class signals.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose OW-CCD setting with a unified spectral perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong results on dual benchmarks, though lacking long-stream analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, though the multi-component method is dense.
- Value: ⭐⭐⭐⭐ Advances CCD to realistic open-world settings with reusable components.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond the Static World: Continual Category Discovery under Visual Drift](../../CVPR2026/self_supervised/beyond_the_static_world_continual_category_discovery_under_visual_drift.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[ECCV 2024\] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](../../ECCV2024/self_supervised/promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)
- [\[CVPR 2026\] Seeing Through the Shift: Causality-Inspired Robust Generalized Category Discovery](../../CVPR2026/self_supervised/seeing_through_the_shift_causality-inspired_robust_generalized_category_discover.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)

</div>

<!-- RELATED:END -->
