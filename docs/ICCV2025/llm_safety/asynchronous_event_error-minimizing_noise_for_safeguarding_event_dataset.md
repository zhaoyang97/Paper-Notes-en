---
title: >-
  [Paper Note] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset
description: >-
  [ICCV 2025][LLM Safety][Unlearnable Examples] This paper proposes UEvs, the first unlearnable example generation method for asynchronous event data. It introduces Event Error-Minimizing Noise (E²MN) and an adaptive proje…
tags:
  - "ICCV 2025"
  - "LLM Safety"
  - "Unlearnable Examples"
  - "Event Camera"
  - "Dataset Protection"
  - "Error-Minimizing Noise"
  - "Asynchronous Event Stream"
date: 2026-05-08
content_hash: 4bf66fd648ab1abc
---

# Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset

**Conference**: ICCV 2025
**arXiv**: [2507.05728](https://arxiv.org/abs/2507.05728)
**Code**: [https://github.com/rfww/uevs](https://github.com/rfww/uevs)
**Area**: Data Security / Event Cameras
**Keywords**: Unlearnable Examples, Event Camera, Dataset Protection, Error-Minimizing Noise, Asynchronous Event Stream

## TL;DR

This paper proposes UEvs, the first unlearnable example generation method for asynchronous event data. It introduces Event Error-Minimizing Noise (E²MN) and an adaptive projection mechanism that prevent unauthorized models from learning from event datasets while preserving utility for legitimate use.

## Background & Motivation

As event camera datasets (e.g., N-Caltech101, DVS128 Gesture) become widely available, the risk of unauthorized use has grown substantially. In the image domain, Unlearnable Examples (UEs) embed imperceptible noise to prevent models from learning true semantic features. However, **directly applying image-based UEs to event data is infeasible** for three reasons:

**Binary polarity constraint**: Event polarities are restricted to $\pm 1$, forming a highly discrete space incompatible with conventional continuous noise.

**Asynchronous spatiotemporal structure**: Event streams are sparse spatiotemporal point clouds $(x, y, t, p)$, not regular 2D grids.

**Representation conversion gap**: Events must first be converted into event stacks before being fed into DNNs, yet noise injected into the stack cannot be directly mapped back to the event stream.

Data corruption approaches (e.g., coordinate shift, polarity flipping) can degrade data quality but are easily neutralized by data augmentation and offer unreliable protection.

## Method

### Overall Architecture

The UEvs pipeline proceeds as follows: event stream → event stack conversion → E²MN generation via surrogate model → adaptive projection and sparsification → unlearnable event stack → retrieval-based reconstruction of the unlearnable event stream.

### Key Designs

#### 1. Event Stack Representation

The event stream is divided into $C=16$ temporal bins. Each pixel value in a bin takes one of three values:
- $0$: event with polarity $p=-1$
- $0.5$: no event
- $1.0$: event with polarity $p=+1$

Using a large number of channels ($C=16$) avoids the event-overwriting problem that arises with single- or three-channel representations.

#### 2. Event Error-Minimizing Noise (E²MN)

The core optimization objective follows a min-min bi-level formulation:
$$\arg\min_\theta \mathbb{E}_{(\mathcal{E},l) \in \mathcal{D}} [\min_\delta \mathcal{L}(f'_\theta(\mathcal{R}(\mathcal{E}) + \delta), l)] \quad \text{s.t.} \|\delta\|_\infty \leq \epsilon$$

**Inner optimization**: Find noise $\delta$ under an $L_\infty$ constraint that minimizes the surrogate model's loss (via PGD).
**Outer optimization**: Update surrogate model parameters to minimize classification loss.

Two noise variants are proposed:
- **Sample-wise noise**: A unique noise perturbation per sample; strongest protection.
- **Class-wise noise**: Shared noise within each class; more efficient and generalizable to new data.

A similarity regularization term is added to the outer loss:
$$\mathcal{L}^* = \lambda_1 \mathcal{L} + \lambda_2 \mathcal{L}_s$$
where $\mathcal{L}_s$ maximizes the discrepancy between clean and unlearnable features, ensuring noise remains effective after projection.

#### 3. Adaptive Projection Mechanism

Continuous noise is projected onto $\{-0.5, 0, +0.5\}$ to ensure compatibility with the event stack representation:
$$\mathbf{P}(\delta) = \begin{cases} -0.5, & \delta_{i,j} < \mu - \tau \times \pi \\ 0.0, & \mu - \tau \times \pi \leq \delta_{i,j} \leq \mu + \tau \times \pi \\ +0.5, & \delta_{i,j} > \mu + \tau \times \pi \end{cases}$$

The parameter $\tau$ balances effectiveness and imperceptibility: larger $\tau$ yields more covert noise with weaker protection, while smaller $\tau$ provides stronger protection at the cost of increased noise visibility.

The semantic effect of the projected noise (confusion matrix):
- $+0.5$ added to an empty position → generates a new event
- $-0.5$ added to an event position → removes the event
- Added to a same-polarity position → no change

#### 4. Event Stream Retrieval and Reconstruction

Reconstructing the unlearnable event stream from the unlearnable event stack proceeds as follows:
- **Original events**: Retrieved from the original stream with their corresponding timestamps.
- **Newly inserted events**: Assigned adaptive timestamps initialized within their corresponding temporal bin $\Delta_t$.

### Loss & Training

- Surrogate model: ResNet18 (SGD, lr=1e-4, momentum=0.9)
- Noise generation: PGD with 10 steps, $\epsilon=0.5$, step size $\alpha=0.8/255$
- Projection balance parameter: $\tau=3/4$
- Termination criterion: surrogate model classification accuracy >99%
- Training constraint: surrogate model trained for only $M=10$ iterations per epoch to prevent over-fitting to true features

## Key Experimental Results

### Main Results (Test accuracy % of various DNNs on unlearnable datasets)

| Noise Type | Model | N-Caltech101 | CIFAR10-DVS | DVS128 Gesture | N-ImageNet |
|---|---|:---:|:---:|:---:|:---:|
| Clean | RN18 | 78.32 | 65.19 | 74.14 | 56.60 |
| CS (Coord. Shift) | RN18 | 50.43 (-27.8) | 46.80 (-18.4) | 75.35 (+1.2) | 42.60 (-14.0) |
| PI (Polarity Inv.) | RN18 | 40.78 (-37.5) | 22.72 (-42.5) | 31.94 (-42.2) | 41.80 (-4.8) |
| **Class-wise** | **RN18** | **1.90 (-76.6)** | **22.33 (-42.9)** | **14.93 (-59.4)** | **10.00 (-46.8)** |
| **Sample-wise** | **RN18** | **0.52 (-77.6)** | **15.51 (-49.6)** | **14.54 (-59.4)** | **10.20 (-46.2)** |
| **Class-wise** | **Swin_B** | **0.52 (-90.2)** | **29.03 (-46.2)** | **28.12 (-55.2)** | **10.00 (-62.0)** |

### Ablation Study (N-Caltech101, sample-wise noise)

| Setting | RN18 | RN50 | VG16 | DN121 | ViT_B | Swin_B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| E1 (Full method) | 0.52 | 6.09 | 14.13 | 11.49 | 1.09 | 21.14 |
| E2 (w/o similarity loss) | 3.85 | 15.22 | 38.31 | 10.17 | 37.51 | 27.51 |
| E3 (mixed $\Delta_c \lor \Delta_s$) | 5.17 | 5.40 | 22.23 | 6.15 | 1.21 | 16.66 |
| E5 (w/ data augmentation) | 13.04 | 8.04 | 14.99 | 10.74 | 9.88 | 25.73 |
| E6 (FGSM instead of PGD) | 27.74 | 33.08 | 44.28 | 34.92 | 8.85 | 24.41 |

### Key Findings

1. On N-Caltech101, UEvs reduces Swin_B accuracy from **90.70% to 0.52%** (class-wise), a drop exceeding 90 percentage points.
2. Strong cross-architecture transferability: a ResNet18 surrogate model yields effective protection across 7 diverse architectures, including ViT and Swin.
3. Satisfactory perceptual quality: PSNR of 20.36 (class-wise) / 18.22 (sample-wise); SSIM of 0.791 / 0.571.
4. The similarity loss $\mathcal{L}_s$ is critical for preserving effectiveness after projection (removing it causes VG16 accuracy to recover from 14.13% to 38.31%).
5. UEvs demonstrates moderate robustness against common data augmentations (random crop/flip/EventDrop).

## Highlights & Insights

- **Gap-filling contribution**: UEvs is the first work to extend the unlearnable examples paradigm to asynchronous event data, addressing the unique challenges of binary polarity, sparsity, and timestamp reconstruction.
- **Elegant projection design**: Continuous noise is mapped to ternary values $\{-0.5, 0, +0.5\}$, directly corresponding to event deletion, no-op, and event insertion — a perfect correspondence with the physical event generation model.
- The **hybrid noise strategy** ($\Delta_c \lor \Delta_s$) offers a flexible solution for practical deployment scenarios.

## Limitations & Future Work

- Evaluation is limited to classification; transferability to other event-based vision tasks such as detection and segmentation remains unexplored.
- Memory consumption of sample-wise noise scales poorly with dataset size.
- In-depth evaluation under adversarial training defenses is lacking.
- An adaptive selection mechanism for the projection parameter $\tau$ has yet to be developed.
- The timestamp quality in the reconstructed event stream may affect downstream temporal tasks.

## Related Work & Insights

- The method inherits the core idea of UEs from Huang et al. (ICLR 2021) while addressing the unique challenges of event data.
- It forms an attack-defense duality with event-based backdoor attack works such as EventTrojan.
- Future directions are suggested toward leveraging generative models (e.g., noise generators) to improve the generation efficiency of sample-wise noise.

## Rating

- Novelty: ⭐⭐⭐⭐ (pioneers dataset protection for event camera data)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 datasets, 7 models, multiple baselines and ablations)
- Writing Quality: ⭐⭐⭐⭐ (clear pipeline; occasional inconsistencies in mathematical notation)
- Value: ⭐⭐⭐⭐ (opens a new research direction in event data security)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](../../NeurIPS2025/llm_safety/masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](../../NeurIPS2025/llm_safety/cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [\[NeurIPS 2025\] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](../../NeurIPS2025/llm_safety/poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)
- [\[NeurIPS 2025\] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](../../NeurIPS2025/llm_safety/enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)

</div>

<!-- RELATED:END -->
