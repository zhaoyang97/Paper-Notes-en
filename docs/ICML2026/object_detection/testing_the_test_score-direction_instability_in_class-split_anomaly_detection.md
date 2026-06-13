---
title: >-
  [Paper Note] Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection
description: >-
  [ICML2026][Object Detection][Anomaly Detection] The authors argue that "class-split" anomaly detection benchmarks are ill-posed when anomaly classes and normal mixture distributions overlap in the representation space—AU…
tags:
  - "ICML2026"
  - "Object Detection"
  - "Anomaly Detection"
  - "OOD Detection"
  - "Class-Split Benchmark"
  - "AUROC Reversal"
  - "Score-Direction Instability"
  - "Neighborhood Class Leakage"
date: 2026-05-08
content_hash: cda1737d1acb90b3
---

# Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection

**Conference**: ICML2026  
**arXiv**: [2606.02601](https://arxiv.org/abs/2606.02601)  
**Code**: None  
**Area**: AI Safety / Anomaly Detection / OOD Evaluation  
**Keywords**: Anomaly Detection, OOD Detection, Class-Split Benchmark, AUROC Reversal, Score-Direction Instability, Neighborhood Class Leakage

## TL;DR
The authors argue that "class-split" anomaly detection benchmarks are ill-posed when anomaly classes and normal mixture distributions overlap in the representation space—AUROC collapses to random or even reverses, with the direction depending on the unknown anomaly class. A training-free "neighborhood class leakage" metric $L_k$ is proposed to diagnose benchmark failure before evaluation.

## Background & Motivation
**Background**: Two mainstream protocols exist for evaluating completely unconditional OOD Anomaly Detection (AD): (i) cross-dataset, e.g., training on CIFAR and treating SVHN as anomalous; (ii) class-split, e.g., CIFAR-10 with 9 classes as normal and 1 as anomalous, rotating through to obtain $K$ AUROCs. The latter is ostensibly "closer to unconditional OOD" because anomalies do not originate from external datasets.

**Limitations of Prior Work**: On natural image datasets, a semantic class designated as an "anomaly" may have samples closer to the core of the normal mixture than a significant portion of normal samples in the representation space. In such cases, all AD scores based on "distance-to-normal" or "local density" fail—AUROC not only collapses toward 0.5 but may even reverse ($\mathrm{AUC}(c)<0.5$), and the preferred score direction varies with the unknown anomaly class. The common community response is to "flip the sign" or use "stronger AD to re-invert," which the authors argue avoids the actual failure.

**Key Challenge**: Evaluation protocols aim to test whether a fixed scoring convention consistently ranks anomalies on the more typical or atypical side. However, under overlapping geometry, different held-out anomaly classes prefer opposite directions ($d(c)=\mathrm{sign}(\mathrm{AUC}(c)-1/2)$ is inconsistent). Thus, even if a method achieves $\mathrm{AUC}>0.5$ on a labeled split, it may be exploiting accidental correlations between class labels and representation geometry rather than learning a class-agnostic concept of "atypicality."

**Goal**: (i) Formalize this failure mode (AUROC collapse/reversal/direction instability); (ii) provide a training-free diagnostic metric in the representation space to predict protocol unreliability for a given (dataset, representation) before running AUROC; (iii) validate consistency between diagnosis and failure using a controlled experimental matrix.

**Key Insight**: The question of whether a benchmark is "well-defined" is translated into whether "class-conditional manifolds overlap in the representation space," which is quantified as a geometric statistic: "what proportion of $k$-NN neighbors have different labels." This does not depend on detector training.

**Core Idea**: Before running AD on class-split benchmarks, perform a "pre-check" using $k$-NN neighborhood class leakage $L_k$. High $L_k$ indicates the protocol is essentially a geometric stress test rather than evidence of OOD capability.

## Method

### Overall Architecture
For a dataset with $K$ semantic classes and a fixed representation mapping $r:\mathcal X\to\mathbb R^d$ (pixel space or a VAE encoder trained unsupervised on normal data), a $K\!-\!1$ vs. $1$ protocol is executed for each class $c$: $c$ is the anomaly, the remaining $K-1$ classes are normal (unlabeled). AD scores (kNN distance / Isolation Forest / LOF) are computed to obtain a set of $\{\mathrm{AUC}(c)\}_{c=1}^K$. Simultaneously, the diagnostic metric $L_k$ is calculated, and the benchmark failure degree is summarized by reversal rate, near-random rate, AUROC variance, and direction instability rate. The core contribution lies in the diagnosis rather than the AD algorithm; thus, the pipeline introduces no new detectors.

### Key Designs

1.  **Neighborhood Class Leakage $\ell_k(i)$ and Dataset-level Pathology Index $L_k$**:
    - **Function**: Quantitatively describes whether "local geometry is consistent with semantic classes" in a fixed representation space to predict if the class-split protocol is well-posed.
    - **Mechanism**: For each sample $i$, find its $k$-nearest neighbors $\mathcal N_k(i)$ in terms of Euclidean distance in the representation space. Define $\ell_k(i)=\frac1k\sum_{j\in\mathcal N_k(i)}\mathbb I[y_j\neq y_i]$. $\ell_k(i)\approx 0$ indicates a pure neighborhood, while $\ell_k(i)\approx 1$ means nearly all neighbors belong to other classes. The dataset-level index is the average: $L_k(\mathcal T;r)=\frac1m\sum_{i=1}^m\ell_k(i)$.
    - **Design Motivation**: (a) Requires no detector training; (b) directly corresponds to the degree of class-conditional manifold overlap, the root cause of failure for distance/density-monotone AD scores; (c) depends only on the representation $r$, enabling comparative experiments between pixel space and VAE latent space.

2.  **Direction Instability Rate $\rho_{\mathrm{dir}}$ and Reversal Rate $\rho_{\mathrm{inv}}$**:
    - **Function**: Characterizes whether the benchmark provides a consistent scoring direction as a single scalar, distinguishing "low AUROC" from "inconsistent AUROC direction."
    - **Mechanism**: Define the sign of deviation from chance as $d(c)=\mathrm{sign}(\mathrm{AUC}(c)-1/2)\in\{-1,0,+1\}$, where $|\mathrm{AUC}(c)-1/2|\le\epsilon$ is treated as 0. Measure direction inconsistency as $\rho_{\mathrm{dir}}(\epsilon)=1-\frac1K\max\{\sum_c\mathbb I[d(c)=+1],\sum_c\mathbb I[d(c)=-1]\}$. $\rho_{\mathrm{dir}}\to 0$ means all classes prefer the same direction; $\rho_{\mathrm{dir}}\to 0.5$ means directions are split 50/50. Reversal rate $\rho_{\mathrm{inv}}=\frac1K\sum_c\mathbb I[\mathrm{AUC}(c)<1/2]$ quantifies pure reversals.
    - **Design Motivation**: Mean AUROC obscures bimodal failures (some classes high, others reversed); $\rho_{\mathrm{dir}}$ captures direction inconsistency that "sign flipping" cannot fix, as the anomaly class is unknown at deployment.

3.  **Controlled Experiment Matrix + Hypothesis Testing Perspective**:
    - **Function**: Implements the abstract notion that "benchmarking is an implicit hypothesis test regarding scoring consistency" into a reproducible design.
    - **Mechanism**: A $3\times 2\times 3$ matrix: Dataset $\in\{\text{Fashion-MNIST, CIFAR-10, Imagenette}\}$ × Representation $\in\{\text{Pixel, VAE Latent}\}$ × Score $\in\{\text{kNN, Isolation Forest, LOF}\}$. Fashion-MNIST serves as a low-overlap negative control. VAEs are trained unsupervised on the normal pool with fixed hyperparameter priors.
    - **Design Motivation**: (a) Uses a "should be well-posed" control (Fashion-MNIST) to prove $L_k$ is non-trivial; (b) uses multiple (representation, detector) pairs to ensure failures are not artifacts of a single detector; (c) explicitly separates the protocol issue from method performance.

### Loss & Training
No new training objectives. VAEs use standard ELBO on the normal pool. kNN/IF/LOF use off-the-shelf implementations with fixed hyperparameter priors.

## Key Experimental Results

### Main Results: Diagnostic Metric $L_k$ vs. Benchmark Failure (Table 1, Average of kNN/IF/LOF)

| Dataset | Representation | $L_k$ | $\rho_{\mathrm{inv}}$ | $\rho_{\mathrm{rnd}}$ | $\sigma^2_{\mathrm{AUC}}$ | $\rho_{\mathrm{dir}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fashion-MNIST | Pixel | 0.2428 | 0.03 | 0.07 | 0.0194 | 0.10 |
| Fashion-MNIST | Latent | 0.2346 | 0.23 | 0.23 | 0.0224 | 0.30 |
| CIFAR-10 | Pixel | 0.7609 | 0.43 | 0.13 | 0.0162 | 0.50 |
| CIFAR-10 | Latent | 0.7885 | 0.50 | 0.03 | 0.0185 | 0.50 |
| Imagenette | Pixel | 0.7815 | 0.43 | 0.43 | 0.0068 | 0.63 |
| Imagenette | Latent | 0.8363 | 0.50 | 0.33 | 0.0092 | 0.63 |

### Key Findings
- **$L_k$ is a strong precursor for benchmark diagnosis**: When $L_k\approx 0.24$ (Fashion-MNIST pixel), $\rho_{\mathrm{inv}}=0.03$ and $\rho_{\mathrm{dir}}=0.10$. At $L_k\approx 0.84$ (Imagenette latent), half the classes reverse, and direction instability reaches 0.63.
- **Failure is not specific to one scorer**: Averages across kNN, Isolation Forest, and LOF show the same pattern, indicating failure is at the protocol-representation geometry level.
- **Pixel vs. VAE Latent spaces do not change the conclusion**: CIFAR-10/Imagenette maintain high $L_k$ and $\rho_{\mathrm{inv}}\approx 0.5$ in both representations, suggesting that unconditional latent spaces do not "save" the protocol itself.
- **Per-class AUROC Scatter (Fig 1)**: In CIFAR-10 pixel space, different held-out classes simultaneously exhibit AUROC clusters far above and far below 0.5, visualizing direction instability.

## Highlights & Insights
- **Training-free Check**: Calculating $L_k$ allows researchers to avoid meaningless comparisons on naturally pathological benchmarks, offering a low-cost upgrade to empirical standards in the AD/OOD community.
- **Debunking "Sign Flipping"**: The authors decouple "using stronger AD to re-invert" (in-sample fitting) from "whether the protocol consistently defines the anomaly direction" (well-posedness at deployment).
- **Transferable Utility**: $L_k$ is not limited to AD; any evaluation relying on "label-geometry alignment" (e.g., OSR, NCD, calibration) can use neighborhood class leakage as a pre-check.

## Limitations & Future Work
- **Lack of Diagnostic Threshold**: While $L_k$ correlates with failure, the paper does not provide an operational threshold (e.g., $L_k \ge \tau$ is pathological).
- **Scope**: Only three AD scores and three datasets were verified. Geometry in self-supervised/pre-trained features (DINO, CLIP) and text/time-series OOD remains to be explored.
- **No Fix Proposed**: The paper is diagnostic. It does not propose how to modify protocols when $L_k$ is high, though weighting AUROC by $L_k$ or switching to controlled geometric splits are possible directions.
- **Theoretical Bounds**: The causal link between $L_k$ and $\rho_{\mathrm{inv}}$ relies on empirical evidence; a formal proof under specific manifold models is missing.
- **Sensitivity to $k$**: $L_k$ scales with $k$. Robust selection strategies for $k$ are needed for cross-paper comparisons.

## Related Work & Insights
- **Comparison to Surveys**: Unlike broad AD surveys that list several failure modes, this work focuses specifically on the class-split protocol, detailing failures at the level of direction and consistency.
- **OOD Baselines**: Traditional OOD research assumes higher AUROC is better. This work warns that high AUROC on a pathological protocol may reflect geometric shortcuts rather than learned OOD concepts.
- **Insight**: Benchmarks using semantic labels for OOD should report $L_k$ and $\rho_{\mathrm{dir}}$. Neighborhood leakage concepts could extend to LLM OOD evaluation in embedding spaces as a retrieval-style diagnostic.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Explicitly formalizing "well-posed benchmarks" and introducing $\rho_{\mathrm{dir}}$ is a clear methodological advancement.
- **Experimental Thoroughness**: ⭐⭐⭐ The matrix is clear but small (3 datasets × 2 representations × 3 scores); lacking SOTA self-supervised/Transformer features.
- **Writing Quality**: ⭐⭐⭐⭐ The argument chain is logical; the clarification on "sign flipping" is essential for the community.
- **Value**: ⭐⭐⭐⭐ Provides a low-cost, immediately usable "pre-check" that could become a standard for reporting benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](../../CVPR2026/object_detection/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/object_detection/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/object_detection/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models](../../CVPR2026/object_detection/the_cote_score_a_decomposable_framework_for_evaluating_document_layout_analysis_.md)

</div>

<!-- RELATED:END -->
