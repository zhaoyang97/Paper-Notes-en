---
title: >-
  [Paper Note] Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection
description: >-
  [ICML2026][Object Detection][Anomaly Detection] The authors point out that "class-split" anomaly detection benchmarks are ill-posed when the anomaly class and the normal mixture distribution overlap in the representation space—AUROC collapses to random or even reverses, with the direction depending on the unknown anomaly class. A training-free "neighborhood class leakage" metric $L_k$ is proposed to diagnose such benchmark failure before evaluation.
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
content_hash: 53b828381fb03394
---

# Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection

**Conference**: ICML2026  
**arXiv**: [2606.02601](https://arxiv.org/abs/2606.02601)  
**Code**: None  
**Area**: AI Safety / Anomaly Detection / OOD Evaluation  
**Keywords**: Anomaly Detection, OOD Detection, Class-Split Benchmark, AUROC Reversal, Score-Direction Instability, Neighborhood Class Leakage

## TL;DR
The authors point out that "class-split" anomaly detection benchmarks are ill-posed when the anomaly class and the normal mixture distribution overlap in the representation space—AUROC collapses to random or even reverses, with the direction depending on the unknown anomaly class. A training-free "neighborhood class leakage" metric $L_k$ is proposed to diagnose such benchmark failure before evaluation.

## Background & Motivation
**Background**: Two main protocols exist for evaluating completely unconditional OOD Anomaly Detection (AD): (i) Cross-dataset, e.g., training on CIFAR and using SVHN as anomalies; (ii) Class-split, e.g., using 9 classes of CIFAR-10 as normal and 1 class as anomalous, rotating through classes to obtain $K$ AUROCs. The latter is ostensibly "closer to unconditional OOD" because anomalies do not come from external datasets.

**Limitations of Prior Work**: On natural image datasets, a semantic class designated as "anomalous" may contain samples that are closer to the core of the normal mixture distribution in the representation space than many normal samples. In such cases, all "distance-to-normal" or "local density-based" AD scores fail—AUROC not only collapses toward 0.5 but may even reverse ($\mathrm{AUC}(c)<0.5$), and the preferred scoring direction varies with the unknown anomaly class. A common response in the community is to "flip the sign" or "use a stronger AD to re-invert it," which the authors argue avoids addressing the actual failure.

**Key Challenge**: Benchmark protocols aim to measure "whether a fixed scoring convention can consistently rank anomalies on the more typical/atypical side." Under overlapping geometry, different held-out anomaly classes prefer opposite directions (inconsistent $d(c)=\mathrm{sign}(\mathrm{AUC}(c)-1/2)$). Therefore, even if a method achieves $\mathrm{AUC}>0.5$ on a labeled evaluation split, it might just be exploiting an accidental correlation between "class labels ↔ representation geometry" rather than learning a true, class-agnostic concept of "atypicality."

**Goal**: (i) Formalize this failure mode (AUROC collapse/reversal/direction instability); (ii) provide a training-free, representation-side diagnostic metric capable of predicting which (dataset, representation) pairs are unreliable before running AUROC; (iii) validate the consistency between diagnostics and failure using a controlled experimental matrix.

**Key Insight**: The question of "whether a benchmark is well-posed" is translated into "whether class-conditional manifolds overlap in the representation space." This overlap is quantified as a purely geometric statistic: "what proportion of $k$-NN neighbors have different labels," independent of detector training.

**Core Idea**: Before running AD on class-split benchmarks, perform a "pre-check" using neighborhood class leakage $L_k$. A high $L_k$ indicates that the protocol is essentially a geometric stress test rather than evidence of OOD capability.

## Method

### Overall Architecture
For a dataset with $K$ semantic classes, a representation mapping $r:\mathcal X\to\mathbb R^d$ (pixel space or a VAE encoder trained unsupervised on normal data) is fixed. For each class $c$, a $K-1$ vs. $1$ protocol is executed: class $c$ is treated as anomalous, while the remaining $K-1$ classes are treated as normal (unlabeled) to train AD scores (kNN distance / Isolation Forest / LOF), resulting in a set of $\{\mathrm{AUC}(c)\}_{c=1}^K$. Simultaneously, the diagnostic metric $L_k$ is computed, and the benchmark failure degree is summarized using (reversal rate, near-random rate, AUROC variance, direction instability rate). The core contribution lies in the diagnostics rather than the AD algorithm, so the pipeline introduces no new detectors.

### Key Designs

**1. Neighborhood Class Leakage $\ell_k(i)$ and Dataset-level Pathological Index $L_k$: Quantifying "Geometry-Semantic Alignment" Before Evaluation**

The root cause of class-split protocol failure is that samples from a class designated as "anomalous" may be closer to the core of the normal mixture than some normal samples. Any AD score monotonic to "distance to normal" or local density will be defeated by this overlap. The authors characterize overlap via a purely geometric statistic: for each sample $i$, identify its $k$ nearest neighbors $\mathcal{N}_k(i)$ in the representation space under Euclidean distance, and define $\ell_k(i)=\frac1k\sum_{j\in\mathcal{N}_k(i)}\mathbb I[y_j\neq y_i]$ (proportion of different classes in the neighborhood; $\approx 0$ indicates a pure neighborhood, $\approx 1$ indicates almost entirely other classes). The dataset average is $L_k(\mathcal T;r)=\frac1m\sum_{i=1}^m\ell_k(i)$. This index requires no detector training and depends only on the representation $r$, allowing for comparisons between pixel space and VAE latent space. A high $L_k$ indicates the protocol is a geometric stress test rather than OOD evidence.

**2. Direction Instability Rate $\rho_{\mathrm{dir}}$ and Reversal Rate $\rho_{\mathrm{inv}}$: Separating "Low AUROC" from "Inconsistent AUROC Direction"**

Mean AUROC can mask bimodal failures where some classes perform well while others reverse. The latter is fatal—if the anomaly class is unknown at deployment, it is impossible to determine whether to flip the scoring sign. The authors define a direction sign $d(c)=\mathrm{sign}(\mathrm{AUC}(c)-1/2)\in\{-1,0,+1\}$ for each class (where $|\mathrm{AUC}(c)-1/2|\le\epsilon$ is mapped to 0). Directional voting inconsistency is measured by $\rho_{\mathrm{dir}}(\epsilon)=1-\frac1K\max\{\sum_c\mathbb I[d(c)=+1],\sum_c\mathbb I[d(c)=-1]\}$. $\rho_{\mathrm{dir}}\to 0$ means all classes prefer the same direction, while $\to 0.5$ means directions are split 50/50. This is supplemented by the pure reversal rate $\rho_{\mathrm{inv}}=\frac1K\sum_c\mathbb I[\mathrm{AUC}(c)<1/2]$, near-random rate $\rho_{\mathrm{rnd}}(\epsilon)=\frac1K\sum_c\mathbb I[|\mathrm{AUC}(c)-1/2|\le\epsilon]$, and AUROC variance $\sigma^2_{\mathrm{AUC}}$. High $\rho_{\mathrm{dir}}$ implies no fixed sign convention can consistently rank anomalies from all classes on the same side, rendering sign-flipping useless.

**3. Controlled Experimental Matrix + Hypothesis Testing Perspective: Reproducible Contrastive Design**

To prove $L_k$ is non-trivial and the failure is not specific to one detector, a $3\times 2\times 3$ matrix is fixed: Dataset $\in\{\text{Fashion-MNIST, CIFAR-10, Imagenette}\}$ × Representation $\in\{\text{Pixel, VAE Latent}\}$ × Scoring $\in\{\text{kNN, Isolation Forest, LOF}\}$. Fashion-MNIST serves as a low-overlap negative control, while CIFAR-10 and Imagenette represent high-overlap complex natural images. VAEs are trained unsupervised only on the normal pool with fixed hyperparameter priors. Class labels are used only for splitting, metric calculation, and diagnostics, not for representation learning. This design distinguishes between protocol-side issues (is this benchmark worth running) and method-side issues (how strong is a specific AD).

### Loss & Training
No new training objectives are introduced. VAEs use standard ELBO on the normal pool. kNN/IF/LOF use off-the-shelf implementations (sklearn-style). All hyperparameters are fixed a priori.

## Key Experimental Results

### Main Results: Diagnostic Metric $L_k$ vs. Benchmark Failure (Table 1, Averaged kNN / IF / LOF)

| Dataset | Representation | $L_k$ | $\rho_{\mathrm{inv}}$ | $\rho_{\mathrm{rnd}}$ | $\sigma^2_{\mathrm{AUC}}$ | $\rho_{\mathrm{dir}}$ |
|---------|----------------|-------|-----------------------|-----------------------|---------------------------|-----------------------|
| Fashion-MNIST | Pixel | 0.2428 | 0.03 | 0.07 | 0.0194 | 0.10 |
| Fashion-MNIST | Latent | 0.2346 | 0.23 | 0.23 | 0.0224 | 0.30 |
| CIFAR-10 | Pixel | 0.7609 | 0.43 | 0.13 | 0.0162 | 0.50 |
| CIFAR-10 | Latent | 0.7885 | 0.50 | 0.03 | 0.0185 | 0.50 |
| Imagenette | Pixel | 0.7815 | 0.43 | 0.43 | 0.0068 | 0.63 |
| Imagenette | Latent | 0.8363 | 0.50 | 0.33 | 0.0092 | 0.63 |

### Predictive Power of Diagnostics for Failure Modes (Trends by $L_k$ Rank)

| Regime | $L_k$ Range | Typical $\rho_{\mathrm{inv}}$ | Typical $\rho_{\mathrm{dir}}$ | Inference |
|--------|-------------|------------------------------|------------------------------|-----------|
| Low Overlap | $\sim 0.24$ | 0.03 | 0.10 | Protocol is well-posed; AUROC reflects class-agnostic atypicality. |
| Moderate Overlap | $\sim 0.23$–0.5 | 0.23 | 0.30 | Directions become unstable; AIROC interpretation requires caution. |
| High Overlap | $\sim 0.76$–0.84 | 0.43–0.50 | 0.50–0.63 | Protocol is pathological; AUROC primarily reflects geometric overlap. |

### Key Findings
- **$L_k$ is a strong precursor for OOD/AD benchmark diagnostics**: When $L_k \approx 0.24$ (Fashion-MNIST pixel), $\rho_{\mathrm{inv}}=0.03$ and $\rho_{\mathrm{dir}}=0.10$. Conversely, when $L_k \approx 0.84$ (Imagenette latent), half the classes reverse and direction instability reaches 0.63.
- **Failures are not detector-specific**: Averaging three types of density/distance-based AD scorers (kNN, IF, LOF) yields the same pattern, suggesting failure originates from the protocol-representation geometry rather than detector artifacts.
- **Pixel vs. VAE Latent spaces do not change the conclusion**: CIFAR-10/Imagenette show high $L_k$ and $\rho_{\mathrm{inv}} \approx 0.5$ in both representations. Changing to a different unconditionally trained latent space does not save the protocol, supporting the argument that the problem lies in the benchmark split.
- **Class-wise AUROC visualization**: In CIFAR-10 pixel space, held-out classes produce two clusters of AUROC (well above and well below 0.5), directly visualizing direction instability.

## Highlights & Insights
- **Training-free "Benchmark Pre-check"**: Calculating $L_k$ beforehand avoids meaningless comparisons on naturally pathological benchmarks, offering a low-cost upgrade to empirical standards in the AD/OOD community.
- **Refuting the "Sign-Flipping" Argument**: The authors distinguish between "using a stronger AD to re-invert results" (in-sample fitting) and "consistent definition of anomaly direction" (well-posedness under unknown anomalies). This distinction has methodological value for the OOD evaluation norm.
- **Transferable Techniques**: $L_k$ is not limited to AD; any evaluation relying on "class label ↔ representation geometry" alignment (e.g., Open-Set Recognition, Novel Class Discovery) can use similar neighborhood checks. $\rho_{\mathrm{dir}}$ can also be generalized to any "$K$-fold hold-out + AUROC" protocol.

## Limitations & Future Work
- **Lack of Diagnostic Threshold**: While $L_k$ correlates strongly with failure, the paper does not provide an operational threshold (e.g., $L_k \ge \tau$ is pathological). Practical application still relies on empirical observation.
- **Scope of Validation**: The study only covers three AD scores and three datasets. It remains to be verified how geometric overlap behaves in self-supervised/pre-trained feature spaces (DINO, CLIP) or for text/time-series OOD.
- **No New AD Algorithm**: The paper is "diagnostic and cautionary." It does not provide a fix for the protocol when $L_k$ is high—a natural extension would be weighting AUROC by $L_k$ or replacing class-splits with controlled geometric splits.
- **Theoretical Characterization**: The causal link between $L_k$ and $\rho_{\mathrm{inv}}/\rho_{\mathrm{dir}}$ is primarily empirical. A provable bound under specific manifold overlap models is missing.
- **Sensitivity to $k$**: $L_k$ values depend on the choice of $k$. A robust selection strategy for $k$ is needed to make $L_k$ comparable across different papers.

## Related Work & Insights
- **vs. Chalapathy & Chawla (2019) / Pang et al. (2021)**: These surveys identify various AD failure modes (e.g., distribution shift). This paper focuses specifically on the class-split protocol and refines failures into direction/reversal/consistency levels.
- **vs. Hendrycks & Gimpel OOD Baselines**: Traditional OOD benchmarks assume higher AUROC is always better. This paper warns that high AUROC on a pathological benchmark may indicate the method learned geometric shortcuts rather than OOD concepts.
- **Insights**: (1) Any OOD/AD benchmark constructed from classification labels should report $L_k$ and $\rho_{\mathrm{dir}}$; (2) Diagnostic ideas can be extended to "pseudo-label split" or "semantic subset OOD" scenarios; (3) For OOD evaluation in the LLM era (e.g., prompt-level anomalies), retrieval-style diagnostics in embedding space are a promising future direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicit formalization of benchmark well-posedness and the introduction of $\rho_{\mathrm{dir}}$ are significant methodological advances for AD evaluation.
- Experimental Thoroughness: ⭐⭐⭐ The controlled matrix is clear but small-scale (3 datasets × 2 representations × 3 scores), lacking coverage of SOTA self-supervised or Transformer features.
- Writing Quality: ⭐⭐⭐⭐ The logical chain is clean, and the clarification on why "sign-flipping" fails to address the underlying issue is vital for the community.
- Value: ⭐⭐⭐⭐ Provides a low-cost, immediate "pre-check" for the OOD/AD community that could refine common benchmark reporting paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](../../CVPR2026/object_detection/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/object_detection/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[CVPR 2025\] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](../../CVPR2025/object_detection/anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[NeurIPS 2025\] AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](../../NeurIPS2025/object_detection/autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)

</div>

<!-- RELATED:END -->
