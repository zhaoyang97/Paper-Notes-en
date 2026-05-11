---
title: >-
  [Paper Note] Domain-Skewed Federated Learning with Feature Decoupling and Calibration
description: >-
  [CVPR 2026][AI Safety][Federated Learning] This paper proposes F²DC, a framework that employs a Domain Feature Decoupler (DFD) and a Domain Feature Corrector (DFC) to decompose local client features in federated learning…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Domain Skew"
  - "Feature Decoupling"
  - "Domain-Aware Aggregation"
  - "Representation Calibration"
date: 2026-05-08
content_hash: 6a2be872b7ef40c2
---

# Domain-Skewed Federated Learning with Feature Decoupling and Calibration

**Conference**: CVPR 2026
**arXiv**: [2603.14238](https://arxiv.org/abs/2603.14238)
**Code**: [GitHub](https://github.com/mala-lab/F2DC)
**Area**: AI Safety
**Keywords**: Federated Learning, Domain Skew, Feature Decoupling, Domain-Aware Aggregation, Representation Calibration

## TL;DR

This paper proposes F²DC, a framework that employs a Domain Feature Decoupler (DFD) and a Domain Feature Corrector (DFC) to decompose local client features in federated learning into domain-robust features and domain-related features. Rather than discarding the latter, F²DC calibrates them to recover entangled class-discriminative information, and combines this with a domain-aware aggregation strategy. The method consistently outperforms state-of-the-art approaches across three multi-domain datasets.

## Background & Motivation

**Domain skew in federated learning**: Unlike label skew, domain skew scenarios involve clients whose data originate from different domains (e.g., driving data under varying weather conditions). Class distributions are similar across clients, but feature distributions differ substantially: $\mathbb{P}_{k_1}(x|y) \neq \mathbb{P}_{k_2}(x|y)$.

**Dimensional collapse**: Domain skew causes local model representations to collapse into narrow low-dimensional subspaces — a large number of singular values of the feature covariance matrix approach zero, indicating that each client fits only its own domain's feature subspace while ignoring others.

**Limitations of elimination-based methods**: Methods such as FDSE attempt to remove domain-specific bias, but domain-related features entangle valuable class-discriminative information (e.g., brushstroke-defined object contours in sketch-domain images). Direct elimination leads to information loss — Grad-CAM visualizations show that FDSE misses giraffe horns and heads in cartoon/sketch domains.

**Core Idea**: Rather than eliminating domain-related features, calibrating them recovers entangled class-relevant cues, thereby promoting more consistent cross-domain decision-making.

## Method

### Overall Architecture

F²DC integrates two core modules and one aggregation strategy into the standard FedAvg framework:
- **Domain Feature Decoupler (DFD)**: Decomposes local features into domain-robust features $f^+$ and domain-related features $f^-$
- **Domain Feature Corrector (DFC)**: Calibrates $f^-$ into corrected features $f^\star$ to capture additional class-discriminative cues
- **Domain-Aware Aggregation (DaA)**: Weights global aggregation according to the domain discrepancy of each client

Architecturally, DFD and DFC are inserted after the last backbone layer (after L4 in ResNet-10). The final feature $\tilde{f} = f^+ + f^\star$ is forwarded to subsequent layers. DFD, DFC, and the auxiliary MLP $\mathbf{m}$ are retained locally and do not participate in global aggregation.

### Key Designs

1. **Domain Feature Decoupler (DFD)**

    - **Function**: Assigns cross-domain robustness scores to each unit in the feature map and separates features into domain-robust and domain-related components.
    - **Design Motivation**: Directly processing raw features leads to overfitting domain bias; domain context must first be isolated to enable subsequent calibration.
    - **Mechanism**: A two-layer CNN (with BN + ReLU) constructs an attribute map $\mathcal{S}_i = \mathcal{A}_D(f_i) \in \mathbb{R}^{C \times H \times W}$. A **Gumbel Concrete distribution** generates pseudo-binary masks $\mathcal{M}_i$ (circumventing the non-differentiability of hard discretization), approaching hard binary values as $\sigma \to 0$. Decoupling: $f_i^+ = \mathcal{M}_i \odot f_i$, $f_i^- = (1 - \mathcal{M}_i) \odot f_i$.
    - **Loss** (separability + discriminability): The separability term minimizes cosine similarity between $f^+$ and $f^-$; the discriminability term encourages $f^+$ to predict the ground-truth label and $f^-$ to favor the highest-confidence incorrect label, with logits predicted by auxiliary MLP $\mathbf{m}$.
    - **Distinction from prior work**: FDSE directly eliminates domain features; DFD adopts a "separate but retain" strategy, preserving domain-related features for downstream calibration.

2. **Domain Feature Corrector (DFC)**

    - **Function**: Extracts class-discriminative cues from $f^-$ that complement $f^+$.
    - **Design Motivation**: $f^-$ entangles domain bias with class information; discarding it entirely forfeits valuable signals.
    - **Mechanism**: A two-layer CNN $\mathcal{A}_C$ with the same architecture as DFD learns a residual: $f_i^\star = f_i^- + (1 - \mathcal{M}_i) \odot \mathcal{A}_C(f_i^-)$.
    - **Loss**: Standard cross-entropy $\mathcal{L}_{DFC} = -y_i \cdot \log(\delta(\mathbf{m}(l_i^\star)))$, injecting correct discriminative supervision.

3. **Domain-Aware Aggregation (DaA)**

    - **Function**: Accounts for per-client domain discrepancy during global aggregation.
    - **Design Motivation**: Naive FedAvg ignores domain diversity; equal-weight aggregation introduces systematic bias.
    - **Mechanism**: A uniform global domain distribution $\mathcal{G} = [1/Q,...,1/Q]$ (where $Q$ is the number of domains) is defined. The domain discrepancy $\mathbf{d}_k$ of client $k$ is computed, and aggregation weights are $\mathbf{p}_k = \sigma(\alpha \cdot n_k/N - \beta \cdot \mathbf{d}_k)$, normalized before aggregation.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{|L|}\sum_{j=1}^{|L|}(\lambda_1 \cdot \mathcal{L}_{DFD}^{L_j} + \lambda_2 \cdot \mathcal{L}_{DFC}^{L_j})$$

Default settings: $|L|=1$ (last layer only), $\lambda_1=0.8$, $\lambda_2=1.0$, Gumbel temperature $\sigma=0.1$, separation temperature $\tau=0.06$, aggregation parameters $\alpha=1.0$, $\beta=0.4$. SGD optimizer, lr=0.01, momentum 0.9, batch size 64, 100 communication rounds, 10 local epochs per round.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (F²DC) | Prev. SOTA (FDSE) | Gain |
|--------|------|------|------------------|------|
| PACS | AVG Acc ↑ | **76.47** | 73.13 | +3.34 |
| PACS | STD ↓ | **5.83** | 6.83 | -1.00 |
| Office-Caltech | AVG Acc ↑ | **66.82** | 63.18 | +3.64 |
| Office-Caltech | STD ↓ | **3.65** | 4.50 | -0.85 |
| Digits | AVG Acc ↑ | **87.23** | 84.15 | +3.08 |
| Digits | STD ↓ | **13.36** | 16.19 | -2.83 |

F²DC consistently outperforms all nine baselines (FedAvg / FedProx / MOON / FPL / FedTGP / FedRCL / FedHEAL / FedSA / FDSE) across all three datasets, with lower STD indicating better cross-domain fairness. Contrastive methods such as MOON perform even below FedAvg on PACS, as enforcing alignment on already-corrupted global representations further degrades performance.

### Ablation Study (PACS)

| Configuration | AVG Acc | STD | Note |
|------|---------|-----|------|
| FedAvg (baseline) | 66.39 | 11.74 | No modules |
| + DFD only | 68.43 | 10.15 | Decoupling only |
| + DFD + DFC | 73.64 | 6.12 | Decoupling + correction |
| + DFD + DaA | 75.33 | 6.80 | Decoupling + domain-aware aggregation |
| + DFD + DFC + DaA | **76.47** | **5.83** | Full F²DC |

### Plug-and-Play Compatibility (PACS)

| Base Method | AVG Acc w/ DFD+DFC | Gain |
|----------|-----------------|------|
| FedAvg | 75.33 | **+8.94** |
| FPL | 75.52 | +4.93 |
| FedHEAL | 75.06 | +1.72 |
| FDSE | 74.79 | +1.66 |

### Key Findings

- **Feature analysis**: $f^+$ achieves AVG=75.13, far surpassing $f^-$ at 57.87; however, calibrated $f^\star$ reaches 73.49, confirming that domain-related features contain recoverable class-discriminative information. The fused $\tilde{f}$ achieves the best result of 76.47.
- **Faster convergence**: F²DC exhibits faster convergence on both Office-Caltech and PACS.
- **Minimal overhead**: No additional communication cost (DFD/DFC are retained locally); training time increases by only 2% (180.67s vs. 176.94s per round).

## Highlights & Insights

1. **"Calibrate rather than eliminate"**: Class-discriminative information entangled within domain bias is valuable. Grad-CAM visualizations intuitively demonstrate how F²DC recovers regions overlooked by conventional methods (e.g., the giraffe's torso).
2. **Gumbel Concrete differentiable separation**: Elegantly resolves the non-differentiability of binary feature splitting, enabling end-to-end training.
3. **Dimensional collapse diagnosis**: Singular value analysis quantitatively identifies the fundamental pathology of domain-skewed FL, offering a generalizable diagnostic tool.

## Limitations & Future Work

1. Decoupling granularity depends on hyperparameter $\tau$; overly aggressive separation degrades performance.
2. The method operates at the feature level only, leaving parameter-level domain bias decoupling unaddressed.
3. Experiments cover only 4-domain settings with ResNet-10; scalability to more domains and larger models remains unverified.
4. Domain-aware aggregation assumes uniform intra-domain class distributions; extensions are needed for scenarios combining domain skew and label skew.

## Related Work & Insights

- **FDSE (CVPR'25)**: Representative of elimination-based decoupling; F²DC's "calibrate and utilize" paradigm constitutes a superior alternative.
- **FedHEAL (CVPR'24)**: Employs selective parameter updating and fair aggregation, but does not address domain bias at the feature level.
- **Insight**: The Gumbel Concrete technique is widely used in NAS and pruning; F²DC demonstrates its novel application to feature selection in federated learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "calibrate rather than eliminate" paradigm is relatively novel in domain-skewed FL; the DFD+DFC design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, nine baselines, comprehensive ablations, plug-and-play validation, efficiency analysis, and visualizations are all provided.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation; rich figures including Grad-CAM, T-SNE, and SVD visualizations.
- **Value**: ⭐⭐⭐⭐ — The modular design facilitates easy integration into existing FL frameworks, offering strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[NeurIPS 2025\] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](../../NeurIPS2025/ai_safety/fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)

</div>

<!-- RELATED:END -->
