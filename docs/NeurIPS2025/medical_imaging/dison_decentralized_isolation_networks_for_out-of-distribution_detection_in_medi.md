---
title: >-
  [Paper Note] DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging
description: >-
  [NeurIPS 2025][Medical Imaging][OOD detection] This paper proposes Decentralized Isolation Networks (DIsoN), which detects OOD samples by training a binary classifier to "isolate" a test sample from training data, and leverages training data information without sharing it through decentralized parameter exchange. The method achieves state-of-the-art performance across 12 OOD detection tasks on 4 medical imaging datasets.
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "OOD detection"
  - "decentralized learning"
  - "isolation networks"
  - "medical imaging safety"
  - "privacy preservation"
date: 2026-05-08
content_hash: fc80ede7f5e0e97f
---

# DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging

**Conference**: NeurIPS 2025
**arXiv**: [2506.09024](https://arxiv.org/abs/2506.09024)  
**Code**: [GitHub](https://github.com/FelixWag/DIsoN)  
**Area**: Medical Imaging
**Keywords**: OOD detection, decentralized learning, isolation networks, medical imaging safety, privacy preservation

## TL;DR
This paper proposes Decentralized Isolation Networks (DIsoN), which detects OOD samples by training a binary classifier to "isolate" a test sample from training data, and leverages training data information without sharing it through decentralized parameter exchange. The method achieves state-of-the-art performance across 12 OOD detection tasks on 4 medical imaging datasets.

## Background & Motivation

**Background**: Safe deployment of ML models in medical imaging requires OOD detection capabilities. Existing methods fall into two categories: post-hoc methods (MSP, Energy, Mahalanobis distance, etc.) and training-time regularization methods (CIDER, PALM, etc.).

**Limitations of Prior Work**:
   - Most OOD detection methods do **not use training data** after deployment, relying solely on indirect representations from model outputs or latent spaces.
   - Methods that do leverage training data (KNN, density estimation) require storing training data or embeddings at the deployment site, which is infeasible due to privacy, regulatory, and storage constraints.
   - Indirect representations (e.g., summary statistics, prototypes, synthetic samples) fail to faithfully capture all characteristics of the original training data.

**Key Challenge**: Direct comparison with training data is most effective for OOD detection, yet data sharing is generally infeasible in clinical settings.

**Key Insight**: Drawing inspiration from the "isolation" principle of Isolation Forest and the parameter exchange mechanism of federated learning, using convergence speed as the OOD score.

**Core Idea**: A binary classifier is trained to separate a single test sample from training data. OOD samples are easily isolated (fast convergence), while in-distribution (ID) samples share features with training data and are difficult to isolate (slow convergence). Decentralized training eliminates the need for data sharing.

## Method

### Overall Architecture
The deployment side (Target Node, TN) holds the test sample $\mathbf{x}_t$, while the model provider (Source Node, SN) holds training data $\mathcal{D}_s$. No data is shared between the two sides; only model parameters are exchanged. An Isolation Network (binary classifier) is trained over multiple communication rounds, and the number of rounds $R$ required for convergence serves as the OOD indicator.

### Key Designs

1. **Isolation Network (Centralized Version)**

    - **Function**: Trains a binary classifier to separate a single test sample $\mathbf{x}_t$ (label 1) from training data $\mathcal{D}_s$ (label 0).
    - **Mechanism**: Constructs a mini-batch $B = B_s \cup \{\mathbf{x}_t^{(1)}, ..., \mathbf{x}_t^{(N)}\}$, addressing class imbalance by oversampling $\mathbf{x}_t$. Loss function:
      $$\mathcal{L}_c(\theta; B_s, \mathbf{x}_t, N) = \frac{1}{|B_s|+N}\left(\sum_{\mathbf{x}_s \in B_s} L(\theta; \mathbf{x}_s, 0) + N \cdot L(\theta; \mathbf{x}_t, 1)\right)$$
    - OOD score = convergence time $K$ (number of steps until the classifier correctly classifies $\mathbf{x}_t$ for $E_{\text{stab}}=5$ consecutive steps and achieves accuracy $\geq \tau=0.85$ on $\mathcal{D}_s$).
    - **Design Motivation**: OOD samples exhibit patterns distinct from training data and are easily separated (small $K$); ID samples share features with training data and are difficult to separate (large $K$).

2. **Decentralized Isolation Networks (DIsoN)**

    - **Function**: Approximates centralized Isolation Network training without sharing data.
    - **Mechanism**: Employs a federated learning-style multi-round communication protocol:
        - Initialization: The SN initializes with the feature extractor of a pretrained model $M_{pre}$ plus a randomly initialized binary classification head.
        - Each round: Both sides perform $E$ local update steps (SN trains on $\mathcal{D}_s$ with label $m=0$; TN trains on $\mathbf{x}_t$ with label $m=1$).
        - Aggregation: $\theta^{(r+1)} = \alpha \cdot \theta_S^{(r,E)} + \beta \cdot \theta_T^{(r,E)}$, where $\beta = 1 - \alpha$.
    - **Theoretical Guarantee (Proposition 3.1)**: When $E=1$ and $\alpha = |B_s|/(|B_s|+N)$, the decentralized update is **exactly equivalent** to the centralized update.
    - OOD score = number of communication rounds $R$ required for convergence.

3. **Class-Conditional DIsoN (CC-DIsoN)**

    - **Function**: Uses the pretrained model to predict the class $\hat{y}$ of $\mathbf{x}_t$; the SN then samples only from training data of the predicted class.
    - **Design Motivation**: ID samples are more similar to same-class training data and thus harder to isolate; narrowing the comparison scope increases the ID/OOD separability.
    - **Implementation**: Requires only that the TN sends the predicted label to the SN, which then filters the mini-batch — a nearly zero-cost addition.

4. **Practical Techniques: Data Augmentation + Instance Normalization**

    - Random cropping, flipping, and color jittering prevent the model from rapidly memorizing a single $\mathbf{x}_t$ (without augmentation, both ID and OOD samples converge quickly, making them indistinguishable).
    - Instance Normalization replaces Batch Normalization (BN is unsuitable for single-sample TN updates).

### Loss & Training
- Binary cross-entropy loss.
- Feature extractor initialized from a pretrained classification model (ResNet18 with Instance Normalization).
- Adam/SGD optimizer; $\alpha=0.8$ is robust across all datasets.
- Each round of local $E$ steps approximates one epoch over the training data.

## Key Experimental Results

### Main Results: OOD Detection on Three Medical Imaging Datasets

| Method | Type | X-Ray AUROC↑ | Derm AUROC↑ | Ultrasound AUROC↑ | Avg. AUROC↑ | Avg. FPR95↓ |
|--------|------|-------------|------------|-------------------|------------|------------|
| MSP | post-hoc | 60.44 | 65.39 | 58.85 | 61.56 | 100.00 |
| fDBD | post-hoc | 68.26 | 63.59 | 60.73 | 64.19 | 84.61 |
| ViM | post-hoc | 62.60 | 68.39 | 59.44 | 63.48 | 78.12 |
| CIDER | regularization | 70.47 | 81.98 | 58.03 | 70.16 | 72.77 |
| PALM | regularization | 65.41 | 77.25 | 59.35 | 67.34 | 74.59 |
| **CC-DIsoN** | **Ours** | **84.94** | **89.54** | **65.62** | **80.00** | **59.20** |

CC-DIsoN achieves 9.8% higher average AUROC and 13.6% lower FPR95 compared to the best baseline (CIDER).

### Ablation Study

| Configuration | Avg. AUROC↑ | Avg. FPR95↓ | Note |
|---------------|------------|------------|------|
| DIsoN (no class conditioning) | 78.3 | 67.4 | Base version |
| **CC-DIsoN** | **80.0** | **59.2** | +1.7% AUROC, −8.2% FPR95 |

### Histopathology MIDOG Experiment (Near/Far OOD)

| Method | Near-OOD Avg. AUROC | Far-OOD Avg. AUROC |
|--------|-------------------|-------------------|
| ViM | 62.4 | 92.6 |
| CIDER | 60.9 | 89.0 |
| PALM | 61.2 | 98.3 |
| **CC-DIsoN** | **69.6** | **98.3** |

CC-DIsoN outperforms the second-best method by 8.4% in the Near-OOD setting.

### Key Findings
- Data augmentation has a substantial impact: without augmentation, AUROC on the dermatology dataset is approximately 76%; with augmentation, it reaches 89.5% (+13.7%), as augmentation prevents rapid memorization of individual samples.
- Performance is robust for $\alpha \in [0.5, 0.95]$; $\alpha=0.8$ achieves the best overall trade-off between detection performance and convergence speed.
- ResNet18 is the optimal network size — Slim-ResNet18 lacks sufficient capacity, while ResNet34 increases computational cost without performance gains.
- CC-DIsoN is robust to prediction errors: even when all ID samples are misclassified, AUROC remains competitive with baselines.
- Per-sample inference takes approximately 40 seconds to 4 minutes, which is acceptable in non-emergency clinical workflows where reports are typically reviewed hours later.

## Highlights & Insights
- **"Isolation difficulty" as an OOD score is an elegant design**: OOD detection is reformulated as measuring the convergence speed of a binary classifier, with clear intuition and a formal theoretical guarantee (Proposition 3.1 proves the equivalence between decentralized and centralized updates).
- **Decentralized design addresses a real deployment bottleneck**: Exchanging only model parameters rather than data perfectly accommodates medical privacy constraints, establishing a novel "OOD detection as a service" paradigm in which ML developers can offer secure utilization of training data as a remote service.
- **Simplicity of class conditioning**: Transmitting a single predicted label to improve performance incurs virtually no additional implementation cost.
- The substitution of Instance Normalization for Batch Normalization is critical for single-sample TN nodes and is transferable to other few-shot or single-sample federated learning scenarios.

## Limitations & Future Work
- Inference overhead is significant (multiple communication rounds per sample), making the method unsuitable for real-time applications.
- The current approach isolates one sample at a time; batched test samples must be processed sequentially, leaving room for efficiency improvements.
- Theoretical equivalence holds strictly only when $E=1$; $E>1$ yields an approximation.
- Experiments focus on OOD types such as artifacts (rulers, annotations) and domain shift; semantic OOD (e.g., rare diseases) remains untested.
- The method relies on a pretrained model's feature extractor for initialization; pretraining quality directly affects DIsoN performance.

## Related Work & Insights
- **vs. Isolation Forest / Deep iForest**: Isolation Forest uses decision trees for isolation and performs poorly on high-dimensional image data (AUROC ~40–56%). DIsoN replaces split node counts with neural networks and convergence speed, making it far more suitable for image data.
- **vs. CIDER / PALM**: These training-time regularization methods require modifying the main model's training pipeline. DIsoN is a post-deployment method that leaves the main model unchanged, offering greater flexibility.
- **vs. KNN-based OOD**: KNN requires storing training embeddings at the deployment site; DIsoN avoids data transmission through parameter exchange.
- **vs. FL-based OOD**: Existing federated OOD methods assume multi-node, multi-distribution training data for detecting distribution shift. DIsoN addresses a single-source, single-target isolation task with fundamentally different motivation and technical design.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The core idea of "isolation difficulty = OOD score" is original; the decentralized design offers genuine practical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, 12 tasks, and comprehensive ablations covering $\alpha$, network size, augmentation, class conditioning, and robustness to misclassification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, theoretical derivations are complete, and experimental design is systematic.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the core practical challenge of leveraging training data for OOD detection under privacy constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection](../../CVPR2025/medical_imaging/openmibood_open_medical_imaging_benchmarks_for_out-of-distribution_detection.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](../../CVPR2026/medical_imaging/the_invisible_gorilla_effect_in_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)
- [\[NeurIPS 2025\] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)
- [\[CVPR 2025\] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](../../CVPR2025/medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)

</div>

<!-- RELATED:END -->
