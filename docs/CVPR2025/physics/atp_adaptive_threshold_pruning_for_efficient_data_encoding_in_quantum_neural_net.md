---
title: >-
  [Paper Note] ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks
description: >-
  [CVPR 2025][Physics & Scientific Computing][Quantum Neural Networks] This paper proposes Adaptive Threshold Pruning (ATP) to adaptively prune low-information data features prior to quantum data encoding. By optimizing thresholds via L-BFGS-B, ATP achieves the highest accuracy in binary classification tasks across four datasets (MNIST, FashionMNIST, CIFAR, PneumoniaMNIST) while significantly reducing entanglement entropy.
tags:
  - "CVPR 2025"
  - "Physics & Scientific Computing"
  - "Quantum Neural Networks"
  - "Adaptive Pruning"
  - "Data Encoding"
  - "Entanglement Entropy"
  - "Quantum Machine Learning"
date: 2026-05-08
content_hash: e39ecc8f832351d8
---

# ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks

**Conference**: CVPR 2025  
**arXiv**: [2503.21815](https://arxiv.org/abs/2503.21815)  
**Code**: None  
**Area**: Quantum Computing / Physics  
**Keywords**: Quantum Neural Networks, Adaptive Pruning, Data Encoding, Entanglement Entropy, Quantum Machine Learning

## TL;DR

This paper proposes Adaptive Threshold Pruning (ATP) to adaptively prune low-information data features prior to quantum data encoding. By optimizing thresholds via L-BFGS-B, ATP achieves the highest accuracy in binary classification tasks across four datasets (MNIST, FashionMNIST, CIFAR, PneumoniaMNIST) while significantly reducing entanglement entropy.

## Background & Motivation

**Background**: Quantum Neural Networks (QNNs) utilize quantum superposition and entanglement for data processing, but are limited by hardware constraints such as insufficient qubit resources, noise, and decoherence. Data encoding is a critical bottleneck in QML, where the encoding strategy directly determines qubit usage efficiency.

**Limitations of Prior Work**: (1) Angle encoding directly maps pixel values to rotation angles, which is simple but introduces redundant data that increases entanglement and circuit complexity; (2) Amplitude encoding is compact but limited in scalability; (3) Excessively high entanglement entropy increases computational complexity and leads to the barren plateau problem.

**Key Challenge**: While higher entanglement enhances expressiveness, excessive entanglement increases hardware overhead and degrades training efficiency, necessitating a balance between the two.

**Goal**: To reduce data redundancy prior to encoding, achieving superior classification performance with fewer quantum resources.

**Key Insight**: Information density varies significantly across different regions in an image. Low-variance regions contribute minimally to classification and can be pruned prior to encoding.

**Core Idea**: Set low-information data regions to zero based on an adaptive threshold to reduce encoding redundancy, while automatically searching for the optimal threshold through bi-level optimization.

## Method

### Overall Architecture

ATP introduces a preprocessing stage prior to data encoding: (1) calculate the average pixel intensity matrix for each class; (2) set positions to zero where the average values of both classes are below a threshold $\tau$; (3) optimize the threshold using L-BFGS-B to maximize test accuracy; (4) perform angle encoding on the pruned data to train the QNN.

### Key Designs

1. **Adaptive Threshold Pruning Function**:

    - Function: Removes low-information data regions that do not contribute to classification.
    - Mechanism: Calculates the average pixel intensity matrices $\bar{x}_0$ and $\bar{x}_1$ for the two binary classes. For position $(i,j)$, if $\bar{x}_0(i,j) < \tau$ and $\bar{x}_1(i,j) < \tau$, the value at this position is set to zero. This is equivalent to retaining only high-information regions valuable for distinguishing the two classes.
    - Design Motivation: Pixels in low-variance regions are nearly indistinguishable between the two classes. Encoding this redundant information not only wastes qubit resources but also introduces unnecessary entanglement.

2. **Bi-level Threshold Optimization (L-BFGS-B)**:

    - Function: Automatically searches for the optimal threshold $\tau^*$.
    - Mechanism: The outer-loop objective is to maximize the test set accuracy as $\tau^* = \arg\max_\tau \text{Acc}_\text{test}(\mathcal{X}_\tau)$, while the inner loop trains the QNN on the pruned data. The L-BFGS-B quasi-Newton method is employed to iteratively optimize within the constraint $[0, \tau_\text{max}]$, achieving efficient gradient descent by approximating the inverse Hessian matrix.
    - Design Motivation: Manual threshold selection cannot adapt to the distribution characteristics of different datasets. Automatic optimization ensures the threshold is tailor-made for the underlying data.

3. **Entanglement Entropy (EE) as an Efficiency Metric**:

    - Function: Evaluates the quantum resource utilization efficiency of the encoding scheme.
    - Mechanism: Entanglement entropy is calculated via the von Neumann entropy of the reduced density matrix, quantifying the degree of correlation between qubits. ATP improves accuracy while lowering EE, achieving "better performance with less entanglement."
    - Design Motivation: Lower EE implies less cross-talk interference among qubits, which facilitates practical deployment on noisy hardware.

### Loss & Training

A three-layer Parameterized Quantum Circuit (PQC) containing XX and ZZ entanglement gates is trained using the COBYLA optimizer. Threshold optimization and QNN training are executed alternately.

## Key Experimental Results

### Main Results

| Class Pairs | Angle | Amplitude | ATP | PCA | SQE |
|------|-------|-----------|-----|-----|-----|
| MNIST(0,1) | 96.0 | 95.5 | **99.0** | 99.0 | 88.0 |
| CIFAR(0,1) | 70.0 | 68.5 | **74.2** | 68.0 | 66.0 |
| PneumoniaMNIST | 81.0 | 68.5 | **87.0** | 80.0 | 75.5 |

### Ablation Study

| Encoding Method | Average EE↓ | Average Accuracy↑ |
|----------|---------|------------|
| Angle | 0.65 | 85.3 |
| Amplitude | 0.55 | 82.1 |
| PCA | 0.56 | 84.6 |
| SQE | 0.41 | 82.7 |
| **ATP** | **0.38** | **87.5** |

### Key Findings
- ATP achieves the highest accuracy and lowest entanglement entropy simultaneously across almost all datasets and class pairs.
- Under depolarizing noise (3-10%), ATP and SQE demonstrate the highest robustness, with an accuracy drop of only 3 to 8 percentage points.
- Combined with adversarial training after FGSM adversarial attacks, ATP maintains the highest accuracy.
- On real quantum hardware (IBM Sherbrooke), ATP shows an average improvement of 7% compared to direct encoding.

## Highlights & Insights
- **Data-level rather than Circuit-level Pruning**: Diverging from the convention of "optimizing quantum circuit structures", this approach directly reduces redundancy from the input data, presenting a simpler and more straightforward methodology.
- **Noise Robustness**: Low-entanglement encoding schemes are inherently more robust against depolarizing noise, as they alleviate the amplification of cross-talk between qubits.
- **Validation on Real Hardware**: Experimental evaluation on IBM quantum computers strengthens the practical significance of the proposed method.

## Limitations & Future Work
- Only binary classification is validated; multi-class classification would require hybrid classical-quantum methods.
- The threshold optimization in ATP increases computational overhead by approximately 15%.
- Testing is currently limited to small-scale images, and its applicability to large-scale high-dimensional data remains to be validated.
- The integration with more complex encoding schemes (such as hybrid amplitude-angle encoding) has not yet been explored.

## Related Work & Insights
- **vs PCA**: Whereas PCA reduces dimensionality via linear transformations, ATP employs variance-based adaptive pruning, demonstrating superior adaptability to data structures.
- **vs SQE**: SQE encodes all features into a single qubit to minimize resource utilization, whereas ATP retains multiple qubits but prunes redundant features.
- The core concept can be transferred to input feature selection and data augmentation strategies in classical deep learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of pruning prior to quantum encoding is clear, though the method is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation spanning 4 datasets, noise, adversarial attacks, and real hardware.
- Writing Quality: ⭐⭐⭐⭐ The structure is coherent, though the explanation of quantum background concepts takes up a substantial portion.
- Value: ⭐⭐⭐⭐ Highly valuable to the quantum computing community, though attention from the mainstream CVPR audience might be limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AQER: A Scalable and Efficient Data Loader for Digital Quantum Computers](../../ICLR2026/physics/aqer_a_scalable_and_efficient_data_loader_for_digital_quantum_computers.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/physics/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICML 2025\] Compact Matrix Quantum Group Equivariant Neural Networks](../../ICML2025/physics/compact_matrix_quantum_group_equivariant_neural_networks.md)
- [\[ICCV 2025\] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers](../../ICCV2025/physics/resq_a_novel_framework_to_implement_residual_neural_networks_on_analog_rydberg_a.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/physics/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)

</div>

<!-- RELATED:END -->
