---
title: >-
  [Paper Note] Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer
description: >-
  [ACL2025][Task Vector] Proposes Neural Parameter Search (NPS), which improves the pruning efficiency of fine-tuned models by searching for optimal weight coefficients in the subspaces of the task vector. It achieves significant improvements across three scenarios: knowledge transfer (+1.5%), model merging (+2.1%), and compression (40% efficiency improvement).
tags:
  - "ACL2025"
  - "Task Vector"
  - "Model Pruning"
  - "Evolutionary Search"
  - "Knowledge Transfer"
  - "Model Merging"
date: 2026-05-08
content_hash: 6ae6db9536ae2134
---

# Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer

**Conference**: ACL2025  
**arXiv**: [2505.18713](https://arxiv.org/abs/2505.18713)  
**Code**: [NPS-Pruning](https://github.com/duguodong7/NPS-Pruning)  
**Area**: Others  
**Keywords**: Task Vector, Model Pruning, Evolutionary Search, Knowledge Transfer, Model Merging  

## TL;DR

Proposes Neural Parameter Search (NPS), which improves the pruning efficiency of fine-tuned models by searching for optimal weight coefficients in the subspaces of the task vector. It achieves significant improvements across three scenarios: knowledge transfer (+1.5%), model merging (+2.1%), and compression (40% efficiency improvement).

## Background & Motivation

- **Redundancy of Fine-Tuned Models**: Parameter updates in fine-tuned models contain a significant amount of redundancy, and different task vector subspaces contribute very differently to performance.
- **Threefold Value of Pruning for Knowledge Management**:
  1. Reduces conflict between the fine-tuned model and the pre-trained model during knowledge transfer, enhancing resilience against catastrophic forgetting.
  2. Reduces parameter interference when merging multiple fine-tuned models, improving multi-task generalization.
  3. Lowers storage costs while maintaining multi-task performance.
- **Limitations of Prior Work**:
    - TIES prunes parameters directly based on magnitude without considering the uneven contributions of different subspaces.
    - DARE randomly selects and scales parameters, lacking fine-grained control.
    - Model Tailor generates masks based on saliency and sensitivity, which incurs high computational overhead.
- **Key Observation**: Parameter subspaces in different magnitude intervals within the task vector contribute significantly differently to model performance (Figure 2), necessitating a more fine-grained re-weighting strategy.

## Method

### Overall Architecture

The core mechanism of NPS is as follows:
1. Compute the task vector $\tau = \theta_{ft} - \theta_{pre}$ (the difference between the fine-tuned and pre-trained parameters).
2. Partition $\tau$ into $M$ subspaces based on parameter magnitudes.
3. Search for the optimal weight of each subspace using an evolutionary algorithm.
4. Prune parameters based on the re-weighted magnitudes.
5. Apply the pruned task vector to three different scenarios.

### Task Vector Subspace Decomposition

The task vector $\tau$ is sorted by parameter magnitude and decomposed into $M$ disjoint subspaces:

$$\tau = \sum_{m=1}^{M} q_m$$

A learnable weight is then assigned to each subspace:

$$\tau = \sum_{m=1}^{M} w_m * q_m$$

Initially, all weights are set to 1, and the optimal weight combination is subsequently search-optimized.

### Evolutionary Search (CMA-ES)

Optimization is performed using Covariance Matrix Adaptive Evolution Strategies (CMA-ES):
- Gradient-free, lightweight, and highly efficient.
- Dynamically adjusts the search distribution via the covariance matrix.
- Updates weights based on the validation accuracy on a calibration dataset.
- Obtains the optimal weights $\{w_1, ..., w_M\}$ upon convergence.

### Magnitude Pruning

After the search is completed, pruning is performed based on the adjusted parameter magnitudes:

$$m_d = \begin{cases} 1, & \text{if } \tau_d \geq \text{sorted}(\tau)[r \times d] \\ 0, & \text{otherwise} \end{cases}$$

The final pruned model is: $\hat{\theta}_{ft} = \theta_{pre} + m \odot \tau$

where $r$ is the sparsity ratio (the proportion of retained parameters).

### Three Application Scenarios

**1. Knowledge Transfer**: Mitigating catastrophic forgetting

$$\hat{\theta}_{ft} = \theta_{pre} + \lambda \cdot m \odot \tau$$

Pruning reduces interference between the fine-tuned model and the pre-trained model, where $\lambda$ controls the transfer intensity.

**2. Knowledge Fusion**: Multi-task model merging

$$\theta_m = \theta_{pre} + \sum_{i=1}^{n}(\lambda_i \cdot m_i \odot \tau_i) / \sum_{i=1}^{n}\lambda_i$$

Pruned task vectors of each task are weight-averaged, where $\lambda_i$ is optimized via evolutionary strategies.

**3. Knowledge Compression**: Efficient storage

$$\hat{\theta}_{ft_1}, ..., \hat{\theta}_{ft_n} = \theta_{pre} + [m_1 \odot \tau_1, ..., m_n \odot \tau_n]$$

Only the pre-trained model, the sparse task vectors, and the binary masks need to be stored, which substantially reduces storage costs.

## Experimental Results

### Knowledge Transfer: LLaVA-1.5 (Vicuna-7B), 10% Keep Rate

| Method | Avg | H-score | Modified Parameters |
|------|-----|---------|-----------|
| Zero-shot | 42.33 | 29.05 | - |
| Fine-tune | 56.42 | 63.40 | 2.7B |
| DARE | 60.12 | 36.64 | 273M |
| Grafting | 61.56 | 60.03 | 273M |
| Model Tailor | 61.87 | 66.94 | 273M |
| **NPS** | **62.38** | **67.54** | **273M** |

While modifying only 273M parameters (10% keep rate), NPS surpasses all baselines in both Avg and H-score, effectively mitigating catastrophic forgetting.

### Knowledge Fusion: Multi-Scenario Model Merging

| Setup | Task Arithmetic | TIES | Consensus TIES | **NPS** |
|------|----------------|------|----------------|---------|
| T5-Base (7-task NLP) | 73.0 | 73.6 | 73.4 | **75.7(+2.1)** |
| T5-Large (7-task NLP) | 80.2 | 80.3 | 80.5 | **82.1(+1.6)** |
| (IA)³ (11 PEFT tasks) | 63.9 | 66.8 | 66.6 | **68.2(+1.4)** |
| LLaMa2 (3 LLM tasks) | 30.4 | 34.2 | 34.4 | **35.3(+0.9)** |
| ViT-B/32 (8 Vision tasks) | 70.1 | 73.6 | 73.3 | **76.5(+3.0)** |
| ViT-L/14 (8 Vision tasks) | 84.5 | 86.0 | 86.2 | **87.6(+1.4)** |
| T5-Base (5 Sentiment Domains) | 33.6 | 34.5 | 34.4 | **35.7(+1.3)** |
| RoBERTa (5 Sentiment Domains) | 38.3 | 39.7 | 39.8 | **40.9(+0.9)** |

NPS achieves the best results across all 8 experimental setups, spanning various modalities and tasks including NLP, vision, multi-modality, and sentiment analysis.

### Knowledge Compression

- Evaluated compression performance across varying numbers of task combinations on 8 vision tasks using ViT-B/32.
- NPS maintains near-original accuracy across all task-count levels (2 to 8 tasks).
- Compared to baseline methods, compression efficiency is improved by approximately 40%.
- NPS maintains accuracy even at a keep rate as low as 0.04, whereas TIES and DARE drop sharply below 0.2.

### Pruning Efficiency Comparison

- When the keep rate is >0.2, most methods maintain performance comparable to the fine-tuned model.
- When the keep rate is <0.2, accuracy drops sharply.
- NPS maintains the original model's accuracy even at a keep rate as low as 0.04.
- Its tolerance to extremely low keep rates far exceeds baselines such as TIES and DARE.

Refers to "sparsity ratio" in the original context as the percentage of kept parameters.

## Highlights & Insights

1. **Value of the Key Observation**: The uneven contribution of different subspaces (Figure 2) serves as the foundation of this work. The observation is simple yet highly insightful.
2. **Gradient-Free Search**: Weight searching using the CMA-ES evolutionary algorithm bypasses the overhead of gradient computation, making the method applicable to extremely large models.
3. **One Method, Three Uses**: The same NPS approach naturally extends to three scenarios—knowledge transfer, fusion, and compression—demonstrating the high versatility of the method.
4. **Robustness to Low Keep Rates**: Retaining accuracy even when keeping only 4% of the parameters indicates that re-weighting makes pruning decisions significantly more precise.
5. **Cross-Modal Validation**: Comprehensively covers multiple modalities including NLP (T5, LLaMa2), vision (ViT, CLIP), and multi-modality (LLaVA).
6. **Compatibility with PEFT**: Experiments on (IA)³ demonstrate that NPS is also applicable to merging adapters from parameter-efficient fine-tuning.

## Limitations & Future Work

1. **Search Overhead**: Although CMA-ES is gradient-free, it requires multiple model evaluations. The number of subspaces $M$ and the population size affect overall efficiency.
2. **Calibration Data Dependence**: Optimization results depend heavily on the quality and representativeness of the calibration data.
3. **Hyperparameter Sensitivity**: The choice of the number of subspaces $M$ impacts the final performance, which is not thoroughly analyzed in the paper.
4. **Magnitude-Based Decomposition Assumption**: Grouping parameters simply by sorting their magnitudes may not be the optimal way to partition subspaces.
5. **Incomplete Evaluation**: LLM experiments only merge 3 tasks, which is relatively small in scale.
6. **Lack of Comparison with Pruning-and-Fine-Tuning**: No comparison is made with traditional workflow of first pruning then continuing to train.

## Related Work & Insights

- **Task Vector**: Task arithmetic proposed by Ilharco et al., which achieves knowledge editing via parameter differences.
- **Model Merging**: Task Arithmetic, TIES-Merging, Fisher Merging, RegMean, etc.
- **Model Pruning**: Traditional pruning such as SparseGPT and Wanda, as well as methods dedicated to fine-tuned models like DARE and Model Tailor.
- **Knowledge Transfer**: WiSE-FT, Model Tailor, etc., which leverage pre-trained model parameters to improve transfer.
- **TALL-masks**: Localizing key task information within the task vector via masking.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ The combination of subspace re-weighting and evolutionary search is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation spanning three scenarios, multiple modalities, and various models.
- **Value**: ⭐⭐⭐⭐ Offers direct practical value for model deployment and multi-task merging.
- **Writing Quality**: ⭐⭐⭐⭐ Intuitive and clear methodology, with systematically organized experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Better Evaluation for Generated Patent Claims](patclaimeval_patent_evaluation.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](../../ICML2025/others/glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[ICML 2025\] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](../../ICML2025/others/function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)

</div>

<!-- RELATED:END -->
