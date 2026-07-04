---
title: >-
  [Paper Note] Autoencoder-Based Hybrid Replay for Class-Incremental Learning
description: >-
  [ICML 2025][Image Generation][Class-Incremental Learning] Proposed the Autoencoder-Based Hybrid Replay (AHR) strategy, which utilizes a Hybrid Autoencoder (HAE) to compress and store samples in the latent space rather than the original input space. By combining Charged Particle System Energy Minimization (CPSEM) and the Repulsion Force Algorithm (RFA) to incrementally embed new class centroids, it reduces the memory complexity from $\mathcal{O}(t)$ to $\mathcal{O}(0.1t)$ in t…
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Class-Incremental Learning"
  - "Hybrid Replay"
  - "Autoencoder"
  - "Repulsion Force Algorithm"
  - "Latent Space Compression"
date: 2026-05-08
content_hash: a81141895403675a
---

# Autoencoder-Based Hybrid Replay for Class-Incremental Learning

**Conference**: ICML 2025  
**arXiv**: [2505.05926](https://arxiv.org/abs/2505.05926)  
**Code**: Source code is accompanied with the paper and will be open-sourced  
**Area**: Image Generation  
**Keywords**: Class-Incremental Learning, Hybrid Replay, Autoencoder, Repulsion Force Algorithm, Latent Space Compression

## TL;DR

Proposed the Autoencoder-Based Hybrid Replay (AHR) strategy, which utilizes a Hybrid Autoencoder (HAE) to compress and store samples in the latent space rather than the original input space. By combining Charged Particle System Energy Minimization (CPSEM) and the Repulsion Force Algorithm (RFA) to incrementally embed new class centroids, it reduces the memory complexity from $\mathcal{O}(t)$ to $\mathcal{O}(0.1t)$ in the worst-case scenario while maintaining SOTA performance.

## Background & Motivation

Class-Incremental Learning (CIL) faces two core challenges: **Catastrophic Forgetting (CF)** and **Task Confusion (TC)**. Existing strategies each have their own limitations:

- **Exemplar Replay**: Performs best but incurs an $\mathcal{O}(t)$ memory overhead, requiring the storage of large amounts of raw data samples, leading to poor scalability.
- **Generative Replay**: Requires small memory, but the generated pseudo-data has poor quality, leading to severe forgetting; furthermore, it shifts the challenge of incremental learning of discriminant models onto the incremental learning of generative models.
- **Generative Classifier**: Requires continuous expansion of architecture with a memory footprint of $\mathcal{O}(t)$, failing to integrate different task features within a single model.

Core observation: An $\mathcal{O}(t)$ memory or computational complexity is inevitable—each time a new task is learned, there must be a mechanism to monitor the $t-1$ constraints imposed by previous tasks on network weights, otherwise the knowledge will be overwritten. Therefore, the key question is: Can the memory complexity be significantly reduced while maintaining $\mathcal{O}(t)$ computational complexity?

| Strategy Type | Memory Complexity | Computational Complexity | Performance |
|---|---|---|---|
| Generative Replay | $\mathcal{O}(cte)$ | $\mathcal{O}(t)$ | non-SOTA |
| Generative Classifier | $\mathcal{O}(t)$ | $\mathcal{O}(cte)$ | non-SOTA |
| Exemplar Replay | $\mathcal{O}(t)$ | $\mathcal{O}(t)$ | SOTA |
| **Hybrid Replay (AHR)** | $\mathcal{O}(0.1t)$ | $\mathcal{O}(t)$ | **SOTA** |

## Method

### Overall Architecture

For each incoming new task $T_\ell$, AHR executes three core steps:

1. **CCE Placement**: Utilizes RFA based on the Euler-Lagrange equation to solve for the optimal positions, allocating centroid locations $\mathcal{P}_\ell = \{p_\ell^j\}_{j=1}^{J_\ell}$ in the latent space for classes of the new task.
2. **HAE Training**: Copies the model from the previous step as a new model, and co-trains it on the new task data and the replay data decoded from old memories, optimizing the reconstruction loss + clustering loss + distillation loss.
3. **Memory Population**: Leverages the Herding strategy to select and store the most representative encoded samples in the latent space.

Key difference: AHR **does not store original raw data**, but stores low-dimensional vectors after **encoding the data into the latent space**. The decoder is designed to **memorize original data pairs** (i.e., deterministic reconstruction) instead of seeking to generalize and generate new samples like a VAE.

### Key Designs

#### Hybrid Autoencoder (HAE)

HAE possesses both discriminative and generative capabilities:

- **Encoder** $\phi: \mathbb{R}^n \to \mathbb{R}^m$: Maps input data to low-dimensional latent representations.
- **Decoder** $\psi: \mathbb{R}^m \to \mathbb{R}^n$: Reconstructs data from latent representations, designed to **memorize** $(z, x)$ pairs rather than generalize.

Design decision: **Intentionally avoids using VAE** because the goal is not to generate new images, but to precisely memorize the encoding-decoding mapping of the training data, enabling the decoder to recover original data with minimal loss.

#### Charged Particle System Energy Minimization (CPSEM) and Repulsion Force Algorithm (RFA)

Optimal distribution of class centroids in the latent space is achieved through physical analogy. Each class centroid embedding (CCE) is viewed as a charged particle, modeled using Coulomb interaction energy:

$$\mathcal{U} = \sum_{i,j=1}^{I,J_i} \frac{(q_i^j)^2}{2} \sum_{i',j' \neq i,j} \frac{1}{\|p_{i'}^{j'} - p_i^j\|}$$

Each particle also possesses kinetic energy $\mathcal{K}_i^j = \frac{1}{2}m_i^j \|v_i^j\|^2$. The optimization goal is to minimize the total energy $\mathcal{E} = \mathcal{U} + \mathcal{K}$, solving the equations of motion for particles via the variational method and Euler-Lagrange equations:

$$\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial v_i^j}\right) = \frac{\partial \mathcal{L}}{\partial p_i^j}$$

Core workflow of RFA:
1. Initialize the new class centroid positions to the mean of the encoder's current outputs.
2. Iteratively calculate the repulsion force vectors among all centroids.
3. Update velocity and position according to the forces until the system energy converges.

Key advantage: Distinct from iCaRL, the CCE of AHR **remains unchanged once placed**, ensuring the stability of the latent space structure.

#### Classification in the Test Phase

Classification is performed directly in the latent space using Euclidean distance:

$$\text{argmin}_{i,j} \|\phi(w_I^*, x) - p_i^j\|$$

Inference is accomplished without decoding—after the encoder maps the sample to the latent space, the nearest class centroid is found.

### Loss & Training

The total loss consists of three components:

**1. HAE Loss (Eq. 1)**:

$$L(x, \hat{x}, z) = \underbrace{\sum \|x_i^{j,k} - \hat{x}_i^{j,k}\|^2}_{L_x: \text{Reconstruction Loss}} + \lambda \underbrace{\sum \|z_i^{j,k} - p_i^j\|^2}_{L_z: \text{Clustering Loss}}$$

- $L_x$: Minimizes the L2 distance between the input and the reconstructed data.
- $L_z$: Pulls samples from the same class closer to their corresponding CCE location in the latent space, with $\lambda$ as a hyperparameter.

**2. Distillation Loss (Data Regularization)**:

$$\|\phi(w_{\ell-1}, D) - \phi(w_\ell, D)\| + \|\psi(v_{\ell-1}, \phi(w_{\ell-1}, D)) - \psi(v_\ell, \phi(w_\ell, D))\|$$

Constrains the consistency of encoder and decoder outputs before and after, preventing catastrophic forgetting.

**3. Training Details**:
- In each SGD iteration, $1/\ell$ of the data comes from the new task, and $(\ell-1)/\ell$ is obtained via on-the-fly decoding from memory.
- Balanced training is adopted (adapted from EEIL).
- Fixed exemplars memory (non-growing) is utilized, where the number of samples per class decreases as tasks increase.
- Optimizer is Adam.
- Encoder: A 2-layer 400 ReLU fully-connected network for MNIST; ResNet-32 for large datasets.
- Decoder: A 3-layer CNN is used for large datasets.

## Key Experimental Results

### Main Results

Compared with more than 10 baseline methods across 5 benchmarks (under fixed compute, matched parameter size, and equal memory budget):

| Dataset | Metric | AHR | Prev. SOTA (REMIND+) | Gain |
|---|---|---|---|---|
| MNIST(5/2) | Accuracy | **97.53** | 95.62 | +1.91 |
| BalancedSVHN(5/2) | Accuracy | **93.02** | 92.15 | +0.87 |
| CIFAR-10(5/2) | Accuracy | **77.12** | 75.49 | +1.63 |
| CIFAR-100(10/10) | Accuracy | **54.43** | 52.36 | +2.07 |
| miniImageNet(20/5) | Accuracy | **48.09** | 45.02 | +3.07 |

AHR achieves the best performance across all 5 benchmarks, especially showing the most significant advantage on the most challenging miniImageNet (+3.07%).

### Ablation Study

| Configuration | CIFAR-100 | miniImageNet | Description |
|---|---|---|---|
| AHR (Full) | **54.43** | **48.09** | Encoding compression + RFA |
| AHR-lossy-mini | 50.29 | 42.39 | Same number of samples, lossy compression |
| AHR-lossless-mini | 50.85 | 42.88 | Same number of samples, lossless |
| AHR-lossless | 56.71 | 49.70 | Same (larger) number of samples, lossless (upper bound) |
| AHR-contrastive | 51.98 | 44.60 | Replaced RFA with contrastive loss |
| AHR-GMM | 49.48 | 42.52 | Replaced RFA with GMM |

### Key Findings

1. **Diversity gains from compression outweigh quality loss**: The improvement of AHR-lossy-mini $\to$ AHR (+4.14/+5.70) is significantly larger than the improvement of AHR $\to$ AHR-lossless (+2.28/+1.61), indicating that "more decoded samples" is more crucial than "perfect samples".
2. **RFA significantly outperforms alternative solutions**: RFA vs. contrastive loss (+2.45/+3.49), RFA vs. GMM (+4.95/+5.57), because RFA can systematically embed new class centroids with minimal displacement.
3. **Advantages are more prominent in small memory settings**: The smaller the memory size, the larger the gap between AHR and baselines.
4. **Extremely low decoder overhead**: The 3-layer CNN decoder has only 1.4-1.8M parameters, which is negligible compared to the memory required to store original raw samples. Under the same total memory budget, AHR can store **7-10x** more encoded samples.
5. **Optimal resource efficiency**: On CIFAR-100, AHR achieves 54.43% using 462 min / 1.4M decoder + 4.6M exemplars, whereas BiC only achieves 52.12% using 473 min / 6M.

## Highlights & Insights

- **Physics-inspired latent space organization**: Analogizes class centroids to charged particles, utilizing Coulomb repulsion to achieve natural separation among incremental tasks—which is more effective than both contrastive learning and GMMs, and the centroids remain unchanged once placed.
- **"Memorizing" decoder vs. generative**: A clever design decision—the decoder prioritizes precise memorization over generalization, avoiding the problem of low-quality pseudo-data commonly seen in generative replay.
- **Direct classification in the latent space**: Eliminates the need for sorting after decoding, leading to high inference efficiency and avoiding the propagation of decoding errors.
- **Orthogonal to existing exemplar replay**: As a compression layer, AHR can be directly integrated into existing exemplar replay strategies.

## Limitations & Future Work

1. **Decoder quality still has an upper bound**: AHR-lossless consistently outperforms AHR, indicating that lossy compression remains a performance bottleneck; more powerful decoder architectures (such as Transformer decoders) might further narrow this gap.
2. **Validated only on image classification**: Its application on other modalities such as NLP or time series remains unexplored.
3. **Computational overhead of RFA**: The particle simulation of CPSEM may slow down when the number of classes is extremely large (due to $O(C^2)$ force calculations).
4. **Rigidity of fixed CCE**: Once placed, centroids are not adjusted, which could lead to crowding in the latent space in extremely long task sequences.
5. **Pre-trained backbones unexplored**: All experiments evaluated networks trained from scratch; incorporating pre-trained models might yield greater improvements.

## Related Work & Insights

- **iCaRL**: AHR borrows its framework of exemplar replay + distillation loss, but substitutes raw-space storage with latent-space storage.
- **REMIND/REMIND+**: Also belonging to hybrid replay, but compresses middle feature layers and performs classification after decoding; AHR is more efficient by classifying directly in the latent space.
- **i-CTRL**: Organizes the latent space using Linear Discriminative Representation, whereas the RFA of AHR is superior.
- **Insights for continual learning**: Data diversity (more samples) might be more critical than data quality (more precise samples)—this finding offers general guidance for memory-constrained scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ - Physics-inspired RFA for latent space organization is novel, though hybrid replay frameworks have precedent.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ - Highly comprehensive, featuring 5 benchmarks, over 10 baselines, detailed ablations, and resource analysis.
- Writing Quality: ⭐⭐⭐⭐ - Clearly structured with complete algorithm pseudocode, but the notation is somewhat dense.
- Value: ⭐⭐⭐⭐ - The method is practical and can be integrated into existing strategies, though it is limited to image classification scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Unsupervised Learning for Class Distribution Mismatch (UCDM)](unsupervised_learning_for_class_distribution_mismatch.md)
- [\[ECCV 2024\] Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](../../ECCV2024/image_generation/diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](../../CVPR2025/image_generation/generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](../../NeurIPS2025/image_generation/coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[ICML 2025\] Directed Graph Grammars for Sequence-based Learning](directed_graph_grammars_for_sequence-based_learning.md)

</div>

<!-- RELATED:END -->
