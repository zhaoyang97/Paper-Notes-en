---
title: >-
  [Paper Note] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits
description: >-
  [NeurIPS 2025][AI Safety][Concept Bottleneck Models] This paper theoretically establishes that the adversarial robustness of Neural Probabilistic Circuits (NPC) depends solely on the attribute recognition model and is independent of the probabilistic circuit. Building on this finding, it proposes RNPC, which achieves provably improved robustness via class-wise inference aggregation, significantly enhancing adversarial robustness while maintaining benign accuracy.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Concept Bottleneck Models"
  - "Probabilistic Circuits"
  - "Adversarial Robustness"
  - "Class-wise Reasoning"
  - "Interpretability"
date: 2026-05-08
content_hash: ed5d677c2857ba3f
---

# Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits

**Conference**: NeurIPS 2025
**arXiv**: [2509.20549](https://arxiv.org/abs/2509.20549)  
**Code**: [https://github.com/uiuctml/RNPC](https://github.com/uiuctml/RNPC)  
**Area**: AI Safety / Adversarial Robustness
**Keywords**: Concept Bottleneck Models, Probabilistic Circuits, Adversarial Robustness, Class-wise Reasoning, Interpretability

## TL;DR

This paper theoretically establishes that the adversarial robustness of Neural Probabilistic Circuits (NPC) depends solely on the attribute recognition model and is independent of the probabilistic circuit. Building on this finding, it proposes RNPC, which achieves provably improved robustness via class-wise inference aggregation, significantly enhancing adversarial robustness while maintaining benign accuracy.

## Background & Motivation

**Concept Bottleneck Models (CBMs)** provide interpretability by introducing an intermediate layer of human-understandable concepts. However, conventional CBMs employ linear predictors on the concept layer, which not only degrades performance but also **undermines robustness**. **Neural Probabilistic Circuits (NPC)** represent a new generation of CBMs comprising two modules: an attribute recognition model that predicts interpretable class attributes, and a probabilistic circuit that learns the joint distribution over attributes and classes while supporting scalable inference. NPCs achieve a favorable balance between performance and interpretability.

Nevertheless, the attribute recognition model in NPC remains a black-box neural network, making it susceptible to **adversarial attacks**—imperceptible input perturbations that manipulate attribute predictions and thereby affect final classification.

**Key Challenge**: The estimation error in NPC is compositional (errors from each module accumulate linearly). Is adversarial robustness similarly compositional? If not, can a more robust inference scheme be designed?

**Key Findings**: The adversarial robustness of NPC depends solely on the attribute recognition model; incorporating the probabilistic circuit yields robustness "for free" (in contrast to conventional CBMs, where the linear layer degrades robustness). This insight motivates the design of RNPC for further robustness improvement.

## Method

### Overall Architecture

RNPC shares the **same model architecture and training procedure** as NPC (attribute recognition model + probabilistic circuit). The sole distinction lies in the **inference strategy**: NPC uses node-wise aggregation, while RNPC uses class-wise aggregation.

### Key Designs

1. **Adversarial Robustness Analysis of NPC (Theorem 3.4)**:

    - Define the predictive perturbation $\Delta_{\theta,w}^{NPC}$ as the worst-case TV distance of the class distribution under adversarial attack.
    - Prove: $\Delta_{\theta,w}^{NPC} \leq \sum_{k=1}^K \mathbb{E}_X[\max_{\tilde{X}} d_{TV}(\mathbb{P}_{\theta_k}(A_k|X), \mathbb{P}_{\theta_k}(A_k|\tilde{X}))]$
    - Implication: The robustness upper bound is determined solely by the attribute recognition model; the probabilistic circuit does not affect robustness—a striking contrast to the compositional nature of NPC's estimation error.

2. **Class-wise Partition of the Attribute Space**:

    - Partition the high-probability attribute node set $V$ by the most likely class: $V = \bigcup_y V_y$
    - Define the inter-class Hamming distance $d_{i,j} = \min_{v_i \in V_i, v_j \in V_j} \text{Ham}(v_i, v_j)$
    - Define the radius $r = \lfloor \frac{d_{min}-1}{2} \rfloor$
    - Define the class neighborhood $\mathcal{N}(y,r)$: $V_y$ together with low-probability nodes within Hamming distance $r$ from $V_y$

3. **RNPC Class-wise Aggregation Inference (Equation 2)**:

    $\Phi_{\theta,w}(Y|X) = \sum_{\tilde{y}} \left(\mathbb{P}_\theta(A_{1:K} \in \mathcal{N}(\tilde{y},r)|X) \cdot \sum_{a_{1:K} \in V_{\tilde{y}}} \mathbb{P}_w(Y|A_{1:K}=a_{1:K})\right)$

    - Core intuition: When an attack perturbs attribute predictions, probability mass flows from the correct node $a_{1:K}^*$ to neighboring nodes. If these nodes remain within $\mathcal{N}(y,r)$ (i.e., the number of corrupted attributes $m \leq r$), the **aggregate weight of the entire class is nearly unchanged**, thereby preserving correct prediction.
    - Contrast with NPC's node-wise aggregation: a drop in a single node's weight directly affects the prediction.

4. **Theoretical Guarantees**:

    - **Robustness upper bound** (Lemma 4.6): The perturbation upper bound of RNPC satisfies $\Lambda_{RNPC} \leq \alpha_\epsilon$, whereas NPC's bound satisfies $\Lambda_{NPC} \leq \frac{|A_1|\cdots|A_K|}{2} \alpha_\epsilon$ (Theorem 4.7)—RNPC's bound is **exponentially smaller** than NPC's.
    - **Compositional estimation error** (Theorem 4.10): The benign error of RNPC remains decomposable as a linear combination of errors from the two modules.
    - **Robustness–accuracy tradeoff** (Theorem 4.11): The distance between the optimal RNPC and the true distribution is governed by the quality of the $V_y$ partition.

### Loss & Training

The attribute recognition model minimizes the sum of cross-entropy losses over all attributes. The probabilistic circuit uses LearnSPN for structure learning and CCCP for parameter optimization. Both modules are trained independently; NPC and RNPC share the same trained models.

## Key Experimental Results

### Main Results (Benign Accuracy)

| Dataset | CBM | DCR | NPC | RNPC |
|--------|------|------|------|------|
| MNIST-Add3 | 99.02 | 98.54 | 99.32 | **99.37** |
| MNIST-Add5 | 99.37 | 99.21 | 99.40 | **99.51** |
| CelebA-Syn | 99.83 | 99.45 | 99.95 | **99.95** |
| GTSRB-Sub | 99.42 | 99.42 | **99.57** | 99.49 |

RNPC matches or slightly surpasses NPC on benign inputs, confirming that the robustness–accuracy tradeoff is negligible in practice.

### Ablation Study (Adversarial Attack PGD-∞, ε=0.11)

| Configuration | MNIST-Add5 Adversarial Accuracy | Notes |
|------|---------|------|
| CBM | <20% | Linear predictor degrades robustness |
| DCR | <25% | Similar to CBM |
| NPC | <40% | Probabilistic circuit neither harms nor improves robustness |
| **RNPC** | **>80%** | Class-wise aggregation yields significant improvement |
| RNPC (r=0) | ~60% | Radius too small; insufficient fault tolerance |
| RNPC (r=r*=2) | **>80%** | Natural radius is optimal |
| RNPC (r=5=K) | 29.9% | Covers the full space; discriminability is lost |

### Key Findings

- NPC and RNPC substantially outperform CBM and DCR in robustness, confirming that the probabilistic circuit provides robustness "for free."
- RNPC achieves over 40% higher adversarial accuracy than NPC on MNIST-Add5, validating the theoretically predicted exponential advantage.
- The radius $r = r^*$ is the optimal choice; both decreasing and increasing it degrade adversarial accuracy.
- **Attack propagation**: On GTSRB, spurious correlations among attributes cause an attack on a single attribute to indirectly affect predictions of others, diminishing RNPC's advantage.
- Adversarial training can effectively mitigate attack propagation by decoupling spurious inter-attribute correlations.

## Highlights & Insights

- The finding that **"the probabilistic circuit provides robustness for free"** is particularly insightful and presents an interesting contrast to the compositional nature of estimation error.
- The class-wise aggregation idea in RNPC is intuitively natural: as long as perturbed attribute predictions remain within the correct class's "neighborhood," the final prediction is unaffected.
- The consistency between theoretical derivations and experimental results is strong, with the ablation study on radius $r$ matching theoretical predictions precisely.
- Inference-time complexity is actually reduced ($|V| \leq \prod_k |A_k|$).

## Limitations & Future Work

- Attack propagation arising from spurious inter-attribute correlations is pervasive in real-world data and can diminish RNPC's advantages.
- The datasets used are relatively small and partly synthetic; validation on larger-scale real-world datasets is needed.
- The attribute space partition and radius depend on the intrinsic structure of the data, which must be known or computable.
- Only white-box norm-bounded attacks are considered; other attack types (e.g., semantic attacks) are not addressed.

## Related Work & Insights

- **vs. standard CBM [Koh et al.]**: The linear predictor in CBM degrades robustness, whereas the probabilistic circuit in NPC/RNPC does not affect robustness.
- **vs. Sinha et al. (2023)**: Their objective is to keep attribute probabilities invariant under attack (via adversarial training), whereas RNPC aims to produce correct predictions even when attribute probabilities change.
- **vs. DCR**: DCR replaces the linear layer with an embedding layer but remains subject to the robustness limitations of linear predictors.

## Rating

- Novelty: ⭐⭐⭐⭐ Class-wise aggregation inference is a natural design choice, but the theoretical finding that "probabilistic circuits provide robustness for free" is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, diverse attacks, and rich ablations, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are tightly integrated; definitions and theorems are clearly presented.
- Value: ⭐⭐⭐⭐ Provides a theoretical framework and practical solution for robustness in interpretable models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](../../ICML2025/ai_safety/solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[CVPR 2026\] Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks](../../CVPR2026/ai_safety/towards_reliable_evaluation_of_adversarial_robustness_for_spiking_neural_network.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](../../ICML2025/ai_safety/understanding_model_ensemble_in_transferable_adversarial_attack.md)

</div>

<!-- RELATED:END -->
