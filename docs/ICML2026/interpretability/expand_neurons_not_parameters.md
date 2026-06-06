---
title: >-
  [Paper Note] Expand Neurons, Not Parameters
description: >-
  [ICML 2026][Interpretability][Neuron Expansion] By keeping the total number of non-zero parameters constant and "splitting" each neuron into $\alpha$ sparse sub-neurons that partition the original input edges…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Neuron Expansion"
  - "Superposition Hypothesis"
  - "Polysemanticity"
  - "Fixed Parameters"
  - "Feature Interference"
date: 2026-05-08
content_hash: e2f75159d98c6183
---

# Expand Neurons, Not Parameters

**Conference**: ICML 2026  
**arXiv**: [2510.04500](https://arxiv.org/abs/2510.04500)  
**Code**: Not disclosed  
**Area**: Interpretability / Superposition Hypothesis / Sparse Networks  
**Keywords**: Neuron Expansion, Superposition Hypothesis, Polysemanticity, Fixed Parameters, Feature Interference

## TL;DR
By keeping the total number of non-zero parameters constant and "splitting" each neuron into $\alpha$ sparse sub-neurons that partition the original input edges, feature interference (polysemanticity) between neurons can be significantly reduced. This leads to consistent accuracy improvements in Boolean tasks and real-world vision tasks like CLIP, CNN, and ImageNet.

## Background & Motivation

**Background**: While neural networks continue to scale, individual neurons are often "polysemantic"—a single neuron encodes multiple features, a phenomenon repeatedly observed in the mechanistic interpretability community. The superposition hypothesis posits that when the number of features exceeds the number of neurons, the network "squeezes" multiple features into the same neuron, leading to mutual interference that harms both interpretability and performance. Another line of research (the Lottery Ticket Hypothesis) found that sparse subnetworks can match or even exceed the accuracy of dense networks, suggesting that "structure" is more critical than "density."

**Limitations of Prior Work**: Existing work to mitigate superposition mostly stays at the "analysis" level (e.g., Sparse Autoencoders (SAEs) learning a sparse dictionary on activations) without changing the underlying network. While pruning or dynamic sparsification methods change structure, their primary goals are parameter compression or inference acceleration rather than "reducing polysemanticity." Using "reducing superposition interference" as an optimization objective to guide architectural design has not been directly addressed.

**Key Challenge**: Under a fixed parameter budget, the number of neurons and the connection density of each neuron are coupled. Making neurons more "specialized" requires more neurons, which typically implies more parameters; conversely, maintaining the parameter count forces the tolerance of polysemanticity. Is it possible to decouple these two axes?

**Goal**: Under the strict constraint of fixed non-zero parameters, make the network "wider" rather than "denser" to verify: (a) whether this reduces collision and interference between features; (b) whether reduced interference directly translates to accuracy gains; (c) whether these gains are most significant in "high superposition pressure" scenarios (few neurons, many features).

**Key Insight**: The root cause of feature interference is that they are forced to "share" the input edges of the same neuron. If the $d$ input edges of a neuron are split into $\alpha$ sub-neurons according to a disjoint partition, where each sub-neuron only sees $d/\alpha$ edges, the probability of two features colliding in the same sub-neuron decreases exponentially, while each feature still has a high probability of being covered by some sub-neuron. Theoretically, it can be proven that the collision probability is $\approx \alpha^{-(2k-1)}$ (where $k$ is the number of literals per clause), while the coverage remains almost unchanged.

**Core Idea**: Use "edge partitioning" as a mechanistic probe—expanding each neuron into $\alpha$ sparse sub-neurons without increasing non-zero parameters. These sub-neurons cover the original neuron's inputs but do not overlap, thereby maximizing feature coverage and minimizing collisions.

## Method

### Overall Architecture

The authors refer to this method as **Fixed Parameter Expansion (FPE)** and emphasize its role as a "mechanistic probe" rather than a deployable recipe. The pipeline consists of four steps:

1.  **Warmup Training**: A dense shallow MLP (input $\mathbf{x}\in\mathbb{R}^d$, hidden width $h$, output $C$ classes) is trained on the target task for 25 epochs to reach near-convergence and learn "stable mixed features."
2.  **Edge Partitioning Expansion**: Using an expansion factor $\alpha > 1$, the hidden layer is expanded to width $h' = \alpha h$. For the original $i$-th neuron's weight $\mathbf{w}_i$, it is copied to $\alpha$ sub-neurons. Then, $\alpha$ disjoint binary masks $\mathbf{m}_{(i_k)}\in\{0,1\}^d$ (satisfying $\sum_k \mathbf{m}_{(i_k)} = \mathbf{1}_d$) are used to partition the input dimensions among these sub-neurons, ensuring they share no input features. The second layer $\mathbf{W}_2 \in \mathbb{R}^{C\times h}$ is expanded to $\mathbf{W}_2'\in\mathbb{R}^{C\times h'}$ by copying each output weight $\alpha$ times.
3.  **Re-sparsification**: Copying the second layer introduces $(\alpha-1)C$ redundant parameters. The authors prune the weights with the smallest absolute values in $\mathbf{W}_1'$ and $\mathbf{W}_2'$ so that the total non-zero parameters strictly equal the original budget. After initialization, masks are no longer updated.
4.  **Fine-tuning**: The dense model and the FPE model are trained for another 25 epochs under identical optimization settings to compare accuracy and interference metrics.

### Key Designs

1.  **Disjoint Edge Partitioning (Fixed Parameter Expansion)**:
    *   **Function**: Expands the number of neurons from $h$ to $\alpha h$ while maintaining $\|\mathbf{W}_1'\|_0 = \|\mathbf{W}_1\|_0$, making each sub-neuron "see" only a disjoint subset of the original input.
    *   **Mechanism**: The weights of the original neuron are directly copied $\alpha$ times, and input dimensions are partitioned using $\alpha$ non-overlapping binary masks. The input support sets of any two sub-neurons are disjoint. In Boolean DNF tasks, the authors prove this reduces the expected collision rate of two clauses into the same sub-neuron to approximately $\alpha^{-(2k-1)}$, with almost no drop in clause coverage.
    *   **Design Motivation**: The authors adopt a feature channel coding perspective—abstract features are "encoded" by a set of neurons (rows) sharing sign patterns. Under a fixed number of rows, there is a finite capacity upper bound; more features inevitably lead to overlapping codes. FPE essentially "increases the number of available rows without increasing the non-zero parameter budget," pushing the feature capacity upper bound higher.

2.  **Mask Generation Strategy: clause-split vs. random-split**:
    *   **Function**: Determines which input dimensions each sub-neuron inherits.
    *   **Mechanism**: In Boolean tasks, the authors use *clause-aware split*—assigning all literals of the same clause to the same sub-neuron. In vision tasks, they compute the Gram matrix $G = \mathbf{W}_1^\top \mathbf{W}_1$ of the first layer, cluster its rows into "pseudo-feature groups," and perform balanced assignment. As a control, *random-split* partitions input dimensions randomly.
    *   **Design Motivation**: Theoretically, clause-split should be optimal as it suppresses collision probability to zero. However, the authors specifically include random-split to test if "simply reducing collisions without precise feature alignment" can capture most of the gains. Experiments confirm that random splitting outperforms the dense baseline in all settings, with clause-split providing only a marginal additional boost. This supports the mechanistic claim that "collision, rather than semantic alignment, is the key driver."

3.  **Interference Quantification: feature capacity + cosine similarity**:
    *   **Function**: Directly measures the strength of "superposition" from weight geometry to perform correlation analysis with accuracy gains.
    *   **Mechanism**: For each feature $i$, capacity is defined as $C_i = (W_{\cdot,i}\cdot W_{\cdot,i})^2 / \sum_j (W_{\cdot,i}\cdot W_{\cdot,j})^2$. The numerator is the "squared weight norm" of the feature itself, while the denominator includes the squared inner products with all other features; higher capacity indicates a larger "exclusive" representation subspace for that feature. Additionally, the average cosine similarity between all pairs of neuron weight vectors is calculated; lower values represent more "orthogonal" neurons and decoupled features. In Figure 3c, the authors perform least-squares regression of the fold change of these two metrics against relative accuracy improvement, identifying a very strong correlation that quantitatively links the chain: "width $\uparrow \to$ interference $\downarrow \to$ accuracy $\uparrow$."
    *   **Design Motivation**: Looking at accuracy gains alone is insufficient to prove that improvement is due to reduced superposition; it could merely be a benefit of width. Feature capacity and cosine similarity are "superposition pressure" metrics read directly from the geometric structure of $\mathbf{W}_1$, providing independent evidence for the mechanistic claim.

### Loss & Training

The task uses standard classification losses: sigmoid + BCE for binary classification, and softmax + cross-entropy for multiclass. No additional regularization terms are used. Training involves two stages: 25-epoch warmup $\to$ application of FPE $\to$ 25-epoch fine-tuning. The dense baseline is trained for a full 50 epochs under the same schedule for a fair comparison. Masks are fixed once determined during FPE initialization (ablations showed that allowing mask updates does not change the conclusion).

## Key Experimental Results

### Main Results

| Task / Setting | Config | Dense Baseline | FPE (random) | FPE (clause/feature) | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Boolean DNF, 8 clauses, 8 neurons, $\alpha=2$ | symbolic | 78.7% | 88.7% | **99.4%** | +26% |
| CLIP-CIFAR-100, 32 pre-expand neurons, $\alpha=4$ | frozen embed | — | $\approx$ matches $1.2\times$ param dense model | — | Param Equivalent Double |
| FashionMNIST / CLIP-ImageNet-100 / CLIP-ImageNet-1k | Multi-width | baseline | Consistent gain | Consistent gain | Significant |
| CIFAR-100 + Trainable CNN backbone (256/512 dim) | joint learning | baseline | Consistent gain | Consistent gain | Max gain at min width |

For CLIP-CIFAR-100 with a small number of neurons, FPE can nearly double the accuracy (Figure 4b). Furthermore, random-split and feature-based split perform almost identically on real data—contrasting with the clear lead of clause-split in Boolean tasks—suggesting that "pseudo-feature clustering" on real data does not perfectly recover the "true" feature structure.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
| :--- | :--- | :--- |
| Increasing $\alpha$ ($2 \to 4$) | Gains continue to rise | Validates theoretical prediction of exponential interference decay with $\alpha$ |
| Increasing clause count (fixed 8 neurons) | Gains rise then saturate at $\approx 16$ clauses | Even wider models cannot fully decouple extreme high superposition |
| Increasing neuron count (fixed 8 clauses) | Gains monotonically decrease | More neurons $\to$ lower superposition pressure $\to$ decreased necessity for FPE |
| Same sub-neuron count but overlapping inputs | Significantly worse than FPE (Table A14) | Proves disjointness is critical, not just simple widening |
| vs DropConnect (same non-zero param budget) | FPE matches or outperforms | Rules out explanation as "just random sparse regularization" |
| Changing split timing | Earlier split is better, late split still outperforms dense | Early feature specialization yields greater benefits |

### Key Findings

*   **Collision, not semantic alignment, is the primary cause**: Random-split outperforms the dense baseline in all settings and matches feature-based split on real vision tasks. This downgrades "correct feature identification" from a necessary condition to a "bonus."
*   **Interference metrics correlate strongly with accuracy**: The degree of feature capacity increase and cosine similarity decrease can linearly predict relative accuracy gains (high $R^2$), a rare instance of quantitatively linking mechanistic metrics to performance.
*   **Greater gain under high superposition pressure**: Scans across clause/neuron ratios in Boolean tasks and category/width scans in CIFAR-100 show that FPE's relative gains are largest when the feature-to-neuron ratio is high. When the dense model already has surplus capacity, gains naturally diminish.
*   **Hardware Friendliness**: The configuration of fixed non-zero parameters with more neurons inherently suits modern accelerators where memory transfer is the bottleneck (provided sparse kernel support is available).

## Highlights & Insights

*   **Applying interpretability insights "in reverse" for architectural design**: Mechanistic interpretability has previously been used almost exclusively to explain trained models. This paper uses the predictions of the superposition hypothesis to guide design choices ("more neurons + sparser edges"), representing a step from "diagnosis" to "prescription"—whereas work like SAEs can help understand superposition, FPE directly translates this understanding into measurable performance gains.
*   **Analytical collision estimates match measured accuracy gains**: The theory yields a collision decay rate of $\alpha^{-(2k-1)}$, and the experimental gain curves across varying $\alpha$ align with the qualitative trends of this prediction. Such a chain of "theory first $\to$ experimental confirmation" is becoming increasingly rare in the LLM era.
*   **Answering "Is simply breaking it up enough?"**: Through strict controlled experiments comparing disjoint and non-disjoint inputs, the authors prove that widening or adding randomness is insufficient. One must explicitly ensure sub-neurons do not share inputs, distinguishing FPE from DropConnect or general sparse regularization.

## Limitations & Future Work

*   **Proof of Concept**: The authors acknowledge that FPE is a "mechanistic proof of concept," not a deployment-ready method. Real-world acceleration depends on sparse kernels. Experiments are concentrated on small models and controlled settings, not yet reaching Transformer scales.
*   **Heuristic Feature Splitting**: Gram clustering on real data neither guarantees recovery of "true" feature structures nor provides a clear lead over random-split, indicating a need for more precise feature attribution tools (where methods like SAEs could potentially help).
*   **Structural Assumptions**: Disjoint input partitioning might be unfriendly to low-dimensional or tightly coupled features (hard-splitting the signal). The effects of FPE on dense embeddings or multimodal token streams remain to be verified.
*   **Future Directions**: Combining FPE with SAEs (using SAEs to extract true features $\to$ applying clause-style split); extending to the depth dimension (partitioning channels between layers); and treating FPE as a "refinement" step for pre-trained large models rather than a train-from-scratch initialization strategy.

## Related Work & Insights

*   **vs. Superposition / SAE Series (Elhage 2022; Cunningham 2023)**: They use sparse dictionaries or toy models to "analyze" superposition; this work is the first to use the same perspective to "reverse-engineer" the underlying network architecture to reduce superposition and directly link interference reduction to accuracy gains.
*   **vs. Lottery Ticket Hypothesis / Pruning (Frankle & Carbin 2018; Han 2015; SparseGPT 2023)**: Pruning trains dense then cuts for compression; FPE does the opposite—trains dense, then "expands and sparsifies" to reduce interference rather than compress, while strictly maintaining the total non-zero parameter count.
*   **vs. Network Growth (Net2Net, dynamic networks)**: Growth methods usually increase total parameters. FPE explicitly forbids parameter growth, decoupling neuron count from parameter count as an independent architectural axis.
*   **vs. DropConnect (Wan 2013)**: DropConnect treats random sparsity as regularization and restores density during inference; FPE is a deterministic, permanent disjoint sparse architecture explicitly aimed at "reducing feature collision."

## Rating
*   **Novelty**: ⭐⭐⭐⭐ First to implement superposition hypothesis as an architectural design principle, though "wide vs. dense" comparisons under fixed parameters have precedents in works like the Lottery Ticket Hypothesis or Golubeva 2020.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Spans Boolean $\to$ CLIP $\to$ ImageNet-1k $\to$ joint CNN, providing mechanistic evidence via interference metrics. However, excludes Transformers and modern large models (scale is small).
*   **Writing Quality**: ⭐⭐⭐⭐ The progression from "theoretical collision estimation $\to$ Boolean verification $\to$ real-world task generalization $\to$ interference metric regression" is clear; the positioning of FPE ("probe, not recipe") is honest.
*   **Value**: ⭐⭐⭐⭐ Provides actionable evidence for the causal chain of "superposition $\to$ polysemanticity $\to$ performance loss" and suggests a new hardware-friendly path for sparsification worthy of testing on Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLMs Lean on Priors, Not Programming Language Semantics](llms_lean_on_priors_not_programming_language_semantics.md)
- [\[ICML 2026\] The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)
- [\[ICML 2026\] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](position_zeroth-order_optimization_in_deep_learning_is_underexplored_not_underpo.md)
- [\[ACL 2026\] Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](../../ACL2026/interpretability/linear_probes_detect_task_format_not_reasoning_mode_in_language_model_hidden_sta.md)
- [\[ICML 2026\] The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level](the_expert_strikes_back_interpreting_mixture-of-experts_language_models_at_exper.md)

</div>

<!-- RELATED:END -->
