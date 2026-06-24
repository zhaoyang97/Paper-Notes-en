---
title: >-
  [Paper Note] Learning on Model Weights using Tree Experts
description: >-
  [CVPR 2025][Interpretability][Weight Space Learning] Discovers that most public models belong to a few Model Trees (fine-tuned from common ancestors), and learning weights within the same Tree is much simpler than across Trees. This paper proposes ProbeX, the first lightweight probing method targeting single hidden layer weights. Through Tucker tensor decomposition, it achieves a 30x reduction in parameter size and realizes the first zero-shot model classification (89.8% accu…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Weight Space Learning"
  - "Model Tree"
  - "Probing Expert"
  - "Zero-Shot Model Classification"
  - "Model Zoo Search"
date: 2026-05-08
content_hash: 4cf45658fecdc8e8
---

# Learning on Model Weights using Tree Experts

**Conference**: CVPR 2025  
**arXiv**: [2410.13569](https://arxiv.org/abs/2410.13569)  
**Code**: [https://horwitz.ai/probex/](https://horwitz.ai/probex/)  
**Area**: Interpretability  
**Keywords**: Weight Space Learning, Model Tree, Probing Expert, Zero-Shot Model Classification, Model Zoo Search

## TL;DR
Discovers that most public models belong to a few Model Trees (fine-tuned from common ancestors), and learning weights within the same Tree is much simpler than across Trees. This paper proposes ProbeX, the first lightweight probing method targeting single hidden layer weights. Through Tucker tensor decomposition, it achieves a 30x reduction in parameter size and realizes the first zero-shot model classification (89.8% accuracy) by aligning model weights with text representations.

## Background & Motivation

1. **Background**: Over a million public models exist on Hugging Face, but most lack adequate documentation, making it difficult for users to determine which model suits their tasks. Weight space learning (metanetwork) attempts to infer model functionality directly from model weights.
2. **Limitations of Prior Work**: Model weights are affected by massive nuisance factors during optimization (e.g., neuron permutation, weight initialization), making cross-model learning extremely difficult. Existing methods (like Neural Graphs) either fail to scale to large models or perform close to random accuracy.
3. **Key Challenge**: Model weights with different initializations differ drastically; even with the same training data, semantic information in weight space is overwhelmed by noise. In reality, however, most public models are not randomly initialized—they are typically fine-tuned from a few base models (e.g., Llama3, DINO), forming a Model Tree structure.
4. **Goal** How to leverage the Model Tree structure to simplify weight space learning and scale it to large models?
5. **Key Insight**: Models within the same Model Tree share initialization weights, which significantly reduces nuisance variance. Therefore, instead of a universal metanetwork, independent lightweight experts (MoE architecture) can be trained for each Tree.
6. **Core Idea**: Grouping models using Model Trees to reduce nuisances + designing a lightweight ProbeX architecture via Tucker decomposition = aligning large-scale model weights to language for the first time.

## Method

### Overall Architecture
Input is a single-layer weight matrix $X \in \mathbb{R}^{d_W \times d_H}$ of a neural network, and output is the task prediction (training class prediction or embedding aligned with text representation). The system contains two stages: (1) routing the model to its corresponding Model Tree via hierarchical clustering; (2) using an independent ProbeX expert within each Tree for prediction. Multiple Trees are combined via MoE (Mixture of Experts).

### Key Designs

1. **Model Tree-aware MoE Routing**:

    - **Function**: Automatically classifies the input model into the correct Model Tree and activates the corresponding expert.
    - **Mechanism**: Hierarchical clustering is applied to model weights in the training set to compute each cluster center $\hat{X}_k$. During inference, assignment is done via nearest neighbor: $R(X) = \arg\min_k \|X - \hat{X}_k\|_2$. The routing accuracy in the experiments is 100%.
    - **Design Motivation**: Motivational experiments demonstrate negative transfer in cross-Tree learning—adding data from other Trees actually degrades performance on a single Tree. MoE decouples the Trees to avoid interference.

2. **ProbeX Single-Layer Probing Architecture**:

    - **Function**: Extracts meaningful representations from a single hidden layer weight matrix using minimal parameters.
    - **Mechanism**: Learnable probe vectors $u_1, ..., u_{r_V}$ are designed and passed through the weight matrix $X$ to obtain responses $z_l = X^T u_l$. They are then encoded using a shared dimensionality reduction matrix $V$ and an encoding matrix $M_l$ unique to each probe: $e_l = M_l \sigma(V^T z_l)$. All probe encodings are aggregated as $e = \sum_l e_l$, and finally mapped to the output $y = Te$ via a prediction head $T$. It is theoretically proven that linear ProbeX is equivalent in expressivity to a dense expert under the Tucker decomposition assumption (Proposition 2), but with a parameter reduction of about 30x.
    - **Design Motivation**: A dense expert requires $d_H \times d_W \times d_Y$ parameters (up to billions), which is infeasible. ProbeX reduces the parameter size from hundreds of millions to 2.3 million through probe-probing + matrix decomposition, cutting training time from hours to 10 minutes.

3. **Weight-to-Language Representation Alignment**:

    - **Function**: Maps model weights to a space shared with CLIP text embeddings, achieving zero-shot model classification.
    - **Mechanism**: Taking fine-tuned Stable Diffusion models as an example, ProbeX is trained to align model encodings $e$ with the CLIP text embeddings of their training data categories, using a CLIP-like contrastive loss—maximizing the cosine similarity of correct pairs. During inference, the ProbeX encoding of the model is calculated, and its cosine distance is computed against the text embeddings of all candidate categories to select the closest one.
    - **Design Motivation**: This is the first time zero-shot capability has been demonstrated in weight space learning. The cross-attention layers of SD models naturally contain text-related information, which facilitates alignment.

### Loss & Training
- Classification task: Cross-entropy loss for 100 binary classification heads (predicting which 50 classes out of CIFAR100 were used for training).
- Alignment task: CLIP-like contrastive loss, maximizing the cosine similarity of correct pairs.
- All parameters ($V, u_l, M_l, T$) are trained end-to-end. A single-layer ProbeX trains in less than 10 minutes on a single GPU (10GB VRAM).

## Key Experimental Results

### Main Results

**Predicting Training Dataset Categories (Discriminative Models, CIFAR100 50/100 classes):**

| Model Tree | Dense Expert Acc | Dense #Params | ProbeX Acc | ProbeX #Params |
|------------|-----------------|---------------|------------|----------------|
| ResNet | 0.713 | 105M (×45) | **0.842** | 2.3M |
| DINO | 0.614 | 59M (×25) | **0.705** | 2.3M |
| MAE | 0.666 | 59M (×25) | **0.765** | 2.3M |
| Sup. ViT | 0.663 | 59M (×25) | **0.885** | 2.3M |

**Zero-Shot Model Classification (SD Generative Models):**

| Dataset | Dense In-dist | Dense Zero-shot | ProbeX In-dist | ProbeX Zero-shot |
|--------|--------------|-----------------|----------------|------------------|
| SD_200 | 0.801 | 0.706 | **0.973** | **0.898** |
| SD_1k | 0.382 | 0.343 | **0.296** | **0.505** |

### Ablation Study

| Configuration | In-dist Acc | Zero-shot Acc | Description |
|------|------------|---------------|------|
| ProbeX (w/o ReLU) | 0.953 | 0.564 | Linear version performs okay in classification but generalizes poorly |
| ProbeX (w/ ReLU) | **0.973** | **0.898** | Nonlinearity significantly boosts zero-shot capability |
| CLIP text encoder | **0.898** | - | Optimal text encoder |
| OpenCLIP | 0.860 | - | Slightly lower |
| BLIP2 | 0.564 | - | Significant gap |

### Key Findings
- **Intra-Tree learning vs. Cross-Tree**: A linear classifier achieves 0.844 accuracy within the same Tree, but only 0.502 (close to random) across Trees, showing a massive gap.
- **Negative transfer phenomenon**: Adding other Trees' data degrades performance on the current Tree, proving the necessity of the MoE design.
- **Nonlinearity is crucial for zero-shot**: ReLU drives zero-shot accuracy up from 56.4% to 89.8%, while only contributing a 2% improvement to in-distribution performance.
- **20 Model Trees cover 50% of models on Hugging Face**, showing that the Tree-aware learning scheme is practically feasible.

## Highlights & Insights
- **Insight from the Model Tree perspective is highly profound**: Discovering the fact that "most public models belong to a few Model Trees" transforms the seemingly impossible cross-architecture weight learning problem into a simple linear problem within Trees. This observation is inherently valuable.
- **Tucker decomposition design of ProbeX**: Proves the mathematical equivalence of probing and dense experts (Propositions 1-2), and compresses parameters via matrix decomposition, offering both theoretical backing and practicality.
- **Zero-shot model classification**: Embeds model weights into the semantic space of CLIP for the first time, initiating a new paradigm of "searching models using text," which has direct application value for searching model repositories like Hugging Face.

## Limitations & Future Work
- **Inability to generalize to unseen Trees**: Newly emerged Model Trees require retraining experts, which, although only taking 10 minutes, is still not zero-cost.
- **Poor zero-shot alignment for discriminative models**: Preliminary experiments show that discriminative models (ViT classifiers) do not align well with text. The authors hypothesize that the cross-attention layers of SD are key to successful alignment, limiting the generalizability of the method.
- **Dependence on known or clusterable Trees**: If a model comes from an unknown Tree not present in the training set, routing might fail.
- **Deeper ProbeX encoders lead to overfitting**: Performance degrades with multi-layer encoders, implying a need for better regularization strategies.

## Related Work & Insights
- **vs Neural Graphs [Kofinas et al.]**: Neural Graphs attempt to handle permutation invariance of model weights using graph neural networks but cannot scale to large models like ViTs. ProbeX bypasses permutation issues via probing and requires extremely low computation.
- **vs StatNN [Unterthiner et al.]**: StatNN extracts simple statistics of weights (mean, variance, quantiles), losing a large amount of structural information. ProbeX actively explores the structure of the weight space using learnable probe vectors.
- **Inspirations for model repository management**: ProbeX can be directly applied to platforms like Hugging Face to automatically generate "training content labels" for undocumented models, and even realize a CLIP-like "text-based model search" function.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Model Tree insight + first single-layer probing + first zero-shot model classification, multiple "firsts"
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale experiments with 14,000 models, covering both discriminative and generative models, but lacks validation on more real-world model registries
- **Writing Quality**: ⭐⭐⭐⭐ Motivational experiments progress systematically, tightly coupling theory and experiments
- **Value**: ⭐⭐⭐⭐⭐ Opens a new paradigm for weight space learning, offering direct utility for model management and search

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Draft and Refine with Visual Experts](../../CVPR2026/interpretability/draft_and_refine_with_visual_experts.md)
- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](../../ACL2026/interpretability/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[ICLR 2026\] Watch the Weights: Unsupervised Monitoring and Control of Fine-tuned LLMs](../../ICLR2026/interpretability/watch_the_weights_unsupervised_monitoring_and_control_of_fine-tuned_llms.md)
- [\[ICML 2026\] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](../../ICML2026/interpretability/from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)

</div>

<!-- RELATED:END -->
