---
title: >-
  [Paper Note] Biologically Plausible Learning via Bidirectional Spike-Based Distillation
description: >-
  [ICLR 2026][Model Compression][Spiking Neural Networks] This paper proposes BSD (Bidirectional Spike-based Distillation), which trains a feedforward spiking network (stimulus → concept for perception) and a feedback spiking network (concept → stimulus for memory recall) by distilling spike features between them. The entire process uses only discrete binary spikes and unsigned error signals, achieving accuracy comparable to backpropagation across image classification/generatio…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Spiking Neural Networks"
  - "Biologically Plausible Learning"
  - "Bidirectional Distillation"
  - "Local Learning"
  - "Contrastive Loss"
date: 2026-05-08
content_hash: 0aaf535319714e26
---

# Biologically Plausible Learning via Bidirectional Spike-Based Distillation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=MmWZ2xVJ7z](https://openreview.net/forum?id=MmWZ2xVJ7z)  
**Code**: https://github.com/alden199/Bidirectional-Spike-Based-Distillation  
**Area**: Model Compression / Spiking Neural Networks / Brain-inspired Learning  
**Keywords**: Spiking Neural Networks, Biologically Plausible Learning, Bidirectional Distillation, Local Learning, Contrastive Loss

## TL;DR
This paper proposes BSD (Bidirectional Spike-based Distillation), which trains a feedforward spiking network (stimulus → concept for perception) and a feedback spiking network (concept → stimulus for memory recall) by distilling spike features between them. The entire process uses only discrete binary spikes and unsigned error signals, achieving accuracy comparable to backpropagation across image classification/generation, text prediction, and temporal regression while satisfying five biological plasticity criteria.

## Background & Motivation

**Background**: Backpropagation (BP) is the foundation of deep learning training, yet it conflicts with neurobiological principles: it requires strict weight symmetry between feedforward and feedback paths, relies on global error signals propagated across the network, enforces distinct forward and backward computational phases, and uses continuous activation values instead of discrete spike communication. To find "biologically plausible" alternatives, various algorithms have been proposed, including Feedback Alignment, Target Propagation (TP/DTP), Predictive Coding, local losses, STDP, and energy-based models.

**Limitations of Prior Work**: Most existing methods compromise on at least one biological criterion. Lv et al. (2025) summarized three core criteria: C1 asymmetric feedforward/feedback weights, C2 synaptic plasticity using only local information, and C3 non-two-phase training. However, models like CCL and DLL, which satisfy these three, still transmit **signed floating-point numbers** between layers, which are neither spikes nor consistent with the fact that neurons fire unsigned spikes. Furthermore, their accuracy significantly lags behind BP in tasks like CIFAR-100 or temporal regression. Conversely, R-STDP, which satisfies all criteria, fails to learn complex tasks (achieving only ~1% on CIFAR-100).

**Key Challenge**: A long-standing trade-off exists between biological fidelity (using discrete spikes, unsigned errors, and local updates) and computational effectiveness. A critical unresolved issue is: **since spikes are binary (0/1), how can they represent the negative values essential for error signals (indicating the direction of weight updates)?**

**Goal**: To design a learning algorithm that satisfies five criteria (C1–C3 plus C4 discrete spikes and C5 unsigned error signals) while matching BP performance across various tasks and architectures.

**Key Insight**: Drawing from cognitive neuroscience, human learning is not a unidirectional "perception → decision" process but an interleaving of bottom-up perception and top-down memory recall. PET experiments show that the early visual cortex (V1/V2) activated during mental imagery overlaps heavily with actual perception. Once high-level concepts (e.g., of birds) are learned, they provide feedback to lower visual areas to sharpen sensitivity to details like beak shapes or wing patterns.

**Core Idea**: Learning is reformulated as a **bidirectional transformation** between two spike representations (stimulus encoding ↔ concept encoding). A feedforward network maps stimuli to concepts, while a feedback network reconstructs stimuli from concepts. They are trained jointly by **mutual distillation of spike features**, using the spike trains themselves as proxies for signed errors, thereby bypassing the limitation that spikes cannot represent negative values.

## Method

### Overall Architecture

BSD trains **a pair of spiking networks with shared layer structures but independent weights**. Each layer contains two types of pyramidal neurons: Type 1 receives lower-level input for the feedforward path (stimulus → concept, handling perception and decision); Type 2 receives higher-level input for the feedback path (concept → stimulus, handling memory reconstruction to assist feedforward learning). During training, the raw input $x$ is fed to the bottom-most Type 1 neurons, while the target is encoded into a spike train $\hat{s}$ and fed to the top-most Type 2 neurons. The two pathways "meet" at every layer and align their features. The process uses only discrete spikes, computes errors locally within single neurons, and executes forward/backward paths simultaneously—satisfying all five biological criteria.

The biological implementation utilizes **three-compartment pyramidal neurons**: the soma, the basal dendrites (receiving feedforward input to generate voltage $v$), and the apical dendrites (receiving feedback signals to generate voltage $\hat{v}$). The basal voltage $v$ drives spike firing $s=\mathrm{SN}(v)$, while the apical voltage $\hat{v}$ acts as a supervisory signal guiding synaptic plasticity at the basal dendrites. Learning consists of **aligning the basal voltage $v$ and apical voltage $\hat{v}$ of the same neuron**—ensuring consistency between "what is seen bottom-up" and "what is imagined top-down."

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Stimulus x"] --> B["Feedforward Pathway Type1<br/>Stimulus → Concept<br/>Basal Voltage v"]
    T["Target-encoded Spikes ŝ"] --> C["Feedback Pathway Type2<br/>Concept → Stimulus<br/>Apical Voltage v̂"]
    B --> D["Layer-wise Local Alignment<br/>ReCo Unsigned Contrastive Loss<br/>v ↔ v̂"]
    C --> D
    D -->|detach (truncate cross-layer gradients)| E["Update local synapses W / Θ only"]
    B -->|Inference: Cosine similarity (Output vs Label spikes)| F["Concept / Decision"]
    C --> G["Reconstruct Stimulus (Generation Task)"]
```

### Key Designs

**1. Bidirectional Spike-Based Distillation: Replacing "Error" with "Spike Features of the Other Path"**

This addresses the core difficulty of spikes being unable to represent signed errors. Instead of propagating abstract errors, BSD makes the feedforward (Type 1) and feedback (Type 2) paths "teachers" for each other. Feedforward: $v_i=\hat{v}'_i=W_{i-1}s_{i-1}, s_i=\mathrm{SN}(v_i)$. Feedback: $v'_i=\hat{v}_i=\Theta_i s'_{i+1}, s'_i=\mathrm{SN}(v'_i)$. Using **independent** weight matrices $W$ and $\Theta$ satisfies C1 (weight asymmetry). At each layer, $v_i$ aligns with the feedback signal $\hat{v}_i$, and $v'_i$ aligns with the feedforward signal $\hat{v}'_i$. Since the supervisory signal is always the voltage of currently firing spikes, it remains a purely positive spike representation, eliminating signed errors (C5).

**2. Three-Compartment Pyramidal Neurons and Local Voltage Alignment: Localizing the Error**

To satisfy C2 (local information only), BSD adopts the three-compartment model: basal dendrites for feedforward, apical dendrites for feedback, and the soma for integration. Learning becomes a **purely local** objective: minimizing the distance between the basal voltage $v$ and apical voltage $\hat{v}$ within a single neuron. Since information processing occurs within individual compartments without cross-layer dependencies, feedforward and feedback distillation can occur **simultaneously**, satisfying C3 (non-two-phase). The neurons use the LIF model to fire discrete 0/1 spikes (C4).

**3. ReCo Unsigned Contrastive Loss + detach: Unsigned Local Objective**

To provide a stable, unsigned local target without global gradients, the authors stack basal/apical voltages into matrices $V_i, \hat{V}_i \in \mathbb{R}^{B \times D_i}$ and define an affinity matrix $[C_i]_{kj} = \frac{v_{i,k} \cdot \hat{v}_{i,j}}{\|v_{i,k}\| \|\hat{v}_{i,j}\|}$. They use a Relaxed Contrastive (ReCo) loss:

$$\mathcal{L}_i = \sum_{k=1}^{B}(1 - [C_i]_{kk})^2 + \lambda \sum_{k=1}^{B} \sum_{j \neq k} \big(\max(0, [C_i]_{kj})\big)^2$$

The first term increases similarity for paired samples (diagonal), while the second term penalizes **positively correlated** non-paired samples. The loss is squared and non-negative (C5). Unlike InfoNCE, ReCo's $\max(0, \cdot)$ **does not penalize non-paired samples that are already orthogonal or negatively correlated**, allowing more flexible embedding spaces. Gradients are localized using `detach()`, ensuring $\mathcal{L}_i$ only updates synapses $W_{i-1}$ connected to the local dendrites.

**4. Frequency-Adaptive Regularization (FFT) for Generation: Balancing Detail and Noise**

For image generation, autoencoders must preserve edges while suppressing noise. BSD applies a Fast Fourier Transform (FFT) to inputs and voltages, splitting signals into low- and high-frequency components. The ReCo regularization coefficient $\lambda$ becomes **frequency-adaptive**: higher frequencies receive a larger $\lambda$ to suppress spurious correlations and preserve edge fidelity, while lower frequencies receive a smaller $\lambda$ to maintain structural coherence without amplifying noise.

### Loss & Training
Layers are aligned using the local ReCo loss between Type 1 and Type 2 basal/apical voltages. The top layer uses Cross-Entropy (classification) or MSE (generation). Weights $W$ and $\Theta$ are updated using their respective local gradients. Calculation graphs are truncated with `detach()` to prevent cross-layer propagation. Feedforward and feedback distillation proceed in parallel.

## Key Experimental Results

### Main Results

Image Classification (Mean of 5 datasets, CNN configuration): BSD satisfies all criteria C1–C5 and achieves accuracy close to BP, significantly outperforming DLL/CCL and R-STDP.

| Method | Criteria satisfied | CIFAR-10 (CNN) | CIFAR-100 (CNN) | Avg (CNN) |
| :--- | :--- | :--- | :--- | :--- |
| BP on ANN | 0/5 | 87.12% | 57.75% | 86.29% |
| BP on SNN | C4 | 87.02% | 57.21% | 86.02% |
| Predictive Coding | C2 | 72.94% | 53.08% | 82.40% |
| DLL | C1–C3 | 70.89% | 38.60% | 77.01% |
| R-STDP | C1–C5 | 33.19% | 1.49% | 50.10% |
| **BSD (Ours)** | **C1–C5** | **84.13%** | **53.48%** | **83.78%** |

Sequence Regression / Text Prediction (RNN): BSD achieves 41.8% on Harry Potter character prediction, outperforming DLL (33.7%) and Predictive Coding (38.8%). For Metr-la, its MSE (0.125) was lower than BP-ANN (0.131).

Image Generation (FID↓):

| Dataset | ANN-BP | FSVAE | BSD (Ours) |
| :--- | :--- | :--- | :--- |
| CIFAR-10 | 127.34 | 175.5 | 168.12 |
| MNIST | 49.56 | 97.06 | 72.39 |

### Ablation Study

Impact of Layer-wise Loss Function (CNN Accuracy):

| Configuration | MNIST | SVHN | CIFAR-10 | CIFAR-100 | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BSD-MSE | 21.10% | 19.46% | 16.93% | 1.58% | Using MSE leads to near non-convergence |
| BSD-InfoNCE | 98.77% | 83.27% | 72.38% | 38.06% | InfoNCE converges but performs worse |
| **BSD (ReCo)** | **99.44%** | **90.81%** | **84.13%** | **53.48%** | Full model |

### Key Findings
- **ReCo is essential for convergence**: Switching to MSE results in failure (1.58% on CIFAR-100), proving that per-neuron alignment requires contrastive unsigned loss.
- **Biological fidelity does not necessitate performance loss**: BSD is the only method satisfying all five criteria while maintaining high performance (~84% average); in contrast, R-STDP achieves only 50.10%.
- **Hamming similarity**: The similarity between Type 1 and Type 2 spikes at the same layer increases to 0.9+ during training, confirming that bidirectional alignment is occurring.

## Highlights & Insights
- **Reframing "Error Propagation" as "Mutual Distillation"**: By using positive spike features from the parallel path as supervision, BSD bypasses the need for signed errors, converting a representation problem into an architectural solution.
- **Three-Compartment Neurons as Local Containers**: This structure naturally enforces C1, C2, and C3 as architectural properties rather than external constraints.
- **Cross-Modal Alignment Perspective**: Aligning basal and apical voltages is treated as cross-modal embedding alignment, successfully integrating CLIP-style contrastive learning into a biologically plausible framework.

## Limitations & Future Work
- **Batch Size Dependency**: As a contrastive method, BSD is sensitive to batch size.
- **Accuracy Gap**: Performance on complex tasks like CIFAR-100 still lags slightly behind BP-ANN (53.48% vs 57.75%).
- **Overhead**: Maintaining two sets of networks and weights doubles memory and computational costs during training.
- **Future Directions**: Scaling to larger architectures, reducing the feedback path's memory footprint, and exploring local contrastive objectives that are less dependent on large batches.

## Related Work & Insights
- **Comparison with DLL/CCL**: Unlike these methods, BSD satisfies C4/C5 by using spikes and unsigned losses, leading to 15%+ higher accuracy on CIFAR-100.
- **Comparison with Predictive Coding**: While PC minimizes prediction error, it typically only satisfies C2; BSD satisfies all five criteria by materializing the prediction as a feedback spiking network.
- **Comparison with R-STDP**: While both satisfy all criteria, STDP struggles with complex supervised tasks where BSD excels due to its contrastive distillation mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bypasses the "signed error" spike representation problem through bidirectional distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across classification, generation, and sequences, though an absolute gap with BP-ANN remains.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical mapping from neurobiological motivation to mathematical derivation.
- Value: ⭐⭐⭐⭐ Provides a scalable paradigm for local learning on neuromorphic hardware.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Otters: An Energy-Efficient Spiking Transformer via Optical Time-to-First-Spike Encoding](otters_an_energy-efficient_spiking_transformer_via_optical_time-to-first-spike_e.md)
- [\[ICLR 2026\] Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression](towards_lossless_memory-efficient_training_of_spiking_neural_networks_via_gradie.md)
- [\[ICLR 2026\] Knowledge Distillation for Large Language Models through Residual Learning](knowledge_distillation_for_large_language_models_through_residual_learning.md)
- [\[CVPR 2026\] A Unified Framework for Knowledge Transfer in Bidirectional Model Scaling](../../CVPR2026/model_compression/a_unified_framework_for_knowledge_transfer_in_bidirectional_model_scaling.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)

</div>

<!-- RELATED:END -->
