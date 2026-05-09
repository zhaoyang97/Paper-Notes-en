---
title: >-
  [Paper Note] Causal Interpretation of Neural Network Computations with Contribution Decomposition
description: >-
  [ICLR2026][Medical Imaging][neural network interpretability] This paper proposes CODEC (Contribution Decomposition), which applies Integrated Gradients to compute the contribution of hidden-layer neurons to the output (rather than analyzing activations alone), and then decomposes these contributions into sparse modes via a Sparse Autoencoder. This approach achieves stronger causal interpretability and network control than activation-based analysis, and is successfully applied to ResNet-50 and a retinal biological neural network model.
tags:
  - ICLR2026
  - Medical Imaging
  - neural network interpretability
  - contribution decomposition
  - sparse autoencoder
  - causal analysis
  - retinal modeling
date: 2026-05-08
content_hash: dad8018abf49e58f
---

# Causal Interpretation of Neural Network Computations with Contribution Decomposition

**Conference**: ICLR2026
**arXiv**: [2603.06557](https://arxiv.org/abs/2603.06557)
**Code**: [https://github.com/baccuslab/CODEC_ICLR_2026](https://github.com/baccuslab/CODEC_ICLR_2026)
**Area**: Medical Imaging
**Keywords**: neural network interpretability, contribution decomposition, sparse autoencoder, causal analysis, retinal modeling

## TL;DR
This paper proposes CODEC (Contribution Decomposition), which applies Integrated Gradients to compute the contribution of hidden-layer neurons to the output (rather than analyzing activations alone), and then decomposes these contributions into sparse modes via a Sparse Autoencoder. This approach achieves stronger causal interpretability and network control than activation-based analysis, and is successfully applied to ResNet-50 and a retinal biological neural network model.

## Background & Motivation

**State of the Field**: Understanding how neural networks transform inputs into outputs is a central problem in explainable AI. Existing methods primarily analyze activation patterns in hidden layers, seeking representations associated with human-interpretable concepts.

**Limitations of Prior Work**: Activation analysis only reflects a neuron's receptive field—i.e., what inputs it responds to—but does not address how that neuron influences the output. A highly activated neuron may exert a positive, negative, or null effect on the output. Existing saliency map methods (Grad-CAM, SmoothGrad) analyze only the input→output mapping and offer limited insight into the causal mechanisms of intermediate layers.

**Root Cause**: Activation $\neq$ contribution. Activation captures only half of the picture via the receptive field; understanding a neuron's causal role also requires its projective field—its influence on downstream computation. Yet existing tools rarely analyze the collaborative contributions of hidden-layer neuron populations.

**Paper Goals**: To establish a general framework for analyzing the collaborative contributions of hidden-layer neuron populations, revealing how they jointly construct network outputs.

**Starting Point**: Inspired by neuroscience—specifically, how distinct neuron types in the retina cooperate to produce functional outputs—the paper extends attribution methods (Integrated Gradients) from the input→output direction to the hidden layer→output direction, computing each hidden neuron's contribution and subsequently decomposing these contributions into cooperative modes via a Sparse Autoencoder.

**Core Idea**: Analyze the *contributions* rather than the *activations* of hidden-layer neurons, and decompose contributions into sparse cooperative modes using a Sparse Autoencoder, yielding causal insights inaccessible to activation-based analysis.

## Method

### Overall Architecture
CODEC consists of four stages: (1) defining the contribution target—selecting the scalar output to be explained (e.g., the top logit); (2) computing contributions—applying Integrated Gradients from the target output to the hidden layer to obtain each neuron's contribution value for each input; (3) decomposing contributions—using a Sparse Autoencoder to factorize the contribution matrix into sparse modes; and (4) visualization—mapping modes back to the input space to reveal which input features drive the output through which modes.

### Key Designs

1. **Hidden-Layer Contribution Computation**:

    - Function: Quantify each hidden-layer neuron's causal contribution to a scalar output target.
    - Mechanism: Integrated Gradients is extended from the "output→input" direction to the "output→hidden layer" direction. For each channel of a convolutional layer, contributions are spatially summed to yield a single scalar. Contributions can be positive or negative—positive values indicate promotion of the output, while negative values indicate suppression—fundamentally distinguishing them from activations (which are always non-negative after ReLU).
    - Design Motivation: Integrated Gradients satisfies the completeness axiom—the sum of all neuron contributions exactly equals the output value. This provides a rigorous mathematical foundation for contribution decomposition, unlike approximate methods such as Grad-CAM.

2. **Sparse Autoencoder Decomposition (Core of CODEC)**:

    - Function: Decompose the $d$-channel × $n$-image contribution matrix into $k$ sparse modes.
    - Mechanism: An encoder $f_{\text{enc}}: \mathbb{R}^d \to \mathbb{R}^k$ computes loadings, which are sparsified via a hard threshold $\tau$. A non-negative dictionary $\mathbf{D} \in \mathbb{R}^{d \times k}_+$ defines the modes. The reconstruction loss is $\mathcal{L} = \|\mathbf{c} - \mathbf{D}\mathbf{z}\|_2^2$ with L1 regularization. The default setting uses $k = 3d$ (overcomplete representation) with threshold 0.9. Training takes approximately 3–7 minutes per layer.
    - Design Motivation: Directly analyzing $d$-dimensional contribution vectors is intractable. The Sparse Autoencoder discovers cooperative patterns among neuron populations—identifying which channels consistently contribute positively or negatively together. Compared with applying a Sparse Autoencoder to activations, applying it to contributions yields modes more closely associated with output classes.

3. **Contribution Mapping Visualization**:

    - Function: Map modes back to the input pixel space, revealing which image regions drive the output through which mode.
    - Mechanism: For the key channels $c$ within mode $m$, the input sensitivity is computed as $A_i^{(c,p)} = J_{y,h_{c,p}} J_{h_{c,p},x_i}$ (Jacobian chain rule), and then element-wise multiplied with the input to obtain the contribution map $C_i^{(m)} = A_i^{(m)} \odot x_i$.
    - Design Motivation: Traditional saliency maps produce a single aggregate input importance map. Contribution mapping decomposes this by mode, revealing how distinct input features (e.g., wood grain, hands, strings) drive the same output through different computational pathways.

4. **Network Control Experiments**:

    - Function: Perform targeted ablation or preservation using channels identified by CODEC.
    - Mechanism: The mode most associated with the target class is identified, and its high-weight channels are extracted. Ablation: these channels are removed and the resulting drop in target-class accuracy is measured. Preservation: only these channels are retained to assess whether they alone suffice for classification. The effectiveness of contribution-based modes is compared against activation-based modes.
    - Design Motivation: If CODEC genuinely captures causal structure, interventions based on contribution modes should be more precise than those based on activation modes.

### Loss & Training
The Sparse Autoencoder is trained on 50,000 images from the ImageNet validation set with a learning rate of 5e-5, batch size of 128, 300 epochs, and an L1 regularization coefficient of 5e-5. ResNet-50 is analyzed layer by layer across different blocks. The average reconstruction $R^2 = 0.85$.

## Key Experimental Results

### Main Results (Contribution vs. Activation)

| Dimension | Contribution | Activation |
|-----------|-------------|------------|
| Sparsity across layers | Consistently higher, increases with depth | Increases but remains lower |
| Dimensions for 95% variance | Higher (~200 at layer 14) | Lower (~150) |
| Max mode–class correlation | **0.45+** (intermediate layers) | ~0.30 |
| Modes exceeding class correlation threshold (>0.2) | **80+** (layer 13) | ~40 |

### Ablation Study (Network Control — Contribution Mode vs. Activation Mode)

| Control Strategy | Contribution Mode | Activation Mode | Notes |
|-----------------|------------------|-----------------|-------|
| Ablation (remove 2% of channels) | Target-class accuracy → ~0% | Target-class → ~20% | Contribution-mode channels are more necessary |
| Preservation (retain only 2% of channels) | Target-class accuracy ~80% | Target-class ~50% | Contribution-mode channels are more sufficient |
| Cross-layer consistency | Significant improvement at Block 7+ | Improvement at Block 7+ but weaker | Semantic information undergoes a qualitative transition at blocks 6–7 |

### Key Findings
- **Progressive decoupling of positive and negative contributions across layers**: In early layers, positive and negative contributions from the same channel are highly correlated (analogous to center-surround receptive fields in the retina); in deeper layers, they progressively decouple as channels become more functionally specialized.
- **Contribution modes are more class-specific than activation modes**: The difference is largest in intermediate layers, suggesting that activations conflate features that are represented but causally irrelevant to the output, whereas contributions filter out this noise.
- **Dynamic receptive fields in the retinal model**: CODEC reveals how combinations of different modes in the retinal model give rise to dynamic instantaneous receptive fields (IRFs)—the same neuron exhibits distinct receptive field properties, ranging from center-surround to orientation-selective, depending on which modes drive it at different times.
- **Robustness of SAE hyperparameters**: Results are insensitive to the L1 regularization coefficient, random seeds, and the non-negativity constraint on the dictionary, degrading only when the threshold is excessively high or the dictionary is too small.

## Highlights & Insights
- **The core insight that activation ≠ contribution**: This is the paper's central message. A highly activated neuron may suppress the output (negative contribution); relying solely on activations completely mischaracterizes its role—a distinction of critical importance for mechanistic interpretability.
- **Cooperative modes over individual neurons**: The paper bypasses the question of which individual neurons matter and directly addresses which neuron combinations matter, consistent with population coding theory in biological neuroscience. A single mode can extract the same visual feature (e.g., "shiny wood") across different classes, making it a more natural unit of analysis than individual neurons.
- **A bidirectional bridge between AI and biology**: The application to a retinal CNN model is not an ancillary experiment but demonstrates the method's core value—CODEC generates experimentally testable hypotheses about which combinations of intermediate neurons drive which output patterns.
- **The block 6–7 inflection point**: Ablation efficacy rises sharply at blocks 6–7, suggesting a qualitative transition in semantic representations at this stage—a finding that can guide model pruning and the selection of feature extraction layers.

## Limitations & Future Work
- **Validation limited to ResNet-50 and ViT-B**: The method has not been systematically tested on larger models (e.g., ViT-L, LLMs). Although the authors mention a scaling analysis on LLMs, the details are insufficient.
- **Weaker performance on ViT**: Ablation effects are less pronounced for ViT than for CNNs; the authors attribute this to ViT's lack of explicit spatial equivariance inductive bias, and note that token spatial aggregation strategies require further investigation.
- **Only positive contributions are analyzed**: The current Sparse Autoencoder decomposition uses only positive contributions; negative contributions (suppressive effects) may encode equally important computational structure.
- **Computational overhead**: Integrated Gradients requires 10 integration steps, and while Sparse Autoencoder training is not prohibitively slow (3–7 minutes per layer), the full pipeline is substantially heavier than simple activation analysis.
- **Independence from training data**: CODEC analyzes at inference time without requiring training data. However, this also means it cannot distinguish contributions attributable to training data biases from those that are genuinely meaningful.

## Related Work & Insights
- **vs. Grad-CAM**: Grad-CAM attributes only the input→output mapping, producing a single saliency map. CODEC analyzes the hidden layer→output direction and decomposes attributions into multiple cooperative modes, conveying far richer information than a single saliency map.
- **vs. SAE on activations (Anthropic's monosemanticity work)**: Anthropic applies Sparse Autoencoders to LLM activations to discover interpretable features. The key distinction of CODEC is that it applies the Sparse Autoencoder to contributions rather than activations, ensuring that the discovered modes carry causal significance rather than mere correlation.
- **vs. Circuit-level mechanistic interpretability**: Circuit-level analysis traces information flow neuron by neuron and is suited to small models. CODEC bypasses this by directly targeting population-level analysis through mode decomposition, making it more applicable to large-scale networks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The distinction between contribution and activation is a profound insight; applying Sparse Autoencoder decomposition to contributions rather than activations is an entirely novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on CNN, ViT, and a biological model with well-designed control experiments, but lacks large-scale validation on LLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ The neuroscience-driven narrative is elegant, with seamless integration of methodology and biological application.
- Value: ⭐⭐⭐⭐⭐ Provides both a new analytical tool and a conceptual framework for mechanistic interpretability.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Intrinsic Lorentz Neural Network](intrinsic_lorentz_neural_network.md)
- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](../../AAAI2026/medical_imaging/unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[ICLR 2026\] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/medical_imaging/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](../../AAAI2026/medical_imaging/mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)

<!-- RELATED:END -->
