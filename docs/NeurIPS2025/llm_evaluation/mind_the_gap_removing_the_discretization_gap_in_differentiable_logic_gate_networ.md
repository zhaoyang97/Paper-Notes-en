---
title: >-
  [Paper Note] Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks
description: >-
  [NeurIPS 2025][LLM Evaluation][Logic Gate Networks] This paper proposes Gumbel Logic Gate Networks (Gumbel LGNs), which inject Gumbel noise into logic gate selection and employ a straight-through (ST) estimator to reduce the discretization gap of differentiable logic gate networks by 98%, achieve a 4.5× speedup in training, and reduce the proportion of unused neurons to 0%.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Logic Gate Networks
  - Gumbel-Softmax
  - Discretization Gap
  - Efficient Inference
  - Loss Landscape Smoothing
date: 2026-05-08
content_hash: a66775d1fa45937a
---

# Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks

**Conference**: NeurIPS 2025
**arXiv**: [2506.07500](https://arxiv.org/abs/2506.07500)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: Logic Gate Networks, Gumbel-Softmax, Discretization Gap, Efficient Inference, Loss Landscape Smoothing

## TL;DR

This paper proposes Gumbel Logic Gate Networks (Gumbel LGNs), which inject Gumbel noise into logic gate selection and employ a straight-through (ST) estimator to reduce the discretization gap of differentiable logic gate networks by 98%, achieve a 4.5× speedup in training, and reduce the proportion of unused neurons to 0%.

## Background & Motivation

Modern deep neural networks, despite their strong performance, impose substantial computational and energy demands that hinder practical deployment. Logic Gate Networks (LGNs) replace arithmetic operations with Boolean logic operations (AND, OR, XOR, etc.), offering an extremely efficient inference paradigm. To enable gradient-based training of LGNs, prior work introduced Differentiable Logic Gate Networks (DLGNs), which apply a continuous relaxation as a weighted combination over all 16 possible binary logic gates.

DLGNs suffer from two core problems:

**Discretization Gap**: Training uses a continuous relaxation, whereas inference requires discretizing to a single logic gate. This results in inference accuracy approximately 3% lower than training accuracy, significantly hindering deployment.

**Slow Convergence**: DLGNs require days or even weeks to converge on CIFAR-10, and nearly half of the neurons fail to collapse to a single logic gate after training (i.e., remain "unused").

The authors' central hypothesis is that a **smoother loss landscape** can simultaneously address both issues — making parameters more robust to discretization while providing better gradient signals to accelerate convergence.

## Method

### Overall Architecture

The core idea of Gumbel LGN is to replace the softmax gate-selection mechanism in DLGNs with **Gumbel-Softmax combined with a straight-through (ST) estimator**.

Each neuron maintains a logit vector $\mathbf{z} \in \mathbb{R}^{16}$ over 16 logic gates. The standard DLGN output is:

$$f_{\mathbf{z}}^{\text{soft}}(a,b) = \sum_{i=1}^{16} \frac{\exp z_i}{\sum_j \exp z_j} \cdot h_i(a,b)$$

Gumbel LGN injects Gumbel noise and performs a hard selection during the forward pass:

$$f_{\mathbf{z}}^{\text{discrete}}(a,b) = h_k(a,b), \quad k = \arg\max_j (z_j + g_j), \quad g_j \sim \text{Gumbel}(0,1)$$

During backpropagation, a soft Gumbel-Softmax approximation is used to compute gradients:

$$\pi_i^{\text{Gumbel}} = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}$$

### Key Designs

**1. Implicit Hessian Regularization**

The authors theoretically show that injecting Gumbel noise is equivalent to smoothing the loss function, expressed as:

$$J(\mathbf{z}) = \mathcal{L}(\text{softmax}(\mathbf{z}/\tau)) + \frac{\pi^2}{12\tau^2} \text{tr}(H_f(\mathbf{z}/\tau)) + O(\tau^{-3})$$

where $\text{tr}(H_f)$ is the trace of the Hessian matrix. This implies that Gumbel noise implicitly penalizes the curvature of the loss landscape, encouraging the optimizer to find flat minima.

**2. Role of the Temperature Parameter $\tau$**

- Small $\tau$ (e.g., 0.1): large $1/\tau^2$ → stronger smoothing, flatter minima
- Large $\tau$ (e.g., 2.0): small $1/\tau^2$ → negligible smoothing effect

A "Goldilocks zone" exists ($\tau \approx 0.25$) that achieves the best balance between convergence speed and final accuracy.

**3. Contribution of the ST Estimator**

The ST estimator aligns the forward-pass behavior during training with that at inference (both performing hard selection), further closing the discretization gap. Ablation studies show that Gumbel noise alone improves convergence and reduces the gap, but incorporating the ST estimator substantially reduces the gap further.

### Loss & Training

Training adopts the same GroupSum operation as DLGNs to obtain class scores. The loss function naturally incorporates the implicit regularization term induced by Gumbel smoothing, with temperature $\tau$ controlling regularization strength. The per-iteration runtime overhead is only approximately 5%.

## Key Experimental Results

### Main Results

Experiments are conducted on CIFAR-10 and CIFAR-100 with a default configuration of depth 12 and width 256K.

| Method | CIFAR-10 Discrete Accuracy | Discretization Gap | Convergence Speed | Unused Gate Ratio |
|--------|---------------------------|-------------------|------------------|------------------|
| DLGN | ~56% | ~3% | Baseline | 49.81% |
| Gumbel LGN | ~57.5% | ~0.06% (↓98%) | 4.5× faster | 0.00% (↓100%) |

**Temperature Ablation** (CIFAR-10, depth 12, width 256k):

| $\tau$ | Max Accuracy | Final Accuracy | Convergence Iterations (×10³) |
|--------|-------------|---------------|-------------------------------|
| 0.01 | 0.547 | 0.546 | 972 |
| 0.10 | 0.574 | 0.570 | 632 |
| 0.25 | 0.573 | 0.568 | 440 |
| 0.50 | 0.573 | 0.572 | 530 |
| 1.00 | 0.578 | 0.575 | 918 |
| 2.00 | 0.490 | 0.480 | 1342 |

### Ablation Study

**ST Estimator Ablation**:
- Soft Gumbel (without ST): converges faster than DLGNs and reduces the discretization gap
- Gumbel LGN (with ST): converges slightly slower than Soft Gumbel, but achieves a substantially larger further reduction in the discretization gap

**Effect of Depth**:
- DLGNs: discretization gap increases with depth
- Gumbel LGNs: maintain consistently low gap across all depths, demonstrating favorable depth scalability

**Shallow-Wide Networks** (depth 6, width 2048K):
- DLGNs exhibit rapid convergence in soft accuracy, but discrete accuracy no longer improves, indicating that width also exacerbates the discretization gap
- Gumbel LGNs achieve higher final accuracy

### Key Findings

1. Gumbel noise makes the Hessian trace more negative (flatter landscape), consistent with theoretical predictions.
2. Loss landscape visualizations confirm that Gumbel LGN is markedly smoother than DLGN.
3. Neuron entropy analysis confirms that nearly all neurons in Gumbel LGNs collapse to a single logic gate (entropy near zero), whereas many neurons in early layers of DLGNs remain uncollapsed.
4. On CIFAR-100, Gumbel LGNs converge in approximately 400K iterations, while DLGNs fail to converge within 48 GPU hours.

## Highlights & Insights

1. **Tight integration of theory and practice**: Lemma 1 formally establishes that Gumbel noise is equivalent to Hessian trace regularization, and this is empirically validated using the Hutchinson estimator.
2. **Scaling NAS to unprecedented search spaces**: The search space of DLGNs reaches $10^{3,699,056}$ (CIFAR-10), far exceeding the $10^{18}$ scale of conventional NAS, demonstrating the viability of DARTS-style methods at extreme scales.
3. **Train-inference alignment**: The ST estimator aligns training behavior with inference, a principle with broad implications for quantization-aware training, NAS, and related fields.
4. **100% neuron utilization** is a compelling result, implying that every parameter in the network is effectively utilized.

## Limitations & Future Work

1. **Limited datasets**: Validation is restricted to CIFAR-10 and CIFAR-100; experiments on more challenging datasets such as ImageNet are absent.
2. **Temperature tuning required**: $\tau$ must be selected manually; adaptive temperature scheduling is a promising future direction.
3. **Theoretical simplification**: The Hessian trace analysis relies on a second-order Taylor approximation; a complete theoretical characterization of the discretization gap remains an open problem.
4. The **interaction between width and depth** is not thoroughly analyzed.
5. Direct comparisons with other efficient neural network approaches (e.g., binary networks, sparse networks) are missing.

## Related Work & Insights

- Core connection to **DARTS** (Liu et al., 2019): DLGNs are essentially a form of differentiable architecture search operating at massive scale.
- **Smooth DARTS** (Chen & Hsieh, 2021) reduces the discretization gap in DARTS by adding noise; Gumbel LGN adopts a similar strategy in the LGN setting with more pronounced effect.
- **SAM** (Foret et al., 2021) seeks flat minima to improve generalization; Gumbel noise achieves a similar objective in this work.
- Kim (2024) also applied Gumbel noise and the ST estimator to DLGNs but did not analyze the discretization gap or convergence behavior.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces Gumbel-Softmax + ST estimator into logic gate networks and establishes a theoretical connection, though the individual technical components have prior precedent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are comprehensive, covering temperature, ST estimator, depth, and width, though the dataset scope is narrow.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with effective integration of theory and experiments; figures and tables are highly informative.
- Value: ⭐⭐⭐⭐ Significant practical implications for deploying logic gate networks; theoretical insights are also valuable to the NAS community.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Idiom Understanding as a Tool to Measure the Dialect Gap](../../ACL2026/llm_evaluation/idiom_understanding_as_a_tool_to_measure_the_dialect_gap.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](../../ACL2026/llm_evaluation/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ICLR 2026\] SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs](../../ICLR2026/llm_evaluation/simpletom_exposing_the_gap_between_explicit_tom_inference_and_implicit_tom_appli.md)
- [\[ICCV 2025\] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](../../ICCV2025/llm_evaluation/sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)
- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](../../ICLR2026/llm_evaluation/improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)

<!-- RELATED:END -->
