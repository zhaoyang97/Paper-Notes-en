---
title: >-
  [Paper Note] Continuous Thought Machines
description: >-
  [NeurIPS 2025][Neural dynamics] This paper proposes the Continuous Thought Machine (CTM), which generates neuron-level temporal dynamics via privately parameterized Neuron-Level Models (NLMs) and employs a neural synchro…
tags:
  - "NeurIPS 2025"
  - "Neural dynamics"
  - "neural synchrony"
  - "adaptive computation"
  - "recurrent architectures"
  - "biologically inspired"
date: 2026-05-08
content_hash: 963251a3e6bbdbca
---

# Continuous Thought Machines

**Conference**: NeurIPS 2025
**arXiv**: [2505.05522](https://arxiv.org/abs/2505.05522)
**Code**: [github.com/SakanaAI/ContinuousThoughtMachines](https://github.com/SakanaAI/ContinuousThoughtMachines)
**Area**: Other
**Keywords**: Neural dynamics, neural synchrony, adaptive computation, recurrent architectures, biologically inspired

## TL;DR
This paper proposes the Continuous Thought Machine (CTM), which generates neuron-level temporal dynamics via privately parameterized Neuron-Level Models (NLMs) and employs a neural synchrony matrix as the core latent representation. The model demonstrates complex reasoning, adaptive computation, and interpretable attention behavior on tasks including maze solving, ImageNet classification, and parity checking.

## Background & Motivation

**Background**: Modern neural networks deliberately abstract away the precise timing and interaction complexity of biological neurons in pursuit of large-scale training efficiency. In standard MLPs and Transformers, neurons execute simple activation functions and lack independent temporal processing capabilities.

**Limitations of Prior Work**: (a) Deep learning lacks the flexibility, efficiency, and common sense of human cognition—properties that may be related to temporal processing; (b) adaptive computation requires auxiliary halting modules (e.g., PonderNet/ACT); (c) recurrent networks possess sequential processing ability, but their state representations are static snapshots rather than dynamic temporal patterns.

**Key Challenge**: Neural timing is critical in biological brains (temporal coding, synchrony, oscillations) yet is entirely absent in AI. Reintroducing the temporal dimension may unlock missing cognitive capabilities.

**Key Insight**: The paper designs a novel architecture with neuron-level temporal processing and synchrony-based representations within a gradient-differentiable framework, striking a balance between biological plausibility and computational feasibility.

## Method

### Overall Architecture
The CTM operates over an **internal time dimension** $t \in \{1, \ldots, T\}$ that is decoupled from the data dimension. Each internal tick executes the following pipeline:

### 1. Synapse Model
A shared synaptic network $f_{\theta_{\rm syn}}$ (U-NET-style MLP) connects the $D$-dimensional latent space:

$$\mathbf{a}^t = f_{\theta_{\rm syn}}([\mathbf{z}^t; \mathbf{o}^t]) \in \mathbb{R}^D$$

The input is the concatenation of the post-activation $\mathbf{z}^t$ from the previous step and the attention output $\mathbf{o}^t$; the output is a pre-activation value. The pre-activation history over the most recent $M$ steps is retained as $\mathbf{A}^t \in \mathbb{R}^{D \times M}$.

### 2. Neuron-Level Models (Key Innovation 1)
Each neuron $d$ possesses a **privately parameterized** NLM $g_{\theta_d}$ (a single-layer MLP) that processes its own $M$-dimensional pre-activation history:

$$z_d^{t+1} = g_{\theta_d}(\mathbf{A}_d^t)$$

Key distinction: in standard networks all neurons share an activation function (ReLU/GELU), whereas in the CTM each neuron has independent weights for processing its own temporal history, enabling rich and diverse temporal dynamics.

### 3. Neural Synchrony as Representation (Key Innovation 2)
The post-activation history $\mathbf{Z}^t = [\mathbf{z}^1, \ldots, \mathbf{z}^t] \in \mathbb{R}^{D \times t}$ is collected, and the synchrony matrix is computed as:

$$\mathbf{S}^t = \mathbf{Z}^t \cdot (\mathbf{Z}^t)^\top \in \mathbb{R}^{D \times D}$$

$\mathbf{S}_{ij}^t$ measures the temporal correlation between the activity patterns of neurons $i$ and $j$. To manage the $O(D^2)$ scale, $D_{\rm out}$ and $D_{\rm action}$ neuron pairs are randomly sampled for output projection and attention queries, respectively:

$$\mathbf{y}^t = \mathbf{W}_{\rm out} \cdot \mathbf{S}_{\rm out}^t, \quad \mathbf{q}^t = \mathbf{W}_{\rm in} \cdot \mathbf{S}_{\rm action}^t$$

### 4. Learnable Temporal Decay
A learnable exponential decay rate $r_{ij} \geq 0$ is introduced for each neuron pair $(i,j)$:

$$\mathbf{S}_{ij}^t = \frac{(\mathbf{Z}_i^t)^\top \cdot \text{diag}(\mathbf{R}_{ij}^t) \cdot \mathbf{Z}_j^t}{\sqrt{\sum_\tau [\mathbf{R}_{ij}^t]_\tau}}$$

where $[\mathbf{R}_{ij}^t]_\tau = \exp(-r_{ij}(t - \tau))$. A high $r_{ij}$ emphasizes recent activity; $r_{ij}=0$ implies no decay.

### 5. Loss & Training
Each tick $t$ produces an output $\mathbf{y}^t$, from which a loss $\mathcal{L}^t$ and a certainty $\mathcal{C}^t = 1 - H(\mathbf{y}^t)/\log C$ are computed. Two ticks are selected for optimization:

$$L = \frac{\mathcal{L}^{t_1} + \mathcal{L}^{t_2}}{2}, \quad t_1 = \arg\min_t \mathcal{L}^t, \quad t_2 = \arg\max_t \mathcal{C}^t$$

Adaptive computation **emerges naturally**: simple samples reach high certainty after few ticks without requiring an explicit halting module.

## Key Experimental Results

### 2D Maze ($39 \times 39$, 100-step path prediction)

| Model | Path Accuracy↑ | Generalization to $99 \times 99$↑ |
|------|-----------|---------------------|
| Feed-Forward | Extremely low (near chance) | — |
| LSTM-1layer (75 ticks) | ~40–50% (short paths) | — |
| LSTM-3layer (50 ticks) | ~50–60% (short paths) | — |
| **CTM (75 ticks)** | **>90% (50+ steps)** | **Effective generalization via repeated application** |

### ImageNet-1K Classification (ResNet-152 backbone, 50 internal ticks)

| Metric | Value |
|------|------|
| Top-1 Accuracy | 72.47% |
| Top-5 Accuracy | 89.89% |
| Early stopping at 80% certainty | Most samples halt in <10 ticks |
| Model calibration | Excellent (averaged probabilities across ticks are well-calibrated) |

### Parity Checking (64-bit sequences, cumulative parity prediction)

| Model | Best Accuracy↑ |
|------|-----------|
| LSTM (parameter-matched) | Unstable, below 90% |
| CTM (50 ticks) | ~95% |
| CTM (75 ticks) | ~98–100% (perfect on some seeds) |
| CTM (100 ticks) | 100% (some seeds) |

### Ablation Study
- Removing NLMs (replacing with standard activation functions): substantial performance drop
- Removing synchrony representation (replacing with snapshot representation): substantial performance drop
- LSTM + synchrony: fails to reproduce CTM's advantage → NLMs and synchrony are both necessary

### Key Findings
- The CTM learns **spatial reasoning without positional encodings** in maze tasks—"imagining" the path ahead through attention (episodic future thinking)
- On ImageNet, a "scanning" behavior emerges: attention dynamically traverses the image across ticks, resembling human visual search
- Neural activity exhibits low-frequency traveling waves analogous to those observed in biological cortex
- More ticks = greater capability (parity checking), but also = greater compute

## Highlights & Insights
- **Synchrony as representation**, rather than as a post-hoc property, constitutes a core paradigm shift: unlike Reichert & Serre, who use synchrony for gating/segmentation, the CTM directly optimizes synchrony patterns during training to encode task-relevant information
- **Design philosophy of NLMs**: each neuron possesses an independent "character" (distinct weights for processing its history), offering an appropriate abstraction of biological neuronal complexity
- **Emergence of adaptive computation**: without PonderNet's explicit halting probabilities or ACT's cumulative thresholds, adaptive computation arises naturally from the min-loss + max-certainty selection scheme alone
- **Architectural generality**: the same CTM, without modification, is applied to maze, classification, and algorithmic tasks, demonstrating broad applicability

## Limitations & Future Work
- The internal sequence dimension extends training time linearly ($T$-fold)—current experimental scale is therefore limited
- Private NLM parameters increase parameter count (though they introduce a new scaling dimension)
- ImageNet accuracy of 72.47% is far below SOTA—the authors explicitly note that no hyperparameter search was conducted; this is an exploration rather than a competition submission
- Theoretical analysis is absent: the expressive capacity and convergence properties of the synchrony representation remain uncharacterized
- Validation on more complex tasks such as language modeling and video understanding is left as future work

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Neural synchrony as a learnable representation + NLMs constitute an entirely new paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task validation (maze/classification/algorithmic/RL) with ablations
- Writing Quality: ⭐⭐⭐⭐ Rich figures and excellent supplementary videos, though the paper is lengthy
- Value: ⭐⭐⭐⭐⭐ Rethinks the abstraction level of neurons; may open a new architectural direction
- Overall: ⭐⭐⭐⭐ Bold innovation; directional significance outweighs current performance numbers

## Related Work & Insights
- **vs. PonderNet / ACT**: These methods require explicit halting modules to learn when to stop. The CTM achieves adaptive computation naturally via min-loss + max-certainty without additional parameters
- **vs. LTCNs (Hasani et al. 2021)**: Liquid time-constant networks use ODEs to govern dynamics but do not employ synchrony as a representation. The CTM's synchrony matrix provides a richer higher-order representation space
- **vs. SNNs**: Discrete spikes and specialized learning rules are incompatible with standard deep learning training. The CTM maintains continuous values and gradient differentiability
- **vs. Reichert & Serre (2013)**: Synchrony is used only as post-hoc gating for segmentation. The CTM treats synchrony as a core latent representation optimized during training
- **vs. RIMs (Goyal et al. 2019)**: Modular asynchronous sub-networks lack a synchrony mechanism. The CTM's neural synchrony provides learnable global coordination
- The NLM paradigm is transferable to Transformers: introducing private temporal history processing for each attention head may enhance chain-of-thought reasoning quality

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[NeurIPS 2025\] AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking](acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)

</div>

<!-- RELATED:END -->
