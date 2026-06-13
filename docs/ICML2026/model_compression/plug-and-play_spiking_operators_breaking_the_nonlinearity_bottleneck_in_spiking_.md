---
title: >-
  [Paper Note] Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers
description: >-
  [ICML 2026][Model Compression][Spiking Neural Networks] The authors decompose the three most challenging nonlinear operators in Transformers (Softmax, SiLU…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Spiking Neural Networks"
  - "Transformer Nonlinear Operators"
  - "ANN-to-SNN Conversion"
  - "LIF Neurons"
  - "Training-free"
date: 2026-05-08
content_hash: 6ee749434272346e
---

# Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers

**Conference**: ICML 2026  
**arXiv**: [2605.20289](https://arxiv.org/abs/2605.20289)  
**Code**: Not yet public  
**Area**: Model Compression / Spiking Neural Networks / ANN-to-SNN / Neuromorphic Hardware  
**Keywords**: Spiking Neural Networks, Transformer Nonlinear Operators, ANN-to-SNN Conversion, LIF Neurons, Training-free  

## TL;DR
The authors decompose the three most challenging nonlinear operators in Transformers (Softmax, SiLU, RMSNorm) into three common primitives: "division / exponential / $\ell_2$ norm." These are implemented as spike-friendly modules using LIF neuron populations and bit-shift scaling. The modules can be assembled like building blocks and are plug-and-play with existing ANN-to-SNN pipelines without any fine-tuning, achieving $<1\%$ accuracy loss on models such as LLaMA-3-8B, Qwen3-8B, and BERT.

## Background & Motivation

**Background**: Deploying large models on neuromorphic hardware (e.g., Loihi, TrueNorth) for event-driven inference is a clear path for energy efficiency optimization. Recently, ANN-to-SNN conversion (direct mapping of activations to spike rates without retraining) has been extended to Transformers and LLMs, represented by works like SpikeZIP-TF and SpikeLLM.

**Limitations of Prior Work**: Most existing ANN-to-SNN works only handle linear operators in Transformers (matrix multiplication, FFN projections). Nonlinear operators like Softmax, SiLU, and RMSNorm are either bypassed or offloaded to an external CPU. The issue is that the core data path of neuromorphic chips only supports lightweight operations like spikes, shifts, and additions, and is not adept at floating-point division, exponentials, or square roots. Offloading nonlinear operators to an external processor introduces significant cross-domain data movement overhead, neutralizing the energy efficiency advantages of spiking computation.

**Key Challenge**: To achieve "strictly spike-only" deployment, these nonlinear operators must also be spiked. However, standard LIF dynamics $v(t) = \lambda v(t-1) + I(t)$, $s(t) = \mathbb{I}[v(t) \geq \theta]$ naturally perform approximately linear "accumulate-threshold-reset" mappings. Forcing division or $\sqrt{\cdot}$ into this framework either requires training or breaks compatibility with existing conversion pipelines.

**Goal**: Design a **training-free**, **modular**, **LIF-primitive-based** implementation of nonlinear operators that can be directly plugged into existing ANN-to-SNN pipelines like SpikeZIP or SpikeLLM without modifying weights or the pipeline.

**Key Insight**: The authors observe that Softmax, SiLU, and RMSNorm share the same algebraic structure: a numerator related to the input and a denominator acting as a non-negative normalization term. If "numerator-denominator-division" can be decomposed into independent modules where each module uses only LIF and shifts, then different nonlinear operators become mere combinations of these primitives.

**Core Idea**: Decompose nonlinear operators into three spike-native primitives—"division + exponential + $\ell_2$ norm"—and recombine them modularly to reconstruct Softmax, SiLU, and RMSNorm.

## Method

### Overall Architecture
The design of NLSpiking is a three-layer structure. The bottom layer consists of three spike-native primitives: Division Neuron Group, PolarNorm Unit ($\ell_2$ norm), and PWL-Exp Unit (exponential). The middle layer provides the "numerator-denominator" decomposition of nonlinear operators: each target operator is rewritten as $\phi(x) = \text{num}(x) / \text{denom}(x)$, with the numerator and denominator approximated by their respective primitives. The top layer consists of NLSpiking operators (NLS-Softmax / NLS-SiLU / NLS-RMS), which invoke the same set of primitives, differing only in the construction of the numerator and denominator. The entire pipeline is decoupled from the original ANN-to-SNN conversion framework; after SpikeZIP-TF converts the linear layers, nonlinear operators can be replaced by NLSpiking versions without any retraining.

### Key Designs

1.  **Division Neuron Group (Spike-native Integer Division)**:
    - **Function**: Uses a population of LIF neurons to achieve an integer approximation of $q \approx I_A / I_B$, treating division as a spike-native primitive rather than a floating-point operation.
    - **Mechanism**: Executed in two stages. The first stage accumulates the spiking denominator $I_B(t)$ over time to get $I_B = \sum_{t=1}^T I_B(t)$, then computes a baseline threshold via a right shift $\theta = I_B \gg n = \lfloor I_B / 2^n \rfloor$, where $n = \log_2(TL)$. Thresholds $\theta_i = i\theta$ are assigned to the $i$-th neuron in the group. The second stage feeds the spiking numerator $I_A(t)$ into the group; neuron $i$ fires if and only if $I_A(t) \geq i\theta$. Finally, the quotient is obtained by counting active neurons $q = \sum_i s_i$ and right-shifting: $\hat q = q \gg n = \lfloor \sum_t I_A(t) / \theta \rfloor$. The process uses only LIF threshold comparisons and shifts.
    - **Design Motivation**: On neuromorphic hardware, "population competition" (which neurons are active) is naturally supported, whereas floating-point division is not. By encoding the "magnitude" of the denominator into the "threshold gradient" of the population, the authors transform division into a look-up problem of "finding the maximum $i$ such that $v(t) \geq i\theta$."

2.  **PolarNorm Unit (CORDIC-Hypot style $\ell_2$ Norm)**:
    - **Function**: Approximates $\|\mathbf v\|_2 = \sqrt{\sum_i x_i^2 + \epsilon d}$ specifically for RMSNorm without using multipliers or square root units.
    - **Mechanism**: Expands the input into $\mathbf v = [x_1, \dots, x_d, \sqrt{\epsilon d}]$ and merges adjacent elements in a binary tree. Each merge uses CORDIC-Hypot iterations: $x_{k+1} = x_k + d_k \cdot y_k / 2^k$, $y_{k+1} = y_k - d_k \cdot x_k / 2^k$ (where $d_k = \text{sign}(y_k)$). After $n$ steps, $x_n \approx \sqrt{x^2 + y^2}$. A fixed gain reciprocal $1/K_n$ (precisely a power of 2) is used for scaling. The process involves only shifts and additions.
    - **Design Motivation**: Directly expanding $x^2$ and $\sqrt{\cdot}$ is nearly impossible in the spiking domain. CORDIC uses iterative rotations to unify these nonlinear operations into "shift + add/sub + sign check," fitting the instruction sets of neuromorphic hardware.

3.  **PWL-Exp Unit (Piecewise Linear Exponential Approximation)**:
    - **Function**: Approximates $e^x$ on the interval $[-L, L]$ using $K$ piecewise linear segments for Softmax and SiLU.
    - **Mechanism**: Divides $[-L, L]$ into $K$ equal segments of width $\gamma = 2L/K$. Each segment uses linear interpolation $e^x \approx ax + b = \frac{e^{x_{i+1}} - e^{x_i}}{x_{i+1} - x_i}(x - x_i) + e^{x_i}$, where $x_i = -L + \gamma i$. Slopes $a$ and intercepts $b$ are precomputed and stored in an 8-bit LUT. Runtime entails one LUT access and one fixed-point multiply-accumulate.
    - **Design Motivation**: Softmax/SiLU rely on $\exp$, but the spiking domain cannot compute exponentials directly. Replacing "runtime exp" with "precomputed 8-bit coefficients + shift scaling" preserves accuracy (analytic error $\varepsilon_{\exp} = \frac{L^2}{2K^2} e^{2L/K}$) while keeping memory overhead within tens of bytes, suitable for on-chip SRAM constraints.

### Loss & Training
The method is **training-free**, introducing no loss functions or fine-tuning. Errors from all approximations are exposed via operator-level replacement after ANN-to-SNN conversion. The authors provide relative error bounds for each operator:

-   Softmax: $|\tilde\phi_i - \phi_i| / \phi_i \leq \frac{2}{1 - \varepsilon_{\exp}}(\varepsilon_{\exp} + \Delta)$
-   SiLU: $|\tilde\phi(x) - \phi(x)| \leq |x| \cdot \frac{2\varepsilon_{\exp}}{1 - \varepsilon_{\exp}} + |x|\Delta$
-   RMSNorm: $|\tilde\phi_i - \phi_i| / \phi_i \leq \frac{\varepsilon_{\text{pol}} + \Delta}{1 - \varepsilon_{\text{pol}}} + \sqrt{d}\Delta$

Where $\Delta = 1/n$ is the quantization step for $(T, L)$-Division, and $\varepsilon_{\text{pol}} = \lceil \log_2 d\rceil \cdot 2^{-2n-1}$ is the CORDIC tree error. Recommended parameters $H=5, K=64$ yield $\varepsilon_{\exp} \leq 3.63 \times 10^{-3}$.

## Key Experimental Results

### Main Results
Model-level evaluation covers: (1) SNN-LLMs converted via SpikeLLM/SpikeZIP; (2) Standard ANN-LLMs not explicitly covered by existing pipelines.

| Model | Task Avg | Original Op | NLSpiking Op | Gain ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA-3-8B (5 tasks avg) | Avg Acc | 0.730 | 0.727 | -0.003 |
| LLaMA-2-7B (5 tasks avg) | Avg Acc | 0.686 | 0.684 | -0.002 |
| Mistral-7B (5 tasks avg) | Avg Acc | 0.724 | 0.724 | +0.000 |
| Qwen3-8B (5 tasks avg) | Avg Acc | 0.734 | 0.748 | **+0.014** |
| SpikeLLM T=2,W2A16 LLaMA-2-7B | Avg Acc | 0.477 | 0.477 | -0.000 |
| SpikeLLM T=2,W2A16 LLaMA-2-13B | Avg Acc | 0.516 | 0.515 | -0.001 |
| SpikeZIP BERT (4 tasks avg) | Avg Acc | 0.807 | 0.810 | +0.003 |

Accuracy changes across WinoGrande, HellaSwag, ArcC, ArcE, and PIQA tasks are $<1\%$, with NLSpiking even slightly outperforming on Qwen3-8B (+1.4%).

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| NLS-Softmax | Lowest mean error across dimensions | Superior to Padé / PWL / Sorbet / hardmax |
| NLS-SiLU | Mean error on par with 64-segment PWL-sigmoid | Best among training-free baselines |
| NLS-RMS | Stable across dimensions | Solves alignment issues in blockwise RMS |
| $H = 3/4/5$ (SiLU range) | Minimal mean/max error | $H=5$ is recommended |
| Increasing $H$ (Softmax, $d=64$) | Monotonic error decrease | Softmax requires larger truncation intervals |

### Key Findings
-   The assumption that "energy for nonlinear operators is negligible" is false in spike-only deployment; once offloaded, data movement costs dominate.
-   Three primitives share a single LUT requiring only $K$ 8-bit and 16-bit values, much smaller than traditional tables and within Loihi's SRAM limits.
-   Latency analysis shows NLSpiking requires only $n$ shift-adds/LUT calls per step with zero data movement, whereas SpikeZIP/SpikeLLM rely on external processors.
-   Experimental results align with theoretical bounds, proving that modular decomposition avoids error accumulation.

## Highlights & Insights
-   Treating "division as a spike-native primitive" is the most counter-intuitive yet vital design. It transforms a hardware-unfriendly operation into population competition, which hardware excels at.
-   Adapting CORDIC for $\ell_2$ norms is an ingenious migration from 70s hardware floating-point techniques to modern SNN operator design.
-   The "numerator-denominator" abstraction provides an extensible template; adding GeLU or LayerNorm in the future would only require new primitive modules rather than a complete redesign.
-   The training-free nature is crucial for practical deployment, allowing NLSpiking to be applied to existing weights without harming precision.

## Limitations & Future Work
-   Ours lacks end-to-end deployment on physical neuromorphic hardware (Loihi/TrueNorth); results are currently based on software simulation.
-   Nonlinear operator coverage is limited to Softmax, SiLU, and RMSNorm; GeLU, Mish, and trigonometric operations in RoPE/ALiBi are not yet covered.
-   $H$ selection is task-dependent (SiLU prefers small $H$, Softmax prefers large $H$), potentially leading to hardware configuration fragmentation.
-   PWL-Exp under 8-bit quantization is near its error floor; further precision may require non-uniform segments or hierarchical LUTs.
-   Future work could combine NLSpiking with Quantization-Aware Training (QAT) to adapt models to LIF noise before conversion.

## Related Work & Insights
-   **vs SpikeZIP-TF (You et al. 2024)**: SpikeZIP spikes linear operators but offloads non-linears; Ours completes the puzzle and can be used as a backend.
-   **vs SpikeLLM (Xing et al. 2025)**: SpikeLLM focuses on saliency-driven allocation; Ours is orthogonal, rewriting the operators themselves.
-   **vs Sorbet (Tang et al. 2025)**: Sorbet uses shift-based operations but requires distillation and fine-tuning; Ours is entirely training-free.
-   **vs FAS / LAS (Chen et al. 2025)**: These focus on lossless conversion but still use external non-linears.
-   **vs XNOR-Net / DoReFa-Net**: These suffer high errors in SiLU/Softmax; Ours demonstrates that "shift-add + LUT" can significantly lower error at similar hardware costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](../../NeurIPS2025/model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[ICML 2026\] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)
- [\[NeurIPS 2025\] S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection](../../NeurIPS2025/model_compression/s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)

</div>

<!-- RELATED:END -->
