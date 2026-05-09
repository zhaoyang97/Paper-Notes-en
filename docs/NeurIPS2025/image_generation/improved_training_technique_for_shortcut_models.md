---
title: >-
  [Paper Note] Improved Training Technique for Shortcut Models (iSM)
description: >-
  [NeurIPS 2025][Image Generation][Shortcut Models] Targeting five key performance bottlenecks of Shortcut Models (compounding guidance, fixed guidance, frequency bias, self-consistency deviation, and curved trajectories), this paper proposes iSM, a unified training framework that incorporates intrinsic guidance, multi-level wavelet loss, scaling optimal transport, and twin EMA strategy, achieving substantial improvements on ImageNet 256×256 with one-step FID 5.27 and four-step FID 2.05.
tags:
  - NeurIPS 2025
  - Image Generation
  - Shortcut Models
  - Flow Matching
  - Few-step Generation
  - CFG Guidance
  - Wavelet Loss
  - Optimal Transport
  - EMA Strategy
date: 2026-05-08
content_hash: 33c5701e0c9cfbae
---

# Improved Training Technique for Shortcut Models (iSM)

**Conference**: NeurIPS 2025
**arXiv**: [2510.21250](https://arxiv.org/abs/2510.21250)
**Code**: Not released
**Area**: Image Generation
**Keywords**: Shortcut Models, Flow Matching, Few-step Generation, CFG Guidance, Wavelet Loss, Optimal Transport, EMA Strategy

## TL;DR
Targeting five key performance bottlenecks of Shortcut Models (compounding guidance, fixed guidance, frequency bias, self-consistency deviation, and curved trajectories), this paper proposes iSM, a unified training framework that incorporates intrinsic guidance, multi-level wavelet loss, scaling optimal transport, and twin EMA strategy, achieving substantial improvements on ImageNet 256×256 with one-step FID 5.27 and four-step FID 2.05.

## Background & Motivation
**State of the Field**: Flow Matching-based diffusion models have achieved remarkable results in image generation, but sampling requires a large number of iterative steps, limiting deployment efficiency. Accelerating sampling is a key research direction.

**Advantages of Shortcut Models**: Shortcut Models (SM) represent an elegant acceleration approach — the network is conditioned on both noise level $t$ and target step size $d$ simultaneously, trained with a self-consistency loss so that a single network supports one-step, few-step, and multi-step sampling.

**Limitations of SM**: Despite the elegant framework design, the practical performance of SM lags far behind comparable methods (one-step FID 10.60 vs. IMM 7.77), impeding widespread adoption.

**Limitations of Prior Work**: The authors identify five systematic issues — (1) exponential compounding amplification of CFG guidance at large step sizes; (2) fixed guidance strength during training, leading to loss of inference flexibility; (3) low-frequency bias from pixel-level loss causing blurry generated images; (4) conflict between the temporal lag of the EMA target network and the self-consistency objective; (5) curved generation trajectories caused by random noise-data pairing.

**Starting Point**: Rather than modifying the core framework of SM, the paper systematically addresses the five bottlenecks above to make SM a genuinely competitive generative paradigm.

## Core Problem

### Problem 1: Compounding Guidance
This is one of the most important findings of the paper. SM constructs self-consistency targets using a fixed CFG strength $w=1.5$ during training. When the model performs one-step generation $s_\theta(x_0, 0, c, 1)$, it implicitly aggregates the effects of $N=128$ intermediate steps. The authors formally prove for the first time that the effective guidance strength at each implicit intermediate step is not $w$ but $w' = w^{\log_2(N)}$. When $N=128, w=1.5$, $w' = 1.5^7 \approx 17$, causing severe color oversaturation and blurring artifacts.

### Problem 2: Inflexible Fixed Guidance
The original SM hard-codes CFG strength as $w=1.5$ during training, making it impossible to adjust the diversity-fidelity trade-off at inference time. Moreover, the optimal $w$ value depends on the number of inference steps, and a fixed value cannot adapt to different scenarios.

### Problem 3: Frequency Bias
SM uses pixel-level $\ell_2$ loss to optimize direct-domain predictions. Neural networks naturally tend to learn low-frequency features first, causing generated images to lack high-frequency texture details and appear overall blurry.

### Problem 4: Self-Consistency Deviation
The EMA target network uses a slow decay rate to maintain training stability, but this causes the target network to represent a "historical state" of the online network. The online network must simultaneously optimize the current trajectory (flow matching) and align with the historical target (self-consistency), creating conflicting learning signals.

### Problem 5: Curved Flow Trajectories
Random noise-data pairing in standard flow matching causes frequent crossing of forward trajectories, forcing the reverse generation process to follow curved paths and increasing the difficulty of large-step prediction.

## Method

### Overall Architecture
iSM preserves the core dual-loss structure of SM (flow matching + self-consistency) and introduces four key improvement components. The final training objective is:
$$\mathcal{L}_{\text{total}}(\theta) = \alpha \mathcal{L}_{\text{velocity}}(\theta) + \beta \mathcal{L}_{\text{guidance}}(\theta) + \gamma \mathcal{L}_{\text{consistency}}(\theta)$$
where $\alpha = \beta = \gamma = 1$.

### Key Design 1: Intrinsic Guidance
- **Function**: Takes guidance strength $w$ as an explicit conditioning input to the network, training the model to directly output CFG-modulated velocity $s_\theta(x_t, t, c, d, w)$
- **Core Idea**:
    - Flow matching objective: trains the base velocity field at $d=0, w=0$, with standard dropout randomly introducing null conditioning
    - Intrinsic guidance objective: trains the model to directly learn CFG scaling behavior at $d=0, w>0$. The target is $s_{\text{velocity}} + w \cdot \text{sg}(s_{\text{guidance}})$, where stop-gradient prevents interference with base predictions
    - Guided self-consistency objective: maintains self-consistency at arbitrary step sizes and guidance strengths for $d>0, w \geq 0$
- **Effect**: Eliminates compounding guidance, supports flexible adjustment of $w$ at inference, enables CFG in a single step, and halves inference time (no additional unconditional forward pass required)
- **Interval Guidance**: Guidance is not applied in high-noise regions ($t < t_{\text{interval}} = 0.3$) to avoid premature mode collapse

### Key Design 2: Multi-Level Wavelet Loss
- **Function**: Uses the Discrete Wavelet Transform (DWT) to decompose predictions and targets into multi-frequency representations, computing the loss in the wavelet domain
- **Core Idea**: Recursively decomposes wavelet subbands to $L=5$ levels (the maximum decomposition depth for a $32 \times 32$ latent space), computing errors independently at each frequency band
- **Effect**: Introduces frequency-aware error signals, forcing the model to recover high-frequency details neglected by $\ell_2$ loss, producing sharper textures

### Key Design 3: Scaling Optimal Transport (sOT)
- **Function**: Decouples the batch size for OT computation from the training batch size, enabling large-scale OT matching
- **Core Idea**: Every $K$ training batches (of size $M$), all $K \times M$ noise-image samples are pooled to compute a global OT plan once, then split back into $K$ mini-batches for training
- **Implementation**: $K=32$, with only approximately 4% additional training time overhead
- **Effect**: Substantially reduces forward trajectory crossings, producing straighter reverse paths and reducing self-consistency and flow matching losses

### Key Design 4: Twin EMA
- **Function**: Maintains two sets of EMA parameters in place of the traditional single EMA
- **Core Idea**:
    - Inference parameters $\theta_{\text{infer}}^-$: slow decay rate (standard), used only for inference to ensure stable high-quality generation
    - Target parameters $\theta_{\text{target}}^-$: fast decay rate ($\rho = 0.95$), used to generate self-consistency targets, closely tracking the current state of the online network
- **Effect**: The self-consistency target seen by the online network more closely reflects the current distribution, eliminating the conflict of "aligning with a historical version," while inference still benefits from the stability of slow decay

## Loss & Training

### Training Configuration
- **Backbone**: SiT-XL/2 (675M parameters), latent space size $32 \times 32$
- **Dataset**: ImageNet 256×256 (class-conditional generation), encoded to latent space using sd-vae-ft-mse
- **Training Iterations**: 800K (main experiments), 250K (ablation studies)
- **Guidance Scale Sampling**: $w$ is discretely sampled from $[0, w_{\max}=3.5]$ in steps of 0.25; excessively high $w_{\max}$ (5.0) introduces unnecessary complexity, while too low (2.0) lacks high-quality guidance
- **Interval Guidance Threshold**: $t_{\text{interval}} = 0.3$; below this threshold $w=0$ (no guidance applied)
- **Wavelet Loss**: $L=5$ levels of DWT decomposition (maximum feasible depth for $32 \times 32$ latent space $= \log_2(32) = 5$)
- **sOT Parameters**: $K=32$; samples are pooled every 32 batches to compute a global OT plan, with approximately 4% additional training time
- **Twin EMA**: Inference parameters $\theta_{\text{infer}}^-$ decay rate 0.9999 (standard slow decay), target parameters $\theta_{\text{target}}^-$ decay rate 0.95 (fast decay)
- **Conditioning Dropout**: During training, conditions are randomly replaced with null conditioning $\varnothing$ with a certain probability, following standard CFG training paradigm

### Inference Procedure
- Inference uses slow-decay EMA parameters $\theta_{\text{infer}}^-$ to generate samples
- The number of sampling steps NFE $\in \{1, 2, 4, 8, 128\}$ can be freely selected without retraining
- Guidance strength $w$ can be freely adjusted at inference time; the network directly outputs CFG-modulated velocity — **no additional unconditional forward pass is required**, reducing inference time by approximately 50% compared to standard CFG
- One-step generation: directly $x_1 = x_0 + s_\theta(x_0, 0, c, 1, w)$; multi-step generation iterates via Euler integration

## Key Experimental Results

### Main Results (ImageNet 256×256, SiT-XL/2, 800K Iterations)

| Model | NFE | FID-50K ↓ |
|------|-----|-----------|
| SM (original) | 1 | 10.60 |
| IMM | 1 | 7.77 |
| **iSM (Ours)** | **1** | **5.27** |
| SM | 2 | — |
| IMM | 2 | 3.99 |
| **iSM** | **2** | **2.44** |
| SM | 4 | 7.80 |
| IMM | 4 | 2.51 |
| **iSM** | **4** | **2.05** |
| SM | 8 | — |
| IMM | 8 | 1.99 |
| **iSM** | **8** | **1.93** |
| SM | 128 | 3.80 |
| **iSM** | **128** | **1.88** |

### Additional Metric Validation (FD-DINOv2 / IS)

| Model | NFE | FD-DINOv2 ↓ | IS ↑ |
|------|-----|-------------|------|
| SM | 1 | 500.92 | 102.66 |
| IMM | 1 | 247.78 | 128.87 |
| **iSM** | **1** | **232.31** | **223.52** |
| SM | 2 | 329.53 | 125.66 |
| IMM | 2 | 152.08 | 173.66 |
| **iSM** | **2** | **107.63** | **302.29** |
| SM | 4 | 265.90 | 136.79 |
| IMM | 4 | 110.88 | 204.95 |
| **iSM** | **4** | **83.70** | **298.23** |

Notably, iSM's improvement on FD-DINOv2 is even more pronounced: at 4 steps, iSM (83.70) reduces the score by more than 3× compared to SM (265.90), indicating that the improvements are not confined to the Inception feature space. The IS metric also demonstrates a substantial advantage; at 2 steps, iSM (302.29) is 2.4× that of SM (125.66).

### Ablation Study (250K Iterations, Incremental Stacking, Per-Block Hyperparameter Search)

The optimal hyperparameters for each component are determined within its corresponding ablation block and carried forward to the search for the next component.

| Component | FID (1-step) | FID (4-step) |
|------|-----------|-----------|
| Intrinsic Guidance ($w_{\max}=3.5$) | 9.62 | 3.17 |
| + Interval Guidance ($t=0.3$) | 8.49 | 2.81 |
| + Wavelet Loss ($L=5$) | 8.12 | 2.64 |
| + sOT ($K=32$) | 7.97 | 2.23 |
| + Twin EMA ($\rho=0.95$) | **6.56** | **2.16** |

### Key Findings on Hyperparameter Sensitivity

- **$w_{\max}$**: 2.0 → 3.5 → 5.0 corresponds to 1-step FID 10.10 → 9.62 → 10.38, exhibiting a sweet spot
- **$t_{\text{interval}}$**: 0.0 → 0.1 → 0.3 → 0.5 corresponds to 4-step FID 3.17 → 3.14 → 2.81 → 2.84; at 0.5, 1-step FID jumps sharply to 19.22 (guidance coverage too narrow)
- **DWT levels**: 0 → 1 → 3 → 5 layers show monotonically increasing effectiveness; 5 layers is the maximum decomposition limit for the latent space
- **sOT $K$ value**: 0 → 1 → 8 → 32, 1-step FID drops from 8.12 → 8.07 → 8.03 → 7.97; 4-step FID drops from 2.64 → 2.51 → 2.28 → 2.23
- **Target EMA decay rate**: 0.9999 → 0.999 → 0.95, 1-step FID drops from 7.97 → 7.43 → 6.56; fast decay yields the largest improvement for one-step generation

### High-Resolution Generalization (ImageNet 512×512, FlowDCN Architecture, 300K Iterations)

| Model | NFE | FID ↓ | Precision ↑ | Recall ↑ |
|------|-----|-------|-------------|----------|
| SM | 1 | 43.81 | 0.56 | 0.11 |
| **iSM** | **1** | **37.05** | **0.60** | **0.55** |
| SM | 4 | 12.16 | 0.86 | 0.19 |
| **iSM** | **4** | **9.94** | 0.78 | **0.62** |

The substantial improvement in Recall (0.11 → 0.55, 0.19 → 0.62) indicates that iSM dramatically improves sample diversity — a direct consequence of intrinsic guidance eliminating the mode collapse caused by compounding guidance. The slight drop in Precision at 4 steps (0.86 → 0.78) is consistent with the expected diversity-fidelity trade-off.

## Highlights & Insights
1. **Formalization of Compounding Guidance**: The paper is the first to rigorously prove the exponential compounding problem of CFG in SM ($w' = w^{\log_2 N}$), explaining the root cause of one-step generation artifacts — an important theoretical insight.
2. **Comprehensive and Systematic Problem Diagnosis**: Rather than addressing issues in isolation, the paper identifies five bottlenecks at once and provides a unified solution framework.
3. **Clear Contribution of Each Component**: The ablation study clearly demonstrates the incremental gains of each improvement, from 9.62 → 6.56 (1-step FID), with well-documented design decisions.
4. **Architecture and Resolution Generalization**: The framework's generality is validated on FlowDCN + 512×512, with the Recall improvement from 0.11 to 0.55 being particularly notable.
5. **Controllable Training Overhead**: sOT adds only ~4% training time, Twin EMA introduces negligible additional parameters, and the overall improvements do not impose significant computational burden.

## Limitations & Future Work
1. **Validation Only on ImageNet**: The absence of experiments on text-to-image (e.g., T2I) or larger-scale datasets leaves the practical applicability to be further confirmed.
2. **Limited Comparison with Distillation Methods**: Distillation methods such as DMD and DMD2 offer more aggressive compression for few-step generation; the paper does not provide a thorough comparison.
3. **Remaining Gap in One-Step FID**: Compared to GANs (StyleGAN-XL 2.30), iSM's one-step FID of 5.27 still shows a notable gap, and the ceiling of the SM framework in extreme one-step scenarios remains to be explored.
4. **Upper Bound of sOT $K$**: The paper stops at $K=32$; whether larger $K$ would yield continued gains is unknown, and increasing $K$ introduces greater inter-batch latency.
5. **Generality of Wavelet Loss**: The multi-level DWT loss is hard-coded for a $32 \times 32$ latent space; different latent resolutions require re-adjusting the number of decomposition levels.

## Related Work & Insights

| Dimension | SM (Original) | IMM | iSM (Ours) |
|------|-----------|-----|-------------|
| Variable Steps | ✅ Single network, multi-step | ✅ Single network, multi-step | ✅ Single network, multi-step |
| CFG Flexibility | ❌ Fixed at training | Partial | ✅ Adjustable at inference |
| 1-Step FID | 10.60 | 7.77 | **5.27** |
| 4-Step FID | 7.80 | 2.51 | **2.05** |
| Frequency Awareness | ❌ | ❌ | ✅ Wavelet Loss |
| OT Matching | ❌ | ❌ | ✅ Scaling OT |
| EMA Strategy | Single EMA | Single EMA | Twin EMA |

**Insights and Connections**:
1. **Intrinsic Guidance is Transferable**: Conditioning on CFG strength as a network input can be directly transferred to other generative models requiring CFG (e.g., consistency models, Flow Matching) and serves as a general performance improvement technique.
2. **Lessons from Wavelet-Domain Loss**: Performing frequency decomposition in latent space rather than pixel space is efficient and effective, and warrants exploration in other latent-space diffusion models.
3. **Twin EMA Concept**: The design of using distinct EMA parameters for training and inference is concise and effective, and may be applicable to other self-consistency or self-distillation frameworks.
4. **Cross-Batch Pooling in sOT**: Achieving large-scale OT at minimal cost can serve as a general training enhancement for Flow Matching.
5. **Complementarity with VeCoR**: VeCoR improves FM via contrastive regularization of velocity fields, while iSM improves SM through training strategies — the two are orthogonal, and SM + VeCoR + iSM may be a combination worth exploring.

## Rating
- Novelty: ⭐⭐⭐⭐ (The theoretical analysis of compounding guidance and the Twin EMA design are novel, though individual components are not entirely new in isolation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablations and diverse metrics, but lacks T2I and larger-scale validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear problem-solution structure, high-quality figures, and rigorous mathematical derivations)
- Value: ⭐⭐⭐⭐ (Systematically improves the competitiveness of SM and provides a practical guide for variable-step generation paradigms)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scaling Offline RL via Efficient and Expressive Shortcut Models](scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)
- [\[NeurIPS 2025\] Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)
- [\[ICCV 2025\] Improved Noise Schedule for Diffusion Training](../../ICCV2025/image_generation/improved_noise_schedule_for_diffusion_training.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)

<!-- RELATED:END -->
