---
title: >-
  [Paper Note] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking
description: >-
  [CVPR 2026][Video Understanding][Spiking Neural Networks] SpikeTrack is proposed as the first RGB visual tracking framework fully compliant with the spike-driven paradigm. Through asymmetric temporal step expansion…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Spiking Neural Networks"
  - "Visual Tracking"
  - "Energy Efficiency"
  - "Asymmetric Architecture"
  - "Memory Retrieval"
date: 2026-05-08
content_hash: 1e1000596de03780
---

# SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking

**Conference**: CVPR 2026
**arXiv**: [2602.23963](https://arxiv.org/abs/2602.23963)  
**Code**: Available (mentioned in paper)  
**Area**: Video Understanding
**Keywords**: Spiking Neural Networks, Visual Tracking, Energy Efficiency, Asymmetric Architecture, Memory Retrieval

## TL;DR
SpikeTrack is proposed as the first RGB visual tracking framework fully compliant with the spike-driven paradigm. Through asymmetric temporal step expansion, unidirectional information flow, and a brain-inspired Memory Retrieval Module (MRM), it achieves SOTA among SNN-based trackers and is on par with ANN-based trackers, while consuming only 1/26 the energy of TransT.

## Background & Motivation
Spiking Neural Networks (SNNs) achieve low-power computation by simulating the spatiotemporal dynamics and spike mechanisms of biological neurons: (i) computation is triggered only on event-driven activation, and (ii) matrix multiplications between spike tensors and weights can be converted to sparse additions. This gives SNNs significant energy efficiency advantages on neuromorphic hardware.

**Problems with existing SNN tracking methods**:

**RGB-based methods** (SiamSNN, Spike-SiamFC++): Although spiking neurons are used, spike signals are decoded into continuous values for computation, failing to achieve fully spike-driven processing, which limits energy efficiency.

**Event camera-based methods**: Directly mimic the ANN one-stream architecture (e.g., OSTrack), concatenating template and search region before feeding into the backbone for bidirectional interaction. This approach has two drawbacks:
   - It does not fully exploit the spatiotemporal correlation dynamics of SNN neurons.
   - Dense bidirectional interaction significantly increases computational overhead.

**Core Research Question**: Can an SNN tracker be designed that adheres to the spike-driven paradigm while fully exploiting spatiotemporal modeling capabilities?

## Method

### Overall Architecture
SpikeTrack consists of three components: a weight-shared spiking backbone, a Memory Retrieval Module (MRM) for unidirectional information transfer, and a prediction head. At inference time, the template branch executes only once during initialization or template update, caching intermediate-layer features as memory; the search branch uses MRM to retrieve target cues from memory and progressively refines target awareness.

### Key Designs

1. **Asymmetric Siamese Backbone**: Asymmetric temporal step inputs with unidirectional information flow

    - **Function**: The template branch expands over $T$ time steps (one template per step), jointly modeling template representations through neuronal spatiotemporal dynamics; the search branch performs efficient single-time-step inference.
    - **Mechanism**: Information flows only from the template branch to the search branch; the computationally intensive template branch runs only at initialization or update, substantially reducing computation.
    - **Backbone**: Spike-Driven Transformer V3, composed of CNN blocks (first two stages) and Transformer blocks (last two stages).
    - **Spiking neuron model**: Normalized Integer LIF (NI-LIF) neurons are adopted, using normalized integer activations during training and converting to equivalent spikes at inference. A key improvement is making the decay factor $\beta_t = \sigma(\theta_t)$ learnable, enabling the network to adaptively model inter-timestep correlations:
    $U[t] = \beta_t H[t-1] + Y[t], \quad S[t] = \text{Clip}(\text{round}(U[t]), 0, D)/D$

2. **Memory Retrieval Module (MRM)**: Brain-inspired memory retrieval for unidirectional information transfer

    - **Function**: Retrieves target cues from memory cached by the template branch to enhance target awareness in the search branch.
    - **Design Motivation**: In neuroscience, recurrent connections in the V1 L2/3 area achieve complete perceptual inference under occlusion through iterative refinement based on prior expectations — naturally suited to template-based tracking.
    - **Mechanism** (three-stage cyclic processing):
        - **Global contour encoding**: Template features $F_Z$ are projected into $K_S$, $V_S$; the memory matrix $M = K_S^T V_S$ is precomputed once at initialization. Search features $F_X$ are temporally expanded into $Q_S^{(0)}$, and global information is retrieved via $Q_S^{(i)'} = \mathcal{SN}(Q_S^{(i)}M \cdot scale)$.
        - **Detail construction**: $T$ dedicated SSConv layers process each time step along the temporal dimension, increasing sensitivity to temporal variations.
        - **Feedback refinement**: Residual connections and projection simulate feedback to higher visual areas.
    - Leverages the linear complexity of spike attention; the precomputed memory matrix is reused across frames.

3. **Prediction Head**: Three-branch center-point prediction

    - **Function**: Predicts the target bounding box from search branch features.
    - **Mechanism**: Three parallel branches respectively predict target center localization (classification), local offset due to resolution reduction, and normalized bounding box width/height. Each branch consists of multiple Conv-BN-NILIF layers.
    - No separate quality scoring module; the localization branch score is used directly as confidence.

### Loss & Training
- **Loss function**: $\mathcal{L} = \mathcal{L}_{class} + \lambda_G \mathcal{L}_{IoU} + \lambda_{L_1} \mathcal{L}_1$, where $\lambda_G=2$, $\lambda_{L_1}=5$
- $\mathcal{L}_{class}$: weighted focal loss; $\mathcal{L}_{IoU}$: generalized IoU loss; $\mathcal{L}_1$: L1 regression loss
- Training data: COCO + LaSOT + TrackingNet + GOT-10k
- Two-stage training: the $T=1$ model is trained for 320 epochs (backbone lr 4e-5, head/MRM lr 4e-4); models with $T>1$ are fine-tuned from the $T=1$ model for 60 epochs.
- Template update: FIFO queue, update interval 25 frames, confidence threshold 0.7.
- Energy computation: $E_{SNN} = \text{FLOPs} \times E_{AC} \times SFR \times T \times D$, $E_{AC}=0.9$ pJ (45nm), far lower than $E_{MAC}=4.6$ pJ.

## Key Experimental Results

### Main Results

| Dataset | Metric | SpikeTrack-B384 | TransT (ANN) | Energy Ratio |
|--------|------|-----------------|--------------|--------|
| LaSOT | AUC | 66.7 | 64.9 | 27.3 vs 75.2 mJ (1/2.8) |
| TrackingNet | AUC | 82.0 | 81.4 | 27.3 vs 75.2 mJ |
| GOT-10k | AO | 73.1 | 72.3 | 27.3 vs 75.2 mJ |
| TNL2K | AUC | 54.8 | 50.7 | 27.3 vs 75.2 mJ |

SpikeTrack-B256-T3 surpasses TransT by 2.2% AUC on LaSOT with only 1/7.6 the energy consumption.

| Dataset | Metric | SpikeTrack-S256 | SpikeSiamFC++ | Gain |
|--------|------|-----------------|---------------|------|
| UAV123 | AUC | 66.2 | 57.8 | +8.4 |
| OTB100 | AUC | 69.4 | 64.4 | +5.0 |
| GOT-10k | AO | 67.8 | - | - |

### Ablation Study

| Configuration | Energy (mJ) | GOT-10k AO | LaSOT AUC | Notes |
|------|----------|------------|-----------|------|
| Baseline (asymmetric) | 8.7 | 71.3 | 66.8 | Baseline |
| One-stream | 22.8 | 70.8 | 65.4 | Energy ↑163%, accuracy ↓ |
| Vanilla Cross-attn | 7.6 | 70.9 | 65.0 | Replaces MRM; accuracy drops |
| Modulation (spike) | 6.8 | 58.3 | 49.9 | AsymTrack approach unsuitable for SNN |
| Mean Fusion | 8.5 | 71.0 | 66.2 | Channel-weighted fusion is superior |
| Fixed Decay | 8.9 | 68.9 | 66.0 | Learnable decay factor performs better |

### Key Findings
- The asymmetric architecture outperforms the one-stream architecture in both accuracy and energy efficiency, demonstrating that SNN spatiotemporal dynamics + MRM is superior to brute-force bidirectional interaction.
- AsymTrack's template modulation approach degrades severely after spiking conversion (AUC 49.9), indicating that using templates as convolutional kernels for signal modulation is incompatible with SNN's coarse-grained representations.
- The learnable decay factor outperforms fixed decay (+ 1.9 LaSOT AUC), providing more flexible control over inter-timestep interactions.
- The performance gap with OSTrack is primarily in Deformation and Fast Motion scenarios, which pose the greatest challenges to deep semantic understanding and re-detection capability in SNNs.
- MRM with $N=1$ iteration is optimal; excessive iterations introduce accumulated errors and over-focus.

## Highlights & Insights
1. **Elegance of the asymmetric design**: The template branch leverages SNN spatiotemporal dynamics over multiple time steps, while the search branch performs efficient single-time-step inference — combining SNN's spatiotemporal modeling strengths with the efficiency of Siamese architectures.
2. **Brain-inspired MRM**: Inspired by recurrent connections in the V1 visual cortex, the precomputed memory matrix enables cross-frame reuse, balancing biological plausibility with engineering efficiency.
3. **First RGB tracking framework** to achieve an energy–accuracy Pareto optimum — surpassing ANN trackers of equivalent accuracy while reducing energy by orders of magnitude.
4. **Six model variants** cover diverse accuracy–power requirements, demonstrating strong scalability.

## Limitations & Future Work
- Performance degrades under similar-object distractor scenarios — spike encoding struggles to convey fine-grained semantic information for discriminating similar targets.
- Template update relies on a simple confidence threshold strategy, lacking a dedicated quality scoring module.
- In long-term tracking (LaSOT), increasing $T$ does not always improve performance, as the simple scoring mechanism can introduce low-quality templates.
- Energy consumption is currently computed theoretically under 45nm process; no validation has been conducted on actual neuromorphic hardware.

## Related Work & Insights
- Inherits the asymmetric Siamese concept from AsymTrack (CVPR'25), replacing ANN template modulation with SNN spatiotemporal dynamics.
- The backbone adopts Spike-Driven Transformer V3, a Meta-Transformer-style SNN.
- The memory precomputation in MRM shares conceptual similarity with KV-cache in Transformers.
- Provides an important reference for applying SNNs to broader video understanding tasks, such as MOT and video segmentation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Asymmetric spike-driven tracking + brain-inspired MRM design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 7 benchmarks, 6 variants, comprehensive ablation and energy analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the correspondence with neuroscience is well articulated.
- **Value**: ⭐⭐⭐⭐ Advances the practical applicability of SNNs in visual tracking.
- **Value**: To be evaluated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](drift-resilient_temporal_priors_for_visual_tracking.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[CVPR 2026\] CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)

</div>

<!-- RELATED:END -->
