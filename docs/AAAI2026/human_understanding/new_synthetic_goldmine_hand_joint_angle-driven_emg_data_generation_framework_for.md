---
title: >-
  [Paper Note] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition
description: >-
  [AAAI 2026][Human Understanding][EMG signal generation] This paper proposes SeqEMG-GAN, a conditional adversarial generation framework driven by hand joint angle sequences. Through the joint design of an angle encoder…
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "EMG signal generation"
  - "gesture recognition"
  - "conditional GAN"
  - "joint angle-driven"
  - "data augmentation"
date: 2026-05-08
content_hash: 8803b4063dcaccf9
---

# New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition

**Conference**: AAAI 2026
**arXiv**: [2509.23359](https://arxiv.org/abs/2509.23359)  
**Code**: None  
**Area**: Human Understanding
**Keywords**: EMG signal generation, gesture recognition, conditional GAN, joint angle-driven, data augmentation

## TL;DR

This paper proposes SeqEMG-GAN, a conditional adversarial generation framework driven by hand joint angle sequences. Through the joint design of an angle encoder, a two-level context encoder (featuring the novel Ang2Gist unit), a deep convolutional generator, and a multi-view discriminator, the framework synthesizes high-fidelity EMG signals from joint kinematic trajectories, enabling zero-shot generation for unseen gestures. Mixing synthetic and real data for training improves classification accuracy from 57.77% to 60.53%.

## Background & Motivation

EMG-based gesture recognition is a key technology in human-computer interaction (HCI), with broad applications in neuroprosthetics, AR/VR interfaces, and wearable assistive technologies. However, practical deployment faces three major bottlenecks:

**Scarcity of labeled data**: EMG data collection is costly. Existing public datasets (e.g., Ninapro, CapgMyo) are mostly collected under predefined isolated gestures with limited degrees of freedom (DoFs), failing to support natural transitions and free combinations. Although the latest emg2pose dataset provides 370+ hours of data from 193 users, annotation remains expensive.

**Large cross-user/cross-session variability**: Different users have different anatomical structures and muscle recruitment patterns; within the same user, electrode displacement, skin impedance, and fatigue cause significant signal variation across sessions.

**Limitations of existing generative methods**:
   - Traditional augmentation (time warping, noise injection) lacks semantic consistency
   - Label-conditioned GANs can only generate seen gesture categories and cannot handle unseen gestures
   - Existing evaluation relies on image-domain metrics (e.g., FID, IS), which are unsuitable for EMG time-series data
   - Fine-grained control and physiological plausibility are not guaranteed

**Core motivation**: Joint angle sequences are an interpretable and compact representation of motion. Grounding synthesis in joint kinematic constraints makes the resulting EMG signals more likely to maintain physiological plausibility under inverse kinematics constraints. Conditioning on joint angles provides fine-grained control and naturally supports zero-shot generation for unseen gestures.

## Method

### Overall Architecture

SeqEMG-GAN consists of four core modules (Figure 1):

**Input**: Joint angle sequence $S = \{s_1, s_2, \ldots, s_T\}$
**Output**: Corresponding EMG signal sequence $\hat{X} = \{\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_T\}$

The global latent context vector $\mathbf{h}_0$ is sampled via the reparameterization trick:
$$\mathbf{h}_0 = \mu(S) + \Sigma(S)^{1/2} \odot \epsilon_s, \quad \epsilon_s \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### Key Designs

#### 1. **Angle Encoder**: Motion Embedding Extraction

The angle encoder maps joint angle sequences into compact latent representations. It outputs motion embedding vectors that capture high-level semantic information about hand posture, serving as the input interface for the entire generation pipeline and providing structured motion information to subsequent modules.

#### 2. **Two-Level Context Encoder + Ang2Gist Unit**: Core Innovation

This is the most important technical contribution of the paper. The context encoder adopts a two-level recurrent structure:

- **Lower level**: A standard GRU encodes motion dynamics
$$\mathbf{i}_t, \mathbf{g}_t = \text{GRU}(\mathbf{s}_t \| \epsilon_t, \mathbf{g}_t-1)$$

- **Upper level**: The novel Ang2Gist unit refines the contextual representation
$$\mathbf{o}_t, \mathbf{h}_t = \text{Ang2Gist}(\mathbf{i}_t, \mathbf{h}_{t-1})$$

**Internal mechanism of the Ang2Gist unit**:

Similar to the GRU structure, it contains an update gate $\mathbf{z}_t$ and a reset gate $\mathbf{r}_t$. The key distinction lies in the computation of the output Gist vector:

$$\mathbf{o}_t = \text{Filter}(\mathbf{i}_t) * \mathbf{h}_t$$

where `Filter(·)` is a 1D depthwise separable convolution (kernel size 5, stride 1, padding 2) that filters the angle embedding $\mathbf{i}_t$ along the temporal dimension. Its parameters are jointly learned with the generator; each filter channel is shared across the gesture dimension to enforce temporal consistency.

**Design Motivation**: Standard GRUs struggle to maintain global contextual consistency while preserving local dynamics. Ang2Gist explicitly fuses instantaneous motion features $\mathbf{i}_t$ with the accumulated semantic state $\mathbf{h}_t$, enabling the generator to condition simultaneously on current motion and historical semantics, thereby producing physiologically plausible and temporally coherent EMG sequences.

#### 3. **EMG Generator and Multi-View Discriminator**

**Generator**: Multi-layer convolutional decoder with transposed convolutions for temporal upsampling:
$$\hat{\mathbf{x}}_t = \mathcal{G}(\mathbf{o}_t)$$

**Multi-view discriminator**: Evaluates the authenticity and semantic alignment of the generated EMG signals, conditioned on joint angles and global context:
$$m_t = \mathcal{D}(\hat{\mathbf{x}}_t, S_t | \mathbf{h}_0)$$

Unlike conventional discriminators that only assess signal-level authenticity, this discriminator jointly evaluates **amplitude continuity, temporal alignment, and waveform morphology** from three perspectives — an important design tailored to the characteristics of EMG signals.

### Loss & Training

**Joint training objective**:
$$\min_\theta \max_\psi (\alpha \mathcal{L}_{\text{GAN}} + \mathcal{L}_{\text{KL}})$$

**Adversarial loss**:
$$\mathcal{L}_{\text{GAN}} = \mathbb{E}_{(x_t, s_t)}[\log D(x_t, s_t, h_0)] + \mathbb{E}_{(\epsilon_t, s_t)}[\log(1 - D(G(\epsilon_t, s_t; \theta), s_t, h_0))]$$

**KL divergence regularization**: Constrains the global latent variable $h_0$ toward the standard normal distribution

**Training configuration**:
- Batch size 32, trained for 100 epochs
- Initial learning rate 0.002, with annealing decay during pretraining
- Learning rate reduced by 10× at epochs 21/24/27 during fine-tuning
- SGD optimizer, momentum 0.9, weight decay 5×10⁻⁴
- Loss weight parameter λ=2
- 2× NVIDIA RTX 3090 Ti

## Key Experimental Results

### Main Results

**Dataset**: Meta emg2pose, currently the largest public wrist EMG gesture dataset. 16-channel EMG (2kHz) + 26 camera-based pose annotations, 193 users, 370 hours, 29 gesture types.

**Signal similarity comparison**:

| Model | Conditional Generation | Unseen Gesture Generation | DTW ↓ | FFT MSE ↓ | EECC ↑ |
|------|:-------:|:-----------:|-------|-----------|--------|
| GAN | ✗ | ✗ | 103.43 | 19.56 | 0.719 |
| StyleTransfer | ✓ | ✗ | 98.53 | 13.53 | 0.782 |
| DCGAN | ✓ | ✗ | 93.44 | 9.68 | 0.792 |
| **Ours** | **✓** | **✓** | **91.76** | **8.76** | **0.817** |

SeqEMG-GAN achieves the best performance on all metrics and is the only method that supports unseen gesture generation.

**Classification accuracy evaluation** (6 micro-gesture classes: swipe left/right/up/down, click, double-click):

| Data Split | SVM | RF | Vemg2pose | NeuroPose | Average |
|----------|----------|----------|----------|----------|------|
| RR (real-only training) | 37.08% | 54.88% | 67.78% | 71.32% | **57.77%** |
| GR (synthetic-only training) | 35.92% | 51.32% | 65.73% | 69.87% | **55.71%** |
| MR (mixed training) | 39.88% | 58.32% | 68.95% | 74.98% | **60.53%** |

- Synthetic-only training degrades accuracy by only 2.06% (57.77% → 55.71%), demonstrating high similarity between synthetic and real data
- Mixed training improves accuracy by 2.76% (57.77% → 60.53%), confirming the data augmentation value of the synthetic data

### Ablation Study

| Configuration | DTW ↓ | FFT MSE ↓ | EECC ↑ |
|------|-------|-----------|--------|
| (a) w/o Angle Encoder | 34.02 | 3.15 | 0.326 |
| (b) w/o GRU | 50.35 | 4.50 | 0.477 |
| (c) w/o Ang2Gist | 62.20 | 5.77 | 0.564 |
| (d) w/o Discriminator | 65.47 | 6.12 | 0.613 |
| **Full Model** | **91.76** | **8.76** | **0.817** |

Note: The values in this ablation table increase monotonically across rows (the full model has the largest DTW and FFT MSE values, which appears inconsistent with lower-is-better conventions — this may reflect an issue with the original paper's metric definition or presentation). The core conclusion remains clear: every component is indispensable, and removing the angle encoder causes EECC to drop sharply from 0.817 to 0.326.

**Cross-user/cross-session generalization**:

| Setting | Sample-wise | Cross-Subject | Cross-Session | Cross User+Stage |
|------|------------|--------------|--------------|-----------------|
| Real only | 57.8±1.0 | 52.6±1.1 | 56.0±0.8 | 48.9±1.2 |
| Real+Synthetic | **60.5±0.9** | **57.1±0.9** | **59.8±0.7** | **54.3±1.0** |

Mixed data consistently improves performance across all evaluation protocols, with the largest gain observed in the cross-user + cross-stage scenario (+5.4%).

### Key Findings

1. **Joint angle conditioning is effective**: Compared to label-conditioned and unconditional GANs, joint angles provide finer-grained semantic control
2. **Synthetic data has augmentation value**: Mixed training consistently outperforms real-only training, especially in cross-user/cross-session scenarios
3. **Deep learning classifiers benefit more**: Traditional ML methods (SVM/RF) are limited on raw EMG, while deep models better leverage synthetic data
4. **Ang2Gist is a critical design**: The explicit fusion of temporal filtering and hidden states ensures semantic consistency
5. **Slide gestures exhibit smoother envelope transitions than Click gestures**, which aligns with their kinematic phases, validating the design rationale of angle-based conditioning

## Highlights & Insights

- **Novel idea of using joint angles as conditioning signals**: Continuous joint angle sequences provide finer-grained and more physically grounded control signals than discrete labels
- **Zero-shot gesture generation capability**: Naturally achieved through joint angle conditioning — any new gesture's EMG can be generated as long as its joint angle sequence is available
- **Multi-view discriminator**: Evaluates EMG fidelity from amplitude, temporal, and morphological perspectives, a beneficial design tailored to biosignal characteristics
- **Evaluation metric design**: A three-dimensional evaluation framework of FFT MSE + DTW + EECC replaces image-domain metrics, better suited to EMG time-series data
- **Broad application prospects**: Neural prosthetic hand control, AI/AR glasses, gesture-based gaming, and more

## Limitations & Future Work

1. The numerical trends in the ablation table require clearer explanation (the full model has the largest DTW yet is claimed to be optimal)
2. Only 6 micro-gesture types are tested; generation quality for more complex or diverse gesture sets remains unverified
3. Cross-user generalization still has room for improvement (57.1% cross-user accuracy remains far from practical deployment)
4. No direct comparison with state-of-the-art diffusion-based methods (e.g., PatchEMG)
5. Physiological plausibility of the generated data lacks biological validation (e.g., whether muscle activation timing conforms to physiological models)
6. The kernel size of the depthwise separable convolution is fixed at 5; the impact of different kernel sizes is not analyzed

## Related Work & Insights

- **emg2pose** (Meta, Salter 2024): Provides a large-scale high-resolution EMG+pose dataset that makes the proposed method feasible
- **PatchEMG** (Xiong 2024): Diffusion-based few-shot EMG generation, representing another important generative approach
- **DCGAN for EMG** (Chen 2022): Converts EMG to grayscale images for image-based generation, but incurs information loss
- Insight: **Modeling the inverse process from kinematics to electromyography** is an underexplored direction; joint angles as an intermediate representation bridge biomechanics and signal generation

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Joint angle-conditioned EMG generation is a novel and well-motivated approach
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers similarity analysis, classification validation, ablation, and cross-domain generalization, though the ablation table contains questionable values
- **Writing Quality**: ⭐⭐⭐ — Some formulas and tables require clearer explanation; the ablation experiment values are suspicious
- **Value**: ⭐⭐⭐⭐ — Practically valuable for addressing the scarcity of EMG data

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](../../CVPR2026/human_understanding/omg-bench_a_new_challenging_benchmark_for_skeleton-based_online_micro_hand_gestu.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[NeurIPS 2025\] CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals](../../NeurIPS2025/human_understanding/cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si.md)
- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)

</div>

<!-- RELATED:END -->
