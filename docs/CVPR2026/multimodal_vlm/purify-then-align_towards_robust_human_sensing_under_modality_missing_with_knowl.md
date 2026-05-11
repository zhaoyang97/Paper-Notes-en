---
title: >-
  [Paper Note] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher
description: >-
  [CVPR 2026][Multimodal VLM][modality missing] This paper proposes PTA (Purify-then-Align), a framework that first *purifies* a noisy multimodal teacher via a meta-learning-driven modality weighting mechanism, then *aligns* each unimodal student through diffusion-model-driven knowledge distillation, enabling unimodal encoders to maintain strong robustness under modality-missing scenarios. PTA achieves state-of-the-art performance on MM-Fi and XRF55.
tags:
  - CVPR 2026
  - Multimodal VLM
  - modality missing
  - knowledge distillation
  - meta-learning
  - diffusion alignment
  - multimodal fusion
date: 2026-05-08
content_hash: da894a0c367f23cc
---

# Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher

**Conference**: CVPR 2026
**arXiv**: [2604.05584](https://arxiv.org/abs/2604.05584)
**Code**: [https://github.com/Vongolia11/PTA](https://github.com/Vongolia11/PTA)
**Area**: Multimodal VLM / Human Sensing
**Keywords**: modality missing, knowledge distillation, meta-learning, diffusion alignment, multimodal fusion

## TL;DR

This paper proposes PTA (Purify-then-Align), a framework that first *purifies* a noisy multimodal teacher via a meta-learning-driven modality weighting mechanism, then *aligns* each unimodal student through diffusion-model-driven knowledge distillation, enabling unimodal encoders to maintain strong robustness under modality-missing scenarios. PTA achieves state-of-the-art performance on MM-Fi and XRF55.

## Background & Motivation

1. **Background**: Multimodal human sensing—integrating depth cameras, LiDAR, WiFi, and other modalities—is a foundational technology for human-computer interaction and intelligent healthcare. Multimodal fusion helps overcome the limitations of individual sensors.

2. **Limitations of Prior Work**: Two core challenges arise: (a) **Representation Gap**: Different sensors exhibit drastically different data representations (e.g., grid pixels in images vs. sparse point clouds in LiDAR), causing information loss when fused directly; (b) **Contamination Effect**: Low-quality or high-noise modalities corrupt high-quality ones during fusion, degrading overall performance.

3. **Key Challenge**: These two problems are **causally linked**—the contamination introduced by low-quality modalities fundamentally impedes the reduction of the representation gap across heterogeneous modalities. Existing approaches (generative reconstruction, shared representation learning, simple fusion, conventional knowledge distillation) each address only one aspect and ignore this causal chain.

4. **Goal**: To construct a unified framework that first resolves the cause (Contamination Effect) and then the effect (Representation Gap), such that each unimodal encoder can operate independently while incorporating cross-modal knowledge.

5. **Key Insight**: A teacher–student paradigm is adopted, where multimodal consensus serves as the teacher to guide each unimodal student. However, the teacher itself may be corrupted by noisy modalities; therefore, the teacher must first be *purified* (Purify) before it can be used to *align* the students (Align).

6. **Core Idea**: Meta-learning-based adaptive weighting addresses modality contamination on the teacher side; diffusion-model-based knowledge distillation addresses representation alignment on the student side.

## Method

### Overall Architecture

Training follows a nested-loop structure. The **outer loop (Purify)** optimizes modality weights $\mathbf{w}$ on a validation set via meta-learning, controlling each modality's contribution to the teacher. The **inner loop (Align)** fixes $\mathbf{w}$ and uses the weighted, "purified" teacher to align each unimodal student's features via diffusion distillation. At inference, each unimodal encoder operates independently without requiring other modalities.

### Key Designs

1. **Purify Stage: Meta-Learning Modality Weighting**

   - **Function**: Adaptively learns the importance weight $\mathbf{w}$ for each modality to suppress noisy or low-contribution modalities.
   - **Mechanism**: Nested optimization—the inner loop optimizes model parameters $\Theta$ on training set $\mathcal{D}_{train}$ with fixed $\mathbf{w}$ (minimizing $\mathcal{L}_{inner} = \mathcal{L}_{task} + \lambda\mathcal{L}_{DiffKD}$); the outer loop evaluates the performance of $\Theta^*(\mathbf{w})$ on validation set $\mathcal{D}_{val}$ and updates $\mathbf{w}$ via gradient $\nabla_\mathbf{w}\mathcal{L}_{outer}$. Weights are Softmax-normalized for stability. Each modality is randomly dropped with uniform probability during training to simulate real-world missing scenarios.
   - **Design Motivation**: X-Fi requires manual tuning of per-modality dropout probabilities (e.g., WiFi/Radar/RFID may require different combinations such as (0.5, 0.5, 0.8)), which becomes intractable as the number of modalities grows. Meta-learning automatically learns weights, entirely eliminating this issue.

2. **Align Stage: Diffusion-Model Knowledge Distillation**

   - **Function**: Distills knowledge from the purified multimodal teacher into each unimodal student.
   - **Mechanism**: The teacher feature is computed as $f_T = \sum_{i \in \mathcal{M}_{all}} \mathbf{w}_i f_i$ (weighted sum over all modalities). Both $f_T$ and $f_S$ are projected into a compressed latent space to obtain $z_T$ and $z_S$. A noise prediction network $\Phi_\phi$ is trained to learn the distribution of $z_T$ (standard diffusion loss $\mathcal{L}_{Diff}$), and $z_S$ is treated as a "noisy version" of $z_T$, which is refined into $\hat{z}_S$ through a reverse denoising process. The total distillation loss is $\mathcal{L}_{DiffKD} = \mathcal{L}_{Diff} + \mathcal{L}_{KD}$, where $\mathcal{L}_{KD} = \text{MSE}(\hat{z}_S, z_T)$.
   - **Design Motivation**: Conventional KL/MSE distillation struggles to bridge the large representation gap between heterogeneous modalities. The denoising process of diffusion models is naturally suited to progressively refine the information-deficient $z_S$ toward the information-rich $z_T$.

3. **Noise Adapter (Adaptive Noise Matching)**

   - **Function**: Dynamically determines the noise level of the student feature for each sample.
   - **Mechanism**: The gap between $z_S$ and $z_T$ varies across samples, and a fixed timestep $t$ cannot accommodate this one-to-many mapping. The Noise Adapter is a compact auxiliary network (1 Bottleneck + Global AvgPool + FC) that predicts a blending coefficient $\gamma \in [0,1]$ to mix the student feature with pure noise: $z_{TS} = \gamma z_S + (1-\gamma)\epsilon_T$. DDIM is then applied for 5 deterministic denoising steps from $z_{TS}$ to obtain $\hat{z}_S$.
   - **Design Motivation**: This addresses the critical problem of unknown student noise level in diffusion distillation—if $z_S$ is already close to $z_T$, minimal denoising is required; if $z_S$ is of poor quality, a noisier starting point allows the diffusion model to perform more thorough refinement.

### Loss & Training

The total inner-loop loss is $\mathcal{L}_{inner} = \mathcal{L}_{task} + \lambda\mathcal{L}_{DiffKD}$ with $\lambda=0.1$. The outer-loop loss is $\mathcal{L}_{outer} = \mathcal{L}_{task}(\Theta^*(\mathbf{w}))$. On MM-Fi, the Adam optimizer is used (model lr=5e-4, meta-learning lr=1e-2, batch=16). On XRF55, model lr=2e-4 and batch=32. All experiments are conducted on an RTX 3090.

## Key Experimental Results

### Main Results

MM-Fi human pose estimation (MPJPE mm ↓):

| Modality | Base.1 | Base.2 | X-Fi | **PTA (Ours)** | Gain |
|----------|--------|--------|------|----------------|------|
| Depth | 102.4 | 102.4 | 96.40 | **84.81** | +12.0% |
| LiDAR | 161.5 | 161.5 | 130.06 | **68.30** | +47.5% |
| WiFi | 227.1 | 227.1 | 210.12 | **182.18** | +13.3% |
| D+L | 111.7 | 108.0 | 89.41 | **64.68** | +27.7% |
| L+W | 167.1 | 206.2 | 111.15 | **74.74** | +32.8% |
| D+L+W | 130.7 | 154.6 | 87.59 | **68.86** | +21.4% |

XRF55 action recognition (accuracy % ↑):

| Modality | Baseline | X-Fi | **PTA (Ours)** | Gain |
|----------|----------|------|----------------|------|
| Radar | 82.1 | 83.9 | **90.03** | +6.13% |
| WiFi | 77.8 | 55.7 | **82.34** | +26.64% |
| RFID | 42.2 | 42.5 | **55.04** | +12.54% |
| R+W+RF | 70.6 | 89.8 | **95.87** | +6.07% |

### Ablation Study

Module ablation on MM-Fi (MPJPE mm ↓):

| Modality | Full | w/o Diff | w/o Meta |
|----------|------|----------|----------|
| Depth | **84.81** | 89.66 | 157.98 |
| LiDAR | **68.30** | 76.27 | 183.04 |
| WiFi | **182.18** | 187.92 | 236.99 |
| D+L | **64.68** | 78.12 | 148.65 |
| D+L+W | **68.86** | 76.79 | 160.34 |

### Key Findings

- **Substantial unimodal performance gains**: The core value of PTA lies in significantly enhancing unimodal encoders—the LiDAR unimodal MPJPE drops from 130.06 (X-Fi) to 68.30 (+47.5%), demonstrating that diffusion distillation effectively injects cross-modal knowledge into unimodal features.
- **The Purify stage is critical**: Removing the meta-learning weights leads to catastrophic performance collapse (D+L MPJPE surges from 64.68 to 148.65), confirming that distilling from an unpurified teacher propagates the negative influence of noisy modalities.
- **Contamination effect of WiFi**: WiFi's unimodal MPJPE of 182.18 is far worse than Depth (84.81) and LiDAR (68.30), yet PTA's L+W fusion (74.74) exhibits only marginal degradation relative to LiDAR alone (68.30), indicating that meta-learned weights successfully suppress WiFi's contamination.
- **Greater gains from homogeneous modality fusion**: On XRF55, full fusion of three RF modalities (Radar+WiFi+RFID) achieves 95.87%, as the representation gap among homogeneous radio-frequency signals is smaller and easier to align.
- X-Fi requires sensitive manual tuning of per-modality dropout rates (WiFi accuracy fluctuates from 29.1% to 55.7% across different configurations); PTA avoids this entirely through uniform dropout.

## Highlights & Insights

- **Causal problem decomposition**: This work is the first to explicitly identify the causal relationship between the Contamination Effect and the Representation Gap, and designs a two-stage solution that addresses cause before effect. This problem-framing approach is transferable to any multi-source information fusion scenario (e.g., data quality issues in multimodal large model training).
- **Diffusion models for feature alignment**: Treating student features as "noisy versions" of teacher features and refining them via denoising constitutes a new paradigm for knowledge distillation. Five-step DDIM ensures efficiency, and the Noise Adapter resolves the critical problem of unknown noise levels.
- **Meta-learning as a replacement for manual hyperparameter tuning**: Using meta-learning to automatically learn modality weights completely eliminates the tedious per-modality dropout probability tuning in multimodal systems—a technique of particular value when the number of modalities is large.

## Limitations & Future Work

- **MPJPE vs. PA-MPJPE trade-off**: PTA is strong on global localization (MPJPE) but shows partial degradation on skeletal structure after removing positional factors (PA-MPJPE), suggesting that diffusion distillation may bias the student toward predicting "mean poses" rather than precise skeletal details.
- Diffusion denoising may introduce generative artifacts under extremely low-quality modalities (WiFi/RFID), as ablation results show that removing diffusion occasionally yields better performance in certain L+W/W+RF scenarios.
- The nested optimization of meta-learning increases training complexity (inner/outer loops plus validation set evaluation).
- Validation is limited to human sensing tasks; generalizability to other multimodal fusion tasks has not been tested.
- Modality missing is simulated via random uniform dropping, without considering more realistic systematic missing patterns (e.g., prolonged sensor failure).

## Related Work & Insights

- **vs. X-Fi**: X-Fi constructs modality-invariant representations but sacrifices peak unimodal performance (LiDAR MPJPE 130.06 vs. PTA's 68.30), and requires sensitive per-modality dropout tuning. PTA achieves superior overall performance by strengthening the unimodal backbone.
- **vs. Generative methods (VAE/GAN reconstruction)**: These approaches attempt to reconstruct raw data for missing modalities, suffer from training instability, and are prone to hallucination. PTA aligns in feature space, avoiding the difficulties of raw data reconstruction.
- **vs. Conventional KD**: Standard knowledge distillation (e.g., MSE distance) struggles to bridge the large representation gap between heterogeneous modalities; diffusion distillation provides a progressive feature refinement pathway.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The causal analysis perspective and the Purify-then-Align paradigm are highly original; the combination of diffusion distillation and meta-learning is also novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two large-scale datasets, full coverage of 7 modality combinations, and in-depth ablation analysis (including per-module ablation and edge-case analysis).
- **Writing Quality**: ⭐⭐⭐⭐ The causal motivation is highly convincing, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐ Makes an important contribution to the multimodal human sensing community; the Purify-then-Align paradigm is generalizable to broader multimodal learning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](ebmc_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)
- [\[CVPR 2026\] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)

</div>

<!-- RELATED:END -->
