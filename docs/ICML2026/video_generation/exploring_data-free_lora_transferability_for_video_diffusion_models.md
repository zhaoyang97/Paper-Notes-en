---
title: >-
  [Paper Note] Exploring Data-Free LoRA Transferability for Video Diffusion Models
description: >-
  [ICML 2026][Video Generation][Video Diffusion] This paper provides the first weight-space analysis of full fine-tuning (FFT) and LoRA for Video Diffusion Models (VDMs). It discovers that both "preserve the singular spect…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Video Diffusion"
  - "LoRA Transfer"
  - "SVD Singular Subspace"
  - "Spectral Routing"
  - "Data-Free"
date: 2026-05-08
content_hash: eb35c2214ad3fba8
---

# Exploring Data-Free LoRA Transferability for Video Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.01929](https://arxiv.org/abs/2605.01929)  
**Code**: https://github.com/Noahwangyuchen/CASA  
**Area**: Video Diffusion Models / LoRA / Parameter-Efficient Transfer  
**Keywords**: Video Diffusion, LoRA Transfer, SVD Singular Subspace, Spectral Routing, Data-Free

## TL;DR
This paper provides the first weight-space analysis of full fine-tuning (FFT) and LoRA for Video Diffusion Models (VDMs). It discovers that both "preserve the singular spectrum and only rotate singular subspaces," yet they exhibit conflicting routing directions on head clusters. Consequently, the authors propose CASA—a data-free "spectral arbitration by cluster" LoRA transfer method. This approach allows LoRAs trained on the Wan2.1 base model to be directly migrated to distilled variants, such as FastWan, without requiring user data or retraining.

## Background & Motivation

**Background**: VDMs like Wan2.1, HunyuanVideo, and Sora have achieved high-fidelity video generation but suffer from extremely slow inference. The community has increasingly adopted various forms of distillation—such as **step distillation** (e.g., Zhang 2025, compressing 50 steps to 4) and **causal distillation** (changing bidirectional attention to causal for streaming generation). These distillations are almost exclusively implemented via full fine-tuning, leading to a VDM ecosystem comprising a family of models with "common roots but distinct weights." Meanwhile, LoRA has become the de facto standard for sharing style and character control (e.g., massive amounts of Wan2.1 LoRAs on HuggingFace).

**Limitations of Prior Work**: Directly applying LoRAs trained on the base model to distilled variants **almost inevitably fails**, resulting in either style loss or structural collapse (Figure 1). Retraining is costly and requires original user data, which is impractical for real scenarios where users only possess LoRA weights. Existing LoRA transfer works (X-Adapter, Trans-LoRA, LoRA-X, ProLoRA) either require data or are only validated on LLMs/Image Diffusion Models, leaving a gap in the VDM domain.

**Key Challenge**: While both FFT and LoRA "gently" modify the base model (keeping singular values nearly unchanged), they **follow different routing paths** within the shared singular subspaces. When FFT has already strongly modulated a specific head cluster's functional path, inserting LoRA updates leads to either "over-activation" (exploding due to same-direction superposition) or "mutual cancellation" (failing due to opposite-direction interference).

**Goal**: (1) Provide a "microscope" for VDM weight space to understand what FFT and LoRA actually modify; (2) Explain the root cause of direct LoRA transfer failure; (3) Design a data-free transfer algorithm to rescue the LoRA.

**Key Insight**: Inspired by Shuttleworth 2025 (identifying "intruder dimensions" in LLM LoRAs), the authors utilize SVD to analyze VDM weights. They find VDM behavior is fundamentally different from LLMs: singular vectors of heads remain nearly unchanged, middle layers show block-wise mixing, and tails are dispersed. Moreover, LoRA does **not** introduce intruder dimensions but strictly maintains the spectral shape. This "spectral rigidity" inspires an analysis from the perspective of a routing matrix $\mathbf{C}=\mathbf{U}^\top\Delta\mathbf{V}$.

**Core Idea**: Treat LoRA transfer as "routing arbitration within singular subspaces." For non-dominant regions, compensate for FFT drift to restore LoRA; for dominant regions, apply a "cap at the maximum of both when exceeding a threshold" to prevent over-activation, achieving data-free transfer.

## Method

### Overall Architecture
Input to CASA: Source model $\mathbf{W}_s$, LoRA trained on source $\Delta_{\text{lora}}=\mathbf{BA}$, and distilled target model $\mathbf{W}_t$ (from which $\Delta_{\text{fft}}=\mathbf{W}_t-\mathbf{W}_s$ is derived). Output: A new LoRA $(\mathbf{B}',\mathbf{A}')$ compatible with the target model. The process is performed independently per layer:
(1) Compute SVD of $\mathbf{W}_s$ to obtain $\mathbf{U}_s,\mathbf{S}_s,\mathbf{V}_s$;
(2) Project both updates onto the source singular basis to obtain routing matrices $\mathbf{C}_{\text{lora}}, \mathbf{C}_{\text{fft}}$;
(3) Construct clusters within the top-k (covering 90% energy) subspace;
(4) Update $\mathbf{C}_{\text{casa}}$ using two different rules based on whether the routing falls into the "dominant region";
(5) Project back to the weight space and apply low-rank decomposition to obtain the new LoRA.

### Key Designs

1.  **Routing Matrix + Cluster Construction**:
    - **Function**: Translates "weight updates" into "information flow between singular directions" and clusters them into stable units based on coupling strength.
    - **Mechanism**: Routing is defined as $\mathbf{C}=\mathbf{U}_s^\top\Delta\mathbf{V}_s$, where the row is the receiver and the column is the sender. A large $\mathbf{C}(i,j)$ indicates the $j$-th sender strongly pushes information to the $i$-th receiver. In the top-$k$ subspace, edges are connected based on predicted rotation strength $\mathbf{R}(i,j)=|\mathbf{C}_{\text{lora}}(i,j)|/(|\sigma_i-\sigma_j|+\epsilon)$ exceeding a threshold $\tau$. Connected components form clusters.
    - **Design Motivation**: Experiments reveal block-wise mixing in the middle spectrum (aligned with step-like singular value plateaus), consistent with Davis-Kahan perturbation theory—the smaller the singular value gap, the easier the mixing. Normalizing routing strength by $\sigma_i-\sigma_j$ captures these "locally degenerate regions," ensuring clusters capture true functional units.

2.  **Dominant Routing Region Identification**:
    - **Function**: Labels clusters as "dominant" or "non-dominant" based on FFT routing energy density to determine subsequent arbitration strategies.
    - **Mechanism**: For each cluster $\mathcal{G}_m$, the send/receive energy density is calculated as $\rho_m^{\text{send}}=\frac{1}{|\mathcal{G}_m|}\sum_{i\in\mathcal{G}_m}\|\mathbf{C}_{\text{fft}}(:,i)\|_2$ and $\rho_m^{\text{recv}}$. Clusters exceeding the quantile threshold $q_{\text{dom}}$ enter the dominant sets $\mathcal{G}_{\text{dom}}^{\text{send/recv}}$. A routing entry $(i,j)$ is marked $\mathcal{D}(i,j)=1$ if $i$ is in the receiver dominant set or $j$ is in the sender dominant set.
    - **Design Motivation**: Empirical findings show that FFT concentrates routing energy in a few head clusters ("generation highways"), while LoRA is uniformly distributed. Conflicts in these highways jeopardize generation quality; LoRA injection in **non-dominant regions** poses little risk and can be safely restored.

3.  **Two-level Arbitration Rules (CASA Core)**:
    - **Function**: Determines for each routing $(i,j)$ whether to "fully restore LoRA" or "truncate to a safe envelope."
    - **Mechanism**:
        - **Non-dominant region** $\mathcal{D}=0$: $\mathbf{C}_{\text{casa}}(i,j)=\mathbf{C}_{\text{lora}}(i,j)-\mathbf{C}_{\text{fft}}(i,j)$, such that the final routing $\mathbf{C}_{\text{fft}}+\mathbf{C}_{\text{casa}}=\mathbf{C}_{\text{lora}}$, perfectly recovering the LoRA.
        - **Dominant region** $\mathcal{D}=1$: Calculate "over-activation risk" $\mathbf{S}(i,j)=\mathbf{E}(i,j)\cdot\text{Context}(i,j)$, where $\mathbf{E}=\max(0,\mathbf{C}_{\text{lora}}\mathbf{C}_{\text{fft}})$ is non-zero only for same-direction updates, and $\text{Context}$ is the cosine similarity of the cluster pair. If $\mathbf{S}$ exceeds quantile $q_{\text{act}}$, use $\mathbf{C}_{\text{casa}}(i,j)=\max(|\mathbf{C}_{\text{lora}}|,|\mathbf{C}_{\text{fft}}|)\cdot\text{sign}(\mathbf{C}_{\text{lora}})-\mathbf{C}_{\text{fft}}$ to cap the recovered strength; otherwise, maintain $\mathbf{C}_{\text{lora}}$.
    - **Design Motivation**: Section 3.4 reveals that LoRA-FFT directions in head clusters lack a uniform orientation. "Indiscriminate restoration" is bound to fail. The essence of CASA is **capping only at high-risk same-direction positions while compensating for FFT drift elsewhere**—preserving the generation highway while maximizing LoRA style recovery.

### Loss & Training
**No training, no data**. CASA is a closed-form weight operation: SVD $\rightarrow$ Routing Projection $\rightarrow$ Clustering $\rightarrow$ Threshold Arbitration $\rightarrow$ Back-projection $\rightarrow$ Truncated SVD back to low-rank $(\mathbf{B}',\mathbf{A}')$. Hyperparameters consist only of three quantile thresholds ($\tau,q_{\text{dom}},q_{\text{act}}$), which are adaptively selected based on cluster/routing distributions.

## Key Experimental Results

### Main Results

Wan2.1-T2V-1.3B $\to$ Distilled Variants (FastWan-1.3B / Rolling Forcing), LoRA: Steamboat-Willie & Jinx-v2:

| LoRA | Target Model | Method | Quality Score↑ | CSD (%)↑ |
|------|-------------|------|---------------|----------|
| Steamboat-Willie-1.3B | FastWan2.1-T2V-1.3B | Direct Reuse | 1.27 | 78.35 |
| Steamboat-Willie-1.3B | FastWan2.1-T2V-1.3B | **CASA** | **1.58** | **81.49** |
| Steamboat-Willie-1.3B | Rolling Forcing | Direct Reuse | 2.31 | 71.03 |
| Steamboat-Willie-1.3B | Rolling Forcing | **CASA** | **2.45** | — |

Results on 14B scale (FastWan-14B, Krea Realtime) + Film-Noir/Steamboat-Willie-14B LoRA show consistent trends, with CASA consistently outperforming Direct Reuse in Quality and Style Similarity.

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full CASA | Optimal | Routing + Dominance Identification + Arbitration modules all present |
| w/o cluster (per-entry handling) | Quality drops | Loss of block-wise synergy |
| w/o Dominance ID (unified arb.) | Style CSD plunges | Excessive truncation of LoRA |
| w/o Arbitration (direct restore in dominant) | Artifacts appear | Same-direction superposition $\rightarrow$ Over-activation |
| Higher threshold $q_{\text{dom}}$ | Stronger style, prone to collapse | Aggressive reduction of the dominant region |

### Key Findings
- **Strong Spectral Rigidity in VDM**: Relative changes in singular values for both FFT and LoRA are $\le 0.3\%$. Unlike LLMs where LoRA significantly lifts leading singular values, VDM adaptation relies almost purely on **subspace rotation** rather than energy redistribution.
- **LoRA does not introduce intruder dimensions in VDM**: Head singular vectors maintain near-perfect diagonal alignment, contrasting sharply with the "abnormal directions" reported in LLMs. This is a crucial behavioral difference for future PEFT designs.
- **Distinct Routing Structures**: FFT concentrates energy on few head clusters, while LoRA spreads energy evenly. Conflicts occur only at the intersection of head clusters, which justifies CASA's "selective arbitration."
- **Cluster-level Arbitration is Necessary**: Reducing arbitration granularity from clusters to single entries significantly degrades performance, as singular directions within a plateau are interchangeable; separate treatment disrupts cluster synergy.

## Highlights & Insights
- The **"Spectral Rigidity + Subspace Rotation" framework** provides a clean characterization of VDM PEFT and may become a standard analytical tool. Specifically, the "routing matrix" perspective ($\mathbf{C}=\mathbf{U}^\top\Delta\mathbf{V}$) is highly portable.
- **Complete Data-Free Operation** is the most practical contribution. It allows for the direct conversion of LoRAs without training data or GPU time, which is highly significant for open-source community distribution.
- The discovery that **both same-direction and opposite-direction updates cause conflicts** is counter-intuitive and insightful—same-direction superposition in head clusters is equally fatal (pushing the highway too far). CASA's logic of magnitude × direction layers is robust.

## Limitations & Future Work
- Validated only on two types of distillation (step/causal) and two Wan scales (1.3B/14B). Performance on other models (HunyuanVideo, CogVideoX) is untested, and spectral rigidity in different backbones (DiT vs. U-Net) remains unknown.
- Evaluation metrics are limited to Quality Score and CSD style similarity, lacking fine-grained metrics for motion consistency or temporal coherence.
- While thresholds are "non-tunable" quantiles, they may drift across model scales.
- Compatibility with non-pure low-rank variants (DoRA, LoRA-FA, Adapter) is not discussed.

## Related Work & Insights
- **vs. ProLoRA**: ProLoRA is also data-free and projects LoRA into target weight subspaces but ignores routing energy distribution. CASA’s differentiated treatment of dominant/non-dominant regions is superior for VDMs.
- **vs. LoRA-X**: LoRA-X konstrains updates during training; CASA reshapes routing at conversion time, offering a complementary positioning.
- **vs. Shuttleworth 2025 (LLM intruder dim)**: This work serves as a counter-example—LoRA does **not** introduce intruder dimensions in VDMs, suggesting such behavior depends on modality/architecture rather than being an inherent property of LoRA.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive spectral + routing analysis for VDMs; unique arbitration design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Convincing across scales and distillation types, though lacks fine-grained evaluation.
- Writing Quality: ⭐⭐⭐⭐ Logical progression (Rigidity $\to$ Subspace $\to$ Routing $\to$ Interference).
- Value: ⭐⭐⭐⭐⭐ Data-free utility directly benefits industrial deployment and the open-source community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[ICML 2026\] Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models](where_concept_erasure_should_occur_concept-layer_alignment_in_text-to-video_diff.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ECCV 2024\] Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](../../ECCV2024/video_generation/exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)
- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](../../CVPR2026/video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)

</div>

<!-- RELATED:END -->
