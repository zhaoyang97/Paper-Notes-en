---
title: >-
  [Paper Note] Exploring Data-Free LoRA Transferability for Video Diffusion Models
description: >-
  [ICML 2026][Video Generation][Video Diffusion] This work is the first to analyze the weight space of full fine-tuning (FFT) and LoRA for video diffusion models (VDM)…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Video Diffusion"
  - "LoRA Transfer"
  - "SVD Singular Subspace"
  - "Spectral Routing"
  - "Data-Free"
date: 2026-05-08
content_hash: 9e88b73b64a58982
---

# Exploring Data-Free LoRA Transferability for Video Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.01929](https://arxiv.org/abs/2605.01929)  
**Code**: https://github.com/Noahwangyuchen/CASA  
**Area**: Video Diffusion Models / LoRA / Parameter-Efficient Transfer  
**Keywords**: Video Diffusion, LoRA Transfer, SVD Singular Subspace, Spectral Routing, Data-Free

## TL;DR
This work is the first to analyze the weight space of full fine-tuning (FFT) and LoRA for video diffusion models (VDM), finding that both "preserve the singular spectrum and only rotate the singular subspace," but their routing directions conflict on head clusters. Based on this, the authors propose CASA—a data-free "cluster-wise spectral arbitration" LoRA transfer method that directly migrates LoRA trained on the base Wan2.1 to distilled variants like FastWan, without any user data or retraining.

## Background & Motivation

**Background**: VDMs such as Wan2.1, HunyuanVideo, and Sora can already generate high-fidelity videos, but inference is extremely slow. The community thus favors various distillation methods—**step distillation** (e.g., Zhang 2025 compressing 50 steps to 4) and **causal distillation** (changing bidirectional attention to causal for streaming generation). These distillations are almost always implemented via full fine-tuning, resulting in a VDM ecosystem with "families of models sharing the same origin but differing in weights." Meanwhile, LoRA has become the de facto standard for style/character control uploads and sharing (with many Wan2.1 LoRA models on HuggingFace).

**Limitations of Prior Work**: Directly applying LoRA trained on the base model to distilled variants almost always fails—either style is lost or structure collapses (see Fig. 1). Retraining is costly and requires user data, which is infeasible in real scenarios (users only have LoRA weights, not the training set). Existing LoRA transfer works (X-Adapter, Trans-LoRA, LoRA-X, ProLoRA) either require data or are only validated on LLMs/image diffusion models, with almost no work on VDMs.

**Key Challenge**: Both FFT and LoRA "gently" modify the base (singular values barely change), but they take different routing paths in the shared singular subspace. When FFT has already strongly modulated a head cluster's functional pathway, injecting LoRA updates can cause "over-activation" (explosive additive effects) or "mutual cancellation" (opposite directions nullifying each other).

**Goal**: (1) Provide a "microscope" for VDM weight space to understand what FFT and LoRA actually change; (2) explain the root cause of LoRA transfer failure; (3) design a data-free transfer algorithm to recover LoRA.

**Key Insight**: Inspired by Shuttleworth 2025 (which found LoRA introduces intruder dimensions in LLMs), the authors also use SVD to analyze VDM weights, but find VDMs behave very differently from LLMs—head singular vectors in VDMs are almost unchanged, middle shows block-wise mixing, tail disperses, and LoRA does **not** introduce intruder dimensions, instead strictly preserving the spectral shape. This "spectral rigidity" motivates analyzing updates from the routing matrix perspective $\mathbf{C}=\mathbf{U}^\top\Delta\mathbf{V}$.

**Core Idea**: View LoRA transfer as "routing arbitration in the singular subspace"—directly compensate FFT drift in non-dominant regions to recover LoRA, and in dominant regions, "truncate to the maximum of the two" if above threshold to prevent over-activation, thus achieving data-free transfer.

## Method

### Overall Architecture
CASA inputs: source model $\mathbf{W}_s$, LoRA trained on the source $\Delta_{\text{lora}}=\mathbf{BA}$, and distilled target model $\mathbf{W}_t$ (thus $\Delta_{\text{fft}}=\mathbf{W}_t-\mathbf{W}_s$). Output: new LoRA $(\mathbf{B}',\mathbf{A}')$ applicable to the target model. The process is performed independently for each layer:
(1) SVD on $\mathbf{W}_s$ to obtain $\mathbf{U}_s,\mathbf{S}_s,\mathbf{V}_s$;
(2) Project both updates onto the source singular basis to obtain routing matrices $\mathbf{C}_{\text{lora}}, \mathbf{C}_{\text{fft}}$;
(3) Cluster in the top-k (covering 90% energy) subspace;
(4) Update $\mathbf{C}_{\text{casa}}$ using two rules depending on whether in dominant routing region;
(5) Back-project to weight space, perform low-rank decomposition to obtain new LoRA.

### Key Designs

1. **Routing Matrix + Cluster Construction**:

    - **Function**: Translates "weight update" into "information flow between singular directions," and clusters into stable groups based on coupling strength.
    - **Mechanism**: Define routing $\mathbf{C}=\mathbf{U}_s^\top\Delta\mathbf{V}_s$, with rows as receivers and columns as senders; large $\mathbf{C}(i,j)$ means the $j$-th sender strongly pushes to the $i$-th receiver. Select the smallest $k$ such that $\sum_{i=1}^k\sigma_i^2/\sum_i\sigma_i^2\ge 0.9$, and in the top-$k$ subspace, connect nodes with predicted rotation strength $\mathbf{R}(i,j)=|\mathbf{C}_{\text{lora}}(i,j)|/(|\sigma_i-\sigma_j|+\epsilon)$ above threshold $\tau$; connected components form clusters.
    - **Design Motivation**: Experiments show block-wise mixing in the middle spectrum (aligned with step-like singular value plateaus), consistent with Davis-Kahan perturbation theory—smaller singular value differences lead to more mixing. Normalizing routing strength by $\sigma_i-\sigma_j$ captures these "locally degenerate regions," allowing clusters to stably capture true functional units.

2. **Dominant Routing Region Identification**:

    - **Function**: Labels clusters as "dominant/non-dominant" based on FFT routing energy density, determining subsequent arbitration strategy.
    - **Mechanism**: For each cluster $\mathcal{G}_m$, compute sending/receiving energy densities $\rho_m^{\text{send}}=\frac{1}{|\mathcal{G}_m|}\sum_{i\in\mathcal{G}_m}\|\mathbf{C}_{\text{fft}}(:,i)\|_2$ and $\rho_m^{\text{recv}}$; clusters above quantile threshold $q_{\text{dom}}$ enter $\mathcal{G}_{\text{dom}}^{\text{send/recv}}$. Routing entry $(i,j)$ is marked $\mathcal{D}(i,j)=1$ if $i$ is in the receiver dominant set or $j$ in the sender dominant set.
    - **Design Motivation**: Section 3 empirically shows FFT concentrates routing energy in a few head clusters ("generation main roads"), while LoRA distributes energy evenly; only conflicts in these main roads truly degrade generation quality. **Non-dominant regions** have low risk for LoRA injection and can be safely restored.

3. **Two-Level Arbitration Rule (CASA Core)**:

    - **Function**: For each routing entry $(i,j)$, decide whether to "directly restore LoRA" or "truncate to a safe envelope."
    - **Mechanism**:
        - **Non-dominant region** $\mathcal{D}=0$: $\mathbf{C}_{\text{casa}}(i,j)=\mathbf{C}_{\text{lora}}(i,j)-\mathbf{C}_{\text{fft}}(i,j)$, so the final routing $\mathbf{C}_{\text{fft}}+\mathbf{C}_{\text{casa}}=\mathbf{C}_{\text{lora}}$, perfectly restoring LoRA.
        - **Dominant region** $\mathcal{D}=1$: Compute "over-activation risk" $\mathbf{S}(i,j)=\mathbf{E}(i,j)\cdot\text{Context}(i,j)$, where $\mathbf{E}=\max(0,\mathbf{C}_{\text{lora}}\mathbf{C}_{\text{fft}})$ is nonzero only for same-direction, and $\text{Context}$ is the cosine similarity of the cluster pair (providing collective direction evidence). If $\mathbf{S}$ exceeds quantile $q_{\text{act}}$, use $\mathbf{C}_{\text{casa}}(i,j)=\max(|\mathbf{C}_{\text{lora}}|,|\mathbf{C}_{\text{fft}}|)\cdot\text{sign}(\mathbf{C}_{\text{lora}})-\mathbf{C}_{\text{fft}}$, capping the restored strength to the maximum of the two; otherwise, keep $\mathbf{C}_{\text{lora}}$.
    - **Design Motivation**: Section 3.4 finds that LoRA-FFT directions in head clusters are sometimes strongly aligned (additive explosion), sometimes strongly opposed (mutual cancellation), with no consistent direction. Thus, "blind restoration" always fails; CASA's essence is to **cap only high-risk same-direction entries, and otherwise compensate FFT drift**—preserving generation main roads while maximally restoring LoRA style.

### Loss & Training
**No training, no data required.** CASA is a closed-form weight operation: SVD → routing projection → clustering → threshold arbitration → back-projection → truncated SVD to low-rank $(\mathbf{B}',\mathbf{A}')$. Only three quantile thresholds $\tau,q_{\text{dom}},q_{\text{act}}$ are needed, adaptively set based on cluster/routing distribution, with no tuning required.

## Key Experimental Results

### Main Results

Wan2.1-T2V-1.3B → distilled variants (FastWan-1.3B / Rolling Forcing), LoRA: Steamboat-Willie & Jinx-v2:

| LoRA | Target Model | Method | Quality Score↑ | CSD (%)↑ |
|------|-------------|--------|---------------|----------|
| Steamboat-Willie-1.3B | FastWan2.1-T2V-1.3B | Direct Reuse | 1.27 | 78.35 |
| Steamboat-Willie-1.3B | FastWan2.1-T2V-1.3B | **CASA** | **1.58** | **81.49** |
| Steamboat-Willie-1.3B | Rolling Forcing | Direct Reuse | 2.31 | 71.03 |
| Steamboat-Willie-1.3B | Rolling Forcing | **CASA** | **2.45** | — |

For 14B scale (FastWan-14B, Krea Realtime) + Film-Noir/Steamboat-Willie-14B LoRA, the trend is consistent: CASA stably outperforms Direct Reuse in both Quality and style similarity.

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Full CASA | Optimal | All three modules: routing + dominant identification + arbitration |
| w/o cluster (single entry) | Quality drops | Loses block-wise synergy |
| w/o dominant identification (uniform arbitration) | Style CSD plummets | All LoRA is truncated |
| w/o arbitration (direct restoration in dominant region) | Artifacts | Same-direction addition → over-activation |
| Higher $q_{\text{dom}}$ threshold | Stronger style but unstable | Fewer dominant regions, more aggressive |

### Key Findings
- **VDM spectral rigidity is extremely strong**: Relative change in singular values for FFT and LoRA is $\le 0.3\%$, unlike LLMs where "LoRA significantly raises leading singular value"—indicating VDM adaptation relies almost entirely on **subspace rotation** rather than energy redistribution.
- **LoRA does not introduce intruder dimension in VDMs**: Head singular vectors remain almost perfectly diagonally aligned, in sharp contrast to the "low cosine similarity anomalous directions" reported for LLMs in Shuttleworth 2025; this is a key behavioral difference for VDM-LoRA, with implications for future PEFT design.
- **FFT and LoRA have completely different routing structures**: FFT concentrates energy in a few head clusters (generation main roads), while LoRA distributes energy evenly; conflicts only occur in the intersection of head clusters, which is precisely the basis for CASA's "selective arbitration."
- **Arbitration must be at the cluster level, not single entry**: Experiments show that reducing arbitration granularity from cluster to single entry significantly degrades performance—since singular directions within a plateau are interchangeable, separate handling destroys intra-cluster synergy.

## Highlights & Insights
- The **"spectral rigidity + subspace rotation" framework** provides a clean characterization of VDM PEFT and may become a standard analysis tool for future VDM adaptation methods; especially viewing $\mathbf{C}=\mathbf{U}^\top\Delta\mathbf{V}$ as a "routing matrix" is highly portable.
- **Completely data-free** is the most practical value of this work: practitioners can directly convert a LoRA file + distilled model weights without original training data or GPU time, which is highly meaningful for open-source LoRA distribution.
- The counterintuitive finding that **both same-direction and opposite-direction are conflicts** is interesting—intuitively, only opposite directions are bad, but in fact, same-direction addition in head clusters is equally fatal (generation main roads are overdriven); CASA logically separates "risk" into magnitude × direction.

## Limitations & Future Work
- Only two types of distillation (step / causal) and two Wan scales (1.3B/14B) are tested; not evaluated on large models like HunyuanVideo, CogVideoX, Sora-style; whether spectral rigidity holds for different backbones (DiT vs U-Net) is unknown.
- Evaluation metrics are limited to VideoAlign's Quality Score + CSD style similarity, lacking finer-grained motion consistency / temporal coherence assessments; not sensitive to failure modes where "style is preserved but motion collapses."
- Although the three quantile thresholds require no tuning, they may drift across model scales; the paper does not provide robustness curves for cross-scale scenarios.
- Assumes LoRA is low-rank BA structure; compatibility with non-pure low-rank variants like DoRA / LoRA-FA / Adapter is not discussed.

## Related Work & Insights
- **vs ProLoRA**: ProLoRA is also data-free, projecting LoRA into the target weight subspace; but ProLoRA does not consider routing energy distribution and treats all singular directions equally. CASA introduces differentiated handling of dominant/non-dominant regions, showing clear superiority on VDMs.
- **vs LoRA-X**: LoRA-X constrains updates to selected singular directions during "training"; CASA reshapes routing at "conversion" time, requiring no retraining, making them complementary.
- **vs Shuttleworth 2025 (LLM intruder dim)**: This work is a counterexample—LoRA does **not** introduce intruder dim in VDMs, indicating that whether LoRA introduces new directions depends on modality/architecture, not an inherent property of LoRA itself.

## Rating
- Novelty: ⭐⭐⭐⭐ First complete spectral + routing analysis for VDM, unique CASA arbitration rule design
- Experimental Thoroughness: ⭐⭐⭐⭐ Two scales × two distillation types × multiple LoRA are convincing, but lacks finer-grained evaluation
- Writing Quality: ⭐⭐⭐⭐ Analysis progresses logically (spectral rigidity → subspace → routing → interference), with clear reasoning
- Value: ⭐⭐⭐⭐⭐ Data-free is truly important, directly benefiting industrial deployment and open-source community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[NeurIPS 2025\] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](../../NeurIPS2025/video_generation/s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)

</div>

<!-- RELATED:END -->
