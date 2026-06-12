---
title: >-
  [Paper Note] Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR
description: >-
  [ICML 2026][Multimodal VLM][MLLM OCR] Addressing OCR scenarios in MLLMs, the authors apply stop-gradient (Detached Skip-Links) to shallow skip branches within a multi-layer ViT→LLM fusion architecture. Simultaneously…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "MLLM OCR"
  - "multi-level feature fusion"
  - "stop-gradient"
  - "reconstruction probe"
  - "training stability"
date: 2026-05-08
content_hash: eb37a17bd733f201
---

# Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR

**Conference**: ICML 2026  
**arXiv**: [2603.20020](https://arxiv.org/abs/2603.20020)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: MLLM OCR, multi-level feature fusion, stop-gradient, reconstruction probe, training stability  

## TL;DR
Addressing OCR scenarios in MLLMs, the authors apply stop-gradient (Detached Skip-Links) to shallow skip branches within a multi-layer ViT→LLM fusion architecture. Simultaneously, they propose $R$-Probe—a reconstruction probe initialized with the first 1/4 layers of the LLM—to diagnose whether visual tokens effectively deliver fine-grained information to the language model.

## Background & Motivation

**Background**: Current MLLMs perform strongly in high-level semantic dialogues but lag significantly in "low-level perception" tasks such as OCR, dense text recognition, and small object grounding. Existing works typically view ViT (especially those trained via CLIP-style contrastive learning) as a bottleneck and suggest two paths: either adding auxiliary supervision like reconstruction loss (Fini et al., Tschannen et al.) or adopting multi-layer fusion to feed shallow features containing geometric/pixel information into the LLM (DenseConnector, DeepStack, ML, etc.).

**Limitations of Prior Work**: While multi-layer fusion is reasonable in the "forward" pass—as shallow features indeed contain stroke-level details essential for OCR—the authors find that naive fusion poses risks in the "backward" pass. Semantic gradients from the LLM's next-token loss propagate directly to shallow ViT blocks via skip branches, "scattering" the original attention maps that encode low-level structures. This leads to training instability, slow convergence, and the destruction of pre-trained spatial priors.

**Key Challenge**: Shallow features are valuable for supplementary local details in the "forward propagation," but their optimization direction conflicts with the semantic goals of the deep LLM in the "backward gradient." Forcing shallow layers to update according to semantic loss is equivalent to using the wrong optimizer for layers originally specialized in low-level patterns.

**Goal**: (i) Eliminate gradient interference while retaining the benefits of multi-layer fusion; (ii) Provide a diagnostic tool to directly determine whether "visual tokens truly deliver details to the LLM," rather than relying solely on downstream benchmarks.

**Key Insight**: Treat "feature aggregation" and "gradient propagation" as two decouplable processes—the former proceeds forward via concatenation, while the latter is controlled independently via stop-gradient.

**Core Idea**: Use $\text{sg}(\cdot)$ (stop-gradient) to cut gradients from shallow skip branches, allowing them to contribute forward features without receiving backward updates. Additionally, a lightweight decoder initialized by the first few layers of the LLM is used to reconstruct image pixels as a diagnostic signal for information arrival.

## Method

### Overall Architecture
The overall method follows a standard ViT→Adapter→LLM multimodal structure but introduces two modifications at the ViT end: (1) Applying stop-gradient to "shallow skip groups" before the multi-layer features enter the adapter; (2) Attaching a Transformer decoder + MLP initialized with the first 1/4 layers of the LLM during diagnostic or optional auxiliary training phases to reconstruct pixels from post-adapter visual tokens. Training consists of two stages: adapter pre-training (frozen ViT and LLM) → FFT/SFT (full model fine-tuning).

### Key Designs

1.  **Detached Skip-Links (Gradient-Decoupled Multi-layer Fusion)**:
    - **Function**: Merges intermediate ViT features from different depths into the main branch, ensuring only "deep groups" carry backward gradients while "shallow groups" participate only in the forward pass.
    - **Mechanism**: After selecting intermediate blocks $\{\ell_1,\dots,\ell_K\}$, they are divided by depth into $\mathbf{h}_{\text{shallow}}$ (e.g., blocks 6, 12) and $\mathbf{h}_{\text{deep}}$ (e.g., blocks 18, 23). The adapter input is $\mathbf{z}=\text{MLP}([\mathbf{h}_{\text{main}};\mathbf{h}_{\text{deep}};\text{sg}(\mathbf{h}_{\text{shallow}})])$. Theoretical analysis formulates the second moment of the full estimator gradient as $\mathbb{E}[\|\mathbf{g}_{\text{full}}\|^2]=\|\mathbf{m}+\mathbf{s}\|^2+\text{tr}(\Sigma_m+\Sigma_s+\Sigma_{ms}+\Sigma_{ms}^\top)$ and proves that in early training, the skip path satisfies variance dominance ($\text{tr}(\Sigma_s)\ge c\cdot\text{tr}(\Sigma_m)$, $c\gg 1$) and is near-orthogonal to the main path ($\cos(\mathbf{g}^{\text{main}},\mathbf{g}^{\text{skip}})\approx 0$). Thus, detaching skip gradients improves the effective Signal-to-Noise Ratio (SNR) $\eta(\mathbf{g})=\|\mathbb{E}[\mathbf{g}]\|^2/\mathbb{E}[\|\mathbf{g}\|^2]$.
    - **Design Motivation**: Visualizing [CLS] attention maps of the 4th ViT block reveals that full gradient backpropagation scatters structured attention, whereas detaching maintains pre-trained spatial consistency. This corresponds to the intuition that "shallow layers should not be rewritten by semantic loss."

2.  **$R$-Probe (LLM-Aligned Reconstruction Probe)**:
    - **Function**: Quantifies whether visual tokens from a trained MLLM retain details and whether they are friendly to the LLM's decoding style.
    - **Mechanism**: Freezes the ViT and adapter, then attaches a shallow Transformer decoder + MLP to reconstruct pixels. Crucially, the decoder is initialized with the first 1/4 layers of the target LLM (e.g., Llama-3.1-8B), ensuring its capacity is limited and its "way of seeing the world" is consistent with the LLM. High reconstruction quality indicates that visual tokens contain information and reside in a subspace easily consumable by the LLM.
    - **Design Motivation**: Traditional benchmarks conflate "vision encoding failure" with "language reasoning failure." $R$-Probe isolates the diagnostic signal at the vision-language interface. Experiments show it is sensitive to feature quality (the detached configuration reaches MSE < 0.75 in 1689 steps vs. 2158 steps for the baseline).

3.  **Context-Aware Reconstruction Sequence & Optional Auxiliary Loss**:
    - **Function**: Enables $R$-Probe to simulate real OCR inference (observing a large image + prompt → reconstructing a specific text region) rather than unconditional auto-encoding.
    - **Mechanism**: Images are tiled into $448\times 448$ blocks, and $14\times 14$ ViT patches are compressed into visual tokens via $2\times 2$ pooling. The sequence structure is $\mathcal{S}=[\mathbf{E}_{\text{context\_img}},\mathbf{E}_{\text{text}},\mathbf{E}_{\text{target\_img}}]$, with global 2D RoPE applied to preserve spatial relations.
    - **Design Motivation**: Unconditional reconstruction ignores whether visual information is usable by the LLM. Conditional reconstruction forces the probe to utilize text prompts and context, aligning with the "contextual observation → target decoding" process in OCR.

### Loss & Training
Two-stage training: adapter pre-training (5M multimodal samples, frozen ViT+LLM) → FFT+SFT (2M task samples, full model fine-tuning). The backbone defaults to Llama-3.1-8B + 300M–400M ViT. Detached Skip-Links simply adds $\text{sg}(\cdot)$ at concatenation points in the forward pass; $R$-Probe adds a shallow decoder when used as an auxiliary loss.

## Key Experimental Results

### Main Results
22 benchmarks are grouped into four categories: STEM, General, Alignment, and OCR. The table below compares the category average scores with three representative multi-layer fusion methods under identical initialization, data, and settings.

| Setting | STEM | General | Align. | OCR | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| PE baseline (No Fusion) | 63.0 | 53.2 | 72.6 | 65.2 | 61.1 |
| DenseConnector (DC) | 63.2 | 54.0 | 72.5 | 66.7 | 62.0 |
| DC + detach | 64.2 | 54.4 | 72.8 | 67.6 | 62.6 |
| ML | 63.5 | 54.1 | 72.6 | 66.9 | 62.1 |
| ML + detach | 63.1 | 54.0 | 73.2 | 68.1 | 62.5 |
| DeepStack | 63.8 | 54.5 | 73.2 | 67.6 | 62.6 |
| **Ours (PE-best)** | **64.1** | **54.6** | **73.6** | **68.3** | **63.0** |

Consistent improvements were observed across four ViT backbones (Perception Encoder, InternViT-300M, AimV2-L, SigLip2-So400M), with OCR gains usually ranging from +1.8 to +3.1 points.

### Ablation Study
Two core hyperparameters: sampling stride $S$ (density of intermediate layers) and detached layer count $D$ (from the shallowest upwards).

| Configuration | Observation | Interpretation |
|:---|:---|:---|
| Small stride ($S=3,4$) | Significantly better than sparse fusion ($S=12$) | Multi-layer fusion is effective; higher density is better |
| Detach shallow layers | Robust improvement across various $S$ | Shallow layers are the primary source of "semantic gradient poisoning" |
| Detach deeper layers | Leads to instability and degradation | Deeper layers align with LLM goals and should not be detached |
| $R$-Probe as Aux Loss | Significant OCR gain, slight drop in Reasoning | OCR data bias introduces distribution shift |

### Key Findings
- Improvements are most significant for InternViT-300M (OCR +1.9, Align +7.4), indicating the method benefits ViT models with weaker initial alignment the most.
- During early training (first ~1.3k steps), the variance of skip branch gradients $\text{tr}(\Sigma_s)$ is significantly larger than the main branch, and $\cos(\mathbf{g}^{\text{main}},\mathbf{g}^{\text{skip}})$ is near zero—validating the theoretical basis for detachment.
- The ranking of $R$-Probe reconstruction steps matches downstream OCR performance, serving as a cost-effective diagnostic tool.

## Highlights & Insights
- **Decoupling Forward Features from Backward Gradients**: While $\text{sg}(\cdot)$ is a standard technique, this work provides clear SNR theoretical explanations and empirical validation in the context of MLLM multi-layer fusion. This is transferable to any "shallow + deep joint training" architecture.
- **Using LLM Layers for Probe Decoders**: This ensures the "evaluator" and "consumer" share the same inductive bias, preventing the disconnect between standalone autoencoder metrics and actual LLM performance.
- **Zero Cost**: The core modification is a simple `.detach()`. It is engineering-wise drop-in for existing MLLM pipelines and orthogonal to architectures like DenseConnector or DeepStack.

## Limitations & Future Work
- Theoretical results (Proposition 4.3) only cover the "early training stage" without formalizing whether detachment remains optimal in late-stage convergence.
- $R$-Probe as an auxiliary loss biases the model towards OCR-style data, causing slight drops in STEM/General scores. A balanced scheduling strategy is needed.
- Scalability across much larger LLMs or non-document scenarios (e.g., scene text in the wild) remains to be verified.

## Related Work & Insights
- **vs. DenseConnector / DeepStack / ML**: These focus on "which layers to fuse and how." Detachment is an orthogonal training-side improvement.
- **vs. Perception Tokens / SeTok**: These works explicitly preserve details via new tokens or reconstruction targets which require architectural changes; $R$-Probe is a non-intrusive diagnostic tool.
- **vs. H-detach (Arpit et al., 2018)**: Shares the philosophy of selective gradient cutting to stabilize training, extending it from LSTMs to multimodal ViT-LLM fusion with SNR-based theoretical backing.

## Rating
- Novelty: ⭐⭐⭐⭐ (Applying stop-gradient to MLLM fusion with SNR analysis and pixel-reconstruction diagnosis is a novel combination)
- Experimental Thoroughness: ⭐⭐⭐⭐ (22 benchmarks, 4 ViT backbones, 5M+2M scale)
- Writing Quality: ⭐⭐⭐⭐ (Clear five-stage structure: Motivation-Theory-Diagnosis-Ablation-Comparison)
- Value: ⭐⭐⭐⭐ (Minimal engineering cost, orthogonal to existing methods, diagnostic tool is independently usable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)
- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](../../CVPR2026/multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)
- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](../../ICCV2025/multimodal_vlm/g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](../../CVPR2026/multimodal_vlm/multi-modal_image_fusion_via_intervention-stable_feature_learning.md)

</div>

<!-- RELATED:END -->
