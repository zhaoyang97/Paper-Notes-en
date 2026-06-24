---
title: >-
  [Paper Note] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs
description: >-
  [ECCV 2024][Multimodal VLM][Multimodal Large Language Models (MLLMs)] Proposes X-Former, a lightweight Transformer module that fuses complementary visual features from CLIP-ViT (contrastive learning) and MAE-ViT (masked image modeling) through a dual cross-attention mechanism. It significantly outperforms BLIP-2 on fine-grained visual understanding tasks using only 1/10 of the training data.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multimodal Large Language Models (MLLMs)"
  - "Contrastive Learning"
  - "Masked Image Modeling"
  - "Visual Representation"
  - "Q-Former"
date: 2026-05-08
content_hash: fbd9f85a57c13f5a
---

# X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs

**Conference**: ECCV 2024  
**arXiv**: [2407.13851](https://arxiv.org/abs/2407.13851)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models (MLLMs), Contrastive Learning, Masked Image Modeling, Visual Representation, Q-Former

## TL;DR

Proposes X-Former, a lightweight Transformer module that fuses complementary visual features from CLIP-ViT (contrastive learning) and MAE-ViT (masked image modeling) through a dual cross-attention mechanism. It significantly outperforms BLIP-2 on fine-grained visual understanding tasks using only 1/10 of the training data.

## Background & Motivation

Current Multimodal Large Language Models (MLLMs) commonly adopt CLIP-ViT as the visual encoder. However, CLIP is trained on contrastive learning and primarily focuses on low-frequency signals and global patterns, showing obvious limitations in capturing fine-grained visual details such as object orientation, structural details, spatial relations, and multi-instance recognition. MAE-ViT, trained via masked image modeling, excels at understanding local and high-frequency visual features, but simply concatenating the two representation spaces does not yield effective results.

The authors validated two key findings through experiments:

**Simple concatenation of CLIP and MAE features performs on par with BLIP-2**, indicating a huge discrepancy between the information provided by the two encoders, making it difficult for the model to learn global and local information simultaneously.

**Early cross-attention yields slight improvements but adds a large number of parameters (75M)**, and even causes a performance drop on the GQA dataset.

**Key Challenge**: How to effectively fuse Contrastive Learning (CL) and Masked Image Modeling (MIM) visual representations without a substantial parameter increase, enabling the LLM to understand both global semantics and local details simultaneously.
**Key Insight**: X-Former addresses this by designing a dual cross-attention interaction mechanism, aligning the MAE features with global semantics guided by a reconstruction loss.

## Method

### Overall Architecture

X-Former adopts a two-stage training paradigm:
- **Stage 1 (Pre-training)**: Learns global + local visual representations from two frozen visual encoders (CLIP-ViT and MAE-ViT).
- **Stage 2 (LLM Alignment)**: Aligns the output of X-Former with a frozen LLM.

The framework consists of four components: a frozen CLIP-ViT encoder, a frozen MAE-ViT encoder, a frozen MAE decoder, and a trainable X-Former module.

### Key Designs

1. **Q-Former Base Module**:

    - **Function**: Extracts global semantic visual features from CLIP-ViT using learnable query vectors.
    - **Mechanism**: Query vectors interact with each other via self-attention layers and interact with frozen image features via cross-attention layers.
    - **Design Motivation**: Inherits the successful architecture of BLIP-2, which, however, only captures global representations.

2. **Dual Cross-Attention Module (Core of X-Former)**:

    - **Function**: Integrates MAE's local detailed features on top of the Q-Former output.
    - **Mechanism**: Performs cross-attention in two steps:
        - **Step 1**: Use MAE features $M$ as Query, and Q-Former output $Z_q$ as Key/Value $\rightarrow$ generates semantically enhanced MAE features $M'$ (injecting global semantic information into local MAE features).
        - **Step 2**: Use the enhanced MAE features $M'$ as Key/Value, and $Z_q$ as Query $\rightarrow$ generates the final enhanced query $Z'$ (injecting local detailed information into global queries).
    - **Design Motivation**: Align first, then fuse. Directly fusing two highly discrepant representations is ineffective; hence, bridging intermediates are used to progressively align the two representation spaces.

3. **MAE Masking and Reconstruction**:

    - **Function**: Applies random masking (50% ratio) on the input image, and feeds the enhanced MAE features $M'$ into a frozen MAE decoder to reconstruct the masked regions.
    - **Mechanism**: The reconstruction loss forces the network to extract meaningful local information from MAE instead of learning shortcuts.
    - **Design Motivation**: Without the reconstruction target, the network fails to utilize MAE features effectively (as validated by a drastic performance drop in ablation studies).

### Loss & Training

**Stage 1 Pre-training (4 loss functions)**:

| Loss Function | Role | Attention Mask |
|---------|------|-----------|
| ITC (Image-Text Contrastive) | Maximizes image-text similarity of positive pairs | Unimodal self-attention mask to prevent query-text interaction |
| ITM (Image-Text Matching) | Binary classification of whether an image-text pair matches | Bidirectional self-attention mask allowing queries and text to attend to each other |
| ITG (Image-grounded Text Generation) | Generates corresponding text conditioned on the image | Multimodal causal self-attention mask |
| Reconstruction | Reconstructs the masked image regions of MAE | Applied on the enhanced MAE features $M'$ |

**Stage 2 LLM Alignment**:
- Maps the X-Former output $Z'$ to the LLM embedding space via a fully connected layer.
- Trains using only the language modeling loss, freezing all visual encoders and the LLM.
- Reconstruction loss is not used (ablation shows adding reconstruction loss in Stage 2 is actually counterproductive).

**Training Details**:
- Stage 1 is trained for 9 epochs, and Stage 2 is trained for 1 epoch.
- CLIP-ViT uses ViT-G from EVA-CLIP, while MAE uses ViT-H.
- LLM uses the OPT model (available in two scales: 2.7B and 6.7B).
- Training data consists of only 14M image-text pairs (compared to 129M used in BLIP-2), which is about 1/10 of BLIP-2's volume.
- Training time increases by ~10%, and GPU memory usage increases by ~4.7%.

## Key Experimental Results

### Main Results: Zero-Shot VQA

| Dataset | Metric | X-Former (OPT 6.7B) | BLIP-2 (OPT 6.7B) | Gain |
|--------|------|---------------------|-------------------|------|
| VQAv2 | Overall Acc | **55.0** | 52.4 | +2.6% |
| VQAv2 | Number Acc | **37.8** | 30.8 | +7.0% |
| GQA | Acc | **34.9** | 33.1 | +1.8% |
| OKVQA | Acc | **34.2** | 31.5 | +2.7% |

### Fine-Grained Visual Perception

| Task | Dataset | X-Former | BLIP-2* (129M) | BLIP-2 (14M) | Note |
|------|--------|---------|---------------|-------------|------|
| Object Counting (OC) | COCO | **39.64** | 34.3 | 25.88 | Outperforms BLIP-2 trained on 129M data |
| Object Counting (OC) | VCR | **27.24** | 18.9 | 21.12 | Significantly outperforms |
| Multi-class Identification (MCI) | COCO | **69.44** | 69.44 | 61.5 | Comparable performance |
| Multi-class Identification (MCI) | VCR | 69.28 | **74.16** | 65.3 | Slightly lower |

### Ablation Study

| Configuration | VQAv2 | GQA | OKVQA | Note |
|------|-------|-----|-------|------|
| X-Former (Full) | **55.0** | **34.9** | **34.2** | Best |
| W/o Reconstruction Loss (Absent in both Stage 1 & 2) | 33.1 | 25.4 | 12.1 | Catastrophic performance drop |
| Stage 1 with + Stage 2 with Reconstruction | 52.4 | 32.2 | 29.2 | Reconstruction is unnecessary for Stage 2 |
| Replacing MAE with CLIP L26 Layer | 53.7 | 32.6 | 31.2 | MAE outperforms intermediate CLIP layers |
| Simple Concatenation (110M params) | 52.3 | 32.1 | 31.9 | Ineffective |
| Early Cross-Attention (183M params) | 53.8 | 32.7 | 31.5 | Over-parameterized but worse performance |

### Key Findings
- Improvements are most significant in object counting tasks (COCO +13%, VCR +6.1%), proving the enhanced understanding of local details.
- Achieves superior performance over the official BLIP-2 checkpoint (trained on 129M data) using only 1/10 of the data size.
- Reconstruction loss is the absolute key—without it, the network fails to effectively utilize MAE features.
- Visual computing overhead increases only slightly: training time +10%, GPU memory +4.7%, and inference latency ~890ms vs. ~680ms.

## Highlights & Insights
- **Elegant Design for Complementary Fusion**: The dual cross-attention "align-first-then-fuse" strategy is more effective and parameter-efficient than simple concatenation or early interactions.
- **Dual Role of Reconstruction Loss**: It acts as both an alignment signal for MAE features and a mechanism to prevent the network from taking shortcuts and ignoring local information.
- **Incredible Data Efficiency**: Surpassing BLIP-2 trained on 129M data with only 14M data demonstrates that high-quality visual representations are more crucial than sheer data volume.
- **Plug-and-play**: X-Former can replace the original Q-Former in other MLLM frameworks.

## Limitations & Future Work
- Experimented only on OPT, without evaluating on stronger LLMs (e.g., LLaMA, Vicuna).
- Inference latency increases by about 30% (~890ms vs. ~680ms) due to the incorporation of the additional MAE encoder.
- Performance on the MCI task in VCR is slightly lower than BLIP-2, indicating minor sacrifice in global understanding.
- Lacks a fair comparison against instruction-tuning methods such as LLaVA.
- Future work can explore other MIM variants (e.g., BEiT, SimMIM) to replace MAE.

## Related Work & Insights
- **BLIP-2**: Direct baseline and structural foundation, upon whose Q-Former design the X-Former is extended.
- **MMVP**: Similarly utilizes self-supervised encoders but relies on instruction tuning.
- **Joint Training of CL + MIM**: Previously only explored in vision pre-training, without being applied to vision-language (VL) understanding.
- **Insight**: Visual encoders with different pre-training objectives do encode complementary information; the key lies in the design of the fusion mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of using dual cross-attention to fuse CL and MIM is clear and effective, though the overall framework is built heavily upon BLIP-2.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset validation and comprehensive ablation studies are provided, but comparisons with a wider range of MLLM methods are missing.
- Writing Quality: ⭐⭐⭐⭐ Complete logical chain of motivation-experiment-analysis, presenting step-by-step progressions from failed attempts to the proposed solution.
- Value: ⭐⭐⭐⭐ Highly data-efficient and plug-and-play, holding practical reference value for improving MLLM visual representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation](../../CVPR2026/multimodal_vlm/guiding_diffusion-based_reconstruction_with_contrastive_signals_for_balanced_vis.md)
- [\[ECCV 2024\] CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learning_with_augmented_pr.md)
- [\[ICCV 2025\] DisenQ: Disentangling Q-Former for Activity-Biometrics](../../ICCV2025/multimodal_vlm/disenq_disentangling_q-former_for_activity-biometrics.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)

</div>

<!-- RELATED:END -->
