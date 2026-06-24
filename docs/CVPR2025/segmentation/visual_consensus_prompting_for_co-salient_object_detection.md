---
title: >-
  [Paper Note] Visual Consensus Prompting for Co-Salient Object Detection
description: >-
  [CVPR 2025][Segmentation][Co-Salient Object Detection] This paper is the first to introduce the parameter-efficient prompt learning paradigm into the Co-Salient Object Detection (CoSOD) task. It proposes Visual Consensus Prompting (VCP) by embedding the processes of consensus extraction and dispersion into learnable prompts. Under the condition of freezing the foundation model, it outperforms 13 fully fine-tuned methods with extremely few trainable parameters.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Co-Salient Object Detection"
  - "Visual Prompt Learning"
  - "Parameter-Efficient Fine-Tuning"
  - "Consensus Prompting"
  - "Transformer"
date: 2026-05-08
content_hash: 30a5a267d39cd3ba
---

# Visual Consensus Prompting for Co-Salient Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2504.14254](https://arxiv.org/abs/2504.14254)  
**Code**: [https://github.com/WJ-CV/VCP](https://github.com/WJ-CV/VCP)  
**Area**: Segmentation / Saliency Detection  
**Keywords**: Co-Salient Object Detection, Visual Prompt Learning, Parameter-Efficient Fine-Tuning, Consensus Prompting, Transformer

## TL;DR

This paper is the first to introduce the parameter-efficient prompt learning paradigm into the Co-Salient Object Detection (CoSOD) task. It proposes Visual Consensus Prompting (VCP) by embedding the processes of consensus extraction and dispersion into learnable prompts. Under the condition of freezing the foundation model, it outperforms 13 fully fine-tuned methods with extremely few trainable parameters.

## Background & Motivation

**Background**: Co-Salient Object Detection (CoSOD) aims to detect commonly occurring salient objects from a group of relevant images. Existing methods generally adopt a three-stage architecture: (1) encoding multi-scale features; (2) consensus extraction and dispersion; (3) predictive output. The training paradigm relies entirely on full parameter fine-tuning.

**Limitations of Prior Work**: (1) Architectural level—the encoding stage and consensus extraction are isolated. Encoder features are used to extract consensus, but the carefully extracted consensus cannot provide timely feedback to guide the encoding process. The encoder is only adjusted indirectly via gradient backpropagation at the very end of training, lacking effective interaction between encoding and consensus. (2) Training paradigm level—fully fine-tuning all parameters (including large pre-trained models) is parameter-inefficient and incurs high computational and storage costs. Moreover, CoSOD datasets are limited in scale, and full fine-tuning might damage the general knowledge representation in pre-trained models.

**Key Challenge**: The core of the CoSOD task lies in "consensus," but existing architectures isolate consensus extraction and feature encoding, making effective interaction between the two impossible. Meanwhile, the full fine-tuning paradigm becomes increasingly impractical as foundation models scale up.

**Goal**: Design an interactively effective and parameter-efficient CoSOD architecture that integrates consensus information into prompts to achieve deep interaction between encoding and consensus, while requiring only minimal trainable parameters.

**Key Insight**: The authors found that if the consensus extraction and dispersion processes are embedded into learnable visual prompts, the "efficient interaction between encoding and consensus" required by CoSOD naturally corresponds to the "prompt-guided frozen model" in the prompt learning paradigm. The prompts interact with features at every layer of the Transformer, realizing real-time guidance of consensus on encoding.

**Core Idea**: Construct consensus representations as task-specific visual prompts. The frozen model is "awakened" to perform the CoSOD task by mining consensus from frozen embeddings via a Consensus Prompt Generator (CPG) and injecting consensus into each Transformer block via a Consensus Prompt Disperser (CPD).

## Method

### Overall Architecture

The foundation model adopted is SegFormer (pre-trained on ImageNet), with all parameters frozen. Inputting a group of related images into SegFormer yields multi-scale embedded features of four scales. CPG mines the co-salient object representation from these frozen embeddings to generate the consensus prompt $P_{Co}$. CPD utilizes $P_{Co}$ in combination with the embedding prompt $P_{Em}$ and the handcrafted prompt $P_{Hand}$ to form the visual consensus prompt $P_{Visual}^{Co}$, which is injected into each layer of the frozen Transformer. Finally, a simplified prediction head generates the co-salient prediction maps.

### Key Designs

1. **Consensus Prompt Generator (CPG)**:

    - Function: Mines intra-group co-salient object representations from frozen embedding features to generate consensus prompts.
    - Mechanism: Operates in three steps. (a) **Saliency Estimation**: Predefines $j$ learnable saliency seeds, performs clustering through residual computation with embedding features and soft assignment probabilities, to obtain updated saliency seed representations. The updated seeds interact with embedding features to generate saliency estimation maps $M^s$: $M^s = conv[MLP(L_2(S_{seed}^{update})), P_{em}]$. (b) **Consensus Seed Selection**: Uses the saliency estimation maps to filter out non-salient regions, leaving the remaining pixel embeddings as consensus seeds $Co_{seed}$. Computes the correlation score of each seed with the intra-group mean salient feature, and selects the top-k most representative seeds $Co_{seed}^{rep}$. (c) **Consensus Prompt Generation**: Maps the representative seeds back to the original embedding space as dynamic convolution kernels, which are then enhanced by spatial attention to obtain the consensus prompt $P_{Co}$.
    - Design Motivation: Unlike previous methods that utilize extra saliency detection datasets for auxiliary training, CPG "discovers" saliency seeds in the embedding space via prototype learning. The two-step strategy of filtering out non-salient regions first before extracting consensus successfully excludes background interference.

2. **Consensus Prompt Disperser (CPD)**:

    - Function: Integrates consensus prompts into each Transformer block, generating task-specific visual consensus prompts to guide the frozen model.
    - Mechanism: CPD integrates three types of prompts—consensus prompts $P_{Co}$ (intra-group consensus information from CPG), embedding prompts $P_{Em}$ (a dimension-reduced version of frozen embedding features), and handcrafted prompts $P_{Hand}$ (traditional visual features generated via Fast Fourier Transform). The three are fused to form the visual consensus prompt $P_{Visual}^{Co}$ and are adaptively injected into different depth layers of the frozen Transformer. The number of trainable parameters is controlled via a reduction factor $r$ ($C_r = C_s / r$) to achieve parameter efficiency.
    - Design Motivation: Simply using simple trainable parameters as visual prompts (such as EVP) fails to model intra-group consensus relationships, resulting in poor performance on the CoSOD task. When consensus information is injected into the prompts, the prompts are no longer generic foreground cues but task-specific co-saliency cues.

3. **Multi-Scale Saliency Supervision and Simplified Prediction Head**:

    - Function: Ensures the accuracy of saliency estimation through multi-stage supervision and simplifies the decoding process.
    - Mechanism: Imposes CoSOD label supervision on the saliency estimation maps $\{M^s\}_{s=1}^4$ across all four stages of SegFormer, maintaining consistency between the saliency estimation and the consensus attention targets. Simplifies the original SegFormer decoder into a lightweight prediction head and integrates a classifier.
    - Design Motivation: Multi-scale supervision provides more direct gradient signals for CPG's saliency seed learning, avoiding gradient dissipation caused by backpropagation solely from the final output.

### Loss & Training

The CoSOD standard labels are used to supervise the multi-scale saliency estimation maps and the final prediction maps. All parameters of SegFormer are frozen, and only the adjustable parameters in CPG, CPD, and the prediction head are trained. The training dataset includes various combinations of COCO-9k and DUT-class.

## Key Experimental Results

### Main Results

Comparison with 13 SOTA methods on three benchmark datasets (Training set: S+D, i.e., COCO-SEG + DUT-class):

| Dataset | Metric | SCED (ACMM23) | CONDA (ECCV24) | **VCP (Ours)** | Gain |
|--------|------|--------------|----------------|----------------|------|
| CoCA | $S_m$↑ | 0.741 | 0.763 | **0.819** | +5.6% |
| CoCA | $F_m$↑ | 0.610 | 0.640 | **0.708** | +6.8% |
| CoCA | MAE↓ | 0.084 | 0.089 | **0.054** | -2.7% |
| CoSOD3k | $S_m$↑ | 0.865 | 0.862 | **0.895** | +3.0% |
| CoSal2015 | $S_m$↑ | 0.894 | 0.900 | **0.927** | +2.7% |

### Ablation Study

| Configuration | CoCA $S_m$↑ | CoCA $F_m$↑ | Description |
|------|-----------|-----------|------|
| EVP (Simple prompts only) | 0.686 | 0.510 | Simple trainable parameters cannot model consensus |
| VCP (C+D training) | 0.774 | 0.660 | Using a smaller training set |
| VCP (S+D training) | **0.819** | **0.708** | Full model, SOTA |

### Key Findings

- On the CoCA dataset, which best reflects the model's robustness, VCP significantly outperforms all full fine-tuning methods with a +6.8% $F_m$ improvement, demonstrating that parameter efficiency and performance can coexist.
- EVP (simple prompt learning for SOD) performs extremely poorly on CoSOD (CoCA $S_m$ of only 0.686), proving that consensus modeling is the core difficulty of CoSOD and simple prompts cannot solve it.
- VCP outperforms fully fine-tuned methods under both training set configurations (C+D and S+D), indicating that the method is insensitive to training data.
- The paradigm of freezing the foundation model + a small number of trainable parameters yielded better results than full fine-tuning, likely because it prevents the limited CoSOD data from disrupting the pre-trained knowledge.

## Highlights & Insights

- **Architectural Innovation of Consensus Embedding Prompts**: Perfectly combines the unique consensus concept of CoSOD with the generic prompt learning paradigm. Injecting consensus as prompts into every layer achieves deep interaction between encoding and consensus—something traditional three-stage architectures cannot do.
- **"Anomalous" Advantage of Parameter Efficiency**: Freezing the large model and tuning only a few parameters to surpass full fine-tuning is no accident. CoSOD dataset scales are limited (tens of thousands of images), where full fine-tuning might overfit and damage pre-trained representations, whereas prompt learning preserves pre-trained knowledge.
- **Prototype Learning Replacing External Models**: CPG estimates saliency through clustering of learnable saliency seeds, without requiring additional SOD datasets or pre-trained SOD models, making the design compact.
- **top-k Consensus Seed Selection**: Choosing the $k$ most representative seeds from all pixel embeddings of the entire group as the consensus representation cleverly achieves cross-image consensus discovery.

## Limitations & Future Work

- The foundation model is restricted to SegFormer, and its performance has not been verified on larger-scale models (e.g., ViT-Large, SAM).
- The choice of the number of saliency seeds $j$ and consensus seeds $k$ in CPG may need adjustment for different datasets.
- The entire group of images must be processed together during inference, and variations in group size may affect performance.
- Extensions to open-vocabulary or natural language-guided CoSOD scenarios have not been explored.

## Related Work & Insights

- **vs SCED/GCoNet+**: These methods employ traditional three-stage architectures and full fine-tuning, with $S_m$ not exceeding 0.741 on CoCA. VCP reaches 0.819 via prompt learning, demonstrating the superiority of the new paradigm.
- **vs EVP/VSCode**: EVP designs simple prompts for SOD, and VSCode introduces task/domain-specific prompts. However, neither models inter-group consensus, leading to poor performance when directly applied to CoSOD. The core innovation of VCP lies in "consensus as prompts."
- **vs VPT**: VPT pioneered visual prompt learning, but its prompts are generic learnable embeddings. VCP's prompts are consensus representations dynamically generated from data, making them more task-specific.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of embedding consensus into prompts is highly ingenious, perfectly matching CoSOD task characteristics with the prompt learning paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across 13 methods, three datasets, and six evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, but the method section contains many symbols, and the CPG workflow is somewhat complex.
- Value: ⭐⭐⭐⭐⭐ Achieves an excellent balance between parameter efficiency and performance, pointing out a new direction of prompt learning for the CoSOD community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Generalizable Co-Salient Object Detection via Mixed Content-Style Modulation](../../CVPR2026/segmentation/generalizable_co-salient_object_detection_via_mixed_content-style_modulation.md)
- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](../../ECCV2024/segmentation/self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting](rocket-1_mastering_open-world_interaction_with_visual-temporal_context_prompting.md)

</div>

<!-- RELATED:END -->
