---
title: >-
  [Paper Note] DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment
description: >-
  [ECCV 2024][Image Generation][EEG-to-image generation] This paper proposes DreamDiffusion, which leverages temporal masked signal modeling for large-scale pre-training of an EEG encoder to learn robust brainwave representations. It then aligns the EEG-text-image space using additional supervision from a CLIP image encoder, and finally utilizes a pre-trained Stable Diffusion model to generate high-quality images directly from EEG signals, achieving portable and low-cost "thoug…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "EEG-to-image generation"
  - "brain signal decoding"
  - "temporal masked signal modeling"
  - "CLIP alignment"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: 65d68ac381567a8e
---

# DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment

**Conference**: ECCV 2024  
**arXiv**: [2306.16934](https://arxiv.org/abs/2306.16934)  
**Code**: [https://github.com/bbaaii/DreamDiffusion](https://github.com/bbaaii/DreamDiffusion)  
**Area**: Diffusion Models / Brain-Computer Interface  
**Keywords**: EEG-to-image generation, brain signal decoding, temporal masked signal modeling, CLIP alignment, Stable Diffusion

## TL;DR

This paper proposes DreamDiffusion, which leverages temporal masked signal modeling for large-scale pre-training of an EEG encoder to learn robust brainwave representations. It then aligns the EEG-text-image space using additional supervision from a CLIP image encoder, and finally utilizes a pre-trained Stable Diffusion model to generate high-quality images directly from EEG signals, achieving portable and low-cost "thoughts-to-image" generation.

## Background & Motivation

**Background**: Significant breakthroughs have been made in the text-to-image generation field (e.g., DALL-E, Stable Diffusion). Recent work has also begun to explore direct image generation from brain signals, such as MinD-Vis, which leverages fMRI signals to generate impressive images. Such "thought-to-image" research holds vast prospects in neuroscience and human-computer interaction (HCI).

**Limitations of Prior Work**: (1) Existing fMRI-based methods, while effective, rely on expensive, non-portable fMRI equipment that requires professional operation, greatly limiting practical applications; (2) Although EEG is portable and low-cost, it is characterized by high noise, limited information content, and significant individual differences, making direct image generation from EEG highly challenging; (3) Paired EEG-image data is extremely scarce, making it difficult to train end-to-end conditional generative models; (4) There is a massive discrepancy between the feature space of EEG signals and the already-aligned text-image space of Stable Diffusion.

**Key Challenge**: To achieve direct image generation from EEG, two fundamental issues must be resolved: how to extract effective semantic representations from noisy and scarce EEG signals, and how to align EEG representations with the text-image space of pre-trained diffusion models.

**Goal**: (1) How to leverage large-scale unlabeled EEG data (without paired images) to learn robust EEG representations? (2) How to align the EEG space with the CLIP text-image space using minimal paired EEG-image data?

**Key Insight**: The authors observe that EEG signals possess strong temporal characteristics, unlike the spatial characteristics of fMRI. Therefore, instead of masking spatial information as in MAE and MinD-Vis, this paper proposes masked modeling in the temporal domain. Furthermore, the CLIP image encoder is leveraged as a bridge: since CLIP's text and image spaces are already aligned, pulling EEG representations closer to the CLIP image space is equivalent to pulling them closer to the text space, thereby making them compatible with Stable Diffusion.

**Core Idea**: Learn robust representations from large-scale noisy EEG data using temporal masked pre-training, and then bridge the EEG space with SD's text-image space using the CLIP image encoder to achieve direct generation of high-quality images from EEG.

## Method

### Overall Architecture

The pipeline of DreamDiffusion consists of three stages: (1) Temporal masked signal pre-training stage—using a large amount of unpaired EEG data (approximately 120,000 samples from 400+ subjects on the MOABB platform) to pre-train the EEG encoder through self-supervised masked learning; (2) Stable Diffusion fine-tuning stage—using a small amount of paired EEG-image data (ImageNet-EEG dataset) to fine-tune the EEG encoder and the cross-attention layers of Stable Diffusion; (3) CLIP alignment stage—utilizing additional supervision from a CLIP image encoder to align the EEG embeddings with CLIP image embeddings for better compatibility with SD. During inference, only the EEG signal is input to obtain conditional embeddings through the encoder, which then steer SD to generate corresponding images.

### Key Designs

1. **Temporal Masked Signal Modeling**:

    - **Function**: Learn generalizable and robust EEG representations from large-scale unlabeled EEG data.
    - **Mechanism**: EEG signals are 2D data of 128 channels $\times$ time steps. Unlike MAE, which masks spatial dimensions, this paper incorporates the temporal characteristics of EEG signals by partitioning the signal into tokens along the temporal dimension (every 4 adjacent time steps form a token), randomly masking 75% of the tokens, and then reconstructing the masked temporal segments using a ViT-Large-style encoder-decoder architecture. The reconstruction loss is MSE, computed only on masked patches. After 500 epochs of pre-training, the decoder is discarded, and only the encoder is kept for downstream tasks.
    - **Design Motivation**: EEG signals are highly noisy and vary significantly across individuals, making traditional supervised learning struggle to capture robust representations. Self-supervised masked pre-training can utilize large amounts of unpaired EEG data. Temporal masking is preferred over spatial masking because EEG has high temporal resolution but low spatial resolution, meaning richer semantic information is contained in temporal dynamics. Experiments show that a 75% masking ratio is optimal, which differs from the lower masking ratios typically used in NLP.

2. **Fine-tuning with SD**:

    - **Function**: Seamlessly inject EEG conditions into the diffusion model by leveraging the generation capabilities of pre-trained SD.
    - **Mechanism**: The output of the pre-trained EEG encoder is transformed via a projection layer into conditional embeddings $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$ that match the dimension of SD text embeddings. The EEG conditional information is injected into the U-Net through the cross-attention mechanism, where $Q = W_Q \cdot \varphi_i(z_t)$, $K = W_K \cdot \tau_\theta(y)$, and $V = W_V \cdot \tau_\theta(y)$. During fine-tuning, the parameters of the EEG encoder and the cross-attention heads of the U-Net are optimized, while the remaining parts of SD are frozen. The standard SD loss is used: $\mathcal{L}_{SD} = \mathbb{E}_{x,\epsilon,t}[\|\epsilon - \epsilon_\theta(x_t, t, \tau_\theta(y))\|_2^2]$.
    - **Design Motivation**: SD has already learned powerful image generation capabilities from large-scale text-image data. Fine-tuning the cross-attention layers allows the EEG condition to replace the text condition in the existing generation pipeline. Keeping most parameters of the U-Net frozen preserves generation quality while preventing overfitting on small datasets.

3. **CLIP Space Alignment**:

    - **Function**: Further optimize EEG embeddings to align them with CLIP's image-text space.
    - **Mechanism**: Since SD uses the CLIP text encoder to generate conditional embeddings, EEG embeddings must be as close as possible to the CLIP space to effectively drive generation. This paper extracts CLIP embeddings of paired images using a CLIP image encoder $E_I$, and then maps EEG embeddings to the same space via a projection layer $h$. The alignment loss is defined as cosine distance: $\mathcal{L}_{clip} = 1 - \frac{E_I(I) \cdot h(\tau_\theta(y))}{|E_I(I)| \cdot |h(\tau_\theta(y))|}$. The CLIP model of this process is kept frozen.
    - **Design Motivation**: Relying solely on limited paired EEG-image data for end-to-end SD fine-tuning makes it difficult to align EEG and text spaces accurately. As a bridge, because CLIP's text and image spaces are highly aligned, pulling EEG closer to the CLIP image space implicitly pulls it closer to the text space. Experiments show that even without pre-training, using only CLIP alignment yields reasonable results, highlighting the importance of CLIP supervision.

### Loss & Training

Training involves loss functions across three stages: (1) MSE reconstruction loss for the pre-training stage; (2) SD diffusion loss $\mathcal{L}_{SD}$ for the fine-tuning stage; (3) CLIP alignment loss $\mathcal{L}_{clip}$ added on top of the SD loss for the alignment stage. SD version 1.5 is used. EEG signals are pre-processed by filtering to 5-95Hz, cropped to a length of 512, and the encoder is based on the ViT-Large architecture. Pre-training is performed for 500 epochs, and fine-tuning for 300 epochs. All experiments use data from Subject 4.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Brain2Image | Description |
|---|---|---|---|---|
| ImageNet-EEG | 50-way Top-1 Acc (%) | 45.8 | - | Evaluates semantic accuracy using a pre-trained ImageNet classifier |
| ImageNet-EEG | Image Quality | Significantly Better | Low quality, blurry | Qualitative comparison shows DreamDiffusion far outperforms Brain2Image |

### Ablation Study

| Configuration | Top-1 Acc (%) | Description |
|---|---|---|
| Full Model (MSM pre-training + CLIP + large encoder) | 45.8 | Optimal configuration |
| No pre-training + No CLIP + large encoder (Model 1) | 4.2 | Validates the necessity of pre-training and CLIP |
| No pre-training + No CLIP + small encoder (Model 2) | 3.7 | Small encoder also fails to prevent overfitting |
| No pre-training + With CLIP + large encoder (Model 3) | 32.3 | CLIP alignment alone yields substantial improvement |
| No pre-training + With CLIP + small encoder (Model 4) | 24.5 | Large encoder shows an advantage |
| With pre-training + With CLIP + mask ratio 0.25 (Model 5) | 19.7 | Masking ratio is too low |
| With pre-training + With CLIP + mask ratio 0.50 (Model 6) | - | Intermediate performance |
| With pre-training + With CLIP + mask ratio 0.75 | 45.8 | Optimal masking ratio |

### Key Findings

- Temporal masked pre-training is crucial: accuracy is only 4.2% without pre-training, but reaches 45.8% with it.
- CLIP alignment is extremely important: even without pre-training, adding CLIP supervision alone improves accuracy from 4.2% to 32.3%.
- A 75% masking ratio is optimal. Unlike the low masking ratios in NLP, this reveals that EEG signals possess redundancy characteristics similar to visual signals.
- A large encoder (297M parameters) significantly outperforms a small encoder (18.3M), indicating that EEG signal modeling requires sufficient model capacity.
- In some failed generation cases, classes with similar shapes or colors are confused, indicating that EEG signals provide coarse-grained information at the category level.

## Highlights & Insights

- Successfully achieves direct high-quality image generation from EEG signals for the first time, significantly reducing costs and lowering the barrier to entry compared to fMRI.
- The design of temporal masked pre-training is insightful—since EEG has high temporal resolution but low spatial resolution, temporal masking is much more appropriate than spatial masking.
- Cleverly leverages CLIP as a bridge for EEG-text-image space alignment, avoiding the difficulties of direct alignment.
- Good design of data sourcing—pre-training with data from 400+ subjects on the MOABB platform enhances cross-subject generalization.
- The method name "DreamDiffusion" is highly appealing, hinting at potential applications in dream visualization.

## Limitations & Future Work

- EEG signals currently only provide coarse-grained semantic information at the category level and cannot capture precise visual details (e.g., easily confusing different objects with similar colors).
- Experiments only utilize data from Subject 4; generalization across different subjects remains unverified.
- The amount of paired EEG-image data is extremely small (2000 images $\times$ 6 subjects), which severely limits fine-tuning efficacy.
- The pixel-level correspondence between generated images and original stimulus images is weak, focusing more on semantic-level matching.
- More sophisticated EEG channel selection strategies or attention mechanisms could be introduced to extract finer signal features.
- Future work could explore complementary fusion or joint training of EEG and fMRI.
- Contrastive learning could be introduced to enhance the performance of EEG pre-training.

## Related Work & Insights

- **MinD-Vis**: Brain signal image generation based on fMRI, using SC-MBM+DC-LDM. Effective but requires expensive equipment.
- **Brain2Image**: Early work on EEG-based image generation using LSTM+GAN/VAE, showing limited quality.
- **Stable Diffusion**: A powerful text-to-image diffusion model used as the generation engine in this paper.
- **MAE**: Masked Autoencoder. This paper adopts its masked pre-training concept but adapts it to the temporal domain.
- **CLIP**: A vision-language alignment model. This paper leverages its image encoder to bridge and align the EEG space.
- **Insights**: The paradigm of pre-training + CLIP alignment can be extended to other modalities (such as electromyography (EMG) or eye-tracking signals) for image generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to achieve high-quality EEG-to-image generation; temporal masked pre-training and CLIP bridging designs are novel.
- **Experimental Thoroughness**: ⭐⭐⭐ Detailed ablation studies, but lacks cross-subject evaluation and more quantitative metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, easy-to-understand method pipeline, and rich charts/figures.
- **Value**: ⭐⭐⭐⭐ Groundbreaking work driving the development of portable and low-cost "thought-to-image" technology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)

</div>

<!-- RELATED:END -->
