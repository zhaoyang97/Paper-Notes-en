---
title: >-
  [Paper Note] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing
description: >-
  [CVPR 2026][Image Generation][Zero-watermarking] This paper identifies that relational distances between image patch pairs remain invariant under AI editing, and exploits this invariance to build Rel-Zero, a zero-watermarking framework that achieves robust content authentication against diverse generative edits without modifying the original image.
tags:
  - CVPR 2026
  - Image Generation
  - Zero-watermarking
  - image editing robustness
  - patch-pair relational invariance
  - content authentication
  - diffusion models
date: 2026-05-08
content_hash: 5e950dfdef262335
---

# Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing

**Conference**: CVPR 2026
**arXiv**: [2603.17531](https://arxiv.org/abs/2603.17531)
**Code**: None
**Area**: Image Generation
**Keywords**: Zero-watermarking, image editing robustness, patch-pair relational invariance, content authentication, diffusion models

## TL;DR
This paper identifies that relational distances between image patch pairs remain invariant under AI editing, and exploits this invariance to build Rel-Zero, a zero-watermarking framework that achieves robust content authentication against diverse generative edits without modifying the original image.

## Background & Motivation

**Background**: Digital watermarking is a critical technique for protecting image copyright and authenticating content integrity. Existing methods fall into two categories: embedding-based watermarking (injecting signals into images) and zero-watermarking (extracting feature fingerprints stored in an external database without modifying the image).

**Limitations of Prior Work**: Embedding-based methods (e.g., VINE, RobustWide) must inject strong signals to withstand diffusion model editing, which inevitably introduces perceptible distortion and degrades image quality. Zero-watermarking methods preserve perfect image quality but rely on global features (SIFT, absolute feature descriptors from deep classifiers) — precisely the features that generative models are most capable of altering — resulting in poor robustness.

**Key Challenge**: A fundamental fidelity–robustness trade-off exists: embedding-based methods sacrifice quality for robustness, while zero-watermarking methods preserve quality at the cost of robustness. In high-precision domains such as medical imaging and autonomous driving, watermark-induced noise may lead to catastrophic consequences.

**Goal**: Achieve high robustness against generative AI editing without modifying the original image (zero-watermarking).

**Key Insight**: Through large-scale empirical analysis, the authors find that although AI editing drastically alters the pixel values and absolute features of individual patches, the pairwise relational distances between patches exhibit remarkable invariance: $d_{ij}^{\text{after}} \approx \alpha \cdot d_{ij}^{\text{before}} + \beta$, where $\alpha \approx 1$, $\beta \approx 0$, and $R^2 > 0.95$.

**Core Idea**: Leverage the editing invariance of patch-pair relational distances as the foundation for zero-watermarking, encoding the watermark as an index set of stable patch pairs.

## Method

### Preliminary Finding: Editing Invariance of Patch-Pair Distances

Prior to proposing the method, the authors conduct a large-scale empirical study. They randomly sample 10,000 images from the UltraEdit and MagicBrush datasets, covering three editing scenarios: deterministic regeneration (2,000 images), global editing (4,000 images), and local editing (4,000 images). Each image is partitioned into $N=256$ non-overlapping patches, represented by RGB mean vectors $\{v_i\}_{i=1}^N$, and the L2 distance differences of all $\binom{N}{2}$ patch pairs before and after editing are computed.

**Key Findings**: Distance differences exhibit a near-zero mean and tight distribution with no systematic bias. Further distance–distance correlation analysis reveals a strong linear relationship $d_{ij}^{\text{after}} \approx \alpha \cdot d_{ij}^{\text{before}} + \beta$, with slope $\alpha \approx 1$, intercept $\beta \approx 0$, coefficient of determination $R^2 > 0.95$, and Spearman correlation $\rho \approx 1$. This reveals a **near-affine invariance** in the feature space: relative inter-patch distances undergo only uniform scaling after editing.

Two levels of theoretical explanation are provided: (1) Diffusion-based editing models are trained with explicit or implicit content/structure preservation losses (LPIPS, L1/L2 reconstruction losses) that penalize unnecessary perturbations, making cross-patch relational structure a core invariant preserved by optimization; (2) Semantic editing corresponds to low-dimensional directions in latent space, which, upon decoding, impose approximately uniform transformations on image statistics. When the transformation is approximately affine, $v_i' \approx Av_i + b$, it follows that $v_i' - v_j' \approx A(v_i - v_j)$, so distances are only scaled while relational ordering is preserved.

### Overall Architecture
Rel-Zero comprises three stages: (1) **Stable patch-pair identification** — a VAE is used to simulate editing, identifying ground-truth invariant patch pairs as training targets; (2) **Patch relationship learning** — a lightweight edge predictor is trained to predict stable patch pairs from a single image; (3) **Watermark generation and verification** — the top-K predicted pairs are extracted as the zero-watermark. Crucially, at inference time only the network from stage (2) is required; given a single image, it outputs a watermark index set without needing the VAE or any editing operations.

### Key Designs

1. **Stable Patch-Pair Identification (Training Target Construction)**

    - **Function**: Construct the ground-truth stable patch-pair set $\mathcal{E}_g$ for training.
    - **Mechanism**: A pretrained VAE (inspired by VINE) is used to simulate generative editing. The original and VAE-reconstructed images are each passed through a ViT to extract patch-level features $\mathcal{F} = \phi_{\text{vit}}(\mathbf{I})$. The stability score of each patch pair is computed as $s_{ij} = \exp(-|d_{ij} - \hat{d}_{ij}|)$, and the top-K pairs with the highest stability scores are selected as ground truth.
    - **Design Motivation**: VAE reconstruction has a structurally similar effect on patch relationships as diffusion editing (inspired by VINE), but at a fraction of the computational cost — no full diffusion editing pipeline is required. Using high-dimensional ViT features rather than RGB vectors for distance computation captures richer semantic relationships. Note that the discovery phase uses RGB means for analysis, whereas the method phase upgrades to ViT features, enhancing representational capacity.

2. **Patch Relationship Learning (Edge Predictor)**

    - **Function**: Train a lightweight predictor to identify which patch pairs are stable from a single image.
    - **Mechanism**: A fully connected pair set $\mathcal{E}$ is constructed from the $N$ patch features extracted by ViT. The feature of each pair $(i,j)$ is $\mathbf{f}_i \oplus \mathbf{f}_j \oplus \|\mathbf{f}_i - \mathbf{f}_j\|_2$ (concatenation + distance), which is passed through an MLP $\psi$ and sigmoid $\sigma$ to produce the prediction score $p_{ij} = \sigma(\psi(\mathbf{f}_i \oplus \mathbf{f}_j \oplus \|\mathbf{f}_i - \mathbf{f}_j\|_2))$.
    - **Design Motivation**: A simple MLP suffices — ablation experiments show that Transformers or GATs actually blur fine-grained inter-patch distance differences (Transformer drops to 92.11%, GAT to 94.45%, while MLP achieves 97.43%). The key information lies in the **local distance features** of patch pairs; attention mechanisms mix patch representations and thereby impair precise distance discrimination. This reflects a "less is more" design philosophy.

3. **Watermark Generation and Verification**

    - **Function**: Generate and verify zero-watermarks based on predictor outputs.
    - **Mechanism**: During generation, the top-K most confident predicted pairs $\mathcal{E}_p = \text{Top-K}(\Phi(\phi_{\text{vit}}(\mathbf{I})))$ are stored as watermark indices. During verification, the same top-K pairs $\mathcal{E}_p'$ are extracted from the suspect image, and the Jaccard overlap $\eta = |\mathcal{E}_p \cap \mathcal{E}_p'| / K$ is computed as the authentication criterion.
    - **Design Motivation**: Encoding the watermark as a patch-pair index set rather than numerical features naturally accommodates affine-transformation invariance — relational ordering rather than absolute values is preserved. The index set can be hashed and encrypted for storage in an external database (a secure storage scheme is provided in the appendix). The verification threshold is calibrated based on a target false positive rate (FPR = 0.1%) to ensure high-confidence authentication.

### Loss & Training
A standard binary cross-entropy loss is used to train the edge predictor:
$$\mathcal{L}_{BCE} = -\sum_{i \neq j} [y_{ij} \log(\hat{y}_{ij}) + (1-y_{ij})\log(1-\hat{y}_{ij})] / N(N-1)$$
where $y_{ij}=1$ for top-K invariant pairs (positive samples) and $y_{ij}=0$ for remaining pairs (negative samples). The positive-to-negative ratio is approximately $K : \binom{N}{2}-K$, yielding severe class imbalance ($K=50$ vs. $\sim$19,000 negative samples), yet BCE remains effective in this setting.

**Implementation Details**: ViT-B/16 serves as a frozen feature extractor (not updated during training); the VAE from Stable Diffusion v1.4 is used to generate training targets; $K=50$ pairs; patch size $16 \times 16$ ($N=196$ patches for 224×224 images); trained on COCO; NVIDIA A100 GPU.

## Key Experimental Results

### Main Results

| Method | Type | PSNR↑ | Regen | Pix2Pix | Magic | Ultra | CtrlN | Cropout | Scale | Contrast | Bright | Gauss |
|--------|------|-------|-------|---------|-------|-------|-------|---------|-------|----------|--------|-------|
| DWT-DCT | Embedding | 40.38 | 0.09 | 0.04 | 0.05 | 0.32 | 0.56 | 10.35 | 6.78 | 30.18 | 51.88 | 12.45 |
| RobustWide | Embedding | 41.93 | 90.41 | 97.23 | 81.97 | 80.45 | 82.11 | 95.31 | 96.45 | 98.93 | 98.89 | 98.12 |
| VINE | Embedding | 37.34 | 99.98 | 97.46 | 94.58 | 99.96 | 93.04 | 54.87 | 76.43 | 98.43 | 97.90 | 98.37 |
| ConZWNet | Zero-wm | ∞ | 0.10 | 0.02 | 0.01 | 5.13 | 2.41 | 98.75 | 97.43 | 96.22 | 96.56 | 98.75 |
| FGPCET | Zero-wm | ∞ | 1.13 | 0.54 | 0.11 | 7.25 | 3.22 | 89.31 | 84.78 | 86.31 | 85.44 | 84.67 |
| **Rel-Zero** | **Zero-wm** | **∞** | **85.13** | **89.65** | **95.63** | **96.55** | **97.43** | **98.45** | **98.57** | **96.45** | **97.93** | **95.12** |

All values are TPR@(0.1% FPR). **Core conclusions**:
- Rel-Zero decisively outperforms prior zero-watermarking methods (other zero-watermarks achieve TPR < 10% under generative editing, while Rel-Zero reaches 85–97%).
- On local editing tasks (Ultra/CtrlN), Rel-Zero even surpasses embedding-based VINE and RobustWide.
- Under conventional perturbations, Rel-Zero maintains 98%+ robustness, as uniform transformations preserve the geometric structure of patch-pair relationships.
- VINE performs poorly on Cropout (54.87%) and Scaling (76.43%), whereas Rel-Zero is naturally robust to these distortions.

### Ablation Study

| Configuration | TPR@(0.1% FPR) | Note |
|--------------|----------------|------|
| Ours (ViT + MLP) | 97.43 | Full model |
| ViT → ResNet-18 | 84.13 | Weaker backbone yields insufficient features |
| ViT → ResNet-50 | 85.21 | ResNet still inferior to ViT's patch-level representations |
| MLP → Transformer+MLP | 92.11 | Attention blurs distance differences |
| MLP → GAT+MLP | 94.45 | GAT has similar issues but is slightly better |

### Uniqueness Analysis
1,000 images are sampled from each of COCO, UltraEdit, and MagicBrush; the Jaccard overlap $\eta_{a,b}$ between watermarks of all image pairs is computed. Results show that cross-image overlaps concentrate near zero with minimal variance, confirming that the learned relational pairs constitute **image-specific signatures** rather than generic templates.

### Parameter Analysis
- **Effect of Top-K**: Robustness increases steadily with $K$, saturating at $K=50$. ControlNet-Inpainting and UltraEdit are most robustly handled; Regeneration remains the most challenging scenario.
- **Effect of Patch Size**: $14 \times 14$ and $16 \times 16$ yield comparable performance; $32 \times 32$ causes a sharp drop — overly coarse partitioning weakens relational modeling and results in too few patch pairs.

### Key Findings
- The ViT backbone contributes most significantly — ViT natively produces patch-level features that are more sensitive to relational distance variations. ResNet, despite strong feature extraction capability, lacks patch-wise structure.
- A simple MLP outperforms Transformers/GATs — pair prediction is fundamentally a distance estimation task; attention mechanisms mix patch representations and impair precise distance discrimination.
- Conventional perturbations (noise, scaling, contrast, brightness) are essentially uniform image transformations that do not alter the relative relationships between patch pairs, making Rel-Zero naturally robust to them.
- Global editing (e.g., Regeneration) remains the greatest challenge, as large-scale semantic changes may disrupt a subset of patch-pair relationships.

## Highlights & Insights
- **The discovery of patch-pair relational invariance** is particularly insightful. Through statistical analysis of 10,000 images, the authors identify a near-perfect linear relationship ($R^2 > 0.95$) between pre- and post-edit patch-pair distances, providing a solid empirical and theoretical foundation for zero-watermarking.
- **Using VAE reconstruction to simulate diffusion editing for training data generation** is a clever design — it reduces computational overhead by orders of magnitude while approximating the structural effects of diffusion-based editing.
- **Encoding watermarks as graph index sets (edge sets)** is a paradigm worth adopting more broadly — it transfers naturally to video watermarking (spatio-temporal patch pairs) and 3D model watermarking (voxel-pair relationships).

## Limitations & Future Work
- **Resolution constraints**: All training and evaluation are conducted at 224×224; effectiveness on high-resolution images (e.g., 4K medical imagery) remains unverified. At higher resolutions, the number of patches $N$ grows dramatically, and the number of pairs scales as $O(N^2)$, posing computational efficiency challenges.
- **Generalization across editing models**: Testing is limited to five editing models; generalization to more powerful future editors (e.g., video-diffusion-based editing, 3D-aware editing) is unknown.
- **Adversarial security**: An adversary aware of the patch partitioning scheme, the value of $K$, and the ViT architecture may design targeted attacks to corrupt specific patch-pair relationships.
- **Class imbalance**: The extreme imbalance between positive and negative samples ($K=50$ vs. $\sim$19,000) under BCE loss suggests that focal loss or adaptive sampling strategies could be explored.
- **Scalable extensions**: Multi-scale patch partitioning for enhanced robustness; temporal extension to video watermarking; adaptive patch partitioning guided by semantic segmentation.

## Related Work & Insights
- **vs. VINE/RobustWide (embedding-based)**: These methods incorporate editing models into optimization via adversarial training, achieving strong robustness at the cost of image quality degradation (VINE's PSNR is only 37.34 dB) and substantial training overhead. Rel-Zero maintains perfect fidelity (PSNR = ∞) while achieving comparable or superior performance on local editing (Ultra: 96.55% vs. VINE 99.96%; CtrlN: 97.43% vs. VINE 93.04%) and conventional perturbations.
- **vs. ConZWNet/FGPCET (zero-watermarking)**: Both are zero-watermarking approaches but follow fundamentally different principles. Prior methods rely on absolute deep feature descriptors or handcrafted features — precisely what generative models excel at altering — leading to near-complete failure under AI editing (TPR < 10%). By discovering and exploiting relational invariance, Rel-Zero improves robustness by two orders of magnitude.
- **vs. DWT-DCT (traditional)**: Frequency-domain embedding methods fail completely under AI editing (TPR < 1%), demonstrating that frequency-domain signals are entirely destroyed during diffusion-based reconstruction.
- **Broader implications**: The relational invariance insight transfers to other authentication scenarios — for example, leveraging inter-facial-region relational consistency in deepfake detection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The discovery of patch-pair relational invariance is insightful, though the overall framework is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Testing spans multiple editing models with uniqueness analysis and parameter ablations, but high-resolution experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The narrative chain from observation to hypothesis to validation to method is exceptionally clear and coherent.
- **Value**: ⭐⭐⭐⭐ — Introduces a new paradigm for zero-watermarking with practical applicability in high-fidelity scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[CVPR 2026\] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](../../NeurIPS2025/image_generation/towards_robust_zero-shot_reinforcement_learning.md)

<!-- RELATED:END -->
