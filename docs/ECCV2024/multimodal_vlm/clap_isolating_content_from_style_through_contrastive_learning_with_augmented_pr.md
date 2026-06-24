---
title: >-
  [Paper Note] CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts
description: >-
  [ECCV2024][Multimodal VLM][CLIP] From the perspective of causal generative models, this paper proposes CLAP (Contrastive Learning with Augmented Prompts). It trains a lightweight disentanglement network using text prompt augmentation and contrastive learning to separate content and style within CLIP pre-trained features. Trained solely on text, CLAP simultaneously improves representation quality for both image and text modalities, achieving consistent gains in zero-shot class…
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "CLIP"
  - "Content-Style Disentanglement"
  - "Contrastive Learning"
  - "Text Augmentation"
  - "Causal Representation Learning"
date: 2026-05-08
content_hash: 4719097d0f348ebb
---

# CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts

**Conference**: ECCV2024  
**arXiv**: [2311.16445](https://arxiv.org/abs/2311.16445)  
**Code**: [YichaoCai1/CLAP](https://github.com/YichaoCai1/CLAP)  
**Area**: Multimodal VLM  
**Keywords**: CLIP, Content-Style Disentanglement, Contrastive Learning, Text Augmentation, Causal Representation Learning

## TL;DR

From the perspective of causal generative models, this paper proposes CLAP (Contrastive Learning with Augmented Prompts). It trains a lightweight disentanglement network using text prompt augmentation and contrastive learning to separate content and style within CLIP pre-trained features. Trained solely on text, CLAP simultaneously improves representation quality for both image and text modalities, achieving consistent gains in zero-shot classification, few-shot classification, and adversarial robustness.

## Background & Motivation

**Background**: CLIP and other contrastive vision-language models have achieved strong generalization capabilities through large-scale image-text contrastive pre-training, becoming widely used in downstream tasks such as zero-shot classification and prompt learning.

**Limitations of Prior Work**: Features learned by CLIP conflate content (e.g., class semantics) and style (e.g., image texture, lighting, prompt phrasing). This leads to three key issues:

- **Sensitivity to prompts**: Significant discrepancies in zero-shot performance exist across different text prompts (e.g., "a photo of a [class]" vs "[class]").
- **Limited few-shot performance**: Distribution shifts in few-shot scenarios allow style information to interfere with classification.
- **Adversarial vulnerability**: Adversarial attacks essentially alter style information, making conflated features prone to deception.

**Key Challenge**: Theoretically (von Kügelgen et al., NeurIPS 2021), content-style disentanglement can be achieved by applying soft interventions to all style variables followed by contrastive learning. However, in practice, image augmentations struggle to sufficiently alter all style factors—for instance, transforming a photo into a sketch is extremely difficult in the image space, whereas in the text space, it merely requires changing a few words.

**Key Insight**: Vision and language data share the same latent space (comprising content $c$ and style $s$). Thus, disentanglement networks can be trained using augmentations in the text modality and directly transferred to the vision modality. Text is inherently highly semantic and logically structured, offering much more precise control over style attributes than images.

**Core Idea**: Use template-based prompt engineering (deleting attributes, swapping order, inserting noise) to construct style-variant text pairs. A lightweight MLP disentanglement network is then trained via InfoNCE contrastive learning on top of a frozen CLIP encoder to extract pure content features.

## Method

### Overall Architecture

The CLAP pipeline consists of three steps:

1. **Training Phase**: On top of the frozen CLIP text encoder, train a disentanglement network $f_c$ using template-based prompts and their augmented counterparts.
2. **Transfer Phase**: Since vision and language share the latent space, the trained $f_c$ is directly transferred for visual features.
3. **Inference Phase**: For an image $x$, the feature is $f_c^* \circ f_x^*(x)$; for a text $t$, the feature is $f_c^* \circ f_t^*(t)$. Cosine similarity is computed for zero-shot classification.

### Key Designs

#### Key Design 1: Causal Generative Model (Theoretical Basis)

- **Function**: Establishes a causal generative model for vision-language data, partitioning the latent space into content $c$ and style $s$.
- **Mechanism**: $s := g_s(c)$, $x := g_x(c,s)$, $t := g_t(c,s)$, $y := g_y(c)$. The label $y$ is determined solely by the content, while the image and text are generated from the shared $(c,s)$ via different generation processes.
- **Design Motivation**: Provides theoretical guarantees—when all style variables are modified via soft intervention, contrastive learning can block-identify content variables. This justifies the use of data augmentation to disentangle content.

#### Key Design 2: Text Prompt Augmentation Strategy

- **Function**: Designed augmentation methods specifically targeting template prompts to maximize style variation without changing the content.
- **Mechanism**: Based on the structured prompt "a [art style] [image type] of a [object size] [object color] [class]", five augmentation operations are designed:
    - **OSD** (Object Size Deletion): Deletes size descriptions.
    - **OCD** (Object Color Deletion): Deletes color descriptions.
    - **ITD** (Image Type Deletion): Deletes the image type.
    - **ASD** (Art Style Deletion): Deletes the art style.
    - **SPO** (Swapping Prompt Order): Swaps the order of prompt segments.
    - **IGN** (Inserting Gaussian Noise): Inserts Gaussian noise (mean 0, std 0.02) into the tokenized prompt.
- **Design Motivation**: Similar to random masking in images, but text augmentation can precisely delete specific style attributes without harming content. For example, deleting "realistic" only changes style, leaving the category designation "car" intact. Image masking, conversely, is prone to corrupting both content and style.

#### Key Design 3: Disentanglement Network Architecture (Residual MLP + Zero Init)

- **Function**: Designed a lightweight network appended to the CLIP encoder to extract content from conflated features.
- **Mechanism**: Adopts a residual MLP structure, where the main branch includes a normally initialized Linear layer $\rightarrow$ SiLU $\rightarrow$ a **zero-initialized** (no bias) Linear layer. The shortcut branch preserves original input features. During inference, a weight coefficient $\alpha$ controls the fusion ratio between the main branch output and the input features.
- **Design Motivation**: Zero initialization is inspired by the zero-conv concept in ControlNet, ensuring that the network output equals the input features in the early training stages (optimizing from the pre-trained CLIP space) to avoid disrupting existing representations from a random starting point.

### Loss & Training

CLAP's training objective is a two-term InfoNCE loss:

$$f_c^* = \arg\min_{f_c} \mathbb{E}_{\{t_i\} \in \mathcal{D}_t} \left[ \mathcal{L}(f_c \circ f_t^*; \{t_i, \tilde{t}_i\}, \tau) + \lambda \mathcal{L}(f_c \circ f_t^*; \{t_i^c, \tilde{t}_i\}, 1) \right]$$

- **First Term**: The original prompt $t_i$ and the augmented prompt $\tilde{t}_i$ form positive pairs, while augmented prompts from other samples serve as negative pairs, with temperature $\tau$.
- **Second Term**: The class name $t_i^c$ and the augmented prompt $\tilde{t}_i$ form positive pairs, increasing the magnitude of variation between prompt pairs, weighted by $\lambda$, with temperature fixed at 1.
- Training data consists entirely of synthetic template prompts (480 per class, combined from 10 colors $\times$ 3 sizes $\times$ 8 image types $\times$ 2 styles), requiring no real images.
- Uses Adam optimizer, lr=0.0001, up to 8000 steps, with early stopping (if no improvement over 5 checkpoints).

## Key Experimental Results

### Main Results

Evaluation is conducted on four multi-domain datasets—PACS, VLCS, OfficeHome, and DomainNet—using ViT-B/16 CLIP.

**Zero-shot Results (Average top-1 acc %)**:

| Prompt | Method | PACS | VLCS | OfficeHome | DomainNet | Overall |
|--------|--------|------|------|------------|-----------|---------|
| ZS(C) "[class]" | CLIP | 95.7 | 76.4 | 79.8 | 57.8 | 77.4 |
| | Im.Aug | 96.5 | 79.5 | 77.0 | 51.5 | 76.1 |
| | **CLAP** | **97.2** | **82.6** | **81.0** | **58.7** | **79.9** |
| ZS(PC) "a photo of a [class]" | CLIP | 96.1 | 82.4 | 82.5 | 57.7 | 79.7 |
| | **CLAP** | **97.2** | **83.4** | **83.0** | **59.0** | **80.6** |
| ZS(NC) "[noise][class]" | CLIP | 90.8 | 68.3 | 71.5 | 51.0 | 70.4 |
| | **CLAP** | **97.2** | **81.0** | **73.5** | **52.6** | **76.1** |

CLAP outperforms both CLIP and Im.Aug across all prompt types and datasets. Notably, on the noisy prompt ZS(NC), CLAP's overall gain reaches +5.7% (70.4 $\rightarrow$ 76.1).

**Few-shot Results**: In 1-shot scenarios, CLAP outperforms linear probing on CLIP by +10% (PACS), +3.5% (VLCS), +2.5% (OfficeHome), and +1.5% (DomainNet).

**Adversarial Robustness (avg top-1 acc % under attacks)**:

| Setting | Method | FGSM Avg | PGD-20 Avg | CW-20 Avg | Overall |
|---------|--------|----------|------------|-----------|---------|
| ZS(C) | CLIP | 58.2 | 13.0 | 11.0 | 29.2 |
| | Im.Aug | 62.7 | 13.2 | 11.0 | 31.1 |
| | **CLAP** | **65.8** | **14.0** | **12.1** | **32.7** |
| 1-shot | CLIP | 42.2 | 16.9 | 6.8 | 23.7 |
| | **CLAP** | **50.7** | **28.6** | **9.2** | **31.9** |

### Ablation Study

**Ablation of Prompt Augmentation Combinations (VLCS Dataset)**:

| Augmentation Combination | ZS(Avg.) ↑ | R ↓ | δ ↓ | Δ(NC) ↓ |
|---------|-----------|-----|-----|---------|
| CLIP baseline | 77.3| 6.1 | 2.8 | 8.1 |
| EDA | 81.6 | 1.9 | 0.9 | 2.3 |
| OSD+OCD+ITD+ASD+SPO | 82.0 | 1.2 | 0.6 | 1.7 |
| OSD+OCD+ITD+ASD | 80.1 | 2.5 | 1.2 | 3.0 |
| **OSD+OCD+SPO+IGN** | **82.6** | **0.8** | **0.4** | **1.6** |

The optimal combination is OSD+OCD+SPO+IGN, achieving the highest average accuracy and lowest variance.

**Ablation of Prompt Sources (VLCS Dataset)**:

| Source | ZS(Avg.) ↑ | R ↓ | δ ↓ |
|------|-----------|-----|-----|
| CLIP baseline | 77.3 | 6.1 | 2.8 |
| LLM (ChatGPT-3.5) | 78.2 | 3.2 | 1.5 |
| Random | 81.6 | 0.7 | 0.3 |
| PromptStyler | 81.2 | 2.7 | 1.2 |
| Template (ours) | 81.6 | 1.9 | 0.9 |

Conclusion: Simpler prompt formats (Random, Template) actually yield better performance.

### Key Findings

1. **Text augmentation greatly outperforms image augmentation**: CLAP consistently surpasses Im.Aug across all metrics. Moreover, Im.Aug performs worse than the CLIP baseline on some datasets (such as DomainNet), showing that image augmentations fail to sufficiently vary all style factors.
2. **Significant improvement in prompt robustness**: CLAP reduces prompt-to-prompt performance fluctuation (R) from 2.7% to 1.1%, and decreases performance drop due to noisy prompts (Δ(NC)) from 7.0% to 3.8%.
3. **High training efficiency**: Since training only relies on text data, CLAP finishes in about 11 minutes on PACS/VLCS, and about 47 minutes on DomainNet (compared to 3.3 hours for Im.Aug).
4. **Cross-model generalization**: Repeated experiments on ViT-L/14 and ResNet50x16 show that CLAP consistently improves zero-shot performance and reduces variance.
5. **t-SNE visualization**: Representations from CLAP present clearer inter-class separation and tighter intra-class clustering.

## Highlights & Insights

1. **Connecting theory and practice through a causal lens**: The content-style causal disentanglement theory proposed by von Kügelgen et al. is practically applied to CLIP-like models for the first time, establishing a complete pathway from SCM to engineering methods.
2. **Clever exploitation of cross-modal transfer**: Training the disentanglement network in the text modality and transferring it directly to the vision modality takes advantage of the aligned latent space of CLIP, significantly reducing training costs (no real images required).
3. **Deep insight on Text Augmentation > Image Augmentation**: The logical and structured nature of text makes it far more suitable than high-dimensional images for property-wise style interventions, a highly inspiring observation.
4. **Zero-init residual design philosophy**: Borrowing the zero-conv concept from ControlNet for an MLP setup ensures that fine-tuning starts from the pre-trained space, preventing catastrophic forgetting.
5. **Extremely lightweight method**: The disentanglement network consists of only a two-layer MLP, adding no inference latency and requiring no modifications to the weights of the CLIP encoders.

## Limitations & Future Work

1. **Reliance on template prompt structure**: The augmentation strategies (OSD, OCD, etc.) are specifically designed for template-structured prompts; generalizing them to free-form text descriptions may require new augmentation strategies.
2. **Class coverage of synthetic data**: Training prompts require a pre-defined list of class names, so applicability to open-vocabulary scenarios remains to be validated.
3. **Assumption of complete style variable perturbation**: The theory requires all style factors to change to guarantee block identifiability, which practical augmentations may not fully satisfy.
4. **Evaluation limited to classification tasks**: CLAP has not yet been validated on broader vision-language tasks such as retrieval, detection, or segmentation.
5. **Dataset-dependent hyperparameters**: Hyperparameters like $\alpha$, $\tau$, $\lambda$, and latent dim need to be adjusted for different datasets.

## Related Work & Insights

- **Causal Representation Learning**: The content-style theory of von Kügelgen et al. (NeurIPS 2021) is the direct foundation of this work. Future work could integrate identifiable causal models (Liu et al., ICLR 2024) for stronger theoretical guarantees.
- **Prompt Learning**: Methods like CoOp, CoCoOp, and MaPLe adapt to downstream tasks by learning prompts but do not modify the CLIP representation itself. CLAP offers an orthogonal direction of improvement—disentangling features first, followed by prompt learning.
- **PromptStyler**: Also exploits style diversity for augmentation but focuses on source-free domain generalization, whereas CLAP targets general representation improvement.
- **Zero-init in ControlNet**: The zero-initialization design concept can be extended to more pre-trained model fine-tuning scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using text augmentation instead of image augmentation, derived from causal theory, is novel, and the concept of cross-modal disentanglement network transfer is highly creative.
- **Technical Quality**: ⭐⭐⭐⭐ — Clear theoretical motivation, sound methodological design, and comprehensive experimental comparison (4 datasets $\times$ multiple evaluation settings).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations (augmentation combination, prompt source, hyperparameters, model scales), though validation on a wider range of task types is lacking.
- **Value**: ⭐⭐⭐⭐ — Lightweight and plug-and-play; training requires only text and no images, making it engineering-friendly.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear narrative path from causal model $\rightarrow$ theoretical motivation $\rightarrow$ method $\rightarrow$ experiments, with high-quality tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](x-former_unifying_contrastive_and_reconstruction_learning_for_mllms.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[ECCV 2024\] ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling](artvlm_attribute_recognition_through_vision-based_prefix_language_modeling.md)

</div>

<!-- RELATED:END -->
