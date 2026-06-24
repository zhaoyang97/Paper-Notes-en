---
title: >-
  [Paper Note] DreamRelation: Bridging Customization and Relation Generation
description: >-
  [CVPR 2025][Image Generation][Relation-Aware Customized Generation] DreamRelation proposes a relation-aware customized image generation framework. Through three key designs—a decoupled data engine, Keypoint Matching Loss (KML), and local token injection—it maintains the identity consistency of multiple subjects while accurately generating textual relations (such as hugging, riding, etc.) between them, outperforming existing methods across the board on RelationBench.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Relation-Aware Customized Generation"
  - "Multi-Subject Customization"
  - "Keypoint Matching Loss"
  - "Local Feature Injection"
  - "LoRA Fine-Tuning"
date: 2026-05-08
content_hash: ca2536825c8f83eb
---

# DreamRelation: Bridging Customization and Relation Generation

**Conference**: CVPR 2025  
**arXiv**: [2410.23280](https://arxiv.org/abs/2410.23280)  
**Code**: [https://github.com/shi-qingyu/DreamRelation](https://github.com/shi-qingyu/DreamRelation)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Relation-Aware Customized Generation, Multi-Subject Customization, Keypoint Matching Loss, Local Feature Injection, LoRA Fine-Tuning

## TL;DR
DreamRelation proposes a relation-aware customized image generation framework. Through three key designs—a decoupled data engine, Keypoint Matching Loss (KML), and local token injection—it maintains the identity consistency of multiple subjects while accurately generating textual relations (such as hugging, riding, etc.) between them, outperforming existing methods across the board on RelationBench.

## Background & Motivation

1. **Background**: Customized image generation has made significant progress. Methods based on fine-tuning (DreamBooth, Custom Diffusion) and training-based approaches (MS-Diffusion, SSR-Encoder) can robustly maintain user-provided subject identities.
2. **Limitations of Prior Work**: Existing multi-subject customization methods ignore the relations between subjects. When users provide two subject images and a text describing their relation (e.g., "hugging"), synthesized results often fail to correctly express this relation, resulting in either missing relations or mixed-up identities.
3. **Key Challenge**: (a) Lack of suitable training data to decouple identity and relation information; (b) in terms of model design, the overly dominant control of image prompts causes textual relation descriptions to be ignored; (c) global features extracted by CLIP lack local details (such as features of 'hands'), leading to subject blending/confusion in overlapping scenarios.
4. **Goal**: To accurately generate text-described relations between subjects while maintaining multiple subject identities—specifically, "relation-aware customized image generation".
5. **Key Insight**: The authors observe that relations are primarily reflected in the pose changes of subjects. However, existing methods crop subjects as image prompts, which introduces a "copy-paste" effect (where poses remain unchanged and relations cannot be learned). Therefore, this issue must be addressed simultaneously from both the data and model levels.
6. **Core Idea**: By using a relation-aware data engine to decouple identity and relation, and then explicitly guiding pose adjustment via Keypoint Matching Loss and avoiding subject confusion through local token injection, relation-aware customized generation is achieved.

## Method

### Overall Architecture
DreamRelation is built on MS-Diffusion. The inputs include two subject images $c_i$ (image prompts) and a relation text prompt $c_t$; the output requires maintaining both subject identities while accurately reflecting the relation described in the text. The entire framework consists of three core phases: (1) constructing training triplets $(x_k, c_i, c_t)$ using a relation-aware data engine; (2) injecting LoRA into the text cross-attention layers of the U-Net for relation learning; and (3) introducing KML and local token injection to enhance relation generation and prevent subject confusion.

### Key Designs

1. **Relation-Aware Data Engine**:

    - **Function**: Constructs high-quality training triplet data for relation learning.
    - **Mechanism**: Leveraging the multi-turn conversation capability of DALL-E 3, the authors generate triplet images containing the same subjects but in different poses: relation target image $x_k$, separate identity images $c_i$, and relation description text $c_t$. With the prompt "The photo of the same", DALL-E 3 is instructed to remember and maintain subject identity. Then, X-Pose is used to detect keypoints, SAM to generate masks, and LLaVA to produce captions.
    - **Design Motivation**: Crucially, directly cropping subjects from $x_k$ to serve as $c_i$ leads to a "copy-paste" effect—the cropped image's pose is identical to the target image, which prevents the model from learning the pose variations induced by relations. By generating $c_i$ with identical identity but different poses, the model is forced to learn relation details from $c_t$, thereby decoupling identity and relation.

2. **Keypoint Matching Loss (KML)**:

    - **Function**: Explicitly supervises subject poses in the latent space, guiding the model to generate correct, relation-matching poses.
    - **Mechanism**: X-Pose is used to detect 17 keypoints for each subject in $x_k$ and $c_i$. During training, the U-Net outputs $\hat{\epsilon}$ to predict $\hat{z}_0$, and the MSE loss between the predicted keypoint locations in the VAE latent space $\hat{z}_0$ and the corresponding keypoints in $\mathcal{E}(c_i)$ is calculated: $\mathcal{L}_{KML} = \frac{1}{n_{kp}} \mathbb{E} \| \mathcal{E}(c_i)[c_{kp}^{c_i}] - \hat{z}_0[c_{kp}^{x_k}] \|_2^2$. The overall loss is $\mathcal{L} = \mathcal{L}_{denoise} + \lambda \cdot \mathcal{L}_{KML}$, where $\lambda = 1e{-3}$.
    - **Design Motivation**: Relations are tightly coupled with poses (e.g., "hugging" requires crossed arms, "riding" requires feet on pedals). It is difficult to control poses precisely solely with the diffusion loss. KML imposes constraints directly on latent keypoints, enabling the model to learn to adjust subject poses for different relations.

3. **Local Token Injection**:

    - **Function**: Extracts fine-grained local features from image prompts to prevent subject confusion in overlapping scenarios.
    - **Mechanism**: The last layer of the CLIP Image Encoder is modified to retrieve dense features $h_{dense}$, which are aligned with global features through a self-distillation approach. During inference, dense features are block-pooled to obtain local tokens $tok_{local}$, which are then concatenated with image-level tokens $tok_{image}$ and fed into the ID extractor: $q = q + \text{Attention}(\text{concat}[q, tok_{image}, tok_{local}])$.
    - **Design Motivation**: CLIP's global image features are too coarse and lack local detail (such as hand features), causing failure in distinguishing the two subjects when generating relations like "shaking hands". Introducing dense local features provides precise local information, effectively avoiding subject confusion.

### Loss & Training
- Overall loss: $\mathcal{L} = \mathcal{L}_{denoise} + \lambda \cdot \mathcal{L}_{KML}$
- LoRA (rank=4) is only injected into $W_q, W_{k_t}, W_{v_t}, W_{out}$ of the U-Net text cross-attention layers, with other parameters frozen
- Fine-tuning takes 500 steps, using 2x A100s, batch size=8, taking 10 minutes
- Only 3.1M trainable parameters, compatible with any SDXL-based model
- Image cross-attention scaling factor $\gamma = 0.6$

## Key Experimental Results

### Main Results

| Dataset/Setting | Metric | DreamRelation | MS-Diffusion | ReVersion+MS | Gain |
|------------|------|--------------|--------------|--------------|------|
| RelationBench Single-Subject | CLIP-T | **30.6** | 26.5 | 27.8 | +2.8 |
| RelationBench Single-Subject | CLIP-R | **21.4** | 18.8 | 19.3 | +2.1 |
| RelationBench Multi-Subject | CLIP-T | **28.9** | 26.9 | 27.2 | +1.7 |
| RelationBench Multi-Subject | DINO | **62.1** | 58.8 | 59.7 | +2.4 |

### Ablation Study

| Configuration | CLIP-T | CLIP-R | CLIP-I| DINO | Notes |
|------|--------|--------|--------|------|------|
| Full Model | **28.9** | **20.4** | **75.4** | **62.1** | Full Model |
| w/o Relation-aware Data | 27.3 | 19.4 | 75.3 | 59.8 | Using the cropped data engine, CLIP-T drops by 1.6 |
| w/o Local Token Injection | 28.5 | 19.5 | 75.1 | 59.9 | Removing local tokens, DINO drops by 2.2 |
| w/o Keypoint Matching Loss | 27.4 | 19.2 | 75.2 | 61.2 | Removing KML, CLIP-T drops by 1.5 |

### Key Findings
- **Data engine contributes the most**: Removing the relation-aware data engine leads to a significant decrease in both CLIP-T and CLIP-R, proving that decoupling identity and relation in data construction is the key to success.
- **KML is crucial for relation generation**: Removing KML leads to the largest drop in CLIP-R (-1.2), demonstrating that keypoint-level pose supervision directly impacts relation accuracy.
- **Local tokens prevent subject confusion**: Removing local tokens drops DINO from 62.1 to 59.9, showing that local features are particularly important for maintaining identity consistency in overlapping scenarios.
- The method can be directly transferred to SDXL base models, and is also effective for text-to-image relation generation.

## Highlights & Insights
- **Incredibly clever data decoupling strategy**: Utilizing DALL-E 3's multi-turn dialogue capabilities to generate image pairs with identical identities but distinct poses naturally decouples identity and relation. This data construction strategy can be generalized to any generation task that requires decoupling multiple attributes.
- **Latent space keypoint constraints**: KML operates in the VAE latent space rather than pixel space, which maintains spatial alignment with the diffusion loss, thereby avoiding gradient mismatch. Introducing structured constraints in latent space can be extended to tasks like pose control and human generation.
- **Only 3.1M trainable parameters**: By only adding LoRA to the text cross-attention layers, extremely lightweight fine-tuning is achieved while maintaining compatibility with all SDXL models—requiring only 10 minutes of training.

## Limitations & Future Work
- **Limited subject categories**: The data engine relies on DALL-E 3 to generate common categories (such as animals, dolls); its capability to preserve identities of rare objects has not been verified.
- **Limited relation types**: The 25 relations in the experiments are mostly spatial and interaction relations; generalization to more abstract relations (e.g., "protecting", "chasing") remains unknown.
- **Two-subject limitation**: Although qualitative results for three subjects are presented, quantitative evaluation is limited to two subjects; the stability of multi-subject expansion needs further validation.
- **Evaluation metric limitations**: CLIP-R only extracts relation words to calculate similarity, which might not be fully accurate for evaluating complex relations.

## Related Work & Insights
- **vs MS-Diffusion**: MS-Diffusion uses bounding boxes to guide multi-subject generation, but cannot handle subject confusion when boxes overlap, nor does it consider relations. DreamRelation extends this with relation learning capabilities, resolving overlapping confusion via KML and local tokens.
- **vs ReVersion**: ReVersion learns relation embeddings via textual inversion but relies on co-occurrence images and cannot customize subject identities. DreamRelation supports both identity preservation and relation generation via data decoupling.
- **vs ADI**: ADI only handles relation generation without supporting customization. DreamRelation is the first approach to simultaneously support single/multi-subject customization and relation generation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define the "relation-aware customized generation" task and propose a complete solution, with a novel data engine decoupling strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Proposes the RelationBench evaluation suite with thorough ablations, but lacks user studies and comparisons with more baselines.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, logically sound method motivations, and high-quality illustrations.
- Value: ⭐⭐⭐⭐ Fills the gap between customized generation and relation generation, offering practical value for multi-subject interaction scene generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] LoRACLR: Contrastive Adaptation for Customization of Diffusion Models](loraclr_contrastive_adaptation_for_customization_of_diffusion_models.md)
- [\[CVPR 2025\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)
- [\[CVPR 2025\] Visual Persona: Foundation Model for Full-Body Human Customization](visual_persona_foundation_model_for_full-body_human_customization.md)

</div>

<!-- RELATED:END -->
