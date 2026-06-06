---
title: >-
  [Paper Note] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Federated Learning] This paper proposes FedMVP, which, under a federated learning setting, employs a PromptFormer network to fuse image visual features with LLM-generated category attribute te…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Federated Learning"
  - "CLIP Prompt Learning"
  - "Multimodal Prompting"
  - "Visual Prompt Tuning"
  - "Cross-Domain Generalization"
date: 2026-05-08
content_hash: 51df0b14f18ea205
---

# FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2504.20860](https://arxiv.org/abs/2504.20860)  
**Code**: [https://github.com/mainaksingha01/FedMVP](https://github.com/mainaksingha01/FedMVP)  
**Area**: Multimodal VLM
**Keywords**: Federated Learning, CLIP Prompt Learning, Multimodal Prompting, Visual Prompt Tuning, Cross-Domain Generalization

## TL;DR
This paper proposes FedMVP, which, under a federated learning setting, employs a PromptFormer network to fuse image visual features with LLM-generated category attribute text features, generating dynamic multimodal visual prompts injected into CLIP's visual encoder. FedMVP achieves substantial improvements of 1.57%–2.26% over existing federated prompt learning methods across 20 datasets and three generalization settings.

## Background & Motivation
Federated learning (FL) enables multiple clients to collaboratively train a global model without sharing raw data. Vision-language models (VLMs) such as CLIP are natural candidates for FL owing to their strong generalization ability; however, their large parameter counts incur prohibitive communication overhead. Prompt tuning mitigates this by learning only lightweight prompt tokens to adapt CLIP, requiring communication of only ~0.37% of parameters, making it inherently suitable for FL.

Nevertheless, existing FL prompt learning methods suffer from severe **generalization degradation**:

- **Text prompt tuning (TPT, e.g., PromptFL)**: Learns static context that, once fixed, cannot generalize to unseen categories.
- **Visual prompt tuning (VPT, e.g., FedVPT)**: Similarly limited by static context.
- **Conditioned prompt methods**: FedTPG relies solely on class-name textual information, while FedCoCoOp relies solely on image visual features — under the high heterogeneity of FL, single-modality conditioning is insufficient.

**Key Challenge**: The high data heterogeneity in FL (each client's data spans disjoint classes and domains) demands prompts that generalize across categories and domains, yet existing methods draw conditioning information from only a single modality.

**Core Idea**: **Dual-modality conditioning** — simultaneously exploiting (1) visual features of the input image and (2) LLM-generated textual attribute descriptions of categories to produce prompts, fused via cross-attention. Attributes facilitate generalization to unseen categories (which share attributes with seen ones), while image features facilitate generalization to unseen domains (capturing textures and abstract concepts that attributes cannot describe).

## Method

### Overall Architecture
Each client locally trains a PromptFormer network (the sole trainable module), whose generated multimodal visual prompts are injected into a frozen CLIP visual encoder. After training, only the lightweight PromptFormer parameters are sent to the server for FedAvg aggregation.

### Key Designs
1. **LLM Attribute Generation**:

    - **Function**: GPT-4o is used to generate rich textual attribute descriptions for each class name.
    - **Example**: "giraffe" → "Exceptionally long neck, unique coat pattern with irregular brown patches, ..."
    - **Design Motivation**: Class labels alone carry limited semantic information. Attributes provide fine-grained, category-shared descriptions — the attribute "two legs" for "chicken" can transfer to the unseen class "seagull."

2. **PromptFormer Network**:

    - **Function**: Fuses image patch embeddings and text attribute embeddings via cross-attention to generate multimodal visual prompts.
    - **Core Architecture**:
        - Attribute embedding extraction: $\mathbf{A}_i = \{\mathcal{E}_t(\text{LLM}(c_k))\}_{k=1}^K$
        - Linear projection for dimension alignment: $\mathbf{A}' = T_{\text{proj}}(\mathbf{A})$ (512→768)
        - Cross-attention fusion:
       $$\mathbf{P}(\mathbf{A}', \mathbf{E}) = \text{FFN}(\text{CrossAttention}(\mathbf{Q}_\mathbf{E}, \mathbf{K}_{\mathbf{A}'}, \mathbf{V}_{\mathbf{A}'}))$$
       where $\mathbf{Q}_\mathbf{E} = \mathbf{E}W_\mathbf{Q}$ (image patches as queries), $\mathbf{K}_{\mathbf{A}'} = \mathbf{A}'W_\mathbf{K}$ (attributes as keys/values)
        - 4-head cross-attention + LayerNorm + two-layer FFN
    - **Design Motivation**: Through cross-attention, image patch features learn to attend to relevant attribute features. For instance, patches depicting "legs" attend to the "four legs" attribute — when an unseen class sharing that attribute appears, the prompt naturally encodes the relevant information.

3. **Visual Prompt Injection**:

    - **Function**: Concatenates the generated multimodal prompts $\mathbf{P} \in \mathbb{R}^{m \times d_v}$ ($m=4$) to the input of the visual encoder.
    - **Reformulated input**: $\mathbf{I} = [\mathbf{z}; \mathbf{E}; \mathbf{P}] \in \mathbb{R}^{(1+b+m) \times d_v}$
    - **Design Motivation**: Unlike methods such as FedTPG that inject prompts into the text encoder, visual prompt tuning more directly influences visual feature representations and supports instance-level dynamic prompting (a distinct prompt per image).

4. **Lightweight LoRA Fine-Tuning**:

    - **Function**: When the initial training loss of a client falls below a threshold $\sigma=0.5$, the PromptFormer parameters are frozen and only the injected LoRA matrices are trained.
    - **Design Motivation**: Clients with limited data are prone to overfitting; LoRA reduces trainable parameters by $267\times$ while simultaneously lowering communication overhead.

### Loss & Training
- **CLIP cross-entropy loss**: $\mathcal{L}_{ce} = -\mathbb{E}_{(\mathbf{x},y)} y \log p(y|\mathbf{I})$
- **Consistency loss**: $\mathcal{L}_{con} = 1 - \cos(\mathcal{E}_v(\mathbf{I}), \mathcal{E}_v(\mathbf{x}'))$, constraining representation consistency between two augmented views of the same image.
- **Total loss**: $\mathcal{L}_{total} = \mathcal{L}_{ce} + \alpha \cdot \mathcal{L}_{con}$, $\alpha = 10$
- **Text features**: $\mathbf{t}_k = \mathcal{E}_t([\text{"A photo of [CLASS]"}; \text{LLM}(c_k)])$, concatenating the hand-crafted template with LLM attributes.
- SGD optimizer, learning rate 0.003, weight decay 1e-5, batch size 128, 8-shot per class.

## Key Experimental Results

### Main Results (Base-to-New Generalization, 9 Datasets)

| Method | Local Acc | Base Acc | New Acc | HM |
|------|----------|---------|---------|-----|
| ZS-CLIP | 76.72 | 70.51 | 75.78 | 74.24 |
| PromptFL | 81.75 | 74.47 | 71.70 | 75.74 |
| FedTPG | 80.75 | 73.68 | 76.02 | 76.70 |
| FedMaPLe | 81.63 | 74.44 | 70.62 | 75.29 |
| **FedMVP (Ours)** | **81.89** | **75.37** | **77.82** | **78.27** |
| Gain | +0.14 | +0.90 | **+1.79** | **+1.57** |

### Ablation Study / Domain Generalization

| Setting (DomainBed MSST) | PACS | OfficeHome | VLCS | TerraInc | DomainNet | Avg |
|----------------------|------|-----------|------|---------|----------|------|
| ZS-CLIP | 96.16 | 81.49 | 83.29 | 33.98 | 57.13 | 70.41 |
| FedTPG | 90.99 | 82.78 | 69.77 | 26.79 | 56.82 | 65.43 |
| FedCLIP | 96.29 | 81.74 | 82.70 | 36.58 | 57.85 | 71.03 |
| **FedMVP (Ours)** | **97.28** | **84.15** | **85.12** | **37.36** | **61.17** | **73.02** |
| Gain | +0.99 | +1.37 | +1.83 | +0.78 | **+2.29** | **+1.99** |

| ImageNet Domain Generalization (SSMT) | IN | INV2 | IN-S | IN-A | IN-R | Avg |
|---------------------|------|------|------|------|------|------|
| FedTPG | 69.51 | 62.90 | 47.65 | 49.97 | 76.35 | 59.22 |
| **FedMVP (Ours)** | **70.87** | **63.72** | **50.93** | **51.76** | **77.23** | **60.91** |
| Gain | +1.36 | +0.82 | **+3.28** | +1.43 | +0.74 | **+1.69** |

| Component Ablation | Base-to-New HM | MSST DG Avg |
|---------|---------------|------------|
| ZS-CLIP | 74.24 | 70.41 |
| $f_\Theta$ only | 75.94 | 71.85 |
| $f_\Theta$ + $\mathcal{L}_{con}$ | 76.27 | 72.14 |
| w/o LoRA | 77.41 | 72.58 |
| **FedMVP (Full)** | **78.27** | **73.02** |

### Key Findings
- **Multimodal conditioning is critical**: FedMVP surpasses FedTPG (text-only conditioning) by 1.79% on unseen classes and FedCoCoOp (vision-only conditioning) by 11.82%.
- **Particularly strong gains on IN-Sketch (+3.28%)**: Attributes remain invariant between real images and sketches (e.g., "four legs"), validating the cross-domain transferability of attribute features.
- **Most prompt learning methods underperform ZS-CLIP on domain generalization**: Only FedCLIP and FedMVP exceed the zero-shot baseline, indicating that poorly designed prompt learning can lead to source-domain overfitting.
- **LoRA effectively prevents overfitting**: Removing LoRA (w/o LoRA) causes a 0.86% drop in HM.
- **FedMVP converges approximately 10× faster than FedTPG** (in communication rounds); despite transmitting twice as many parameters per round, total communication cost is lower.
- **Cross-dataset generalization remains the most challenging setting**: FedMVP underperforms ZS-CLIP on OxfordPets and StanfordCars, likely due to attribute overlap among fine-grained categories.

## Highlights & Insights
1. **First to introduce LLM-generated attributes in FL**: Attributes serve as a shared knowledge bridge across categories, effectively promoting cross-category generalization.
2. **Intuitive cross-attention design**: Visual patches serve as queries to retrieve relevant attributes, enabling prompts to automatically attend to shared attributes when encountering new categories.
3. **Visual rather than text prompting**: Contrary to the mainstream approach, FedMVP injects prompts into the visual encoder, enabling instance-level dynamic prompting.
4. **Adaptive LoRA strategy**: Automatically switches between full-parameter and LoRA training based on per-client data volume, balancing performance and overfitting prevention — a fine-grained adaptation to FL heterogeneity.

## Limitations & Future Work
1. Reliance on GPT-4o for attribute generation increases deployment cost and introduces dependency on an external API.
2. Performance falls below ZS-CLIP on fine-grained datasets (OxfordPets, StanfordCars); attribute overlap is a likely root cause, motivating finer-grained attribute design.
3. Only the ViT-B/16 backbone is evaluated; performance on larger models (e.g., ViT-L/14) remains unverified.
4. The LoRA switching threshold $\sigma=0.5$ is manually set; an adaptive threshold may be preferable.
5. Scalability and communication efficiency as the number of clients grows warrant further analysis.

## Related Work & Insights
- **Relation to FedTPG**: FedTPG uses only class names as textual conditioning to generate text prompts; FedMVP additionally incorporates image visual conditioning and LLM attribute conditioning, and injects prompts into the visual encoder rather than the text encoder.
- **Relation to MaPLe**: MaPLe injects prompts into both text and visual encoders but requires unfreezing portions of the encoder weights, making it unsuitable for FL.
- **Relation to LaBo/VFC**: These methods augment CLIP with LLM-generated descriptions but are not designed for FL; FedMVP is the first to introduce this strategy into federated learning.
- **Insight**: In highly heterogeneous distributed settings, **multimodal conditioning information is more effective than any single modality** — suggesting that future FL methods should exploit all available modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multimodal conditioning and visual prompt tuning is novel; the PromptFormer design is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20 datasets, three generalization settings, multiple FL baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables; notation is slightly complex.
- Value: ⭐⭐⭐⭐ Provides a practical solution for federated VLM adaptation with significant cross-domain generalization improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)

</div>

<!-- RELATED:END -->
