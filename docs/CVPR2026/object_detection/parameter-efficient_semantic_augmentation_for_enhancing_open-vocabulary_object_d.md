---
title: >-
  [Paper Note] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection
description: >-
  [CVPR 2026][Object Detection][open-vocabulary object detection] HSA-DINO proposes a multi-scale prompt bank that learns hierarchical semantic prompts from the image feature pyramid to enrich text representations…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "open-vocabulary object detection"
  - "parameter-efficient fine-tuning"
  - "semantic augmentation"
  - "prompt bank"
  - "domain adaptation"
date: 2026-05-08
content_hash: dbf9f1503856128b
---

# Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection

**Conference**: CVPR 2026
**arXiv**: [2604.04444](https://arxiv.org/abs/2604.04444)
**Code**: N/A
**Area**: Object Detection / Open-Vocabulary
**Keywords**: open-vocabulary object detection, parameter-efficient fine-tuning, semantic augmentation, prompt bank, domain adaptation

## TL;DR

HSA-DINO proposes a multi-scale prompt bank that learns hierarchical semantic prompts from the image feature pyramid to enrich text representations, and employs a semantics-aware router to dynamically determine at inference time whether domain-specific augmentation should be applied. This design achieves a superior balance between domain adaptation and open-vocabulary generalization, attaining the best harmonic mean (H) scores across three vertical-domain datasets.

## Background & Motivation

1. **Background**: Open-vocabulary object detection (OVOD) has achieved impressive zero-shot detection performance in general scenarios (e.g., OV-COCO), driven by large-scale pre-training methods such as GLIP, Grounding DINO, and OV-DINO.

2. **Limitations of Prior Work**: (a) Pre-trained OVOD models suffer severe performance degradation on vertical domains (e.g., insect classification ArTaxOr, remote sensing DIOR, underwater UODD), because fine-grained categories are scarce and semantically weak in pre-training data. (b) Full fine-tuning improves target-domain performance but substantially impairs generalization to the general domain (e.g., OV-DINO's mAP_coco drops from 50.6 to 36.1 after fine-tuning on ArTaxOr). (c) Existing prompt methods (predefined templates, CoOp) lack multi-faceted visual-semantic descriptions.

3. **Key Challenge**: A fundamental tension exists between domain adaptation and open-vocabulary generalization — parameter updates targeting downstream tasks inevitably corrupt pre-trained semantic knowledge.

4. **Goal**: Within a parameter-efficient fine-tuning framework, the paper seeks to (a) enrich text representations with dense visual semantics to improve alignment, and (b) automatically select an appropriate semantic strategy at inference time so that domain adaptation does not degrade open-vocabulary capability.

5. **Key Insight**: The multi-scale feature pyramid of an OVOD model inherently encodes hierarchical semantics ranging from coarse to fine (e.g., high-level context such as "flower" vs. low-level texture such as "spotted wings"), which can serve as auxiliary prompts for category labels. Moreover, explicitly modeling content and domain information enables a more accurate router that resolves the difficulty of distinguishing domain distributions.

6. **Core Idea**: Prompt representations selected via multi-scale visual features are used to augment the text representations of category labels, coupled with a semantics-aware router that explicitly disentangles content and domain information, enabling dynamic switching between augmented and original semantic strategies at inference time.

## Method

### Overall Architecture

HSA-DINO is built upon the OV-DINO architecture. During training, LoRA is integrated into the image encoder to learn domain-specific visual features. For each training image, multi-scale feature maps retrieve relevant prompts from the MSPB; these prompts are concatenated with category label embeddings and fed into the text encoder. The detector fuses image features, text features, and detection queries to produce predictions. At inference time, the SAR determines whether to use domain-augmented semantics or the original pre-trained semantics based on the input image.

### Key Designs

1. **Multi-Scale Prompt Bank (MSPB)**:

    - **Function**: Serves as a bridge between the visual and text encoders, learning domain-specific prompts from hierarchical image semantics to augment text representations.
    - **Mechanism**: Maintains $N$ (key, prompt) pairs $\{(\mathbf{k}_i, \mathbf{P}_i)\}_{i=1}^N$, where each key $\mathbf{k}_i \in \mathbb{R}^D$ shares the same dimensionality as the image features and each prompt $\mathbf{P}_i \in \mathbb{R}^{D \times M}$ consists of $M$ learnable vectors. Feature maps are extracted at $S$ scales, globally average-pooled, and matched against all keys via cosine similarity; the best-matching key and its corresponding prompt are selected at each scale. The $S$ selected prompts are concatenated with the category label embedding: $\mathbf{t}_p^k = \text{concat}(\mathbf{P}_1; ...; \mathbf{P}_S; [\text{CLS}]_k)$, and the result is fed into the text encoder.
    - **Design Motivation**: Feature maps at different scales capture semantics at different granularities (high-level context vs. low-level texture). The key–value matching in the prompt bank allows the text encoder to access multi-level visual descriptions relevant to the image content, providing richer conditioning than fixed templates or single-scale global features.

2. **Semantics-Aware Router (SAR)**:

    - **Function**: Dynamically determines at inference time whether the input belongs to the downstream or general domain, and selects the corresponding semantic strategy accordingly.
    - **Mechanism**: Given input image $\mathbf{x}$, features $\tilde{f}$ are extracted; their mean $\mu$ and standard deviation $\sigma$ are computed as domain statistics $\mathcal{D} = \{\mu, \sigma\}$. The domain component is removed to obtain the content embedding $c = \frac{\tilde{f} - \mu}{\sigma + \epsilon}$. An autoencoder reconstructs $c$ as $\hat{c}$, and the domain statistics are added back: $\hat{f} = \hat{c} \cdot \sigma + \mu$. The reconstruction error $d_{err} = |\hat{f} - \tilde{f}|^2$ is compared against threshold $\tau$: if $d_{err} < \tau$, domain-augmented semantics are used; otherwise, pre-trained semantics are used.
    - **Design Motivation**: Training an autoencoder directly on image features (as in DDAS/MoEAdapter4CL) leads to highly overlapping reconstruction errors across domains, causing routing confusion. By explicitly disentangling content and domain statistics and reconstructing only the content part, the distribution overlap is substantially reduced, improving routing accuracy.

3. **LoRA Integration and Auxiliary Losses**:

    - **Function**: Efficiently learns domain-specific visual features while optimizing the prompt bank.
    - **Mechanism**: LoRA is integrated exclusively into the image encoder (leaving the text encoder intact) to learn hierarchical domain visual features. Two auxiliary losses are introduced: a matching loss $\mathcal{L}_m = \sum_{s=1}^S (1 - \gamma(\tilde{\mathbf{z}}^s, \mathbf{k}_{i_s}))$ that pulls selected keys closer to the corresponding scale image features, and an orthogonality loss $\mathcal{L}_p = \frac{1}{N(N-1)} \sum |\langle \mathbf{P}_i, \mathbf{P}_j \rangle|$ that encourages semantic diversity among prompts.
    - **Design Motivation**: The matching loss ensures that keys learn domain knowledge from image samples; the orthogonality loss prevents prompts from collapsing into homogeneous representations.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_{DINO} + \lambda_m \mathcal{L}_m + \lambda_p \mathcal{L}_p$
- $\mathcal{L}_{DINO}$ comprises focal loss, regression loss, GIoU loss, and denoising loss.
- The SAR autoencoder is trained separately with an MSE reconstruction loss for 24 epochs.
- Hyperparameters: $N=10,\ M=12,\ S=3,\ \tau=0.039,\ \lambda_m=0.7,\ \lambda_p=0.3$.
- Fine-tuning: 24 epochs, batch size 16, AdamW with lr = 1e-3.

## Key Experimental Results

### Main Results

Comparison of the harmonic mean (H) between downstream task performance and OV-COCO:

| Method | ArTaxOr mAP_tgt/mAP_coco/H | DIOR H | UODD H |
|------|---------------------------|--------|--------|
| ZiRa (PEFT) | 81.5/44.1/**57.2** | 49.9 | 46.5 |
| OV-DINO (PEFT) | 78.5/24.0/36.8 | 22.1 | 47.6 |
| **HSA-DINO** | 76.8/49.9/**60.5** | **53.0** | **49.6** |

OV-COCO+ extended evaluation:

| Method | w ArTaxOr | w DIOR | w UODD |
|------|-----------|--------|--------|
| ZiRa | 46.9 | 44.4 | 46.0 |
| **HSA-DINO** | **52.3** | **50.1** | **50.5** |

### Ablation Study

Component-wise contributions on the ArTaxOr dataset:

| V-LoRA | MSPB | SAR | mAP_tgt | mAP_coco | H |
|:------:|:----:|:---:|---------|----------|------|
| ✗ | ✗ | ✗ | 1.4 | 50.6 | 2.7 |
| ✓ | ✗ | ✗ | 61.6 | 22.7 | 33.2 |
| ✓ | ✓ | ✗ | 79.1 | 0.5 | 1.0 |
| ✓ | ✗ | ✓ | 59.5 | 50.4 | 54.6 |
| ✓ | ✓ | ✓ | **76.8** | **49.9** | **60.5** |

### Key Findings

- **MSPB substantially improves domain adaptation**: Adding MSPB raises mAP_tgt from 61.6 to 79.1 (+17.5), but severely degrades general-domain performance (mAP_coco drops from 22.7 to 0.5).
- **SAR is the key to balance**: Incorporating SAR recovers mAP_coco from 0.5 to 49.9 (close to the pre-trained level of 50.6), while mAP_tgt decreases only marginally to 76.8.
- SAR surpasses DDAS by 8.2 H_mean points (54.4 vs. 46.2), because explicit content–domain disentanglement substantially reduces reconstruction error overlap.
- Comparison of text semantic augmentation strategies: MSPB (54.4) > AttriCLIP (53.0) > CoOp (52.1) > predefined templates (49.9).
- Optimal hyperparameters: bank size $N=10$, prompt length $M=12$, routing threshold $\tau=0.039$.

## Highlights & Insights

- **"Augment but switchable" design philosophy**: Rather than pursuing a single model that generalizes across all domains, the method trains domain-specific augmentation and dynamically switches via the router. This sidesteps the fundamental conflict between adaptation and generalization, offering a practical and elegant solution.
- **Content–domain disentanglement for routing**: By applying instance normalization to separate content and domain statistics before reconstruction, the method significantly reduces distribution overlap compared to DDAS, which applies the autoencoder directly to image features. This idea is transferable to other scenarios requiring domain-aware routing.
- **Multi-scale prompt bank as a visual–textual bridge**: Enabling the text encoder to access multi-scale visual semantic information from the image is more expressive than global features combined with fixed templates.

## Limitations & Future Work

- The SAR threshold $\tau$ is a fixed scalar (0.039); different downstream domains may have different optimal thresholds in theory, although the paper demonstrates that a unified SAR is still effective.
- Prompt selection in the MSPB relies on globally average-pooled scale features, discarding spatial local information.
- Each fine-tuning run trains a dedicated MSPB and SAR for a single downstream task; multiple downstream tasks require multiple training runs.
- Potential directions: exploring unified prompt banks under multi-task joint training; using finer-grained regional features (e.g., RoI features) to guide prompt selection.

## Related Work & Insights

- **vs. ZiRa**: ZiRa employs dual-norm penalties to constrain the residual detection branch for continual learning. Its H score is slightly lower than HSA-DINO, with a larger gap on DIOR (49.9 vs. 53.0), suggesting that norm constraints are less flexible than dynamic routing.
- **vs. CoOp/AttriCLIP**: These methods rely on single-scale global features for prompt selection, which is less semantically rich than the multi-scale prompt bank.
- **vs. MR-GDINO**: The memory-and-retrieval mechanism preserves some pre-trained knowledge but reduces mAP_coco to 0.1 on UODD, almost entirely sacrificing generalization capability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-scale prompt bank and content–domain disentangled router are novel designs, though the overall framework is a composition of PEFT and routing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three vertical domains + OV-COCO + OV-COCO+ extended evaluation; ablations are thorough and visualizations are rich.
- **Writing Quality**: ⭐⭐⭐⭐ Technical descriptions are detailed and clear; figures are intuitive; motivation is well argued.
- **Value**: ⭐⭐⭐⭐ Addresses the practical problem of domain adaptation vs. generalization in OVOD; the use of H as a comprehensive evaluation metric also offers methodological reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[CVPR 2026\] HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](herod_heuristic_inspired_reasoning_data_efficient_rod.md)
- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)

</div>

<!-- RELATED:END -->
