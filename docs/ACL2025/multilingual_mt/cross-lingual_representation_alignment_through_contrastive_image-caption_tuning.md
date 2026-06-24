---
title: >-
  [Paper Note] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-Lingual Alignment] This paper explores a method for cross-lingual representation alignment without parallel corpora. By performing contrastive learning on multilingual image-caption pairs (CLIP-style), text representations in different languages are implicitly aligned in a shared visual space. It demonstrates that even languages unseen during the pre-training of the encoder (such as Quechua) can be integrated into the align…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Alignment"
  - "Contrastive Learning"
  - "Image-Text"
  - "Low-Resource Languages"
  - "Multilingual Representations"
date: 2026-05-08
content_hash: 92c82a63bb663374
---

# Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning

**Conference**: ACL 2025  
**arXiv**: [2505.13628](https://arxiv.org/abs/2505.13628)  
**Code**: Yes (github.com/nkrasner/cl-clip-align)  
**Area**: Multilingual Translation  
**Keywords**: Cross-Lingual Alignment, Contrastive Learning, Image-Text, Low-Resource Languages, Multilingual Representations

## TL;DR

This paper explores a method for cross-lingual representation alignment without parallel corpora. By performing contrastive learning on multilingual image-caption pairs (CLIP-style), text representations in different languages are implicitly aligned in a shared visual space. It demonstrates that even languages unseen during the pre-training of the encoder (such as Quechua) can be integrated into the alignment framework using this approach.

## Background & Motivation

Encoder language models remain the mainstream approach for extracting textual semantic information. However, the internal representations of multilingual encoders (such as XLM-R) are often "disjointed"—the representation of a sentence in language A may bear little resemblance to its translation in language B. This misalignment primarily stems from inter-lingual imbalances and domain mismatches in the pre-training data.

Traditional cross-lingual alignment methods almost exclusively rely on parallel corpora (bitext), the acquisition of which is both expensive and time-consuming for low-resource languages. In contrast, **describing an image is far easier than translating a text**—speakers of any language can generate captions for an image. Language documentation efforts also frequently yield multimedia materials with monolingual audio or transcriptions.

Thus, the core hypothesis is: **images can serve as a bridge modality for cross-lingual alignment; if multilingual texts are aligned with the same image, they will also align implicitly with each other**.

## Method

### Overall Architecture

The text encoder (XLM-Roberta-Large) and the vision encoder (ViT-Base) are connected via contrastive learning. Given image-caption pairs, the model is trained using a standard CLIP-style contrastive loss:

$$S = E_c \cdot E_i^\top \ast t$$
$$L(E_i, E_c) = \text{CrossEntropy}(S, I)$$

where $E_c$ and $E_i$ represent the text and image representations respectively, $t$ is a learnable temperature parameter, and $I$ is an identity matrix.

### Key Designs

1. **Comparative Experiments with Four Data Configurations**:

    - **Eng-only**: Monolingual English MS-COCO image-caption pairs (118k)
    - **Eng-Pivot**: English captions + alternated Spanish/Japanese/Hindi translated captions (text-to-text alignment, similar to LaBSE)
    - **Multilingual**: Each image is alternated with a caption in one of English, Spanish, Japanese, or Hindi (text-to-image alignment)
    - **Multilingual+Quechua**: Quechua (a language unseen during encoder pre-training) is added on top of the Multilingual configuration

2. **Encoder Architecture Processing**: Since XLM-R and ViT have different hidden dimensions, linear layers are added after their respective outputs to map them to a 512-dimensional joint space. For the first half epoch, only the linear layers are trained (warm-up), after which the encoders are unfrozen for full-parameter fine-tuning.

3. **Handling of Unseen Languages During Pre-training**: The pre-training of XLM-R does not include Indigenous languages of Latin America like Quechua. By including Quechua image captions in the training data rotation, whether unseen languages can be integrated into the aligned representation space is explored.

### Loss & Training

- Standard contrastive learning loss (cross-entropy with temperature scaling)
- Employs the MS-COCO dataset (118k English image-caption pairs)
- Generates Spanish/Japanese/Hindi/Quechua translations via Google Translate
- Linear layer warm-up -> full-parameter fine-tuning

## Key Experimental Results

### Experiment 1: Bitext Retrieval Accuracy (Flores-200 Dataset)

| Encoder | All (203 languages) | XLM-R Seen (92) | Unseen (111) | Quechua |
|--------|-------------|-------------------|-------------|---------|
| XLM-R (Original) | 0.5 | 0.6 | 0.4 | 0.5 |
| Eng-Only | 18.3 | 27.5 | 10.7 | 7.2 |
| Eng-Pivot (Text-Text) | **62.2** | **92.6** | **37.1** | 13.1 |
| Multilingual (Text-Image) | 55.7 | 82.2 | 33.7 | 18.0 |
| Multilingual+Quechua | 50.4 | 76.6 | 28.6 | **29.2** |

### Experiment 3: XNLI Cross-lingual NLI Accuracy (Selected Languages)

| Encoder | en | es | hi | de | zh | ar | Average (12 languages) |
|--------|----|----|----|----|----|----|-------------|
| XLM-R | 50 | 44 | 44 | 43 | 44 | 45 | 43.8 |
| Eng-Only | 53 | 50 | 46 | 49 | 48 | 47 | 48.0 |
| Eng-Pivot | **67** | **65** | **60** | **64** | **62** | **61** | **61.8** |
| Multilingual | 55 | 52 | 51 | 52 | 51 | 51 | 51.3 |
| +Quechua | 56 | 53 | 51 | 53 | 51 | 51 | 51.6 |

### Key Findings

1. **Text-to-Image Alignment Enables Text-to-Text Alignment**: Although the Multilingual model is only trained on text-to-image alignment, it achieves a 55.7% accuracy in the bitext retrieval task, vastly outperforming the 0.5% of the original XLM-R. While it lags behind direct text-to-text alignment (62.2%), the result is highly significant.

2. **Unseen Languages Can Be Integrated**: After adding Quechua captions, the bitext retrieval accuracy for Quechua substantially increases from 18.0% to 29.2%, while the performance on other languages remains largely unaffected (the minor overall drops are primarily due to the reduction in data volume for other languages to maintain comparable data scales).

3. **Downstream Task Quality Improves Rather Than Declines**: On the XNLI cross-lingual NLI task, the text-image aligned encoder outperforms the original XLM-R (51.3 vs. 43.8), demonstrating that image alignment does not overwrite text features useful for downstream tasks; instead, it yields improvements.

4. **Adding Quechua Even Benefits NLI in Other Languages**: The English NLI score of Multilingual+Quechua (56) is higher than that of Multilingual (55), and is comparable or better across almost all languages. Broader language coverage seems beneficial for NLI tasks.

5. **Limited Effect of Monolingual English Training**: Although Eng-Only increases the retrieval accuracy from 0.5% to 18.3%, it falls far short of multilingual training, indicating that cross-lingual alignment requires multilingual signals.

## Highlights & Insights

- **Simplicity and Effectiveness**: The overall approach requires only image-caption pairs (monolingual captions are sufficient) without any parallel corpora. For low-resource languages, annotating image captions is far easier than obtaining parallel translation corpora.
- **Images as a Semantic Bridge**: This hypothesis is validated by the experiments—descriptions of the same image in different languages naturally align text representations in a shared visual space. While it does not match direct text-to-text alignment, it serves as a highly practical bootstrapping approach for low-resource languages.
- **Intuitive t-SNE Visualization**: The alignment effect is clearly demonstrated visually, moving from the distinct language-based clusters in the original XLM-R to translated sentence pairs being close to each other after multilingual image alignment.

## Limitations & Future Work

- Although multilingual image-text alignment is effective, a performance gap remains compared to parallel corpora-based methods (55.7% vs. 62.2% bitext retrieval accuracy).
- The translations were generated using Google Translate, which introduces translation quality bias (especially for Quechua, where translation quality may be poorer).
- A slight decline in overall performance occurred after adding Quechua, making it difficult to clearly decouple the impact of reduced per-language data volume from the introduction of the new language.
- Generalization to other model architectures (e.g., mBERT, larger XLM-R) remains unverified.
- Only one downstream task (NLI) was tested; broader task coverage (e.g., NER, sentiment analysis) would better validate the generalizability.
- Utilizing human-annotated non-English captions instead of machine translation could be considered.

## Related Work & Insights

- Direct comparison with LaBSE (text-to-text contrastive alignment) highlights both the feasibility and the performance gaps of using images as a bridge.
- Muraoka et al. (2023) introduced image representations into NLU task inputs to improve cross-lingual transfer, supporting the core hypothesis of this work.
- For the field of language documentation—where many endangered languages have illustrated text materials but lack parallel translations—this approach can serve as a vital first step in building NLP tools.
- Integrating stronger visual encoders (e.g., SigLIP, EVA-CLIP) or larger image-caption datasets could help close the gap with parallel corpora-based methods.

## Rating

| Dimension | Rating (1-5) |
|------|-----------|
| Novelty | 3.5 |
| Experimental Thoroughness | 3.5 |
| Writing Quality | 4 |
| Value | 3.5 |

The research direction is clear and the hypothesis is novel (images as a cross-lingual bridge), but the method itself (CLIP-style contrastive learning) is relatively standard. The experimental setup is reasonable but limited in scale (only 118k data, 4-5 languages, 1 downstream task). As a short paper (ACL Findings), the quality is sound.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](modular_sentence_encoders.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2025\] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)

</div>

<!-- RELATED:END -->
