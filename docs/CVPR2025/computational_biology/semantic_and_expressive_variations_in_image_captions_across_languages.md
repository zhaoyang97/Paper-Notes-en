---
title: >-
  [Paper Note] Semantic and Expressive Variation in Image Captions Across Languages
description: >-
  [CVPR 2025][Computational Biology][Multilingual Image Captioning] This work systematically demonstrates significant distributional differences in semantic content (objects, relations, attributes) and expressive style (concreteness, tone, authenticity) in image captions across different languages. Multilingual caption sets provide richer visual information compared to monolingual ones (+46% objects, +66.1% relations, +66.8% attributes), providing empirical support for training…
tags:
  - "CVPR 2025"
  - "Computational Biology"
  - "Multilingual Image Captioning"
  - "Semantic Variation"
  - "Multilingual VLM Training"
  - "Scene Graph Analysis"
  - "Cross-cultural Visual Perception"
date: 2026-05-08
content_hash: 9fb486e75fb67b72
---

# Semantic and Expressive Variation in Image Captions Across Languages

**Conference**: CVPR 2025  
**arXiv**: [2310.14356](https://arxiv.org/abs/2310.14356)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Multilingual Image Captioning, Semantic Variation, Multilingual VLM Training, Scene Graph Analysis, Cross-cultural Visual Perception

## TL;DR

This work systematically demonstrates significant distributional differences in semantic content (objects, relations, attributes) and expressive style (concreteness, tone, authenticity) in image captions across different languages. Multilingual caption sets provide richer visual information compared to monolingual ones (+46% objects, +66.1% relations, +66.8% attributes), providing empirical support for training vision models on multilingual data.

## Background & Motivation

Currently, vision-language models are primarily trained on English image-text pairs, and non-English data is often filtered out during preprocessing. However, extensive research in cross-cultural psychology shows that people from different linguistic and cultural backgrounds systematically perceive and describe the same visual scene differently:

- Americans tend to describe focal objects and their attributes, while Japanese speakers tend to describe relationships between objects.
- The complex morphosyntactic system of German provides a refined understanding of spatial relationships.
- Russian verbs of motion require speakers to specify directionality, mode of transportation, etc.

Are these differences reflected in visual datasets and model behaviors? If so, training models exclusively on English data may miss unique visual concepts brought by other languages. This paper hypothesizes that **the linguistic bias of training data restricts the breadth of the visual world that a model "sees"**.

## Method

### Overall Architecture

The study revolves around a core question: Do multilingual caption sets cover more and more diverse visual information than monolingual caption sets? This is measured along two dimensions: **semantic variation** (scene graph size = number of objects, relations, and attributes) and **expressive variation** (concreteness, analytic style, clout, authenticity, tone, and embedding space coverage). The analysis is conducted on human-annotated data (Crossmodal dataset) and model-generated data (LLaVA, Vertex API).

### Key Designs

1. **Scene Graph-based Semantic Variation**:
    - **Function**: Quantify the differences in "what is being said" across descriptions in different languages.
    - **Mechanism**: For each image $i$, a monolingual scene graph $\text{mono}_i^l$ and a multilingual scene graph $\text{multi}_i^L$ are constructed. FLAN-T5 is used to parse captions into a scene graph $\mathcal{G} = \text{SG}(c)$, containing $(object, attribute)$ and $(subject, predicate, object)$ tuples. Semantic concepts are normalized using WordNet path similarity and cosine similarity to merge different texts referring to the same concept. Finally, $\mathbb{E}[M(\bigcup \text{SG}(\text{mono}_i^l))]$ is compared with $\mathbb{E}[M(\bigcup_{l \in L} \text{SG}(\text{multi}_i^L))]$.
    - **Design Motivation**: Scene graphs are standard representations for measuring the content of visual descriptions, and the size of the deduplicated and merged scene graph directly reflects information coverage. An LLM-based parser (instead of traditional syntactic parsing) is utilized because descriptions translated from multiple languages often contain complex semantic relations.

2. **LIWC-based Expressive Variation**:
    - **Function**: Quantify the differences in "how it is said" across descriptions.
    - **Mechanism**: Five linguistic metrics (concreteness, analytic, clout, authenticity, tone) + embedding space coverage are utilized. Expressive coverage is defined as $C_M(\mathcal{T}) = \max(M(\mathcal{T})) - \min(M(\mathcal{T}))$, and the embedding space uses the maximum pairwise cosine distance. The coverage of multilingual sets is compared against monolingual sets.
    - **Design Motivation**: Even if two descriptions "say the same thing", differences in expressive style (e.g., more concrete vs. more abstract, more analytical vs. more narrative) can provide models with information from different perspectives.

3. **Translation for Fair Comparison**:
    - **Function**: Eliminate confounding factors caused by differences in linguistic tools.
    - **Mechanism**: GPT-4 is used to translate all foreign-language captions into English, analyzing content differences on a "common linguistic ground". Human evaluation shows a translation fidelity of $\mu = 4.68/5.0$, with 98.42% of visually important information faithfully preserved.
    - **Design Motivation**: Parser, embedding, and tokenizer tools possess language specificity or linguistic bias, making direct cross-lingual comparisons unfair. Translating all descriptions into the same language allows for the comparison of deep semantic content differences rather than surface-level linguistic variations.

### Loss & Training

This is an analytical work and does not propose new training objectives. In the fine-tuning experiments, the GIT model is fine-tuned on the Crossmodal training set using standard image captioning loss (cross-entropy), with the SPICE F1-score as the evaluation metric (to measure conceptual overlap rather than superficial syntactic alignment).

## Key Experimental Results

### Main Results (XM Dataset, Human Annotated)

| Metric | 3×English | 3×German | 3×Japanese | avg mono | en-fr-zh (multi) | avg multi | Gain |
|------|-----------|----------|------------|----------|------------------|-----------|------|
| Objects | 2.59 | 3.16 | 3.41 | 2.98 | 3.71 | 4.35 | +46.0% |
| Relations | 1.54 | 1.94 | 1.99 | 1.77 | 2.41 | 2.94 | +66.1% |
| Attributes | 1.27 | 1.97 | 2.47 | 1.78 | 2.36 | 2.97 | +66.8% |
| Tone range | 8.62 | 9.74 | 9.18 | 10.04 | 13.78 | 15.40 | +53.4% |
| Embedding cov. | .38 | .43 | .42 | .42 | .54 | .52 | +23.8% |

### Model-Generated Captions (LLaVA & Vertex)

| Source | Metric | avg mono | avg multi | Gain |
|------|------|----------|-----------|------|
| LLaVA | Objects | 4.78 | 5.93 | +24.1% |
| LLaVA | Relations | 3.95 | 4.54 | +14.9% |
| LLaVA | Embeddings | .29 | .47 | +62.1% |
| Vertex | Objects | 3.48 | 4.17 | +19.8% |
| Vertex | Relations | 2.77 | 3.40 | +22.7% |

### Fine-Tuning Experiments (SPICE F1-score, Vertex)

| Fine-Tuning Lang → Eval Lang | en | de | fr | zh | multi |
|---------------------|-----|-----|-----|-----|-------|
| en | **.225** | .213 | .248 | .199 | .230 |
| de | .229 | **.234** | .240 | .202 | .219 |
| zh | .218 | .215 | .236 | **.242** | .216 |
| multi | .230 | .226 | .253 | .224 | **.235** |

### Key Findings

- **Multilingual caption sets systematically cover more visual concepts**: On average, scene graphs from any two languages share only 63.1% of objects and 39.5% of relations.
- **Not an artifact of "language models"**: Variations in multilingual captions from a single model are close to those of English captions generated across multiple different models (92.4% objects, 98.4% relations), indicating that the differences stem from the languages themselves rather than model switching.
- **Validated via Visual Genome**: The multilingual caption set achieves 23.9% higher recall on VG-annotated objects than its monolingual counterparts, proving that these differences are not "hallucinations."
- **Models internalize language-specific distributional features**: Models fine-tuned on language $X$ perform best on the test set of language $X$, while multilingually fine-tuned models archive consistently strong performance across all languages.

## Highlights & Insights

- **Redefining the "Curse of Multilinguality"**: Shifting perspective from "multilinguality degrades monolingual performance" to "multilinguality provides richer visual concepts," offering a new justification for multilingual training.
- **Translation does not equal native content**: Translating English into other languages misses out on the specific content distributions of "native languages," explaining why models trained on native multilingual data outperform those trained on translated data.
- **Warning against "perceptual monoculture"**: Dominance of English-led modeling might cause vision models to only internalize the perceptual preferences of English speakers.
- **Innovation in research methodology**: The analytical framework using scene graph union, normalization, and coverage measurement can be widely applied to other multimodal data analysis tasks.

## Limitations & Future Work

- Only 7 languages (all major ones) are analyzed. Low-resource languages and more diverse language families remain unexamined.
- The dataset size is moderate (XM has ~3.6K images), leaving the robust validation of conclusions on larger scales for future work.
- Translation relies heavily on GPT-4. Despite high human-evaluated fidelity, translation could still introduce minor information loss.
- Fine-tuning experiments are restricted to the GIT model; the generalization to larger, modern models (such as InternVL) remains unexplored.
- Cross-effects between culture and language are not fully disentangled (e.g., people speaking the same language across different cultures might also exhibit differences).

## Related Work & Insights

- Connection with multilingual VLMs like PaLI-X: This study provides theoretical support for "why we should train on multilingual data."
- Research on visual perception differences in cross-cultural psychology offers an essential perspective to the computer vision community.
- Insight: Contrastive learning frameworks could be designed to leverage stylistic and semantic variations across multilingual descriptions of the same image to learn richer visual representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically quantify semantic and expressive variations in multilingual image captions with a unique perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive analysis covering both human annotations and model generation, two dimensions of semantic and expressive traits, Visual Genome control, and fine-tuning verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous logical reasoning where every step includes verification showing "why these variances are not illusions"; the methodology is highly instructive.
- **Value**: ⭐⭐⭐⭐ Provides an empirical foundation for multilingual vision-language research, influencing training data strategies and model design decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[ICML 2025\] Weisfeiler and Leman Go Gambling: Why Expressive Lottery Tickets Win](../../ICML2025/computational_biology/weisfeiler_and_leman_go_gambling_why_expressive_lottery_tickets_win.md)
- [\[CVPR 2026\] FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics](../../CVPR2026/computational_biology/feast_fully_connected_expressive_attention_for_spatial_transcriptomics.md)
- [\[ICCV 2025\] Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines](../../ICCV2025/computational_biology/integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov.md)
- [\[ICLR 2026\] Automatic Image-Level Morphological Trait Annotation for Organismal Images](../../ICLR2026/computational_biology/automatic_image-level_morphological_trait_annotation_for_organismal_images.md)

</div>

<!-- RELATED:END -->
