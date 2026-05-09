---
title: >-
  [Paper Note] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates
description: >-
  [ACL 2026][LLM Safety][knowledge coordinates] This paper proposes Knowledge Coordinate-conditioned pre-training (KoCo), which maps each document to a three-dimensional semantic coordinate (Source, Content, Stability) and injects this as a natural language prefix during pre-training. This endows the model with explicit context-awareness, yielding performance gains across 10 downstream tasks, approximately 30% faster convergence, and effective hallucination mitigation.
tags:
  - ACL 2026
  - LLM Safety
  - knowledge coordinates
  - conditional pre-training
  - hallucination mitigation
  - data contextualization
  - pre-training acceleration
date: 2026-05-08
content_hash: e909c4a6bd9614a9
---

# KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates

**Conference**: ACL 2026  
**arXiv**: [2604.12397](https://arxiv.org/abs/2604.12397)  
**Code**: N/A  
**Area**: LLM Safety / Pre-training  
**Keywords**: knowledge coordinates, conditional pre-training, hallucination mitigation, data contextualization, pre-training acceleration

## TL;DR

This paper proposes Knowledge Coordinate-conditioned pre-training (KoCo), which maps each document to a three-dimensional semantic coordinate (Source, Content, Stability) and injects this as a natural language prefix during pre-training. This endows the model with explicit context-awareness, yielding performance gains across 10 downstream tasks, approximately 30% faster convergence, and effective hallucination mitigation.

## Background & Motivation

**Background**: Standard LLM pre-training treats corpora as flat token sequences and optimizes the negative log-likelihood of next-token prediction uniformly across all tokens—regardless of whether a token originates from a peer-reviewed theorem or a casual social media post. This stands in sharp contrast to human learning, where readers naturally contextualize information based on its source and role.

**Limitations of Prior Work**: Recent improvements fall into two categories. Metadata-aware pre-training methods (e.g., MeCo) prepend URL prefixes to identify sources, but URLs are overly fine-grained, rely on prior mappings, and lack objectivity. Data selection methods (e.g., ASK-LLM) filter high-quality data via classifiers but operate in a binary fashion—retaining high-quality data and discarding the rest—which diverges from human learning. Humans do not simply "delete" low-quality information; instead, they contextualize it according to its source and nature.

**Key Challenge**: Existing methods either provide overly superficial contextual signals (URLs) or discard "low-quality" data rather than helping the model understand its limitations. A more structured approach is needed to help the model perceive each document's position in the knowledge space.

**Goal**: To design a simple yet effective pre-training method that equips models with human-like context-awareness during pre-training by providing each document with an objective knowledge coordinate description.

**Core Idea**: Inspired by the DIKW (Data–Information–Knowledge–Wisdom) hierarchy, each document is mapped to a three-dimensional semantic space—Source, Content, and Stability—and injected as a natural language prefix during pre-training, enabling the model to distinguish "timeless physical theorems" from "transient social opinions."

## Method

### Overall Architecture

KoCo transforms the standard pre-training objective from $P(x)$ to the conditional distribution $P(x|\mathcal{T})$, where $\mathcal{T} = (s, c, t)$ is the knowledge coordinate triple of a document. The pipeline proceeds as follows: (1) a lightweight language model (Qwen-3-4B) serves as an annotator to predict three-dimensional coordinate labels from each document's URL and text; (2) the labels are concatenated into a natural language prefix (e.g., "Source: Academic; Content: Reference; Stability: Evergreen") and prepended to the original text; (3) during pre-training, loss is computed only over document tokens, with the prefix masked out. The training objective is:

$$\mathcal{L}_{\text{KoCo}} = -\sum_{i=1}^{n} \log P_\theta(x_i | x_{<i}, \mathcal{T})$$

### Key Designs

1. **Three-Dimensional Knowledge Coordinate System**

    - **Function**: Provides each document with an objective meta-description that is independent of specific semantic topics.
    - **Mechanism**: Three orthogonal dimensions are defined—**Source** (provenance: Academic/Media/Community/Personal, etc., 10 categories), **Content** (content type: Instructional/Pedagogical/Discussion/Opinion, etc., 11 categories), and **Stability** (temporal stability: Ephemeral/Decaying/Long-term/Evergreen, 4 categories). Over 99.5% of documents in the DCLM corpus can be successfully mapped to this coordinate system.
    - **Design Motivation**: Unlike the superficial signals provided by URLs, the three-dimensional coordinates characterize the intrinsic properties of information, simulating the human cognitive process of "understanding the source and nature of information." The three dimensions capture complementary information, as confirmed by ablation experiments.

2. **Conditional Inference**

    - **Function**: Guides model behavior at inference time by specifying knowledge coordinate prefixes.
    - **Mechanism**: Task-specific coordinate prefixes are designed for different benchmarks (e.g., {Source: Media; Content: Discussion} for Social IQA; {Source: Academic; Content: Pedagogical} for LogiQA). More critically, specifying reliable source prefixes (e.g., {Source: Publication; Content: Instructional; Stability: Long-term}) yields up to 3.78% improvement on TruthfulQA.
    - **Design Motivation**: KoCo introduces at the pre-training stage control signals typically reserved for the alignment stage, allowing users to suppress unreliable outputs at inference time by simply specifying knowledge coordinates.

3. **Annotator Independence Verification**

    - **Function**: Demonstrates that KoCo's gains stem from the coordinate conditioning mechanism itself rather than from knowledge distillation from the annotator model.
    - **Mechanism**: A BERT-base model with only 110M parameters (far smaller than the 0.6B model being pre-trained) is used as an alternative annotator, trained on 50K labeled samples to generate coordinates for KoCo pre-training.
    - **Design Motivation**: If KoCo's improvements were due to knowledge distillation from Qwen-3-4B, using a weaker annotator should significantly degrade performance. Experiments show comparable results across both annotators, ruling out the knowledge distillation hypothesis.

### Loss & Training

Loss is computed only over document tokens during pre-training, with prefix tokens masked. Pre-training continues from the MeCo 1.6B checkpoint using a 100 GB subset of the DCLM corpus. When training from scratch, KoCo demonstrates approximately 30% convergence acceleration on both 0.3B and 0.6B models.

## Key Experimental Results

### Main Results

Continued pre-training from the MeCo 1.6B checkpoint, evaluated on 10 downstream tasks:

| Method | COPA | ARC-e | ARC-c | CSQA | IFEval | OBQA | PIQA | SIQA | LogiQA | TruQA | Avg |
|--------|------|-------|-------|------|--------|------|------|------|--------|-------|-----|
| MeCo (URL prefix) | 82.0 | 75.4 | 44.4 | 64.0 | 20.0 | 50.8 | 73.0 | 52.9 | 25.5 | 36.3 | 52.42 |
| Standard continued pre-training | 82.0 | 74.6 | 42.8 | 59.5 | 22.2 | 49.6 | 72.9 | 52.7 | 24.9 | 35.2 | 51.64 |
| Data selection | 82.0 | 75.0 | 44.6 | 63.3 | 22.4 | 49.0 | 74.0 | 52.6 | 25.2 | 35.5 | 52.36 |
| **KoCo** | **83.0** | **77.4** | 44.1 | 61.8 | **25.5** | **51.2** | **74.8** | **53.4** | **26.9** | **36.6** | **53.48** |

### Ablation Study

| Setting | ARC-e | ARC-c | OBQA | PIQA | Avg |
|---------|-------|-------|------|------|-----|
| w/o Source | 76.2 | 44.1 | 50.2 | 73.7 | 53.43 |
| w/o Content | 76.6 | 43.6 | 51.2 | 74.1 | 53.46 |
| w/o Stability | 76.7 | 43.1 | 51.0 | 73.8 | 53.32 |
| KoCo (full) | 77.4 | 44.1 | 51.2 | 74.8 | **53.48** |

### Key Findings

- KoCo uses the same data as MeCo (DCLM corpus) and achieves significant performance improvements without introducing additional data, with an average gain of 1.06%.
- Standard continued pre-training actually degrades the MeCo checkpoint's performance, and data selection methods only match it—indicating that simple data manipulation is insufficient and that more structured conditioning signals are required.
- Conditional inference yields a 3.78% improvement on TruthfulQA, far exceeding gains on other tasks, suggesting that the model has learned associations between source reliability and factuality.
- Using unreliable source prefixes (e.g., Personal/x.com + Opinion + Ephemeral) reduces TruthfulQA accuracy to 34.75%, while reliable source prefixes raise it to 40.39%.
- PCA visualization shows that KoCo-trained models clearly separate factual and opinion-based statements in the representation space.

## Highlights & Insights

- **Cognition-Inspired Design**: The three-dimensional knowledge coordinates simulate the human cognitive process of "understanding the source and nature of information," yielding a conceptually simple yet empirically effective design.
- **A New Path for Hallucination Mitigation**: By teaching the model to distinguish reliable from unreliable sources during pre-training, KoCo offers a fundamental approach to reducing hallucinations rather than addressing them post hoc.
- **Bridging Pre-training and Alignment**: KoCo advances control signals from the alignment stage to the pre-training stage, suggesting that certain alignment objectives can be moved upstream to simplify downstream fine-tuning.
- **Retaining Low-Quality Data**: Unlike data selection methods, KoCo retains all data while annotating its properties, enabling the model to learn from all data with an understanding of its context.

## Limitations & Future Work

- Experiments are limited to models of 0.3B–1.6B parameters; effectiveness at larger scales remains to be validated.
- Annotator accuracy (approximately 75–83% agreement with commercial models) introduces noise that may limit coordinate precision.
- The three-dimensional coordinate system is manually designed; whether additional dimensions or automatically discovered coordinates would be more effective remains an open question.
- The approximately 30% pre-training acceleration is verified only on small models trained from scratch; confirmation at larger scales is needed.
- Conditional inference requires users to manually select appropriate coordinate prefixes; automated selection mechanisms warrant further investigation.

## Related Work & Insights

- **vs. MeCo (URL prefix)**: MeCo uses URLs as source identifiers, which are overly fine-grained and rely on prior mappings; KoCo uses structured three-dimensional coordinates to provide more objective and informative conditioning signals.
- **vs. Data Selection Methods**: Data selection adopts a binary "retain/discard" strategy; KoCo retains all data while annotating their properties, enabling the model to understand information of varying quality in context.
- **vs. RLHF/SFT Alignment**: KoCo introduces controllability at the pre-training stage, offering a new paradigm of "upstream alignment."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of knowledge coordinates is novel, the three-dimensional taxonomy is well-designed, and the cognition-inspired motivation is convincing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation covers 10 downstream tasks, from-scratch pre-training, ablation studies, and annotator independence verification, though model scale remains relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clear, the method is concise, and the analysis is thorough; the discussion of complementarity and limitations is particularly strong.
- **Value**: ⭐⭐⭐⭐ The method is simple and reproducible, with practical implications for pre-training data utilization and hallucination mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ICLR 2026\] Self-Destructive Language Model](../../ICLR2026/llm_safety/self-destructive_language_model.md)
- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](../../ICLR2026/llm_safety/unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)

</div>

<!-- RELATED:END -->
