---
title: >-
  [Paper Note] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates
description: >-
  [ACL 2026][LLM Pretraining][Knowledge coordinates] Ours proposes Knowledge Coordinate conditioning (KoCo), which maps each document into a three-dimensional semantic coordinate (Source, Content…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Knowledge coordinates"
  - "conditional pre-training"
  - "hallucination mitigation"
  - "data contextualization"
  - "pre-training acceleration"
date: 2026-05-08
content_hash: 9b0691406e2eed5e
---

# KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates

**Conference**: ACL 2026  
**arXiv**: [2604.12397](https://arxiv.org/abs/2604.12397)  
**Code**: None  
**Area**: LLM Safety / Pre-training  
**Keywords**: Knowledge coordinates, conditional pre-training, hallucination mitigation, data contextualization, pre-training acceleration

## TL;DR

Ours proposes Knowledge Coordinate conditioning (KoCo), which maps each document into a three-dimensional semantic coordinate (Source, Content, Stability). These are injected into pre-training as natural language prefixes, granting the model explicit context-aware capabilities. This approach improves performance across 10 downstream tasks, accelerates convergence by approximately 30%, and effectively mitigates hallucinations.

## Background & Motivation

**Background**: Standard LLM pre-training treats the corpus as a flat sequence of tokens, indiscriminately optimizing the negative log-likelihood of the next token—regardless of whether that token originates from a peer-reviewed theorem or a casual social media conversation. This contrasts sharply with human learning, where readers naturally contextualize information based on its source and role.

**Limitations of Prior Work**: Recent attempts at improvement fall into two categories. Metadata-aware pre-training (such as MeCo) identifies sources via URL prefixes, but URLs are often too fine-grained, rely on prior mappings, and lack objectivity. Data selection methods (such as ASK-LLM) use classifiers to filter high-quality data but employ a binary approach—retaining high-quality data while discarding the rest. Unlike these methods, humans do not simply "delete" low-quality information but rather contextualize it based on its source and nature.

**Key Challenge**: Existing methods provide either superficial context signals (URLs) or directly discard "low-quality" data instead of helping the model understand its limitations. A more structured way is needed for the model to perceive the position of each document within the knowledge space.

**Goal**: Design a simple and effective pre-training method that, by providing an objective knowledge coordinate description for each document, allows the model to acquire human-like context-awareness during the pre-training phase.

**Key Insight**: Inspired by the DIKW (Data-Information-Knowledge-Wisdom) hierarchy, each document is mapped to a three-dimensional semantic space—Source, Content, and Stability. This is injected into pre-training as a natural language prefix, allowing the model to distinguish between "eternal physical theorems" and "transient social opinions."

## Method

### Overall Architecture

KoCo transforms the standard pre-training objective from $P(x)$ to a conditional distribution $P(x|\mathcal{T})$, where $\mathcal{T} = (s, c, t)$ is a triplet of knowledge coordinates for the document. The workflow consists of: (1) Using a lightweight language model (Qwen-3-4B) as a tagger to predict 3D coordinate labels based on the document's URL and text; (2) Concatenating these labels into a natural language prefix (e.g., "Source: Academic; Content: Reference; Stability: Evergreen") and prepending it to the original text; (3) Computing the loss only on the document tokens during pre-training (masking the prefix part). The training objective is:

$$\mathcal{L}_{\text{KoCo}} = -\sum_{i=1}^{n} \log P_\theta(x_i | x_{<i}, \mathcal{T})$$

### Key Designs

1.  **3D Knowledge Coordinate System**
    - **Function**: Provides an objective meta-description for each document that is independent of specific semantic topics.
    - **Mechanism**: Defines three orthogonal dimensions—**Source** (10 categories such as Academic/Media/Community/Personal), **Content** (11 categories such as Instructional/Pedagogical/Discussion/Opinion), and **Stability** (4 categories: Ephemeral/Decaying/Long-term/Evergreen). Over 99.5% of documents in the DCLM corpus can be successfully mapped to this coordinate system.
    - **Design Motivation**: Unlike the superficial signals provided by URLs, the 3D coordinates originate from the essential attributes of information, simulating the human cognitive process of "understanding information source and nature." Ablation experiments confirm these three dimensions capture complementary information.

2.  **Conditional Inference Control**
    - **Function**: Guides model behavior during inference by specifying knowledge coordinate prefixes.
    - **Mechanism**: Specific coordinate prefixes are designed for different tasks (e.g., {Source: Media; Content: Discussion} for Social IQA, {Source: Academic; Content: Pedagogical} for LogiQA). More importantly, specifying reliable source prefixes (e.g., {Source: Publication; Content: Instructional; Stability: Long-term}) yields an improvement of up to 3.78% on TruthfulQA.
    - **Design Motivation**: KoCo introduces control signals during pre-training that are typically only available in the alignment stage, allowing users to suppress unreliable outputs during inference by simply specifying knowledge coordinates.

3.  **Tagger Independence Verification**
    - **Function**: Proves that the gains of KoCo come from the coordinate conditioning mechanism itself rather than knowledge distillation from the tagger model.
    - **Mechanism**: A BERT-base model with only 110M parameters (significantly smaller than the 0.6B pre-trained model) is used as an alternative tagger. It generates coordinates for KoCo pre-training after being trained on 50K labeled samples.
    - **Design Motivation**: If KoCo's improvements resulted from distilling knowledge from Qwen-3-4B, using a much weaker tagger should significantly degrade performance. Experiments show both taggers achieve comparable results, ruling out the knowledge distillation hypothesis.

### Loss & Training

During pre-training, the loss is calculated only on the document tokens, with prefix tokens being masked. Continued pre-training was performed on a MeCo 1.6B checkpoint using a 100GB subset of the DCLM corpus. When pre-training from scratch, KoCo demonstrated approximately 30% convergence acceleration on both 0.3B and 0.6B models.

## Key Experimental Results

### Main Results

Evaluated on 10 downstream tasks using the MeCo 1.6B checkpoint for continued pre-training:

| Method | COPA | ARC-e | ARC-c | CSQA | IFEval | OBQA | PIQA | SIQA | LogiQA | TruQA | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MeCo (URL Prefix) | 82.0 | 75.4 | 44.4 | 64.0 | 20.0 | 50.8 | 73.0 | 52.9 | 25.5 | 36.3 | 52.42 |
| Standard Continued Pre-training | 82.0 | 74.6 | 42.8 | 59.5 | 22.2 | 49.6 | 72.9 | 52.7 | 24.9 | 35.2 | 51.64 |
| Data Selection | 82.0 | 75.0 | 44.6 | 63.3 | 22.4 | 49.0 | 74.0 | 52.6 | 25.2 | 35.5 | 52.36 |
| **KoCo** | **83.0** | **77.4** | 44.1 | 61.8 | **25.5** | **51.2** | **74.8** | **53.4** | **26.9** | **36.6** | **53.48** |

### Ablation Study

| Setting | ARC-e | ARC-c | OBQA | PIQA | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| w/o Source | 76.2 | 44.1 | 50.2 | 73.7 | 53.43 |
| w/o Content | 76.6 | 43.6 | 51.2 | 74.1 | 53.46 |
| w/o Stability | 76.7 | 43.1 | 51.0 | 73.8 | 53.32 |
| KoCo (Full) | 77.4 | 44.1 | 51.2 | 74.8 | **53.48** |

### Key Findings

- KoCo significantly improves performance (by 1.06% on average) using the same data as MeCo (DCLM corpus) without introducing extra data.
- Standard continued pre-training actually degrades MeCo checkpoint performance, while data selection only maintains parity—indicating that simple data manipulation is insufficient and structured conditional signals are necessary.
- Conditional inference improves TruthfulQA by 3.78%, far exceeding other tasks—the model successfully learned the association between source reliability and factuality.
- Using unreliable source prefixes (e.g., Personal/x.com + Opinion + Ephemeral) drops the TruthfulQA score to 34.75%, while reliable source prefixes increase it to 40.39%.
- PCA visualization shows that the model trained with KoCo clearly separates factual and opinionated statements in the representation space.

## Highlights & Insights

- **Cognitive-inspired Design**: The 3D knowledge coordinates simulate the human cognitive process of "understanding information source and nature," featuring a simple concept with significant results.
- **A New Path for Hallucination Mitigation**: By teaching the model to distinguish between reliable and unreliable sources during pre-training, it provides a fundamental method for reducing hallucinations rather than post-hoc fixes.
- **Bridging Pre-training and Alignment**: KoCo moves control signals from the alignment stage up to the pre-training stage, suggesting that certain alignment objectives can be shifted upstream to simplify downstream fine-tuning.
- **Retaining Low-quality Data**: Unlike data selection methods, KoCo retains all data but labels its nature, allowing the model to learn from all data while understanding the context.

## Limitations & Future Work

- The experimental scale is limited to 0.3B-1.6B models; effectiveness on larger-scale models remains to be verified.
- Tagger accuracy (approximately 75-83% consistency with commercial models) contains noise, which may limit the precision of the coordinates.
- The 3D coordinate system is hand-designed; it is unclear whether additional dimensions or automatically discovered coordinates would be superior.
- The conclusion regarding ~30% pre-training acceleration was only verified on small models trained from scratch; it remains to be confirmed in large-scale settings.
- Conditional inference requires users to manually select appropriate coordinate prefixes; automated selection mechanisms require further research.

## Related Work & Insights

- **vs MeCo (URL Prefix)**: MeCo uses URLs as source identifiers, which are too fine-grained and rely on priors; KoCo uses structured 3D coordinates to provide more objective and informative conditional signals.
- **vs Data Selection**: Data selection uses a binary "keep/discard" strategy; KoCo retains all data while tagging its attributes, enabling the model to understand information of varying quality within context.
- **vs RLHF/SFT Alignment**: KoCo introduces controllability into the pre-training phase, offering a new perspective for "upstream alignment."

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of knowledge coordinates is novel, the 3D classification system is well-designed, and the cognitive-inspired motivation is persuasive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 10 tasks, pre-training from scratch, ablation studies, and tagger independence verification are all complete, though model sizes are relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, concise methodology, and deep analysis. The discussion on complementarity and limitations is particularly excellent.
- Value: ⭐⭐⭐⭐ The method is simple and reproducible, offering practical guidance for pre-training data utilization and hallucination mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](../../NeurIPS2025/llm_pretraining/nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[ICML 2026\] FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](../../ICML2026/llm_pretraining/flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment.md)
- [\[ACL 2026\] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)

</div>

<!-- RELATED:END -->
