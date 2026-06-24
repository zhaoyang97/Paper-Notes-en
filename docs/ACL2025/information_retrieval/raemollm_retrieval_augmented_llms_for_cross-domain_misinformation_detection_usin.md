---
title: >-
  [Paper Note] RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information
description: >-
  [ACL2025][Information Retrieval & RAG][Cross-Domain Misinformation Detection] This paper proposes RAEmoLLM, the first RAG framework based on emotional information retrieval. It utilizes the implicit embeddings of an emotional LLM to construct a retrieval database, providing emotion-related few-shot examples for cross-domain misinformation detection. Without the need for fine-tuning, RAEmoLLM improves performance by up to 15.64%, 31.18%, and 15.73% on three benchmarks respecti…
tags:
  - "ACL2025"
  - "Information Retrieval & RAG"
  - "Cross-Domain Misinformation Detection"
  - "Retrieval-Augmented Generation"
  - "Emotional Information"
  - "In-Context Learning"
  - "Few-Shot Learning"
date: 2026-05-08
content_hash: ed1f32dc541ee29a
---

# RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information

**Conference**: ACL2025  
**arXiv**: [2406.11093](https://arxiv.org/abs/2406.11093)  
**Code**: [lzw108/RAEmoLLM](https://github.com/lzw108/RAEmoLLM)  
**Area**: Information Retrieval  
**Keywords**: Cross-Domain Misinformation Detection, Retrieval-Augmented Generation, Emotional Information, In-Context Learning, Few-Shot Learning

## TL;DR

This paper proposes RAEmoLLM, the first RAG framework based on emotional information retrieval. It utilizes the implicit embeddings of an emotional LLM to construct a retrieval database, providing emotion-related few-shot examples for cross-domain misinformation detection. Without the need for fine-tuning, RAEmoLLM improves performance by up to 15.64%, 31.18%, and 15.73% on three benchmarks respectively compared to other few-shot methods.

## Background & Motivation

**Misinformation Proliferation**: Misinformation on the Internet spreads widely across multiple domains such as education, politics, and health, causing severe harm to social life and stability, requiring substantial time and effort to verify.

**Challenges of Cross-Domain Detection**: Models trained on specific domains often perform poorly and make inaccurate predictions when encountering samples from new domains. Cross-domain misinformation detection has become a pressing global issue.

**High Cost of Existing Methods**: Current cross-domain detection methods rely on time-consuming fine-tuning and complex model architectures (such as Mixture of Experts/MoE multi-domain models), making it difficult to adapt quickly to new domains.

**LLM Methods Restricted to In-Domain**: Although LLMs perform exceptionally well in many tasks, existing LLM-based misinformation detection methods focus only on in-domain tasks and lack cross-domain capabilities.

**Neglect of Emotional Information**: Misinformation authors often deliberately choose specific emotions to attract readers' attention and resonance. However, existing methods seldom exploit sentiment and emotion features. ConspEmoLLM, the only prior counterpart, also requires fine-tuning and lacks cross-domain capabilities.

**Gap in ICL + Emotion Retrieval**: In-context learning (ICL) works without fine-tuning, requiring only task instructions and a few examples. However, no prior work has applied ICL based on emotional information retrieval to cross-domain misinformation detection.

## Method

### Overall Architecture

RAEmoLLM comprises three modules, forming a complete retrieval-augmented inference pipeline:
1. **Index Construction Module**: Extracts emotional embeddings of all domain data using an emotional LLM to construct a retrieval database.
2. **Retrieval Module**: Retrieves the Top-K relevant examples from the source domain based on emotional similarity.
3. **Inference Module**: Uses the retrieved examples as few-shot demonstrations to guide the LLM in determining the veracity of target domain content.

### Key Designs

**Dual Utilization of Emotional Information**:
- **Implicit Emotional Information**: Utilizes the final hidden layer of EmoLLaMA-chat-7B (a 4096-dimensional vector) as the emotional embedding to capture deep emotional semantics in the text.
- **Explicit Emotional Information**: Extracts labels from five emotional dimensions (EIreg emotion intensity, EIoc emotion classification, Vreg sentiment intensity, Voc sentiment classification, Ec emotion detection), which can be optionally added to the prompt.

**Validation of Emotional Analysis**: The paper validates through t-tests that real and fake information exhibit significant differences in emotional distribution:
- In the AMTCele dataset, fake information is associated with significantly higher levels of anger and fear, and significantly lower levels of joy.
- t-tests on implicit embeddings indicate that Top-K similarity within the same class is significantly higher than across classes, and PCA visualization also shows distinct separation of different classes in the latent space.

**Retrieval Process**:
1. Target and source domain data are encoded into emotional embeddings $E_T$ and $E_S$ respectively using EmoLLaMA.
2. For each target-domain embedding $e_t$, the cosine similarity with all source-domain embeddings $e_s$ is calculated.
3. The K source-domain samples (containing text and label) with the highest similarities are selected as few-shot examples.

**Inference Templates**:
- Template 1 (Implicit only): [Task Instruction] + [Target Text] + [Retrieved Examples] → [Output]
- Template 2 (Implicit + Explicit): Adds explicit emotional labels on top of Template 1 for both target texts and retrieved examples.

### Loss & Training

- **No Fine-Tuning**: The entire framework is based on in-context learning and does not fine-tune any LLM.
- **Flexible Base Models**: Multiple LLMs are supported as inference engines (ChatGPT, GPT-4o, Mistral, Llama, Gemma, Vicuna, etc.).
- **Number of Retrieved Examples**: 4 few-shot examples are used by default. Experiments show that increasing the number of examples does not necessarily improve performance; too many examples may introduce noise.

## Key Experimental Results

### Datasets

| Dataset | Domains | Type | Size |
|:---:|:---:|:---:|:---:|
| AMTCele | 7 domains | Fake news detection | 980 articles |
| PHEME | 9 domains | Rumor detection | 6425 tweets |
| COCO | 12 domains | Conspiracy theory detection | 2581 tweets |

### Main Results (F1 Score)

| Model | AMTCele | PHEME | COCO |
|:---:|:---:|:---:|:---:|
| BERT (Fine-tuned) | 0.5322 | 0.7208 | 0.6356 |
| RoBERTa (Fine-tuned) | 0.4730 | 0.7204 | 0.6388 |
| MDFEND | 0.5815 | 0.5829 | 0.7793 |
| EDDFN | 0.6951 | 0.6816 | 0.5917 |
| Mistral-7b-zs | 0.6926 | 0.5936 | 0.4673 |
| Mistral-7b-random | 0.6889 | 0.6227 | 0.7287 |
| **Mistral-7b-Vreg** | **0.7404** | **0.6788** | **0.7898** |
| **Mistral-7b-Vreg-addexpl** | **0.7717** | **0.6920** | **0.7931** |
| GPT4o-zs | 0.8813 | 0.6228 | 0.7396 |
| **GPT4o-Vreg** | **0.8884** | **0.6992** | **0.8326** |

### Ablation Study of Different Embedding Retrievals (Mistral-7b, F1)

| Retrieval Method | AMTCele | PHEME | COCO |
|:---:|:---:|:---:|:---:|
| Vreg (Emotion) | **0.7404** | **0.6788** | **0.7898** |
| Vreg-addexpl | **0.7717** | **0.6920** | **0.7931** |
| Semantic (RoBERTa) | 0.6904 | 0.6718 | 0.7771 |
| SentiBERT | 0.6984 | 0.6663 | 0.7687 |

### Key Findings

1. **Emotional retrieval significantly outperforms random few-shot**: Across all LLM backbones, examples retrieved based on Vreg emotional embeddings outperform randomly sampled examples, with maximum improvements of 15.64% (AMTCele/Gemma-2b), 31.18% (PHEME/Llama3.2-1b), and 15.73% (COCO/Vicuna-7b).
2. **Complementarity of implicit and explicit emotional information**: Adding explicit Vreg labels (Vreg-addexpl) further improves performance in most cases, demonstrating that implicit embeddings and explicit labels provide complementary emotional signals.
3. **Emotional embeddings outperform semantic embeddings**: Compared to RoBERTa semantic embeddings and SentiBERT emotional embeddings, EmoLLaMA's emotional embeddings lead to the best retrieval performance, validating the utility of specialized emotional LLMs.
4. **More retrieved examples is not always better**: Increasing the number of retrieved examples to 8/16/32/64 does not always improve performance; too many examples can introduce cross-class noise. The default of 4 examples serves as a good trade-off.
5. **Simply adding explicit emotion is ineffective**: random-addexpl (random sampling + explicit emotion) barely outperforms random alone, indicating that the key lies in the emotion-driven **retrieval** rather than simple emotion-label concatenation.

## Highlights & Insights

- **Unique value of the emotional perspective**: Employs emotional information as a bridge for cross-domain misinformation detection. Capitalizing on the insight that "misinformation often carries specific emotional patterns," the framework utilizes distribution differences in the emotional embedding space to select high-quality few-shot examples.
- **Practical plug-and-play solution without fine-tuning**: The entirely ICL-based framework offers plug-and-play adaptability to new domains, and its actual deployment cost is far lower than fine-tuning approaches.
- **Rigorous statistical validation**: Verification of the correlation between emotional information and misinformation classes using t-tests and PCA visualization strengthens the theoretical foundation of the method.
- **Broad compatibility with LLMs**: Validated as effective across 9 different LLMs (ranging from 1B to GPT-4o), demonstrating the generalizability of the framework.

## Limitations & Future Work

1. **Relatively weak performance on the PHEME dataset**: In the short-text rumor detection task, fine-tuned models sometimes outperform RAEmoLLM, likely due to the lack of rich emotional signals in short texts.
2. **Dependence on emotional LLMs**: The quality of EmoLLaMA-chat-7B embeddings dictates the core performance of the framework. Performance may degrade if the emotional LLM underperforms on new languages or domains.
3. **Computational overhead**: The approach requires encoding all data via the emotional LLM beforehand, followed by pairwise cosine similarity computations, which might become a bottleneck on large-scale datasets.
4. **Sole focus on the Vreg dimension**: Main experiments were based on Vreg (sentiment intensity) among the five emotional dimensions. Other dimensions (such as specific emotion types) remain unexplored for retrieval strategies.
5. **Domain division assumptions**: The framework assumes clear boundaries between the source and target domains, which may require additional handling for domains with ambiguous boundaries.

## Related Work & Insights

### vs ConspEmoLLM (Liu et al., 2024)
ConspEmoLLM is the only prior work using emotional LLMs for misinformation detection, but it requires fine-tuning, lacks cross-domain capabilities, and does not fully exploit emotional information. RAEmoLLM achieves fine-tuning-free cross-domain detection via ICL + emotional retrieval and utilizes both implicit and explicit emotional information, offering a more lightweight and broadly applicable solution.

### vs MDFEND (Nan et al., 2021) / CANMD (Yue et al., 2022)
These are specialized cross-domain misinformation detection methods employing complex architectures like MoE. MDFEND achieves an F1 of only 0.58 on AMTCele and PHEME, and CANMD achieved 0.73 on PHEME. RAEmoLLM (Mistral-7b-Vreg-addexpl) substantially outperforms these approaches on AMTCele and COCO without training any parameters.

### vs Standard RAG (Semantic Retrieval)
The variant using RoBERTa semantic embeddings for retrieval (Mistral-7b-semantic) yields F1 scores of 0.690/0.672/0.777, whereas emotional embedding retrieval (Vreg) achieves 0.740/0.679/0.790. The difference is particularly pronounced on AMTCele (+5.0%), indicating that the emotional dimension is better suited for cross-domain transfer in misinformation detection than pure semantics.

## Rating

- **Novelty**: 7/10 — The combination of emotional information + RAG for cross-domain misinformation detection is novel, though individual components (RAG, emotional LLMs, ICL) are existing techniques.
- **Experimental Thoroughness**: 8/10 — Comprehensive and detailed with 3 datasets, 9 LLMs, and various ablation studies (embedding types, retrieval sizes, explicit/implicit).
- **Writing Quality**: 7/10 — Well-structured and rigorously validated with statistical emotional analyses, though the abundance of tables makes it slightly lengthy.
- **Value**: 7/10 — Provides a practical, training-free solution for cross-domain misinformation detection, with inspiring thoughts on emotion-driven retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Contradiction Detection in RAG-Based Chatbots](contradiction_detection_in_rag-based_chatbots.md)
- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](../../CVPR2025/information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[ACL 2025\] On Synthetic Data Strategies for Domain-Specific Generative Retrieval](on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)
- [\[ACL 2025\] Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs](automatic_benchmark_generation_from_scientific_papers_via_retrieval-augmented_ll.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
