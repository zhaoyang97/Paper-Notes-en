---
title: >-
  [Paper Note] A Survey on Patent Analysis: From NLP to Multimodal AI
description: >-
  [ACL 2025][Multimodal VLM][patent analysis] A systematic survey of NLP and multimodal AI applications in four core patent analysis tasks (classification, retrieval, quality analysis, and generation), proposing a taxonomy based on the patent lifecycle, and revealing methodological evolution trends from Word2Vec+LSTM to BERT/GPT and multimodal models, along with key research gaps.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "patent analysis"
  - "NLP"
  - "multimodal AI"
  - "PLM"
  - "LLM"
date: 2026-05-08
content_hash: 45ba401f72206ca1
---

# A Survey on Patent Analysis: From NLP to Multimodal AI

**Conference**: ACL 2025  
**arXiv**: [2404.08668](https://arxiv.org/abs/2404.08668)  
**Code**: [GitHub](https://github.com/AI4Patents-survey)  
**Area**: Multimodal VLM  
**Keywords**: patent analysis, NLP, multimodal AI, PLM, LLM

## TL;DR

A systematic survey of NLP and multimodal AI applications in four core patent analysis tasks (classification, retrieval, quality analysis, and generation), proposing a taxonomy based on the patent lifecycle, and revealing methodological evolution trends from Word2Vec+LSTM to BERT/GPT and multimodal models, along with key research gaps.

## Background & Motivation

**Background**: The global volume of patent data is growing exponentially, with the USPTO and EPO processing hundreds of thousands of patent applications annually. Patent examination involves multiple stages such as classification, retrieval, quality analysis, and drafting, which traditionally rely heavily on the expertise and time commitment of human examiners. Recently, breakthrough advancements in pre-trained language models (PLMs) and large language models (LLMs) in natural language processing have brought unprecedented opportunities for automated patent analysis.

**Limitations of Prior Work**: Existing patent AI surveys (Gomez & Moens 2014, Krestel et al. 2021, Ali et al. 2024) exhibit three critical limitations: first, they fail to cover recent advancements in PLM/LLM applications; second, they lack a systematic taxonomy based on task dimensions and methodological characteristics; third, they overlook the potential of multimodal learning (patent text + images) in retrieval and classification. The unique legal language structures of patent texts (e.g., nested claim structures) and the non-natural characteristics of patent images (black-and-white line drawings, annotated numbers) prevent the direct transfer of general NLP methods.

**Key Insight**: This work proposes a new taxonomy based on patent lifecycle tasks, organizing literature across two dimensions: four core tasks (classification, retrieval, quality analysis, and generation) and three classes of methods (traditional NNs, ensemble models, and PLMs/LLMs), thereby providing researchers with a roadmap for building task-specific methods. Meanwhile, a public GitHub repository is maintained to continuously update the categorized paper list.

## Method

### Overall Architecture

The survey is structured hierarchically around four core tasks in the patent lifecycle: **Patent Classification** (IPC/CPC hierarchical multi-label classification) $\rightarrow$ **Patent Retrieval** (prior art retrieval for text and images) $\rightarrow$ **Patent Quality Analysis** (predicting metrics like forward citations and patent family size) $\rightarrow$ **Patent Generation** (automated drafting of abstracts, claims, etc.). Within each task, relevant works are organized chronologically by their methodological evolution stages (Traditional ML $\rightarrow$ Traditional NN $\rightarrow$ PLM $\rightarrow$ LLM/Multimodal Models).

### Key Designs

1. **Three-Stage Evolution of Patent Classification Methods**:

    - **Function**: Automatically assign patents to multiple labels within the IPC/CPC hierarchical classification system.
    - **Mechanism**: Evolution from early Word2Vec+LSTM/GRU (Grawe et al. 2017, Risch & Krestel 2018) to ensemble methods combining multiple embeddings and deep models (Kamateri et al. 2022 used Bi-LSTM + Bi-GRU + multiple partitioning techniques), and further to BERT/SciBERT/XLNet fine-tuning (Roudsari et al. 2022 achieved a precision of 0.82), showing progressive performance improvements. The latest Sentence-BERT+KNN method (Bekamiri et al. 2024) achieves the best performance in recall and F1. Additionally, Ghauri et al. (2023) first applied CLIP+MLP to patent image classification (flowcharts, circuit diagrams, technical drawings, etc.).
    - **Design Motivation**: Patent texts contain highly technical jargon and complex structures; domain-adaptive pre-training (e.g., SciBERT) can better capture semantics in the patent domain.

2. **Multimodal Fusion Trends in Patent Retrieval**:

    - **Function**: Retrieve relevant patent documents and images based on queries (text or images) to support novelty assessment and infringement analysis.
    - **Mechanism**: Text retrieval evolved from SVM + word embeddings (Setchi et al. 2021) to BERT (Kang et al. 2020) and Sentence-BERT + TransE knowledge graph embeddings (Siddharth et al. 2022). Image retrieval progressed from CNN/ResNet50 (Kucer et al. 2022) to self-supervised deep metric learning (Higuchi et al. 2023 used InfoNCE + ArcFace). The state-of-the-art work by Lo et al. (2024) integrates BLIP-2 and GPT-4V for joint patent text-image retrieval, employing a distribution-aware contrastive loss to address long-tail class issues.
    - **Design Motivation**: Patent retrieval inherently requires cross-modal understanding—design patents are predominantly image-based, while utility patents are text-based. Multimodal fusion ensures comprehensive coverage.

3. **Rapid Penetration of LLMs in Patent Generation**:

    - **Function**: Automatically generate patent abstracts, independent claims, dependent claims, and specifications.
    - **Mechanism**: Transitioned from GPT-2 fine-tuning for claim generation (Lee & Hsiang 2020a) to Patentformer utilizing T5/GPT-J to generate specifications from claims + drawings (Wang et al. 2024a), and subsequently to RLHF-based PatentGPT-J (Lee 2024) and multi-agent frameworks (Wang et al. 2024b using Qwen2/LLaMA3/GPT-4o). A significant finding is that general LLMs (e.g., Llama-3, GPT-4) outperform domain-specific models in claim generation (Jiang et al. 2024).
    - **Design Motivation**: Patent drafting demands precise legal language and technical descriptions; the powerful text generation capabilities of LLMs can drastically reduce the time cost for patent attorneys.

## Key Experimental Results

### Main Results

**Patent Classification Performance Comparison** (USPTO Dataset):

| Method | Embedding | Model | Precision | Classification Level |
|------|------|------|-----------|---------|
| Risch & Krestel (2018) | FastText | GRU | 0.53 | Full Text |
| Lee & Hsiang (2020b) | — | BERT-base | 0.74 (acc) | Subclass |
| Roudsari et al. (2022) | Word2Vec/FastText | XLNet | **0.82** | Title/Abstract |
| Bekamiri et al. (2024) | SBERT | KNN | Best recall/F1 | Claim/Title/Abstract |

**Patent Retrieval Method Comparison**:

| Method | Data Type | Model | Training Method | Dataset |
|------|---------|------|---------|--------|
| Setchi et al. (2021) | Text | SVM/NB/RF | Supervised | — |
| Pustu-Iren et al. (2021) | Text + Image | RoBERTa+CLIP | Pre-training | EPO |
| Kucer et al. (2022) | Image | ResNet50 | Fine-tuning | DeepPatent |
| Lo et al. (2024) | Text + Image | BLIP-2+GPT-4V | Pre-training + Supervised | DeepPatent2 |

### Ablation Study

| Dimension of Analysis | Findings | Impact |
|---------|------|------|
| Text vs. Image Retrieval | Multimodal Transformer models > Unimodal | Highest mAP |
| Classification Level (Section→Subgroup) | Finer levels show more significant drops in accuracy | Subclass achieves at most 0.74 |
| Patent Document Components | Claim > Abstract > Full text | Information density impact |
| General vs. Domain-specific LLMs | General LLMs $\ge$ Domain-specific models | Stronger generalization |

### Key Findings

- The introduction of PLMs significantly boosted patent classification precision from 0.53 to 0.82, with domain-adaptive pre-trained models like SciBERT showing superior understanding of technological language.
- Multimodal retrieval is a clear trend—patent images (black-and-white line drawings) differ drastically from natural images, requiring specialized vision encoders.
- General LLMs (GPT-4, Llama-3) surprisingly outperform domain-specific models (PatentGPT-J) in patent generation, reflecting the generalization advantages of large-scale pre-training.
- Patent quality analysis lacks unified "gold standard" evaluation metrics—forward citations remain the only metric directly correlated with actual value.
- Patent texts generated by LLMs face hallucination risks and legal compliance challenges, where RLHF and RAG represent promising directions for development.

## Highlights & Insights

- This work proposes the first systematic taxonomy based on patent lifecycle tasks, filling a critical gap in existing surveys that lack task-oriented organization.
- It systematically outlines a clear evolutionary pathway from traditional NN $\rightarrow$ PLM $\rightarrow$ LLM $\rightarrow$ Multimodal models, offering a roadmap for follow-up research.
- It identifies four crucial future directions: multimodal patent foundation models, RAG-based hallucination mitigation, patent knowledge graph construction, and cross-jurisdictional retrieval.
- It maintains a public GitHub repository for continuous updates, offering high practical value.
- It reveals a significant methodological gap between the patent domain and general NLP—models currently used in patent classification lag far behind state-of-the-art LLMs.

## Limitations & Future Work

- The survey focuses predominantly on academic methods, offering limited coverage of AI systems actually deployed in industry (e.g., USPTO, EPO).
- There is a lack of unified benchmark comparisons across methods—differences in dataset subsets, classification hierarchies, and evaluation metrics make horizontal comparison difficult.
- The discussion on how the special linguistic structures of patent texts (such as the nested legal language of claims) influence model design is not deep enough.
- Crucial practical deployment challenges, such as data labeling costs and model interpretability, are left undiscussed.
- The discussion on multimodal methods is relatively thin, lacking a unified multimodal benchmark.

## Related Work & Insights

- **vs. Gomez & Moens (2014)**: An early NLP+patent survey; this work covers brand-new developments in the PLM/LLM era.
- **vs. Krestel et al. (2021)**: Focused on information extraction; this work expands to four major tasks: classification, retrieval, quality analysis, and generation.
- **vs. Ali et al. (2024)**: Surveys AI methods but overlooks recent LLM trends and multimodal fusion methods.

## Rating

- Novelty: ⭐⭐⭐ The taxonomy is innovative, but survey-style works are inherently limited.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive literature coverage, but lacks unified experimental verification.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich tables, and strong organizational logic.
- Value: ⭐⭐⭐⭐ Provides a valuable panoramic map and guidance on future directions for the patent AI field.
---
title: >-
  [Paper Reading] A Survey on Patent Analysis: From NLP to Multimodal AI
description: >-
  [Multimodal] A comprehensive survey of NLP and multimodal AI applications in patent analysis, proposing a new taxonomy based on patent lifecycle tasks. It covers four major tasks—classification, retrieval, quality analysis, and generation—revealing the evolution of existing approaches from traditional NNs to PLMs/LLMs and pointing out future directions.
tags:
  - Multimodal
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](../../AAAI2026/multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[ACL 2025\] MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation](mira_empowering_one-touch_ai_services_on_smartphones_with_mllm-based_instruction.md)
- [\[ACL 2026\] A Survey of Deep Learning for Geometry Problem Solving](../../ACL2026/multimodal_vlm/a_survey_of_deep_learning_for_geometry_problem_solving.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](../../ICCV2025/multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
