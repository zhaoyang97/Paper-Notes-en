---
title: >-
  [Paper Note] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs
description: >-
  [ACL 2025][Text Generation][media bias mitigation] Constructs a multi-document event relation graph (containing four types of intra-document event relations, cross-document event coreferences, and event-level moral foundations) and injects bias information into LLMs via two strategies: graph serialization and graph prompt tuning, generating unbiased neutralized summaries that outperform baselines in both content preservation and bias mitigation.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "media bias mitigation"
  - "neutralized summarization"
  - "event relation graph"
  - "multi-document summarization"
  - "graph prompt tuning"
date: 2026-05-08
content_hash: 38f95642b1c4904e
---

# Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.12978](https://arxiv.org/abs/2506.12978)  
**Code**: [https://github.com/yuanyuanlei-nlp/multi_doc_summarization_acl_2025](https://github.com/yuanyuanlei-nlp/multi_doc_summarization_acl_2025)  
**Area**: Text Generation  
**Keywords**: media bias mitigation, neutralized summarization, event relation graph, multi-document summarization, graph prompt tuning

## TL;DR

Constructs a multi-document event relation graph (containing four types of intra-document event relations, cross-document event coreferences, and event-level moral foundations) and injects bias information into LLMs via two strategies: graph serialization and graph prompt tuning, generating unbiased neutralized summaries that outperform baselines in both content preservation and bias mitigation.

## Background & Motivation

**Background**: News media is increasingly polarized, with journalists embedding ideological bias into reports through content framing. Most research focuses on bias detection (determining the political stance of an article), while studies on bias mitigation (generating neutral text) remain relatively scarce. Lee et al. (2022) proposed the neutralized summarization task: given multiple articles with different stances, generate a neutral summary.

**Limitations of Prior Work**: Existing neutralized summarization methods mainly rely on basic text-to-text generation, lacking awareness of bias. When generating summaries directly, LLMs still carry vocabulary-level and information-level biases, and may even hallucinate. The models are unaware of which events are objective facts commonly reported by all sides, versus selective reporting by a specific party.

**Key Challenge**: Bias mitigation requires the model to understand "where the bias comes from"—whether from emotional wording (lexical bias) or selective reporting of certain events (informational bias). Plain text inputs cannot clearly convey this structured bias distribution information to the model.

**Goal**: How to enable LLMs to perceive the bias distribution during summary generation, thereby simultaneously mitigating both lexical and informational biases without compromising the quality of content preservation.

**Key Insight**: Events and event relationships play a key role in bias detection—articles with different stances selectively report different events, connect events with different narrative logics, and attach different moral judgments to events. Constructing a cross-document event relation graph can systematically reveal the sources of bias.

**Core Idea**: Encode bias distribution information using a multi-document event relation graph, and inject it into LLMs via a dual-channel of hard prompt (graph textualization) and soft prompt (graph embedding) to guide the generation of neutralized summaries.

## Method

### Overall Architecture

Input: A set of news articles (typically 3) reporting on the same event but from different stances. Output: A neutralized summary. The pipeline consists of two stages: (1) multi-document event relation graph construction—extracting events, predicting moral attributes, extracting four types of intra-document event relations and cross-document event coreferences; (2) graph injection into LLMs—graph textualization to convert the graph into text as a hard prompt, and graph prompt tuning using GAT to encode graph embeddings as a soft prompt, both of which are complementarily fed into a frozen LLM to generate the summary.

### Key Designs

1. **Multi-document Event Relation Graph Construction**:

    - **Function**: Systematically characterize the bias distribution across multiple articles.
    - **Mechanism**: Extracts event words using an event detector trained on MAVEN; assigns moral labels to each event using a moral classifier trained on EMONA (covering 10 categories across 5 dimensions, such as Care/Harm, Fairness/Cheating); predicts four types of intra-document event relations (coreference, temporal, causal, subevent) using a relation extractor jointly trained on MAVEN-ERE; and links different articles using a cross-document event coreference system. In the graph, nodes represent events, attributes denote moral labels, and edges signify event relations.
    - **Design Motivation**: Cross-document coreferences reveal content selection biases (which events are commonly reported vs. selectively reported); intra-document relations reflect narrative framing biases; moral labels expose opinion-based biases.

2. **Graph Textualization (Hard Prompt)**:

    - **Function**: Convert graph structure information into text that can be directly read by the LLM.
    - **Mechanism**: Converts the graph into two tables—an event table $T_{event}$ (event ID, event text, moral judgment) and a relation table $T_{relation}$ (source event, relation type, target event)—which are concatenated as text and encoded into a hard prompt $h_t = \text{TextEmbedder}(T_{event}; T_{relation})$ through the LLM's text embedder.
    - **Design Motivation**: Textualization retains the structural information of the graph while leveraging the LLM's natural language understanding capabilities to interpret event relations.

3. **Graph Prompt Tuning (Soft Prompt)**:

    - **Function**: Allow the model to learn directly from the graph structure through learnable graph embeddings.
    - **Mechanism**: Initializes event embeddings using Longformer, concatenates them with moral label embeddings, and updates them via a relation-aware GAT—where the attention weight $\alpha_{ij} = \text{softmax}((W^Q e_i)(W^K r_{ij})^T)$ incorporates the relation type. A global graph node is introduced to aggregate graph embeddings via GAT, which are then projected into the LLM representation space through a two-layer MLP: $\hat{h}_g = W_2(W_1 h_g + b_1) + b_2$.
    - **Design Motivation**: Hard prompts strengthen instructions, while soft prompts directly fine-tune, complementing each other.

### Loss & Training

The LLM (Llama-2 / LED) is frozen, while the GAT and projection layers are trained using standard autoregressive cross-entropy loss. Llama-2 utilizes LoRA (rank=8, alpha=16, dropout=0.05) with a learning rate of 1e-4, while LED is trained with a learning rate of 1e-5. The maximum input length is 2048, the maximum output length is 512, and training is conducted for 5 epochs.

## Key Experimental Results

### Main Results

| Method | Rouge-1 | Rouge-2 | Rouge-L | BLEU-2 | polarization↓ | sum-arousal↓ |
|------|---------|---------|---------|--------|--------------|-------------|
| GPT-4 | 42.36 | 16.49 | 26.30 | 19.04 | 75.86 | 5.34 |
| GPT-4 + graph | 42.61 | 18.67 | 30.82 | 19.09 | 31.77 | 3.60 |
| LED baseline | 40.30 | 18.63 | 30.24 | 17.30 | 31.97 | 2.45 |
| LED + full model | 42.96 | 20.66 | 32.74 | 19.09 | **28.14** | **1.97** |
| Llama-2 baseline | 42.26 | 19.25 | 30.88 | 19.15 | 30.30 | 2.81 |
| Llama-2 + full model | **45.14** | **22.30** | **34.02** | **21.89** | 27.89 | 2.46 |

### Ablation Study

| Configuration | Rouge-1 | Rouge-2 | polarization↓ | sum-arousal↓ |
|------|---------|---------|--------------|-------------|
| Llama-2 baseline | 42.26 | 19.25 | 30.30 | 2.81 |
| + event moral | 43.82 | 20.65 | 29.05 | 2.51 |
| + in-doc relations | 44.74 | 21.31 | 28.57 | 2.68 |
| + cross-doc coreference | 44.53 | 20.78 | 28.16 | 2.60 |
| + all (full model) | 45.14 | 22.30 | 27.89 | 2.46 |

### Key Findings

- The multi-document event relation graph simultaneously improves content preservation (Rouge/BLEU) and bias elimination (polarization/arousal), demonstrating they are not contradictory.
- The three components of the graph (moral labels, intra-document relations, and cross-document coreferences) each contribute complementary information and are all indispensable.
- Human evaluation validates the automatic metrics: after adding the graph, lexical bias improves from 83.33 to 91.02, informational bias from 84.61 to 89.74, and the non-hallucination rate from 68.42 to 84.21.
- Although GPT-4 excels in content quality (non-hallucination rate of 89.74), its bias score remains high (polarization of 75.86), indicating that even powerful LLMs require guided structured bias information.
- Qualitative analysis shows that the graph helps the model filter out single-source biased information, restore omitted consensus events, and eliminate hallucinations.

## Highlights & Insights

- **Ingenious design of event relation graphs as bias carriers**: Cross-document event coreferences naturally reveal content selection biases, intra-document relations reflect narrative framing differences, and moral labels directly denote opinion-based biases. This structurally introduces prior knowledge of bias detection into generation tasks.
- **Transferable Hard + Soft dual-channel injection paradigm**: Textualization allows the model to "know" the graph structure, while graph embeddings enable the model to "learn" the graph semantics. This paradigm is applicable to any task requiring the injection of structured knowledge into LLMs (e.g., knowledge graphs, causal graphs).
- **No modification to the LLM core**: Freezing the LLM + lightweight GAT + LoRA ensures high practicality.

## Limitations & Future Work

- The event relation extractor is relatively weak at identifying implicit relations; thus, the quality of graph construction is limited by upstream NLP tools.
- Evaluation is conducted only on a single dataset (NeuS), covering only U.S. political news.
- Moral label classification is based on the Moral Foundations Theory, which may not apply to all cultural backgrounds.
- End-to-end training (joint optimization of graph construction and summary generation) has not been explored.

## Related Work & Insights

- **vs NeuS (Lee et al., 2022)**: NeuS pioneered the task but used a pure text-to-text method. Ours is the first to inject bias indicator signals (event relation graphs) into the generation process.
- **vs Bang et al. (2023)**: They used polarity minimization loss to reduce bias, focusing only on the lexical level. Ours addresses both the lexical and informational levels simultaneously.
- **vs GPT-4 prompting**: Even GPT-4 + CoT is inferior to fine-tuned Llama-2 + graph, illustrating that bias mitigation requires structured bias information rather than relying solely on the model's inherent capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ Combining event relation graphs with LLMs for bias mitigation is novel, though the individual sub-modules rely on existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete automatic evaluation, human evaluation, ablation studies, and qualitative analysis, though evaluated on only one dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive illustrations, and detailed methodological descriptions.
- Value: ⭐⭐⭐⭐ Practically meaningful for media bias mitigation, and the paradigm of injecting structured knowledge into LLMs offers valuable reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](dpp_diverse_multidoc_summary.md)
- [\[ACL 2025\] Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [\[ACL 2025\] PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)
- [\[ACL 2025\] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)

</div>

<!-- RELATED:END -->
