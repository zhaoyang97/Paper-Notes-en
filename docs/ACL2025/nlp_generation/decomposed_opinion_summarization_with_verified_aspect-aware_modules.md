---
title: >-
  [Paper Note] Decomposed Opinion Summarization with Verified Aspect-Aware Modules
description: >-
  [ACL 2025][Text Generation][Opinion Summarization] This study decomposes the opinion summarization task into three progressively verifiable modules—Aspect Identification, Opinion Consolidation, and Meta-Review Synthesis. By using zero-shot prompting on LLMs, a domain-independent modular processing pipeline is achieved, generating more traceable and comprehensive summaries across three domains: peer reviews, business reviews, and product reviews.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Opinion Summarization"
  - "Aspect-aware"
  - "Modular Decomposition"
  - "LLM Prompting"
  - "Meta-review Generation"
date: 2026-05-08
content_hash: 2f4566cb2d2f8066
---

# Decomposed Opinion Summarization with Verified Aspect-Aware Modules

**Conference**: ACL 2025  
**arXiv**: [2501.17191](https://arxiv.org/abs/2501.17191)  
**Code**: None  
**Area**: Text Generation  
**Keywords**: Opinion Summarization, Aspect-aware, Modular Decomposition, LLM Prompting, Meta-review Generation

## TL;DR

This study decomposes the opinion summarization task into three progressively verifiable modules—Aspect Identification, Opinion Consolidation, and Meta-Review Synthesis. By using zero-shot prompting on LLMs, a domain-independent modular processing pipeline is achieved, generating more traceable and comprehensive summaries across three domains: peer reviews, business reviews, and product reviews.

## Background & Motivation

**Background**: Opinion summarization aims to aggregate voluminous online reviews (e.g., hotel reviews, product reviews, and peer reviews) into a concise meta-review. Existing approaches include extractive methods (selecting representative sentences), generative methods (end-to-end neural generation), and hybrid methods (clustering followed by generation).

**Limitations of Prior Work**: Extractive methods are traceable but lack coherence; generative methods (end-to-end) behave as black boxes, making them non-traceable and uncontrollable; and hybrid methods (such as HIRO) organize inputs through clustering, but these clusters are aspect-agnostic, which can result in irrelevant or ambiguous groupings. Furthermore, most methods are restricted by the context window of LLMs, making it difficult to handle hundreds or thousands of reviews.

**Key Challenge**: The need for comprehensive coverage of various review aspects versus the infeasibleness, uncontrollability, and non-traceability of end-to-end processing for large volumes of reviews.

**Goal**: (1) To design a domain-agnostic modular approach that makes the opinion summarization process transparent and inspectable; (2) To ensure that the generated summaries comprehensively cover all review aspects mentioned; (3) To leverage intermediate outputs to assist humans in writing summaries more efficiently.

**Key Insight**: Inspired by Chain-of-Thought and Decomposed Prompting, the complex opinion summarization task is explicitly decomposed into three subtasks, with each managed by a dedicated LLM module. The key difference lies in the decomposition being rooted in task and domain knowledge (aspect definitions), rather than being automatic and knowledge-independent.

**Core Idea**: Utilizing review aspects (e.g., "cleanliness," "location," "service" for hotels) as the organizing axis, the summarization is decomposed into a three-step pipeline: Aspect Identification $\rightarrow$ Aspect-specific Opinion Consolidation $\rightarrow$ Cross-aspect Meta-Review Synthesis, where each step can be independently verified.

## Method

### Overall Architecture

Given a set of reviews $R_i$ regarding an entity (e.g., a hotel, a scientific paper) and a predefined set of aspects $A_d$ for the domain, the system executes the following process: (1) **Aspect Identification**: extract text fragments related to each aspect from each review, forming aspect-level clusters; (2) **Opinion Consolidation**: generate aspect-specific summaries for each aspect cluster separately; (3) **Meta-Review Synthesis**: integrate the summaries of all aspects into a complete, coherent meta-review. All three modules are implemented via zero-shot LLM prompting.

### Key Designs

1. **Aspect Identification**:

    - **Function**: Extracts text fragments related to specific aspects from the raw reviews.
    - **Mechanism**: Given an aspect name (e.g., "Clarity") and its definition (e.g., "the readability, structure, and language of the paper"), the LLM is prompted to extract relevant fragments from each review. Since reviews can be processed individually, this stage is unaffected by context window constraints. Extracted fragments are clustered by aspect into $C_{a_i} = \{f_1, f_2, ...\}$. This semantic-definition-based "clustering" is more precise and interpretable than similarity-based unsupervised clustering.
    - **Design Motivation**: Existing hybrid methods (such as HIRO) use sentence-embedding-based clustering to organize reviews, but the resulting clusters may not correspond to meaningful review aspects. Directly categorizing text fragments with domain-knowledge-defined aspects guarantees the interpretability and coverage of the clustering.

2. **Opinion Consolidation**:

    - **Function**: Summarizes multiple review fragments within the same aspect into a concise aspect-specific summary.
    - **Mechanism**: A divide-and-conquer strategy is adopted—generating summaries for each aspect individually is far simpler than generating a complete, multi-aspect summary all at once. The LLM is prompted to synthesize all opinion fragments in an aspect cluster to generate a concise aspect summary $o_{a_i}$. For instance, summarizing three sentences in the "Clarity" cluster ("needs better readability", "chaotic structure", "unclear figures") into "the clarity of the paper needs improvement."
    - **Design Motivation**: Direct end-to-end generation from hundreds of reviews leads to information loss or runs into context window limitations. By segmenting by aspects, the input size for each subtask remains manageable, and each aspect summary can be independently verified for its faithfulness to the source reviews.

3. **Meta-Review Synthesis**:

    - **Function**: Integrates the summaries from all aspects into a fluent and complete meta-review.
    - **Mechanism**: After concatenating the aspect-specific summaries generated by Opinion Consolidation, the LLM is prompted to write a comprehensive meta-review that covers all aspects. Since the input consists of already refined aspect summaries, which are short and well-structured, the LLM can easily generate a high-quality output.
    - **Design Motivation**: The final meta-review needs to establish transitions between different aspects and maintain readability. This step is essentially a short-text, multi-document summarization task, distinct from the information extraction and consolidation operations in the first two steps.

### Loss & Training

The entire method requires no training and relies solely on the zero-shot prompting capabilities of pretrained LLMs. The backbone models used include GPT-4o, Llama-3.1-70B-Instruct, and Llama-3.1-8B-Instruct. For the fine-tuning baseline (FT-Llama 8B), training is conducted for 5 epochs using the AdaFactor optimizer with a learning rate of 1e-6.

## Key Experimental Results

### Main Results

Scientific Paper Reviews (PeerSum Dataset):

| Method | Coverage↑ | G-Eval↑ | AlignScore-R↑ |
|------|-----------|---------|---------------|
| Sentiment CoT-GPT-4o (SOTA) | 0.96 | 0.75 | 0.72 |
| FT-Llama 8B | 0.87 | 0.60 | 0.33 |
| Aspect-aware decomp.-GPT-4o (Ours) | **0.95** | **0.76** | 0.68 |
| Aspect-aware decomp.-Llama 70B (Ours) | **0.97** | **0.86** | **0.75** |
| Automatic decomp.-Llama 70B | 0.76 | 0.57 | 0.59 |
| Chunk-wise decomp.-Llama 70B | 0.88 | 0.76 | 0.69 |

Hotel Reviews (SPACE Dataset):

| Method | Coverage↑ | G-Eval↑ | AlignScore-R↑ |
|------|-----------|---------|---------------|
| HIRO-abs (SOTA) | 0.87 | 0.62 | 0.83 |
| Aspect-aware decomp.-GPT-4o (Ours) | **1.00** | **0.90** | 0.81 |
| Aspect-aware decomp.-Llama 70B (Ours) | **0.99** | **0.86** | **0.85** |

### Ablation Study

Module Contribution Analysis (Llama 70B, Coverage↑ / AlignScore-S↑):

| Configuration | Hotels | Sneakers | Academic Papers |
|------|------|--------|----------|
| AI+OC+MS (Full Model) | 0.99/0.80 | 0.83/0.74 | 0.97/0.79 |
| OC+MS (w/o Aspect Identification) | 0.99/0.83 | 0.69/0.72 | 0.98/0.78 |
| AI+MS (w/o Opinion Consolidation) | 0.55/0.62 | 0.61/0.69 | 0.97/0.75 |
| AI†+OC+MS (Human-annotated fragments) | — | — | 0.97/0.69 |

### Key Findings

- **Opinion Consolidation (OC) is the most critical module**: Removing the OC module causes the Coverage in the hotel and sneakers domains to drop from 0.99 to 0.55 and 0.83 to 0.61 respectively, showing that the intermediate consolidation step is vital for final summary quality.
- **Model-extracted fragments are more helpful than human annotations**: In the peer-review domain, using model-extracted fragments (0.79 AlignScore) outperforms human-annotated fragments (0.69), potentially because model extractions encompass more relevant information.
- **Llama-70B outperforms GPT-4o in aspect identification**: Scoring F1 of 0.46 vs. 0.40 respectively, demonstrating that large open-source models are highly competitive in following structured instructions.
- **Human evaluation consistently prefers the proposed approach**: In human evaluations, crowdsourced workers chose the proposed system's summaries more frequently than gold-standard reference summaries in most domains.
- **Assisting human writing is effective**: Providing intermediate outputs from the proposed method reduces the time humans spend writing summaries by 14.7% and doubles quality preference.

## Highlights & Insights

- **Generality of the decomposition strategy**: Applying the modular "large-to-subtask" decomposition concept to opinion summarization ensures that each module can be independently verified and replaced. This framework can easily migrate to any summarization/analysis tasks requiring large volume inputs.
- **Elegant injection of domain knowledge**: Rather than injecting domain knowledge through training data, this approach guides the model using aspect definitions (a few sentences of natural language descriptions), enabling adaptation to new domains at an extremely low cost.
- **Practical value of intermediate outputs**: It is not just the final summary that is useful; the intermediate aspect fragment clusters and aspect summaries themselves can assist human workflow. This "human-in-the-loop" design philosophy is highly valuable.

## Limitations & Future Work

- **Evaluation restricted to English data**: All three experimental datasets are in English, leaving performance in multilingual scenarios unexplored.
- **Room for prompt optimization**: The paper acknowledges that systematic prompt engineering optimization was not performed; more refined prompt design could further boost performance.
- **Aspect definitions require manual predefinition**: Each domain requires a predefined set of aspects and descriptions; although the adaptation cost for new domains is low, it is non-zero.
- **Unaddressed correlation between aspects**: The three-step pipeline is linear, meaning potential associations between aspects (e.g., "price" and "value for money") are not modeled.
- **Lack of mechanisms to handle bias and harmful content**: Generated summaries might inadvertently amplify certain biased viewpoints.

## Related Work & Insights

- **vs. HIRO**: HIRO utilizes hierarchical index based on sentence embeddings to organize reviews, resulting in aspect-independent clustering. This work directly leverages aspect definitions to classify text fragments, yielding more precise and interpretable clusters. On hotel reviews, the Coverage of the proposed approach reaches 1.00, whereas HIRO only secures 0.87.
- **vs. Decomposed Prompting (Khot et al.)**: Decomposed Prompting automatically predicts task decomposition and modules without leveraging domain knowledge. This study demonstrates that task-aware manual decomposition outperforms automatic decomposition, lifting the Coverage on peer-reviews from 0.76 to 0.97.
- **vs. Sentiment Consolidation (Li et al. 2024)**: Li et al. specifically target peer-reviews by organizing comments using sentiment labels. The proposed method is more generalizable, directly adaptable to business and product reviews.

## Rating

- Novelty: ⭐⭐⭐ Modular decomposition is not a new concept per se, but aspect-based decomposition and consolidation in opinion summarization represents an effective engineering innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceptionally comprehensive across three domains, featuring multi-model comparisons, ablation studies, human evaluations, and human-AI collaboration experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ A 37-page long paper, detailed and clear in exposition, with a rigorous experimental design.
- Value: ⭐⭐⭐⭐ Provides a practical modular framework for large-scale review processing, with direct references for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [\[ACL 2025\] DTCRS: Dynamic Tree Construction for Recursive Summarization](dtcrs_dynamic_tree_construction_for_recursive_summarization.md)
- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [\[ACL 2025\] An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)
- [\[ACL 2025\] What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations](video_text_summarization.md)

</div>

<!-- RELATED:END -->
