---
title: >-
  [Paper Note] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education
description: >-
  [ACL 2025][LLM (Other)][STEM Education Retrieval] This paper proposes a multi-style multimodal retrieval task and dataset SER (24,000+ query pairs) designed for STEM education scenarios, alongside a lightweight retrieval model Uni-Retrieval based on a Prompt Bank. It extracts query style features through prototype learning and dynamically selects prompt vectors to enhance retrieval performance across various styles (text, sketch, art, low-resolution, speech)…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "STEM Education Retrieval"
  - "Multi-style Retrieval"
  - "Prompt Bank"
  - "Prototype Learning"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 63ddaf717a659837
---

# Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education

**Conference**: ACL 2025  
**arXiv**: [2502.05863](https://arxiv.org/abs/2502.05863)  
**Code**: None  
**Area**: Other  
**Keywords**: STEM Education Retrieval, Multi-style Retrieval, Prompt Bank, Prototype Learning, Vision-Language Models

## TL;DR

This paper proposes a multi-style multimodal retrieval task and dataset SER (24,000+ query pairs) designed for STEM education scenarios, alongside a lightweight retrieval model Uni-Retrieval based on a Prompt Bank. It extracts query style features through prototype learning and dynamically selects prompt vectors to enhance retrieval performance across various styles (text, sketch, art, low-resolution, speech), surpassing existing methods on both STEM education retrieval and traditional retrieval datasets.

## Background & Motivation

1. **Background**: AI-powered education (AI4EDU) is rapidly developing, with an increasing demand for resource retrieval in STEM education. Current retrieval systems are primarily designed for natural text-image matching using pre-trained models such as CLIP and BLIP.

2. **Limitations of Prior Work**: (1) Existing retrieval models predominantly optimize natural text-image matching, ignoring the diverse querying modalities in educational scenarios—teachers may search for instructional materials via speech, hand-drawn sketches, or low-resolution photos; (2) The polysemy and ambiguity of retrieval expressions are particularly prominent in educational contexts (abstract concepts require multiple interpretations); (3) There is a lack of specialized evaluation benchmark datasets for STEM education retrieval.

3. **Key Challenge**: Educational scenarios require diverse querying styles (text, sketch, speech, low-resolution images, etc.), but existing models are only optimized for a single style, failing to satisfy the diverse needs of educational scenarios.

4. **Goal**: How to build an efficient STEM education retrieval system capable of adapting to multiple querying styles?

5. **Key Insight**: Treating different querying styles as learnable "prototypes," storing and composing semantic information of various styles using a Prompt Bank, and dynamically retrieving and concatenating prompt tokens to adapt to any query style.

6. **Core Idea**: Storing prototype information of different query styles in a sustainably updatable Prompt Bank, and dynamically selecting prompt tokens via prototype matching to extend the input representation, thereby achieving unified multi-style retrieval.

## Method

### Overall Architecture

Uni-Retrieval consists of three sub-modules: (1) Prototype Learning Module—generates prototype feature vectors for each query style; (2) Prompt Bank—stores learnable key-value pairs, where the key is used to match the query style prototype, and the value is the prompt token injected into the model; (3) Feature Extractor—a vision-language model based on ViT + Transformer. Input Query $\rightarrow$ Prototype Matching $\rightarrow$ Select Prompt Tokens $\rightarrow$ Concatenate to Input Sequence $\rightarrow$ Extract Features through Frozen Backbone $\rightarrow$ Compute Similarity Ranking.

### Key Designs

1. **Prototype Learning Module**:

    - **Function**: Extracting representative feature vectors for each query style.
    - **Mechanism**: Given $m$ query samples $x_0^i$ of a certain style, a pre-trained style encoder $f$ is used to extract features $E_0^i = f(x_0^i)$, then average pooling is performed over all sample features of this style to obtain the prototype $P_j = \text{AvgPool}(\sum_{i=0}^m E_j^i)$. The style encoder can be a pre-trained style classifier (for image style queries) or a text encoder (for text queries), depending on the query type.
    - **Design Motivation**: Prototype vectors compress the core semantic information of each style and are used to subsequently retrieve the most matching prompt tokens from the Prompt Bank.

2. **Prompt Bank**:

    - **Function**: Storing learnable style knowledge and supporting dynamic composition to adapt to seen and unseen styles.
    - **Mechanism**: The Bank contains $N$ key-value pairs $\{(k_1, P_1), ..., (k_N, P_N)\}$. Given the prototype features of the input query, the top-$n$ most matching keys are retrieved via cosine similarity $\gamma$, the corresponding prompt tokens are fetched, and prefixed to the input sequence: $x_p = [CLS; P_{j_1}; P_{j_2}; ...; P_{j_n}; x_e]$. For unseen query styles, the Bank automatically combines tokens of multiple similar styles for representation. Both the keys and the prompt tokens are learnable parameters, which are jointly updated during training.
    - **Design Motivation**: The hash table-like structure enables fast matching (analogous to the hidden state design of TTT and Mamba). The key advantages are composability and generalizability—unseen styles can be represented by combining the tokens of known styles.

3. **Feature Extractor with Inference Strategy**:

    - **Function**: Conducting retrieval ranking based on a frozen vision-language model.
    - **Mechanism**: The vision encoder uses ViT (initialized with OpenCLIP) and the text encoder uses a Transformer (gpt-neo tokenizer), both of which are fully frozen to preserve the original semantic space. The vision and text prompt tokens share parameters to align modalities. The CLS token is used as the global representation. Audio queries can optionally be processed via GPT-4o for speech-to-text post-processing.
    - **Design Motivation**: Freezing the backbone significantly reduces trainable parameters. Only the keys and tokens of the Prompt Bank are trained (adding only about 26M parameters), making training highly efficient.

### Loss & Training

Triplet Loss is used: 

$$\mathcal{L} = \max\{0, \mu + d(\delta(x_f), \delta(x_r)) - d(\delta(x_f), \delta(x_h))\}$$ 

where $x_f$ represents query features, $x_r$ is the positive sample, $x_h$ is the negative sample, and $d$ denotes the normalized cosine distance. A regularization term for Prompt Bank key matching is added: 

$$\min_{k,p,L} \mathcal{L} + \lambda \sum_{K_x} \gamma(q(x), k_{si})$$ 

which encourages keys to align with prototype features. During inference, test-time adaptation is supported to update the Prompt Bank for domain-specific knowledge.

## Key Experimental Results

### Main Results

Retrieval performance on the SER dataset:

| Method | Text→Image R@1 | Sketch→Image R@1 | Art→Image R@1 | Low-Res→Image R@1 |
|------|-------|-------|-------|-------|
| CLIP | 54.6 | 47.3 | 46.8 | 53.7 |
| BLIP | 55.8 | 48.2 | 47.5 | 51.5 |
| CLIP-Finetune | 71.4 | 71.0 | 52.2 | 71.2 |
| FreestyleRet | 80.1 | 75.3 | 73.0 | 78.0 |
| **Uni-Retrieval** | **83.2** | **84.5** | **76.9** | **87.4** |

### Ablation Study

Efficiency analysis:

| Method | Parameters | Q2I Inference (ms) | Text→Image Acc |
|------|--------|-------------|---------------|
| CLIP | 427M | 68ms | 54.6 |
| LanguageBind | 1200M | 372ms | 60.2 |
| GASKN | 33M | 12ms | 55.7 |
| **Uni-Retrieval** | 453M (+26M) | 77ms (+9ms) | **83.2** (+28.6) |

Multi-style hybrid queries:

| Method | T→I | T+S→I | I→T | I+S→T |
|------|-----|-------|-----|-------|
| CLIP-Finetune | 54.6 | 55.3 (+0.7) | 47.4 | 46.6 (-0.8) |
| VPT | 69.9 | 72.0 (+2.1) | 73.9 | 74.1 (+0.2) |
| **Uni-Retrieval** | 83.2 | **87.4 (+4.2)** | 81.7 | **83.3 (+1.6)** |

### Key Findings

- Uni-Retrieval completely surpasses all baselines on all retrieval styles, with Sketch$\rightarrow$Image R@1 reaching 84.5% (9.2 percentage points higher than FreestyleRet) and Low-Res$\rightarrow$Image R@1 reaching 87.4%.
- Only adding 26M parameters and 9ms inference time, yet yielding a massive performance boost (Text$\rightarrow$Image +28.6%), which is extremely efficient.
- Uni-Retrieval achieves even greater gains in multi-style hybrid queries (+4.2 vs CLIP-Finetune +0.7), demonstrating that the compositions in Prompt Bank are indeed effective.
- It remains competitive on traditional retrieval datasets as well, not limited only to STEM educational scenarios.

## Highlights & Insights

- **The composability design of the Prompt Bank is highly ingenious**: Decoupling and storing style information allows unseen styles to be represented through linear combinations of tokens from known styles, inheriting a natural generalizability. This design paradigm can be transferred to other tasks requiring style/domain adaptation.
- **Extreme efficiency of freezing the backbone and training only the Prompt Bank**: Achieving a 50%+ performance improvement while introducing only about 6% additional parameters serves as an excellent paradigm for parameter-efficient learning.
- **The construction of the SER dataset is highly worth referencing**: 6,000 source images $\times$ 6 styles (text, speech, sketch, art, low-resolution) $\times$ 22+ STEM subjects. Its construction pipeline (human curation + AIGC) strikes a good balance between quality and diversity.

## Limitations & Future Work

- Although the SER dataset covers 22+ subjects, the sample distribution per subject may be imbalanced.
- Audio queries are processed via GPT-4o for speech-to-text, which is an indirect pipeline; end-to-end audio-image retrieval could be explored.
- Prototype learning depends heavily on the quality of pre-trained encoders; for highly specialized STEM diagrams (such as circuit diagrams or chemical structures), dedicated encoders might be necessary.
- Sensitivity analysis regarding the number of tokens $N$ in the Prompt Bank and the top-$n$ retrieved tokens remains unexplored.
- Queries in real-world educational scenarios may be even more ambiguous and incomplete, requiring a more robust design.

## Related Work & Insights

- **vs FreestyleRet**: The strongest baseline for multi-style retrieval currently, which this work fully surpasses on SER. The key distinction lies in the Prompt Bank's capability for continuous updating and composition.
- **vs CoCoOP / MaPLe**: Prompt learning-based methods, but their prompts are globally shared and cannot be dynamically selected based on the input style. Uni-Retrieval's Prompt Bank is query-conditioned.
- **vs VPT**: Visual Prompt Tuning also inserts learnable prompts into visual tokens, but lacks a style-conditioned selection mechanism.
- Increasingly diversified retrieval demands in the AI4EDU field provide new application-driven directions for multimodal retrieval research.

## Rating

- Novelty: ⭐⭐⭐⭐ The dynamic prompt selection combining Prompt Bank and prototype matching is an innovative design. The contributions of the new task and dataset are also valuable.
- Experimental Thoroughness: ⭐⭐⭐ The main experiments present broad coverage but lack some crucial ablation studies (e.g., the size of Prompt Bank, the impact of top-$n$).
- Writing Quality: ⭐⭐⭐ Generally clear, though descriptions in some parts lack conciseness, and some mathematical notations are inconsistent.
- Value: ⭐⭐⭐⭐ The new dataset and task definition drive advancements in both educational AI and multi-style retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [\[ACL 2025\] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)

</div>

<!-- RELATED:END -->
