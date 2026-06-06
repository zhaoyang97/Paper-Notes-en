---
title: >-
  [Paper Note] ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models
description: >-
  [AAAI 2026][AIGC Detection][Knowledge Overshadowing] This paper proposes ActiShade, a framework that detects "overshadowed" key phrases in LLM multi-hop reasoning via Gaussian noise perturbation…
tags:
  - "AAAI 2026"
  - "AIGC Detection"
  - "Knowledge Overshadowing"
  - "Multi-Hop Reasoning"
  - "Retrieval-Augmented Generation"
  - "Gaussian Perturbation Detection"
  - "Contrastive Learning Retriever"
date: 2026-05-08
content_hash: 128fe46d411d4354
---

# ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2601.07260](https://arxiv.org/abs/2601.07260)  
**Code**: Not available  
**Area**: AIGC Detection
**Keywords**: Knowledge Overshadowing, Multi-Hop Reasoning, Retrieval-Augmented Generation, Gaussian Perturbation Detection, Contrastive Learning Retriever

## TL;DR
This paper proposes ActiShade, a framework that detects "overshadowed" key phrases in LLM multi-hop reasoning via Gaussian noise perturbation, retrieves supplementary documents using a customized contrastive learning retriever, and iteratively reformulates queries to mitigate error accumulation caused by knowledge overshadowing. ActiShade significantly outperforms DRAGIN and other state-of-the-art methods on HotpotQA, 2WikiMQA, and MuSiQue.

## Background & Motivation
Multi-hop reasoning requires LLMs to integrate multiple conditions to answer questions correctly. Existing multi-round RAG methods (e.g., IRCoT, Iter-RetGen, DRAGIN) rely on LLM-generated content as retrieval queries for subsequent rounds. However, LLMs exhibit **Knowledge Overshadowing** during generation — certain critical conditions in the query are overshadowed by other dominant conditions, leading to incomplete or inaccurate generation. For example, when a query simultaneously mentions "Te Deum in D Major" and "Gloria in D Major," the former may dominate and cause the LLM to neglect the latter, resulting in retrieval of irrelevant documents and compounding errors across iterations.

This problem is particularly severe in multi-hop scenarios, where each step of the reasoning chain depends on the output of the previous step — once key information is overshadowed at any step, all subsequent steps deviate from the correct path. Existing methods either use LLM outputs directly as queries (Iter-RetGen) or decompose questions into sub-questions (SelfASK), neither of which directly addresses the LLM's selective neglect of conditions during generation.

## Core Problem
How to detect and activate knowledge overshadowed by LLMs during multi-hop reasoning, so that multi-round retrieval can supplement neglected critical information and reduce error accumulation?

This problem is significant because: (1) knowledge overshadowing is an intrinsic limitation of LLMs, distinct from hallucination — it stems from uneven attention allocation across multiple input conditions; (2) this bias amplifies exponentially across iterative rounds; (3) existing methods lack targeted mechanisms to address this phenomenon.

## Method

### Overall Architecture
ActiShade is an iterative multi-round retrieval framework. Each round consists of three modules:
1. **Knowledge Overshadowing Detection (GaP)**: Given the current query, detects which key phrase is being neglected by the LLM.
2. **Overshadowed-Phrase-Guided Retrieval**: Concatenates the query and the overshadowed phrase as input to a trained retriever to find supplementary documents.
3. **Query Reformulation**: Uses retrieved documents to prompt the LLM to generate a new query for the next iteration.

Termination condition: the LLM determines that the current query has become a single-hop question (one additional retrieval round suffices), or the maximum number of iterations is reached. Finally, the original question and all relevant documents retrieved across iterations are fed to the LLM to produce the final answer.

### Key Designs

1. **GaP (Gaussian Perturbation-based Detection)**:

    - **Step 1 – Key Phrase Extraction**: SpaCy is used to extract named entities and meaningful tokens (NOUN/ADJ/VERB/PROPN/NUM/ADV), removing stopwords to obtain a candidate key phrase set $P=\{p_1,...,p_n\}$.
    - **Step 2 – Gaussian Perturbation**: For each candidate phrase $p_i$, Gaussian noise is injected into its embedding: $\tilde{H}_{p_i} = H + m_{p_i} \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$, where $m_{p_i}$ is a binary mask with ones only at the token positions of $p_i$.
    - **Step 3 – Overshadowing Measurement**: The cosine similarity between the LLM output distributions before and after perturbation is compared. The candidate phrase with the highest similarity is identified as overshadowed — the intuition being that an unchanged output after noise injection indicates the model was not utilizing that phrase in the first place.

   Compared to the prior CoDA method (which directly deletes tokens), GaP preserves the complete structure of the query and only adds noise in the embedding space without disrupting the reasoning chain. Experiments show that CoDA can actually hurt performance in multi-hop settings.

2. **Fine-Grained Contrastive Learning Retriever**:

    - Documents are categorized into three types: **positive** (relevant to both the query and the overshadowed phrase), **semi-positive** (relevant to the query but not directly related to the overshadowed phrase), and **negative** (irrelevant).
    - Two loss terms are designed: $\mathcal{L}_1$ encourages positive documents to score higher than semi-positive and negative documents; $\mathcal{L}_2$ encourages semi-positive documents to collectively score higher than negative documents.
    - The combined loss is $\mathcal{L} = \alpha \mathcal{L}_1 + (1-\alpha) \mathcal{L}_2$ ($\alpha=0.7$), enforcing the ranking $D^+ > D^* > D^-$.
    - The retriever is fine-tuned from contriever-msmarco on training data constructed from MuSiQue (5,000 samples: 3,500 train / 750 validation / 750 test).
    - Input is the concatenation of the query and the overshadowed phrase.

3. **Query Formulation**:

    - **Document Selection**: The LLM assesses the relevance of each retrieved document (outputting Yes/No probabilities) and selects the one with the highest probability.
    - **Query Generation**: Based on the selected document, the LLM generates a new query that makes implicit reasoning results explicit. For example, "What is the famous bridge in the birthplace of the composer of Gloria in D Major?" → after retrieving a document about Vivaldi → "What is the famous bridge in the birthplace of Antonio Vivaldi?"
    - **Termination Judgment**: The LLM determines whether the new query is a single-hop question; if so, one additional retrieval round is performed before termination.

## Key Experimental Results

| Dataset | Metric | ActiShade | Prev. SOTA (DRAGIN) | Gain |
|--------|------|-----------|---------------------|------|
| MuSiQue (Llama-3-8B) | F1 | 26.94 | 22.61 | +4.33 |
| 2WikiMQA (Llama-3-8B) | F1 | 56.33 | 52.52 | +3.81 |
| HotpotQA (Llama-3-8B) | F1 | 46.02 | 42.31 | +3.71 |
| MuSiQue (Qwen2.5-7B) | F1 | 26.11 | 22.01 | +4.10 |
| HotpotQA (Qwen2.5-7B) | F1 | 50.47 | 45.87 | +4.60 |
| MuSiQue (Qwen2.5-14B) | F1 | 27.47 | 24.11 | +3.36 |
| HotpotQA (Qwen2.5-14B) | F1 | 53.29 | 49.87 | +3.42 |

### Ablation Study
- **GaP vs. CoDA**: Under the multi-round retrieval setting, GaP (F1=26.94) substantially outperforms CoDA (F1=21.23); CoDA performs even below the detection-free baseline on MuSiQue and 2WikiMQA, confirming that token removal disrupts multi-hop reasoning chains.
- **Fine-Grained Contrastive Learning (FCL) vs. Standard Contrastive Learning (SCL)**: FCL substantially outperforms SCL in positive document Recall@1 (75.33 vs. 57.84), with correspondingly higher downstream QA F1 (26.94 vs. 24.10).
- **Document Selection Step**: Removing LLM-based selection and using the top-ranked retrieved document directly decreases F1 from 26.94 to 25.10 on MuSiQue.
- **Noise Standard Deviation σ**: The optimal value is 0.1; performance remains relatively stable in the range [0.05, 0.5], indicating robustness to this hyperparameter.
- **Model Scale**: Performance improves with model size (7B→14B), demonstrating scalability.
- **Cross-Dataset Generalization**: The retriever is trained only on MuSiQue but consistently outperforms all baselines on HotpotQA and 2WikiMQA.

## Highlights & Insights
- **Elegant Design of Knowledge Overshadowing Detection**: Gaussian perturbation replaces token deletion to detect neglected information. The core intuition — "if the output remains unchanged after noise injection, the model was not using that information to begin with" — is simple yet highly effective.
- **Three-Level Document Classification**: The positive/semi-positive/negative distinction is more fine-grained than standard binary contrast. Semi-positive documents are "documents useful for the question but irrelevant to the current reasoning step," a design that aligns closely with the step-by-step decomposition nature of multi-hop reasoning.
- **Explicit Representation of Implicit Reasoning**: Query reformulation not only rephrases the question but encodes intermediate reasoning results into the new query (e.g., "composer of Gloria" → "Vivaldi"), enabling more precise subsequent retrieval.
- **Strong Generalization**: The retriever, trained only on MuSiQue, generalizes to other datasets, suggesting it has learned a general capability of "attending to overshadowed phrases."

## Limitations & Future Work
- **Embedding-Level Access Requirement**: GaP requires access to the LLM's token embedding layer and output distribution, making it inapplicable to closed-source API-based models.
- **Computational Overhead**: Each candidate phrase requires a separate forward pass to compute the perturbed output distribution, incurring non-trivial cost when many candidates are present.
- **Limited Model Scale Evaluation**: Hardware constraints precluded testing on larger models (e.g., 70B); it remains unclear whether knowledge overshadowing is less severe in larger models.
- **SpaCy Dependency**: Key phrase extraction relies on SpaCy's NER and POS tagging, which may be inaccurate for non-English languages or domain-specific text.
- **Single Overshadowed Phrase per Round**: Only the single most overshadowed phrase is selected per round, whereas multiple phrases may be simultaneously overshadowed in practice.
- **Future Directions**: (1) Parameter-efficient overshadowing detection to avoid full forward passes per candidate; (2) transferring the GaP concept to visual information overshadowing detection in vision-language models; (3) joint handling of multiple overshadowed phrases.

## Related Work & Insights
- **vs. DRAGIN**: DRAGIN detects information needs via self-attention and dynamically triggers retrieval during generation. ActiShade goes further by not only determining "when to retrieve" but also identifying "what information is being neglected" and customizing the retrieval strategy accordingly. The fundamental distinction is that DRAGIN is reactively triggered, whereas ActiShade actively detects and compensates.
- **vs. SelfASK**: SelfASK decomposes complex questions into a sequence of sub-questions. ActiShade performs no explicit decomposition; instead, it naturally resolves questions step by step through overshadowing detection and query reformulation, avoiding the instability associated with decomposition quality.
- **vs. CoDA**: CoDA also detects knowledge overshadowing but via token deletion. ActiShade's GaP replaces deletion with Gaussian perturbation, preserving the structural integrity of the query. Experiments confirm that CoDA can produce adverse effects in multi-hop settings.

The fine-grained three-level contrastive loss design (positive/semi-positive/negative) is broadly applicable to any retrieval task requiring fine-grained relevance ranking. The "explicit representation of implicit reasoning" strategy in query reformulation shares conceptual similarities with CoT refinement and could potentially be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Knowledge overshadowing detection is a novel perspective, and the GaP method is cleverly designed, though the overall framework remains a variant of multi-round RAG.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, three LLMs, multi-dimensional ablation (detection method / retriever training / query reformulation), interpretability visualization, and cross-dataset generalization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and detailed method descriptions; the case study is intuitive, though notation is somewhat dense.
- **Value**: ⭐⭐⭐⭐ — Knowledge overshadowing is a genuine pain point in multi-hop RAG, and the solution is practical; however, the requirement for embedding-level access limits its applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](../../NeurIPS2025/aigc_detection/reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](../../ICML2026/aigc_detection/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)
- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](../../NeurIPS2025/aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)
- [\[AAAI 2026\] BAID: A Benchmark for Bias Assessment of AI Detectors](baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)

</div>

<!-- RELATED:END -->
