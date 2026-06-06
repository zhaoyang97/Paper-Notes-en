---
title: >-
  [Paper Note] SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification
description: >-
  [ACL2026][LLM Safety][Event Causality Identification] SERE argues that example selection in Event Causality Identification (ECI) should not rely solely on semantic similarity. Instead…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Event Causality Identification"
  - "Structural Retrieval"
  - "In-Context Learning"
  - "Causal Hallucination"
  - "ConceptNet"
date: 2026-05-08
content_hash: 2afce486d28c0b58
---

# SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification

**Conference**: ACL2026  
**arXiv**: [2605.03701](https://arxiv.org/abs/2605.03701)  
**Code**: https://github.com/DMIRLAB-Group/SERE  
**Area**: LLM Safety / Event Causality Identification  
**Keywords**: Event Causality Identification, Structural Retrieval, In-Context Learning, Causal Hallucination, ConceptNet

## TL;DR
SERE argues that example selection in Event Causality Identification (ECI) should not rely solely on semantic similarity. Instead, it proposes retrieving examples with structurally similar conceptual paths, syntactic trees, and causal patterns to reduce causal over-prediction in LLMs during few-shot reasoning.

## Background & Motivation
**Background**: ECI requires models to determine whether a causal relationship exists between two events in a context. While traditional methods rely on fine-tuning encoders like BERT/RoBERTa, LLMs enable training-free or few-shot reasoning, which is suitable for ECI scenarios with scarce labeled data or requiring rapid transfer.

**Limitations of Prior Work**: A typical issue when prompting LLMs directly for ECI is causal hallucination, where models over-predict the presence of causality. While CoT can slightly improve the reasoning process, it fails to stably reduce false positives; pattern-based methods like Dr.ECI improve recall but often at the expense of precision.

**Key Challenge**: Similarity in ECI is fundamentally not about general sentence semantic similarity, but about the similarity of structural relationships between events. Two examples might share words like "rain" and "road" but have opposite causal labels; conversely, examples with dissimilar surface forms might share the same causal structure, making them better in-context demonstrations.

**Goal**: The authors aim to build an example retrieval framework that does not require fine-tuning LLMs, making few-shot examples closer to the causal reasoning structure of the target instance, thereby significantly improving precision while maintaining recall.

**Key Insight**: SERE decomposes "structure" into three types of signals: the conceptual path from the source event to the target event on ConceptNet, the dependency syntactic tree of the context, and predefined causal graph patterns. The first two handle continuous scoring, while the latter performs hard filtering.

**Core Idea**: The framework first ranks candidate examples using conceptual path edit distance and syntactic tree edit distance, then filters them using causal patterns extracted by an LLM. Finally, a small number of structurally similar, label-balanced (positive/negative) examples are selected for LLM reasoning.

## Method

### Overall Architecture
SERE consists of three modules. First is Joint Structural Metrics, which calculates conceptual path similarity and syntactic structure similarity for the target and corpus instances, then aggregates them into a structural score. Second is Causal Pattern Filtering, where a lightweight LLM prompt extracts coarse-grained causal patterns for candidate examples, retaining only those consistent with the target. Third is LLM Reasoning, which incorporates the retrieved few-shot examples into the prompt for the Reasoner to output "Yes" or "No" in a binary classification format.

In terms of workflow, SERE acts more as a retriever for "selecting appropriate cases for LLMs" rather than another causal classifier. Its key hypothesis is that LLMs are not entirely incapable of causal reasoning but are misled by incorrect demonstrations or surface semantics in few-shot settings; as long as the demonstration structures are aligned, the model's decision boundaries become more conservative and precise.

### Key Designs
1.  **Conceptual Path Metric**:
    - **Function**: Measures whether the conceptual relationship between two event pairs in an external common-sense graph is similar.
    - **Mechanism**: Event spans are matched to ConceptNet nodes using Contriever-msmarco, and Neo4j is used to query the shortest path between the source and target event nodes. After obtaining conceptual paths for both the target and candidate samples, normalized similarity is calculated using edit distance as $1-ED(path_x,path_q)/\max(|path_x|,|path_q|)$.
    - **Design Motivation**: Common-sense paths between events can provide implicit relationship clues when explicit causal connectors are absent, which is closer to the structural requirements of ECI tasks than comparing sentence embeddings.

2.  **Syntactic Metric**:
    - **Function**: Measures whether the syntactic organization of the context expressing the causal relationship is similar.
    - **Mechanism**: Dependency trees are built using spaCy, with multi-sentence texts connected to a manual root node. The Tree Edit Distance (TED) between the target and candidate samples is then calculated. Similarity is defined as $e^{-0.05\cdot TED(tree_x,tree_q)}$, which is then weighted and combined with the conceptual path score.
    - **Design Motivation**: Causal relations are often carried by clauses, adverbials, prepositional structures, or cross-sentence coreference. Syntactic trees capture "how events are expressed," complementing the external knowledge from ConceptNet.

3.  **Causal Pattern Filtering & Balance**:
    - **Function**: To avoid retrieving examples with similar structures but different causal graph types, while suppressing the LLM's bias toward predicting causality.
    - **Mechanism**: SERE defines coarse-grained causal patterns such as Direct, Chain, Collider, Fork, and Coreference. Positive examples have their patterns extracted via an LLM PatternExtractor, while negative examples are labeled "No." Only candidates consistent with the target pattern are kept. A top-k selection is then made from high-scoring examples, maintaining a balanced ratio of positive and negative demonstrations.
    - **Design Motivation**: Providing only positive examples exacerbates causal hallucination, while providing semantic neighbors might introduce opposing labels. Pattern filtering and balancing together transform the prompt into a "structured discriminative boundary."

### Loss & Training
The primary method does not train LLMs or update classifier parameters. In implementation, the ConceptNet node matching threshold is 0.6, weights for conceptual path and syntactic scores are 0.5 each, top-2 demonstrations are selected by default, and LLM temperature is set to 0 to reduce randomness. Main experiments use GPT-4o-mini and Gemini-1.5-pro APIs; the appendix validates transplanting SERE to a fine-tuning setting using LoRA on Qwen2.5-3B-Inst.

## Key Experimental Results

### Main Results
The main experiments cover three ECI datasets: ESC, CTB, and MAVEN-ERE. Metrics used are Precision, Recall, and F1. The following table highlights F1 scores, emphasizing SERE's improvement over LLM baselines.

| LLM | Method | ESC F1 | CTB F1 | MAVEN-ERE F1 | Main Change |
|-----|------|--------|--------|--------------|----------|
| GPT-4o-mini | Base | 42.3 | 10.1 | 36.9 | Direct prompt, high recall but low precision |
| GPT-4o-mini | CoT | 43.1 | 11.6 | 39.4 | Slight improvement |
| GPT-4o-mini | Dr.ECI | 46.1 | 15.1 | 40.7 | Structural reasoning improves recall |
| GPT-4o-mini | SERE | 49.9 | 20.0 | 42.3 | Best across all three datasets |
| Gemini-1.5-pro | Base | 37.0 | 9.0 | 34.6 | Similar over-prediction issues |
| Gemini-1.5-pro | Dr.ECI | 41.3 | 13.3 | 37.3 | Improvement but insufficient |
| Gemini-1.5-pro | SERE | 45.2 | 17.4 | 39.9 | Stable improvement |

On GPT-4o-mini, SERE improves over the Base by 7.6, 9.9, and 5.4 F1 points on ESC, CTB, and MAVEN-ERE respectively. More importantly, it primarily reduces false positives by increasing Precision rather than simply increasing Recall.

### Ablation Study

| Configuration | ESC F1 | CTB F1 | MAVEN-ERE F1 | Description |
|------|--------|--------|--------------|------|
| SERE | 49.9 | 20.0 | 42.3 | Full structural retrieval |
| Conceptual Path Only | 46.7 | 18.9 | 39.2 | External paths are effective but incomplete |
| Syntactic Only | 44.5 | 18.9 | 38.7 | Syntactic structure alone is weak |
| Causal Pattern Only | 47.1 | 17.3 | 38.3 | Pattern constraints control false positives but lack coverage |
| w/o Conceptual Path | 46.6 | 18.8 | 40.8 | Performance drops across all datasets |
| w/o Syntactic | 48.1 | 18.9 | 40.5 | Syntax is complementary to other signals |
| w/o Causal Pattern | 47.5 | 18.2 | 39.4 | Hard filtering provides significant contribution |

| Retrieval Type | ESC F1 | CTB F1 | MAVEN-ERE F1 | Conclusion |
|----------|--------|--------|--------------|------|
| Base | 42.3 | 10.1 | 36.9 | No examples |
| Random | 46.6 | 13.9 | 39.1 | Examples themselves are helpful |
| Contriever-msmarco | 46.2 | 18.8 | 37.9 | Semantic retrieval is unstable |
| BM25 | 46.5 | 17.1 | 33.4 | Surface retrieval hurts MAVEN-ERE |
| SERE | 49.9 | 20.0 | 42.3 | Structural retrieval is the most stable |

### Key Findings
- The advantage of SERE lies not in making the model more aggressive in predicting causality, but in making it more cautious. The paper notes that Base and CoT tend toward low Precision and high Recall; SERE reduces instances where non-causal relations are misidentified by using structurally similar examples.
- Top-k is not better when larger. Top-2 performs best on ESC and MAVEN-ERE, while top-4 is slightly better only on CTB; performance generally declines at top-6, suggesting that excessive demonstrations dilute the structural signal.
- In the fine-tuning experiments in the appendix, Qwen2.5-3B-Inst + SERE achieved 76.4/93.3 F1 on ESC/CTB, outperforming both Base and CPATT in the same setting. This indicates that structural examples are not only applicable to API ICL but also serve as an effective data organization method for fine-tuning.
- Regarding cost, SERE takes an average of 21.85 seconds, higher than CoT (4.02s) and Dr.ECI (14.26s). However, its output token count is lower than Dr.ECI; the extra time is primarily consumed by CPU-based structural matching and tree edit distance calculations.

## Highlights & Insights
- The paper identifies a key aspect of ECI: similar sentences do not equal similar causal structures. This observation is intuitive but directly explains why standard dense retrieval picks incorrect examples for causal tasks.
- The three signals—Conceptual Path, Syntactic Tree, and Causal Pattern—correspond to external common sense, linguistic expression, and causal graph types, respectively. The combination method is clear and interpretable.
- SERE restricts the LLM's role to pattern extraction and final judgment, with the core retrieval logic controlled by structural algorithms. This "structural constraint + LLM reasoning" division reduces drift inherent in pure prompting methods.
- For safety-sensitive causal inference tasks, improving Precision is of significant practical value. Reporting fewer causal relationships is sometimes more reliable than hallucinating causal chains.

## Limitations & Future Work
- The cost of structural retrieval is high, particularly for tree edit distance and ranking candidates in large corpora; practical deployment would require caching, indexing, or parallelization.
- Causal Pattern Filtering relies on LLM extraction; errors in extraction directly lead to filtering out correct examples or retaining incorrect ones.
- Experiments mainly cover three English ECI datasets; the quality of conceptual path matching in cross-lingual, open-domain news, or specialized domain texts requires further validation.
- The paper only uses structural signals for example retrieval and does not further train a structure-aware retriever, which may limit efficiency and generalization in large-scale settings.

## Related Work & Insights
- **vs Dr.ECI**: Dr.ECI decomposes reasoning via predefined causal patterns; SERE adopts the pattern idea for example filtering and adds ConceptNet paths and syntactic tree similarity.
- **vs BM25 / Contriever ICL**: Traditional retrievers optimize for surface or semantic similarity; SERE optimizes for task structural similarity, which better fits the discriminative needs of ECI.
- **vs CPATT**: While CPATT is a fine-tuning-based structural model, SERE focuses on training-free few-shot LLM calls. The appendix shows the SERE approach can be transferred to fine-tuning settings.
- **Implications for other tasks**: Tasks such as event temporal relation extraction, argument relation identification, and legal causal chain judgment could likely benefit from "structure-first" example retrieval rather than relying solely on embedding similarity.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying three types of structural signals to ECI few-shot retrieval is a precise approach to the problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis including main experiments, ablation, retrieval baselines, top-k, fine-tuning, and cost.
- Writing Quality: ⭐⭐⭐⭐ Logical structure; though some implementation details in the appendix are long, they aid reproducibility.
- Value: ⭐⭐⭐⭐ Highly practical reference for causal reasoning, structured ICL, and mitigating LLM over-prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs](astra_an_automated_framework_for_strategy_discovery_retrieval_and_evolution_for_.md)
- [\[ICCV 2025\] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](../../ICCV2025/llm_safety/asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
