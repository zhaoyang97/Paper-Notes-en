---
title: >-
  [Paper Note] From Documents to Segments: A Contextual Reformulation for Topic Assignment
description: >-
  [ACL 2026][Interpretability][Topic Modeling] This paper shifts the basic unit of topic assignment from documents to segments, proposing Segment-Based Topic Allocation (SBTA) and the SemEval-STM dataset. It demonstrates t…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Topic Modeling"
  - "Text Segmentation"
  - "Topic Contamination"
  - "SemEval-STM"
  - "segment intrusion"
date: 2026-05-08
content_hash: 52365921cebb758d
---

# From Documents to Segments: A Contextual Reformulation for Topic Assignment

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.17714](https://arxiv.org/abs/2605.17714)  
**Code**: Dataset https://huggingface.co/datasets/LG-AI-Research/SemEval-STM; GitHub repo not provided yet  
**Area**: Topic Modeling / Interpretable Text Analysis / NLP Understanding  
**Keywords**: Topic Modeling, Text Segmentation, Topic Contamination, SemEval-STM, segment intrusion  

## TL;DR
This paper shifts the basic unit of topic assignment from documents to segments, proposing Segment-Based Topic Allocation (SBTA) and the SemEval-STM dataset. It demonstrates that assigning topics by semantic segments in multi-topic short texts significantly improves topic purity, interpretability, and downstream retrieval utility.

## Background & Motivation
**Background**: Traditional topic modeling typically uses the document as the fundamental unit, representing each document as a mixture of one or more topics. LDA, BERTopic, and recent LLM-based topic modeling models all follow this paradigm, merely enhancing topic generation, labeling, or semantic representation.

**Limitations of Prior Work**: In real-world applications, many texts do not discuss only a single topic. A product review might cover price, quality, service, and appearance simultaneously; employee feedback might address compensation, culture, and promotion. Document-level topic assignment mixes these disparate topics, leading to topic contamination: retrieving a specific topic returns entire multi-topic documents rather than the most relevant specific statements.

**Key Challenge**: The objective of topic modeling is to obtain clear, interpretable, and retrievable topic sets, yet documents are often much coarser than the topics themselves. The more heterogeneous a document is, the more likely document-level assignment is to introduce noise into topic clusters.

**Goal**: The authors aim to formally redefine topic assignment: assigning topics to short, semantically consistent text segments instead of documents, and creating a dataset and task to evaluate this setting.

**Key Insight**: The paper leverages aspect-based sentiment analysis (ABSA) data, as ABSA naturally uses aspect labels that serve as proxy topics. LLMs are used to extract text spans corresponding to each aspect.

**Core Idea**: By changing the basic unit of topic allocation to segments, each topic aggregates truly relevant semantic fragments rather than full documents containing off-topic content.

## Method
SBTA can be viewed as a "granularity reconstruction of topic modeling." It does not require reinventing all topic modeling algorithms but rather changes the units of input and output: documents are first partitioned into segments expressing single or few related topics, followed by topic assignment, cluster evaluation, and human consistency assessment at the segment level.

### Overall Architecture
Given a corpus $\mathcal{D}=\{d_1,\ldots,d_D\}$ and $K$ topics, Document-Based Topic Allocation (DBTA) associates topics with the entire document. SBTA constructs a set of segments $\mathcal{Q}_d$ for each document, where each segment is a combination of a contiguous token span $[i:j]$ and a set of topics $\mathcal{T}$. If a user queries topic $k$, the system returns $\mathcal{Q}_{d,k}=\{Q\in\mathcal{Q}_d|k\in\mathcal{T}(Q)\}$, which represents the document segments actually discussing that topic.

At the data level, the authors constructed SemEval-STM. Based on the laptop and restaurant domains of SemEval-2016 ABSA, aspect labels are used as topic proxies. LLMs first identify relevant segments for each topic, and the authors perform post-processing, manual merging, and reassignment to create a benchmark that supports direct comparisons between DBTA and SBTA.

### Key Designs
1. **Segment-based Topic Allocation Task Definition**:

	- **Function**: Transitions the topic assignment unit from the document level to the segment level.
	- **Mechanism**: A segment is defined as $([i:j],\mathcal{T})$, where $[i:j]$ is a contiguous text span and $\mathcal{T}$ is the set of topics involved in that span. A segment typically contains only one or a few topics, ensuring higher purity than the full document.
	- **Design Motivation**: In practical analysis, users often want to identify "which specific sentences discuss price / service / quality" rather than retrieving long reviews containing multiple unrelated topics.

2. **SemEval-STM Construction Process**:

	- **Function**: Provides an evaluable dataset for SBTA.
	- **Mechanism**: Use o3-mini to extract maximal contiguous spans based on topics and documents; topics with fewer than 10 segments were discarded. The laptop domain was reduced from 76 to 33 topics and then manually merged into 23; the restaurant domain was organized into 11 topics. DBTA and SBTA utilize the same topic set for fair comparison.
	- **Design Motivation**: Directly using document-level data would give SBTA an unfair advantage; SemEval-STM targets short texts where multi-topic overlap exists but off-topic content is the primary focus of the comparison.

3. **Segment Intrusion Evaluation**:

	- **Function**: Evaluates the semantic consistency of segment topic clusters from a human interpretability perspective.
	- **Mechanism**: Four task types were constructed: single/double intrusion and cross-domain easy / intra-domain hard. Humans or LLMs must identify an "intruder" segment that does not semantically belong to a given topic cluster. Higher success rates indicate more consistent topic clusters.
	- **Design Motivation**: Traditional word intrusion only checks topic keywords and cannot measure contextual coherence; segment intrusion better aligns with the object granularity of SBTA.

### Loss & Training
The paper does not propose an end-to-end neural training loss but rather focuses on task reformulation, data construction, and evaluation protocols. Experiments utilize LDA, BERTopic, and various LLM-based topic assignment methods as baselines. For LLM methods, a segment and a predefined list of candidate topics are provided to the model to select the most relevant topics, similar to the assignment phase in TopicGPT but at the segment level.

## Key Experimental Results

### Main Results
| Comparison | Domain | DBTA | SBTA | Conclusion |
|--------|--------|------|------|------|
| DB Index ↓ | Laptop | 20.1768 | 6.2767 | SBTA clusters are more compact |
| CH Index ↑ | Laptop | 3.0037 | 15.5184 | SBTA shows stronger inter-class separation |
| Silhouette ↑ | Laptop | -0.0522 | 0.0460 | SBTA improves from negative to positive |
| XB Index ↓ | Laptop | 95.8645 | 10.8348 | SBTA significantly reduces intra-class mixing |
| DB Index ↓ | Restaurant | 70.9506 | 6.6657 | DBTA is particularly chaotic on restaurant data |
| CH Index ↑ | Restaurant | 1.7204 | 22.6709 | SBTA reveals clearer topic structures |
| Silhouette ↑ | Restaurant | -0.0303 | 0.0222 | SBTA is more separable |
| XB Index ↓ | Restaurant | 1233.5519 | 12.1985 | Segment-level assignment vastly reduces topic contamination |

### Ablation Study
| Task / Metric | Method | Laptop | Restaurant | Description |
|-------------|------|--------|------------|------|
| Label-based F1 ↑ | LDA | 0.3577 | 0.4512 | Traditional topic models perform weakly |
| Label-based F1 ↑ | BERTopic | 0.5102 | 0.6692 | Embedding clustering shows significant gains |
| Label-based F1 ↑ | DeepSeek-v3 | 0.7383 | 0.8278 | LLM assignment is highly effective |
| Label-based F1 ↑ | Claude-3.7-Sonnet | 0.7182 | 0.8353 | Best model on the restaurant domain |
| Inter-annotator $\kappa$ | Laptop intrusion | easy single 1.0000 / easy double 1.0000 / hard single 0.9519 / hard double 0.8650 | - | "Hard double" is most difficult, but consistency remains high |
| Inter-annotator $\kappa$ | Restaurant intrusion | easy single 0.9514 / easy double 0.9550 / hard single 0.9753 / hard double 0.8800 | - | Segment intrusion evaluation has stable human consensus |

### Key Findings
- SBTA significantly outperforms DBTA across clustering metrics, indicating that topic contamination stems primarily from the document unit being too coarse rather than model weakness.
- After topic shuffling, SBTA's clustering metrics drop more sharply, showing that the original SBTA structure carries strong semantic organization; DBTA was less sensitive to shuffling, exposing its already loose clusters.
- Traditional coherence metrics are unstable for SBTA because they rely on word co-occurrence, which is naturally sparse in short segments.
- LLMs are significantly stronger at label-based topic assignment than LDA/BERTopic, but segment intrusion results show many models still perform below human levels in fine-grained semantic consistency.

## Highlights & Insights
- The core contribution of this paper is the shift in unit rather than stacking models. it identifies that many interpretability issues in topic modeling arise because "documents are not the appropriate atomic unit."
- The construction of SemEval-STM is clever; using ABSA aspect labels provides natural weak supervision, avoiding expensive manual topic labeling from scratch while retaining the complexity of multi-topic interweaving.
- Segment intrusion is an insightful evaluation metric. It moves from "how similar are topic words" to "whether these semantic fragments can be perceived by humans as the same category," aligning more closely with actual analysis needs.
- For practical feedback, surveys, logs, and review analysis, SBTA is more aligned with workflows than DBTA because users typically require specific evidence sentences rather than entire documents.

## Limitations & Future Work
- Segment extraction relies on LLMs; despite manual post-processing, boundary inconsistencies and automatic system bias may persist.
- Traditional coherence metrics do not match SBTA's span-level objectives, suggesting a need for redesigned automatic evaluation frameworks.
- SemEval-STM consists primarily of short reviews; segment construction in long documents, news, transcripts, or customer service dialogues requires further validation.
- Many experiments use predefined topic lists. Real-world unsupervised deployment requires a full loop of topic generation and segment assignment, along with evaluation of topic drift and label merging quality.

## Related Work & Insights
- **vs LDA / BERTopic**: While LDA and BERTopic output document-level distributions or clusters, SBTA changes the fundamental object, enabling these methods or LLM assignments to operate on purer fragments.
- **vs TopicGPT**: TopicGPT uses fragments as interpretive evidence, but topics remain primarily document-oriented; this paper elevates segments to a formal topic assignment unit.
- **vs Topic Segmentation**: Topic segmentation focuses on where to split a document, while SBTA focuses on which topic to assign to the resulting segments and how to use them for modeling; the two are complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The task reformulation is clear and addresses a fundamental assumption in topic modeling with effective corrections.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ The data construction, DBTA/SBTA comparisons, LLM benchmarks, and intrusion evaluations are comprehensive, though long-document scenarios are lacking.
- **Writing Quality**: ⭐⭐⭐⭐☆ Motivations and examples are easy to follow; appendix tables are extensive.
- **Value**: ⭐⭐⭐⭐☆ Highly practical for user feedback analysis and enterprise text insights, providing a fine-grained evaluation direction for LLM-based topic modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)
- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ACL 2026\] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)

</div>

<!-- RELATED:END -->
