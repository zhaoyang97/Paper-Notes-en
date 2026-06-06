---
title: >-
  [Paper Note] UniversalRAG: Retrieval-Augmented Generation for Multimodal Corpora
description: >-
  [ACL 2026][Multimodal VLM][Retrieval-Augmented Generation] UniversalRAG proposes a universal any-to-any RAG framework that utilizes modality-aware routing and granularity-aware retrieval to dynamically select the most ap…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Retrieval-Augmented Generation"
  - "Multimodal"
  - "Routing Mechanism"
  - "Modality Gap"
  - "Granularity-aware Retrieval"
date: 2026-05-08
content_hash: 1f4d60c8cbbb2c35
---

# UniversalRAG: Retrieval-Augmented Generation for Multimodal Corpora

**Conference**: ACL 2026  
**arXiv**: [2504.20734](https://arxiv.org/abs/2504.20734)  
**Code**: https://github.com/universalrag  
**Area**: Multimodal VLM / Retrieval-Augmented Generation  
**Keywords**: Retrieval-Augmented Generation, Multimodal, Routing Mechanism, Modality Gap, Granularity-aware Retrieval

## TL;DR

UniversalRAG proposes a universal any-to-any RAG framework that utilizes modality-aware routing and granularity-aware retrieval to dynamically select the most appropriate knowledge sources from heterogeneous multimodal corpora (text, images, videos at various granularities). This approach avoids the modality gap problem inherent in unified embedding spaces and significantly outperforms single-modality and unified methods across 10 benchmarks.

## Background & Motivation

**Background**: Existing RAG methods are either limited to text corpora or extended to other modalities like images/videos, but typically remain modality-specific. When multimodal knowledge is required, the most straightforward approach is to aggregate all modality corpora into a unified embedding space.

**Limitations of Prior Work**: Although multimodal encoders attempt to align semantics across different modalities, a "modality gap" persists in practice—queries tend to cluster more closely with knowledge items of the same modality. This leads to retrieval bias toward intra-modal knowledge and the omission of relevant content from other modalities. Furthermore, different queries require different granularities of knowledge (simple factual questions need passages, while complex analytical questions need full documents or videos), and fixed-granularity retrieval is often suboptimal.

**Key Challenge**: How can a unified framework handle multimodal knowledge while avoiding retrieval bias caused by the modality gap and dynamically adjusting retrieval granularity based on query complexity?

**Goal**: To design a "one-stop" RAG framework that flexibly adapts to knowledge requirements across different modalities and granularities while maintaining retrieval efficiency.

**Key Insight**: Rather than forcing all modalities into a unified space, it is more effective to maintain modality-specific embedding spaces and use intelligent routing to dynamically select the most appropriate modality-granularity pairs.

**Core Idea**: A two-layer routing mechanism: (1) Modality-aware routing: predicts the required modalities and routes to the corresponding corpora; (2) Granularity-aware routing: further selects the most suitable granularity (passage/document/table/snippet/full video, etc.) within each modality.

## Method

### Overall Architecture

The end-to-end pipeline of UniversalRAG is divided into three key stages:

1.  **Corpus Organization**: Multiple independent corpora are constructed based on modality and granularity. For text, passage-level and document-level are stored separately; for video, short snippets (under 3 minutes) and full videos are stored; images remain as-is. Each corpus uses an independent modality-specific encoder to generate embeddings.
2.  **Routing Decision**: Given a query $\mathbf{q}$, the routing module $\mathcal{R}$ predicts the most suitable set of modality-granularity pairs $\{(m_1, g_1), (m_2, g_2), ...\}$, supporting both single-modality and cross-modal retrieval. The router also outputs a "No Retrieval" option for simple questions.
3.  **Generation**: Relevant content is retrieved from the corpora selected by the router and passed to an LVLM to generate the final answer.

### Key Designs

1.  **Modality-aware routing**:
    *   **Function**: Precisely identifies the modalities required by the query (text, image, video, etc.) to avoid the modality gap problem in cross-modal embedding spaces.
    *   **Mechanism**: Theoretical analysis (Proposition 1) proves that if the modality bias $\alpha$ is sufficiently large relative to the variance of semantic relevance, modality-specific retrieval strictly outperforms unified space retrieval. Specifically, routing is modeled as a multi-label classification problem to predict $\mathcal{R}(\mathbf{q}) = M_{\mathbf{q}} \subseteq \{\text{text, image, video, table}\}$. During inference, sigmoid probabilities for all modality-granularity pairs are thresholded (typically 0.8) to select combinations for parallel retrieval.
    *   **Design Motivation**: The root cause of the modality gap is that unified encoders are forced to align heterogeneous data from different sources into a single space. Maintaining independent modality-specific embedding spaces and controlling traffic with simple routing logic avoids numerical alignment difficulties while supporting seamless integration of new modalities.

2.  **Granularity-aware retrieval**:
    *   **Function**: Creates multi-level granularity representations for each modality (Text: passage/document; Video: snippet/full), allowing queries to route to the optimal granularity to balance information sufficiency and noise.
    *   **Mechanism**: The output space of the routing module is expanded to $\mathcal{R}: Q \to \{\varnothing\} \cup \mathcal{P}(\bigcup_{m \in M} \{m\} \times G_m)$, where $G_m$ is the set of granularities for modality $m$. Ground truth is automatically labeled based on dataset characteristics (e.g., multi-hop questions in HybridQA are labeled as document-level, while factual questions in NQ are passage-level) to train a multi-hot encoded routing classifier. The paper proves (Proposition 2) that different queries benefit from different granularities.
    *   **Design Motivation**: Fixed granularity easily leads to deficiencies at both ends—excessively fine granularity dilutes context, while excessively coarse granularity introduces irrelevant noise. Query-level granularity adaptation is dynamically adjusted through routing.

3.  **Flexibility in Routing Implementation**:
    *   **Function**: Two strategies coexist—training-based routing (fine-tuned using small models like Qwen3-VL-2B or InternVL3.5-1B) and training-free routing (using in-context learning with GPT-5 or Qwen3-VL-8B).
    *   **Mechanism**: The training-based router uses multi-label multi-hot encoding with cross-entropy loss for fast inference. The training-free router uses prompt templates (including target descriptions and few-shot examples) to support open-domain adaptation.
    *   **Design Motivation**: Training-based routers achieve extremely high in-domain accuracy (95%+) but poor out-of-distribution generalization; training-free routers offer stronger generalization. Robust plug-and-play solutions are achieved through an ensemble strategy balancing both.

## Key Experimental Results

### Main Results

Comprehensive comparison across 10 benchmarks (using Qwen3-VL-8B-Instruct as the generation backbone):

| Dataset | UniversalRAG (Trained) | UniRAG (Unified) | GME (Unified) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| MMLU | 74.39 | 70.06 | 70.41 | +4.3% |
| NQ | 38.65 EM | 19.30 EM | 20.05 EM | +99.2% |
| HotpotQA | 50.61 F1 | 29.71 F1 | 29.91 F1 | +70.4% |
| HybridQA | 11.05 EM | 2.85 EM | 3.00 EM | +288% |
| MRAG | 23.20 Acc | 19.05 Acc | 19.20 Acc | +21.3% |
| Average | 42.40 | 32.93 | 33.88 | **28.7% Relative Gain** |

### Ablation Study

| Configuration | Modality Accuracy | Average Metric | Description |
| :--- | :--- | :--- | :--- |
| Random Routing | 14.29% | 31.75 | Baseline by probability only |
| UniRAG (Unified Space) | 25.00% | 33.86 | Modality bias leads to skewed retrieval |
| UniversalRAG-Qwen3-2B | 95.28% | 42.40 | Routing accuracy is near-perfect |
| UniversalRAG-GPT-5 (Free) | 68.22% | 41.68 | Stronger generalization |
| Oracle (Perfect Routing) | 100% | 42.45 | Upper bound |

### Key Findings

*   **Modality Routing Effect**: VLM2Vec-V2 relies entirely on text retrieval regardless of the query, resulting in failure on video tasks; UniversalRAG adaptively retrieves balanced information from various modalities.
*   **Marginal Contribution of Granularity**: Increasing granularity levels from 1→2→3→4 yields continuous performance improvements, though the trend is not strictly monotonic.
*   **Efficiency Gains**: On large-scale corpora (>1M entries), UniversalRAG's end-to-end latency is lower than unified methods because modality-specific retrieval avoids searching across a massive unified index.
*   **OOD Generalization**: On 6 OOD datasets, the performance of the training-based router declines, while the training-free GPT-5 router remains stable.

## Highlights & Insights

*   **Theoretical Characterization of Modality Gap**: Through the formal analysis in Proposition 1, the paper provides a rigorous proof that modality-specific retrieval is superior to unified space retrieval under sufficiently large modality bias. This provides a solid theoretical foundation for "why routing works."
*   **Minimalist Architecture Philosophy**: Instead of modifying underlying encoders or introducing complex alignment mechanisms, the modality gap is elegantly bypassed through a "divide and conquer" routing strategy. This simplicity makes the framework naturally extendable to new modalities.
*   **Sophistication of Two-layer Routing**: Joint prediction of modality-granularity pairs is more powerful than step-by-step prediction because it supports cross-modal granularity combinations (e.g., HybridQA requiring both tables and passages simultaneously).
*   **Value of Training-free Routing**: Through carefully designed few-shot prompts, an LLM can serve as a router without any parameter updates, and its generalization surpasses that of fine-tuned smaller models.

## Limitations & Future Work

**Author-acknowledged Limitations**:
*   The accuracy of routing depends on high-quality training data, yet existing benchmarks lack explicit labels for "which modality/granularity should be retrieved for this query."
*   The number of granularity levels is limited by labeling availability. The paper only implements two levels for text/video; finer multi-level granularities require manual annotation.

**Self-identified Limitations**:
*   While routing latency is constant-time, it may be relatively significant for extremely small-scale queries.
*   Information complementarity during multimodal fusion might be restricted by the router.
*   On smaller language models (e.g., a 2B-parameter router), routing accuracy drops from 95% to approximately 90%.

## Related Work & Insights

*   **vs. UniRAG / GME / VLM2Vec-V2** (Unified Multimodal Encoding): These methods force alignment of all modalities into a single space; this paper rigorously proves the inherent modality bias of such approaches through experiments and theory.
*   **vs. MultiRAG** (Multi-corpus Fusion): MultiRAG simply retrieves from all corpora and mixes them, which can introduce noise. UniversalRAG uses explicit routing to refine the selection.
*   **vs. Adaptive RAG Series** (Query Complexity Adaptation): Works like Adaptive-RAG and ROWEN adjust retrieval strategies within a single modality. UniversalRAG crosses modality boundaries.
*   **Insights**: In multimodal system design, "decoupling + routing" often outperforms "forced unification + complex alignment"; theory-driven design provides confidence for heuristic decisions; training-free solutions are efficient paths for rapid adaptation.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of the modality gap solved fundamentally via a dual-layer routing + granularity design; strong original theoretical analysis.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 benchmarks covering 7 modalities, 6 OOD datasets for generalization, 3 LVLM backbones, and comparison between training-based and training-free routing.
*   Writing Quality: ⭐⭐⭐⭐ Clear logic, intuitive visualizations, and rigorous theoretical sections.
*   Value: ⭐⭐⭐⭐⭐ Addresses a core pain point in the RAG field (multimodal knowledge integration) with a highly generalizable and plug-and-play framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ICCV 2025\] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs](../../ICCV2025/multimodal_vlm/autocompose_automatic_generation_of_pose_transition_descriptions_for_composed_po.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)
- [\[ICML 2026\] SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval](../../ICML2026/multimodal_vlm/solar_self-supervised_joint_learning_for_symmetric_multimodal_retrieval.md)

</div>

<!-- RELATED:END -->
