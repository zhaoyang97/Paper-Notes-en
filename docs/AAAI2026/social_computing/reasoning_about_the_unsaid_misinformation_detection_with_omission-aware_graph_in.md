---
title: >-
  [Paper Note] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference
description: >-
  [AAAI 2026][Social Computing][Misinformation Detection] This paper proposes OmiGraph, the first omission-aware misinformation detection framework. By constructing omission-aware graphs, leveraging LLMs to reason about omission intent, and employing omission-guided message passing and aggregation mechanisms, OmiGraph extracts deception patterns from "what is unsaid," achieving average gains of +5.4% F1 and +5.3% ACC on bilingual datasets.
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Misinformation Detection"
  - "Omission-Aware"
  - "Graph Neural Networks"
  - "Large Language Models"
  - "Information Manipulation"
date: 2026-05-08
content_hash: f4b1d4331bfed777
---

# Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference

**Conference**: AAAI 2026
**arXiv**: [2512.01728](https://arxiv.org/abs/2512.01728)  
**Code**: [GitHub](https://github.com/ICTMCG/OmiGraph)  
**Area**: Social Computing
**Keywords**: Misinformation Detection, Omission-Aware, Graph Neural Networks, Large Language Models, Information Manipulation

## TL;DR

This paper proposes OmiGraph, the first omission-aware misinformation detection framework. By constructing omission-aware graphs, leveraging LLMs to reason about omission intent, and employing omission-guided message passing and aggregation mechanisms, OmiGraph extracts deception patterns from "what is unsaid," achieving average gains of +5.4% F1 and +5.3% ACC on bilingual datasets.

## Background & Motivation

Deceptive strategies in misinformation fall into two main categories: (1) **deception by commission** — explicitly fabricating false content; and (2) **deception by omission** — implicitly withholding critical information, leading readers to draw erroneous conclusions from an incomplete picture.

Existing misinformation detection methods focus almost exclusively on the former, extracting deceptive signals from "what is said":
- **Style/emotion-based methods** (e.g., DualEmo): analyze linguistic style and affective signals;
- **Commonsense conflict-based methods** (e.g., MD-PCC): leverage external knowledge to detect contradictions between content and facts;
- **Evidence verification-based methods** (e.g., RAV, RAFTS): verify the veracity of claims against external evidence.

However, **deception by omission is pervasive and far more covert in practice**. Psychological research demonstrates that people are more susceptible to manipulation when information is selectively presented. For instance, a news article about a protest may deliberately omit background causes and causal relationships, amplifying the surface impression of police-civilian conflict — a "strategic silence" that existing methods fail to detect.

Detecting omission-based deception poses three core challenges:

**Implicit signal recovery**: Omitted information is absent from the target article and cannot be directly observed;

**Dynamic omission relations**: The relationship between stated and omitted content is highly variable — it may reflect ordinary editorial choices or deliberate causal concealment;

**Omission pattern modeling**: Effective integration of omitted content and its relational structure is required to build holistic deception awareness.

## Method

### Overall Architecture

OmiGraph comprises three core components (as illustrated in Figure 2):
1. **Omission-Aware Graph Construction**: recovers omitted information using contextual surroundings and constructs a graph structure;
2. **Omission-Oriented Relation Modeling**: reasons about omission relations both within and across graph sources;
3. **Omission-Guided Message Passing and Aggregation**: extracts omission-based deception patterns for detection.

OmiGraph is designed as a **plug-and-play enhancement module** compatible with any existing misinformation detector.

### Key Designs

1. **Omission-Aware Graph Construction**

   Core insight: Different news reports covering the same event naturally provide complementary perspectives, which can serve as resources for recovering omitted information.

   **Contextual environment construction**: Given a target news article $n_{\text{tgt}}$, semantically similar news articles are retrieved from a candidate pool $\mathcal{P}$ published within $T$ days, using BERT-based cosine similarity. The top-$K$ articles form the contextual environment $\mathcal{C}_{\text{ctx}}$:
   $\mathcal{C}_{\text{ctx}} = \{n_u \mid n_u \in \text{TopK}(\cos(\mathbf{h}_{\text{tgt}}, \mathbf{h}_u), n_u \in \mathcal{P})\}$

   **Fine-grained node initialization**: Both target and contextual news articles are decomposed into sentence-level atomic segments as graph nodes $\mathcal{V}$. This fine-grained decomposition enables precise identification of omission boundaries, as opposed to coarse-grained full-article representations. Graph edges include intra-source edges $\mathcal{E}_{\text{intra}}$ (connecting segments within the same article) and inter-source edges $\mathcal{E}_{\text{inter}}$ (connecting the target article to contextual articles).

2. **Omission-Oriented Relation Modeling**

   **Intra-source relations**: Model semantic dependency between segments within the same article, revealing how internal segments interact to maintain narrative coherence or facilitate deception. These are realized via learnable edge embeddings:
   $\mathbf{e}_{\text{intra}}^{ij} = \text{MLP}(\mathbf{h}_i \| \mathbf{h}_j \| \text{diff}(\mathbf{h}_i - \mathbf{h}_j))$

   **Inter-source relations**: The key innovation — leveraging **LLMs to dynamically reason about omission intent**. Rather than relying on predefined relation types, OmiGraph exploits the contextual understanding of LLMs to infer "why specific information was omitted":
   $\mathbf{e}_{\text{inter}}^{ij} = \text{PLM}(\mathcal{M}(s_{\text{tgt}}^i, s_{\text{ctx}}^j))$

   The LLM returns free-text descriptions of omission intent (e.g., "to downplay the political motivations behind the action"), which are then encoded into edge attributes via a pretrained language model. This design enables the framework to **dynamically capture diverse omission patterns without predefined relation taxonomies**.

3. **Omission-Guided Message Passing and Aggregation**

   **Local attention-based message passing**: Omission relations encoded in edge embeddings guide information propagation. Edge attributes are augmented with type-specific learnable embeddings and incorporated into attention weight computation:
   $\alpha_{ij} = \text{softmax}((\mathbf{h}_i^{(l-1)} + \hat{\mathbf{e}}_t^{ij}) \cdot (\mathbf{h}_j^{(l-1)} + \hat{\mathbf{e}}_t^{ij}))$

   **Global aggregation**: A super-root node $\mathbf{h}_{\text{root}}$ is introduced as a central aggregator for global information, mitigating over-smoothing and over-squashing issues that arise from relying solely on multi-layer local message passing:
   $\mathbf{h}_{\text{root}}^{(l)} = \mathbf{h}_{\text{root}}^{(l-1)} + \sum_i \text{softmax}(\mathbf{W}\mathbf{h}_i^{(l-1)} + b) \cdot \mathbf{h}_i^{(l-1)}$

   Global information is then back-propagated to individual nodes via residual fusion, ensuring that segment-level omission patterns are contextualized within the overall narrative structure.

### Loss & Training

The final omission-aware representation $\mathbf{h}_{\text{omi}}$ is obtained by mean-pooling the target news graph node features, then fused with the conventional detector's features $\mathbf{h}_{\text{com}}$ for prediction:
$$\hat{y} = \text{fuse}(\mathbf{h}_{\text{omi}} \| \mathbf{h}_{\text{com}})$$

Optimized with standard binary cross-entropy loss:
$$\mathcal{L}_{\text{cls}} = -y \log(\hat{y}) - (1-y)\log(1-\hat{y})$$

Training details: AdamW optimizer, batch size 64, learning rate $2 \times 10^{-5}$, feature dimension 256, MLP hidden size $[128, 128]$. bert-base-uncased is used for English; bert-base-chinese for Chinese. GPT-4o-mini is used as the LLM for omission intent reasoning.

## Key Experimental Results

### Main Results

OmiGraph is applied as an enhancement module to multiple baseline detectors ("+ Ours" denotes the addition of OmiGraph):

| Baseline | EN macF1 | +OmiGraph | Gain | ZH macF1 | +OmiGraph | Gain |
|----------|-----------|-----------|------|-----------|-----------|------|
| BERT | 0.7111 | **0.7530** | +4.19% | 0.7851 | **0.8407** | +5.56% |
| DualEmo | 0.7194 | **0.7557** | +3.63% | 0.7958 | **0.8417** | +4.59% |
| MSynFD | 0.7317 | **0.7608** | +2.91% | 0.8054 | **0.8496** | +4.42% |
| LLM | 0.5556 | **0.7259** | +17.03% | 0.6992 | **0.8336** | +13.44% |
| PCoT | 0.6508 | **0.7062** | +5.54% | 0.8020 | **0.8383** | +3.63% |
| NEP | 0.7274 | **0.7596** | +3.22% | 0.8288 | **0.8585** | +2.97% |
| RAV | 0.7189 | **0.7433** | +2.44% | 0.7930 | **0.8354** | +4.24% |
| RAFTS | 0.6016 | **0.6771** | +7.55% | 0.7427 | **0.7870** | +4.43% |

All improvements are statistically significant at $p < 0.005$.

### Ablation Study

| Configuration | EN macF1 | ZH macF1 | Description |
|---------------|-----------|-----------|-------------|
| OmiGraph (Full) | **0.7530** | **0.8407** | Complete model |
| w/o Seg (no fine-grained segmentation) | ~0.735 | ~0.825 | Coarse-grained representation impedes omission reasoning |
| w/o Textual (no LLM omission intent) | ~0.730 | ~0.820 | Structural connectivity cannot substitute semantic reasoning |
| w/o Intra (no intra-source relations) | ~0.740 | ~0.830 | Internal dependencies provide important contextual cues |
| w/o GlobalAgg (no global aggregation) | ~0.742 | ~0.832 | Global narrative understanding is critical for detecting systematic omissions |

### Key Findings

1. **LLM baselines show the largest gains** (+17.03% EN / +13.44% ZH): This indicates that even powerful language models struggle to effectively detect misinformation without explicit omission modeling.
2. **Complementary gains over methods already using external information**: Even methods such as NEP that already incorporate external news benefit from OmiGraph, as it contributes along the dimension of "information completeness gaps" rather than "factual contradictions."
3. **Distributional differences in omission types**: Misinformation exhibits higher proportions of comparative omissions and stakeholder omissions, while true news shows higher rates of complexity omissions — revealing distinct editorial motivations.
4. **Feasibility of LLM simulation**: In settings without access to external news corpora, using LLMs to simulate the contextual environment still yields competitive performance, at a lower token cost than methods such as PCoT.
5. **Hyperparameter robustness**: Performance remains stable across varying numbers of contextual nodes $k$; optimal GNN depth is $l=2$ for English and $l=3$ for Chinese.

## Highlights & Insights

1. **Pioneering "omission" perspective**: OmiGraph is the first to extend misinformation detection from "what is said" to "what is unsaid" — a long-neglected yet critical dimension of deception, with solid grounding in psychological theory.
2. **Novel use of LLMs for omission intent reasoning**: Rather than predefining relation types, the framework prompts LLMs to freely generate textual descriptions of omission intent, which are then encoded as edge attributes — an elegant combination of LLM reasoning capacity and GNN structural learning.
3. **Plug-and-play framework design**: OmiGraph augments any existing detector without modifying its original architecture, offering strong practical utility and adaptability.
4. **Systematic analysis of eight omission types**: The omission typology derived from large-scale data analysis provides a valuable theoretical foundation for future research.

## Limitations & Future Work

1. **Dependence on external news corpora**: The standard setting requires large-scale contemporaneous news corpora (1M+ articles in English, 580K+ in Chinese), incurring high acquisition costs. Although an LLM simulation alternative is proposed, it comes at some performance cost.
2. **LLM inference cost**: Inter-source relation reasoning requires LLM calls for each segment pair, and token consumption may become a bottleneck as the number of segments grows.
3. **Timeliness constraints**: The contextual environment is constructed from news published within $T$ days prior, which may yield insufficient reference articles for breaking news events.
4. **Text-only modality**: The current framework analyzes omissions only in textual content, without addressing omissions in images or videos in multimodal news.
5. **Absence criterion for omissions**: The determination of whether an omission exists currently relies on LLM judgment, lacking a more rigorous formal definition.

## Related Work & Insights

- This work fills a gap in the misinformation detection literature by addressing omission-based deception, offering a complementary perspective to existing commission-based detection methods.
- The approach of **modeling news relationships via graph structures** is generalizable to related tasks such as rumor tracing and news credibility assessment.
- The design of **LLMs as relation reasoning engines** offers a transferable paradigm for other NLP tasks requiring implicit inference.
- The eight-category omission typology lays the groundwork for future automated "news information completeness assessment" tools.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First omission-aware misinformation detection framework; unique and consequential perspective
- **Technical Depth**: ⭐⭐⭐⭐ — LLM+GNN integration is well-motivated; message passing mechanism is complete
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Bilingual datasets, 8 baselines, comprehensive ablations, clear case studies
- **Value**: ⭐⭐⭐⭐ — Plug-and-play design is practical, though reliance on external corpora and LLMs introduces cost
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear, motivation builds progressively, case studies are compelling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)

</div>

<!-- RELATED:END -->
