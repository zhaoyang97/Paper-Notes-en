---
title: >-
  [Paper Note] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines
description: >-
  [ACL 2026][Information Retrieval & RAG][Generative Retrieval] This paper proposes AuthGR, the first framework to systematically integrate document authority into generative retrieval. It utilizes VLM-based multimodal aut…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Generative Retrieval"
  - "Authority"
  - "GRPO"
  - "Multimodal Scoring"
  - "Web Search"
date: 2026-05-08
content_hash: a1c6d1c839a4d1bc
---

# From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines

**Conference**: ACL 2026  
**arXiv**: [2604.13468](https://arxiv.org/abs/2604.13468)  
**Code**: None  
**Area**: Information Retrieval / Search Engines  
**Keywords**: Generative Retrieval, Authority, GRPO, Multimodal Scoring, Web Search

## TL;DR
This paper proposes AuthGR, the first framework to systematically integrate document authority into generative retrieval. It utilizes VLM-based multimodal authority scoring, a three-stage progressive training pipeline (CPT $\rightarrow$ SFT $\rightarrow$ GRPO), and a hybrid ensemble deployment pipeline. Significant user engagement improvements were validated through large-scale A/B testing on the Naver commercial search engine.

## Background & Motivation

**Background**: Generative Information Retrieval (GenIR) reformulates retrieval as a text generation task, directly generating document identifiers (DocID). Recently, it has been widely applied in industrial scenarios such as e-commerce search, food delivery, and financial services. However, existing methods primarily optimize for semantic relevance.

**Limitations of Prior Work**: In high-risk domains such as health and finance, relying solely on semantic relevance may rank unverified personal blogs (e.g., health influencers) at the same level as official medical associations. Integrating authority into GenIR faces three challenges: (1) Defining authority—textual clues alone cannot distinguish trustworthy sources from well-disguised promotional websites; (2) Learning authority—injecting authority concepts without compromising semantic relevance is non-trivial; (3) Deployment—directly replacing existing rankers is impractical.

**Key Challenge**: Highly relevant documents are not necessarily trustworthy documents. In high-risk domains, recommending unreliable information to users can lead to serious consequences. However, existing GenIR systems lack any mechanism to distinguish between authoritative and non-authoritative sources.

**Goal**: Design the first authority-aware GenIR framework that prioritizes displaying trustworthy documents while maintaining relevance.

**Key Insight**: (1) Use VLMs to simulate human multimodal judgments of webpage authority (textual content + visual design + advertisement distribution); (2) Use GRPO preference optimization to enable the model to prioritize high-authority documents among candidates; (3) Deploy collaboratively with existing rankers via a hybrid ensemble pipeline.

**Core Idea**: Use authority scores as reward signals for GRPO, allowing the generative retrieval model to learn a preference for authoritative sources while maintaining semantic relevance.

## Method

### Overall Architecture
AuthGR consists of three components: (1) Multimodal Authority Scoring—VLMs quantify document authority from textual and visual signals; (2) Three-stage Training Pipeline—Continual Pre-training (CPT) $\rightarrow$ Supervised Fine-tuning (SFT) $\rightarrow$ GRPO Authority-aware Optimization; (3) Hybrid Ensemble Deployment Pipeline—working in synergy with existing rankers.

### Key Designs

1.  **Multimodal Authority Scoring**:
    - **Function**: Automated large-scale quantification of document trustworthiness to replace manual evaluation.
    - **Mechanism**: The VLM processes both textual signals (title, body, URL metadata) and visual signals (page screenshots). It evaluates three core dimensions—expertise, officiality, and public interest—based on a comprehensive scoring rubric, while checking for commercial intent and harmfulness. It outputs an authority score $\text{Authority}(d) = f_{\text{VLM}}(T(d), V(d)) \in [0, 100]$ and natural language justifications.
    - **Design Motivation**: Promotional content often mimics authoritative language at the text level, making it difficult to identify by text alone. Visual cues (e.g., ad density, page layout quality) are critical signals for distinguishing true authority from well-disguised fakes. VLMs serve as a scalable proxy for human evaluators.

2.  **Three-stage Training Pipeline**:
    - **Function**: Progressively inject authority awareness into the generative retrieval model.
    - **Mechanism**: (1) Continual Pre-training (CPT)—Pre-trained on 9.85 million search logs to let the model learn associations between Query-URL-Title-Body. Host-level URLs are used as DocIDs (e.g., "plus.gov.kr") to expose source identity. (2) Supervised Fine-tuning (SFT)—Fine-tuned on 3.95 million high-quality query-document pairs to learn DocID generation from queries. (3) GRPO Authority Optimization—Samples $G$ candidate DocIDs for each query, using the authority score as a reward to optimize the policy via group relative advantage $A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$.
    - **Design Motivation**: Direct SFT treats all valid documents equally and fails to capture relative differences in authority. Weighted cross-entropy is pointwise and lacks an exploration mechanism. GRPO enables the model to prioritize high-authority sources through intra-group contrast.

3.  **Hybrid Ensemble Deployment Pipeline**:
    - **Function**: Seamlessly integrate AuthGR with existing rankers to maintain recall while injecting authority signals.
    - **Mechanism**: Existing rankers provide a relevance score $S_{\text{rel}}(d|q)$, and AuthGR generates a set of authoritative DocIDs $\mathcal{D}_{\text{auth}}(q)$, deriving an authority score $S_{\text{auth}} = \frac{N - \text{rank}(d) + 1}{N}$ through linear decay. The final score is $S_{\text{final}} = S_{\text{rel}} + \lambda \cdot S_{\text{auth}} \cdot \mathbb{I}[d \in \mathcal{D}_{\text{auth}}]$.
    - **Design Motivation**: Directly replacing existing rankers is too risky. The ensemble solution maintains the recall of existing systems while using AuthGR’s knowledge to boost the ranking of trustworthy documents.

### Loss & Training
Three stages: CPT uses standard language modeling loss; SFT uses negative log-likelihood; GRPO uses a PPO-style clipped objective based on authority scores. Trained on Korean data from the Naver commercial search engine.

## Key Experimental Results

### Main Results

| Model | Scale | P@3 | R@5 | R@10 |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3 (ICL) | 32B | 0.0821 | 0.1176 | 0.1570 |
| K-EXAONE (ICL) | 236B | 0.1366 | 0.1918 | 0.2656 |
| **Ours** (AuthGR) | **3B** | Matches 14B baseline | — | — |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Full AuthGR (CPT+SFT+GRPO) | Optimal | Complete three-stage pipeline |
| w/o CPT | Decrease | Lacks domain knowledge prior |
| w/o GRPO (CPT+SFT only) | Low Authority | Fails to distinguish authority levels |
| Online A/B Test | User Engagement ↑ | Large-scale commercial validation |
| Human Evaluation | Quality Score ↑ | Expert verification |

### Key Findings
- AuthGR’s 3B model matches the 14B baseline in offline evaluation (4.7x increase in parameter efficiency).
- Large-scale online A/B testing confirms a significant increase in real user engagement.
- Human evaluation verifies high consistency between authority scores and human judgment.
- The GRPO stage is key to injecting authority awareness; SFT alone cannot learn the relativity of authority.
- Multimodal (text + visual) scoring is more accurate than text-only scoring because promotional content is adept at mimicking authoritative text.

## Highlights & Insights
- **Paradigm Extension from Relevance to Authority**: This work systematically introduces authority signals into GenIR for the first time, expanding from "finding what is relevant" to "finding what is trustworthy." This direction is crucial in an era of information overload and misinformation.
- **VLM as a Scalable Proxy for Authority Assessment**: Leveraging the multimodal understanding capabilities of VLMs to automate webpage authority assessment is a modern upgrade to traditional PageRank logic.
- **Industrial-grade Validation**: Large-scale A/B testing and human evaluation on the Naver commercial search engine provide strong practical validation, distinguishing it from most academic works that only include offline experiments.

## Limitations & Future Work
- Validated only on Korean web search; authority standards may differ across languages and cultures.
- The computational cost of VLM scoring is high, posing efficiency challenges for large-scale real-time scoring.
- Authority assessment may contain biases—certain legitimate but niche information sources might be undervalued.
- Host-level DocID granularity may erroneously treat content of different quality under the same domain identically.

## Related Work & Insights
- **vs PageRank/TrustRank**: Traditional methods rely on link structures; AuthGR evaluates authority directly from content and visuals.
- **vs Standard GenIR**: Standard GenIR only optimizes for relevance; AuthGR adds the authority dimension.
- **vs TREC Health Misinformation Track**: While TREC focuses on trustworthiness in the health domain, AuthGR provides a generalized framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to systematically integrate authority into GenIR, highly significant for industry.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Complete validation system including offline, online A/B, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and systematic method, though some technical details are brief.
- Value: ⭐⭐⭐⭐⭐ Directly impacts the trustworthiness of commercial search engines, with profound social significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)
- [\[ACL 2026\] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)
- [\[ACL 2026\] GLIER: Generative Legal Inference and Evidence Ranking for Legal Case Retrieval](glier_generative_legal_inference_and_evidence_ranking_for_legal_case_retrieval.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)

</div>

<!-- RELATED:END -->
