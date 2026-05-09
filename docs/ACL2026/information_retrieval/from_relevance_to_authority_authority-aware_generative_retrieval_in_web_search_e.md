---
title: >-
  [Paper Note] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines
description: >-
  [ACL 2026][Generative Retrieval] This paper proposes AuthGR, the first framework to systematically integrate document authority into generative retrieval. It combines VLM-based multimodal authority scoring, a three-stage progressive training pipeline (CPT→SFT→GRPO), and a hybrid ensemble deployment pipeline. The approach is validated through large-scale A/B testing on Naver's commercial search engine, demonstrating significant improvements in user engagement.
tags:
  - ACL 2026
  - Generative Retrieval
  - Authority
  - GRPO
  - Multimodal Scoring
  - Web Search
date: 2026-05-08
content_hash: 2cdcbeeadb962041
---

# From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines

**Conference**: ACL 2026
**arXiv**: [2604.13468](https://arxiv.org/abs/2604.13468)
**Code**: None
**Area**: Information Retrieval / Search Engines
**Keywords**: Generative Retrieval, Authority, GRPO, Multimodal Scoring, Web Search

## TL;DR
This paper proposes AuthGR, the first framework to systematically integrate document authority into generative retrieval. It combines VLM-based multimodal authority scoring, a three-stage progressive training pipeline (CPT→SFT→GRPO), and a hybrid ensemble deployment pipeline. The approach is validated through large-scale A/B testing on Naver's commercial search engine, demonstrating significant improvements in user engagement.

## Background & Motivation

**State of the Field**: Generative Information Retrieval (GenIR) reformulates retrieval as a text generation task that directly generates document identifiers (DocIDs). It has seen widespread adoption in industrial settings such as e-commerce search, food delivery, and financial services. However, existing methods primarily optimize for semantic relevance.

**Limitations of Prior Work**: In high-stakes domains such as health and finance, optimizing solely for semantic relevance may rank unverified personal blogs (e.g., health influencers) on par with official medical associations. Integrating authority into GenIR poses three challenges: (1) *defining authority*—textual cues alone cannot distinguish trustworthy sources from well-disguised promotional sites; (2) *learning authority*—injecting authority awareness without degrading semantic relevance is non-trivial; (3) *deployment*—directly replacing existing rankers is impractical.

**Root Cause**: A highly relevant document is not necessarily a trustworthy one. In high-stakes domains, surfacing unreliable information can have serious consequences, yet existing GenIR systems have no mechanism to distinguish authoritative sources from non-authoritative ones.

**Paper Goals**: Design the first authority-aware GenIR framework that prioritizes trustworthy documents while preserving relevance.

**Starting Point**: (1) Use a VLM to simulate human multimodal judgment of webpage authority (textual content + visual design + advertisement distribution); (2) Apply GRPO preference optimization to teach the model to favor high-authority documents among candidates; (3) Deploy collaboratively with existing rankers via a hybrid ensemble pipeline.

**Core Idea**: Authority scores serve as reward signals for GRPO, enabling the generative retrieval model to learn a preference for authoritative sources while maintaining semantic relevance.

## Method

### Overall Architecture
AuthGR consists of three components: (1) *Multimodal Authority Scoring*—a VLM quantifies document authority from textual and visual signals; (2) *Three-Stage Training Pipeline*—domain-adaptive continual pre-training → supervised fine-tuning → GRPO authority-aware optimization; (3) *Hybrid Ensemble Deployment Pipeline*—collaborative integration with existing rankers.

### Key Designs

1. **Multimodal Authority Scoring**:

    - **Function**: Automatically quantifies document trustworthiness at scale, replacing manual evaluation.
    - **Mechanism**: A VLM jointly processes textual signals (title, body, URL metadata) and visual signals (page screenshots), evaluating three core dimensions—professionalism, officiality, and public interest—while checking for commercial intent and harmfulness. The output is an authority score $\text{Authority}(d) = f_{\text{VLM}}(T(d), V(d)) \in [0, 100]$ accompanied by natural language rationales.
    - **Design Motivation**: Promotional content frequently mimics authoritative language at the textual level, making text-only identification unreliable. Visual cues (e.g., advertisement density, page layout quality) are critical signals for distinguishing genuine authority from well-crafted imitation. The VLM serves as a scalable proxy for human evaluators.

2. **Three-Stage Training Pipeline**:

    - **Function**: Progressively injects authority awareness into the generative retrieval model.
    - **Mechanism**: (1) *Domain Continual Pre-Training (CPT)*—pre-trains on 9.85 million search logs to learn query–URL–title–body associations, using host-level URLs as DocIDs (e.g., "plus.gov.kr") to expose source identity. (2) *Supervised Fine-Tuning (SFT)*—fine-tunes on 3.95 million high-quality query–document pairs to learn DocID generation from queries. (3) *GRPO Authority Optimization*—samples $G$ candidate DocIDs per query, uses authority scores as rewards, and optimizes the policy via group-relative advantage $A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$.
    - **Design Motivation**: Standard SFT treats all valid documents equally and cannot capture relative authority differences. Weighted cross-entropy is pointwise and lacks an exploration mechanism. GRPO leverages intra-group contrastive learning to teach the model to prefer high-authority sources among candidates.

3. **Hybrid Ensemble Deployment Pipeline**:

    - **Function**: Seamlessly integrates AuthGR with existing rankers, injecting authority signals while preserving recall.
    - **Mechanism**: The existing retriever provides a relevance score $S_{\text{rel}}(d|q)$; AuthGR generates an authoritative DocID set $\mathcal{D}_{\text{auth}}(q)$; a linearly decayed authority score is computed as $S_{\text{auth}} = \frac{N - \text{rank}(d) + 1}{N}$. The final score is $S_{\text{final}} = S_{\text{rel}} + \lambda \cdot S_{\text{auth}} \cdot \mathbb{I}[d \in \mathcal{D}_{\text{auth}}]$.
    - **Design Motivation**: Directly replacing the existing ranker carries high operational risk. The ensemble approach preserves the recall of the existing system while leveraging AuthGR's authority knowledge to promote trustworthy documents.

### Loss & Training
The three stages employ: standard language modeling loss (CPT); negative log-likelihood (SFT); and a PPO-style clipped objective with authority-score-based rewards (GRPO). All models are trained on Korean-language data from Naver's commercial search engine.

## Key Experimental Results

### Main Results

| Model | Scale | P@3 | R@5 | R@10 |
|-------|-------|-----|-----|------|
| Qwen3 (ICL) | 32B | 0.0821 | 0.1176 | 0.1570 |
| K-EXAONE (ICL) | 236B | 0.1366 | 0.1918 | 0.2656 |
| AuthGR | **3B** | matches 14B baseline | — | — |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| Full AuthGR (CPT+SFT+GRPO) | Best | All three stages |
| w/o CPT | Degraded | Lacks domain knowledge prior |
| w/o GRPO (CPT+SFT only) | Lower authority | Cannot distinguish authority levels |
| Online A/B Test | User engagement ↑ | Large-scale commercial validation |
| Human Evaluation | Quality score ↑ | Expert-validated |

### Key Findings
- AuthGR's 3B model matches the 14B baseline in offline evaluation, yielding a 4.7× parameter efficiency gain.
- Large-scale online A/B testing confirms significant improvements in real-user engagement.
- Human evaluation validates strong alignment between authority scores and human judgments.
- The GRPO stage is critical for injecting authority awareness—SFT alone cannot capture the relational nature of authority.
- Multimodal (text + visual) scoring outperforms text-only scoring because promotional content is adept at mimicking authoritative language.

## Highlights & Insights
- **Paradigm extension from relevance to authority**: AuthGR is the first work to systematically introduce authority signals into GenIR, broadening the objective from "find the relevant" to "find the trustworthy." This direction is of critical importance in an era of information overload and misinformation.
- **VLM as a scalable proxy for authority assessment**: Leveraging the multimodal understanding capabilities of VLMs to automate webpage authority evaluation represents a modern upgrade over traditional PageRank-style approaches.
- **Industrial-grade validation**: Large-scale A/B testing and human evaluation on Naver's commercial search engine provide exceptionally strong practical validation, distinguishing this work from most academic studies that rely solely on offline experiments.

## Limitations & Future Work
- Validation is limited to Korean-language web search; cross-lingual and cross-cultural authority standards may differ.
- The computational cost of VLM scoring is substantial, posing efficiency challenges for large-scale real-time evaluation.
- Authority assessment may introduce bias—legitimate but niche sources could be systematically undervalued.
- Host-level DocID granularity may incorrectly conflate content of varying quality hosted under the same domain.

## Related Work & Insights
- **vs. PageRank/TrustRank**: Traditional methods rely on link structures, whereas AuthGR assesses authority directly from content and visual signals.
- **vs. Standard GenIR**: Standard GenIR optimizes only for relevance; AuthGR adds an authority dimension.
- **vs. TREC Health Misinformation Track**: The latter focuses on credibility in the health domain, while AuthGR provides a general-purpose framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to systematically integrate authority into GenIR, with significant industrial implications.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation combining offline evaluation, online A/B testing, and human assessment.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the method is well-organized, though some technical details are underspecified.
- Value: ⭐⭐⭐⭐⭐ Directly improves the trustworthiness of commercial search engines, with substantial societal impact.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)

<!-- RELATED:END -->
