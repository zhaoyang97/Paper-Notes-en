---
title: >-
  [Paper Note] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation
description: >-
  [ACL 2026][Recommender Systems][Conversational Recommendation] This paper proposes HARPO, a framework that redefines conversational recommendation as a structured decision-making problem optimizing for recommendation qua…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Conversational Recommendation"
  - "Agentic Reasoning"
  - "Preference Optimization"
  - "Tree Search"
  - "Recommendation Quality"
date: 2026-05-08
content_hash: da96e5b3925288ab
---

# HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation

**Conference**: ACL 2026  
**arXiv**: [2604.10048](https://arxiv.org/abs/2604.10048)  
**Code**: [https://anonymous.4open.science/r/HARPO-D881](https://anonymous.4open.science/r/HARPO-D881)  
**Area**: Recommender Systems  
**Keywords**: Conversational Recommendation, Agentic Reasoning, Preference Optimization, Tree Search, Recommendation Quality

## TL;DR
This paper proposes HARPO, a framework that redefines conversational recommendation as a structured decision-making problem optimizing for recommendation quality. Through hierarchical preference learning, tree search reasoning guided by value networks, virtual tool manipulation, and multi-agent refinement, HARPO significantly outperforms existing methods across ReDial, INSPIRED, and MUSE benchmarks.

## Background & Motivation

**Background**: Conversational Recommender Systems (CRS) aim to help users discover items matching their preferences through natural language interaction. Recently, LLM-based CRS methods have achieved strong performance on proxy metrics such as Recall@K and BLEU.

**Limitations of Prior Work**: High scores on proxy metrics do not necessarily equate to high-quality user-aligned recommendations. Existing methods primarily optimize intermediate objectives like retrieval accuracy, generation fluency, or tool-calling success rather than the recommendation quality itself. For instance, "something casual for a summer wedding" might be misinterpreted as everyday casual wear instead of occasion-appropriate wedding attire; such responses may score high on automated metrics but yield low user satisfaction.

**Key Challenge**: There is a fundamental misalignment between CRS training/evaluation objectives (proxy metrics) and actual recommendation quality. Proxy metrics are only weakly correlated with user-aligned recommendation quality.

**Goal**: To model conversational recommendation as a structured decision-making problem that explicitly optimizes recommendation quality, rather than treating quality as a byproduct of response generation.

**Key Insight**: From a decision-reasoning perspective, the system should explicitly reason over multiple candidate recommendation strategies, evaluate their expected quality, and select recommendations based on user-alignment criteria rather than proxy signals.

**Core Idea**: Recommendation quality is decomposed into interpretable dimensions (relevance, diversity, satisfaction, and engagement) via hierarchical preference learning. A learned value network then guides tree search reasoning to explore candidate recommendation paths.

## Method

### Overall Architecture
HARPO consists of four components: STAR (Structured Tree search Agentic Reasoning), CHARM (Contrastive Hierarchical Reward Alignment), BRIDGE (Cross-domain Transfer), and MAVEN (Multi-agent Refinement), all sharing a pre-trained language model backbone. The input is the dialogue context, and the output is a response containing recommendations. The optimization objective is the recommendation quality $\mathcal{Q}$ rather than proxy metrics.

### Key Designs

1.  **STAR: Structured Tree search Reasoning**:
    - **Function**: Explicitly explores multiple candidate recommendation strategies via tree search to select the highest-quality path.
    - **Mechanism**: Each reasoning node is represented as $s=(\mathbf{h}, \tau, \mathbf{v}, d)$, containing the dialogue context encoding, current thoughts, predicted virtual tool operations, and search depth. The value network decomposes quality into four dimensions (relevance, diversity, satisfaction, and engagement), each predicted by a dedicated head and weighted by learnable weights. At each node, $b$ candidate next steps are generated, and beam search is used to select the optimal path.
    - **Design Motivation**: Unlike direct generation, tree search allows the system to explore, compare, and refine recommendation decisions before generating the final response.

2.  **CHARM: Contrastive Hierarchical Reward Alignment**:
    - **Function**: Decomposes recommendation quality into interpretable dimensions and learns context-dependent dimensional weights.
    - **Mechanism**: Each quality dimension is modeled by a dedicated reward head $R_k(\mathbf{h}) = \tanh(\mathbf{W}_k^{(2)} \cdot \text{GELU}(\mathbf{W}_k^{(1)} \cdot \mathbf{h}))$, with outputs restricted to $[-1,1]$. Context-dependent weights $\mathbf{w} = \text{softmax}(\mathbf{W}_{\text{meta}} \cdot [\text{Enc}(\mathbf{h}); \mathbf{e}_d] + \mathbf{b})$ are learned in a meta-learning fashion. Training is conducted using a margin-based preference optimization loss.
    - **Design Motivation**: The importance of different quality dimensions varies across dialogue contexts and domains; adaptive weighting avoids information loss associated with a single scalar reward.

3.  **BRIDGE: Cross-domain Transfer via VTO Abstraction**:
    - **Function**: Realizes the transfer of recommendation reasoning across domains through Virtual Tool Operations (VTO) and adversarial domain adaptation.
    - **Mechanism**: Adversarial domain adaptation uses a gradient reversal layer to learn domain-invariant representations, while a learnable domain gate $\mathbf{z}' = \sigma(\mathbf{g}_d) \odot \mathbf{z} + (1-\sigma(\mathbf{g}_d)) \odot \mathbf{h}$ preserves useful domain-specific signals. VTO decouples high-level reasoning operations from specific tool implementations, mapping them dynamically at runtime.
    - **Design Motivation**: Existing tool-augmented methods are tightly coupled to domain-specific tool implementations, which limits transferability.

### Loss & Training
The total loss comprises a preference optimization loss $\mathcal{L}_{\text{pref}}$, domain adaptation loss $\mathcal{L}_{\text{domain}}$, task preservation loss $\mathcal{L}_{\text{task}}$, and agent consistency loss $\mathcal{L}_{\text{agree}}$.

## Key Experimental Results

### Main Results
Recommendation performance on the ReDial dataset:

| Method | R@1 | R@10 | R@50 | MRR@10 | User Sat. | Engage. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KBRD | 2.9 | 16.7 | 36.2 | 7.4 | 0.42 | 0.38 |
| UniCRS | 3.8 | 18.1 | 37.4 | 8.4 | 0.45 | 0.41 |
| GPT-4 | — | — | — | — | — | — |
| **Ours (HARPO)** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full HARPO | Best | Complete model |
| w/o STAR | Significant Drop | Removed tree search reasoning |
| w/o CHARM | Obvious Drop | Removed hierarchical preference optimization |
| w/o BRIDGE | Cross-domain Drop | Removed domain transfer module |
| w/o MAVEN | Slight Drop | Removed multi-agent refinement |

### Key Findings
- HARPO achieves an average improvement of 17-21% over the strongest baseline (GPT-4), with even larger gains in user-alignment metrics.
- The largest improvement occurs on the INSPIRED dataset (R@10 is 45.7% higher than GPT-4), as social dialogues require reasoning over implicit preferences.
- Human evaluation confirms that recommendation quality, explanation quality, and overall scores are significantly superior to GPT-4 (+0.55, +0.50, +0.55).
- The Pearson correlation coefficient between the CHARM reward model and independent human judgments reaches 0.64-0.73.

## Highlights & Insights
- The insight that the misalignment between proxy metrics and recommendation quality is a fundamental issue in the CRS field represents a paradigm shift.
- VTO abstraction decouples reasoning logic from specific tools, similar to interface design in software engineering, providing an elegant solution for transferable reasoning.
- Multi-dimensional quality decomposition combined with context-adaptive weighting avoids the information compression issues inherent in single reward signals.

## Limitations & Future Work
- The CHARM reward model itself may contain biases; while correlated with human judgment, it is not a perfect substitute.
- Tree search reasoning increases computational overhead at inference time, necessitating latency considerations for practical deployment.
- The scale of experimental datasets is limited (ReDial has 10k dialogues), lacking large-scale validation.
- Future work could explore extending quality dimensions to more fine-grained user preference modeling.

## Related Work & Insights
- **vs UniCRS**: While UniCRS unifies recommendation and generation, it still optimizes proxy metrics. HARPO directly optimizes recommendation quality.
- **vs RecMind**: RecMind uses self-motivated reasoning but lacks quality-guided search; HARPO’s value network provides quality-oriented guidance.
- **vs GPT-4**: Even powerful general-purpose models that perform well on proxy metrics fall behind the specifically optimized HARPO on user-alignment metrics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Redefining conversational recommendation as a quality-optimized decision problem is a novel approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes three datasets, human evaluation, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Deep problem analysis and clear framework logic.
- **Value**: ⭐⭐⭐⭐ Offers important methodological insights for the CRS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ACL 2026\] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)

</div>

<!-- RELATED:END -->
