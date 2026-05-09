---
title: >-
  [Paper Note] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation
description: >-
  [ACL 2026][Recommender Systems][Conversational Recommendation] This paper proposes HARPO, a framework that reformulates conversational recommendation as a structured decision-making problem explicitly optimizing for recommendation quality. HARPO integrates four components—hierarchical preference learning, value-network-guided tree search reasoning, virtual tool operation abstraction, and multi-agent refinement—achieving significant improvements over existing methods on three benchmarks: ReDial, INSPIRED, and MUSE.
tags:
  - ACL 2026
  - Recommender Systems
  - Conversational Recommendation
  - Agentic Reasoning
  - Preference Optimization
  - Tree Search
  - Recommendation Quality
date: 2026-05-08
content_hash: 8936d6b8be3ea303
---

# HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation

**Conference**: ACL 2026
**arXiv**: [2604.10048](https://arxiv.org/abs/2604.10048)
**Code**: [https://anonymous.4open.science/r/HARPO-D881](https://anonymous.4open.science/r/HARPO-D881)
**Area**: Recommender Systems
**Keywords**: Conversational Recommendation, Agentic Reasoning, Preference Optimization, Tree Search, Recommendation Quality

## TL;DR
This paper proposes HARPO, a framework that reformulates conversational recommendation as a structured decision-making problem explicitly optimizing for recommendation quality. HARPO integrates four components—hierarchical preference learning, value-network-guided tree search reasoning, virtual tool operation abstraction, and multi-agent refinement—achieving significant improvements over existing methods on three benchmarks: ReDial, INSPIRED, and MUSE.

## Background & Motivation

**State of the Field**: Conversational Recommender Systems (CRS) aim to help users discover items matching their preferences through natural language interaction. Recent LLM-based CRS approaches have achieved strong performance on proxy metrics such as Recall@K and BLEU.

**Limitations of Prior Work**: High proxy metric scores do not necessarily reflect high-quality, user-aligned recommendations. Existing methods primarily optimize intermediate objectives—retrieval accuracy, generation fluency, or tool invocation—rather than recommendation quality itself. For example, the query "something casual for a summer wedding" may be misinterpreted as everyday casual wear rather than occasion-appropriate attire; such responses score well on automatic metrics yet yield low user satisfaction.

**Root Cause**: There is a fundamental misalignment between CRS training and evaluation objectives (proxy metrics) and actual recommendation quality. Proxy metrics correlate only weakly with user-aligned recommendation quality.

**Paper Goals**: To model conversational recommendation as a structured decision-making problem that explicitly optimizes recommendation quality, rather than treating quality as a byproduct of response generation.

**Starting Point**: From a decision-making and reasoning perspective, the authors argue that a system should explicitly reason over multiple candidate recommendation strategies, evaluate their expected quality, and select recommendations based on user-alignment criteria rather than proxy signals.

**Core Idea**: Decompose recommendation quality into interpretable dimensions (relevance, diversity, satisfaction, and engagement) via hierarchical preference learning, and employ a learned value network to guide tree search reasoning over candidate recommendation paths.

## Method

### Overall Architecture
HARPO comprises four components: STAR (Structured Tree-search Agentic Reasoning), CHARM (Contrastive Hierarchical Agentic Reward Modeling), BRIDGE (cross-domain transfer), and MAVEN (Multi-Agent refinement), all sharing a pretrained language model backbone. The input is the dialogue context; the output is a recommendation-bearing response; and the optimization target is recommendation quality $\mathcal{Q}$ rather than proxy metrics.

### Key Designs

1. **STAR: Structured Tree-Search Agentic Reasoning**

    - **Function**: Explicitly explores multiple candidate recommendation strategies via tree search and selects the highest-quality path.
    - **Mechanism**: Each reasoning node is represented as $s=(\mathbf{h}, \tau, \mathbf{v}, d)$, encoding dialogue context, current thought, predicted virtual tool operations, and search depth. A value network decomposes quality into four dimensions (relevance, diversity, satisfaction, engagement), each predicted by a dedicated head and combined via learnable weights. At each node, $b$ candidate next steps are generated, and beam search selects the optimal path.
    - **Design Motivation**: Unlike direct recommendation generation, tree search allows the system to explore, compare, and refine recommendation decisions before producing the final response.

2. **CHARM: Contrastive Hierarchical Agentic Reward Modeling**

    - **Function**: Decomposes recommendation quality into interpretable dimensions and learns context-dependent dimension weights.
    - **Mechanism**: Each quality dimension is modeled by a dedicated reward head $R_k(\mathbf{h}) = \tanh(\mathbf{W}_k^{(2)} \cdot \text{GELU}(\mathbf{W}_k^{(1)} \cdot \mathbf{h}))$, with outputs constrained to $[-1,1]$. Context-dependent weights are learned via a meta-learning approach: $\mathbf{w} = \text{softmax}(\mathbf{W}_{\text{meta}} \cdot [\text{Enc}(\mathbf{h}); \mathbf{e}_d] + \mathbf{b})$. Training uses a margin-based preference optimization loss.
    - **Design Motivation**: The relative importance of quality dimensions varies across dialogue contexts and domains; adaptive weighting avoids the information loss inherent in a single scalar reward.

3. **BRIDGE: Cross-Domain Transfer and VTO Abstraction**

    - **Function**: Enables cross-domain transfer of recommendation reasoning via Virtual Tool Operations (VTO) and adversarial domain adaptation.
    - **Mechanism**: A gradient reversal layer is used for adversarial domain adaptation to learn domain-invariant representations, while a learnable domain gate $\mathbf{z}' = \sigma(\mathbf{g}_d) \odot \mathbf{z} + (1-\sigma(\mathbf{g}_d)) \odot \mathbf{h}$ preserves useful domain-specific signals. VTO decouples high-level reasoning operations from concrete tool implementations, enabling dynamic mapping at runtime.
    - **Design Motivation**: Existing tool-augmented methods are tightly coupled to domain-specific tool implementations, limiting transferability.

### Loss & Training
The total loss comprises a preference optimization loss $\mathcal{L}_{\text{pref}}$, a domain adaptation loss $\mathcal{L}_{\text{domain}}$, a task preservation loss $\mathcal{L}_{\text{task}}$, and an agent consistency loss $\mathcal{L}_{\text{agree}}$.

## Key Experimental Results

### Main Results
Recommendation performance on the ReDial dataset:

| Method | R@1 | R@10 | R@50 | MRR@10 | User Sat. | Engage. |
|--------|-----|------|------|--------|-----------|---------|
| KBRD | 2.9 | 16.7 | 36.2 | 7.4 | 0.42 | 0.38 |
| UniCRS | 3.8 | 18.1 | 37.4 | 8.4 | 0.45 | 0.41 |
| GPT-4 | — | — | — | — | — | — |
| **HARPO** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Full HARPO | Best | Complete model |
| w/o STAR | Significant drop | Tree search reasoning removed |
| w/o CHARM | Noticeable drop | Hierarchical preference optimization removed |
| w/o BRIDGE | Cross-domain drop | Domain transfer module removed |
| w/o MAVEN | Slight drop | Multi-agent refinement removed |

### Key Findings
- HARPO achieves an average improvement of 17–21% over the strongest baseline (GPT-4), with larger gains on user-alignment metrics.
- The largest gains are observed on the INSPIRED dataset (R@10 45.7% above GPT-4), where social dialogue requires reasoning about implicit preferences.
- Human evaluation confirms significant superiority over GPT-4 on recommendation quality, explanation quality, and overall rating (+0.55, +0.50, +0.55).
- The CHARM reward model achieves Pearson correlations of 0.64–0.73 with independent human judgments.

## Highlights & Insights
- Identifying the misalignment between proxy metrics and recommendation quality as a fundamental problem in CRS represents a paradigm-shifting insight for the field.
- The VTO abstraction decouples reasoning logic from concrete tool implementations—analogous to interface design in software engineering—offering an elegant solution for transferable reasoning.
- Multi-dimensional quality decomposition combined with context-adaptive weighting avoids the information compression inherent in a single reward signal.

## Limitations & Future Work
- The CHARM reward model may itself carry biases; although it correlates with human judgments, it is not a perfect substitute.
- Tree search reasoning introduces additional inference-time computational overhead, which must be considered for latency-sensitive deployment.
- The experimental datasets are limited in scale (ReDial: ~10,000 dialogues), leaving large-scale validation insufficient.
- Future work could explore extending quality dimensions to finer-grained user preference modeling.

## Related Work & Insights
- **vs. UniCRS**: UniCRS unifies recommendation and generation but still optimizes proxy metrics, whereas HARPO directly optimizes recommendation quality.
- **vs. RecMind**: RecMind employs self-motivated reasoning but lacks quality-guided search; HARPO's value network provides explicit quality-oriented guidance.
- **vs. GPT-4**: Even a powerful general-purpose model performs well on proxy metrics but lags behind the specialized optimization of HARPO on user-alignment metrics.

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulating conversational recommendation as a quality-optimized decision problem is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, human evaluation, and ablation studies are all included.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough and the framework design logic is clearly articulated.
- Value: ⭐⭐⭐⭐ Offers important methodological insights for the CRS community.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)

<!-- RELATED:END -->
