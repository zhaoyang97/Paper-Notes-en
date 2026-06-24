---
title: >-
  [Paper Note] AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise
description: >-
  [NeurIPS 2025][LLM (Other)][Bayesian Surprise] AutoDiscovery proposes Bayesian Surprise as an objective reward signal for open-ended scientific discovery — estimating the KL divergence between prior and posterior belief distributions via LLM sampling, combined with MCTS and progressive widening to explore the hypothesis space. On 21 real-world datasets, the method produces 5–29% more surprising discoveries than greedy/beam search baselines. Human evaluation confirms that Baye…
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "Bayesian Surprise"
  - "Open-ended Discovery"
  - "MCTS"
  - "Hypothesis Generation"
  - "LLM Agent"
date: 2026-05-08
content_hash: 06dce8b604fa1ec8
---

# AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise

**Conference**: NeurIPS 2025
**arXiv**: [2507.00310](https://arxiv.org/abs/2507.00310)  
**Code**: [https://github.com/allenai/autodiscovery](https://github.com/allenai/autodiscovery)  
**Area**: Automated Scientific Discovery
**Keywords**: Bayesian Surprise, Open-ended Discovery, MCTS, Hypothesis Generation, LLM Agent

## TL;DR
AutoDiscovery proposes Bayesian Surprise as an objective reward signal for open-ended scientific discovery — estimating the KL divergence between prior and posterior belief distributions via LLM sampling, combined with MCTS and progressive widening to explore the hypothesis space. On 21 real-world datasets, the method produces 5–29% more surprising discoveries than greedy/beam search baselines. Human evaluation confirms that Bayesian Surprise aligns with expert "surprise" ratings (0.67), substantially outperforming LLM self-evaluated "novelty" and "usefulness."

## Background & Motivation

**Background**: Goal-driven automated scientific discovery requires humans to specify research questions. Open-ended discovery — where the system autonomously explores without predefined objectives — is more ambitious but lacks reliable intrinsic reward signals.

**Limitations of Prior Work**: (a) Diversity heuristics are insufficient — the hypothesis space is vast, and uniform exploration wastes the evaluation budget. (b) Human proxy metrics ("interestingness," "novelty," "usefulness") are subjective, inconsistent across experts, and unreliable to automate — experiments show that LLM-evaluated "interestingness" is nearly uncorrelated with human "surprise."

**Key Challenge**: There is no objective, automatically computable reward signal for open-ended discovery that aligns with human scientific intuition.

**Goal**: To define and implement open-ended scientific discovery driven by Bayesian Surprise.

**Key Insight**: Bayesian Surprise = KL divergence between posterior and prior beliefs — a hypothesis that is "surprised" by experimental evidence (i.e., substantially shifts beliefs) constitutes an interesting finding. Prior/posterior Beta distribution parameters are estimated via LLM sampling.

**Core Idea**: LLM sampling estimates prior/posterior beliefs → Beta-Bernoulli KL divergence = Bayesian Surprise → serves as a reward signal for MCTS-driven hypothesis space exploration.

## Method

### Overall Architecture
**Reward**: For hypothesis $H$, the LLM samples $n$ times to estimate the prior $P(\theta_H)$ and posterior $P(\theta_H|\mathcal{V}_D)$ → Beta-Bernoulli fitting → KL divergence = Bayesian Surprise $\text{BS}(H, \mathcal{V}_D)$. **Search**: MCTS + progressive widening → UCT balances exploration/exploitation → each iteration: selection → expansion → execution (hypothesis verification) → backpropagation of surprise. **Agent**: Multi-agent architecture comprising a hypothesis generator, experiment programmer, analyst, reviewer, and reviser.

### Key Designs

1. **Bayesian Surprise Estimation**:

    - Function: Quantifies the degree to which experimental evidence shifts belief in a hypothesis.
    - Mechanism: The LLM samples $n$ binary (true/false) judgments for hypothesis $H$; with $k_{prior}$ "true" responses, the prior is $P_{est}(\theta_H) = \text{Beta}(1+k_{prior}, 1+n-k_{prior})$. The posterior is estimated analogously after experimental verification. $\text{BS} = D_{KL}(P_{post} \| P_{prior})$. An additional belief-shift condition requires the expected posterior to cross a threshold $\delta=0.5$ (i.e., flipping from "likely true" to "likely false" or vice versa).
    - Design Motivation: In information theory, the magnitude of belief change equals information gain — precisely capturing the essence of "surprise." The Beta-Bernoulli model is the simplest conjugate pair, and binary LLM sampling suffices for estimation.

2. **MCTS + Progressive Widening**:

    - Function: Efficiently searches the hypothesis space for high-surprise hypotheses.
    - Mechanism: $\text{UCT}(H) = \frac{\sum_{h \in subtree(H)} S(h)}{N(H)} + C\sqrt{\frac{2\log N(H_{parent})}{N(H)}}$. Progressive widening limits each node to at most $kN^\alpha$ children. Four-phase iteration: selection → expansion → execution → backpropagation.
    - Design Motivation: Greedy search becomes trapped in local optima (repeatedly exploring near the first high-surprise hypothesis found); UCT in MCTS balances exploration depth and breadth.

3. **LLM-based Deduplication (HAC)**:

    - Function: Merges semantically equivalent hypotheses to avoid redundant evaluation.
    - Mechanism: Text embedding → hierarchical agglomerative clustering (HAC) → each merge decision is adjudicated by GPT-4o (merged if >70% of sampled votes indicate "equivalence").
    - Design Motivation: Identically worded hypotheses with different phrasings waste the evaluation budget — deduplication is critical for efficiency.

### Loss & Training
- No training — purely inference-time search.
- Budget: 500 hypothesis evaluations.
- Evaluated on 21 real-world datasets (5 from DiscoveryBench + 15 from BLADE + 1 from SEA-AD).

## Key Experimental Results

### Main Results (Cumulative Surprise, 500 Iterations)

| Search Method | Cumulative Surprise | vs. AutoDiscovery |
|---|---|---|
| Repeated Sampling (baseline) | ~20–25 | −5 to −29% |
| Last-K Linear | ~25–30 | ~−15% |
| Greedy Tree | ~25–30 | ~−25% |
| Beam Search | ~30 | ~−10% |
| **AutoDiscovery (MCTS)** | **40+** | — |

AutoDiscovery achieves the best performance on 17 out of 21 datasets.

### Human Evaluation (1,620 LLM-surprising hypotheses, 3 experts per hypothesis)

| Reward Signal | Human Surprise | Human Interestingness | Human Usefulness |
|---|---|---|---|
| **Bayesian Surprise** | **0.67** | 0.73 | 0.79 |
| LLM Surprise | 0.11 | 0.76 | 0.80 |
| LLM Interestingness | 0.15 | 0.74 | 0.78 |
| LLM Usefulness | 0.21 | 0.73 | 0.78 |

### Ablation Study / Validation

| Metric | Result |
|---|---|
| Experimental Validity | 98.58% (Gwet's AC1 = 0.97) |
| Implementation Validity | 98.01% (Gwet's AC1 = 0.98) |
| Deduplication Accuracy | 90.76% |

### Key Findings
- Bayesian Surprise aligns with human "surprise" at 0.67, far exceeding LLM self-evaluation (0.11–0.21) — demonstrating that subjective metrics are unreliable while information-theoretic metrics are robust.
- "Interestingness" and "usefulness" scores are nearly identical across all reward signals (~0.73–0.80), rendering them ineffective as discriminative metrics.
- MCTS search efficiency does not degrade over iterations (unlike greedy/beam search), as UCT automatically balances exploration and exploitation.
- The belief-shift condition is important — it filters out low-quality "surprises" that only marginally adjust beliefs.

## Highlights & Insights
- **Bayesian Surprise is the first successful reward signal for open-ended discovery**: All prior attempts (diversity, interestingness, novelty) are either insufficiently objective or cannot be automated reliably.
- **Quantitative evidence that LLM subjective evaluation is unreliable**: LLM-judged "surprise" correlates with human ratings at only 0.11 — a strong warning about the limitations of LLM-as-Judge.
- **Application of MCTS to scientific discovery** demonstrates the cross-domain value of search algorithms — from Go to scientific hypothesis spaces.

## Limitations & Future Work
- Assumes that the LLM's knowledge frontier approximates the human knowledge frontier (an assumption that will gradually hold as models improve).
- The reasoning process is unsupervised (supervised reasoning could improve sample efficiency in future work).
- Evaluation is limited to data-driven discovery (no wet-lab experiments; limited literature-based discovery).
- Deployment requires academic caution and peer-review safeguards.

## Related Work & Insights
- **vs. MOOSE-Chem/OpenScienceAgent**: These are goal-driven discovery systems (requiring a research question as input); AutoDiscovery operates in an open-ended setting.
- **vs. Curiosity-driven RL**: Curiosity = prediction error; Bayesian Surprise = belief change — the latter is more appropriate for scientific discovery.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bayesian Surprise + MCTS-driven open-ended scientific discovery constitutes an entirely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 21 datasets + 4 search baselines + 1,620-hypothesis human evaluation + agent validity verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical motivation is clear; experimental design is rigorous.
- Value: ⭐⭐⭐⭐⭐ Potentially opens a new direction for autonomous scientific discovery with LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[NeurIPS 2025\] Cultural Alien Sampler: Open-ended Art Generation Balancing Originality and Coherence](cultural_alien_sampler_open-ended_art_generation_balancing_originality_and_coher.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)
- [\[ACL 2025\] Enough Coin Flips Can Make LLMs Act Bayesian](../../ACL2025/llm_nlp/coin_flips_bayesian.md)
- [\[ACL 2025\] Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](../../ACL2025/llm_nlp/enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)

</div>

<!-- RELATED:END -->
