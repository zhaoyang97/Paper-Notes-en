---
title: >-
  [Paper Note] The Lock-in Hypothesis: Stagnation by Algorithm
description: >-
  [ICML 2025][LLM (Other)][Feedback loop] This paper proposes and formalizes "The Lock-in Hypothesis": the human-AI feedback loop formed during LLM training and deployment solidifies users' pre-existing beliefs, leading to an irreversible loss of collective viewpoint diversity and potentially locking the population into incorrect beliefs.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Feedback loop"
  - "Belief lock-in"
  - "Diversity loss"
  - "Human-AI interaction"
  - "Bayesian modeling"
date: 2026-05-08
content_hash: 98188b9a1c97f603
---

# The Lock-in Hypothesis: Stagnation by Algorithm

**Conference**: ICML 2025  
**arXiv**: [2506.06166](https://arxiv.org/abs/2506.06166)  
**Code**: None (Website: thelockinhypothesis.com)  
**Area**: LLM/NLP  
**Keywords**: Feedback loop, Belief lock-in, Diversity loss, Human-AI interaction, Bayesian modeling

## TL;DR

This paper proposes and formalizes "The Lock-in Hypothesis": the human-AI feedback loop formed during LLM training and deployment solidifies users' pre-existing beliefs, leading to an irreversible loss of collective viewpoint diversity and potentially locking the population into incorrect beliefs.

## Background & Motivation

Large Language Models (LLMs) increasingly influence human beliefs and values, generating a self-reinforcing feedback loop:

1. AI learns values from human data (pre-training + post-training phases)
2. AI influences human viewpoints during interaction
3. The influenced viewpoints are reabsorbed by AI
4. The loop repeats indefinitely

This dynamics is similar to the "Echo Chamber" effect in recommender systems, but with a fundamental difference: recommender systems optimize **individually** (tailored to each user), whereas LLMs predominantly optimize **collectively** (unifying alignment via techniques such as RLHF for all users). Consequently, LLMs could induce a lock-in effect at the collective level.

Three major gaps in prior work:
- Most focus on the **unidirectional** influence of AI on humans, neglecting the bidirectional feedback loop.
- Lack of a **mechanistic explanation** for the feedback loop.
- Most evidence comes from **laboratory** environments rather than real-world usage scenarios.

## Method

### Overall Architecture

This paper progressively argues the lock-in hypothesis from three levels:

1. **Formal Bayesian Model** (§3): Construct a mathematical framework for multi-agent belief updates, deriving the necessary and sufficient conditions for lock-in to occur.
2. **Agent-based LLM Simulation** (§4): Use GPT-4.1-Nano to simulate the belief evolution process of 100 agents.
3. **Causal Inference on Real-world Data** (§5): Test changes in conceptual diversity following GPT model iterations on the WildChat-1M dataset.

### Key Designs

#### 1. Bayesian Belief Update Model

Consider $N$ agents estimating an unknown parameter $\mu \in \mathbb{R}$. Each agent $i$ at time step $t$ receives a noisy observation $o_{i,t} \sim \mathcal{N}(\mu, \sigma_i^2)$ and maintains:

- **Private belief**: Posterior based on their own observations $\mathcal{N}(\hat{\mu}_{i,t}, p_{i,t}^{-1})$
- **Aggregated belief**: Posterior integrating their own and others' information $\mathcal{N}(\hat{\nu}_{i,t}, q_{i,t}^{-1})$

The key lies in: agent $i$ can only observe the **aggregated beliefs** of agent $j$, rather than their private beliefs. This leads to double-counting of information, forming a feedback loop.

#### 2. Trust Matrix

Define a trust matrix $\mathbf{W} \in \mathbb{R}_{\geq 0}^{N \times N}$, where $w_{i,j}$ represents the level of trust agent $i$ places in agent $j$'s belief:

- $w_{i,j} = 0$: Complete disregard
- $0 < w_{i,j} < 1$: Partial trust (discounted precision)
- $w_{i,j} = 1$: Complete trust
- $w_{i,j} > 1$: Excessive trust

For the human-LLM interaction scenario, a specific trust matrix is constructed: 1 LLM + $(N-1)$ human users, where the LLM trusts each human with weight $\lambda_1$ (preference learning strength), and humans trust the LLM with weight $\lambda_2$. There is no direct communication among humans.

#### 3. Phase Transition Conditions for Lock-in (Core Theorem)

**Theorem 3.2 (Feedback loops cause collective lock-in)**: When the spectral radius $\rho(\mathbf{W}) > 1$ of the trust matrix $\mathbf{W}$, there exists at least one agent whose belief almost surely **does not** converge to the true value:

$$\Pr\left[\lim_{t \to \infty} \hat{\mu}_{i,t} = \mu\right] = 0$$

Conversely, when $\rho(\mathbf{W}) < 1$, all agents almost surely converge to the true value.

**Corollary 3.3 (Human-LLM lock-in condition)**: The critical condition for lock-in to occur is:

$$(N-1)\lambda_1 \lambda_2 > 1$$

This is an **extremely weak** condition. For example, when $N = 101$, it only requires $\lambda_1, \lambda_2 > 0.1$ to trigger lock-in—meaning humans and AI trust each other's reported beliefs with a discount factor of less than 10.

#### 4. Lineage Diversity Metric

To evaluate the shift in conceptual diversity, the authors propose a **lineage diversity** metric:

About 5.44 million natural language concepts are extracted from the WildChat corpus, and a concept hierarchy tree $\mathcal{T}$ is constructed via hierarchical clustering. For a concept set $\mathcal{C}$, define:

$$D_{\text{lineage}}(\mathcal{C}; \mathcal{T}) = \frac{\log|\mathcal{T}| - \log \mathbb{E}_{u,v \sim \text{Unif}(\mathcal{C})} [|\mathcal{T}| / |\mathcal{T}_{l(u,v)}|]}{\log |\mathcal{T}|}$$

where $l(u,v)$ is the lowest common ancestor of concepts $u$ and $v$. This metric is normalized to $[0, 1]$, where 1 indicates complete diversity and 0 represents complete homogeneity.

### Loss & Training

This work does not involve model training in the conventional sense, yet the core "training dynamics" are reflected in the recurrence equations of multi-agent Bayesian updates:

- **Precision Update**: $\mathbf{q}_{t+1} = \mathbf{p}_{t+1} + \mathbf{W} \cdot \mathbf{q}_t$
- **Belief Update**: $\hat{\boldsymbol{\nu}}_{t+1} \odot \mathbf{q}_{t+1} = \hat{\boldsymbol{\mu}}_{t+1} \odot \mathbf{p}_{t+1} + \mathbf{W}(\hat{\boldsymbol{\nu}}_t \odot \mathbf{q}_t)$

When $\rho(\mathbf{W}) > 1$, the precision $\mathbf{q}_t$ grows exponentially. However, this growth is driven by the double-counting of information in the feedback loop, rather than actual new evidence. This leads the collective to become extremely overconfident in **incorrect beliefs**.

## Key Experimental Results

### Main Results

#### Agent Simulation Experiments (§4)

100 agents simulated by GPT-4.1-Nano interact over 200 rounds on 4 r/ChangeMyView topics:

| Topic | Initial Belief Distribution | Final Belief Distribution | Diversity Change |
|------|------------|------------|----------|
| Trump & Discourse | Bimodal distribution | Concentrated at extreme value ~0.1 | Sharp decline in entropy |
| Population Decline | Bimodal distribution | Concentrated at extreme value ~0.9 | Sharp decline in entropy |
| Citizens United | Bimodal distribution | Concentrated at extreme value ~0.9 | Sharp decline in entropy |
| RBG Legacy | Bimodal distribution (mean ~0.5) | Concentrated near ~0.9 | Sharp decline in entropy |

#### WildChat Real-world Data Analysis (§5)

| Diversity Metric | GPT-4 Trend | GPT-3.5t Trend | GPT-4-0125 Switch | GPT-3.5t-0613 Switch | GPT-3.5t-0125 Switch |
|-----------|-----------|-------------|----------------|-------------------|-------------------|
| Lineage (value-laden) | ↓ (p<.05) | ↑ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) |
| Lineage (all) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) |
| Depth (value-laden) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p=.41) |
| Topic Entropy | ↓ (p=.07) | ↓ (p=.09) | ↓ (p<.05) | ↓ (p<.05) | ↑ (p=.06) |
| Jaccard Distance | ↓ (p<.05) | ↑ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p=.63) |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| $(N-1)\lambda_1\lambda_2 = 0.9$ | Converges to true value | Spectral radius < 1, feedback loop self-decays, no lock-in occurs |
| $(N-1)\lambda_1\lambda_2 = 1.0$ | Tends to true value but overconfident | Critical point, precision explosion begins to occur |
| $(N-1)\lambda_1\lambda_2 = 1.1$ | Converges to false value | Spectral radius > 1, each simulation locks in on a different false belief |
| Template messages removed | Lineage ↓ (p<.05) | Diversity decline is still observed after excluding API usages |
| Value-laden concepts only | Lineage ↓ (p<.05) | Lock-in effect is more significant in moral/political/religious domains |
| Per-user regression | ↓ (p=.07) | Remaining negative after controlling for user identity and other confounding variables, moderate support |

### Key Findings

1. **Phase Transition Phenomenon**: There is a clear critical threshold $(N-1)\lambda_1\lambda_2 = 1$, beyond which the collective inevitably locks into incorrect beliefs.
2. **Belief Convergence Patterns**: Three convergence patterns are observed in simulation—opinion flipping, hedging, and extreme convergence.
3. **Model Updates Accelerate Diversity Loss**: Following the launch of new GPT versions, the conceptual diversity of human messages in WildChat exhibits a **discontinuous sudden drop**.
4. **Behavioral Divergence between GPT-4 and GPT-3.5**: The conceptual diversity of value-laden domains for GPT-4 users continuously declines, whereas the trend for GPT-3.5 users is more ambiguous.

## Highlights & Insights

- **Three-layered Argument (Theory-Simulation-Empirics)**: From the mathematical Bayesian model to LLM agent simulations and empirical causal inference on real-world data, the logical progression is highly rigorous.
- **Extremely Weak Phase Transition Condition**: The condition $(N-1)\lambda_1\lambda_2 > 1$ is easily met in practice, implying that lock-in might already be happening.
- **Novel Evaluation Metric**: The proposed Lineage Diversity utilizes conceptual hierarchies, capturing semantic-level diversity shifts better than traditional Shannon entropy.
- **Regression Kink Design (RKD)**: Cleverly leverages GPT version updates as an exogenous event for natural experiments, partially solving the causal inference challenge.
- **Connecting Recommender Systems with LLM Alignment Literature**: Highlights that echo chamber research in recommender systems operationally targets the individual level, whereas the LLM lock-in effect operates at the collective level, establishing an important conceptual distinction.

## Limitations & Future Work

1. **No Randomized Controlled Trial (RCT)**: The WildChat analysis cannot rule out all time-series confounding variables; cooperation with AI labs for authentic RCTs is required.
2. **Over-simplified Simulation**: The 100 agents are simulated by the same LLM, departing from actual human belief update dynamics; the introduction of new empirical evidence is not accounted for.
3. **Ambiguous Results for GPT-3.5**: Hypothesis 1 was not consistently supported on GPT-3.5, which might relate to the division of labor between low-end and high-end models, but in-depth analysis is lacking.
4. **Prompting-dependent Concept Extraction**: Both concept extraction and value-laden tagging rely on a GPT-4o-mini prompting pipeline, potentially introducing systematic bias.
5. **Focus Only on Diversity Loss**: Loss of diversity is a necessary but not sufficient condition for lock-in; empirical evidence demonstrating absolute lock-in remains absent.
6. **Lack of Mitigation Strategies**: The paper only identifies the problem without proposing concrete algorithmic or policy-based intervention strategies.

## Related Work & Insights

- **Recommender System Echo Chambers**: Research on echo chamber effects in recommender systems (Cinelli et al., 2021) is the closest analogy, but LLM’s collective alignment mechanism accentuates the lock-in effect on a population scale.
- **Model Collapse** (Shumailov et al., 2024): Performance degradation when models are trained on their own generated data shares the core structure of "feedback loops" with the lock-in hypothesis.
- **Iterative Learning Theory** (Griffiths & Kalish, 2007): Individuals learn from others who learn in the same manner. This work incorporates considerations of topological networks and mutual trust on top of that.
- **Blind Spots of RLHF**: Current alignment methods treat human feedback as an uninfluenceable oracle (Bai et al., 2022), ignoring the reverse influence of LLMs on human preferences.
- **Inspiration for Future Research**: How can the feedback loop in preference learning be broken? Could controlling $\lambda_1$ (reducing preference learning intensity) or $\lambda_2$ (prompting users to think independently) keep $(N-1)\lambda_1\lambda_2 < 1$?

## Rating

| Dimension | Score (1-5) | Description |
|------|----------|------|
| Novelty | 5 | First to formalize and empirically examine the lock-in effect of the human-LLM feedback loop. |
| Theoretical Depth | 5 | Rigorous Bayesian modeling; elegant derivation of phase transition conditions. |
| Experimental Thoroughness | 4 | Three-layered proof, but causal inference on WildChat still has limitations. |
| Writing Quality | 4 | Clear structure, though LaTeX formula-dense; some readers may require a strong mathematical background. |
| Impact | 5 | Profound implications for AI safety, alignment, and governance. |
| **Overall Score** | **4.6** | Outstanding interdisciplinary work combining robust theory and solid empirics. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Binary Hypothesis Testing for Softmax Models and Leverage Score Models](binary_hypothesis_testing_for_softmax_models_and_leverage_score_models.md)
- [\[ACL 2025\] Literature Meets Data: A Synergistic Approach to Hypothesis Generation](../../ACL2025/llm_nlp/literature_meets_data_hypothesis.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](../../ACL2025/llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](../../ACL2025/llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)

</div>

<!-- RELATED:END -->
