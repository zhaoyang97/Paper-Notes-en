---
title: >-
  [Paper Note] Bayesian Social Deduction with Graph-Informed Language Models
description: >-
  [ACL 2026][Social Computing][Social Reasoning] This paper proposes GRAIL (Graph Reasoning Agent Informed through Language), a hybrid reasoning framework that externalizes probabilistic reasoning to a factor graph model w…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Social Reasoning"
  - "Probabilistic Graphical Models"
  - "Theory of Mind"
  - "Game Agents"
  - "Human-Computer Interaction"
date: 2026-05-08
content_hash: 6981851c45bca18f
---

# Bayesian Social Deduction with Graph-Informed Language Models

**Conference**: ACL 2026  
**arXiv**: [2506.17788](https://arxiv.org/abs/2506.17788)  
**Code**: [Project Page](https://camp-lab-purdue.github.io/bayesian-social-deduction)  
**Area**: LLM Agent / Social Reasoning  
**Keywords**: Social Reasoning, Probabilistic Graphical Models, Theory of Mind, Game Agents, Human-Computer Interaction

## TL;DR

This paper proposes GRAIL (Graph Reasoning Agent Informed through Language), a hybrid reasoning framework that externalizes probabilistic reasoning to a factor graph model while using LLMs for language understanding and interaction. It marks the first time an agent has defeated human players in the social deduction game Avalon (67% win rate), with resource consumption significantly lower than large-scale reasoning models.

## Background & Motivation

**Background**: While LLMs excel in general reasoning, social reasoning in multi-agent scenarios with hidden information—inferring others' beliefs, intentions, and deceptions—remains an open challenge. Social deduction games like Avalon provide structured environments to evaluate this capability.

**Limitations of Prior Work**: (1) The largest reasoning models (e.g., DeepSeek-R1 671B) can solve simple reasoning but require excessive tokens and computation; (2) Performance drops sharply when distilled into smaller models; (3) Pure LLM approaches struggle with constrained probabilistic reasoning across long time horizons; (4) High latency in large models prevents real-time interaction with humans.

**Key Challenge**: Social reasoning requires constrained probabilistic reasoning (e.g., the hard constraint of "only 2 evil players") and long-term belief tracking, but LLMs are inherently token-level reasoners and are not adept at such structured reasoning.

**Goal**: To build a social deduction agent capable of real-time competition against humans, achieving or exceeding the performance of large reasoning models even when using smaller models.

**Key Insight**: A hybrid architecture—externalizing belief reasoning to a Probabilistic Graphical Model (Factor Graph + Belief Propagation), while the LLM focuses on language understanding and dialogue generation.

**Core Idea**: Decouple structured reasoning from linguistic ability: the factor graph tracks role beliefs (interpretable and efficient), while the LLM provides linguistic priors and dialogue generation.

## Method

### Overall Architecture

GRAIL consists of three components: (1) Factor Graph—performs probabilistic reasoning over player roles using Max-Product Belief Propagation for MAP inference; (2) LLM—parses dialogues to extract linguistic priors and generates dialogue messages; (3) Heuristic Action Policy—selects game actions (proposing teams, voting) based on beliefs (marginal probabilities).

### Key Designs

1.  **Factor Graph Role Reasoning**:
    - **Function**: Maintains and updates role beliefs for each player under hard constraints (exactly 2 evil players).
    - **Mechanism**: Variable nodes $\mathcal{R} = \{r_1,...,r_6\}$ represent player roles (0=Good/1=Evil). Game state variables $\mathcal{S}$ include team composition, votes, and mission outcomes. Factor functions are approximated by neural networks $F = p(r_j|\{p_i,v_i,o_i\})$, trained on 100,000 historical games.
    - **Design Motivation**: Factor graphs naturally support reasoning with hard constraints and incremental belief updates, making them more precise and reliable than token-based reasoning in LLMs.

2.  **LLM Language Prior Integration**:
    - **Function**: Incorporates unstructured social signals from dialogue into probabilistic reasoning.
    - **Mechanism**: The LLM determines whether a player's belief should "increase/decrease/remain unchanged" ($\delta_j^t$), which is converted into a prior $p(r_j^t) = 0.5 \pm \beta^t$. Here, $\beta^t$ increases as the game progresses (conservative early on, confident later).
    - **Design Motivation**: Structured data lacks dialogue-level information, yet dialogue contains critical social reasoning cues such as contradictions or implied alliances.

3.  **Neural Network Approximation of Factor Functions**:
    - **Function**: Addresses the infeasibility of high-dimensional conditional probability tables.
    - **Mechanism**: Simple feed-forward networks estimate $p(r_j|\text{game state})$. Ego-centric input transformations are used to eliminate position bias, and shared networks eliminate bias between factors.
    - **Design Motivation**: Traditional probability tables are impractical in high-dimensional settings; neural networks provide flexible approximations and can be trained with only 2.5K-5K games.

### Loss & Training

The factor function networks are trained using binary classification loss. No end-to-end Reinforcement Learning (RL) is required; the LLM is utilized via in-context prompting. GRAIL uses GPT-4.1 as the underlying LLM, though ablation studies show it maintains a 75% win rate even with Llama-3.1-8B.

## Key Experimental Results

### Main Results (Agent-Agent)

| Good Agent | Opponent Type | Avg. Win Rate |
| :--- | :--- | :--- |
| Random | Various Evil | 0.00 |
| ReCon (GPT-4.1) | Various Evil | 0.43 |
| GPT-o4-mini (Reasoning) | Various Evil | 0.40 |
| DeepSeek-R1 (671B) | Various Evil | 0.71 |
| **GRAIL (GPT-4.1)** | **Various Evil** | **0.75** |

### Human Experiments

| Condition | Win Rate | Contribution Score | Helpfulness Score |
| :--- | :--- | :--- | :--- |
| GRAIL vs. Humans | **67%** | Higher than reasoning baseline & some humans | Higher than reasoning baseline & some humans |
| GPT-o4-mini vs. Humans | 27% | Lower | Lower |

### Ablation Study

| Configuration | Finding |
| :--- | :--- |
| Factor Graph Only | Robust to model size; 8B model still reaches 75% win rate. |
| LLM Only | Extremely sensitive to model size; 8B performance drops significantly. |
| GRAIL 8B Llama vs. Reasoning 70B DS-R1 | GRAIL 8B achieves a higher win rate. |

### Key Findings

- GRAIL outputs over 10x fewer tokens than reasoning baselines, demonstrating extreme computational efficiency.
- The factor graph provides a "performance floor," maintaining high win rates even with minimal model sizes.
- Linguistic priors accelerate belief convergence—reaching high confidence by round 3 with priors, compared to rounds 4-5 without.
- Reasoning agents exhibit counter-intuitive phenomena: the 405B Llama performed worse than the 70B version due to sycophancy bias.
- GRAIL's hallucination rate is lower than that of reasoning agents across all model sizes.

## Highlights & Insights

- The first language agent to defeat human players in controlled experiments (67% win rate).
- Hybrid architecture insight: Externalizing structured reasoning that LLMs struggle with allows each component to play to its strengths.
- Factor Graphs + Belief Propagation represent an elegant revival of classical AI methods in the LLM era.
- Humans were unaware of AI participation yet rated GRAIL higher than some human teammates.

## Limitations & Future Work

- Evaluated only as the "Good" faction; deception and lying capabilities were not tested.
- Excluded special roles (e.g., Merlin), simplifying game complexity.
- Training factor functions requires a large amount of historical game data.
- Future work could extend to more complex games with incomplete information.

## Related Work & Insights

- **DeepRole (Serrino et al., 2019)**: An Avalon agent trained via self-play (no dialogue).
- **ReCon (Wang et al., 2023)**: An LLM-based reasoning agent for Avalon.
- Application of Probabilistic Graphical Models in social reasoning (Xu et al., 2024a).
- Hybrid neuro-symbolic reasoning is a vital research direction—for structured reasoning tasks, a combination of specialized models + LLMs outperforms pure large models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Hybrid architecture defeats humans for the first time; an elegant combination of classical AI and LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across Agent-Agent, Human experiments, model size ablations, and architectural ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem setup and rigorous human experiment design.
- Value: ⭐⭐⭐⭐⭐ A significant breakthrough for LLM social reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)
- [\[ACL 2026\] Content Fuzzing for Escaping Information Cocoons on Social Media](content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)
- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)

</div>

<!-- RELATED:END -->
