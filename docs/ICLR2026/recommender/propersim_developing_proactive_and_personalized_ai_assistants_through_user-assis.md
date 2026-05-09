---
title: >-
  [Paper Note] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation
description: >-
  [ICLR 2026][Recommender Systems][proactive agent] This paper proposes ProPerSim, a simulation framework that models daily behaviors of 32 user personas grounded in the Big Five personality model within the Smallville household environment. The AI assistant makes proactive recommendation decisions every 2.5 minutes and learns user preferences via DPO, improving user satisfaction from 2.2/4 to 3.3/4 over a 14-day simulation—providing the first empirical validation of jointly achieving proactivity and personalization.
tags:
  - ICLR 2026
  - Recommender Systems
  - proactive agent
  - personalization
  - user simulation
  - DPO
  - Big Five personality
  - generative agents
date: 2026-05-08
content_hash: 09dcdaab70980f9c
---

# ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation

**Conference**: ICLR 2026
**arXiv**: [2509.21730](https://arxiv.org/abs/2509.21730)
**Code**: [GitHub](https://github.com/jiho283/ProPerSim)
**Area**: Recommender Systems
**Keywords**: proactive agent, personalization, user simulation, DPO, Big Five personality, generative agents

## TL;DR
This paper proposes ProPerSim, a simulation framework that models daily behaviors of 32 user personas grounded in the Big Five personality model within the Smallville household environment. The AI assistant makes proactive recommendation decisions every 2.5 minutes and learns user preferences via DPO, improving user satisfaction from 2.2/4 to 3.3/4 over a 14-day simulation—providing the first empirical validation of jointly achieving proactivity and personalization.

## Background & Motivation

**Background**: LLM-based assistants are evolving in two separate directions: proactive recommendation and personalization. Proactive Agent (Lu et al., 2024) explores proactive recommendations without accounting for individual preferences, while personalization methods (e.g., RLHF) adapt to users but still require user-initiated interaction.

**Limitations of Prior Work**:
- Proactivity alone → recommending a steakhouse to a vegetarian (Figure 1), with mismatches in both timing and content relative to personal preferences
- Personalization alone → even highly accurate recommendations are missed if the user must initiate the interaction
- Large-scale collection of real behavioral data faces prohibitive costs and privacy challenges; human subject experiments are extremely expensive
- Existing proactive research is event-driven (triggered only when a user performs some action), leaving time-based continuous monitoring unexplored

**Key Challenge**: Learning "when to recommend" and "what to recommend" simultaneously requires large-scale user–assistant interaction data, yet collecting such data in practice is infeasible.

**Goal**: Unify proactivity and personalization within a simulation environment to develop AI assistants that adapt to individual users over time.

**Key Insight**: Simulate realistic user behavior with LLM-based user agents equipped with rich Big Five personality-grounded personas, collect preference data within the simulation, and apply DPO training.

**Core Idea**: Generative Agents for user simulation + personalized rubric-based evaluation + DPO preference learning → a continuously improving proactive and personalized closed loop.

## Method

### Overall Architecture

The system comprises three components: (1) a persona-driven user agent that generates daily action sequences $\{(A_i, \text{Range}_i)\}$ in a household environment; (2) an AI assistant that observes user behavior every $T=2.5$ minutes and decides whether to make a recommendation $R_t = \mathcal{A}_\theta(A_t, S_t^{(a)})$; and (3) the user agent that scores each recommendation using a personalized rubric $\text{Score}_t = \mathcal{E}(P, r, A_t, R_t, S_t^{(u)})$.

### Key Designs

1. **Big Five Personality-Driven User Persona System**:
    - **Function**: Constructs 32 diverse user personas to drive behavior generation and recommendation evaluation.
    - **Mechanism**: Each persona is defined by five Big Five dimensions (High/Low Extraversion / Agreeableness / Openness / Conscientiousness / Neuroticism) plus six extended attributes (age, background, interests, lifestyle, daily planning needs, and long-term goals). GPT-4o generates attribute values to ensure consistency with personality traits. UMAP + HDBSCAN validates the separability and diversity of the 32 personas.
    - **Design Motivation**: The Big Five model is the most empirically validated personality framework in psychology. Different personality combinations naturally yield different recommendation preferences—low-extraversion personas prefer solitary activities, while high-conscientiousness personas prefer structured recommendations.

2. **Four-Dimensional Personalized Evaluation Rubric**:
    - **Function**: Provides a set of four evaluation dimensions selected via an AMT survey of 353 participants, with persona-specific criteria generated for each dimension.
    - **Mechanism**: Starting from 10 candidate dimensions, AMT voting excludes those with less than 50% support (Diversity and Interruption), retaining: Personal Preference (content alignment), Frequency (recommendation rate), Timing (contextual appropriateness), and Communication & Safety (communication style + safety). Dimension-specific criteria are customized by GPT-4o per persona (e.g., for a low-extraversion persona: "I prefer receiving recommendations no more than once every two hours"). Evaluation is performed by Gemini 2.0 Flash with binary scoring per dimension.
    - **Design Motivation**: Evaluation criteria must simultaneously reflect the general importance of task dimensions (from large-scale survey) and individual differences (from persona customization). The two-layer design ensures both a consensus foundation and personalized flexibility.

3. **RAG + DPO Preference-Aligned ProPerAssistant**:
    - **Function**: Constructs a proactive recommendation assistant capable of continuously learning from user feedback.
    - **Mechanism**: The internal state $S_t^{(a)}$ contains structured episodic memory (detailed records of the past 10 minutes, with earlier content compressed into 1h/4h summaries) plus top-5 historically similar interactions retrieved via OpenAI embeddings. At each timestep, $n=2$ candidate recommendations (including a "no recommendation" option) are generated; user scores form preference pairs that are stored in a replay buffer. At the end of each day, 200 samples are randomly drawn from the buffer for DPO training:
      $$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\!\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)$$
    - **Design Motivation**: DPO avoids the complexity of reward model training in RLHF. The replay buffer, inspired by experience replay in reinforcement learning, prevents forgetting of early interactions. LoRA fine-tuning of LLaMA 3.3 70B (4-bit quantization) balances performance and efficiency.

### Loss & Training

**Base model**: LLaMA 3.3 70B (4-bit quantization), fine-tuned with LoRA. **DPO training**: 200 samples randomly drawn from the accumulated replay buffer at the end of each day; candidate count $n=2$. **Simulation setup**: timestep $T=2.5$ minutes; each persona is simulated for 14 days. **Per-persona simulation cost**: approximately 10 days × 1 A100 GPU + ~$30 in API fees.

## Key Experimental Results

### Main Results — Method Comparison

| Method | Day 1 Avg. Score | Day 14 Avg. Score | Notes |
|--------|-----------------|-------------------|-------|
| No Memory | ~2.1 | ~2.2 | Current action only |
| AR Memory (A,R) | ~2.3 | ~2.3 | Historical actions + recommendations |
| ARS Memory (A,R,Score) | ~2.6 | ~2.5 | Scores added to prompt |
| **ProPerAssistant** | **~2.2** | **~3.3** | DPO preference learning |

### Persona Dimension Analysis

| Analysis Dimension | Best Persona | Worst Persona | Reason for Gap |
|-------------------|-------------|--------------|---------------|
| Final Score | 3.8/4 | 2.5/4 | Differences in preference complexity |
| Preference Profile | Simple philosophical/creative | Data-driven/debate-oriented | Latter requires multi-dimensional matching |
| Time Window | Flexible | Strict (6–9AM / 21:00+) | Narrow windows are harder to adapt to |

### Key Findings
- ProPerAssistant improves rapidly from Day 2 onward and sustains its lead, with daily average scores approaching 3.4/4, demonstrating that DPO preference learning far outperforms in-context reward signals (ARS Memory).
- Recommendation frequency decreases from an initial ~24 per hour to ~6 per hour, indicating the assistant learns that "not recommending" is equally important.
- The success rate (proportion of recommendations with score ≥ 3) improves from 51.06% to 71.51%.
- Low-extraversion personas show greater improvement (household setting aligns with preference for solitary activities); low-openness personas also improve more (consistent preferences are easier to learn).
- The Frequency and Timing dimensions show the most significant gains; Personal Preference improves more modestly—because as total recommendation volume decreases, the proportion of high-quality recommendations actually rises (0.77 → 0.83).
- Human evaluation confirms high quality: behavior naturalness 8.25/10, persona consistency 8.02/10, evaluation reasonableness rate 90.54%.

## Highlights & Insights
- **First unified proactivity + personalization framework**: Bridges the gap between two independently studied research directions and defines a new task formulation.
- **Time-driven vs. event-driven proactivity**: Decision-making at every $T$ timesteps more closely approximates continuous monitoring by real assistants and is more natural than event-driven approaches.
- **DPO >> in-context reward**: ARS Memory directly incorporates scores into the prompt but performs far worse than DPO training—explicit preference learning is necessary, as in-context reward signals are insufficient to drive genuine adaptation.
- **"Not recommending" is a critical capability**: The assistant's learned suppression of recommendations (frequency reduced by 4×) is as important as improvements in recommendation content quality.

## Limitations & Future Work
- Computational cost is extremely high (~10 A100 days + $30 API per persona); a full experiment across 32 personas requires approximately 320 GPU-days.
- Both user behavior and evaluation are LLM-simulated rather than human-generated—the gap between simulated and real-world behavior has not been quantified.
- The framework is limited to the household setting (Smallville house) and has not been extended to work, social, or outdoor environments.
- The DPO candidate count $n=2$ is constrained by cost; more candidates could provide richer preference signals.
- Only immediate rewards are optimized; delayed rewards such as long-term satisfaction and recommendation diversity are not considered.

## Related Work & Insights
- **vs. Proactive Agent (Lu et al., 2024)**: Lu et al. train a proactive agent using 6,790 training events but do not account for individual preference differences; ProPerSim achieves personalization through persona-driven simulation.
- **vs. Generative Agents (Park et al., 2023)**: Park et al. conduct social simulations with 25 agents; ProPerSim extends the generative agent framework to user–assistant interaction simulation, adding evaluation dimensions and preference learning.
- **vs. Personalized RLHF**: Conventional personalization is achieved through one-time alignment; ProPerAssistant achieves continuous adaptation through a daily-accumulating replay buffer.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unifying proactivity and personalization is a meaningful new direction; the simulation framework is comprehensively designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 32 personas, 4 baselines, personality dimension analysis, and human evaluation—though real-user validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear, evaluation design is systematic, and persona examples are richly illustrated.
- **Value**: ⭐⭐⭐⭐ Provides a valuable simulation platform and baseline for personal assistant research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](../../ACL2026/recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)
- [\[ACL 2026\] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](../../ACL2026/recommender/scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](../../ACL2026/recommender/horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)

<!-- RELATED:END -->
