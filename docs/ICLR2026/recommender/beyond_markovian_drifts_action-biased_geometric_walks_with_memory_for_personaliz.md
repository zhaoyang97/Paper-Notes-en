---
title: >-
  [Paper Note] Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization
description: >-
  [ICLR 2026][Recommender Systems][Paper Note] This paper proposes the "Structured Walk Hypothesis" (SWH) to challenge the prevailing "Markovian Drift Hypothesis" (MDH) in personalized summarization. It introduces Walk2Pers, a lightweight encoder-decoder model that characterizes user preference evolution as an action-biased geometric walk with dual memory channels,
tags:
  - ICLR 2026
  - Recommender Systems
date: 2026-05-08
content_hash: c5b6a955dd4a92b0
---
# Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HvOKarTubb](https://openreview.net/forum?id=HvOKarTubb)  
**Code**: To be confirmed  
**Area**: Personalized Recommendation / Personalized Summarization  
**Keywords**: Personalized Summarization, User Preference Modeling, Geometric Random Walk, Dual Memory Channels, Action Conditioning  

## TL;DR
This paper proposes the "Structured Walk Hypothesis" (SWH) to challenge the prevailing "Markovian Drift Hypothesis" (MDH) in personalized summarization. It introduces Walk2Pers, a lightweight encoder-decoder model that characterizes user preference evolution as an action-biased geometric walk with dual memory channels, decomposable into magnitude and orientation (continuity vs. novelty). It significantly outperforms specialized summarizers and Large Language Models (LLMs) across three benchmarks.

## Background & Motivation
- **Background**: Personalized document summarization assists readers by focusing on "content of interest," which is subjective and time-varying. Leading approaches in news recommendation and summarization (e.g., graph diffusion RWR/Personalized PageRank, short-memory neural encoders like NAML/NRMS/EBNR, and prompted mid-sized LLMs) almost universally assume that user preferences follow a **memoryless or short-memory random walk** along the interaction graph—where each new state depends primarily on the most recent interaction.
- **Limitations of Prior Work**: The authors categorize these practices under the **Markovian Drift Hypothesis (MDH)**: history is compressed into a final state/seed vector/hidden state/prompt window, causing long-range action dynamics to be overwritten. Graph diffusion lacks action semantics; neural encoders compress long histories into shallow memories; LLMs are constrained by prompt length and lack persistent reinforcement or inhibition. In datasets like PENS containing click/skip logs, user interests drift across fine-grained subtopics; including long histories in prompts degrades the performance of SOTA LLMs.
- **Key Challenge**: Preference evolution requires **persistent, asymmetric memory of "likes" (click reinforcement) vs. "dislikes" (skip inhibition)** and must distinguish whether evolution continues along the existing trajectory (continuity) or pivots (novelty). MDH fails both requirements by converging everything into a single-step drift.
- **Goal**: To test whether MDH holds for personalized summarization and to provide a more faithful, interpretable, and lightweight modeling alternative.
- **Core Idea**: **Structured Walk Hypothesis (SWH)**—updates to preference states caused by interactions (click/skip/summarize) are decomposed into **(i) Magnitude** (push intensity) + **(ii) Orientation** (continuity vs. novelty), supplemented by **dual memory channels** (reinforcement/inhibition) and a **summary-request-specific drift term**. This theoretically approximates a first-order action-conditioned kernel and is instantiated as Walk2Pers.

## Method

### Overall Architecture
The method abstracts user history into a two-layer structure: the lower **User Interaction Graph (UIG)** records nodes (user/document/summary) and action edges (click/skip/summarize/summGen); the upper layer compresses each interaction into a "behavior dual" b-cell `b=⟨action, tail node⟩`, forming a trajectory via nextBehavior edges. Walk2Pers utilizes a T5-base encoder-decoder framework: the SWH-Encoder performs memory-augmented geometric walks along the b-layer to obtain contextualized embeddings; the Predictor forecasts the next b-node; the Inverse Approximator extracts latent summary intents (s-nodes); the Contextualizer uses cross-attention to fuse summary intent, user history, and query documents; finally, a fine-tuned T5 decoder (top layers only) generates the personalized summary.

```mermaid
flowchart LR
  A[User Interaction Logs<br/>click/skip/summarize] --> B[UIG → b-cell Trajectory]
  B --> C[SWH-Encoder<br/>Dual Memory + Geometric Step]
  C --> D[Predictor<br/>Predict next b-node]
  D --> E[Inverse Approximator<br/>Extract latent s-node]
  E --> F[Contextualizer<br/>Cross-attention fusion]
  F --> G[T5 Decoder<br/>Top-layer fine-tuning]
  G --> H[Personalized Summary]
```

### Key Designs

**1. Unified Update for Structured Walk: Decomposing "One-step Drift" into a Triad of Geometric Step, Memory, and Drift.** MDH assumes the next state depends only on the previous state: $e^{(t+1)}_{b,u}=f(e^{(t)}_{b,u}, a^{(t)}, q)+\epsilon^{(t)}$, where history is folded into a recency prior $q$. SWH rewrites this into an additive family:

$$e^{(t+1)}_{b,u}=e^{(t)}_{b,u}+\underbrace{\mathrm{mag}(a^{(t)})\big(\cos\theta(a^{(t)})\,u^{(t)}+\sin\theta(a^{(t)})\,o^{(t)}\big)}_{\Phi:\ \text{Geometric step, Continuity vs. Novelty}}+\underbrace{\Psi(h^+_t,h^-_t)}_{\Psi:\ \text{Dual Memory}}+\underbrace{\delta\cdot\mathbb{I}[a^{(t)}=\text{summGen}]}_{\Delta:\ \text{Summary Drift}}$$

Where $\mathrm{mag}(\cdot)$ controls step size (single click = small step, repeated = large), and the rotation angle $\theta$ interpolates between the momentum axis $u^{(t)}$ (continuation of interest) and the orthogonal novelty axis $o^{(t)}$ (pivot). This decomposition draws from JODIE's trajectory dynamics and RotatE/ChronoR's angular relations but adds explicit action bias and memory.

**2. Dual Memory Channels Ψ and Summary Drift Δ: Asymmetric Reinforcement and Inhibition.** Walk2Pers uses two channels to represent history $h^{(t)}=\omega^{(t)}h^{(+,t)}+(1-\omega^{(t)})h^{(-,t)}$ ($\omega$ is learnable). The positive channel accumulates reinforcement signals from clicks, while the negative channel accumulates inhibition from skips, using asymmetric update rules:

$$h^{(+,t_i)}=h^{(+,t_{i-1})}+m^{(t_i)}\odot c^{(t_i)}_{tl};\qquad h^{(-,t_i)}=h^{(-,t_{i-1})}\odot(1-m^{(t_i)})+c^{(t_i)}_{tl}$$

Gate $m^{(t_i)}=\mathrm{SoftMax}(W_h h^{(t_{i-1})}+W_c c^{(t_i)}_{tl})$. Summary requests (summGen) trigger a drift vector $\Delta^{(t)}=(I-e^{(t-1)}_{tl})\cdot e^{(t)}_{tl}$, pushing the preference state toward a more concise representation.

**3. Action-Biased b-cell and Geometric Step Implementation.** Nodes are initialized with T5-base; actions are encoded as 4D one-hot vectors. The b-cell fuses action gating and tail-node content $c^{(t_i)}_{tl}=\tanh(f^{(a,t_i)}\odot e^{(t_i)}_{tl})$, where $f^{(a,t_i)}=\mathrm{AGD}(e_a,t_i)\odot h^{(t)}$ leverages the action gates of the AGD baseline. The encoder is supervised by two objectives: a next-node prediction head $\mathcal{L}_{next}$ and a position classification alignment term $\mathcal{L}_{align}$ to ensure intermediate b-nodes can be recovered from contextualized embeddings: $\mathcal{L}_{enc}=\alpha\mathcal{L}_{align}+(1-\alpha)\mathcal{L}_{next}$.

**4. User-Contextualized Attention (T5-UCA): Rewriting Documents via Preference Prisms.** The decoder leverages T5-base with two variants: T5-CA uses cross-attention to contextualize documents with latent summary intents; T5-UCA further uses the user trajectory state to gate document embeddings—suppressing aspects aligned with negative memory $h^-$ and amplifying those aligned with $h^+$. This ensures Alice and Bob receive different summaries for the same document.

## Key Experimental Results

### Main Results
Next b-node prediction task (PENS, 151 candidates):

| Category | Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MDH | NAML | 0.498 | 0.001 | 0.0004 | 0.0007 |
| MDH | NRMS | 0.499 | 0.0009 | 0.0002 | 0.0004 |
| MDH | EBNR | 0.499 | 0.0009 | 0.0003 | 0.0005 |
| MDH | SMD (Ours) | 0.415 | 0.094 | 0.052 | 0.065 |
| MDH | AGD (Ours) | 0.446 | 0.113 | 0.069 | 0.073 |
| SWH | Walk2Pers-Enc. w/o Geo | 0.474 | 0.121 | 0.082 | 0.132 |
| SWH | **Walk2Pers-Enc. Full** | **0.532** | **0.23** | **0.198** | **0.249** |

PENS Personalized Summarization (PerSEval metrics):

| Category | Model | PSE-JSD | PSE-SU4 | PSE-METEOR |
|---|---|---|---|---|
| Oracle (Clues) | BigBird-Pegasus | 0.253 | 0.143 | 0.168 |
| LLM (2-shot) | DeepSeek-14B | 0.248 | 0.094 | 0.097 |
| LLM (2-shot) | Gemini-2.5-Flash | 0.222 | 0.104 | 0.124 |
| MDH Encoders | AGD + T5-UCA | 0.286 | 0.214 | 0.248 |
| SWH (Ours) | **Walk2Pers Full + T5-UCA** | **0.452** | **0.383** | **0.449** |

### Ablation Study

| Variant | PSE-JSD | PSE-SU4 | PSE-METEOR |
|---|---|---|---|
| AGD + T5-UCA (Pure MDH) | 0.286 | 0.214 | 0.248 |
| Walk2Pers w/o Geometric + T5-UCA | 0.306 | 0.334 | 0.321 |
| Walk2Pers Full + T5-CA | 0.418 | 0.341 | 0.422 |
| Walk2Pers Full + T5-UCA | 0.452 | 0.383 | 0.449 |

### Key Findings
- **RQ1: MDH is insufficient.** Short-memory neural encoders (NAML/NRMS/EBNR) show AUC near random (≈0.5) and ranking metrics near zero (MRR≤0.001), indicating compressed hidden states carry almost no predictive signal.
- **RQ2: SWH components provide systematic Gain.** Adding dual memory and drift (w/o geometric) surpasses AGD; adding geometric magnitude-orientation steps yields a significant jump in performance.
- **RQ3: Superiority over LLMs.** Ours significantly outperforms all LLMs (e.g., +0.20/0.29/0.35 over DeepSeek-14B), while chain-of-thought prompting often lags behind MDH baselines.
- **Cross-task Transferability**: Walk2Pers trained for summarization outperforms dedicated baselines on the MIND news recommendation leaderboard (MRR +1.2, nDCG@10 +3.5).

## Highlights & Insights
- **Hypothesis-Driven Research**: Instead of simply stacking models, the paper explicitly names and tests the prevalent MDH before proposing the falsifiable SWH.
- **Geometric Interpretability**: Decomposing updates into magnitude and orientation makes preference shifts readable, unlike opaque hidden states.
- **Asymmetric Dual Memory**: Splitting "likes" and "dislikes" into channels with different update rules aligns with psychological reinforcement/inhibition and persists across long histories.
- **Lightweight Strength**: A T5-base-scale framework outperforms 13B-235B LLMs, suggesting structural priors are more critical than scale for this task.

## Limitations & Future Work
- Evaluation relies heavily on PerSEval; the trade-off between personalization and absolute summary quality (e.g., ROUGE) or fluency is not fully detailed.
- Action sets are restricted to four one-hot types; fine-grained or implicit feedback (e.g., dwell time) is not explored.
- Geometric steps introduce hyperparameters ($\alpha, \omega, \theta$), whose sensitivity across datasets is primarily discussed in the appendix.

## Related Work & Insights
- **Personalized Summarization Metrics**: PerSEval is adopted for its high correlation with human judgment.
- **Dynamic Embedding and Angular Modeling**: Inspired by JODIE’s trajectory dynamics and RotatE’s angular relations.
- **Insight**: Explicit memory channels combined with geometric decomposition may replace implicit attention/GRU aggregation in sequential/session recommendations, particularly for long-range asymmetric feedback.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Explicitly challenges MDH and proposes a geometric-memory alternative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-task validation against MDH, specialized models, and LLMs.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from hypothesis to formula to instantiation.
- **Value**: ⭐⭐⭐⭐ Demonstrates that lightweight models with proper priors can beat LLMs; transferable to recommendation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](../../ACL2026/recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[ICLR 2026\] More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences](more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user.md)
- [\[ICLR 2026\] Low-pass Personalized Subgraph Federated Recommendation](low-pass_personalized_subgraph_federated_recommendation.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](../../AAAI2026/recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)

</div>

<!-- RELATED:END -->
