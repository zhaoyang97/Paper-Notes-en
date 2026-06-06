---
title: >-
  [Paper Note] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning
description: >-
  [ICML 2026][Audio & Speech][Full-duplex speech dialogue] This paper proposes FLAIR: a framework that enables full-duplex Spoken Dialogue Models (SDLM) to replace the standard `<SIL>` tokens during the "user listening" ph…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Full-duplex speech dialogue"
  - "implicit thinking"
  - "ELBO"
  - "think-while-listening"
  - "variational inference"
date: 2026-05-08
content_hash: 6b82d53eed931970
---

# The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning

**Conference**: ICML 2026  
**arXiv**: [2603.17837](https://arxiv.org/abs/2603.17837)  
**Code**: TBD  
**Area**: Speech Dialogue / Full-duplex SDLM / Latent Reasoning  
**Keywords**: Full-duplex speech dialogue, implicit thinking, ELBO, think-while-listening, variational inference

## TL;DR
This paper proposes FLAIR: a framework that enables full-duplex Spoken Dialogue Models (SDLM) to replace the standard `<SIL>` tokens during the "user listening" phase with continuous latent reasoning. By utilizing an ELBO training objective and a non-causal "global expert" to provide the posterior, the causal LLM learns to "think while listening" using a sequence of embedding vectors, significantly improving answer quality without introducing any inference latency.

## Background & Motivation
**Background**: Full-duplex SDLMs (e.g., Moshi, SALMONN-omni) process "listening" and "speaking" as dual concurrent streams. The model must output a text token at every time step. During user speech, the standard practice is to repeatedly output `<SIL>` placeholder tokens to align the text stream length with the audio stream.

**Limitations of Prior Work**: Outputting only sentinel tokens while listening wastes the computational capacity of the listening window. A direct alternative is to port CoT (Chain of Thought) from NLP by generating explicit text reasoning chains synchronously. however, speech streams are **causal**: reasoning cannot precede unfinished utterances. Furthermore, users may finish speaking at any moment; if a model is locked into a predetermined text reasoning chain, interrupting it to switch to a response introduces significant state management latency.

**Key Challenge**: High-quality reasoning requires "computational steps," yet full-duplex real-time requirements demand **causality + zero extra latency + no explicit CoT labeling**, three constraints that simultaneously impede explicit reasoning schemes.

**Goal**: To enable the model to perform continuous "thinking" during the listening phase and translate these thoughts into improved response quality, without breaking causality, increasing inference latency, or requiring additional reasoning datasets.

**Key Insight**: The authors abandon the premise that "thinking must be explicitly presented as discrete tokens." Instead, they model "internal cognition" as a **continuous latent variable trajectory** that evolves with user input. Since no ground-truth labels exist, they employ variational inference (ELBO) with a non-causal "global expert" serving as the posterior provider.

**Core Idea**: Formulate "think-while-listening" as a latent variable modeling problem using ELBO. The causal SDLM aligns its prior distribution with the posterior of a non-causal expert (which has seen the full dialogue) via KL divergence, internalizing global reasoning capabilities into the implicit prior during the streaming listening phase.

## Method

### Overall Architecture
FLAIR modifies the input of full-duplex SDLMs during the listening phase. Conventionally, `<SIL>` tokens are written to the text stream while the user speaks; FLAIR replaces these with **latent reasoning embeddings** $Z$. The workflow is as follows:

- **During Training**: In addition to the causal LLM $P_\theta$, a non-causal "global expert" $Q_\phi$ is introduced. It observes the entire dialogue (user audio $X$ + assistant text embeddings $H^{txt}$) and outputs ideal latent reasoning embeddings $Z_t$ at each step as "soft labels." A temporal indicator $G_t \in \{0,1\}$ is also introduced to mark whether the current step is assistant speaking (1) or user speaking (0).
- **During Inference**: The expert is discarded, leaving only the causal LLM and a timing predictor $\hat{G}_t$. When a speaking step is predicted, standard text tokens are output. When a listening step is predicted, the text logits of the current step are transformed via a softmax-weighted vocabulary embedding matrix to obtain the next input embedding $Z_t$, while `<SIL>` is sent to the audio side.

The mechanism is fully step-aligned, requiring no chunks or extra computation during inference.

### Key Designs

1. **Vocab-weighted Recursive Latent Embedding**:
    - **Function**: Implements "implicit thinking" as a sequence of evolving continuous embeddings during the listening phase, serving as input for the next LLM step.
    - **Mechanism**: Rather than feeding the previous hidden state $h^l_{t-1}$ directly back (as in Coconut), it passes through the LM head to obtain text logits $y^{txt}_{t-1}$. Then, $Z_{t-1} = \text{Softmax}(y^{txt}_{t-1}) E$ is calculated as a weighted average over the vocabulary embedding matrix $E \in \mathbb{R}^{|V|\times d}$. Thus, "implicit thoughts" always reside on the semantic manifold spanned by vocabulary embeddings, equivalent to "soft tokens."
    - **Design Motivation**: Directly recycling hidden states can introduce train/test mismatch due to distributional differences between hidden and embedding spaces. Using softmax-over-vocab maintains interpretability (one can `argmax` the "soft tokens" at any time) and constrains the search to the learned semantic space.

2. **ELBO + Global-aware Expert SFT**:
    - **Function**: Provides supervision for each implicit step during the listening phase without CoT ground-truth labels.
    - **Mechanism**: The objective is converted into a conditional ELBO for $\log P_\theta(Y^{txt}|X)$: $$\log P_\theta(Y^{txt}|X) \geq \mathbb{E}_{q_\phi(Z|X,Y^{txt})}[\log P_\theta(Y^{txt}|Z,X)] - \text{KL}[q_\phi(Z|X,Y^{txt}) \| P_\theta(Z|X)]$$. The first term is the **conditional reconstruction loss** $\mathcal{L}_{reco}$ (calculated via teacher forcing only during assistant speaking steps for the NLL of the next text token). The second term is the **variational regularization** $\mathcal{L}_{regu}$ (calculated only during user speaking steps to pull the causal LLM's predicted vocab distribution toward the expert's distribution $W^e_t$ using stop-gradient). The expert $Q_\phi$ is a non-causal encoder that uses future information to estimate the posterior; the causal LLM internalizes this global posterior as a prior that can be generated in a streaming fashion via KL alignment.
    - **Design Motivation**: Teacher forcing is key to SFT efficiency, but implicit tokens lack ground truth. ELBO elegantly models "ideal implicit labels" through an expert-estimated approximate posterior, bypassing the need for CoT datasets and avoiding the instability of RL or resampling. ELBO does not hinder subsequent RL post-training and is orthogonally stackable.

3. **Latent Reasoning Timing Predictor**:
    - **Function**: Decides whether to "think" or "speak" at each time step during inference.
    - **Mechanism**: The final hidden state $h^l$ is passed through an MLP to obtain $\hat{G}_t \in [0,1]$, supervised by a BCE loss $\mathcal{L}_{time} = -\sum_t [G_t \log \hat{G}_t + (1-G_t)\log(1-\hat{G}_t)]$. During inference, if $\hat{G}_t=1$, a text token is generated; if $\hat{G}_t=0$, vocab-weighted embedding recursion is performed.
    - **Design Motivation**: Full-duplex systems must autonomously decide when to start/stop vocalizing (turn-taking/barge-in). Binding this directly to latent steps makes "starting to speak after finishing thinking" an emergent behavior of the model rather than being triggered by external VAD or rules, ensuring that latency is not locked by explicit CoT.

### Loss & Training
The total loss is $\mathcal{L}_{elbo} = \mathcal{L}_{reco} + \alpha \cdot \mathcal{L}_{regu} + \beta \cdot \mathcal{L}_{time}$. Training consists of three phases: (i) Pre-training: standard full-duplex speech continuation (outputting `<SIL>`); (ii) Latent Reasoning SFT in two sub-stages: first using only $\mathcal{L}_{reco}$ to let the expert learn to produce viable implicit labels, then joint training of the causal LLM and expert using the full $\mathcal{L}_{elbo}$; (iii) Speech Synthesizing SFT: freezing the backbone to train the streaming TTS module (CosyVoice 2 flow-matching). Data includes 530k hours of synthetic speech continuation + 70k hours of instruction QA + 20k hours of ASR-QA.

## Key Experimental Results

### Main Results (QA benchmarks, sampled from Table 1)

| Dataset | Metric | FLAIR w/o thk | FLAIR w/ thk | Gain | Remarks |
|---------|--------|---------------|--------------|------|---------|
| LlamaQ | Acc % | 73.0 | **78.0** | +5.0 | Best Full-Duplex |
| TriviaQA | Acc % | 54.4 | 56.2 | +1.8 | — |
| SDQA | Acc % | 3.80 | 3.85 | +0.05 | GPT-Score |
| AlpacaEval | GPT | 3.80 | 3.85 | +0.05 | Open-ended QA |
| CommonEval | GPT | 3.54 | 3.65 | +0.11 | Human recordings |
| OpenbookQA | Acc % | 72.9 | 74.2 | +1.3 | Multiple Choice |
| **MMSU** | Acc % | 50.2 | **56.2** | **+6.0** | Largest reasoning gain |

Compared to full-duplex baselines like Moshi (54.5 LlamaQ), SALMONN-omni (73.6 LlamaQ), and Freeze-Omni, FLAIR w/ thk achieves state-of-the-art (SOTA) in full-duplex tasks like LlamaQ, MMSU, and OpenbookQA. It is comparable to or outperforms half-duplex large models like Kimi-Audio and Baichuan-Audio.

### Dialogue Dynamics (Table 2 Impatient Dataset + Table 3 Full-Duplex-Bench)

| Config | E2E? | Turn-taking lat. ↓ | Barge-in lat. ↓ | Barge-in succ. ↑ | MOS ↑ |
|--------|------|--------------------|------------------|-------------------|-------|
| Moshi | ✓ | — | 0.81 | 55.1 | 3.9 |
| ORISE | ✓ | 0.43 | 0.61 | 96.8 | 4.2 |
| FLAIR w/o thk | ✓ | 0.33 | 0.49 | 100 | 4.3 |
| FLAIR w/ thk | ✓ | 0.39 | 0.46 | **100** | 4.3 |

With latent reasoning, turn-taking latency increased slightly from 0.33s to 0.39s, while barge-in latency decreased (0.46s), and the success rate remained at 100%. On Full-Duplex-Bench, TOR and GPT-4o scores did not degrade (barge-in GPT-4o score increased from 4.08 to 4.22).

### Key Findings
- **Reasoning-heavy tasks benefit most**: MMSU +6.0, LlamaQ +5.0; whereas fact-extraction tasks like WebQ saw only +1.3 or slight decreases, verifying that implicit steps perform actual "reasoning" rather than adding noise.
- **No additional inference latency**: The listening phase inherently waits for the user to finish; converting these token slots to latent steps recovers compute at "zero cost." The 0.06s increase in turn-taking latency is within tolerance.
- **t-SNE Visualization (Fig. 3)**: When user audio embeddings, target text embeddings, and latent reasoning embeddings are projected to 2D, the latent embeddings extend from the audio cluster toward the text cluster along a "feasible manifold," forming a "bridge." This suggests that latent steps perform cross-modal alignment and planning rather than simple feature echoing.
- **Expert failure leads to collapse**: Ablations (mentioned in Appendix E) show the non-causal expert is the source of the ELBO signal. If the expert is downgraded to causal, $\mathcal{L}_{regu}$ loses its pull, and latent steps degrade to fitting silent tokens.

## Highlights & Insights
- **ELBO as an "Implicit CoT Label Generator" for SFT**: Unlike prior implicit reasoning methods like Coconut/CODI that rely on expensive resampling or RL, FLAIR uses a non-causal expert + KL alignment to achieve teacher-forcing level efficiency in supervising "what to think." This is transferable to any scenario where future information is available during training but not inference (e.g., simultaneous translation, low-latency streaming ASR).
- **Vocab-weighted Embedding Recursion**: Compared to raw hidden state recycling, this detail maintains interpretability and avoids embedding space mismatch, serving as an undervalued key in latent reasoning implementation.
- **Timing Predictor for Autonomous Turn-taking**: Modeling "when to speak" as a learned endogenous signal effectively absorbs VAD/turn-taking decisions into the LLM forward pass, representing a template for a unified "listen-think-speak" architecture.
- **Transferable Concept**: The abstract idea that "internal cognition is a latent variable" can be applied to any streaming task with wasted waiting time—such as "observation idle frames" in robot policies or interval frames in video generation for implicit planning.

## Limitations & Future Work
- **Dependency on Expert Convergence**: Two-stage SFT relies heavily on the expert learning a viable latent distribution in the first stage. If expert capacity or data is insufficient, the second-stage KL may pull in the wrong direction.
- **Interpretability limited to argmax**: While latent embeddings can be `argmax`-ed into "soft tokens" for inspection, it remains unclear whether these long sequences are human-readable or correspond to distinct named cognitive segments (lack of qualitative analysis beyond t-SNE).
- **Synthetic Training Data**: 530k+70k+20k hours are mostly synthetic English assistant-style corpora. Generalization across languages, accents, and noisy environments has not been deeply tested.
- **Varying Task Returns**: Small gains in fact-extraction tasks (WebQ) suggest that the "latent step" mechanism does not provide positive returns for all tasks. Future work should consider a task router to gate the "thk" module.

## Related Work & Insights
- **vs. Coconut / CODI**: While both perform "hidden state recycling," FLAIR uses vocab-weighted soft embeddings + ELBO supervision for more stable signals and is the first to apply this to streaming SDLM scenarios.
- **vs. STITCH ("think-while-talking")**: STITCH inserts explicit CoT chunks during the speaking phase by breaking up chunks; FLAIR hides thinking entirely within the listening phase, maintaining zero causal latency. Both are orthogonally combinable.
- **vs. SHANKS / Arora et al. (streaming CoT in SDLM)**: SHANKS uses explicit CoT, which either requires custom CoT datasets or breaks causality. FLAIR’s use of ELBO to bypass CoT labeling is its core methodological differentiator.
- **vs. Moshi / Freeze-Omni / SALMONN-omni**: All share a "streaming encoder + LLM + streaming TTS" architecture. FLAIR only modifies the input semantics of the listening phase, making it nearly plug-and-play for these existing SDLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce latent reasoning into speech LLMs; the ELBO + global expert formulation is a clean, original combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 QA benchmarks + 2 dialogue dynamics evaluations + t-SNE, though many ablation details are in the Appendix and the sensitivity to latent step length is missing from the main tables.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation—formula—architecture—experiment loop is clear; ELBO derivation directly maps to implementation loss terms.
- Value: ⭐⭐⭐⭐⭐ Provides a zero-latency solution for the long-accepted waste of "listening compute" in full-duplex models, representing a paradigm shift for real-time speech agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](../../ACL2026/audio_speech/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ACL 2026\] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](../../ACL2026/audio_speech/sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](../../ACL2026/audio_speech/mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ICLR 2026\] Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](../../ICLR2026/audio_speech/stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)

</div>

<!-- RELATED:END -->
