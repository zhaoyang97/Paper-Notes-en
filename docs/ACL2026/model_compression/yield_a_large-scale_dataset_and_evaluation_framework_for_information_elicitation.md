---
title: >-
  [Paper Note] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents
description: >-
  [ACL 2026][Model Compression][Information Elicitation] This paper introduces the Information Elicitation Agent (IEA) as a novel conversational paradigm, releases YIELD — the first large-scale human-human information elicitation dialogue dataset (2,281 conversations, 26M tokens) — formalizes the elicitation process as a finite-horizon POMDP, and proposes dedicated evaluation metrics (Conformity, Progression, TLR). Experiments demonstrate that fine-tuning on YIELD significantly improves LLM alignment with authentic elicitation behavior.
tags:
  - ACL 2026
  - Model Compression
  - Information Elicitation
  - Dialogue Dataset
  - Reinforcement Learning
  - Conversational Agents
  - POMDP
date: 2026-05-08
content_hash: 2ae2edb2a7c8ea97
---

# YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents

**Conference**: ACL 2026
**arXiv**: [2604.10968](https://arxiv.org/abs/2604.10968)
**Code**: [https://github.com/infosenselab/yield](https://github.com/infosenselab/yield)
**Area**: Model Compression
**Keywords**: Information Elicitation, Dialogue Dataset, Reinforcement Learning, Conversational Agents, POMDP

## TL;DR

This paper introduces the Information Elicitation Agent (IEA) as a novel conversational paradigm, releases YIELD — the first large-scale human-human information elicitation dialogue dataset (2,281 conversations, 26M tokens) — formalizes the elicitation process as a finite-horizon POMDP, and proposes dedicated evaluation metrics (Conformity, Progression, TLR). Experiments demonstrate that fine-tuning on YIELD significantly improves LLM alignment with authentic elicitation behavior.

## Background & Motivation

**Background**: Most conversational agents (CAs) are designed for user-driven interaction, where the user controls the agenda and direction while the agent provides assistance. Prominent datasets such as MultiWOZ and SGD are oriented toward this paradigm.

**Limitations of Prior Work**: Many real-world scenarios — academic interviews, judicial proceedings, investigative journalism — require an agent to proactively elicit information from the user in service of the agent's own institutional or task objectives. This "information elicitation" differs fundamentally from conventional CA design in three respects: (1) the locus of conversational control shifts to the agent, which must actively guide the dialogue through questioning; (2) success is defined by the amount of valuable information obtained rather than the resolution of a user problem; (3) there is no single optimal question, but rather multiple possible directions, each potentially yielding valuable information.

**Key Challenge**: Despite clear demand for IEAs, dedicated datasets, formal frameworks, and evaluation metrics to support research in this area are absent. Existing dialogue datasets have short conversation lengths (DSTC2 averages 14.49 turns), making them incapable of capturing long-horizon elicitation strategies.

**Goal**: (1) Define IEA as a new conversational paradigm; (2) construct the first large-scale IEA dataset; (3) formalize the information elicitation problem; (4) design purpose-built evaluation metrics.

**Key Insight**: Data are collected from authentic human-human elicitation dialogues (oral histories, judicial hearings, academic interviews, investigative journalism), ensuring that the dataset reflects genuine elicitation behavior patterns.

**Core Idea**: Information elicitation is modeled as a POMDP, with entity novelty serving as a proxy reward signal; offline reinforcement learning (AWR) is used to fine-tune LLMs so that they learn to ask questions in the manner of human elicitors.

## Method

### Overall Architecture

YIELD comprises four core components: (1) a large-scale dataset — 2,281 human-human elicitation dialogues across four domains; (2) a POMDP formalization — modeling the elicitation process as sequential decision-making under partial observability; (3) offline reinforcement learning training — using AWR with an entity novelty reward; and (4) dedicated evaluation metrics — Conformity, Progression, and Turn-Length Ratio.

### Key Designs

1. **POMDP Formalization and State Representation**

    - **Function**: Formalizes the information elicitation process as an optimizable sequential decision-making problem.
    - **Mechanism**: The interviewee possesses unobservable hidden information (state $X_t$); the elicitor takes action $A_t$ (a natural-language question); the environment returns observation $O_{t+1}$ (the interviewee's response). Because the hidden state space is unbounded, the hidden-layer representation of a causal language model, $S_t = f_\theta(H_t^s)$, is used in place of a conventional belief state. The reward is defined as the number of named entities appearing for the first time in the interviewee's response: $R_{t+1} = |\mathcal{E}_{t+1} \setminus \mathcal{E}_{\leq t}|$.
    - **Design Motivation**: The POMDP framework naturally fits the elicitation scenario — an elicitor can never fully observe the interviewee's knowledge state and can only obtain information indirectly through questioning. The entity novelty reward, while simple, effectively quantifies the marginal information gain of each question.

2. **Offline Reinforcement Learning Training (AWR)**

    - **Function**: Trains the LLM to preferentially generate questions that elicit more novel information.
    - **Mechanism**: Advantage-Weighted Regression is employed: a linear value head estimates the state value $v_\psi(S_t)$, and the advantage function is computed to reweight training samples. Elicitor utterances with higher advantage (i.e., those that elicit more new information) receive higher training weights. LoRA is used for parameter-efficient fine-tuning, jointly optimizing a policy loss and a value loss: $\mathcal{L}(\theta, \psi) = \mathcal{L}_\pi(\theta) + \mathcal{L}_v(\psi)$.
    - **Design Motivation**: Unlike standard SFT, AWR accounts for the downstream impact of each utterance on the entire subsequent conversation, enabling the model to learn long-horizon elicitation strategies rather than turn-by-turn imitation.

3. **Dedicated Evaluation Metric Suite**

    - **Function**: Assesses IEA elicitation capability along multiple dimensions.
    - **Mechanism**: (1) **Conformity** — measures whether model outputs conform to the distributional patterns of real elicitors, using perplexity and response length; (2) **Progression** — measures whether the conversation continuously advances rather than stalling on the same topic, computed via a decayed cosine-distance window; (3) **Turn-Length Ratio** — the ratio of the interviewee's average response length to the elicitor's, reflecting the principle that effective elicitors ask concisely to draw out extended responses.
    - **Design Motivation**: Conventional metrics (BLEU, task success rate) fail to capture the essence of elicitation dialogue — conversational momentum, questioning efficiency, and stylistic conformity to authentic elicitation behavior.

### Loss & Training

The AWR-weighted policy loss is: $\mathcal{L}_\pi(\theta) = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \bar{w}_i \log \pi_\theta(A_i | S_i)$, where weights $\bar{w}_i$ are scaled by the advantage function. Dialogue data are segmented using a sliding window of 6 turns, with discount factor $\gamma=0.9$ and temperature $\alpha=0.25$. Training is conducted on 3 A6000 GPUs.

## Key Experimental Results

### Main Results

Conformity evaluation (perplexity and response length comparison):

| Model | Academic Perplexity | Judicial Perplexity | Academic Response Length | Real Response Length |
|-------|--------------------|--------------------|------------------------|---------------------|
| Llama-3.1-8B Prompt | 46.9 | 22.6 | 39.5 tokens | 16.9 tokens |
| Llama-3.1-8B SFT | 10.9 | 10.9 | 11.2 tokens | 16.9 tokens |
| Llama-3.1-8B ORL | 12.5 | 11.3 | 11.6 tokens | 16.9 tokens |

### Ablation Study

| Configuration | Key Finding | Remark |
|--------------|------------|--------|
| Prompt-only vs. SFT/ORL | Perplexity drops 3–4× | Fine-tuning substantially improves alignment with authentic elicitation behavior |
| SFT vs. ORL | SFT achieves slightly lower perplexity | ORL sacrifices per-token likelihood to optimize long-horizon strategy |
| DeepSeek-R1 Prompt | Response length 414–472 tokens | Verbose meta-reasoning of the reasoning model severely deviates from elicitation style |
| 3B vs. 8B models | Performance comparable | Indicates that YIELD data, rather than model scale, is the key factor |

### Key Findings

- Prompt-only approaches produce elicitor utterances far longer than those of real elicitors (39–53 vs. 17–39 tokens), and the prompts themselves are lengthy (540–648 tokens), making them highly inefficient.
- DeepSeek-R1's reasoning mode is entirely unsuitable for elicitation tasks — it generates excessively long meta-reasoning prefixes, which are only suppressed after fine-tuning.
- ORL-trained models are competitive with SFT on the Progression metric and exhibit response-length distributions closer to those of real elicitors.
- Human evaluation corroborates the automatic metrics: models trained on YIELD substantially outperform prompt-based methods in elicitation quality.

## Highlights & Insights

- **The introduction of the IEA concept is pioneering** — it explicitly defines "agent-initiated information elicitation" as a distinct conversational paradigm, in sharp contrast to the conventional "user asks, agent answers" model, and provides a unified framework encompassing academic interviews, judicial hearings, and investigative journalism.
- **The entity novelty reward is elegantly simple yet effective** — using NER-extracted new entity counts as a proxy for elicitation success avoids the subjectivity of defining "information value," while multiple constraints prevent reward hacking.
- **The dataset construction methodology is exemplary** — manually annotated from authentic human-human dialogues drawn from public-domain or CC-licensed sources, with an average of 171 turns per dialogue, far exceeding existing datasets (13–20 turns).

## Limitations & Future Work

- Evaluation is conducted solely in an offline setting; the actual elicitation effectiveness of IEAs in live user interactions remains untested.
- The entity novelty reward is too coarse to measure the "depth" or "relevance" of information; more nuanced reward design is a critical direction for future work.
- The dataset is entirely in English, and certain domains have limited data (only 129 dialogues for investigative journalism).
- The ethical boundaries of IEA behavior — specifically, the appropriate extent of agent-driven elicitation — warrant further discussion.

## Related Work & Insights

- **vs. MultiWOZ/SGD**: These datasets target user-driven task-completion dialogue, averaging 13–20 turns per conversation. YIELD targets agent-driven information elicitation, averaging 171 turns per conversation — fundamentally different in both scale and paradigm.
- **vs. LLM-as-Interviewer**: Existing work in this vein primarily focuses on LLMs simulating interviewers but lacks systematic datasets and formal frameworks. YIELD provides a complete research infrastructure spanning data, theory, and evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Defines the IEA paradigm; the dataset, formalization, and evaluation metrics are all first-of-their-kind.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparisons and human evaluation are thorough, though online interaction experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Concepts are clearly defined; the paper is rigorously structured with a complete logical chain from motivation to method to evaluation.
- **Value**: ⭐⭐⭐⭐ The pioneering dataset and framework offer long-term value to the community, though the application scope is relatively specialized.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)
- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)

<!-- RELATED:END -->
