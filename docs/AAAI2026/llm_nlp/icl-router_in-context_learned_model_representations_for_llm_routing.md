---
title: >-
  [Paper Note] ICL-Router: In-Context Learned Model Representations for LLM Routing
description: >-
  [AAAI 2026][LLM/NLP][model routing] This paper proposes ICL-Router, a two-stage training framework (query reconstruction + ICL model routing) that encodes LLM capability profiles as in-context vectors…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "model routing"
  - "in-context vectors"
  - "capability profiling"
  - "scalability"
  - "LLM collaboration"
date: 2026-05-08
content_hash: 490e6b63dd58e4f6
---

# ICL-Router: In-Context Learned Model Representations for LLM Routing

**Conference**: AAAI 2026
**arXiv**: [2510.09719](https://arxiv.org/abs/2510.09719)  
**Code**: [GitHub](https://github.com/lalalamdbf/ICL-Router)  
**Area**: LLM Routing / Model Selection
**Keywords**: model routing, in-context vectors, capability profiling, scalability, LLM collaboration

## TL;DR

This paper proposes ICL-Router, a two-stage training framework (query reconstruction + ICL model routing) that encodes LLM capability profiles as in-context vectors, enabling scalable dynamic model routing. New models can be incorporated without retraining the router, achieving state-of-the-art performance on both in-distribution and out-of-distribution tasks.

## Background & Motivation

**Background**: Different LLMs excel at different tasks (e.g., DeepSeek at reasoning, Qwen at coding). Model routing dynamically assigns queries to the most suitable model to maximize overall performance, and has become an important research direction for multi-model collaboration.

**Limitations of Prior Work**:
   - **RouterDC**: Trains query and model embeddings via dual contrastive learning, but assumes a fixed model pool; adding new models requires retraining the router.
   - **EmbedLLM**: Trains the router via an encoder-decoder framework; similarly requires retraining to accommodate new models.
   - **MODEL-SAT**: Describes models using manually designed "capability instructions," avoiding retraining but requiring human-crafted instructions per benchmark and relying on prior knowledge.
   - Core issue: Model representations are overly simplistic (fixed embeddings or handcrafted descriptions) and lack scalability.

**Key Challenge**: LLMs are released at a rapid pace, so routing methods must incorporate new models at low cost; yet accurate capability representations require large-scale evaluation—creating an efficiency–accuracy tension.

**Goal**: Develop a scalable routing framework that can incorporate new models without retraining the router.

**Key Insight**: Represent model capability profiles as in-context vectors—a model's performance (correct/incorrect) across diverse queries can serve as in-context demonstrations, condensed into compact vectors for the router, rather than training dedicated embeddings per model.

**Core Idea**: Two-stage training—first train a projector and router to understand vector representations of queries (via a reconstruction task to align the semantic space), then use each model's performance on a query set (query vectors + correct/incorrect labels) as in-context vectors to train the router to predict model–query compatibility. A new model only needs to be evaluated on a small query set to obtain its capability profile.

## Method

### Overall Architecture

ICL-Router consists of three components:
1. **Embedding model** $f_{emb}$: Encodes queries into embedding vectors.
2. **Projector** $f_{proj}$: Aligns the embedding dimension with the router's input dimension.
3. **LLM Router**: Receives the query vector and model capability profile vectors, and predicts the best model.

Training proceeds in two stages: **Query Reconstruction** (semantic space alignment) → **ICL Model Routing** (routing decision learning).

### Key Designs

**Module 1: Query Reconstruction Training (Stage 1)**

- **Function**: Given a query $q_n$, the embedding model and projector produce vector $v_n$; the router is trained to reconstruct the original query text from $v_n$.
- **Mechanism**:
    - $e_n = f_{emb}(q_n) \in \mathbb{R}^{d_{Emb}}$
    - $v_n = f_{proj}(e_n) \in \mathbb{R}^{d_{Router}}$
    - Autoregressive reconstruction: minimize $\mathcal{L}_{rec} = -\frac{1}{NT_n}\sum_{n}\sum_{t}\log P(q_n^{(t)}|q_n^{(<t)}, v_n)$
- **Design Motivation**: The reconstruction task forces the projector's output vectors to retain complete query semantic information and simultaneously trains the router to interpret these vectors—a prerequisite for accurate routing, since the router cannot make sound routing decisions without understanding query semantics.

**Module 2: ICL Model Routing Training (Stage 2)**

- **Function**: For each candidate LLM $\mathcal{M}_t$, a capability profile $\mathbf{P}_t = ((v_1,c_1), ..., (v_K,c_K))$ is constructed, where $v_k$ is the query vector and $c_k$ indicates whether the model answered correctly ('Yes'/'No'); the router is trained to predict model–query compatibility accordingly.
- **Mechanism**:
    - The query set $\mathscr{Q}$ consists of high-difficulty queries (answered correctly by only a few LLMs) to maximize discriminability.
    - Joint training of projector and router with cross-entropy loss: $\mathcal{L}_{ce} = -\frac{1}{TN}\sum_t\sum_n \log P(y_{t,n}|(\mathbf{P}_t, q_n))$
    - Capability profiles are fed to the router as in-context vectors (rather than thousands of raw query texts), substantially compressing context length.
- **Design Motivation**:
    - Traditional methods learn a fixed embedding per model, requiring retraining upon adding new models; the ICL approach only requires evaluating a new model on the query set to obtain a plug-and-play capability profile.
    - Selecting high-difficulty queries ensures that capability profiles are sufficiently discriminative—if all models answer a query correctly, it carries no routing value.

**Module 3: Inference and New Model Integration**

- **Function**: At inference time, the router outputs the probability of correctly answering the new query for each candidate model and selects the one with the highest probability.
- **Mechanism**: $\mathcal{M}^* = \arg\max_t p(\text{'Yes'}|\mathcal{M}_t, q')$
- **New Model Integration**: Only requires evaluating the new model $\mathcal{M}_{T+1}$ on query set $\mathscr{Q}$ to obtain $\mathbf{P}_{T+1}$, which is then directly fed to the router—zero retraining required.
- **Design Motivation**: Capability profiles are decoupled from the router—the router learns "how to make decisions based on capability profiles" rather than "embeddings of specific models."

### Loss & Training

- Stage 1: Autoregressive reconstruction loss $\mathcal{L}_{rec}$
- Stage 2: Standard cross-entropy $\mathcal{L}_{ce}$
- Both stages jointly train projector parameters $\theta_{proj}$ and router parameters $\theta_{router}$

## Key Experimental Results

### Main Results

8 candidate LLMs (7–9B scale), 5 in-distribution (ID) + 5 out-of-distribution (OOD) benchmarks:

**In-Distribution Tasks (ID)**:

| Method | OlympiadBench | BBH | LogicBench | MMLUPro | MBPP | Avg |
|--------|---------------|-----|------------|---------|------|-----|
| Best Single Model | 74.26 | 75.62 | 78.03 | 58.84 | 79.21 | 73.19 |
| RouterDC | 73.56 | 73.49 | 77.24 | 58.20 | 79.11 | 72.32 |
| EmbedLLM | 71.45 | 79.02 | 78.92 | 64.06 | 77.34 | 74.16 |
| MODEL-SAT | 73.02 | 71.14 | 74.80 | 63.61 | 76.00 | 71.71 |
| **ICL-Router** | **74.16** | **80.52** | **79.03** | **67.53** | **80.53** | **76.30** |

ICL-Router achieves an average of 76.30, surpassing the best single model by 3.11 points, EmbedLLM by 2.14 points, and RouterDC by 3.98 points.

**Out-of-Distribution Tasks (OOD)**: ICL-Router also achieves state-of-the-art performance, demonstrating strong generalization.

### Ablation Study

- Necessity of the query reconstruction stage: removing Stage 1 prevents the router from understanding vector semantics, resulting in a significant performance drop.
- Number of in-context demonstrations: performance steadily improves and converges as query set size $|\mathscr{Q}|$ increases.
- Model scalability: ICL-Router's performance continues to improve as the candidate model pool grows, whereas MODEL-SAT's improvement plateaus (Figures 2–3).

### Key Findings

- Even with a small candidate model pool (8 models at 7–9B scale), the router already outperforms any single model by 7.2 points.
- The selection of high-difficulty queries is critical to routing quality—easy queries lack discriminability.
- In-context vector representations are more effective than both fixed embeddings and handcrafted instructions.
- Strong OOD generalization: a router trained on ID benchmarks can still make reasonable routing decisions on unseen benchmarks.

## Highlights & Insights

- **Scalability is the key contribution**: new models are integrated without any retraining, which is highly practical given the rapid pace of LLM development.
- The two-stage training design is elegant: query reconstruction ensures semantic space alignment, while ICL routing leverages the aligned vectors for capability matching.
- The in-context vector concept draws on the core idea of ICL (learning without training) and applies it to model representation, offering a novel perspective.
- The high-difficulty query selection strategy—retaining only queries that only a few models can answer correctly—maximizes the information content of capability profiles.

## Limitations & Future Work

- The selection and scale of query set $\mathscr{Q}$ substantially affect performance, but optimal query selection strategies remain underexplored.
- Candidate models cover only the 7–9B scale; performance on larger models (70B+) or cross-scale routing is unknown.
- The router itself is an LLM with non-negligible inference cost, which may become a bottleneck in latency-sensitive settings.
- Capability profiles are static (evaluated once on a fixed query set); if a model undergoes subsequent fine-tuning, re-evaluation is required.
- Multi-criteria routing (balancing performance, latency, and cost) is a promising direction for future work.

## Related Work & Insights

- **RouterDC**: Dual contrastive learning routing, but with a fixed model pool.
- **EmbedLLM**: Encoder-decoder routing; similarly not scalable.
- **MODEL-SAT**: Routing based on handcrafted capability instructions; scalable but requires manual design.
- **Vector-ICL** (Zhuang 2024): The source of the in-context vector concept, projecting continuous data into the LLM embedding space.
- **GraphRouter**: Focuses on balancing performance and computational cost, complementing this work's focus on pure performance maximization.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The method design is clear (two-stage training + in-context capability profiles), scalability addresses a practical pain point, and experiments are comprehensive (10 benchmarks, covering both ID and OOD settings). One point deducted for insufficient analysis of the query selection strategy and inadequate discussion of the router's own inference overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] In-Context Routing (ICR): Train Once, Use Everywhere via Attention-Level Implicit ICL](../../ICML2026/llm_nlp/train_once_reuse_everywhere_generalizable_implicit_in-context_learning_by_routin.md)
- [\[AAAI 2026\] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](an_invariant_latent_space_perspective_on_language_model_inve.md)

</div>

<!-- RELATED:END -->
