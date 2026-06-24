---
title: >-
  [Paper Note] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment
description: >-
  [ACL 2025][Medical LLM][medical dialogue system] This paper proposes MedRef, a medical dialogue system that integrates a knowledge refinement mechanism and a dynamic prompt adjustment strategy. It filters irrelevant knowledge graph triplets using latent variables, conducts joint entity-action prediction, and dynamically builds system prompts via a triplet filter and an exemplar selector, achieving SOTA performance on both the MedDG and KaMed benchmarks.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "medical dialogue system"
  - "knowledge refining"
  - "dynamic prompt"
  - "entity prediction"
  - "knowledge graph"
date: 2026-05-08
content_hash: c1c4f2225bacb60c
---

# Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment

**Conference**: ACL 2025  
**arXiv**: [2506.10877](https://arxiv.org/abs/2506.10877)  
**Code**: [GitHub](https://github.com/simon-p-j-r/MedReF)  
**Area**: Medical NLP  
**Keywords**: medical dialogue system, knowledge refining, dynamic prompt, entity prediction, knowledge graph

## TL;DR

This paper proposes MedRef, a medical dialogue system that integrates a knowledge refinement mechanism and a dynamic prompt adjustment strategy. It filters irrelevant knowledge graph triplets using latent variables, conducts joint entity-action prediction, and dynamically builds system prompts via a triplet filter and an exemplar selector, achieving SOTA performance on both the MedDG and KaMed benchmarks.

## Background & Motivation

**Background**: Medical Dialogue Systems (MDS) aim to support multi-turn, context-aware conversations between doctors and patients. These systems need to track patients' evolving health conditions and generate accurate responses utilizing medical domain knowledge. Existing methods often retrieve relevant entities from Medical Knowledge Graphs (MedKG) to enhance response generation.

**Limitations of Prior Work**: 
   - Retrieval-Augmented Generation (RAG) approaches often introduce irrelevant knowledge, which conversely degrades the quality of generated responses.
   - While Large Language Models (LLMs) improve fluency, they are highly sensitive to prompt structure and content.
   - Existing prompt designs lack the capability to dynamically adjust based on real-time patient information.

**Key Challenge**: Retrieved entities from the knowledge graph suffer from substantial noise, and a single static prompt template struggles to adapt to the highly diverse patient conditions.

**Goal**: (1) Refine the retrieved knowledge to provide more accurate response guidance; (2) dynamically adjust the system prompt to adapt to specific patient circumstances.

**Key Insight**: Incorporating latent variable modeling for knowledge refinement (filtering out irrelevant triplets) combined with joint entity-action prediction, followed by utilizing a Triplet Filter and a Demo Selector to dynamically construct multi-component prompts.

**Core Idea**: Utilizing VAE-style latent variables to refine retrieval results from medical knowledge graphs, and subsequently dynamically adjusting the prompt's knowledge triplets and dialogue exemplars to elevate entity accuracy and generation quality in medical dialogues.

## Method

### Overall Architecture

MedRef consists of three stages: (1) encoding dialogue history and retrieving a relevant entity subgraph from MedKG; (2) knowledge refinement + joint entity-action prediction; (3) dynamic prompt adjustment + LLM response generation.

### Key Designs

1. **Input Representation**: MedBERT is utilized as the encoder backbone to encode patient utterances and physician responses, yielding a context representation $e_{\bar{c}_t}$. A 1-hop subgraph $G_{\bar{x}_t}^0$ of historical entities retrieved from MedKG is encoded using a Graph Attention Network (GAT) to obtain the subgraph representation $e_{\bar{x}_t}^{G_0}$. Concurrently, the dialogue action representation $e_{\bar{a}_t}$ is encoded.

2. **Knowledge Refining Mechanism (KRM)**: A latent variable $z_t$ is introduced to model the prior distribution $p_\theta(z_t|\bar{c}_t, G_{\bar{x}_t}^0) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$ and the posterior distribution $q_\phi(z_t|\bar{c}_t, G_{\bar{x}_t}^0, x_t) = \mathcal{N}(\mu_\phi, \Sigma_\phi)$ (where the posterior utilizes information from the ground-truth target entities $x_t$). $z_t$ is sampled and passed through a decoder, which is then connected to the raw entity embeddings via a residual connection: $e_{\bar{x}_t}^G = f_{dec}(z_t) + e_{\bar{x}_t}^{G_0}$, thereby filtering out irrelevant knowledge while retaining entity information highly relevant to the current dialogue context.

3. **Joint Entity-Action Prediction**: A cross-attention module $f_{ca}$ is employed to model the interaction among context, refined entities, and historical actions. After fusion via a GRU, linear layers with sigmoid functions are applied to predict target entities $\hat{x}_t = \sigma(W_x \tilde{e}_{\bar{x}_t}^G + b_x)$ and target actions $\hat{a}_t = \sigma(W_a \tilde{e}_{\bar{a}_t} + b_a)$. The strong correspondence between entities and actions (e.g., "symptom" corresponding to "symptom inquiry", "disease" to "diagnosis") justifies this joint prediction design.

4. **Triplet Filter**: Transforms the retrieved 1-hop subgraph into a set of triplets, calculates entity frequencies, and iteratively filters them utilizing a threshold $\tau$ (incremented starting from 1): $Tri_{\bar{x}_t}^\tau = \{(e_{head}, r, e_{tail}) | \min(\#e_{head}, \#e_{tail}) \geq \tau\}$, until the count of triplets does not exceed a predefined maximum value $M=25$.

5. **Demo Selector**: Selects the most relevant exemplar dialogues through a three-step alignment: (a) Entity alignment—matching training set dialogues based on first-turn utterance entities; (b) Similarity alignment—encoding and selecting the nearest dialogues via cosine similarity; (c) Span alignment—extracting focal snippets using a sliding window $\xi=2$ as the final exemplars.

6. **Dynamic Prompt Structure**: $\mathcal{P} = [\mathcal{I}; \mathcal{H}; \mathcal{K}; \mathcal{E}]$, which comprises task instructions, history details (dialogue context + entities + actions), evidence details (predicted entities/actions + filtered triplets), and related exemplars.

### Loss & Training

Two-stage training:

Stage 1 (Entity-Action Prediction): $\mathcal{L} = \lambda_x \mathcal{L}_x + \lambda_a \mathcal{L}_a + \lambda_{kl} \mathcal{L}_{kl}$, where $\lambda_x=1, \lambda_a=0.05, \lambda_{kl}=0.05$. $\mathcal{L}_x$ and $\mathcal{L}_a$ are BCE losses, and $\mathcal{L}_{kl}$ represents the KL divergence between the prior and posterior distributions.

Stage 2 (Response Generation): The prediction module is frozen, and ChatGLM3-6B is fine-tuned using LoRA (rank=8, $\alpha$=32) to maximize the response likelihood $\mathcal{L}_{gen} = -\sum_t \log \sum_k p_{gen}(r_{t_k}|r_{t_{<k}}, \mathcal{P})$.

## Key Experimental Results

### Main Results

| Method | MedDG B-1 | B-4 | E-F1 | R-1 | KaMed B-1 | B-4 | E-F1 | R-1 |
|------|-----------|-----|------|-----|-----------|-----|------|-----|
| DFMed | 41.74 | 22.48 | 21.54 | 28.90 | 39.59 | 20.30 | 21.33 | 27.67 |
| GPT-4o | 42.19 | 23.32 | 13.15 | 13.99 | 41.88 | 23.34 | 13.86 | 13.94 |
| ChatGLM3-6B | 33.16 | 17.97 | 17.43 | 29.27 | 32.03 | 16.68 | 20.56 | 28.02 |
| **MedRef** | **43.51** | **23.04** | **22.70** | **30.07** | **40.47** | **21.28** | **21.96** | **28.14** |

MedRef achieves comprehensive superiority on MedDG: outperforming GPT-4o by **16.08%** on ROUGE-1 and by **11.05%** on Entity-F1.

### Ablation Study

| Variant | MedDG B-1 | E-F1 | R-1 | KaMed B-1 | E-F1 | R-1 |
|------|-----------|------|-----|-----------|------|-----|
| MedRef (Full) | 43.51 | 22.70 | 30.07 | 40.47 | 21.96 | 28.14 |
| w/o KRM | 42.58 | 21.94 | 29.88 | 40.29 | 21.51 | 27.95 |
| w/o Demo | 41.80 | 21.84 | 29.69 | 39.07 | 20.09 | 27.35 |
| w/o Kg | 41.76 | 21.58 | 29.86 | 39.82 | 20.55 | 28.09 |
| E-A&Cxt only | 41.63 | 21.30 | 28.68 | 39.30 | 20.81 | 26.72 |
| Cxt only | 33.16 | 17.43 | 29.27 | 32.03 | 20.56 | 28.02 |

### Human Evaluation

| Method | Fluency (FLU) | Knowledge Accuracy (KC) | Overall Quality (OQ) |
|------|------------|---------------|-------------|
| Ground-truth | 3.70 | 3.75 | 3.95 |
| DFMed | 3.42 | 3.57 | 3.65 |
| E-A&Cxt only | 2.91 | 3.05 | 3.14 |
| **MedRef** | **3.55** | **3.68** | **3.79** |

### Key Findings

- **The Knowledge Refinement Mechanism (KRM) is the most critical component**—its removal leads to the most significant drop across all evaluation metrics.
- **Blindly increasing the number of knowledge triplets is detrimental**: Weak Kg (randomly selecting triplets without filtering) performs worse than the full MedRef.
- **Randomly selecting exemplar dialogues (Weak Demo) is also harmful**, indicating that the target-driven alignment process of the Demo Selector is highly essential.
- Though GPT-4o achieves a high BLEU score, its Entity-F1 is quite low (13.15 vs. 22.70), as generating verbose, QA-style responses leads to poor entity accuracy.
- The human evaluation scores of MedRef are closest to the Ground-truth.

## Highlights & Insights

- **VAE-style knowledge refinement** serves as the core innovation. Guiding the prior with the posterior distribution to learn to filter out irrelevant retrieved knowledge is more principled than simple attention-based filtering.
- The **multi-component dynamic prompt** design offers valuable insights: modularizing different types of information (instructions, history, evidence, exemplars) allows each module to be optimized independently.
- The **frequency iterative filtering of the Triplet Filter** is simple yet effective—high-frequency entities tend to be more central, naturally screened out via an increasing frequency threshold.
- The **three-step alignment of the Demo Selector** (entity $\rightarrow$ similarity $\rightarrow$ span) ensures semantic relevance while keeping the prompt length under control.

## Limitations & Future Work

- The BLEU score of MedRef on KaMed is slightly lower than those of HuatuoGPT-II and GPT-4o. The authors attribute this to the high complexity of KaMed spanning over 100 departments, which also indicates the limitations of this method in dealing with ultra-large-scale knowledge bases.
- The system relies on CMeKG (Chinese Medical Knowledge Graph); migrating to English scenarios requires replacing the underlying knowledge source.
- The two-stage training pipeline is complex. Training the prediction and generation modules separately may not represent the optimal end-to-end framework.
- The Demo Selector requires pre-indexing dialogue instances in the training set; retrieval efficiency during online inference is not discussed.
- Generalization capabilities on open-domain medical questions are not evaluated.

## Related Work & Insights

- **DFMed** (Xu et al., 2023) is the primary baseline for comparison; MedRef incorporates knowledge refinement and dynamic prompts on top of it.
- **VRBot** (Li et al., 2021) models patient states and physician actions, whereas MedRef further refines retrievals from the knowledge graph.
- The idea in **MedPIR** (Zhao et al., 2022) of recalling vital information as prefixes echoes the prompt design philosophy of MedRef.
- Insight: In knowledge-graph-augmented generation, "post-retrieval refinement" is far more critical than "direct retrieval."

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The VAE modeling for knowledge refinement and the three-step demo selection offer decent originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Highly thorough evaluation over two datasets, exhaustive ablations, human evaluations, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear layout, rigorous mathematical formulation, and convincing case analysis.
- **Value**: ⭐⭐⭐⭐ — Holds practical value for medical dialogue systems; source code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2025\] Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine](pattern_recognition_or_medical_knowledge_the_problem_with_multiple-choice_questi.md)
- [\[ACL 2025\] Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice](are_llms_effective_psychological_assessors_leveraging_adaptive_rag_for_interpret.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)

</div>

<!-- RELATED:END -->
