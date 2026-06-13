---
title: >-
  [Paper Note] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition
description: >-
  [ICCV 2025][Multimodal VLM][Open-world activity recognition] This paper proposes ProbRes, a framework that leverages a probabilistic residual search strategy based on jump diffusion…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Open-world activity recognition"
  - "egocentric vision"
  - "jump diffusion"
  - "structured search"
  - "VLM"
date: 2026-05-08
content_hash: a0cefc58f63ea25f
---

# ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition

**Conference**: ICCV 2025
**arXiv**: [2504.03948](https://arxiv.org/abs/2504.03948)  
**Code**: To be released  
**Area**: Multimodal VLM
**Keywords**: Open-world activity recognition, egocentric vision, jump diffusion, structured search, VLM

## TL;DR

This paper proposes ProbRes, a framework that leverages a probabilistic residual search strategy based on jump diffusion, combined with ConceptNet commonsense priors and VLM likelihood estimation, to efficiently navigate large-scale search spaces in open-world egocentric activity recognition. ProbRes substantially reduces the number of VLM queries while improving recognition accuracy.

## Background & Motivation

Open-world egocentric activity recognition requires models to infer ongoing activities from a vast, unconstrained hypothesis space, which fundamentally differs from conventional closed-set classification. The main challenges include:

**Search space explosion**: Activities are composed of action–object combinations, whose cardinality grows exponentially with the degree of openness, making exhaustive enumeration computationally infeasible.

**Limitations of VLMs**: Although VLMs such as CLIP and LAVILA exhibit strong zero-shot generalization, they rely on exhaustive enumeration for inference, making them inefficient for large-scale open-world reasoning.

**Ambiguous openness definitions**: Existing work lacks a unified definition of "open world." Different factors (objects, actions, domains) independently contribute uncertainty, hampering fair method comparison and generalization evaluation.

To address this, the authors first propose a clear **openness hierarchy** (L0–L3):
- **L0**: All activity categories are predefined (conventional zero-shot)
- **L1**: Atomic concepts (actions/objects) are known, but their compositions are undefined
- **L2**: The domain is known (e.g., "cooking"), but the search space is unconstrained
- **L3**: No prior knowledge; the search space must be constructed from scratch

## Method

### Overall Architecture

ProbRes is an adaptive search framework based on jump diffusion. Its core idea is to efficiently navigate the text embedding space of a VLM guided by structured priors. The framework consists of three stages:

1. **Exploration**: Prior-driven sampling to discover candidate activities
2. **Exploitation**: VLM likelihood-guided fine-grained search
3. **Residual Refinement**: Concept decomposition and re-ranking

### Search Space Construction

**Prior space construction**: ConceptNet is used to estimate prior probabilities over action–object pairs. ConceptNet is modeled as a directed graph, and semantic scores are computed via the sum of edge weights along the shortest path with an exponential decay factor $\lambda^i$:

$$f(a_{\text{action}}, a_{\text{object}}) \leftarrow f(a_{\text{action}}, a_{\text{object}}) \cdot R(a_{\text{action}}, a_{\text{object}})$$

where $R(\cdot)$ is a relation adjustment weight that penalizes negative relations (e.g., NotCapableOf) and amplifies positive ones (e.g., UsedFor). The prior is normalized into a probability distribution.

**Search space ordering**: Activities are ordered by Euclidean distance $d(a_i, a_j) = \|\phi(a_i) - \phi(a_j)\|$ in the VLM embedding space $\phi(a)$, with an anchor $a_{\text{ref}}$ selected and activities ranked by distance, ensuring that semantically similar activities are adjacent in the search space.

### Adaptive Search: Jump Diffusion

**Exploration stage** uses prior-guided sampling:

$$P_{\text{explore}}(a) = \frac{\lambda P_{\text{prior}}(a) + (1-\lambda) \frac{1}{|\mathcal{S}|}}{\sum_{a'} \lambda P_{\text{prior}}(a') + (1-\lambda)}$$

$\lambda \in [0,1]$ controls the trade-off between prior-guided and uniform random sampling. $\lambda \approx 1$ corresponds to strong prior guidance; $\lambda \approx 0$ reduces to a random walk.

**Exploitation stage** transitions to likelihood-driven search:

$$P_{\text{guided}}(a) = \frac{P_{\text{prior}}(a) \cdot P_{\text{likelihood}}(v|a)}{\sum_{a'} P_{\text{prior}}(a') \cdot P_{\text{likelihood}}(v|a')}$$

Bayesian posterior updates progressively concentrate the search on high-likelihood regions.

### Concept Decomposition and Re-ranking

Top-$k$ candidates are drawn from the refined set $\mathcal{A}_{\text{refine}}$. Each activity is decomposed into action and object components, and alignment scores are computed separately:

$$S_{\text{final}}(a) = P_{\text{likelihood}}(v|a) + \lambda_a S_a + \lambda_o S_o$$

where $S_a = v^T \phi(a_{\text{action}})$ and $S_o = v^T \phi(a_{\text{object}})$. This hierarchical re-ranking assigns higher confidence to candidates with semantically consistent components.

### Loss & Training

ProbRes is an inference-time framework and involves no training loss. The core optimization objective is:

$$a^* = \arg\max_{a \in \mathcal{S}} [P_{\text{search}}(a) + \lambda_a S_a + \lambda_o S_o]$$

### Key Designs

- VLM backbones: EGOVLP and LAVILA
- Prior source: ConceptNet
- Search spaces for L2/L3 generated by Gemini 2.0 Flash
- $\lambda = 0.5$; $T = 3000$ (small datasets) / $1000$ (large datasets)
- $\lambda_a, \lambda_o \in [0.3, 0.7]$
- Inference time: ~2 seconds per video (RTX 3090)

## Key Experimental Results

### Main Results

| Method | GTEA Gaze VLM Calls | GTEA Gaze WUPS | GTEA Gaze+ VLM Calls | GTEA Gaze+ WUPS | EK100 VLM Calls | EK100 WUPS |
|------|------|------|------|------|------|------|
| ALGO+LAVILA | N/A | 49.42 | N/A | 53.38 | N/A | 34.47 |
| LAVILA | 380 | 51.31 | 405 | 53.27 | 29100 | 43.52 |
| LAVILA+ProbRes | **110** | **53.34** | **175** | **53.82** | **3000** | **43.55** |
| EGOVLP | 380 | 46.77 | 405 | 51.48 | 29100 | 39.84 |
| EGOVLP+ProbRes | **110** | **49.25** | **178** | **53.98** | **3000** | **40.53** |

**Key Findings**: ProbRes reduces VLM queries from 380 to 110 on GTEA Gaze (71% reduction) and from 29,100 to 3,000 on EK100 (90% reduction), while maintaining or improving accuracy.

### L2/L3 Openness Evaluation

| Setting | Method | VLM Calls | GTEA Activity WUPS | EK100 Activity WUPS |
|------|------|------|------|------|
| L2 | LAVILA | 37191 | 38.34 | 30.64 |
| L2 | LAVILA+ProbRes | **1500** | **43.28** | **31.92** |
| L3 | LAVILA | 195714 | 46.06 | 31.06 |
| L3 | LAVILA+ProbRes | **5000** | **47.71** | **32.58** |

Under the most open L3 setting, VLM queries decrease from 195,714 to 5,000, with a performance gain of 1.65 WUPS.

### Ablation Study

| Ablation | WUPS Change | Notes |
|------|------|------|
| Remove ConceptNet prior | Significant drop | Largest impact under L3; unstructured search increases uncertainty |
| Remove re-ranking | Drop in Exact Match | Re-ranking is critical for distinguishing semantically similar but incorrect predictions |
| $\lambda=0$ (likelihood only) | Performance drop | Premature convergence to high-confidence but incorrect activities |
| $\lambda \approx 0.6$ (optimal) | Best | Optimal balance between exploration and exploitation |

## Highlights & Insights

1. **Dual benefit of efficiency and accuracy**: The core contribution of ProbRes is demonstrating that structured search can replace exhaustive enumeration, achieving superior performance with only 10% of the VLM queries.
2. **Openness hierarchy (L0–L3)**: A systematic taxonomy of open-world recognition challenges that provides a unified evaluation framework for the field.
3. **Critical role of prior knowledge**: Experiments show that ConceptNet priors are far more efficient than random search when the search space is unconstrained.
4. **Effectiveness of concept decomposition**: Separately evaluating action and object components effectively mitigates noise in VLM phrase-level embeddings.

## Limitations & Future Work

1. Dependence on VLM pretraining biases may lead to suboptimal search trajectories.
2. VLM text embeddings lack semantic organization, causing irrelevant jumps during search.
3. LLM-generated search spaces for L2/L3 settings require careful filtering, as they may be overly broad or noisy.
4. ConceptNet's coverage is limited and may be insufficient for highly specialized domains.

## Related Work & Insights

- **Zero-shot activity recognition**: KGL (2022) employs ConceptNet knowledge graphs; ALGO (2024) iteratively refines predictions via affordance priors.
- **VLM backbones**: EGOVLP, LAVILA, EgoVLPv2, and others focus on egocentric video–text embeddings.
- **Open-world detection**: Open World DETR, UMB, and related works handle novel-class bias in object detection.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] DisenQ: Disentangling Q-Former for Activity-Biometrics](disenq_disentangling_q-former_for_activity-biometrics.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[NeurIPS 2025\] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/openhoi_open-world_hand-object_interaction_synthesis_with_multimodal_large_langu.md)

</div>

<!-- RELATED:END -->
