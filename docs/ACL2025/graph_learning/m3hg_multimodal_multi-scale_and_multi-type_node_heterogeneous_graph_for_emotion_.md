---
title: >-
  [Paper Note] M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations
description: >-
  [ACL 2025][Graph Learning][Multimodal Emotion Cause Analysis] This paper proposes the M3HG model, which explicitly models emotional and causal contexts in conversations by constructing a multimodal multi-type node heterogeneous graph. It fuses semantic information at both intra-utterance and inter-utterance scales to achieve end-to-end extraction of emotion-cause triplets in multimodal conversations. Additionally, it constructs MECAD, the first Chinese multi-scenario MECTEC d…
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Multimodal Emotion Cause Analysis"
  - "Heterogeneous Graph Attention Network"
  - "Emotion-Cause Triplet Extraction"
  - "Multi-scale Semantic Fusion"
  - "Conversation Emotion Analysis"
date: 2026-05-08
content_hash: 3dbe2618238cb467
---

# M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations

**Conference**: ACL 2025  
**arXiv**: [2508.18740](https://arxiv.org/abs/2508.18740)  
**Code**: [https://github.com/redifinition/M3HG](https://github.com/redifinition/M3HG)  
**Area**: Graph Learning  
**Keywords**: Multimodal Emotion Cause Analysis, Heterogeneous Graph Attention Network, Emotion-Cause Triplet Extraction, Multi-scale Semantic Fusion, Conversation Emotion Analysis

## TL;DR

This paper proposes the M3HG model, which explicitly models emotional and causal contexts in conversations by constructing a multimodal multi-type node heterogeneous graph. It fuses semantic information at both intra-utterance and inter-utterance scales to achieve end-to-end extraction of emotion-cause triplets in multimodal conversations. Additionally, it constructs MECAD, the first Chinese multi-scenario MECTEC dataset.

## Background & Motivation

**Background**: Multimodal emotion cause triplet extraction in conversations (MECTEC) is a key task in social media analysis, aiming to simultaneously extract triplets $\mathcal{P}=\{(\boldsymbol{U}_j^e, \boldsymbol{U}_j^c, y_j^e)\}$ consisting of emotion utterances, cause utterances, and emotion categories from conversations containing text, audio, and video.

**Limitations of Prior Work**: 
   - Scarcity of datasets: Only one ECF dataset currently exists, which comes from a single source (only the *Friends* TV show) and severely lacks scenario diversity.
   - Existing methods do not explicitly model emotion and cause contexts, neglecting the fusion of semantic information at different scales (inter-utterance/intra-utterance).
   - Inability to effectively handle cases where cause utterances appear after emotion utterances.

**Key Challenge**: Emotion attribution theory indicates that the relationship between emotion and cause is revealed by specific contexts (e.g., emotional words in text, tone in audio, facial expressions in video), but existing methods treat multimodal fusion and context extraction as independent processes.

**Goal**: How to explicitly capture emotion and cause contexts in multimodal conversations, while effectively fusing semantic information at different scales to extract emotion-cause triplets.

**Key Insight**: Designing a heterogeneous graph that introduces multi-type nodes (emotion context nodes, cause context nodes, utterance supernodes, conversation supernodes) and various edge relations to perform multi-scale semantic fusion on the graph.

**Core Idea**: Unifying the modeling of emotion/cause contexts, multimodal features, and multi-scale semantic relations using a multi-type node heterogeneous graph.

## Method

### Overall Architecture

M3HG is an end-to-end model comprising four core components:
1. Unimodal feature extraction
2. Heterogeneous graph construction
3. Multi-scale semantic fusion
4. Emotion-cause classification

### Key Designs

1. **Unimodal feature extraction**: 

    - Text: SA-RoBERTa extracts $\boldsymbol{E}^t \in \mathbb{R}^{n \times d_t}$, then encodes local context using multi-head self-attention to obtain $\boldsymbol{H}^t$.
    - Audio: Wav2Vec2 extracts $\boldsymbol{E}^a$, which is encoded via GRU + LayerNorm + FFN: $\boldsymbol{H}^m = LN(\boldsymbol{E}^m + \boldsymbol{E}'^m + FFN(\boldsymbol{E}'^m))$
    - Video: DenseNet extracts $\boldsymbol{E}^v$, also encoded using GRU.
    - Multi-modalities are mapped to a unified dimension $d_h$ using linear layers.

2. **Heterogeneous graph construction**: 

    - **Four node types**: Emotion context node $N^e$, cause context node $N^c$, utterance supernode $SN^u=\{N^t, N^a, N^v\}$, conversation supernode $SN^d$.
    - **Five edge relations**: Same speaker edge $r_{ss}$, different speaker edge $r_{ds}$, global connection edge $r_{gc}$, emotion connection edge $r_{ec}$, cause connection edge $r_{cc}$.
    - Global connection edges bidirectionally connect each utterance supernode with the conversation supernode, **supporting scenarios where cause utterances appear after emotion utterances for the first time**.

3. **Multi-scale semantic fusion (HGAT)**: 

    - **Intra-utterance**: Fuses tri-modal semantic information into emotion/cause context nodes within a single utterance via the meta-path $\Phi_{intra}$. Attention calculation:
    $$\alpha_{ij}^\phi = \frac{\exp(\sigma(\boldsymbol{a}_\phi^T \cdot [\boldsymbol{H}'_i \| \boldsymbol{H}'_j]))}{\sum_{k \in \mathcal{N}_i^\phi} \exp(\sigma(\boldsymbol{a}_\phi^T \cdot [\boldsymbol{H}'_i \| \boldsymbol{H}'_k]))}$$
    - **Inter-utterance**: Propagates semantic information across different utterances via the meta-path $\Phi_{inter}$, achieving global information aggregation through the conversation supernode $SN^d$.
    - After fusion, node features are updated using a semantic attention mechanism + PFFN.

4. **Emotion-cause classification**: 

    - Emotion nodes $\boldsymbol{Z}_i^e$ are fed into an Emotion MLP to predict the emotion category $\hat{y}_i^e$.
    - Cause nodes $\boldsymbol{Z}_i^c$ are fed into a Cause MLP to predict the cause indicator $\hat{y}_i^c$.
    - For an utterance pair $(U_i, U_j)$, RBF kernel function is used to calculate the relative position encoding $RPE_{ij}$, which is concatenated and fed into an MLP to determine the causal relationship:
    $$\hat{y}_{ij}^{ec} = \sigma(MLP(\boldsymbol{Z}_j^e \| \boldsymbol{Z}_i^c \| RPE_{ij}))$$

### Loss & Training

- Focal Loss is used to address class imbalance: $$\mathcal{L}^\beta = -\frac{1}{N^\beta}\sum_{i=1}^{N^\beta}\alpha^\beta(1-\hat{y}_i^\beta)^\gamma \log(\hat{y}_i^\beta)$$
- Joint optimization of losses for the three tasks (emotion prediction, cause prediction, and emotion-cause pair prediction).

## Key Experimental Results

### Main Results

| Dataset | Method | Modality | 6 Avg F1 | 4 Avg F1 |
|--------|------|------|----------|----------|
| ECF | HiLo (SOTA) | T,A,V | 33.04 | 35.81 |
| ECF | GPT-4o (5-shots) | T | 29.13 | 30.30 |
| ECF | **M3HG (T)** | T | 37.46 | 39.95 |
| ECF | **M3HG (T,A,V)** | T,A,V | **40.07** | **41.96** |
| MECAD | SHARK (SOTA) | T | 27.58 | 29.99 |
| MECAD | GPT-4o (5-shots) | T | 27.16 | 28.42 |
| MECAD | **M3HG (T,A,V)** | T,A,V | **32.82** | **34.59** |

### Ablation Study

| Modality Combination | ECF 6 Avg | ECF 4 Avg | MECAD 6 Avg | MECAD 4 Avg |
|----------|-----------|-----------|-------------|-------------|
| T | 37.46 | 39.95 | 30.81 | 32.55 |
| T+A | 39.10 | 40.97 | 32.16 | 33.73 |
| T+V | 38.90 | 40.72 | 31.95 | 33.52 |
| T+A+V | 40.07 | 41.96 | 32.82 | 34.59 |

### Key Findings

- M3HG outperforms all baselines (including multimodal ones) using only the text modality, proving the effectiveness of the graph structure design.
- Compared with the SOTA method HiLo, M3HG improves the 6 Avg by 21.28% and the 4 Avg by 17.17% on ECF.
- On sample-scarce categories such as Disgust and Fear, M3HG improves upon GPT-4o by 31.36% and 58.46%, respectively.
- Every additional modality yields performance improvements, validating the necessity of multimodal fusion.
- The MECAD dataset contains 989 conversations and 10,516 utterances from 56 TV series, achieving a Fleiss's Kappa of 0.6932.

## Highlights & Insights

- **Exquisite Heterogeneous Graph Design**: The heterogeneous graph with four node types and five edge relations unifies three problems: multimodal fusion, context modeling, and emotion-cause association.
- **First to Handle "Cause-after-Effect" Scenarios**: Bidirectionally connecting each utterance to the conversation node via global connection edges breaks the constraint of forward-only dependencies.
- **Multi-scale Fusion**: Intra-utterance fusion captures multimodal emotion/cause cues within a single utterance, while inter-utterance fusion propagates conversation-level context. The two scales complement each other.
- **Dataset Contribution**: MECAD is the first Chinese multi-scenario MECTEC dataset, significantly enhancing data diversity in this field.

## Limitations & Future Work

- External knowledge (such as commonsense knowledge graphs) is not integrated, limiting the accuracy of emotion and cause prediction.
- The model cannot process excessively long conversations due to constraints on the language model's input length.
- Error propagation may occur during the multimodal fusion process, especially when information from different modalities is inconsistent.
- The MECAD dataset originates from TV dramas, which still exhibits gaps compared with real social media conversations.

## Related Work & Insights

- **Emotion Cause Analysis**: The evolution from text to multimodality: RECCON $\rightarrow$ ConvECPE $\rightarrow$ ECF $\rightarrow$ MECAD.
- Heterogeneous Graph Attention Networks (HGAT) (Wang et al., 2019)'s meta-path mechanism provides a flexible framework for information propagation among multi-type nodes.
- Focal Loss effectively mitigates the issue of class imbalance.
- Future work can consider incorporating LLMs for few-shot or zero-shot reasoning in emotion cause analysis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The heterogeneous graph design is novel; the four node types and five edge relations completely model multiple aspects of the MECTEC task.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive comparisons on two datasets, along with modality ablation and comparisons with GPT-4o.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and precise illustrations.)
- **Value**: ⭐⭐⭐⭐ (Dual contribution of the MECAD dataset and the M3HG model, advancing the field of multimodal emotion cause analysis.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs](multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs.md)
- [\[CVPR 2025\] DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition](../../CVPR2025/graph_learning/dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition.md)
- [\[ICLR 2026\] Multi-Scale Diffusion-Guided Graph Learning with Power-Smoothing Random Walk Contrast for Multi-View Clustering](../../ICLR2026/graph_learning/multi-scale_diffusion-guided_graph_learning_with_power-smoothing_random_walk_con.md)
- [\[NeurIPS 2025\] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](../../NeurIPS2025/graph_learning/heterogeneous_swarms_jointly_optimizing_model_roles_and_weights_for_multi-llm_sy.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)

</div>

<!-- RELATED:END -->
