---
title: >-
  [Paper Note] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment
description: >-
  [ICML2026][Information Retrieval & RAG][Knowledge Conflict] This paper introduces the REAL framework, which redefines knowledge conflicts in KI-VQA using "Reasoning-Pivots" (atomic nodes/edges in a reasoning chain that m…
tags:
  - "ICML2026"
  - "Information Retrieval & RAG"
  - "Knowledge Conflict"
  - "KI-VQA"
  - "Reasoning-Pivot"
  - "Contrastive Decoding"
  - "Multimodal RAG"
date: 2026-05-08
content_hash: 27d9ce1ccbeafdaa
---

# REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment

**Conference**: ICML2026  
**arXiv**: [2602.14065](https://arxiv.org/abs/2602.14065)  
**Code**: TBD  
**Area**: information_retrieval  
**Keywords**: Knowledge Conflict, KI-VQA, Reasoning-Pivot, Contrastive Decoding, Multimodal RAG

## TL;DR
This paper introduces the REAL framework, which redefines knowledge conflicts in KI-VQA using "Reasoning-Pivots" (atomic nodes/edges in a reasoning chain that must rely on external evidence for completion). By training a pivot-aware conflict discriminator via RPA-SFT and a training-free contrastive decoding strategy through RPGD, the method achieves improvements of +3.8%, +1.6%, and +3.6% on E-VQA, InfoSeek, and A-OKVQA, respectively.

## Background & Motivation

**Background**: Knowledge-Intensive VQA (KI-VQA) has become a mainstream configuration for MLLMs and Multimodal RAG—supplementing visual and parametric memory deficiencies through the retrieval of external passages like Wikipedia. Existing works primarily focus on retrieval precision, rerankers, and knowledge structural organization.

**Limitations of Prior Work**: Open-domain retrieval inevitably introduces noise and contradictory evidence, forming "knowledge conflicts" (e.g., an artist being cited as both Italian and Spanish). Current conflict resolution paradigms suffer from two main issues: (1) **Poor generalization of conflict detection**—semantic matching rules based on entities/keywords are fragile and cannot adapt to the massive external knowledge and complex evidence interactions in KI-VQA; (2) **Lack of in-model conflict constraints**—existing methods rely on external knowledge reorganization or contrastive prompt intervention, but the diverse manifestations of the same conflict type in KI-VQA lead to inconsistent resolution behaviors and unpredictable reasoning.

**Key Challenge**: The traditional definition of "entity mismatch = conflict" ignores the sequential and conditional nature of KI-VQA reasoning chains. In multi-hop reasoning $\{e_{img} \xrightarrow{p_1} e_2 \xrightarrow{p_2} \cdots \xrightarrow{p_n} e_n\}$, intermediate nodes $e_2,\ldots,e_n$ are inherently different from the initial visual entity $e_{img}$; furthermore, identical property types (e.g., location/nationality) can appear at different stages of the reasoning chain, causing keyword matching to misjudge them as equivalent.

**Goal**: (1) Reformulate what constitutes a "true conflict"; (2) Use a unified signal to simultaneously train a discriminator and guide decoding to resolve conflicts in a closed loop.

**Key Insight**: KI-VQA is decomposed into discrete reasoning chains, and contradictions are determined only at factual points bound to "Reasoning-Pivots"—differences in entities/keywords outside these pivots are treated as benign noise.

**Core Idea**: Conflict detection is first constrained to key nodes in the reasoning chain via Reasoning-Pivot extraction. This same pivot signal then drives SFT training and guides logit-level contrastive decoding.

## Method

### Overall Architecture
The REAL pipeline consists of three components: (1) **REAL-VQA Dataset**—A high-quality multi-hop conflict training set (4,149 training / 629 test) automatically constructed via Wikipedia + GPT-4o, with pivot-level annotations and 5 ground-truth passages per sample; (2) **RPA-SFT (Reasoning-Pivot Aware SFT)**—Using special tokens `<RPivot></RPivot>` to wrap reasoning pivots, training the model to extract pivots from questions/passages before performing conflict discrimination; (3) **RPGD (Reasoning-Pivot Guided Decoding)**—A training-free contrastive decoding strategy that constructs a "conflict-dominant" path via Patch Shuffle, uses the discriminator's pivot set for adaptive gating, and applies Gram-Schmidt projection to decouple conflict directions from standard logits. These three components form a "data $\rightarrow$ discriminator $\rightarrow$ decoding" loop.

### Key Designs

1.  **Reasoning-Pivot Formalization + REAL-VQA Data Construction**:
    - **Function**: Collects all indispensable nodes and edges in the KI-VQA reasoning chain $e_1 \xrightarrow{p_1} e_2 \xrightarrow{p_2} y$ into a pivot set $\mathcal{P}=\{e_1,p_1,e_2,p_2,y\}$, and strictly defines conflict as logically mutually exclusive assertions targeting the same pivot: $\mathcal{K}_{conflict}=\{u\in\mathcal{P}\mid\exists a_i,a_j\in\mathcal{I}_u, a_i\wedge a_j\rightarrow\bot\}$.
    - **Mechanism**: Dataset construction follows three principles: high multi-hop complexity (maximizing pivot breadth), common-property aggregation (increasing pivot density), and knowledge-deficit induction (filtering visually solvable samples). Conflicts are generated using a rewrite-based strategy: replacing the ground-truth pivot $p_{gt}$ with $p_{neg}$ and having GPT-4o rewrite the passage in the true Wikipedia context of $p_{neg}$ to ensure internal factual consistency while creating precise contradictions with visual evidence. Quality is guaranteed via vote-of-confidence filtering and manual verification.
    - **Design Motivation**: To address the fundamental issue that "entity/keyword mismatch $\neq$ true conflict" by determining contradictions only at key reasoning nodes and excluding noise like irrelevant positional information.

2.  **RPA-SFT: Dual-Mechanism Pivot-Aware Training**:
    - **Function**: Trains a discriminator capable of accurately extracting reasoning-pivots from questions and retrieved passages to judge conflicts, avoiding shortcut learning associated with simple binary labels.
    - **Mechanism**: (a) **Token-Level Pivot Perception**: Adding `<RPivot>` / `</RPivot>` special tokens to the vocabulary and explicitly wrapping all pivots in input and target sequences as semantic anchors in the embedding space; (b) **Multi-Stage Reasoning Training**: Structuring the target output into three steps: ① question pivot extraction $\rightarrow$ ② passage pivot extraction (guided by question pivots) $\rightarrow$ ③ binary conflict label output based on logical consistency within the pivot set.
    - **Design Motivation**: Converts the discrimination task into an explicit logical verification process, forcing the model to make decisions based on "assertion comparison for the same pivot" rather than relying on dataset artifacts.

3.  **RPGD: Training-Free Pivot-Guided Decoding**:
    - **Function**: Utilizes the pivot set $\mathcal{K}$ from the RPA-SFT discriminator during inference to targetedly suppress conflict directions in logits without damaging normal reasoning tokens.
    - **Mechanism**: A three-stage pipeline: (a) **Patch Shuffle**: Randomly shuffling visual patch embeddings to construct a "conflict-dominant" path $L_{conf}=M(x,\text{Shuffle}(v))$, which preserves part-level features and distribution magnitude while breaking object-level topology to force reliance on contradictory text; (b) **Adaptive Gating**: Initializing a gate matrix $\alpha\in\mathbb{R}^{B\times V}$ with a global baseline $\varepsilon$, and enhancing gate strength for pivot-related vocabulary indices $\mathcal{K}$ via $\alpha_{b,v}\leftarrow\varepsilon+\beta\cdot\sigma(\kappa L_{conf}(b,v))$; (c) **Gram-Schmidt Orthogonalization**: Calculating the projection coefficient $c=\langle L_{std},L_{conf}\rangle/(\|L_{conf}\|_2^2+\delta)$ to get the projected component $L_{proj}=c\cdot L_{conf}$. The final logit is $L_{final}=L_{std}-\alpha\odot L_{proj}$.
    - **Design Motivation**: Unlike direct logit subtraction which may harm shared reasonable structures, Gram-Schmidt strictly removes components geometrically aligned with the conflict; Patch Shuffle safely constructs conflict paths compared to masking; adaptive gating ensures suppression follows pivot signals.

### Loss & Training
RPA-SFT uses the standard SFT objective with a structured target sequence. The number of retrieved documents is $k=5$, aligned with baselines like EchoSight; trained on 8 H20 GPUs. RPGD is completely training-free.

## Key Experimental Results

### Main Results
KI-VQA accuracy main results (compared with SOTA, bold is best):

| Model | Method | InfoSeek (All) | E-VQA (All) | Gain vs. Prev. SOTA |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-8B | REAL (Ours) | **44.1** | **41.4** | +1.6 / +3.8 |
| InternVL3.5-8B | REAL (Ours) | 43.8 | 39.2 | Leading at scale |
| InternVL3-8B | VLM-PRF | 42.5 | 39.2 | Prev. SOTA |
| LLaMA3.1-8B | ReflectiVA | 40.2 | 35.5 | — |
| LLaVA-1.5-7B | EchoSight | 26.8 | 28.5 | — |

On A-OKVQA, REAL (LLaVA-1.5-7B) achieves MC=80.3 / DA=68.3, outperforming QACap (Claude 3.5), proving transferability to commonsense reasoning.

### Ablation Study
Conflict Discrimination (MCC / F1, key cross-domain results):

| Model | Method | REAL-VQA MCC | E-VQA MCC | ScienceQA MCC | MMKC MCC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-8B | Zero-shot | 19.0 | 85.4 | 64.5 | 23.4 |
| Qwen3-VL-8B | Few-shot CoT | 19.4 | 86.9 | 67.4 | 42.4 |
| Qwen3-VL-8B | Standard SFT | 89.4 | 82.6 | 87.0 | 38.2 |
| Qwen3-VL-8B | RPA-SFT (Ours) | **98.1** | **93.4** | **87.9** | **52.9** |

RPGD Component Ablation (Qwen3-VL-8B on E-VQA):

| Patch Shuffle | Adaptive Gating | Gram-Schmidt | Single-Hop | All |
| :--- | :--- | :--- | :--- | :--- |
| ✗ | ✗ | ✗ | 42.4 | 38.1 |
| ✓ | ✓ | ✓ | **45.5** | **41.4** |

### Key Findings
- **RPA-SFT outperforms standard SFT by +14.7 MCC on the unseen MMKC dataset**, indicating that pivot-level supervision brings true generalization rather than overfitting.
- **RPGD components are indispensable**: Removing Patch Shuffle, Adaptive Gating, or Gram-Schmidt leads to performance drops of 1.6-2.0%.
- **Cross-model transferability**: RPGD consistently brings +3~7 point improvements across different model scales (LLaVA-1.5 to Qwen3-VL) as a plug-in.

## Highlights & Insights
- **Paradigm Shift in Conflict Definition**: Replacing "entity mismatch" with "logical mutual exclusion on reasoning pivots" fundamentally resolves misjudgments caused by natural entity differences in multi-hop chains.
- **End-to-End Signal Reuse**: Using pivots as the same semantic entity across data construction, SFT targets, and decoding gates prevents signal mismatch between modules.
- **Patch Shuffle Logic**: Maintaining distribution magnitude while destroying topology is more effective at exposing conflict signals than information deletion (masking), without introducing distribution shifts.
- **Mathematical Framework**: Gram-Schmidt projection combined with adaptive gating provides a clean way to peel off conflict-aligned logit components while avoiding over-penalization.

## Limitations & Future Work
- **Data Dependency**: REAL-VQA relies on GPT-4o for pivot annotation; its quality is bounded by the teacher model's multi-hop capabilities.
- **Enumeration Assumption**: For open-ended QA requiring implicit reasoning without clear chains, the pivot set might collapse.
- **Conflict Types**: Currently focuses on RAG context-memory conflicts; intra-memory and pure image-text conflicts are not yet explicitly modeled.
- **Inference Overhead**: RPGD requires two forward passes, doubling the reasoning cost.

## Related Work & Insights
- **vs. ReflectiVA / VLM-PRF**: REAL shifts the workload to internal pivot discrimination and decoding without requiring retriever modifications.
- **vs. NoteMR / mKG-RAG**: Unlike knowledge graph indexing, REAL performs discrete determinations of "where the conflict lies," which is more robust against noise in multi-hop reasoning.
- **vs. Traditional Contrastive Decoding**: REAL refines the contrastive direction by projecting it specifically onto pivot tokens and using orthogonalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/information_retrieval/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](../../AAAI2026/information_retrieval/mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](../../ACL2026/information_retrieval/counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](../../ACL2026/information_retrieval/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)

</div>

<!-- RELATED:END -->
