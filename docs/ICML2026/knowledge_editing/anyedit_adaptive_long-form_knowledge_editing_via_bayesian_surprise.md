---
title: >-
  [Paper Note] AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise
description: >-
  [ICML 2026][Knowledge Editing][Bayesian Surprise] AnyEdit++ utilizes token-level Bayesian Surprise to identify semantic transition points in long-form text…
tags:
  - "ICML 2026"
  - "Knowledge Editing"
  - "Bayesian Surprise"
  - "Adaptive Chunking"
  - "Long-Form Knowledge Editing"
  - "Structural Independence"
  - "Causal Locality"
date: 2026-05-08
content_hash: 42762f8f0635c612
---

# AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise

**Conference**: ICML 2026  
**arXiv**: [2606.01053](https://arxiv.org/abs/2606.01053)  
**Code**: The paper claims GitHub availability, but the specific URL is not in the local cache  
**Area**: Knowledge Editing / Long-Form Knowledge Editing  
**Keywords**: Bayesian Surprise, Adaptive Chunking, Long-Form Knowledge Editing, Structural Independence, Causal Locality  

## TL;DR
AnyEdit++ utilizes token-level Bayesian Surprise to identify semantic transition points in long-form text, replacing the fixed-window slicing of AnyEdit with structure-aware Bayes-Chunk. It consistently improves BLEU and BERT Score across long-form knowledge editing tasks including mathematics, code, news, and poetry.

## Background & Motivation
**Background**: Knowledge editing aims to incorporate specific facts or knowledge into model parameters without retraining the entire large model, while minimizing damage to unrelated knowledge. Locate-and-edit methods like ROME, MEMIT, and AlphaEdit typically localize edits to the FFN output matrices of key layers. By optimizing a local perturbation or target value, the model is conditioned to generate a new object when encountering a specific subject/relation.

**Limitations of Prior Work**: This paradigm is natural for triplet facts but encounters capacity bottlenecks with long-form knowledge such as mathematical derivations, code snippets, news narratives, or poetry. Long text is not a single point-fact but a sequence of semantic units with internal dependencies; compressing an entire segment into a single perturbation vector often leads to generation collapse, broken logic chains, or partial memorization.

**Key Challenge**: AnyEdit previously addressed length limits by decomposing long-form editing into autoregressive multi-segment edits, using multiple anchor keys and perturbations to update weights. However, AnyEdit uses fixed-window slicing, where boundaries are agnostic to semantic structure. This may split function definitions, mathematical conditions, or narrative turns, resulting in anchor keys that are semantically ambiguous or highly correlated with adjacent segments, causing interference during weight updates.

**Goal**: This work does not aim to redesign the entire knowledge editing algorithm but rather addresses two specific questions in long-form editing: first, where should long text be partitioned to ensure segment independence; second, where should control signals be injected to influence subsequent generation most effectively.

**Key Insight**: The paper observes that a language model's internal belief state does not move smoothly during inference; expectations for the next token change significantly at new arguments, events, code structure shifts, or reasoning jumps. Bayesian Surprise quantifies this "belief revision" intensity, making high surprise points natural semantic boundaries.

**Core Idea**: Replace fixed window lengths with the model's own token-level surprise values to slice long text into semantically coherent segments. By placing edit perturbations on the token immediately preceding high-surprise segments, the system reduces cross-segment crosstalk and enhances local control.

## Method
The architecture of AnyEdit++ is restrained: it retains the autoregressive editing and MEMIT-style closed-form weight updates of AnyEdit, only replacing the "segmentation" and "anchor selection" components. This design functions as a plug-and-play module rather than a completely new editor.

### Overall Architecture
The input is long-form knowledge (e.g., a math CoT, code implementation, or narrative). The model first calculates the surprisal for each token in the original sequence to generate an information density curve. Bayes-Chunk then selects local peaks from this curve as boundaries to partition the text into semantic units. For the $j$-th chunk, the system uses the hidden state of the token preceding the boundary as an anchor to optimize a local perturbation $\delta_j$ in the target layer, encouraging the model to generate the current chunk following that anchor. Once all chunk key-value pairs are collected, they are written into the target weight matrix via a single multi-edit closed-form update using MEMIT/AnyEdit.

Specifically, standard locate-and-edit methods construct a key $k$ and target value $v^*$, updating the FFN output matrix $W_{out}$ such that $W_{out}k \approx v^*$. AnyEdit extends this to multiple segments where the $t$-th segment has its own anchor key $k_t$ and perturbation $\delta_t$, resulting in an edit dataset $D_{edit}=\{(k_t,v_t)\}_{t=1}^{M}$. AnyEdit++ maintains this solver but derives $k_t$ from semantic boundaries instead of fixed intervals.

### Key Designs
1. **Bayes-Chunk Adaptive Semantic Segmentation**:
    - **Function**: Selects partition boundaries based on token-level Bayesian Surprise to avoid fragmenting semantic units.
    - **Mechanism**: When processing prefix $y_{<t}$, the model maintains a prior belief distribution $\pi_t$, updated to $\pi_{t+1}$ after seeing $y_t$. Theoretical Bayesian Surprise is defined as $D_{KL}(\pi_{t+1}\|\pi_t)$, approximated in practice by information surprisal $S(y_t)\approx -\log P(y_t|y_{<t};\theta)$. Bayes-Chunk selects peaks in the surprisal curve and sorts them as boundaries $B=\{b_1,\ldots,b_M\}$.
    - **Design Motivation**: Fixed windows only ensure uniform length, not semantic integrity. High-surprisal tokens often correspond to transitions in logic, code structure, or narrative, making them ideal points for creating internally consistent and distinct segments.

2. **Structural Independence for Interference Reduction**:
    - **Function**: Explains why semantic boundaries stabilize multi-segment editing by making anchor keys across segments nearly orthogonal.
    - **Mechanism**: Closed-form updates for multi-segment editing can be viewed as an aggregation of multiple rank-1 updates. The paper provides a crosstalk bound showing that the interference on the $j$-th segment is proportional to $\sum_{t\neq j}\|\delta_t\|_2\cdot |k_t^T A k_j|$, where $A$ is the precision matrix from pre-training statistics. High similarity between segment keys leads to overwriting or crosstalk.
    - **Design Motivation**: Segments from Bayes-Chunk are more dispersed in both semantic embedding and anchor key space. The paper reports a reduction in average cross-segment similarity from 0.594 (fixed window) to 0.509 (Bayes-Chunk), visualized via key heatmaps showing weaker off-diagonal correlations.

3. **Causal Locality for Precursor Anchor Selection**:
    - **Function**: Determines that edit perturbations should be injected at the token preceding a high-surprisal segment.
    - **Mechanism**: For a target token $y_t$, the paper defines positional controllability as $\kappa(i\to t)=\|\nabla_{h_i}L(y_t)\|_2$. Theoretical analysis suggests that in Transformer residual flows, the backward pass from $t-1$ acts as a "vertical channel" preserving amplitude, while influence from $t-k$ must pass through attention weights, diluting the signal. Thus, for $k>1$, $\Delta\kappa_k=\kappa(t-1\to t)-\kappa(t-k\to t)>0$.
    - **Design Motivation**: High-surprisal tokens denote where the semantic trajectory shifts; the preceding hidden state is the most direct control entry point. Placing perturbations here requires fewer parameter changes and causes fewer side effects than distant historical tokens.

### Loss & Training
Optimization in AnyEdit++ occurs at two levels. Locally, for each Bayes-Chunk segment, the system optimizes perturbation $\delta_t$ to maximize the generation probability of the current chunk, conditioned on previous segments and historical perturbations. Globally, all $(k_t, v_t)$ pairs are fed into a MEMIT-style least-squares update, aiming to satisfy edit segments while maintaining general knowledge through a covariance constraint $C$. For fair comparison, both AnyEdit and AnyEdit++ use MEMIT as the base editor; the authors also integrated Bayes segmentation into FT-UKE to verify its general applicability.

## Key Experimental Results

### Main Results
The evaluation uses EditEverything, UnKE, and CounterFact datasets. EditEverything covers seven sectors: Math, Code, Physics, Chemistry, Biology, News, and Poetry. Performance is measured using BLEU and BERT Score (all-MiniLM-L6-v2).

| Model | Method | EditEverything Avg BLEU | EditEverything Avg BS | Main Changes vs AnyEdit |
|------|------|--------------------------|-------------------------|--------------------------|
| Llama-3.1-8B-Instruct | MEMIT | 42.61 | 82.74 | Traditional triplet editors are insufficient |
| Llama-3.1-8B-Instruct | AnyEdit | 72.64 | 94.23 | Fixed-window long text editing already shows significant improvement |
| Llama-3.1-8B-Instruct | AnyEdit++ | 75.00 | 94.50 | BLEU +2.36, BS +0.27 |
| Llama-2-7B | AnyEdit | 42.30 | 86.33 | Fixed windows are more fragile on weaker models |
| Llama-2-7B | AnyEdit++ | 50.13 | 87.66 | BLEU gain ~+8, BS +1.33 |
| Qwen-2.5-7B-Instruct | AnyEdit | 81.81 | 95.28 | Baseline is already strong on reasoning models |
| Qwen-2.5-7B-Instruct | AnyEdit++ | 85.33 | 96.29 | BLEU +3.52, BS +1.01 |

The crucial observation is the consistent gain across models, with the largest benefits seen in models like Llama-2-7B that are easily overwhelmed by long-form editing. Gains are particularly pronounced in Math and Code categories; e.g., in Code on Llama-2-7B, AnyEdit++ outperforms AnyEdit by nearly 20 BLEU points, indicating that structured logic benefits most from semantic partitioning.

| Method | UnKE BLEU | UnKE BS | CounterFact BLEU | CounterFact BS | Avg BLEU | Avg BS |
|------|-----------|---------|------------------|----------------|-----------|---------|
| MEMIT | 24.76 | 76.50 | 32.21 | 75.79 | 28.49 | 76.15 |
| AlphaEdit | 21.34 | 73.86 | 23.51 | 72.42 | 22.43 | 73.14 |
| AnyEdit | 79.02 | 95.88 | 86.27 | 97.85 | 82.65 | 96.87 |
| AnyEdit++ | 81.57 | 96.03 | 90.69 | 98.29 | 86.13 | 97.16 |

Reference benchmarks confirm that AnyEdit++ does not compromise basic editing capabilities on unstructured QA or factual datasets.

### Ablation Study
The paper provides structural independence analysis and plug-and-play verification with FT-UKE.

| Analysis Item | Fixed Window / Original | Bayes-Chunk / With Bayes | Note |
|--------|-------------------|------------------------------|------|
| Avg semantic similarity (EditEverything) | 0.594 | 0.509 | Bayes-Chunk segments are more independent, reducing crosstalk |
| FT-UKE Avg BLEU/BS (Llama-3.1-8B) | 99.90 / 99.99 | 99.95 / 99.99 | Minimal gain as original is near saturation |
| FT-UKE Avg BLEU/BS (Qwen-2.5-7B) | 99.52 / 99.93 | 99.57 / 99.96 | Shows transferability of segmentation strategy |
| QwQ-Edit Long CoT Math | AnyEdit benchmark | AnyEdit++ higher across all length/logic density groups | Structured logic benefits most from structural slicing |

### Key Findings
- **Structural Sensitivity**: Gains from Bayes-Chunk correlate with text structure. Fixed windows often break conditions or function definitions in Math and Code, where adaptive boundaries provide the highest utility.
- **Metric Nuance**: BERT Score improvements are smaller than BLEU gains. AnyEdit++ primarily improves precise generation and structural details rather than general semantic proximity.
- **Independence Proof**: Reduced similarity in anchor key space provides evidence that Bayesian boundaries help the solver distinguish between multiple edit targets.
- **Scalability**: Experiments on 300 long CoT math samples show that AnyEdit++ maintains its advantage as logic density and length increase.

## Highlights & Insights
- The transition of "text segmentation" from an engineering hyperparameter to a model-internal state reading is elegant. Surprisal curves directly reflect information jumps perceived by the model.
- Strong alignment between theory and method: Structural independence justifies *where to cut*, while causal locality justifies *where to edit*.
- Incremental cost is low. It requires no external boundary detectors or memory banks, utilizing only the target LLM's own token probabilities.
- Transferability: The surprisal-based chunking could potentially assist other tasks like long CoT distillation or code patch learning.

## Limitations & Future Work
- The paper lacks an extensive discussion of failure modes, such as cases where surprisal peaks are triggered by noisy formatting or tokenization anomalies rather than semantic shifts.
- Computing surprisal requires an additional forward pass, which might increase latency and memory overhead in extreme-scale batch editing.
- Heavy reliance on the model's own calibration. If a model is unfamiliar with a domain, its surprisal might trigger at "rare" tokens instead of logical turning points.
- While generated similarity is high, fine-grained evaluation of locality, portability, and multi-hop consistency is needed.

## Related Work & Insights
- **vs ROME / MEMIT**: AnyEdit++ inherits the closed-form updates of MEMIT but extends the scope to multiple long-form segments, specifically addressing crosstalk between multiple keys.
- **vs AlphaEdit**: While AlphaEdit focuses on null-space constraints to preserve unrelated knowledge, AnyEdit++ optimizes the topology of the edit target itself.
- **vs AnyEdit**: AnyEdit introduced the autoregressive sequence; AnyEdit++ identifies and fixes the "fixed-window" bottleneck within that framework.
- **vs FT-UKE**: By improving fine-tuning-based editing through Bayes segmentation, the paper suggests that "structure-aware slicing" is a universal component for long-form editing.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Using Bayesian Surprise for segmentation is intuitive but highly effective when grounded in causality and independence theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Extensive coverage across models and datasets, though could benefit from deeper locality/failure analysis.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear methodology and strong theoretical motivation.
- **Value**: ⭐⭐⭐⭐☆ Highly practical for long-form editing tasks and provides a reusable pattern for long-sequence decomposition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](../../ACL2026/knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)

</div>

<!-- RELATED:END -->
