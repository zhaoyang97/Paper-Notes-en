---
title: >-
  [Paper Note] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion
description: >-
  [ICML 2026][Graph Learning][Hyper-relational Knowledge Graphs] This paper proposes the "Fact Generation" task, extending hyper-relational knowledge graph (HKG) completion from "filling a single gap" to "generating comple…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Hyper-relational Knowledge Graphs"
  - "Fact Generation"
  - "Masked Discrete Diffusion"
  - "Contextual Message Passing"
  - "Link Prediction"
date: 2026-05-08
content_hash: 1bd2f14868be52ae
---

# Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion

**Conference**: ICML 2026  
**arXiv**: [2605.24064](https://arxiv.org/abs/2605.24064)  
**Code**: https://github.com/bdi-lab/KREPE (Available)  
**Area**: Graph Learning / Knowledge Graph Representation / Generative Modeling  
**Keywords**: Hyper-relational Knowledge Graphs, Fact Generation, Masked Discrete Diffusion, Contextual Message Passing, Link Prediction

## TL;DR
This paper proposes the "Fact Generation" task, extending hyper-relational knowledge graph (HKG) completion from "filling a single gap" to "generating complete facts from arbitrary mask patterns or even from scratch." It introduces KREPE, the first generative HKG representation learning method. KREPE utilizes contextual message passing to encode intra-fact and inter-fact dependencies and models the joint conditional distribution of missing components via masked discrete diffusion. It achieves SOTA performance in link prediction across three HKG benchmarks and significantly outperforms strong LLM baselines (e.g., GPT-5.2 / Gemini 3 Pro) in fact generation (e.g., 0.855 vs. 0.343 for generation from scratch on WikiPeople-).

## Background & Motivation

**Background**: HKGs extend traditional triples $(h,r,t)$ into "primary triples + qualifier key-value pairs" $((h,r,t),\{(k_i,v_i)\})$, enabling the expression of complex multi-dimensional facts (typical examples include Wikidata and YAGO). Mainstream HKG completion methods (StarE, HyperFormer, HAHE, MAYPL, etc.) model the task as **Link Prediction**: assuming exactly one position in a fact is `?`, the model ranks candidates from the entity/relation set for that position.

**Limitations of Prior Work**: The single-gap assumption is severely detached from reality. In real-world queries, the number of missing components is uncertain—subject and relation might be missing simultaneously, an entire fact might be unknown, or only a qualifier might be known to infer the primary triple. Once multiple positions are missing, scoring-based methods face a combinatorial explosion (the candidate space grows exponentially with the number of qualifiers). Forced extensions (e.g., HAHE's multi-position prediction) can only handle predefined fixed mask patterns, at most missing one element per qualifier pair.

**Key Challenge**: The intrinsic requirement of HKG completion is "generating new facts under uncertain mask patterns," whereas existing architectures are essentially "discriminative scoring + single mask template." These two are misaligned in both training objectives and inference workflows. Directly applying LLMs is also problematic—LLM-based KG methods (like KICGPT) follow a "KG model proposes candidates, LLM reranks" pipeline, but since the KG backbone cannot handle multi-gap queries, reranking cannot even begin.

**Goal**: (1) Formally define "Fact Generation," a new task covering arbitrary mask patterns (including fully empty); (2) Design a single representation learning framework capable of **both** link prediction and fact generation, unifying the different models/objectives.

**Key Insight**: Treat missing components as a sampling problem from the joint conditional distribution $P_\theta(x_{\text{mask}} \mid \zeta, G)$. This naturally introduces **masked discrete diffusion**, a mechanism that inherently supports arbitrary subset masking and iterative reconstruction. Additionally, it is observed that dependencies in HKGs are both "intra-fact" (mutual constraints between head/relation/tail/qualifier within a fact) and "inter-fact" (shared semantics of an entity across multiple facts), requiring modeling at both the message passing and diffusion noise levels.

**Core Idea**: Use **contextual message passing** (explicitly excluding a component's own information during updates to force dependency on context) for representation, and use **bi-level noise + AO-AR diffusion objective** (simultaneously perturbing the observed subgraph and query mask patterns) for training. Link prediction is treated as a special case of "single-mask fact generation," unified within the same model.

## Method

### Overall Architecture
The input is an HKG $G=(\mathcal{V},\mathcal{R},\mathcal{H})$ and a masked query $\zeta$ (where an arbitrary subset of a fact is replaced by `?`). During training, each epoch randomly partitions the training fact set $\mathcal{H}_{\text{train}}$ into an "observed set $\mathcal{H}_{\text{obs}}$" and a "target set $\mathcal{H}_{\text{tgt}}$." $L$ layers of contextual message passing on $\mathcal{H}_{\text{obs}}$ produce entity/relation representations. Facts in $\mathcal{H}_{\text{tgt}}$ are masked according to a distribution $\mathcal{D}_\xi$ and used as queries. The model outputs a distribution over candidates ($\mathcal{V}$ or $\mathcal{R}$) for each mask position, optimizing the negative log-likelihood (AO-AR loss). During inference, $\mathcal{H}_{\text{train}}$ acts as the observed set. A single forward pass yields distributions for all mask positions—candidates are ranked for **link prediction**, while top-$p$ sampling and iterative denoising are used for **fact generation** (even for $((?,?,?),\{(?,?)\})$ from scratch).

### Key Designs

1.  **Contextual Message Passing (CM)**:
    - **Function**: Generates hierarchical representations for each entity/relation considering both intra-fact companions and cross-fact structures, serving as the basis for diffusion decoding.
    - **Mechanism**: Decomposes each fact into "relation-entity pairs" $(p,e)$ projected by role $\rho\in\{\texttt{head},\texttt{tail},\texttt{qual}\}$ as $z_{(p,e)}^{(l)} = W_\rho^{(l)}[p^{(l-1)};e^{(l-1)}]$. The fact representation $z_\xi^{(l)}$ is the sum of all pairs. When updating a component $e$, a "self-excluding" contextual message $m_{\xi\to(p,e)}^{(l)} = \text{MLP}^{(l)}\big((z_\xi^{(l)} - z_{(p,e)}^{(l)})/(n_\xi+1)\big)$ is calculated, followed by concatenating paired relation/entity representations to get $m_{\xi\to e}^{(l)}$ and $m_{\xi\to p}^{(l)}$. Messages from multiple facts are aggregated via multi-head attention.
    - **Design Motivation**: Ablation (i) replacing contextual messages with $z_\xi$ directly (retaining self-information) dropped WD50K link prediction MRR from 0.419 to 0.408 and fact generation accuracy from 0.717 to 0.552. "Explicitly kicking out self" forces the model to use surrounding structures to infer identity, providing the inductive bias needed for generative reconstruction. Simultaneously, entities/relations use **shared** initial tokens $z_{\text{ENT}}, z_{\text{REL}}$ (Ablation iv shows per-entity independent embeddings cause WD50K MRR to crash to 0.272), preventing the model from degenerating into "memorizing each ID."

2.  **Probabilistic Masked Fact Decoding**:
    - **Function**: Converts "components to be predicted" into explicit probability distributions over candidates ($\mathcal{V}$ or $\mathcal{R}$), supporting both discriminative (ranking) and generative (sampling) inference.
    - **Mechanism**: Masked components are initialized as learnable vectors $x_{\text{ENT}}$ or $x_{\text{REL}}$. During message passing, "the query $\zeta$ only affects the masks themselves and is not allowed to pollute known components." The final layer $x_{\text{mask}}^{(L)}$ is dot-producted with all candidate final representations $c^{(L)}$ followed by softmax: $P_\theta(x \mid \zeta, G) = \text{Softmax}_{c\in\mathcal{C}}(x_{\text{mask}}^{(L)} \cdot c^{(L)})$. Local query information is encoded in $x_{\text{mask}}^{(L)}$ while global HKG structure is in $c^{(L)}$.
    - **Design Motivation**: Dot product is used instead of cosine similarity (Ablation vi shows a 0.6 point drop) because the **norm** of the representation vector carries information about "confidence in the candidate given the context." Mask updates depend unidirectionally on $\zeta$ to prevent mask noise from polluting known components during training, ensuring the discriminative capability for link prediction.

3.  **Bi-level Noising + AO-AR Diffusion Objective**:
    - **Function**: Enables the model to handle generation under "arbitrary mask patterns" on "varying observed subgraphs," allowing both single-gap filling and from-scratch generation.
    - **Mechanism**: **Inter-fact noise**—stochastic structure sampling partitions the training set into $\mathcal{H}_{\text{obs}}$ and $\mathcal{H}_{\text{tgt}}$ each epoch, forcing representations to derive from changing observed subgraphs. **Intra-fact noise**—for $\xi\in\mathcal{H}_{\text{tgt}}$, the number of masks $n_{\text{mask}}\sim\mathcal{U}(\{1,\dots,2n_\xi+3\})$ is sampled, then $n_{\text{mask}}$ components are masked without replacement to form $\zeta$. The training objective uses Any-Order Auto-Regressive (AO-AR) loss (equivalent to time-independent reparameterization for absorbing-state discrete diffusion): $\mathcal{L}_{\text{AO-AR}} = \mathbb{E}_{\xi,\mathcal{M}_\zeta}\big[-\sum_{(x,y)\in\mathcal{M}_\zeta} \log P_\theta(x=y \mid \zeta,(\mathcal{V},\mathcal{R},\mathcal{H}_{\text{obs}}))\big]$. Fact generation uses iterative denoising during inference.
    - **Design Motivation**: Ablation (ii) turning off stochastic structure sampling crashed WD50K link prediction MRR from 0.419 to 0.296, proving that "varying observed subgraphs during training" is key to unifying discrimination and generation. Ablation (v) replacing AO-AR with standard link prediction cross-entropy dropped fact generation accuracy from 0.717 to 0.038, confirming that link prediction objectives inherently fail to learn joint distributions.

### Loss & Training
A single AO-AR loss (Eq. 7) is used. Training once supports three downstream tasks: entity prediction, relation prediction, and fact generation. The mask upper bound $2n_\xi+3$ covers all possible patterns, including "from scratch." Discriminative ranking and generative sampling share the same representations and parameters.

## Key Experimental Results

### Main Results
Datasets: WD50K, WikiPeople-, WikiPeople. Metrics: MRR / Hit@10 / Hit@1 for link prediction; Accuracy (LLM-as-a-judge, GPT-5.2) for fact generation.

**Link Prediction (Entity Prediction, All positions)**

| Dataset | Metric | KREPE | Prev. SOTA (MAYPL) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| WD50K | MRR | **0.419** | 0.411 | +0.008 |
| WD50K | Hit@10 | **0.580** | 0.572 | +0.008 |
| WikiPeople- | MRR | **0.522** | 0.521 | +0.001 |
| WikiPeople | MRR | **0.491** | 0.488 | +0.003 |
| WikiPeople | Hit@10 | **0.642** | 0.635 | +0.007 |

Relation prediction (WD50K All positions) MRR improved from HDiff's 0.956 to **0.968**, with Hit@10 reaching 0.995.

**Fact Generation (Accuracy, higher is better)**

| Dataset | Setting | KREPE | Strongest LLM Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| WikiPeople- | Scratch | **0.855** | 0.343 (Random+Gemini 3 Pro) | +0.51 |
| WD50K | Scratch | **0.717** | 0.351 (Neighbor+Gemini) | +0.37 |
| WikiPeople | Scratch | **0.777** | 0.326 (Few-shot+Gemini) | +0.45 |
| WD50K | Arbitrary Masking | **0.604** | 0.604 (Random+Gemini) | Equal |
| WikiPeople- | Targeted | **0.600** | 0.394 (Neighbor+Gemini) | +0.21 |

On WikiPeople- Scratch, the Valid & Novel Rate reached 0.351 vs. LLM's 0.242, with expected generation attempts at 2.85 vs. 4.13—more accurate and more novel.

### Ablation Study
(WD50K Link Prediction MRR / Generation from Scratch Accuracy)

| Config | LP MRR | FG Acc | Description |
| :--- | :--- | :--- | :--- |
| Full KREPE | 0.419 | 0.717 | Full model |
| (i) w/o Context Msg | 0.408 | 0.552 | No self-exclusion, generation drops 16.5 pts |
| (ii) w/o Stochastic Sampling | 0.296 | 0.545 | No obs/tgt split, LP crashes |
| (iii) w/o Attention | 0.408 | 0.673 | Attention replaced by mean pooling |
| (iv) Individual Init | 0.272 | 0.466 | Per-entity/relation independent embeddings |
| (v) w/ LP Loss | 0.415 | **0.038** | Replaced with discriminative cross-entropy, generation zeroed |
| (vi) w/ Cosine Sim | 0.419 | 0.711 | Cosine similarity instead of dot product |

### Key Findings
- **AO-AR diffusion objective is the fundamental source of generation capability**: Replacing it with standard link prediction loss (v) reduces accuracy nearly to zero, yet retaining it does no harm to link prediction (LP MRR 0.419 vs 0.419)—key evidence for a unified discriminative/generative model.
- **"Excluding self" and "Shared initial tokens" are counter-intuitive but necessary designs**: Ablations (i) and (iv) show that including the component's own information allows the model to take shortcuts, while individual embeddings cause WD50K MRR to crash to 0.272.
- **LLMs fail significantly in HKG fact generation**: Even with 1000 random facts as context, Gemini 3 Pro's scratch generation accuracy is only 0.343, far below KREPE's 0.855. Furthermore, there is a dispute over whether LLMs "generate" or "recall pre-training data," whereas KREPE's result is more reliable due to explicit training data.
- **Generation quality and novelty coexist**: V&N Rate 0.351 + expectation of 2.85 attempts for a new fact. Qualitative analysis (Table 7) shows KREPE can generate reasonable facts for "Toy Story" like "(nominated for, Oscars Best Score), {(subject of, 68th Oscars), (nominee, R. Newman)}" across multiple qualifiers, and can generate facts across diverse domains like movies, sports, and literature "out of thin air."

## Highlights & Insights
- **Task definition itself is a contribution**: Raising HKG completion from "fill-in-the-blank" to "conditional fact generation" provides a new benchmark dimension for future work. Previous HKG models could not perform this task directly; the authors provided 7 strong baselines (2 discriminative variants + 5 LLM prompting strategies) for comparison.
- **"Bi-level noise" is the key trick for unifying discrimination and generation**: Varying the observed subgraph (inter-fact) + varying the mask pattern (intra-fact) expands the "training distribution" enough so that link prediction naturally becomes a special case of fact generation. This approach is transferable to any structural prediction task requiring both discrimination and generation (e.g., table completion, code completion).
- **"Excluding self contextual message" + "shared initial tokens"**: An elegant combination in representation learning—forcing the model to reconstruct identities solely from context. It essentially turns every fact into a self-supervised denoising sample, being particularly friendly to low-resource/long-tail entities.
- **Dot product vs. cosine similarity**: Magnitude contributes substantially (Ablation vi drops 0.6 points). In generative representation, "how confident the candidate is in this context" and "how similar it is to the mask representation" are distinct; forced normalization loses the former.

## Limitations & Future Work
- **Transductive setting only**: Only entities/relations seen during training can be predicted. Handling new entities (inductive scenarios) is listed as future work.
- **Quadratic inference complexity w.r.t. query length**: $\mathcal{O}(|\zeta|^2 d^2 (L + |\mathcal{V}| + |\mathcal{R}|))$. When the candidate set $|\mathcal{V}|$ is huge (e.g., millions of entities in Wikidata), the softmax layer remains a bottleneck.
- **Potential Bias in LLM-as-a-judge**: Although validated with multi-judge + 10% human sub-set (Pearson 0.997 / 0.987), there's a risk of "stylistic bias" between the judge and generative models. Some of KREPE's high scores might stem from LLMs preferring "structured, conservative" facts.
- **Mask upper bound $2n_\xi+3$ is empirical**: It's unclear if this is sufficient for very long qualifier sequences; in real HKGs, some facts can have dozens of qualifiers.

## Related Work & Insights
- **vs. MAYPL (Lee & Whang, 2025)**: A discriminative HKG model from the same group that encodes HKG structure but only performs single-gap link prediction. KREPE reuses the structural encoding but shifts to the AO-AR diffusion objective to enable generation—and even outperforms it on link prediction (WD50K All MRR 0.419 vs. 0.411).
- **vs. HDiff (Luo et al., 2025)**: Also uses diffusion on HKGs, but for **denoising continuous embeddings to rerank candidates**. KREPE uses **discrete diffusion to directly model joint conditional distributions**, enabling generation. This distinction highlights that the value of diffusion in structural data lies in "explicit probability," not just "embedding regularization."
- **vs. LLM-based KG (KICGPT, MuKDC, etc.)**: The retrieve-then-rerank paradigm faces a deadlock with multi-gap queries. KREPE generates using the HKG's own probabilistic model, avoiding dependence on external retrievers.
- **vs. GPHT / Triple Set Prediction (Zhang et al., 2024)**: GPHT handles "triples from scratch" via ranking, which is barely manageable in KG candidate spaces. For HKGs, the candidate space grows exponentially with qualifiers, making the ranking paradigm fail—this is the hard constraint necessitating KREPE's generative paradigm.
- **Transferable Insight**: The combination of AO-AR / masked discrete diffusion + role-aware message passing is worth trying for any structural generation task with variable components and rich dependencies (e.g., drug-target networks, table schema reasoning, protein interaction prediction).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Simultaneously provides a new task (fact generation) and the first generative HKG framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × 3 downstream tasks, compared against 15+ HKG baselines and 7 LLM strategies with 6 ablations + human validation.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions, precise ablation mapping; formulas are dense, and Bi-level noising require careful reading of Figure 2.
- Value: ⭐⭐⭐⭐⭐ Achieves new SOTA while bridging "HKG completion" and "probabilistic generation," pushing a paradigm shift for the KG community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](../../AAAI2026/graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)

</div>

<!-- RELATED:END -->
