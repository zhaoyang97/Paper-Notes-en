---
title: >-
  [Paper Note] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction
description: >-
  [ACL 2026][LLM/NLP][Cross-document relation extraction] This paper proposes HCRE, a model that reformulates cross-document relation extraction from direct classification over a large relation set into layer-wise hierarchical classification guided by a constructed relation tree. A predict-then-verify inference strategy is designed to mitigate inter-layer error propagation. HCRE achieves substantial improvements over both SLM and LLM baselines on the CodRED benchmark.
tags:
  - ACL 2026
  - LLM/NLP
  - Cross-document relation extraction
  - hierarchical classification
  - large language models
  - error propagation mitigation
  - predict-then-verify strategy
date: 2026-05-08
content_hash: 651e7f5043b9aa35
---

# HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction

**Conference**: ACL 2026
**arXiv**: [2604.07937](https://arxiv.org/abs/2604.07937)
**Code**: [https://github.com/XMUDeepLIT/HCRE](https://github.com/XMUDeepLIT/HCRE)
**Area**: NLP Understanding
**Keywords**: Cross-document relation extraction, hierarchical classification, large language models, error propagation mitigation, predict-then-verify strategy

## TL;DR
This paper proposes HCRE, a model that reformulates cross-document relation extraction from direct classification over a large relation set into layer-wise hierarchical classification guided by a constructed relation tree. A predict-then-verify inference strategy is designed to mitigate inter-layer error propagation. HCRE achieves substantial improvements over both SLM and LLM baselines on the CodRED benchmark.

## Background & Motivation

**Background**: Cross-document relation extraction (RE) aims to identify relations between entities distributed across different documents. More than half of the relational facts in Wikidata span multiple documents. Existing methods predominantly follow a "small language model (SLM) + classifier" paradigm.

**Limitations of Prior Work**: The limited language understanding capacity of SLMs constrains further progress in cross-document RE. Preliminary experiments conducted by the authors reveal that directly applying LLMs to this task yields suboptimal results, sometimes underperforming strong SLM baselines. In-depth analysis identifies the root cause as the excessive number of predefined relations (277 in CodRED): (1) a large number of semantically similar relations are difficult to distinguish; and (2) enumerating all relations results in excessively long inputs, distracting LLMs from key information in the documents.

**Key Challenge**: LLMs possess strong language understanding but struggle to handle large-scale relation option sets effectively, while SLMs can accommodate such sets but lack sufficient comprehension capacity.

**Goal**: To reduce the number of relation options an LLM must consider at each inference step, while avoiding the error propagation introduced by hierarchical classification.

**Key Insight**: Preliminary experiments demonstrate that reducing the number of relation options significantly improves LLM performance (see Figure 4), motivating the hierarchical classification design.

**Core Idea**: A hierarchical relation tree is constructed to guide the LLM in reasoning top-down across layers, with only a small number of options considered at each level. A predict-then-verify strategy further mitigates inter-layer error propagation through multi-view verification.

## Method

### Overall Architecture
HCRE consists of two core components: (1) an LLM $\mathcal{M}_1$ for relation prediction; and (2) a hierarchical relation tree constructed from the predefined relation set. The LLM performs layer-wise classification guided by the tree until reaching a leaf node corresponding to the target relation. A predict-then-verify strategy is applied at each layer during inference to enhance reliability.

### Key Designs

1. **Hierarchical Relation Tree Construction**:

    - **Function**: Organizes 277 predefined relations into a tree structure, reducing the number of options at each classification layer.
    - **Mechanism**: A high-capability LLM (e.g., GPT-4o) constructs the tree layer by layer. At each level, a partition criterion $C_l$ (e.g., "by domain") is generated, and relations under each current node are grouped and named as child nodes accordingly. The second level is specifically designed with two nodes—"valid relation" and "no valid relation"—to explicitly separate positive samples from NA instances, alleviating label imbalance. This process recurses until the maximum depth $L$ is reached.
    - **Design Motivation**: The number of child nodes per parent is substantially smaller than the total relation count, effectively reducing the classification difficulty at each step.

2. **Predict-then-Verify (PtV) Inference Strategy**:

    - **Function**: Mitigates error propagation at each tree level through multi-view verification.
    - **Mechanism**: The strategy proceeds in two steps. *Prediction step*: the LLM selects the top-ranked node $\hat{r}_{1st}$ and the second-ranked node $\hat{r}_{2nd}$ from the current-level option set $\mathcal{R}_l$. *Verification step*: each of $\hat{r}_{1st}$ and $\hat{r}_{2nd}$ is replaced by its child nodes, constructing three verification option sets $\mathcal{R}_l^{v_1}, \mathcal{R}_l^{v_2}, \mathcal{R}_l^{v_3}$. The LLM selects the best node from each verification set. If more than half of the auxiliary verification nodes are semantically consistent with $\hat{r}_{1st}$, the prediction is confirmed; otherwise, $\hat{r}_{1st}$ is removed and the process repeats.
    - **Design Motivation**: The verification option sets provide finer-grained semantic information at the child-node level, enabling the LLM to detect subtle distinctions between the top and second-ranked candidates.

3. **Training Data Construction**:

    - **Function**: Constructs training samples for hierarchical classification and the verification step.
    - **Mechanism**: For each original training instance $(x, \mathcal{R}, r)$, the root-to-leaf path is identified and expanded into $L-1$ layer-wise training samples forming $\mathcal{D}_1$. For each sample in $\mathcal{D}_1$, the verification step is simulated to generate three verification samples, forming $\mathcal{D}_2$. The LLM is then fine-tuned on the combined dataset $\mathcal{D}_1 \cup \mathcal{D}_2$.
    - **Design Motivation**: Explicitly training on verification examples enables the LLM to effectively leverage fine-grained information during inference.

### Loss & Training
Standard supervised fine-tuning with cross-entropy loss is applied over the combined dataset $\mathcal{D}_1 \cup \mathcal{D}_2$.

## Key Experimental Results

### Main Results
Results on the CodRED dataset:

| Model | Closed micro F1 | Closed binary F1 | Open micro F1 | Open binary F1 |
|---|---|---|---|---|
| ECRIM (RoBERTa) | 42.54 | 49.47 | 23.39 | 27.60 |
| NEPD (RoBERTa) | 42.96 | 52.67 | 30.12 | 37.04 |
| Vanilla LLaMA | 38.14 | 41.43 | 15.19 | 17.00 |
| **HCRE (LLaMA)** | **45.35** | **58.19** | **34.91** | **49.33** |

### Ablation Study

| Configuration | micro F1 | binary F1 | Note |
|---|---|---|---|
| Full HCRE | 45.35 | 58.19 | Complete model |
| w/o multi-view | 39.37 | 49.63 | Single verification set only |
| w/o PtV | 37.66 | 47.28 | No predict-then-verify strategy |
| w/o LTC | 43.18 | 56.60 | Tree generated directly, not layer-by-layer |
| w/o HRT | 38.14 | 41.43 | No hierarchical tree; direct classification |

### Key Findings
- HCRE outperforms the strongest SLM baseline (NEPD) by 2.39 in closed micro F1 and 5.52 in closed binary F1.
- Gains are more pronounced in the open setting (binary F1 from 37.04 to 49.33), suggesting hierarchical classification is particularly beneficial for long documents.
- The predict-then-verify strategy is the most critical component: its removal leads to a 7.69 drop in micro F1 and a 10.91 drop in binary F1.
- Multi-view verification (3 verification sets vs. 1) yields an additional gain of 1.71 in micro F1.
- Error propagation analysis confirms that the PtV strategy effectively reduces propagation errors at every layer.

## Highlights & Insights
- The preliminary experiments provide a clear empirical diagnosis: excessive relation options are the primary cause of LLM underperformance in cross-document RE. This finding has broader implications—LLMs may face analogous challenges in any large-scale label classification task.
- The predict-then-verify strategy is elegantly designed: replacing parent nodes with their children to construct verification sets essentially leverages finer-grained information to validate coarser-grained judgments.
- The evaluation metric analysis (maximum F1 overstating performance; P@K sensitivity to data scale) provides valuable methodological guidance for the community.

## Limitations & Future Work
- Tree construction relies on GPT-4o, introducing dependency on tree quality and incurring non-trivial cost.
- The verification step increases inference overhead, as multiple LLM calls are required per layer.
- Evaluation is conducted on a single dataset (CodRED); generalizability remains to be confirmed.
- Future work could explore adaptive tree depth or dynamically adjusted verification intensity to balance efficiency and accuracy.

## Related Work & Insights
- **vs. NEPD**: NEPD focuses on long-range dependency modeling but remains constrained by the capacity ceiling of SLMs. HCRE leverages the superior language understanding of LLMs to surpass this limitation.
- **vs. Hierarchical Text Classification (HTC) methods**: Conventional HTC approaches (e.g., DFS-L, BFS-L) lack a verification mechanism, resulting in severe error propagation on cross-document RE.
- **vs. Vanilla LLM**: Preliminary experiments clearly demonstrate that directly applying LLMs to large relation sets is ineffective, confirming that hierarchical decomposition is necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of hierarchical classification and the predict-then-verify strategy is novel and practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations are comprehensive; preliminary experiments provide convincing motivational analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from problem discovery to analysis to solution is exceptionally clear.
- Value: ⭐⭐⭐⭐ — Offers practical guidance for applying LLMs to large-scale classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](../../AAAI2026/llm_nlp/from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[CVPR 2026\] Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo](../../CVPR2026/llm_nlp/bi_cmpstereo_bidirectional_cross_modal_prompting_for_event_frame_asymmetric_stereo.md)
- [\[ACL 2026\] Detoxification for LLM from Dataset Itself](detoxification_for_llm_from_dataset_itself.md)
- [\[ACL 2026\] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)
- [\[ACL 2026\] Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes](route_to_rome_attack_directing_llm_routers_to_expensive_models_via_adversarial_s.md)

</div>

<!-- RELATED:END -->
