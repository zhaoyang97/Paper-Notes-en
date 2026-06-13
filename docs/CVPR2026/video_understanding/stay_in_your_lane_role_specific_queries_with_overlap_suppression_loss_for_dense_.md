---
title: >-
  [Paper Note] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning
description: >-
  [CVPR 2026][Video Understanding][Dense Video Captioning] This paper proposes ROS-DVC, which decouples the shared queries in DETR-based DVC frameworks into independent localization queries and caption queries…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Dense Video Captioning"
  - "Role-Specific Queries"
  - "Overlap Suppression Loss"
  - "Contrastive Alignment"
  - "Concept Guidance"
date: 2026-05-08
content_hash: 23c9e0019d3acd8e
---

# Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning

**Conference**: CVPR 2026
**arXiv**: [2603.11439](https://arxiv.org/abs/2603.11439)  
**Code**: [https://github.com/edwardback/ROS-DVC](https://github.com/edwardback/ROS-DVC)  
**Area**: Video Understanding
**Keywords**: Dense Video Captioning, Role-Specific Queries, Overlap Suppression Loss, Contrastive Alignment, Concept Guidance

## TL;DR
This paper proposes ROS-DVC, which decouples the shared queries in DETR-based DVC frameworks into independent localization queries and caption queries, introduces an Overlap Suppression Loss to penalize temporal overlap between queries, and employs Cross-Task Contrastive Alignment to maintain cross-task semantic consistency. The approach achieves state-of-the-art captioning and localization performance on YouCook2 and ActivityNet Captions.

## Background & Motivation
Dense Video Captioning (DVC) aims to **simultaneously** perform temporal event localization and natural language description in long videos. Early methods adopted a two-stage "localize-then-describe" strategy, with two independently trained modules lacking interaction. PDVC first introduced the DETR architecture into DVC, using a set of learnable queries to parallelly predict event segments and generate descriptions, enabling end-to-end joint optimization.

**Two major pain points of existing query-based DVC**:

**Multi-task interference**: Localization and captioning share the same set of queries, requiring each query to simultaneously perform the two highly distinct tasks of "boundary finding" and "description writing." At the attention level, query attention can neither precisely focus on event boundaries (localization requires broad attending) nor densely attend to fine-grained semantics of key frames (captioning requires dense attending)—the two optimization objectives conflict, leading to ambiguous attention. Although DDVC attempted query decomposition, it merely derives caption queries from localization queries via MLP, resulting in highly similar attention distributions without achieving genuine task separation.

**Temporal redundancy**: Multiple queries tend to capture overlapping temporal intervals, generating redundant descriptions. As shown in Figure 1(a), the baseline model repeatedly detects the same temporal segment and produces identical captions, severely degrading localization precision and description diversity.

**Key Challenge**: Queries must simultaneously serve two heterogeneous tasks, yet the shared representation space causes conflicting optimization directions; additionally, the absence of explicit constraints on inter-query temporal relationships prevents overlap from being eliminated automatically.

**Key Insight**: Rather than requiring a single query to serve dual roles, two independent sets of queries should "stay in their lanes"—localization queries focus on broad temporal context for boundary estimation, while caption queries focus on semantic details of key frames. Explicit loss designs further constrain query behavior: contrastive loss ensures cross-task consistency, and overlap loss penalizes temporal redundancy.

**Core Idea**: Eliminate multi-task interference in DVC via role-specific independent queries, and eliminate temporal redundancy via Overlap Suppression Loss.

## Method

### Overall Architecture
ROS-DVC builds upon a DETR-style parallel encoder-decoder architecture:
- **Input**: Video frame sequences, with frame-level features extracted by pretrained CLIP ViT-L/14
- **Transformer Encoder**: Models temporal context over frame features
- **Decoder**: Two independent sets of learnable queries (localization + caption) each interact with frame features via cross-attention
- **Output**: Four task heads predicting event count, temporal boundaries, caption text, and event concepts respectively

### Key Designs

1. **Role Specific Query Initialization**:

    - **Function**: Separates the conventional single query set in DVC into two independent groups—$\{q_{\text{loc}}^j\}_{j=1}^K$ and $\{q_{\text{cap}}^j\}_{j=1}^K$
    - **Mechanism**: Both query groups are initialized from **fully independent learnable embedding spaces** and each interacts with encoded frame features via cross-attention. Localization queries broadly attend to temporal context at the attention level to estimate event boundaries; caption queries densely attend to key frames to capture semantic details. Both groups share the same decoder and reference the same visual location in cross-attention (defined by the reference points of the localization queries), ensuring consistency in visual grounding.
    - **Design Motivation**: Unlike DDVC, which derives caption queries from localization queries via MLP (introducing dependency and limiting attention diversity), the fully independent initialization in this work allows each query group to optimize toward attention patterns best suited to its task, fundamentally eliminating multi-task interference.

2. **Cross-Task Contrastive Alignment (CTCA) Loss**:

    - **Function**: Ensures semantic consistency between the separated localization and caption queries.
    - **Mechanism**: Hungarian matching determines the set of query indices $\mathcal{M}$ corresponding to ground truth. For each $j \in \mathcal{M}$, the $j$-th caption query $\tilde{q}_{\text{cap}}^j$ and the corresponding localization query $\tilde{q}_{\text{loc}}^j$ form a positive pair, with the remaining localization queries serving as negatives:
    $\mathcal{L}_{\text{CTCA}} = -\sum_{j \in \mathcal{M}} \log \frac{\exp(\text{sim}(\tilde{q}_{\text{cap}}^j, \tilde{q}_{\text{loc}}^j)/\tau)}{\sum_{j'} \exp(\text{sim}(\tilde{q}_{\text{cap}}^j, \tilde{q}_{\text{loc}}^{j'})/\tau)}$
      where $\text{sim}(\cdot)$ denotes cosine similarity and $\tau$ is a temperature parameter.
    - **Design Motivation**: After query separation, without explicit constraints, the two query groups may semantically drift and become inconsistent (i.e., the localization and caption of the same event may not correspond). CTCA pulls corresponding queries closer and pushes non-corresponding queries apart via asymmetric contrastive learning, equipping localization queries with semantic awareness.

3. **Overlap Suppression Loss (OSL)**:

    - **Function**: Explicitly penalizes temporal overlap between queries, encouraging the model to learn distinct, non-overlapping event regions.
    - **Mechanism**: The overlap between predicted intervals $B_i$ and $B_j$ is defined as $P_o(i,j) = \text{IoU}(B_i, B_j)$. A ground-truth alignment score $P_g(i,j) = \text{IoU}(B_i, G_j)$ is defined, from which an adaptive weight is constructed:
    $\alpha = \gamma \cdot P_g + (1-\gamma) \cdot (1-P_g), \quad \gamma \leq 0.5$
      When a prediction is well-aligned with the GT (high $P_g$), $\alpha$ is small and suppression is weak; when the prediction is misaligned, $\alpha$ is large and overlap is strongly penalized. The final loss is:
    $\mathcal{L}_{\text{OSL}} = -\alpha \cdot \log(\beta - P_o)$
      where $\beta$ is a maximum overlap threshold hyperparameter.
    - **Design Motivation**: Naively penalizing all overlap would suppress the correct detection of adjacent events. The GT-aware adaptive weight addresses this: predictions well-matched to GT receive lighter penalization, while redundant predictions deviating from GT are penalized heavily.

4. **Concept Guider**:

    - **Function**: A lightweight MLP auxiliary head that outputs event-level multi-hot concept vectors, enriching the semantic representation of caption queries.
    - **Mechanism**: The top-$N_c$ nouns and verbs from training set captions are extracted as a concept vocabulary, and multi-hot labels $Y^c \in \{0,1\}^{N_c}$ are constructed for each event. The concept head takes caption query output $\tilde{q}_{\text{cap}}$ as input to an MLP + sigmoid to predict concept distribution, trained with cross-entropy:
    $\hat{y}_c = \text{sigmoid}(\text{MLP}(\tilde{q}_{\text{cap}}))$
      This head is not used at inference; it serves only as a training-time auxiliary task to guide caption queries toward encoding high-level semantics.
    - **Design Motivation**: Rather than relying on external memory banks (e.g., CM2), the approach implicitly teaches caption queries to encode core event concepts through an auxiliary task, producing more specific and context-aware descriptions.

### Loss & Training
The total loss combines standard DVC losses with three newly introduced losses:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{giou}}\mathcal{L}_{\text{giou}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{cap}}\mathcal{L}_{\text{cap}} + \lambda_{\text{ec}}\mathcal{L}_{\text{ec}} + \lambda_{\text{CTCA}}\mathcal{L}_{\text{CTCA}} + \lambda_{\text{OSL}}\mathcal{L}_{\text{OSL}} + \lambda_{\text{CG}}\mathcal{L}_{\text{CG}}$$

- Visual encoding: CLIP ViT-L/14, 1 FPS frame sampling
- 2-layer deformable transformer decoder, 4-scale multi-scale features
- YouCook2: $K=50$ queries/group, $F=200$ frames; ActivityNet: $K=10$, $F=100$
- OSL hyperparameters: $\gamma=0.25$, $\beta=1.0$; concept vocabulary size $N_c=30$

## Key Experimental Results

### Main Results — Captioning Performance

| Method | Pretrained | YouCook2 CIDEr↑ | YouCook2 SODA_c↑ | ActivityNet CIDEr↑ | ActivityNet SODA_c↑ |
|--------|-----------|-----------------|-------------------|--------------------|---------------------|
| PDVC | ✗ | 29.69 | 4.92 | 29.97 | 5.92 |
| CM2 | ✗ | 31.66 | 5.34 | 33.01 | 6.18 |
| MCCL | ✗ | 36.09 | 5.21 | 34.92 | 6.16 |
| E2DVC | ✗ | 34.26 | 5.39 | 33.63 | 6.13 |
| **ROS-DVC (Ours)** | **✗** | **39.18** | **7.06** | **35.04** | **6.45** |

On YouCook2, CIDEr exceeds MCCL (which uses an external memory bank) by 3.09 and SODA_c by 1.85; on ActivityNet, CIDEr is the best (35.04), outperforming all non-pretrained methods.

### Main Results — Localization Performance

| Method | YouCook2 Rec.↑ | YouCook2 Pre.↑ | YouCook2 F1↑ | ActivityNet Rec.↑ | ActivityNet F1↑ |
|--------|---------------|---------------|-------------|------------------|----------------|
| PDVC | 22.89 | 32.37 | 26.81 | 53.27 | 54.78 |
| E2DVC | 24.36 | 34.75 | 28.64 | 54.67 | 56.14 |
| **ROS-DVC** | **29.34** | **35.26** | **32.03** | **55.35** | **55.50** |

On YouCook2, F1 exceeds E2DVC by 3.39; on ActivityNet, recall and precision are well-balanced, indicating that the event counter predicts event counts closer to GT.

### Ablation Study

| RSQI | CTCA | OSL | CG | CIDEr↑ | SODA_c↑ | F1↑ | Note |
|------|------|-----|----|--------|---------|-----|------|
| ✗ | ✗ | ✗ | ✗ | 29.69 | 5.39 | 26.81 | Baseline (PDVC) |
| ✓ | ✗ | ✗ | ✗ | 32.33 | 5.43 | 27.00 | Query separation only → CIDEr +2.64 |
| ✗ | ✗ | ✓ | ✗ | 33.60 | 6.79 | 31.22 | OSL only → F1 large gain +4.41 |
| ✗ | ✗ | ✗ | ✓ | 31.40 | 5.62 | 27.69 | Concept guidance only → CIDEr +1.71 |
| ✓ | ✓ | ✗ | ✗ | 34.48 | 5.58 | 27.59 | Query separation + contrastive → CIDEr +4.79 |
| ✓ | ✓ | ✓ | ✓ | **39.18** | **7.06** | **32.03** | **Full model, best on all metrics** |

### Key Findings
- **OSL contributes most to localization**: Adding OSL alone raises F1 from 26.81 to 31.22 (+4.41), the largest improvement among all single components, directly validating the hypothesis that "inter-query temporal overlap is the bottleneck for localization."
- **RSQI + CTCA contributes most to captioning**: Query separation combined with contrastive alignment raises CIDEr from 29.69 to 34.48 (+4.79), demonstrating that task decoupling genuinely unlocks captioning capacity.
- **All four components are necessary**: The full model achieves CIDEr 39.18, significantly exceeding any three-component combination; the modules are complementary rather than redundant.
- **$\gamma=0.25$ is the optimal balance point for OSL**: Smaller $\gamma$ degrades captioning quality; larger $\gamma$ weakens overlap suppression.
- **Removing $\alpha$ (i.e., uniformly penalizing all overlap) reduces Precision**: This validates the necessity of the GT-aware adaptive weight.
- **Query count of 50 is optimal**: Too few queries miss events; too many introduce redundant proposals; 50 achieves the best balance between captioning and localization.

## Highlights & Insights
- **Simplicity and effectiveness of independent query initialization**: Without additional encoders or complex query interaction mechanisms, initializing from different embedding spaces alone allows localization and caption queries to naturally learn distinct attention patterns (broad vs. dense). The design is remarkably clean.
- **The GT-aware adaptive design of OSL** elegantly avoids the over-constraint of "penalizing all overlap," using the $\alpha$ weight to balance "reducing redundancy" and "preserving correct detections."
- **Concept Guider adds no inference overhead**: It acts only as an auxiliary training task to guide query learning and is removed at inference, exemplifying a "training augmentation with zero inference cost" paradigm transferable to other generative tasks.

## Limitations & Future Work
- The captioning head uses LSTM, which is weaker than GPT-2/LLM-based methods (e.g., DDVC uses GPT-2) for generating long descriptions; replacing it with a stronger language model could yield further improvements.
- The concept vocabulary ($N_c=30$) is small and derived from training set statistics, potentially insufficient for open-vocabulary scenarios.
- Validation is limited to YouCook2 (cooking) and ActivityNet (human activities); generalization to other video types (e.g., Ego4D, movie understanding) is not verified.
- OSL is based on temporal IoU and assumes events are contiguous temporal segments, making it unsuitable for multi-label or hierarchical event settings.
- Inference speed is not reported; the two-group query decoder theoretically doubles the computation compared to the baseline.

## Related Work & Insights
- **vs. PDVC**: PDVC first applied DETR to DVC with shared queries; this work separates queries and adds explicit loss constraints on that foundation, achieving CIDEr +9.49 and F1 +5.22—highly significant improvements.
- **vs. DDVC**: DDVC derives caption queries from localization queries via MLP, retaining an inherent dependency; the fully independent initialization in this work is more thorough and does not rely on GPT-2.
- **vs. CM2/MCCL**: These methods enhance captioning via external memory banks, increasing system complexity; this work achieves comparable or superior results through internal Concept Guider and role-specific queries, with greater simplicity.
- **Transferability**: The GT-aware overlap penalization mechanism of OSL can be directly transferred to temporal action detection, moment retrieval, and other temporal grounding tasks to address proposal redundancy.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core idea (query separation + overlap loss) is intuitively clear and elegantly designed, though the technical complexity of individual components is modest.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual benchmarks + comprehensive ablation (individual components, combinations, hyperparameters, query count) + qualitative analysis; highly thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and figures are intuitive (the attention comparison in Fig. 1 is particularly convincing), though the Related Work section is slightly verbose.
- **Value**: ⭐⭐⭐⭐ Provides a clear improvement paradigm for query-based DVC; OSL is transferable to other temporal tasks.
- **Value**: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[AAAI 2026\] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs](../../AAAI2026/video_understanding/predicting_video_slot_attention_queries_from_random_slot-feature_pairs.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
