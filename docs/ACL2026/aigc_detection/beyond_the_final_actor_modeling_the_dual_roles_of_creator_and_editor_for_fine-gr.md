---
title: >-
  [Paper Note] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection
description: >-
  [ACL 2026][AIGC Detection][LLM-generated text detection] This paper proposes RACE (Rhetorical Analysis for Creator-Editor Modeling), which leverages Rhetorical Structure Theory (RST) to construct logic graphs that model…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "LLM-generated text detection"
  - "Rhetorical Structure Theory"
  - "creator-editor modeling"
  - "fine-grained classification"
  - "discourse analysis"
date: 2026-05-08
content_hash: 6ac118563ca0233a
---

# Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection

**Conference**: ACL 2026  
**arXiv**: [2604.04932](https://arxiv.org/abs/2604.04932)  
**Code**: [https://race.yang-li.cn](https://race.yang-li.cn)  
**Area**: AIGC Detection  
**Keywords**: LLM-generated text detection, Rhetorical Structure Theory, creator-editor modeling, fine-grained classification, discourse analysis

## TL;DR
This paper proposes RACE (Rhetorical Analysis for Creator-Editor Modeling), which leverages Rhetorical Structure Theory (RST) to construct logic graphs that model the "creator's" cognitive architecture, while extracting discourse unit-level features to capture the "editor's" linguistic style, achieving fine-grained four-class LLM-generated text detection (human-written/LLM-written/LLM-polished human text/human-rewritten LLM text).

## Background & Motivation

**Background**: LLM-generated text detection primarily focuses on binary classification (human-written vs. LLM-written), with recent work introducing a third "mixed text" category for three-class settings.

**Limitations of Prior Work**: Even three-class classification is insufficiently fine-grained—"LLM-polished human text" and "human-rewritten LLM text" have completely different policy implications in practical governance. The former is typically viewed as legitimate writing assistance, while the latter represents evasion of detection and academic misconduct. However, both fall under "mixed text," making them indistinguishable using unified features in traditional methods.

**Key Challenge**: The creator-editor collaboration patterns differ fundamentally between these two hybrid types: LLM-polished human text = human logical framework + LLM expressive style; human-rewritten LLM text = LLM logical framework + human surface perturbations. Unified features struggle to capture these separated dual traces.

**Goal**: Design a detection framework capable of separately modeling "creator" and "editor" contributions to achieve reliable fine-grained four-class detection.

**Key Insight**: Creator identity is deeply embedded in the text's logical organization and argument progression (humans use hierarchical reasoning, LLMs tend toward linear narratives), while editor influence primarily manifests in surface linguistic expression. RST precisely separates these two levels.

**Core Idea**: Parse text using RST to obtain a rhetorical relation tree, convert it into a logic graph to characterize the creator's cognitive fingerprint, while using EDU-level semantic representations to capture the editor's linguistic style.

## Method

### Overall Architecture
RACE processing pipeline: Raw text → RST parsing yields EDU sequence and rhetorical relation tree → Construct multi-relational logic graph (EDUs as leaf nodes, rhetorical relations as internal nodes) → Node feature initialization (descendant span pooling + information bottleneck projection) → Rhetoric-Guided Message Passing (RGCN) → Root Pooling for global representation → Classification.

### Key Designs

1. **Dual Trace Extraction**:

    - Function: Separate creator and editor traces from text
    - Mechanism: Use end-to-end RST parser to parse text into binary constituency tree. Leaf nodes are EDU sequences (representing editor's linguistic units), internal nodes carry rhetorical relation labels (Elaboration, Contrast, etc.), representing creator's logical organization
    - Design Motivation: Statistical analysis shows human creators significantly over-express Attribution and Background relations (citing sources, establishing context), while LLM creators over-express Elaboration and Evaluation (linear information presentation). Even after editing, these structural fingerprints persist—cosine similarity shows texts from the same creator remain consistently closer in rhetorical relation frequency (>0.89)

2. **Logic-Aware Graph Initialization**:

    - Function: Transform RST tree into learnable multi-relational graph
    - Mechanism: Construct graph $\mathcal{G} = (\mathcal{V}_{edu} \cup \mathcal{V}_{rel}, \mathcal{E}, \mathcal{R})$. EDU nodes initialized with PLM MeanPool embeddings; relation nodes initialized via Descendant Span Pooling, recursively using semantic centroids of all descendant EDUs. Then dimensionality reduction to $d_{feat}$ via information bottleneck projection to filter surface noise
    - Design Motivation: Direct one-hot encoding of relation labels provides too sparse information; initializing relation nodes with descendant EDU semantic centroids injects richer contextual information

3. **Rhetoric-Guided Message Passing**:

    - Function: Learn deep representations of human/LLM creator differences on rhetorical relation graph
    - Mechanism: Use $L$-layer RGCN, learning independent transformation matrices for each rhetorical relation. To avoid parameter explosion, use basis decomposition regularization $\mathbf{W}_r^{(l)} = \sum_{k=1}^B \alpha_{rk}^{(l)} \mathbf{V}_k^{(l)}$, sharing $B$ basis matrices. Finally obtain global text representation through root node pooling
    - Design Motivation: Different rhetorical relations carry different logical functions (causal vs. contrastive vs. elaborative), requiring relation-specific propagation rules

### Loss & Training
Joint loss $\mathcal{L}_{total} = \mathcal{L}_{con} + \mathcal{L}_{ce}$: supervised contrastive loss encourages compact intra-class clustering + cross-entropy loss for classification. Backbone uses RoBERTa-base with fine-tuning only the final layer.

## Key Experimental Results

### Main Results
Four-class detection on HART dataset.

| Method | AUROC (Avg) | TPR@1%FPR |
|------|------------|-----------|
| RoBERTa | ~85 | 68.06 |
| CoCo | ~86 | - |
| DeTeCtive | ~87 | - |
| **RACE** | **~92** | **~80** |

### Ablation Study

| Config | AUROC | Note |
|------|-------|------|
| Full RACE | **~92** | full model |
| w/o RST graph (EDU only) | ~87 | Removing creator modeling drops 5 points |
| w/o contrastive loss | ~90 | Feature space insufficiently compact |
| w/o basis decomposition | ~91 | Sparse relations overfit |

### Key Findings
- RACE achieves highest AUROC among 12 baselines and maintains high recall at low false positive rate (1% FPR)
- Creator modeling (RST graph) contributes most significantly—removing it drops 5 points
- Rhetorical relation frequency analysis validates core hypothesis: human-written text has deeper, more complex RST structure; LLM text is flatter
- Texts from the same creator maintain rhetorical relation frequency cosine similarity >0.89 even after editing, proving editing cannot alter creator fingerprints

## Highlights & Insights
- **Creator-Editor dual-role framework** is conceptually clear and powerful—elevating "who was the final operator" to "who established the logical framework + who applied surface modifications"
- **RST as creator fingerprint** discovery is highly compelling—human rhetorical structures are deeper with more Attribution/Background relations; LLMs prefer flat Elaboration/Evaluation structures
- Introduction of low false positive rate metric (TPR@1%FPR) is practical—in high-stakes scenarios like academic integrity detection, reducing false accusations is more important than improving recall

## Limitations & Future Work
- Depends on RST parser quality; current parsers may be insufficiently accurate on certain text types
- Only evaluated on HART dataset; cross-domain generalization capability unknown
- Four-class setting assumes text undergoes only one editing round; multi-round human-LLM interaction scenarios are more complex
- As LLM capabilities improve, their rhetorical structures may increasingly resemble human writing

## Related Work & Insights
- **vs DetectAIve**: Also attempted four-class classification but used unified features; RACE's dual-role modeling is more fine-grained
- **vs CoCo**: CoCo also considers discourse information but doesn't utilize RST's hierarchical structure
- **vs LF-Motifs**: Uses word frequency patterns for detection, cannot capture differences at logical organization level

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Creator-Editor framework + RST creator fingerprint represents highly original design
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 baselines are comprehensive, but single dataset limits generalization validation
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation analysis is highly compelling
- Value: ⭐⭐⭐⭐⭐ First to achieve practical-level four-class fine-grained detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[CVPR 2026\] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](../../CVPR2026/aigc_detection/fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)
- [\[ACL 2026\] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)

</div>

<!-- RELATED:END -->
