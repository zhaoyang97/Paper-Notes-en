---
title: >-
  [Paper Note] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection
description: >-
  [ACL 2026][AIGC Detection][LLM-generated text detection] Proposes RACE (Rhetorical Analysis for Creator-Editor Modeling), which utilizes Rhetorical Structure Theory (RST) to construct logic graphs for modeling the "creat…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "LLM-generated text detection"
  - "Rhetorical Structure Theory"
  - "Creator-Editor modeling"
  - "fine-grained classification"
  - "discourse analysis"
date: 2026-05-08
content_hash: cb63850b7ade4ce1
---

# Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection

**Conference**: ACL 2026  
**arXiv**: [2604.04932](https://arxiv.org/abs/2604.04932)  
**Code**: [https://race.yang-li.cn](https://race.yang-li.cn)  
**Area**: AIGC Detection  
**Keywords**: LLM-generated text detection, Rhetorical Structure Theory, Creator-Editor modeling, fine-grained classification, discourse analysis

## TL;DR
Proposes RACE (Rhetorical Analysis for Creator-Editor Modeling), which utilizes Rhetorical Structure Theory (RST) to construct logic graphs for modeling the "creator's" cognitive architecture while extracting discourse unit-level features to capture the "editor's" linguistic style, achieving fine-grained four-way detection (Human-written, LLM-written, LLM-polished human text, and Human-revised LLM text).

## Background & Motivation

**Background**: LLM-generated text detection primarily focuses on binary classification (Human vs. LLM). Recently, some works have introduced "mixed text" as a third category for ternary classification settings.

**Limitations of Prior Work**: Even ternary classification is insufficiently granular—"LLM-polished human text" and "Human-revised LLM text" lead to vastly different policy consequences in practical regulation. The former is typically viewed as legitimate writing assistance, while the latter is considered a deceptive act to bypass detection. However, both belong to "mixed text," and traditional methods fail to distinguish them using unified features.

**Key Challenge**: The Creator-Editor collaboration patterns differ fundamentally between the two mixed types: LLM-polished human text = human logical framework + LLM expressive style; Human-revised LLM text = LLM logical framework + human expressive perturbations. Unified features struggle to capture these separated dual traces.

**Goal**: To design a detection framework capable of separately modeling the contributions of the "creator" and the "editor" to achieve reliable fine-grained four-way detection.

**Key Insight**: The creator's identity is deeply embedded in the logical organization and argumentative progression of the text (humans use hierarchical reasoning, while LLMs tend toward flat exposition), whereas the editor's influence is primarily reflected in surface linguistic expression. RST is perfectly suited to decouple these two layers.

**Core Idea**: Use RST to parse the text into a rhetorical relation tree, transforming it into a logic graph to characterize the creator's mental fingerprint, while using EDU-level (Elementary Discourse Unit) semantic representations to capture the editor's linguistic style.

## Method

### Overall Architecture
The RACE pipeline: Original text → RST parsing to obtain EDU sequences and rhetorical relation trees → Construction of multi-relational logic graphs (EDUs as leaf nodes, rhetorical relations as internal nodes) → Node feature initialization (descendant span pooling + information bottleneck projection) → Rhetoric-Guided Message Passing (RGCN) → Root Pooling for global representation → Classification.

### Key Designs

1. **Dual Trace Extraction**:
    - **Function**: Separating creator and editor traces from the text.
    - **Mechanism**: An end-to-end RST parser is used to parse text into a binary elective tree. Leaf nodes comprise the EDU sequence (representing the editor's linguistic units), while internal nodes carry rhetorical relation labels (e.g., Elaboration, Contrast), representing the creator's logical organization.
    - **Design Motivation**: Statistical analysis reveals human creators significantly over-express Attribution and Background relations (citing sources, establishing context), whereas LLM creators over-express Elaboration and Evaluation (flattening information). Even after editing, these structural fingerprints persist—cosine similarity shows that texts by the same creator remain consistently closer in rhetorical relation frequency ($>0.89$).

2. **Logic-Aware Graph Initialization**:
    - **Function**: Transforming the RST tree into a learnable multi-relational graph.
    - **Mechanism**: Constructs a graph $\mathcal{G} = (\mathcal{V}_{edu} \cup \mathcal{V}_{rel}, \mathcal{E}, \mathcal{R})$. EDU nodes are initialized with MeanPool embeddings from a PLM. Relation nodes are initialized recursively using Descendant Span Pooling of all descendant EDU semantic centroids. Dimensions are then reduced to $d_{feat}$ via information bottleneck projection to filter surface noise.
    - **Design Motivation**: One-hot encoding for relation labels is too sparse; initializing relation nodes with descendant EDU semantic centroids injects richer contextual information.

3. **Rhetoric-Guided Message Passing**:
    - **Function**: Learning deep representations of human/LLM creation differences on rhetorical graphs.
    - **Mechanism**: Employs $L$ layers of RGCN, learning independent transformation matrices for each rhetorical relation. To prevent parameter explosion, basis decomposition regularization is used: $\mathbf{W}_r^{(l)} = \sum_{k=1}^B \alpha_{rk}^{(l)} \mathbf{V}_k^{(l)}$, sharing $B$ basis matrices. Global text representation is obtained via root node pooling.
    - **Design Motivation**: Different rhetorical relations carry distinct logical functions (Causality vs. Contrast vs. Elaboration) and require relation-specific propagation rules.

### Loss & Training
The joint loss is $\mathcal{L}_{total} = \mathcal{L}_{con} + \mathcal{L}_{ce}$: Supervised contrastive loss encourages compact intra-class clustering, while cross-entropy loss performs classification. The backbone uses RoBERTa-base with only the final layer fine-tuned.

## Key Experimental Results

### Main Results
Four-way detection on the HART dataset.

| Method | AUROC (Avg) | TPR@1%FPR |
|------|------------|-----------|
| RoBERTa | ~85 | 68.06 |
| CoCo | ~86 | - |
| DeTeCtive | ~87 | - |
| **RACE** | **~92** | **~80** |

### Ablation Study

| Configuration | AUROC | Description |
|------|-------|------|
| Full RACE | **~92** | Complete model |
| w/o RST graph (EDU only) | ~87 | Removes creator modeling, drops 5 points |
| w/o contrastive loss | ~90 | Feature space is less compact |
| w/o basis decomposition | ~91 | Overfitting on sparse relations |

### Key Findings
- RACE achieves the highest AUROC among 12 baselines and maintains high recall at a low false positive rate (1% FPR).
- Creator modeling (RST graph) provides the largest contribution—dropping 5 points when removed.
- Rhetorical relation frequency analysis validates the core hypothesis: human-written text has deeper and more complex RST structures, while LLM text is flatter.
- The cosine similarity of rhetorical relation frequencies for the same creator remains $>0.89$ after editing, proving that editing hardly alters the creator's fingerprint.

## Highlights & Insights
- The **Creator-Editor dual-role framework** is conceptually powerful—elevating the question from "who was the final actor" to "who established the logical framework vs. who performed surface modification."
- The discovery of **RST as a creator fingerprint** is compelling—human rhetorical structures are deeper with more citation/background relations, while LLMs prefer flat elaboration/evaluation structures.
- The introduction of the low false positive rate metric (**TPR@1%FPR**) is practical—in high-stakes scenarios like academic integrity detection, avoiding false accusations is more critical than simply increasing recall.

## Limitations & Future Work
- Dependency on the quality of the RST parser; current parsers may be inaccurate for certain text types.
- Evaluation was limited to the HART dataset; cross-domain generalization remains unknown.
- The four-way setting assumes text undergoes only one round of editing; multi-turn human-LLM interaction scenarios are more complex.
- As LLM capabilities improve, their rhetorical structures may increasingly resemble those of humans.

## Related Work & Insights
- **vs DetectAIve**: Also attempted four-way classification but used unified features; RACE is more precise with dual-role modeling.
- **vs CoCo**: CoCo considers discourse information but does not utilize the hierarchical structure of RST.
- **vs LF-Motifs**: Uses word frequency patterns for detection, failing to capture differences at the logical organizational level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Creator-Editor framework + RST creator fingerprint is a highly original design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 baselines are sufficient, though the single dataset limits generalization validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation analysis is extremely persuasive.
- Value: ⭐⭐⭐⭐⭐ First to bring fine-grained four-way detection to a practical level of performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](../../CVPR2026/aigc_detection/fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)
- [\[ACL 2026\] ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability](exagpt_example-based_machine-generated_text_detection_for_human_interpretability.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)

</div>

<!-- RELATED:END -->
