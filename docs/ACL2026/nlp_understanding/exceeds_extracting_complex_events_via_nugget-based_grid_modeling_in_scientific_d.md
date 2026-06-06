---
title: >-
  [Paper Note] EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain
description: >-
  [ACL 2026][NLP Understanding][Event Extraction] The authors identified that scientific literature abstracts present specific EE challenges: **high information density** (5.54 events + 12.82 arguments per 100 tokens) and…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Event Extraction"
  - "Document-level"
  - "Word-Word Relation Grid"
  - "Scientific Literature"
  - "Hierarchical Events"
date: 2026-05-08
content_hash: b9e3981a9c78753d
---

# EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain

**Conference**: ACL 2026  
**arXiv**: [2406.14075](https://arxiv.org/abs/2406.14075)  
**Code**: https://github.com/HammerScholar/EXCEEDS  
**Area**: NLP Understanding / Event Extraction / Information Extraction  
**Keywords**: Event Extraction, Document-level, Word-Word Relation Grid, Scientific Literature, Hierarchical Events

## TL;DR
The authors identified that scientific literature abstracts present specific EE challenges: **high information density** (5.54 events + 12.82 arguments per 100 tokens) and **complex event structures** (overlapping/discontinuous/reverse-order nuggets + sub-events) that traditional EE datasets lack. To address this, they (a) annotated the SciEvents dataset with 2,508 documents and 24,381 events, and (b) proposed EXCEEDS—an end-to-end framework that reformulates EE as multi-label relation classification on an $l \times l$ word-word grid. Using three types of edges (HTL/THL/EAL) to unify the modeling of triggers, arguments, and sub-events, the model outperforms 9 SOTA baselines on main metrics and complex scenarios.

## Background & Motivation
**Background**: Event Extraction (EE) is typically decomposed into Event Detection (ED) and Event Argument Extraction (EAE). Leading approaches include global joint extraction (OneIE), discriminative token classification (PAIE/Tagprime), and generative methods (DEGREE/KnowCoder). F1 scores on existing benchmarks like ACE05, RAMS, and Genia have reached high levels.

**Limitations of Prior Work**: Upon detailed statistical analysis of information density and complex morphology across 9 domain-specific datasets, the authors found two overlooked facts: (1) Scientific text (abstracts) has much higher density than news, legal, or cyber domains—SciEvents contains 5.54 events and 39.49 nugget tokens per 100 tokens, over 3 times that of ACE05 (1.80 events); (2) In scientific text, 33.70% are overlapping nuggets, 25.63% are sub-events, 3.08% are discontinuous nuggets, and 1.01% are reverse-order nuggets—while most existing datasets only annotate continuous nuggets.

**Key Challenge**: Two modeling assumptions of existing EE methods are violated in the scientific domain: (a) Most methods assume non-hierarchical structures (no sub-events) and local contexts (sentence-level), but triggers in scientific abstracts often link to arguments over long distances, and hierarchical trigger-of-trigger sub-event relations are pervasive; (b) Discriminative methods relying on span start/end offsets cannot represent discontinuous or reverse-order nuggets.

**Goal**: (1) Construct a scientific EE dataset capable of evaluating high density and complex structures; (2) Design a method that handles overlapping, discontinuous, reverse-order nuggets, and hierarchical sub-events within a **single end-to-end** framework.

**Key Insight**: The authors draw inspiration from word-word relation grids used in NER (e.g., W2NER). Since span boundary representations fail with complex morphologies, the task is returned to token-pair relations, where "which two tokens are in one nugget" and "which nugget is an argument for another" are unified as multi-label relation predictions on a grid.

**Core Idea**: Simplify EE into "nugget-based relation classification on an $l \times l$ grid"—using HTL (Head-Tail-Link) to connect tokens within a nugget, THL (Tail-Head-Link) with types to close the nugget, and EAL (Event-Argument-Link) to connect trigger heads to argument heads. This structure can encode all complex nugget morphologies and naturally represent hierarchical sub-events.

## Method

### Overall Architecture
The EXCEEDS pipeline proceeds as: Input document $D = \{x_1, \dots, x_l\}$ → RoBERTa-large encoding → BiLSTM for sequential dependencies → Conditional Layer Normalization (CLN) for context adaptation to obtain $\mathbf{H} \in \mathbb{R}^{l \times d}$ → Construct pair-wise grid $\mathbf{G} \in \mathbb{R}^{l \times l \times C_g}$ (each cell is an MLP projection of $[\mathbf{h}_i; \mathbf{h}_j; \mathbf{d}_{i,j}]$, where $\mathbf{d}_{i,j}$ is a relative distance embedding) → $K=2$ layers of 2D Convolutional Residual Grid Refiner for local information aggregation → Linear classification head outputting $\mathbf{Y} \in \mathbb{R}^{l \times l \times |R|}$ → Multi-label zero-threshold binarization → Extraction of the event set using Algorithm 1: first backtrack nugget chains via DFS along HTL edges (requiring a THL-type edge for closure), determine trigger/argument types via THL-type, and finally link arguments to triggers using EAL edges and ontology constraints.

### Key Designs

1.  **Word-Word Event Grid (HTL + THL + EAL edges)**:
    *   **Function**: Encodes all internal nugget structures, nugget types, and cross-nugget trigger-argument/trigger-trigger relations into a single $l \times l$ multi-label relation grid.
    *   **Mechanism**: Each cell $G[i,j]$ stores the relation type $r \in R$ between token pair $(x_i, x_j)$. Three edge types are used: HTL (Head-Tail-Link) marks the sequence of adjacent tokens within a nugget (e.g., $x_i$ followed by $x_j$), naturally supporting discontinuous (skipping non-HTL tokens) and reverse-order (HTL direction is not strictly left-to-right) nuggets. THL (Tail-Head-Link) points from the last token back to the first, with the edge label representing the semantic type (trigger or argument type), serving both to close the nugget and assign a type. EAL (Event-Argument-Link) connects trigger head tokens to argument head tokens; sub-events are represented directly via trigger-to-trigger EALs.
    *   **Design Motivation**: Traditional BIO or span boundary representations assume nuggets are continuous left-to-right segments. These fail for overlapping (one token in two nuggets), discontinuous (interspersed stop words), or reverse-order structures. Grid representation treats "token pairs" as the atomic unit, reducing complex structures into sets of edges on a graph, making them end-to-end learnable.

2.  **CLN + Distance Embeddings + 2D CNN Grid Refiner**:
    *   **Function**: Transforms isolated token-pair representations into mutually aware grid features through local propagation, improving the separability of complex structures.
    *   **Mechanism**: (a) Uses CLN to re-normalize $\mathbf{H}$: $\mathbf{H} = \text{MLP}_\gamma(\mathbf{L}) \odot \frac{\mathbf{L} - \mu}{\sigma + \epsilon} + \text{MLP}_\beta(\mathbf{L})$, making affine parameters context-adaptive; (b) Concatenates relative distance embeddings $\mathbf{d}_{i,j}$ to inject positional signals; (c) Applies $K=2$ residual 2D convolutional blocks $\mathbf{G}^{(k+1)} = \text{Norm}(\mathbf{G}^{(k)} + \mathcal{F}(\mathbf{G}^{(k)}))$ for local aggregation on the grid, allowing the kernel to capture patterns like "trigger-argument" relations which often appear in specific grid localities (e.g., near the diagonal).
    *   **Design Motivation**: A naive pair MLP treats each cell independently, losing patterns such as "multiple cells around a trigger activating together." 2D convolutions are efficient ($O(Kl^2)$) and inject spatial priors. Ablation showed a drop of 0.21% in EC and 0.76% in AC without the Grid Refiner.

3.  **Multi-label Zero-Threshold Loss + Heuristic Decoding**:
    *   **Function**: Uses a unified loss instead of a sigmoid chain when a token pair belongs to multiple relations (e.g., simultaneous HTL and EAL head); ensuring only structurally valid nuggets are generated during decoding.
    *   **Mechanism**: Training uses ZLPR multi-label cross-entropy:
        $$\mathcal{L}_{i,j} = \log(1 + \sum_{r \in \Omega^-} e^{y^r_{i,j}}) + \log(1 + \sum_{r \in \Omega^+} e^{-y^r_{i,j}})$$
        which automatically balances positive and negative labels. Inference applies binarization at $\mathbb{I}[y^r_{i,j} > 0]$ to obtain $\hat{\mathbf{M}}$. Decoding enforces two hard constraints: (i) HTL chains must be closed by a THL-type edge or be discarded; (ii) Arguments without a valid trigger link are discarded.
    *   **Design Motivation**: In complex nugget morphologies, one cell having multiple labels is common; binary sigmoid doesn't account for label interdependencies. ZLPR loss optimizes relative margins for all positive vs. negative instances and is differentiable at zero. Heuristic pruning prevents the DFS from generating exponential HTL chains due to model instability in early training.

### Loss & Training
A single multi-label ZLPR loss trains the entire grid classifier without phased pre-training or curriculum strategies. RoBERTa-large backbone lr=1e-5, other modules lr=1e-3, batch=2, epoch=20, BiLSTM hidden size 1024, grid channels $C_g=256$, refiner $K=2$, kernel=3, dropout=0.1. Validation was skipped in the first few epochs to prevent the DFS from exploding due to instability. Overall complexity is $O(l^2)$ dominated by grid construction, with memory $O(l^2 C_g + l^2 |R|)$.

## Key Experimental Results

### Main Results
Overall F1 scores (%) on SciEvents (TI=Trigger Identification, TC=Trigger Classification, AI/AC=Argument I/C, EC=Event Correlation/sub-event extraction), featuring representative baselines and EXCEEDS:

| Model | TI | TC | AI | AC | EC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OneIE (global) | 75.72 | 62.93 | 30.30 | 28.81 | 37.41 |
| EEQA (Generative) | 74.85 | 62.15 | 37.75 | 35.64 | 44.81 |
| PAIE† (Discriminative) | 73.27 | 63.03 | 43.92 | 42.06 | 47.17 |
| Tagprime (Discriminative) | 73.27 | 63.03 | 44.67 | 42.69 | 47.72 |
| BartGen† (Generative) | 73.27 | 63.03 | 39.85 | 37.81 | 42.75 |
| KnowCoder (LLM-based) | 69.88 | 52.02 | 35.24 | 33.43 | 34.54 |
| **EXCEEDS** | **75.29** | **63.74** | **44.97** | **43.20** | **48.25** |

EXCEEDS ranks first in TC, AI, AC, and EC. It follows OneIE closely in TI (0.43 difference). EAE metrics are +0.30~+0.53 absolute F1 higher than the runner-up Tagprime, and EC is +0.53 higher.

### Ablation Study
**Module Ablation + Complex Scenario Breakdown**:

| Configuration | TC | AC | EC | Description |
| :--- | :--- | :--- | :--- | :--- |
| EXCEEDS Full | 63.74 | 43.20 | 48.25 | Full Model |
| − Contextual encoding | 63.44 | 42.14 | 47.64 | Removes CLN/BiLSTM, largest AC drop (-1.06) |
| − Grid Refiner | 63.41 | 42.44 | 48.04 | Removes 2D CNN aggregation, AC -0.76 |

Complex scenario subsets (F1%, "-" denotes physical inability of baseline to support scenario):

| Model | Discontinuous AC | Overlapping TC | Overlapping AC | Reverse-order AC | Sub-event TC | Sub-event AC | Sub-event EC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Tagprime | – | 55.03 | 18.11 | – | 53.84 | 47.89 | 48.11 |
| PAIE | – | 49.62 | 13.18 | – | 53.66 | 47.34 | 49.08 |
| BartGen | 2.74 | 31.98 | 10.58 | 0.00 | 52.25 | 43.61 | 40.19 |
| KnowCoder | 0.00 | 26.18 | 6.93 | 0.00 | 42.36 | 34.81 | 40.33 |
| **EXCEEDS** | **13.86** | **62.46** | **22.46** | **7.27** | **55.13** | **48.32** | **51.15** |

### Key Findings
*   **Discriminative baselines completely fail with discontinuous and reverse-order nuggets**: Marked as "–" because these methods rely on span offset representations. EXCEEDS' grid relation representation is notably the only one capable of handling all scenarios.
*   **Generative models collapse on complex nuggets**: BartGen/DEGREE/KnowCoder's AC drops toward zero on overlapping nuggets because textual span generation cannot express "one token belonging to two nuggets." EXCEEDS achieves 22.46 (2-3x higher).
*   **CLN/BiLSTM are more critical than Grid Refiner**: Removing contextual modules yields a -1.06 drop in AC, compared to -0.76 for the Grid Refiner, suggesting the bottleneck lies in token representation quality.
*   **Error Analysis**: 89.2% of TI errors and 84.6% of AI errors are "missed" (false negatives) rather than boundary errors, indicating recall is the bottleneck in dense scientific contexts. Classification errors concentrate on semantically similar types (e.g., MDS vs WKS), suggesting a need for schema-aware representations.
*   **Overall AC remains low (43.20%)**: The authors acknowledge SciEvents is a difficult benchmark intended to challenge the research community.

## Highlights & Insights
*   **Elegant Extension of W2NER to EE**: Extending word-word relations to three edge types (intra-nugget, nugget type, and inter-nugget) elegantly represents structures using a single matrix that previously required two-stage pipelines. This "unified graph representation for all tasks" is an insightful paradigm for IE.
*   **Dual-purpose THL-type edges**: The tail→head edge simultaneously completes nugget closure and type assignment, merging the type classifier into the grid and reducing complexity and error propagation.
*   **Direct Sub-event Modeling**: Representing sub-events via trigger→trigger EAL edges avoids traditional multi-stage hierarchical pipelines. This is natural for "Evaluation method X uses dataset Y" nested patterns in scientific literature.
*   **Value of the SciEvents Dataset**: The rigorous effort involved (4 schema iterations, 7 annotators, three-tier quality control) resulting in a high 73% first-pass pass rate is significant. The density statistics provide a standard framework for cross-domain comparison.

## Limitations & Future Work
*   **Restricted to Abstracts**: SciEvents uses ACL abstracts only (2019-2022), missing full-text tables, formulas, and cross-section references where many events are fully detailed.
*   **Narrow Domain**: Data is limited to NLP; the generalizability of the schema to biomedicine, physics, or chemistry writing styles remains unverified.
*   **Complex Scenarios Unsolved**: F1 for reverse-order (7.27) and discontinuous (13.86) AC is far below continuous nuggets (43.20). While the grid can model these, the actual performance remains low.
*   **$O(l^2)$ Memory Bottleneck**: The grid size explodes with document length. The current implementation is limited to short abstracts and lacks chunking or sparsification.
*   **Directions**: (1) Incorporate schema-aware prompts or type embeddings; (2) Use sparse attention or grid tiling to support document-level inputs; (3) Expand SciEvents to multi-domain and multi-modal (tables + formulas) data; (4) Use LLMs for weak supervision to generate silver-standard labels.

## Related Work & Insights
*   **vs Tagprime (Strongest Discriminative Baseline)**: While Tagprime uses token-level sequence labeling and trigger embeddings for EAE (AC of 42.69), EXCEEDS (AC 43.20) demonstrates fundamental modeling superiority in complex scenarios (e.g., overlapping AC 22.46 vs 18.11).
*   **vs OneIE / Joint Extraction**: OneIE has a slight TI advantage (75.72 vs 75.29) by using entity information, but EXCEEDS achieves similar performance without requiring entity supervision, making it more versatile.
*   **vs PAIE / Tagprime / DEEIA (EAE-only)**: These pipeline methods accumulate error from external ED modules. EXCEEDS' end-to-end grid results in superior EC (48.25 vs 47.72).
*   **vs KnowCoder (LLM-based)**: LLaMA2-7B with LoRA significantly underperforms (AC 33.43), indicating that general LLM capability cannot yet replace structural modeling in specialized domains with complex EE requirements.
*   **Inspiration for Other Tasks**: The grid + multi-relational edge approach could be adapted for nested NER, coreference, or AMR parsing. The THL-type edge serves as a useful design pattern for joint structure-type labels.

## Rating
*   Novelty: ⭐⭐⭐⭐ Extending W2NER to EE and sub-events is a clear incremental contribution, though the underlying token-pair grid paradigm was established in NER.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 baselines across three genres, comprehensive complex-scenario evaluations, module ablations, and data statistics make this exhaustive.
*   Writing Quality: ⭐⭐⭐⭐ Figures 2/3 and Algorithm 1 are well-presented; however, the schema relies heavily on appendices, and the low performance in complex scenarios deserves more discussion in the main text.
*   Value: ⭐⭐⭐⭐⭐ SciEvents is a high-quality scientific EE benchmark (24k events), and EXCEEDS provides an end-to-end modeling paradigm essential for scientific knowledge graphs and automatic summarization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis](dimabsa_building_multilingual_and_multidomain_datasets_for_dimensional_aspect-ba.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)

</div>

<!-- RELATED:END -->
