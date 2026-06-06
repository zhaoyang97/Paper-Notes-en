---
title: >-
  [Paper Note] Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance
description: >-
  [ICML 2026][Reinforcement Learning][LVLM] This paper proposes a dual suite for multimodal table reasoning: DiSCo for the training phase…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "LVLM"
  - "Table Image Understanding"
  - "Structure-Content Disentanglement"
  - "Global-to-Local Reasoning"
  - "Sub-table Evidence"
date: 2026-05-08
content_hash: ff026cb35b6f1c63
---

# Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance

**Conference**: ICML 2026  
**arXiv**: [2602.03491](https://arxiv.org/abs/2602.03491)  
**Code**: https://github.com/AAAndy-Zhu/TableVLM  
**Area**: Multimodal VLM / Table Reasoning / Representation Disentanglement / Training-free Inference  
**Keywords**: LVLM, Table Image Understanding, Structure-Content Disentanglement, Global-to-Local Reasoning, Sub-table Evidence  

## TL;DR
This paper proposes a dual suite for multimodal table reasoning: DiSCo for the training phase, which decouples the "skeleton" and "flesh" alignment targets through structure anonymization, allowing the LVLM to learn layouts using only 10K table images; and Table-GLS for the inference phase, which compresses whole-image QA into a minimal verifiable sub-table through a three-step process of "global structure exploration $\rightarrow$ self-refined sub-table extraction $\rightarrow$ evidence-grounded reasoning." The entire system requires no fine-tuning on reasoning data and no external tools, outperforming SFT/RL baselines relying on 82K-97K annotations across 21 benchmarks.

## Background & Motivation

**Background**: Adapting LVLMs for table reasoning currently follows two main paths: first, large-scale SFT or GRPO reinforcement learning (e.g., Table-LLaVA, Table-R1, TURBO), which injects HTML/Markdown/LaTeX strings into the model using hundreds of thousands of table images; second, connecting to external tools (e.g., ReFocus), relying on visual editors and code control for multi-hop reasoning.

**Limitations of Prior Work**: The SFT route requires expensive table reasoning annotations and can trigger catastrophic forgetting of original general reasoning capabilities. The external tool route increases inference latency and system complexity without truly enhancing the model's own structural understanding. Both paths couple the table's **structure (row-column layout, header hierarchy)** and **content (cell semantics)** within the same linearized sequence for learning, forcing the model to simultaneously memorize two entangled signals, resulting in poor cross-layout generalization and low sample efficiency.

**Key Challenge**: Serialized representations like HTML/Markdown naturally interleave structural tokens (`<tr>`, `|`, header tags) and content tokens (cell semantics) into a single long sequence. When the model tries to learn both types of information using the same objective, structural signals are submerged by massive content tokens. Conversely, content understanding depends on a structural skeleton that has not yet been learned, forming a "chicken and egg" mutual barrier.

**Goal**: (1) Use minimal alignment data to let the LVLM learn generalizable table structure representations; (2) Enable the model to robustly answer questions about tables with dense layouts during inference without any additional training or tool calls.

**Key Insight**: The authors observe that the text-semantic reasoning ability of LVLMs is already strong; what is missing is the "table structure" as an independent dimension. If structure learning and content learning can be decoupled—using anonymized "skeleton" tables for structure and using global/local structural coordinates as anchors for content—the model's existing semantic capabilities can be "grafted" onto the structural skeleton. Inference mimics this decoupling: first looking at the skeleton to locate rows and columns, then extracting a small sub-table for evidence reasoning.

**Core Idea**: The "skeleton-flesh" disentanglement is implemented throughout both training (DiSCo dual-path alignment) and inference (Table-GLS three-stage chain), making table capability a "plug-in" module rather than an end-to-end forced injection.

## Method

### Overall Architecture
In the training phase, DiSCo uses 10K table images to simultaneously construct **structure alignment samples** $(I_S,V)\to T_S$ (anonymized HTML/Markdown/LaTeX with cell content replaced by a placeholder $t_p$) and **content alignment samples**—global semi-structured summaries $T_G$ of the form "$M$ rows and $N$ columns, column $m$ describes X" and local cell semantics $T_L$ of the form "Row $m$ Column $n$: [content]". These three targets are jointly fine-tuned using LoRA. In the inference phase, Table-GLS splits single-step QA into three steps: first, the LVLM examines the whole image to provide relevant row/column indices $R,C$ and a reasoning draft $T_t$; second, it self-checks if $R,C$ are sufficient and extracts the minimal interpretable sub-table $T_{sub}$; finally, it performs evidence-grounded reasoning on $T_{sub}$ (retaining the original image as a visual anchor) to output $\hat{y}$. The entire pipeline requires no reasoning-specific annotations or external tools; structural capability comes from DiSCo, while reasoning capability comes from the base LVLM itself.

### Key Designs

1.  **DiSCo Structure Alignment (Skeleton)**:
    - **Function**: Enables the LVLM to separate and learn table layouts (row-column separation, header hierarchies, merged cell spans) from content.
    - **Mechanism**: Anonymizes regular serialized tables $T$ as $T_S=\texttt{Anonymize}(T,t_p)$ by replacing all cell content with a uniform placeholder $t_p$. The training objective $\mathcal{L}_{\text{struct}}=-\mathbb{E}\log P_\theta(T_S\mid I_S,V)$ forces the model to rely solely on visual layout cues in the image (lines, header positions) to predict the skeleton.
    - **Design Motivation**: In traditional HTML alignment, "structure tokens ≪ content tokens," causing structural signals to be submerged. Anonymization zeroes out content signals, forcing the model to learn only the layout, which provides independent supervision for structural capability and improves generalization to unseen merged cells and nested headers (OOD TSD/TCE show the most significant gains here).

2.  **DiSCo Content Alignment (Flesh)**:
    - **Function**: Grafts cell semantics onto the existing structural skeleton, forcing the model to treat "Row $m$ Column $n$" as a coordinate system rather than treating content as a free-form text stream.
    - **Mechanism**: Divided into two layers. **Global**: The model outputs a semi-structured summary $T_G$ (number of rows, columns, and the meaning of each row/column), with loss $\mathcal{L}_{\text{content\_global}}=-\mathbb{E}\log P_\theta(T_G\mid I_G,V)$. **Local**: Given row number $m$ and column number $n$, the model outputs "Row $m$ Column $n$: [content]", with loss $\mathcal{L}_{\text{content\_local}}=-\mathbb{E}\log P_\theta(T_L\mid I_L,V,m,n)$.
    - **Design Motivation**: In traditional HTML alignment, semantics and positions are tied together in a sequence, preventing explicit queries for a specific cell. DiSCo forces content to be anchored to learned structural coordinates, reusing the LVLM's native semantic capabilities while making "Query Row $m$ Column $n$" a native operation—the exact minimal interface required for sub-table extraction in Table-GLS.

3.  **Table-GLS Global-to-Local Reasoning**:
    - **Function**: Transforms whole-table QA from "end-to-end forced reading" into an interpretable chain of "locate skeleton $\to$ extract evidence $\to$ final reasoning," without extra training or tools.
    - **Mechanism**: Phase I (Global Structure Exploration) uses prompt $I_{GSE}$ to drive the model to output a reasoning draft $T_t$ and relevant row/column labels $R, C$, forcing a "where-to-look" decision. Phase II (Self-refined Sub-table Extraction) uses $I_{SSE}$ for the model to self-check if $R,C$ are sufficient and necessary, correcting them if needed, and then extracts the semi-structured sub-table $T_{sub}$. This "plan-before-extract" approach prevents global misalignment from propagating. Phase III (Evidence-grounded Reasoning) allows the model to generate the answer $\hat{y}=\text{LVLM}(I_{EGR},T_{sub},V,q)$ based on $T_{sub}$.
    - **Design Motivation**: Direct QA on entire table images often allows models to take shortcuts via global pattern matching (answering correctly while focusing on irrelevant rows). Explicitly splitting "evidence selection vs. answer derivation" reduces spurious correlations and leaves an explicit reasoning trace for diagnosing OOD errors. The self-reflection step is key—ablation shows that removing SSE drops AIT-QA performance from 76.71 to 73.39.

### Loss & Training
The total DiSCo loss is $\mathcal{L}_{\text{DiSCo}}=\mathcal{L}_{\text{struct}}+\mathcal{L}_{\text{content\_global}}+\mathcal{L}_{\text{content\_local}}$. All LVLMs (Gemma3-12B, Gemma3n-E4B, LLaVA-v1.6-7B, Qwen3-VL-8B/4B/32B) are fine-tuned using LoRA to preserve original reasoning capabilities. Table-GLS is entirely training-free, using vLLM to run the three-stage prompts in a zero-shot setting. The components are orthogonal: DiSCo enhances representation while Table-GLS enhances the reasoning process.

## Key Experimental Results

### Main Results
On 21 table understanding + reasoning benchmarks, with an alignment budget of only 10K table images vs. the 82K-97K of baselines, all evaluated in a zero-shot setting:

| Task Cluster | Configuration | Key Metric | Ours (Qwen3-VL-8B) | Textual (10K) | Textual-All (97K) |
|--------|------|----------|---------------------|----------------|---------------------|
| Table Understanding TSD/TCE/TCL/RCE/MCD | Avg (Seen structures) | accuracy | **DiSCo 42.9-93.5** | 41.0-89.6 | 37.7-89.8 |
| OOD Table Understanding (Unseen layouts) | OOD TSD/TCE/TCL/RCE | accuracy | **DiSCo 65.5-88.4** | 44.8-82.0 | 50.1-86.1 |
| Table Reasoning 8 tasks (HiTab/AIT-QA/InfoTabs etc.) | Full = DiSCo + Table-GLS | avg | **Significantly > GPT-4o-mini & Table-LLaVA-13B** | – | – |

On Qwen3-VL-32B, DiSCo improved OOD TCL from 65.91 to 74.10 and OOD RCE Column from 84.16 to 88.40. On the small Gemma3n-E4B model, OOD TCL rose from 9.00 $\rightarrow$ 14.32 (textual alignment only reached 10.20), indicating that structure-content decoupling is particularly effective in preventing OOD collapse in smaller models.

### Ablation Study
Qwen3-VL-8B + four representative reasoning tasks (Full = DiSCo + Table-GLS):

| Configuration | HiTab | AIT-QA(O) | InfoTabs | PubHealthTab(O) |
|------|-------|-----------|----------|------------------|
| **Full** | 27.35 | **76.71** | **72.67** | **77.14** |
| − GSE (Remove Global Exploration) | 24.30 | 62.82 | 72.09 | 74.92 |
| − SSE (Remove Self-refinement) | **31.41** | 73.39 | 70.20 | 73.94 |
| only Table-GLS (No DiSCo) | 29.76 | 55.58 | 73.59 | 72.76 |
| CoT | 28.17 | 56.75 | 67.98 | 57.52 |
| DiSCo + CoT | 26.40 | 73.78 | 71.00 | 68.33 |
| RoT (row-of-thought) | 33.88 | 55.58 | 61.26 | 58.29 |
| DiSCo + RoT | 26.27 | 69.08 | 66.98 | 72.14 |

### Key Findings
- **Structural Decoupling is Key for OOD**: DiSCo's gains are significantly larger in OOD tasks than in-domain. Textual alignment actually degrades performance on tasks like OOD TCL, suggesting that interleaving structure and content leads to overfitting to specific training set layouts.
- **GSE is a Lifesaver for OOD**: Removing Global Structure Exploration caused the AIT-QA (OOD) score to drop from 76.71 $\rightarrow$ 62.82 (nearly 14 points), while the in-domain HiTab score rose to 31.41. This suggests GSE sacrifices some in-domain speed for strong generalization.
- **DiSCo and Table-GLS Must Team Up**: Table-GLS alone (without DiSCo) reached only 55.58 on AIT-QA. DiSCo + CoT/RoT were also inferior to Full, verifying that representation decoupling in training and path decoupling in inference are complementary.
- **Small Supervision, Large Effect**: Using only 10K images outperformed Textual-All (97K) and SFT-based Table-LLaVA (82K), showing that structure-content decoupling sets a new level of sample efficiency.

## Highlights & Insights
- **"Skeleton/Flesh" as an Elegant Inductive Bias**: Explicitly separating "what the layout is" from "what it contains" prevents alignment tokens from overriding each other. This idea could be transferred to other coupled domains like code (AST structure vs. identifier semantics), UI (layout vs. copy), or chemistry (molecular skeletons vs. substituents).
- **Symmetric Disentanglement Philosophy**: DiSCo decouples representation during training, while Table-GLS decouples "finding evidence vs. using evidence" during inference. This "learned during training $\to$ used during inference" closed loop avoids the common issue where inference prompts pull a generalized model back into a single mode.
- **Plan-before-extract Self-reflection**: Asking the model "Are these rows/columns sufficient?" in the SSE stage is a low-cost but significant engineering trick—valuable as a general template for LVLM agents.

## Limitations & Future Work
- Evaluation is primarily on static table images; joint reasoning for tables embedded in long documents with peripheral text (e.g., full scientific PDFs) is not covered. Structure alignment does not yet consider higher-order table-image-text layouts.
- DiSCo requires HTML/Markdown/LaTeX ground truth for training tables to construct anonymized samples, making it difficult to apply to raw scans or low-quality OCR tables.
- Table-GLS requires three LVLM calls per question, increasing inference latency by approximately 2-3x. For simple questions, the full pipeline is inefficient; an early-exit classifier could be added.
- Lack of a second-chance mechanism for evidence retrieval—SSE runs once; if initial $R, C$ values are far off, correction is limited.

## Related Work & Insights
- **vs. Table-LLaVA / TabPedia / SynTab**: These rely on large-scale SFT (82K-97K) to learn serialized tables with coupled structure/content. DiSCo uses 10K images and achieves comparable in-domain and superior OOD performance without fine-tuning on downstream reasoning.
- **vs. Table-R1 / TURBO / R3V**: These use GRPO/RL for reasoning trajectories but require reasoning-specific reward signals. Table-GLS makes reasoning explicit through three-stage prompting without modifying weights, avoiding capability drift from RL.
- **vs. ReFocus (External Tools)**: ReFocus uses code for multi-hop editing. Table-GLS folds "multi-hop" into self-refined sub-table extraction, requiring no external tool stack and lowering deployment barriers.
- **vs. General CoT/RoT**: Results show DiSCo + Table-GLS outperforms DiSCo + CoT/RoT, proving that structuring the reasoning process into a "global $\to$ local $\to$ evidence" chain is better suited for highly structured inputs like tables.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The philosophy of training-inference dual disentanglement is clear, and cell anonymization for structure alignment is a simple but previously un-systematized trick.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 21 benchmarks + 4 backbones (4B/8B/12B/32B) + in-domain/OOD perspectives + thorough ablation, with open-sourced code and data.
- **Writing Quality**: ⭐⭐⭐⭐ Framework diagrams and three-stage formulas (Eq. 5-7) are clear, and the structure echoes the "Skeleton/Flesh" theme; however, Table 1 is very dense.
- **Value**: ⭐⭐⭐⭐⭐ Reduces sample efficiency requirements for table reasoning to 10K, and the Table-GLS template can be applied to any existing LVLM at zero cost, making it industry-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](../../ICLR2026/reinforcement_learning/metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](../../ICLR2026/reinforcement_learning/fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](../../ICLR2026/reinforcement_learning/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICML 2026\] Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)

</div>

<!-- RELATED:END -->
