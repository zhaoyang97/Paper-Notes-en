---
title: >-
  [Paper Note] On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation
description: >-
  [ICLR 2026][Information Retrieval & RAG][humor generation] This paper proposes HOMER, a framework that constructs a three-role LLM collaboration mechanism (conflicting script extractor + hierarchical imaginator + caption…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "humor generation"
  - "GTVH theory"
  - "script opposition"
  - "imagination tree"
  - "LLM collaboration"
  - "multi-role framework"
date: 2026-05-08
content_hash: 38648a4f61f7db45
---

# On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation

**Conference**: ICLR 2026
**arXiv**: [2602.06423](https://arxiv.org/abs/2602.06423)
**Code**: None
**Area**: Information Retrieval
**Keywords**: humor generation, GTVH theory, script opposition, imagination tree, LLM collaboration, multi-role framework

## TL;DR

This paper proposes HOMER, a framework that constructs a three-role LLM collaboration mechanism (conflicting script extractor + hierarchical imaginator + caption generator) grounded in the GTVH theory of verbal humor. By explicitly modeling script opposition, multi-perspective associative chains, and joke database retrieval to build an imagination tree for creative space expansion, HOMER achieves an average improvement of ~7% over baselines on the New Yorker cartoon benchmark using GPT-4o as the backbone, and significantly outperforms all baselines in human evaluation.

## Background & Motivation

**Background**: Multimodal humor generation is an important task for exploring whether LLMs possess human-level linguistic cognitive complexity. The representative task of funny caption generation requires simultaneous visual understanding, humor reasoning, creative imagination, and linguistic style expression — challenging even for humans.

**Limitations of Prior Work**: Existing methods primarily rely on (a) general prompting (HumorousAI, Phunny), (b) multi-hop reasoning with self-refinement (CLoT), or (c) task-specific fine-tuning (LoL). These approaches depend solely on the LLM's inherent humor mechanisms, capturing only surface-level humor language patterns without enabling deep humor logic reasoning or creative imagination.

**Key Challenge**: (a) LLMs' intrinsic humor generation capability is weak, as validated by multiple studies; (b) existing methods lack creativity — generated "humorous" captions are often merely descriptive; (c) they are uninterpretable — it is unclear where the punchline lies or why the output is funny.

**Key Insight**: This work draws on the classical linguistic humor theory GTVH (General Theory of Verbal Humor), modeling humor creation as the interaction of multiple knowledge resources — with **script opposition** at its core, wherein the collision of two semantic frames generates the punchline (expectation established → expectation violated → surprise/absurdity).

**Theoretical Foundation**: GTVH decomposes humor into five knowledge resources — Script Opposition, Situation, Target, Narrative Strategy, and Language — providing a natural structured guidance framework for humor caption generation.

**Motivating Example**: Consider a cartoon depicting an office meeting with an oversized coffee cup. GPT-4o and CLoT can only describe "meeting + caffeine," whereas HOMER identifies the script opposition of "giant cup vs. normal cup," follows the associative chain "coffee → milk → cow," and generates a caption such as "HR says we can expense a cow now" — conveying a deeper sense of absurdist humor.

## Method

### Overall Architecture

HOMER (Humor-theory-driven multi-role LLM collaboration framework augmented with humor Retrieval) comprises three collaborating LLM roles:

- **Conflicting Script Extractor** $\mathrm{Extract}(\cdot)$: extracts scene description $D$ and script opposition set $\mathcal{C}$ from the image
- **Hierarchical Imaginator** $\mathrm{Imagine}(\cdot)$: identifies key humor targets and expands the creative space via an imagination tree $\mathcal{T}_{\mathrm{im}}$
- **Caption Generator** $\mathrm{Gen}(\cdot)$: integrates all knowledge to generate a humorous caption

The overall pipeline is:

$$\mathrm{Extract}(I) \rightarrow (\mathcal{C}, D), \quad \mathrm{Imagine}(I, \mathcal{C}, D) \rightarrow \mathcal{T}_{\mathrm{im}}, \quad \mathrm{Cap}(I) = \mathrm{Gen}(\Phi(\mathcal{C}, D, \mathcal{T}_{\mathrm{im}}, \Omega))$$

where $\Omega \in NS \times LA$ controls narrative strategy and language style.

### Key Design 1: Conflicting Script Extractor

- **Function**: Extracts from input image $I$ a scene description $D$ (including locations, characters, expressions, actions, etc.) centered on script opposition, and a conflicting script set $\mathcal{C}$.
- **Mechanism**: The LLM first analyzes the scene to generate a description, then a prompt based on GTVH's definition of script opposition (the relationship between two conflicting or contrasting semantic frames) guides the model to extract all relevant conflicting scripts:

$$D = \mathrm{Extract}(I), \quad \mathcal{C} = \mathrm{Extract}(\Phi_{\mathrm{script}}(I, D))$$

- **Design Motivation**: Script opposition is the foundational logic of humor — without conflict, there is no punchline. Ablation studies confirm that removing $\mathcal{C}$ causes the largest performance drop (I+D+$\mathcal{T}_{\mathrm{im}}$ is the worst among all single-module ablation variants). Both $D$ and $\mathcal{C}$ together form the basis of the entire generation process.

### Key Design 2: Hierarchical Imaginator

- **Function**: Identifies candidate humor targets $\{t_i\}$, constructs an imagination tree $\mathcal{T}_{\mathrm{im}}$ through depth-oriented free association (LLM free-association) and breadth-oriented retrieval (joke database), and prunes it using humor relevance scores.
- **Mechanism**:

  (a) **Multi-perspective target identification**: Candidate humor targets are extracted from both local (fine-grained entities from scene description $D$) and global (coarse-grained entities from image $I$) perspectives, serving as the root nodes of the imagination tree.

  (b) **Depth imagination** (backbone chain): For each target $t_i$, the LLM first-order association function $f_{\mathrm{chain}}(\cdot)$ recursively generates an associative chain:

  $$e_{\tau+1}^{(i)} = f_{\mathrm{chain}}(e_{\tau}^{(i)}), \quad \tau = 0, \ldots, n-1$$

  The average chain length $\mathbb{E}[\tau] \approx 4$, enabling progressively deeper creative reasoning.

  (c) **Breadth imagination** (leaf node expansion): For each entity $e_{\tau}^{(i)}$ in the backbone chain, a query embedding $\mathbf{z}_q = f_{\mathrm{emb}}(D, \mathcal{C}, e_{\tau}^{(i)})$ is constructed, and top-$k$ relevant jokes are retrieved from an integrated corpus of 12 joke datasets; extracted tokens serve as leaf nodes.

  (d) **HOMER pruning**: Weakly relevant leaf nodes are filtered using humor relevance scores:

  $$\mathbf{H}(e_{\tau}^{(i)}, \varepsilon) = \mathbf{H}_{\mathrm{rel}}(e_{\tau}^{(i)}, \varepsilon) + \mathbf{H}_{\mathrm{freq}}(\varepsilon) + \mathbf{H}_{\mathrm{div}}(\varepsilon)$$

  where:
  - **Relevance-opposition score** $\mathbf{H}_{\mathrm{rel}}$: a joint function of WordNet-based Wu-Palmer semantic similarity TSS and conceptual opposition CO, balanced via a shaping function $f(x) = x\exp(-x)$: $\mathbf{H}_{\mathrm{rel}} = \mathrm{TSS} + f(\mathrm{TSS}) \cdot \mathrm{CO}$
  - **Humor frequency score** $\mathbf{H}_{\mathrm{freq}}$: geometric mean of token frequency and joke frequency
  - **POS diversity score** $\mathbf{H}_{\mathrm{div}}$: richness of part-of-speech tags (more POS tags → more wordplay opportunities)

- **Design Motivation**: Pure LLM free association tends to be repetitive and limited; retrieved jokes may be noisy. The combination of depth + breadth with subsequent pruning achieves both creative depth and humorous breadth, while filtering out irrelevant content.

### Key Design 3: Caption Generator

- **Function**: Samples creative paths from the imagination tree and generates humorous captions by integrating scene description, script opposition, and narrative strategy.
- **Mechanism**: A key conflicting script $C \in \mathcal{C}$ and humor target $t_i$ are randomly selected; DFS is applied to imagination tree $T_i$ to enumerate all root-to-leaf paths $\mathcal{P}_i$; one imagination path $P_i$ is sampled; and a prompt is constructed:

$$\mathrm{Cap}(I) = \mathrm{Gen}(\Phi(\mathcal{C}, D, P_i, \Omega))$$

- **Design Motivation**: By randomly sampling different paths and targets, diverse humorous captions can be generated (each sample yields a different creative chain), while the five GTVH knowledge resources provide structured constraints to ensure humor quality.

### Key Design 4: Mathematical Design of Humor Relevance Scoring

- **Function**: Quantifies the humor relevance between retrieved joke tokens and backbone entities.
- **Mechanism**: The relevance-opposition score leverages structured semantic relations from WordNet. Target semantic similarity TSS is quantified via Wu-Palmer similarity:

$$\mathrm{TSS}(s_{e_\tau}, s_\varepsilon) = \max_{s_{e_\tau} \in S_{e_\tau}, s_\varepsilon \in S_\varepsilon} \mathrm{Sim}_{\mathrm{wup}}(s_{e_\tau}, s_\varepsilon)$$

Conceptual opposition CO is quantified via Jaccard dissimilarity:

$$\mathrm{CO}(s_{e_\tau}, s_\varepsilon) = 1 - \max_{s_{e_\tau}, s_\varepsilon} \frac{|\mathcal{R}(s_{e_\tau}) \cap \mathcal{R}(s_\varepsilon)|}{|\mathcal{R}(s_{e_\tau}) \cup \mathcal{R}(s_\varepsilon)|}$$

- **Design Motivation**: The essence of humor is "relevant yet unexpected" — semantically connected (high TSS) but conceptually opposed (high CO). The shaping function $f(x) = x\exp(-x)$ ensures: (i) semantic similarity dominates; (ii) opposition is a bounded additive bonus; (iii) both are naturally balanced.

## Key Experimental Results

### Main Results: Humor Caption Generation with GPT-4o Backbone (pass@k, %)

| Method | #Top10 @1 | #Top10 @3 | #200-209 @1 | #200-209 @3 | #1000-1009 @1 | #1000-1009 @3 |
|--------|-----------|-----------|-------------|-------------|---------------|---------------|
| CoT | 45.79 | 70.59 | 57.28 | 82.85 | 61.58 | 86.90 |
| Few-shot | 58.07 | 78.91 | 65.12 | 81.14 | 65.59 | 88.39 |
| Self-consistency | 62.03 | 77.96 | 68.09 | 84.45 | 69.42 | 85.51 |
| HumorousAI | 62.11 | 81.24 | 69.38 | 85.32 | 73.46 | 85.42 |
| CLoT | 61.17 | 75.29 | 59.52 | 72.47 | 68.70 | 78.00 |
| **HOMER** | **66.41** | **83.70** | **73.40** | **88.38** | **76.32** | **90.50** |
| Gain | +6.92% | +3.03% | +5.79% | +3.59% | +3.89% | +2.39% |

### Ablation Study: Module Contributions (GPT-4o, Humor in AI #Top10)

| Configuration | pass@1 | pass@3 | pass@5 |
|---------------|--------|--------|--------|
| Image-Only | 20.20 | 38.30 | 51.00 |
| I+D | 50.60 | 69.49 | 78.00 |
| I+$\mathcal{C}$ | 41.80 | 59.70 | 67.00 |
| I+$\mathcal{T}_{\mathrm{im}}$ | 20.00 | 35.90 | 43.00 |
| I+D+$\mathcal{C}$ | 57.40 | 75.50 | 80.00 |
| I+D+$\mathcal{C}$+$\mathcal{T}_{\mathrm{im}}$ (Full) | **66.41** | **83.70** | **89.18** |

### Human Evaluation (5-point humor rating)

| Method | Humor in AI | Electronic Sheep |
|--------|-------------|-----------------|
| CoT | 2.47 ± 0.67 | 2.20 ± 0.78 |
| CLoT | 2.95 ± 0.77 | 2.53 ± 0.73 |
| HumorousAI | 3.01 ± 0.73 | 2.24 ± 0.81 |
| LoL | 3.16 ± 0.84 | 2.40 ± 0.82 |
| **HOMER** | **3.54 ± 0.59** | **3.31 ± 0.85** |

### Evaluator Reliability

| Evaluator | Humor in AI Ranking Accuracy | Electronic Sheep Ranking Accuracy |
|-----------|------------------------------|-----------------------------------|
| LLaMa 3 | 53.5% | 52.0% |
| Humor-tuned LLaMa3 | 60.0% | 58.0% |
| GPT-4.1 | 68.5% | 67.0% |
| GPT-5 | 73.5% | 70.0% |

## Key Findings

- **Script opposition is the most critical foundational component**: In ablation experiments, removing the conflicting script $\mathcal{C}$ causes the largest performance drop (I+D+$\mathcal{T}_{\mathrm{im}}$ achieves only 34.40 pass@1 vs. 66.41 for the full model), confirming that script opposition is the logical cornerstone of humor generation.

- **The imagination tree requires proper guidance to be effective**: Adding the imagination tree alone (I+$\mathcal{T}_{\mathrm{im}}$) performs worse than using only the image (Image-Only), because imagination without the guidance of script opposition and scene description produces irrelevant and nonsensical content. The $\mathcal{T}_{\mathrm{im}}$ only becomes effective when conditioned on $D$ and $\mathcal{C}$.

- **Strong cross-model generalizability**: HOMER consistently improves performance across all four backbone models — GPT-4o, Claude-4, Qwen-VL (7B), and LLaVA-1.5 (7B) — with larger gains on weaker models (Qwen-VL, LLaVA), where pass@1 improves by up to 24.4%, demonstrating that the framework design rather than backbone capability is the key driver.

- **Good cross-visual-domain generalization**: On the ImgFlip Meme dataset (comprising realistic, cartoon, and synthetic images), HOMER achieves 83.33% pass@1, outperforming CLoT (76.67%) by approximately 5.4%.

- **Human evaluation aligns with automatic metrics**: Among 20 evaluators using a 5-point scale, HOMER is the only method with a mean score above 3.0 (Humor in AI: 3.54; Electronic Sheep: 3.31). Cohen's $\kappa = 0.49$ indicates moderate inter-rater agreement, which is reasonable given the subjectivity of humor.

- **All three humor relevance scoring components are important**: Ablation experiments show that removing any single component (relevance-opposition, frequency, or diversity) leads to significant performance degradation, with the removal of the relevance-opposition score having the greatest impact.

## Highlights & Insights

- **Theory-driven vs. data-driven**: Rather than prompting LLMs to "try to be funny," HOMER systematically decomposes humor generation using the GTVH linguistic theory of humor — every step has a theoretical basis, yielding interpretability and principled design. This is the paper's most fundamental innovation.

- **Creative expansion mechanism of the imagination tree**: Multi-step conceptual leaps — such as from "coffee cups" to "cow" — require deep associative thinking. The combination of LLM associative chains (depth) and joke database retrieval (breadth) enables such creativity: depth association forms the backbone, while breadth retrieval supplements humor associations from everyday life.

- **Exploration of humor evaluation reliability**: The paper systematically compares the ranking accuracy of five evaluators, selecting GPT-5 (73.5%) as the primary evaluator — 23.5% above random chance. This provides a useful reference for the humor NLG evaluation community.

- **Toxic content detection**: Toxicity scores across all 7 Detoxify dimensions remain below 0.03, indicating that theory-driven humor generation does not produce harmful content — making it safer than crude "be funny" prompting approaches.

## Limitations & Future Work

1. **Dependence on GTVH theoretical coverage**: GTVH primarily addresses verbal humor; its coverage of non-verbal humor (e.g., visual puns) may be insufficient. Whether the framework applies to certain humor types (e.g., dark humor, deadpan humor) remains unexplored.

2. **High cost of multi-turn LLM calls**: The sequential invocation of three roles (extract → imagine → generate) combined with joke retrieval incurs substantially higher inference overhead than single-pass prompting, raising practical deployment costs.

3. **Coverage and bias of the joke database**: The 12 English joke datasets offer limited coverage and may exhibit cultural biases. Cross-lingual and cross-cultural humor generation has not been validated.

4. **Subjectivity of evaluation**: Although Cohen's $\kappa = 0.49$ is acceptable for humor evaluation, inter-rater consistency remains limited. The pass@k metric relies on the GPT-5 evaluator, whose ranking accuracy is itself only 73.5%.

## Related Work & Insights

### vs. CLoT (multi-hop reasoning with self-refinement)
CLoT relies on chain-of-thought reasoning for self-improvement but lacks a structured understanding of the nature of humor. HOMER provides a "recipe" for humor generation via GTVH theory, rather than simply prompting the LLM to "think more steps." In experiments, HOMER outperforms CLoT by 5.24% in pass@1 on GPT-4o (66.41 vs. 61.17), with larger gaps on weaker models.

### vs. HumorousAI (general humor prompting)
HumorousAI uses carefully designed prompts to guide GPT-4o in generating humorous captions but still relies on the LLM's inherent humor capability. HOMER's advantage lies in the imagination tree mechanism — by incorporating humor associations from external joke databases beyond LLM training data, it expands the creative space.

### vs. LoL (task-specific fine-tuning)
LoL improves humor generation through fine-tuning but is constrained by the scale and diversity of fine-tuning data. As a plug-and-play framework requiring no fine-tuning, HOMER outperforms LoL by 10.11% in pass@1 on GPT-4o (66.41 vs. 56.30) and offers superior interpretability.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | First systematic integration of GTVH humor theory into a multi-role LLM collaboration framework; the imagination tree + joke retrieval pruning mechanism is elegantly designed |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Consistent and significant improvements across 4 backbone models and 2 datasets; complete evaluation combining human assessment, automatic metrics, and ablation studies |
| Value | ⭐⭐⭐ | Multi-turn LLM calls and joke retrieval incur high inference costs; joke database requires maintenance; however, the framework is plug-and-play without fine-tuning |
| Writing Quality | ⭐⭐⭐⭐ | Clear paper structure, natural introduction of GTVH theory, thorough ablation experiments; case studies are vivid and convincing |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](../../ACL2026/information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](../../ACL2026/information_retrieval/slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](../../ACL2026/information_retrieval/mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ICLR 2026\] RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)
- [\[ICLR 2026\] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)

</div>

<!-- RELATED:END -->
