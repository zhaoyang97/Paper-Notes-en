---
title: >-
  [Paper Note] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning
description: >-
  [ACL 2025][LLM (Other)][molecular structure elucidation] This paper proposes K-MSE (Knowledge-enhanced Molecular Structure Elucidation), a framework that constructs a molecular substructure knowledge base to expand the chemical structure space coverage of LLMs, designs a specialized molecule-spectrum scorer to replace self-evaluation by LLMs, and incorporates Monte Carlo Tree Search (MCTS) to achieve test-time inference scaling. On the MolPuzzle benchmark…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "molecular structure elucidation"
  - "MCTS"
  - "knowledge base"
  - "reward model"
  - "spectral data"
  - "test-time scaling"
date: 2026-05-08
content_hash: 88d60c27b5c4d17a
---

# Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning

**Conference**: ACL 2025  
**Authors**: Xiang Zhuang, Bin Wu, Jiyu Cui, Kehua Feng, Xiaotong Li, Huabin Xing, Keyan Ding, Qiang Zhang, Huajun Chen (Zhejiang University, UCL)  
**arXiv**: [2506.23056](https://arxiv.org/abs/2506.23056)  
**Code**: [GitHub](https://github.com/HICAI-ZJU/K-MSE)  
**Area**: LLM/NLP, Chemical Reasoning, Molecular Structure Elucidation  
**Keywords**: molecular structure elucidation, MCTS, knowledge base, reward model, spectral data, test-time scaling  

## TL;DR

This paper proposes K-MSE (Knowledge-enhanced Molecular Structure Elucidation), a framework that constructs a molecular substructure knowledge base to expand the chemical structure space coverage of LLMs, designs a specialized molecule-spectrum scorer to replace self-evaluation by LLMs, and incorporates Monte Carlo Tree Search (MCTS) to achieve test-time inference scaling. On the MolPuzzle benchmark, it improves the accuracy of GPT-4o-mini and GPT-4o from 3.7% and 27.8% to 27.3% and 39.8%, respectively.

## Background & Motivation

- **Core Problem:** Molecular structure elucidation is a fundamental task in chemical experimental analysis — inferring molecular structures from spectral data such as NMR and IR. Even human experts require 10-15 minutes to process one molecule. Although LLMs have the potential to automate this process, they face two major challenges.
- **Limitations of Prior Work:** (1) LLMs lack comprehensive coverage of the molecular structure space — frequently misidentifying uncommon structures like thiophene as benzene rings (the most common aromatic structure); (2) LLMs cannot accurately evaluate their own reasoning results, lacking the domain knowledge necessary to assess the alignment between predicted molecules and spectral data, which leads to a lack of effective reward signals in tree search reasoning.
- **Research Motivation:** Enhancing chemical structure coverage through external knowledge + providing accurate rewards via a specialized scorer $\rightarrow$ incorporating MCTS to achieve test-time inference scaling for LLMs in molecular structure elucidation.

## Method

### Overall Architecture

K-MSE consists of three components:
1. **Molecular Substructure Knowledge Base** $\mathcal{KB} = \{(s_i, d_i)\}$: Contains substructure SMILES representations and textual descriptions, with cyclic and acyclic substructures extracted from the MOSES molecular database.
2. **Molecule-Spectrum Scorer**: Consists of a molecular encoder $g_m$ (GIN + MLP processing molecular graphs and Morgan fingerprints) and a spectral encoder $g_s$ (Transformer processing chemical shifts, splitting patterns, and coupling constants of C-NMR/H-NMR).
3. **MCTS Reasoning Framework**: Starts by retrieving relevant substructures from the knowledge base $\rightarrow$ iteratively performs Selection (UCT) $\rightarrow$ Expansion (Critique + Rewrite) $\rightarrow$ Evaluation (by the scorer) $\rightarrow$ Backpropagation.

### Key Designs

1. **Knowledge Base Construction**: Automatically extracts molecular substructures from the MOSES database and automatically generates reliable descriptions using LLMs combined with structural info from external tools, balancing both diversity and generalizability.
2. **Specialized Scorer**: The molecular encoder utilizes GIN to encode molecular graphs + MLP to encode Morgan fingerprints. The spectral encoder discretizes NMR chemical shifts and coupling constants into token IDs before feeding them into a Transformer. Training employs the NT-Xent contrastive learning loss to maximize the embedding similarity of matched molecule-spectral pairs.
3. **Dual Role of the Scorer**: It serves both as the MCTS reward model to evaluate candidate molecules ($R(a') = \text{sim}(g_m(m_{a'}), g_s(n))$) and as a retrieval bridge for the knowledge base — using the spectral encoder to encode query spectra and the molecular encoder to encode substructures for Top-k retrieval.

### Loss & Training

The scorer is trained using the NT-Xent contrastive learning loss: maximizing the cosine similarity of correct molecule-spectrum pairs, minimizing the similarity of negative pairs within the batch, where the temperature parameter $\tau$ controls the distribution sharpness. The MCTS backpropagation uses a weighted update of $Q(a) = 0.5 \times Q(a') + 0.5 \times Q(a)$.

## Experiments

### Main Results — MolPuzzle Benchmark (216 molecules, zero-shot)

| Model | Method | Morgan FTS | MACCS FTS | ACC |
|------|------|-----------|-----------|-----|
| GPT-4o-mini | baseline | 0.260 | 0.512 | 0.037 |
| GPT-4o-mini | + Self-Refine | 0.287 | 0.523 | 0.069 |
| GPT-4o-mini | + MCTSr | 0.281 | 0.530 | 0.069 |
| GPT-4o-mini | **+ K-MSE** | **0.470** | **0.651** | **0.273** |
| GPT-4o | baseline | 0.493 | 0.690 | 0.278 |
| GPT-4o | + Self-Consistency | 0.551 | 0.732 | 0.347 |
| GPT-4o | **+ K-MSE** | — | — | **0.398** |
| Llama-3.2-11B | baseline | 0.163 | 0.349 | 0.014 |
| Llama-3.2-11B | **+ K-MSE** | **0.298** | **0.465** | **0.111** |

### Ablation Study

| Ablated Component | Impact on GPT-4o-mini ACC |
|---------|-------------------------|
| Full K-MSE | 0.273 |
| Remove knowledge base | Obvious decrease — LLMs struggle to identify uncommon substructures |
| Replace specialized scorer with LLM | Significant decrease — LLMs fail to accurately evaluate molecule-spectrum match |
| Remove molecular images from Critique | Decrease — Text-only critique struggles to detect structural errors |
| Remove chemical formulas from Critique | Decrease — Lacks chemical constraint information |

### Key Findings

- K-MSE achieves substantial improvements across all base models: GPT-4o-mini ACC +23.6%, GPT-4o ACC +12.0%, Llama-3.2-11B ACC +9.7%
- Existing general reasoning enhancement methods (Self-Refine, MCTSr, MAD) have limited effectiveness in molecular structure elucidation — the core bottleneck is the lack of domain knowledge.
- The specialized scorer is far superior to LLM self-evaluation — LLMs lack the domain knowledge to determine molecule-spectrum matches.
- Substructure information in the knowledge base is crucial for handling uncommon molecular structures.
- As a plug-and-play framework, K-MSE can be combined with any LLM.

## Highlights & Insights

- Finely applies test-time reasoning scaling combined with external knowledge enhancement to the task of molecular structure elucidation for the first time.
- The dual-role design of the scorer, acting as both a reward model and a retrieval bridge, is highly elegant.
- The plug-and-play nature of the framework endows it with high practical value.
- The absolute accuracy improvement of 20%+ on MolPuzzle is extremely significant.

## Limitations & Future Work

- Evaluation is only conducted on the MolPuzzle benchmark, which has a relatively small scale (216 molecules).
- The coverage of the scorer's training data may limit its generalization ability to rare molecular types.
- The increased number of MCTS iterations incurs significant inference time costs (API calls + scorer inference).
- The knowledge base is static, with online expansion or adaptive update mechanisms left unexplored.
- Only NMR and IR spectra are considered, leaving other commonly used analytical data like mass spectrometry (MS) unaddressed.

## Related Work & Insights

- **LLM Chemical Reasoning:** ChemCrow (M. Bran et al., 2024) integrating external tools, ChatDrug (Liu et al., 2024) molecular editing, and STRUCTCHEM (Ouyang et al., 2024) predefined reasoning templates.
- **Tree Search Reasoning:** Tree-of-Thought (Yao et al., 2023), MCTSr (Zhang et al., 2024a), but they lack domain-specific accurate reward models.
- **Molecular Structure Elucidation:** MolPuzzle (Guo et al., 2024) first proposed the LLM benchmark for this task.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Structural Reasoning Improves Molecular Understanding of LLM](structural_reasoning_improves_molecular_understanding_of_llm.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)

</div>

<!-- RELATED:END -->
