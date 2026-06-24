---
title: >-
  [Paper Note] Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization
description: >-
  [ACL 2025][Computational Biology][Protein Language Models] This paper proposes the KPO framework, which constructs a Protein Safety Knowledge Graph (PSKG) combined with a weighted graph pruning strategy to identify "similar but safe" protein pairs, and fine-tunes protein language models using DPO to steer them away from the hazardous sequence space while maintaining functionality.
tags:
  - "ACL 2025"
  - "Computational Biology"
  - "Protein Language Models"
  - "Knowledge Graphs"
  - "DPO"
  - "Biosafety"
  - "Preference Optimization"
date: 2026-05-08
content_hash: 0067259bec9d647f
---

# Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization

**Conference**: ACL 2025  
**arXiv**: [2507.10923](https://arxiv.org/abs/2507.10923)  
**Code**: [GitHub](https://github.com/HICAI-ZJU/KPO)  
**Area**: Computational Biology  
**Keywords**: Protein Language Models, Knowledge Graphs, DPO, Biosafety, Preference Optimization

## TL;DR

This paper proposes the KPO framework, which constructs a Protein Safety Knowledge Graph (PSKG) combined with a weighted graph pruning strategy to identify "similar but safe" protein pairs, and fine-tunes protein language models using DPO to steer them away from the hazardous sequence space while maintaining functionality.

## Background & Motivation

**Background**: Protein Language Models (PLMs) such as ProtGPT2 and ProGen2 have achieved significant success in protein sequence generation, enabling functional optimization and de novo design. These models capture the implicit relationships between sequence, structure, and function by training on massive amounts of protein sequence data.

**Limitations of Prior Work**: Unlike safety issues in textual LLMs (which are ethical/social), the security risks of PLMs have direct physical consequences. They might unintentionally generate harmful protein sequences that enhance viral transmissibility, escape immune responses, or confer drug resistance, which could lead to public health crises or even be exploited to manufacture bioweapons. However, current PLMs focus almost exclusively on functionality and generation performance, paying insufficient attention to safety.

**Key Challenge**: Existing research related to protein safety primarily focuses on introducing safety mutations to known protein sequences, but leaves the higher risk of the generation phase unaddressed—how can models avoid generating harmful proteins when synthesizing entirely new sequences?

**Goal**: How to introduce safety constraints during the fine-tuning of PLMs to minimize the probability of generating harmful proteins while maintaining functionality?

**Key Insight**: Inspired by LLM alignment (RLHF/DPO) in the NLP field, the key innovation lies in using a Protein Safety Knowledge Graph to incorporate domain expert knowledge for constructing preference pairs. Harmful and safe proteins are linked through shared Gene Ontology (GO) functional annotations; leveraging this connection allows for identifying "functionally similar but safe" alternative proteins.

**Core Idea**: Construct a Protein Safety Knowledge Graph (PSKG) to encode biochemical relationships between harmful and safe proteins, mine high-quality preference pairs via graph structures and embedding similarities, and employ DPO to steer PLMs away from the hazardous sequence space.

## Method

### Overall Architecture

The KPO framework consists of three stages: (1) constructing a Protein Safety Knowledge Graph (PSKG) that encodes relationships among harmful proteins $P_H$, safe proteins $P_B$, and Gene Ontology (GO) functional annotations; (2) performing weighted graph pruning to retain the most informative safe protein nodes, thereby reducing computational complexity; (3) identifying the safe proteins most similar to harmful proteins in both the graph structure and embedding space to construct preference pairs for fine-tuning the PLM with DPO.

### Key Designs

1. **蛋白质安全知识图谱(PSKG)构建**:

    - Function: Construct a knowledge graph encoding the relationships between harmful proteins $P_H$ and safe proteins $P_B$
    - Mechanism: Collate harmful proteins annotated with keywords "toxin" and "antigen" from the UniProt database (~18,000 sequences), and retrieve safe proteins from Swiss-Prot after excluding harmful ones. Establish indirect connections through Gene Ontology (GO) functional annotations—if a harmful protein $p_H^i$ and a safe protein $p_B^j$ share a GO term $g_z$, a triplet $(p_H^i, g_z, p_B^j)$ is formed.
    - Design Motivation: The hierarchical structure of GO can capture functional associations ranging from coarse-grained (e.g., "binding activity") to fine-grained (e.g., "DNA-binding transcription factor activity"), ensuring that the PSKG is not just a collection of annotations but a graph structure encoding biological expert knowledge.

2. **加权指标图剪枝**:

    - Function: Crop the large-scale PSKG into a compact subgraph that retains the most critical nodes, significantly reducing computational overhead.
    - Mechanism: Calculate an importance score for each safe protein node as $S(p_B^j) = \alpha \cdot C_{GO}(p_B^j) + \beta \cdot C_{Deg}(p_B^j)$, where $C_{GO}$ measures connections with high-scoring GO nodes and $C_{Deg}$ measures degree centrality. The score of a GO node also synthesizes bridging degree $R(g_z)$ (how many harmful-safe protein pairs it connects) and neighbor span $O(g_z)$ (how many safe proteins it connects).
    - Design Motivation: Retaining the top-50% GO nodes and top-50% safe protein nodes. Experiments show that score distributions exhibit a long-tail pattern, where low-scoring nodes contribute minimal marginal information; pruning cuts computation time in half without performance degradation.

3. **基于图+嵌入的偏好对构造与DPO微调**:

    - Function: Find the most "similar but safe" protein for each harmful protein from the pruned PSKG, constructing preference pairs for DPO fine-tuning.
    - Mechanism: Synthesize graph structural distance and TransE embedding cosine similarity to find matches: $s(p_H^i, p_B^j) = \mu \cdot \frac{1}{\text{dis}(p_H^i, p_B^j)} + (1-\mu) \cdot \cos(e_{p_H^i}, e_{p_B^j})$. For each harmful protein, select the top-M safe proteins to construct preference pairs $(p_B^j, p_H^i)$, and fine-tune using the DPO loss: $L_{KPO} = -\log \sigma(\varphi \cdot [\log P_\theta(p_B^j|x) - \log P_\theta(p_H^i|x)])$.
    - Design Motivation: Unlike random pairing (standard DPO), PSKG-guided pairing ensures that safe and harmful proteins are functionally related, enabling the model to learn subtle differences to "bias towards safe directions within functionally similar spaces".

### Loss & Training

DPO training is implemented using the TRL library with a learning rate of 5e-5, training for approximately 2 hours per epoch on 8×A100 GPUs. The harmful protein dataset is split into training/testing at an 8:2 ratio. The parameter $\varphi$ serves as a scaling factor in the DPO loss to control optimization intensity.

## Key Experimental Results

### Main Results

Safety and functional evaluations across three base PLM models:

| Model | BLAST↓ | MMseq2↓ | ToxinPred3↓ | GB1↑ | GFP↑ |
|------|--------|---------|-------------|------|------|
| ProtGPT2 | 0.269 | 0.325 | 0.070 | 0.030 | 1.526 |
| ProtGPT2+KPO | **0.138** | **0.149** | **0.024** | **0.041** | **2.204** |
| ProGen2 | 0.155 | 0.170 | 0.029 | 0.144 | 1.683 |
| ProGen2+KPO | **0.128** | **0.117** | **0.007** | 0.024 | 1.562 |
| InstructProtein | 0.410 | 0.285 | 0.031 | 0.030 | 1.983 |
| InstructProtein+KPO | **0.086** | **0.079** | **0.003** | **0.191** | **2.319** |

### Ablation Study

Comparison of different preference pair) construction methods on ProtGPT2:

| Method | BLAST↓ | MMseq2↓ | ToxinPred3↓ | Description |
|------|--------|---------|-------------|------|
| DPO (Random pairing) | ~0.18 | ~0.20 | ~0.04 | Without PSKG |
| KPO-random | ~0.16 | ~0.17 | ~0.03 | Randomly pruned PSKG |
| KPO-community | ~0.15 | ~0.16 | ~0.03 | Community detection pruning |
| **KPO (Ours)** | **0.138** | **0.149** | **0.024** | Weighted metric pruning |

### Key Findings

- **Significant Safety Improvements**: KPO reduces BLAST/MMseq2 similarities by 50-80% and decreases ToxinPred3 toxicity predictions by 66-90%.
- **Functionality Preserved or Enhanced**: GFP fitness increases by 44% (ProtGPT2) and 17% (InstructProtein). This performance boost occurs because steering the model away from the toxic space allows it to better explore safe regions with superior functionality.
- **Embedding Space Analysis**: t-SNE visualization shows that the embeddings of KPO-fine-tuned generated sequences are clearly separated from those of harmful proteins.
- **3D Structural Verification**: ColabFold predictions show the RMSD between KPO-generated proteins and harmful proteins increases from ~1.4Å to ~8.0Å, indicating a significantly heightened structural divergence.
- **Graph Pruning Effectiveness**: KPO outperforms KPO-random and KPO-community, proving that weighted metric pruning successfully preserves the most informative nodes.

## Highlights & Insights

- **First Protein Safety Knowledge Graph (PSKG)**: This work conceptualizes migrating security alignment from NLP to the protein domain. Adapting GO annotations to build indirect associations between harmful and safe proteins provides richer biological prior knowledge than simple sequence-similarity-based pairing.
- **Win-Win for Safety and Functionality**: While safety constraints are typically thought to sacrifice performance, KPO demonstrates that steering the model away from the harmful space can actually facilitate the exploration of functionally superior regions. This insight can be applied to safety alignment research in other domains.
- **Alignment Migration from LLM to PLM**: This study demonstrates that textual LLM alignment techniques like DPO can be effectively transferred to protein sequence generation, paving a practical path for cross-modal safety alignment.

## Limitations & Future Work

- **Focus Only on Sequence-Level Safety**: The method does not directly constrain harmful conformations at the 3D structural level and relies on downstream evaluation tools. Future work could integrate structural information predicted by AlphaFold as an additional reward signal.
- **Limitations in Defining Harmful Proteins**: Relying solely on "toxin" and "antigen" keyword searches might miss other types of harmful proteins (such as prions or allergens).
- **Computational Overhead**: Constructing the large-scale PSKG and training the TransE embeddings still require substantial computational resources.
- **Data Security Risks**: The harmful protein dataset used in the paper poses inherent safety risks, and the authors have indicated restricted public access.

## Related Work & Insights

- **vs. Protein Mutation Safety Methods**: Methods like the unlearning approach by Li et al. (2024) focus on introducing safety mutations to known proteins, whereas KPO directly imposes safety constraints during the generation phase, offering broader coverage.
- **vs. Standard DPO**: Ablation studies show that PSKG-guided preference pairing significantly outperforms random pairing, indicating that domain knowledge is crucial for biosecurity alignment.
- **Insights**: This paradigm of "Knowledge Graph-Guided Preference Optimization" can be transferred to other generation tasks requiring domain-specific safety constraints, such as drug molecule generation and chemical reaction prediction.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically address safety during the generation phase of PLMs; the PSKG construction is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three base PLMs + multi-dimensional safety/functionality evaluations + ablation studies + hyperparameter sensitivity + structural analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a serious problem formulation and a complete solution.
- Value: ⭐⭐⭐⭐ Protein AI safety is a crucial and under-explored research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/computational_biology/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](../../AAAI2026/computational_biology/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[ICLR 2026\] DrugTrail: Interpretable Drug Discovery via Structured Reasoning and Druggability‑Tailored Preference Optimization](../../ICLR2026/computational_biology/drugtrail_interpretable_drug_discovery_via_structured_reasoning_and_druggability.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)

</div>

<!-- RELATED:END -->
