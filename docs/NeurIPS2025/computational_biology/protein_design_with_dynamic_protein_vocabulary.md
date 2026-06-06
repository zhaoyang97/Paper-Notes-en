---
title: >-
  [Paper Note] Protein Design with Dynamic Protein Vocabulary
description: >-
  [NeurIPS 2025][Computational Biology][protein design] ProDVa introduces natural protein fragments as a "dynamic vocabulary" for generative protein design…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "protein design"
  - "dynamic vocabulary"
  - "fragment retrieval"
  - "foldability"
  - "functional alignment"
date: 2026-05-08
content_hash: f8b865d58e6251b3
---

# Protein Design with Dynamic Protein Vocabulary

**Conference**: NeurIPS 2025
**arXiv**: [2505.18966](https://arxiv.org/abs/2505.18966)  
**Code**: [GitHub](https://github.com/sornkL/ProDVa)  
**Area**: Medical Imaging
**Keywords**: protein design, dynamic vocabulary, fragment retrieval, foldability, functional alignment

## TL;DR

ProDVa introduces natural protein fragments as a "dynamic vocabulary" for generative protein design, employing a three-component architecture consisting of a text encoder, a protein language model, and a fragment encoder. Using less than 0.04% of the training data required by prior work, ProDVa designs functionally aligned and structurally foldable protein sequences, surpassing the SOTA model Pinal by 7.38% on the pLDDT>70 ratio.

## Background & Motivation

Protein design is a central challenge in biotechnology, aiming to engineer novel proteins with desired functions within a vast sequence space. Recent deep generative models (e.g., ProteinDT, Pinal, PAAG) have enabled function-oriented protein design from textual descriptions, yet they face a fundamental limitation: **the designed proteins often fail to fold into stable three-dimensional structures**.

Classical protein design approaches (e.g., rational design, directed evolution) succeed precisely because they leverage known structures of natural proteins as design scaffolds. This motivates a natural question: **can the foldability of generative models be enhanced by incorporating natural protein fragments (e.g., motifs, functional sites)?**

The authors conduct a key pilot experiment demonstrating that even **randomly** inserting natural protein fragments into generated sequences (the Random+ method) significantly expands distributional diversity and improves foldability. In UMAP visualizations, Random+-generated proteins lie closer to the natural protein distribution than purely random sequences. This finding underpins the entire approach — fragments inherently encode structural priors.

Core limitations of existing methods:
- ProteinDT and PAAG perform poorly on pLDDT and PAE metrics, approaching random baselines
- Pinal achieves better performance but requires 1.76 billion training pairs and lacks open-source training scripts
- ESM3 achieves the lowest PPL but suffers from severe sequence repetition and poor foldability

## Method

### Overall Architecture

ProDVa consists of three core components: a Text Language Model (TextLM), a Protein Language Model (PLM), and a Fragment Encoder (FE). During training, functional annotations (fragment type and description) are automatically retrieved from the InterPro database for each protein sequence. During inference, relevant fragments are dynamically retrieved based on the input text to serve as candidate vocabulary.

### Key Designs

1. **Dynamic Protein Vocabulary**: Protein sequences are decomposed into two types of units — amino acid tokens and natural protein fragments. The static vocabulary $V_{\text{tokens}}$ segments amino acid sequences using a BPE tokenizer, while the dynamic vocabulary is drawn from InterPro-annotated fragments across 8 categories (Domain, Family, Active Site, etc.). Fragments are mapped into the same embedding space as the PLM via the Fragment Encoder. The key innovation is that at each generation step, the model may output either a single token **or** an entire fragment:

$$p(x_i = k | \mathbf{H}_{\text{pre}}) = \frac{\exp(\mathbf{H}_{\text{pre}} \mathbf{W}_{\text{out}}^{(k)})}{\sum_{k' \in V_{\text{tokens}} \cup S} \exp(\mathbf{H}_{\text{pre}} \mathbf{W}_{\text{out}}^{(k')})}$$

2. **Functional Annotation Learning**: Two auxiliary losses are introduced to fully exploit fragment functional annotations:

    - **Type Loss $\mathcal{L}_{\text{TYPE}}$**: Classifies fragments into 8 categories using weighted cross-entropy to handle class imbalance.
    - **Description Loss $\mathcal{L}_{\text{DESC}}$**: Aligns fragment representations with description text representations via InfoNCE contrastive learning, where positive pairs are matched fragment–description pairs and negatives are drawn from other pairs within the same batch.

$$\mathcal{L}_{\text{DESC}} = -\frac{1}{\sum_i |S_i|} \sum_i \sum_j \log \frac{\exp(\text{sim}(\mathbf{u}_{ij}, \mathbf{v}_{ij})/\tau)}{\sum_k \sum_l \exp(\text{sim}(\mathbf{u}_{ij}, \mathbf{v}_{kl})/\tau)}$$

3. **Fragment Retrieval at Inference**: Given an input functional description $t$, PubMedBERT embeddings are used to retrieve the Top-K most similar text–protein pairs (default K=16), from which InterPro extracts the corresponding fragments as the candidate set. Top-k sampling decoding is applied to enhance diversity.

### Loss & Training

Overall training objective: $\mathcal{L} = \mathcal{L}_{\text{NTP}} + \alpha \mathcal{L}_{\text{TYPE}} + \beta \mathcal{L}_{\text{DESC}}$

- TextLM is initialized from GPT-2; PLM and FE are initialized from ProtGPT2.
- Classification heads and description projection layers are used only during training and discarded at inference.
- The retrieval backend uses the txtai framework backed by Faiss.

## Key Experimental Results

### Main Results 1: CAMEO Subset (Functional Keyword-Conditioned Generation)

| Model | #Training Pairs | pLDDT↑ | %>70↑ | PAE↓ | %<10↑ | ProTrek↑ | Keyword Recovery↑ |
|------|--------|--------|-------|------|-------|---------|-------------------|
| Random+(E) | - | 62.38 | 32.65% | 17.23 | 9.28% | 3.29% | 0.00% |
| ProteinDT | 541K | 38.70 | 0.20% | 26.25 | 0.00% | 7.43% | 0.05% |
| Pinal | 1.76B | 66.50 | 47.21% | 14.57 | 33.53% | 14.57% | 30.46% |
| ESM3 | 539M | 59.79 | 31.49% | 17.40 | 21.37% | 3.76% | 5.49% |
| **ProDVa** | **392K** | **75.88** | **77.00%** | **6.39** | **83.88%** | 14.43% | 30.34% |

### Main Results 2: Mol-Instructions (Natural Language Description-Conditioned Generation)

| Model | pLDDT↑ | %>70↑ | PAE↓ | %<10↑ | ProTrek↑ | EvoLlama↑ |
|------|--------|-------|------|-------|---------|----------|
| Pinal | 75.25 | 68.97% | 10.96 | 58.44% | 17.50% | 53.42% |
| Chroma | 59.18 | 20.17% | 15.03 | 28.62% | 2.10% | 40.10% |
| **ProDVa** | **76.86** | **76.35%** | **8.66** | **68.06%** | **17.40%** | 51.10% |

### Ablation Study (vs. Vanilla Multimodal Baseline)

| Configuration | pLDDT | %>70 | PAE | %<10 | ProTrek Score | Notes |
|------|-------|------|-----|------|--------------|------|
| Vanilla (no fragments) | ~72 | ~63% | ~11 | ~58% | ~10% | GPT-2 + ProtGPT2 only |
| ProDVa | 76.86 | 76.35% | 8.66 | 68.06% | 17.40% | Full model |

ProDVa improves pLDDT by 4.63%, reduces PAE by 2.71%, and increases the pLDDT>70 ratio by 13.66%, confirming the contribution of fragments and functional annotations.

### Key Findings

- Even randomly inserted fragments (Random+) improve foldability, validating the effectiveness of fragments as structural priors.
- ProDVa surpasses Pinal on foldability metrics using only 0.02%–0.04% of its training data.
- Analysis of the Top-K retrieval parameter indicates K=16 is optimal; excessive retrieval degrades functional alignment.
- ProDVa also generalizes to unconditional protein generation, outperforming Pinal by 22.76% on pLDDT>70.

## Highlights & Insights

- The core insight is concise and compelling: natural protein fragments inherently encode folding priors, making fragment-based generation more efficient than residue-by-residue synthesis.
- The dynamic vocabulary concept elegantly transfers copy/retrieval mechanisms from NLP to protein design.
- The approach is highly data-efficient — achieving SOTA performance with only 0.04% of the training data, offering strong practical value.

## Limitations & Future Work

- The method relies on InterPro for fragment annotation, which may offer insufficient coverage for newly discovered proteins.
- Linguistic alignment metrics are slightly below Pinal (particularly EvoLlama Score), leaving room for improvement in text comprehension.
- No wet-lab validation is included; foldability evaluation relies entirely on ESMFold predictions.
- The optimal K value requires careful tuning and may differ across tasks.

## Related Work & Insights

- The comparison with Pinal is most central: Pinal follows a structure-then-sequence paradigm, whereas ProDVa operates directly in sequence space while incorporating structural fragment priors.
- The dynamic vocabulary paradigm is generalizable to other biological sequence design tasks (e.g., RNA, peptides).
- The fragment retrieval paradigm resembles RAG and has the potential to be combined with larger-scale protein structure databases.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The dynamic vocabulary + fragment prior idea is original and thoroughly validated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, multiple baselines, ablations, and unconditional generation — comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Motivation is introduced naturally; figures and tables are clear.
- Value: ⭐⭐⭐⭐ High data efficiency, good methodological generalizability, open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[ACL 2026\] ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](../../ACL2026/computational_biology/protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)

</div>

<!-- RELATED:END -->
