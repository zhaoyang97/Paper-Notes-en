---
title: >-
  [Paper Note] De novo generation of functional terpene synthases using TpsGPT
description: >-
  [NeurIPS 2025][Medical Imaging][terpene synthase] TpsGPT fine-tunes a distilled ProtGPT2 Tiny (38.9M parameters) on 79K terpene synthase (TPS) sequences to generate 28K candidate sequences…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "terpene synthase"
  - "protein language model"
  - "ProtGPT2"
  - "de novo design"
  - "wet-lab validation"
date: 2026-05-08
content_hash: 9c1efd2b866e48c7
---

# De novo generation of functional terpene synthases using TpsGPT

**Conference**: NeurIPS 2025
**arXiv**: [2512.08772](https://arxiv.org/abs/2512.08772)
**Code**: [https://github.com/colorfulcereal/TpsGPT](https://github.com/colorfulcereal/TpsGPT)
**Area**: Protein Design / Enzyme Engineering
**Keywords**: terpene synthase, protein language model, ProtGPT2, de novo design, wet-lab validation

## TL;DR
TpsGPT fine-tunes a distilled ProtGPT2 Tiny (38.9M parameters) on 79K terpene synthase (TPS) sequences to generate 28K candidate sequences, which are subsequently filtered through a multi-stage pipeline (perplexity / pLDDT / EnzymeExplorer / CLEAN / InterPro / Foldseek) to yield 7 de novo TPS sequences that are evolutionarily distant (<60% sequence identity) yet structurally conserved. Wet-lab experiments confirm that 2 of the 7 candidates possess TPS enzymatic activity—achieving functional enzyme de novo design at a GPU cost below $200.

## Background & Motivation

**Background**: Terpene synthases (TPS) are a key enzyme family catalyzing the biosynthesis of terpene scaffolds. Terpenes constitute the largest class of natural products (>76,000 compounds), encompassing important pharmaceuticals such as the anticancer drug Taxol. Chemical synthesis of terpenes is costly and multi-step; leveraging TPS enzymes in synthetic biology offers a far more efficient alternative.

**Limitations of Prior Work**: (a) **Directed evolution** is the dominant enzyme engineering approach but is prohibitively expensive (on the order of hundreds of thousands of dollars), slow, and yields sequences highly similar to natural proteins, thereby failing to explore the vast sequence space. (b) Computational approaches such as HMMER can only discover naturally occurring TPS sequences and cannot design novel enzymes. (c) Structure-guided design methods (e.g., RFdiffusion) require detailed knowledge of catalytic sites, which is impractical for TPS given the complexity of its functional mechanism.

**Key Challenge**: TPS is an underrepresented enzyme family with only 1,125 experimentally validated sequences, yet its sequence space is enormous. Existing methods either cannot explore novel sequences (database mining), impose prohibitive costs (directed evolution), or require unavailable structural knowledge (structure-based design).

**Goal**: To generate functional TPS enzymes de novo from sequence data alone—without structural knowledge—at low computational cost.

**Key Insight**: Protein language models (PLMs) have been shown to generate valid proteins within specific families through fine-tuning. However, existing PLM fine-tuning approaches typically require large family-level datasets and conditioning labels. The authors begin with 1,125 seed sequences, mine 79K homologous TPS sequences, and fine-tune a lightweight model (ProtGPT2 Tiny, 38.9M parameters) to reduce cost.

**Core Idea**: Carefully curated enzyme-family-specific dataset + distilled PLM fine-tuning + multi-stage computational filtering = low-cost functional enzyme de novo design.

## Method

### Overall Architecture
**Input**: A dataset of 79K TPS sequences mined from UniProt. **Model**: Fine-tuned ProtGPT2 Tiny (38.9M parameters). **Output**: 28K candidate sequences → multi-stage filtering → 7 candidates → wet-lab validation → 2 confirmed active.

### Key Designs

1. **Dataset Construction**:

    - **Function**: Expand 1,125 experimentally validated TPS seed sequences into a 79K training set.
    - **Mechanism**: (a) HMMER hmmsearch against Pfam/SUPERFAMILY databases is applied to UniProt and BFD to retrieve TPS homologs at scale. (b) Rigorous filtering is applied: sequences outside the 300–1,100 amino acid range are removed; sequences with stronger non-TPS domain matches are excluded; sequences must contain TPS catalytic motifs (DDXXD, NSE/DTE, DXDD); and isopentenyl diphosphate synthases with >80% similarity are filtered out.
    - **Design Motivation**: 1,125 sequences are insufficient to train a PLM directly; computational mining expands the dataset 70-fold. Strict filtering ensures training data quality by retaining only high-confidence TPS sequences.

2. **Data Partitioning Strategy**:

    - **Function**: Prevent data leakage between training and validation sets.
    - **Mechanism**: SpanSeq is used to cluster the 79K sequences at a 30% sequence identity threshold into 6 partitions. Five partitions (~63K sequences) are used for training and one (~16K) for validation, ensuring a maximum pairwise identity of ≤30% between the two sets.
    - **Design Motivation**: Conventional random splitting may allow highly similar sequences to appear in both training and validation sets (data leakage). SpanSeq's similarity-based partitioning guarantees reliable evaluation.

3. **Model Fine-Tuning**:

    - **Function**: Fine-tune the distilled ProtGPT2 on TPS data.
    - **Mechanism**: ProtGPT2 Tiny (38.9M parameters; a distilled version of the 738M original) is trained with block size = 512 tokens, batch size = 64 with gradient accumulation × 8 (effective batch size = 512), learning rate = 1e-4, and up to 4,000 steps. Training loss decreases from 8.4 to 4.94; validation loss decreases from 8.0 to 7.32.
    - **Design Motivation**: The distilled model offers 6× faster inference with comparable perplexity, making large-scale sequence generation (28K sequences) computationally feasible on a single NVIDIA L4 GPU.

4. **Multi-Stage Computational Filtering Pipeline**:

    - **Function**: Select the most functionally promising sequences from 28K candidates.
    - **Mechanism**: Three filtering tiers are applied—
        - **Sequence-level filtering**: Top 10% by perplexity ranking (2,800 sequences); maximum sequence identity to the training set ≤60% (ensuring evolutionary distance).
        - **Function-level filtering**: EnzymeExplorer TPS detection score ≥0.7; CLEAN-predicted EC number belonging to the terpene biosynthesis pathway; InterPro domain prediction detecting TPS-specific domains.
        - **Structure-level filtering**: ESMFold pLDDT ≥70 (structural stability); Foldseek TM-score between 0.6 and 0.9 (structurally similar to, but not identical to, training set members).
    - **Design Motivation**: Each filtering tier validates candidates from a distinct dimension—sequence quality (perplexity), functional plausibility (enzyme classification), and structural viability (folding confidence)—progressively reducing the pool from 28K → 2,800 → 77 → 7.

5. **Wet-Lab Validation**:

    - **Function**: Heterologous expression and activity assay in *Saccharomyces cerevisiae*.
    - **Mechanism**: All 7 candidate genes are expressed in the engineered yeast strain JWY501 (engineered to overproduce the GGPP substrate), and products are detected by LC-MS. Extracted ion chromatograms (XIC) at the $\text{C}_{20}\text{H}_{36}\text{O}_2$ mass confirm that TpsGPT1 and TpsGPT2 produce diterpene-like products (e.g., angelicene-type compounds).
    - **Design Motivation**: Computational validation, however comprehensive, cannot substitute for experimental evidence—wet-lab testing is the ultimate proof of functional de novo enzyme design.

### Loss & Training
- Standard autoregressive language modeling loss (next-token prediction).
- Learning rate search over {1e-6, 1e-5, 1e-4, 1e-3}; 1e-4 selected (best validation loss).
- Maximum steps searched over {1200, 1875, 3000, 4000}; 4,000 selected (convergence).
- Total GPU cost < $200 on a single NVIDIA L4.

## Key Experimental Results

### Main Results — Computational Validation of 7 TPS Candidates

| Sequence ID | EnzymeExplorer ↑ | pLDDT ↑ | TM-Score | Max Seq ID | CLEAN EC | Experimental Activity |
|-------------|------------------|---------|----------|------------|----------|-----------------------|
| **TpsGPT1** | 0.75 | 78 | 0.73 | 49.67% | Terpene synthase (4.2.3.75) | **✓** |
| **TpsGPT2** | 0.72 | 74 | 0.79 | 59.72% | Squalene synthase (2.5.1.21) | **✓** |
| TpsGPT3 | 0.73 | 74 | 0.84 | 60.00% | Cyclic dienol synthase | Pending |
| TpsGPT4 | 0.73 | 70 | 0.65 | 60.08% | Squalene synthase | Pending |
| TpsGPT5 | 0.78 | 80 | 0.72 | 59.75% | β-Amyrin synthase | Pending |
| TpsGPT6 | 0.73 | 71 | 0.69 | 57.33% | Squalene synthase | Pending |
| TpsGPT7 | 0.74 | 71 | 0.72 | 52.19% | Cycloartenol synthase | Pending |

### Filtering Pipeline Efficiency

| Filtering Stage | Sequences Remaining | Rejection Rate |
|-----------------|--------------------:|---------------:|
| Raw generation | 28,000 | — |
| Perplexity Top 10% | 2,800 | 90% |
| pLDDT ≥70 | ~1,120 (40%) | 60% |
| EnzymeExplorer ≥0.7 | 77 | 93% |
| MaxID ≤60% + Foldseek 0.6–0.9 | 7 | 91% |
| Wet-lab validation | 2/7 active | 29% |

### Key Findings
- **pLDDT distribution**: 40% of generated sequences achieve structural confidence ≥70, indicating that TpsGPT has learned to generate foldable proteins.
- **Evolutionary distance**: The 7 candidates exhibit maximum sequence identities of 49.67%–60.08% to the training set, confirming that they represent genuinely novel, evolutionarily distant sequences.
- **Structural conservation**: All candidates have TM-scores between 0.6 and 0.9, placing them within the same structural family without being simple copies of known sequences.
- TpsGPT1, the most evolutionarily distinct candidate (maxID = 49.67%), is experimentally validated as active, demonstrating that the model successfully explores distal regions of sequence space.
- The detected products contain oxygen ($\text{C}_{20}\text{H}_{36}\text{O}_2$); whether they constitute canonical TPS products remains to be confirmed and warrants further mechanistic investigation.

## Highlights & Insights
- **Extreme cost efficiency**: <$200 GPU cost with a 38.9M-parameter model achieves functional enzyme de novo design, substantially lowering the barrier to protein design. Compared with the hundreds-of-thousands-of-dollars cost of directed evolution, this represents an order-of-magnitude improvement.
- **Dataset construction is the critical bottleneck**: The careful construction of a high-quality TPS dataset—expanding 1,125 seed sequences to 79K via HMMER, multi-level filtering, and catalytic motif verification—underpins the entire pipeline. Data engineering proves no less important than model selection.
- **Multi-dimensional filtering strategy**: The combination of sequence-level (perplexity + identity), function-level (three classification/domain prediction tools), and structure-level (pLDDT + TM-score) filtering precisely identifies 7 candidates from 28K, with each tier grounded in clear physicochemical or biological rationale.
- **Generalizable framework**: The pipeline (data mining → PLM fine-tuning → generation → multi-stage filtering → experimental validation) is directly transferable to other underrepresented enzyme families (e.g., lysozymes, P450s).

## Limitations & Future Work
- Only 2 of 7 candidates are experimentally confirmed active (29% success rate), and the oxygenated products have yet to be definitively assigned as canonical terpenes.
- The model does not support conditional generation; it cannot be directed to produce a specific TPS subclass (e.g., monoterpene / sesquiterpene / diterpene synthase).
- Only sequence information is utilized; 3D structural information (e.g., catalytic pocket geometry) is not incorporated to guide generation.
- The limited capacity of ProtGPT2 Tiny (38.9M parameters) may constrain the breadth of sequence space exploration.
- The filtering pipeline depends on multiple external tools (EnzymeExplorer, CLEAN, InterPro, ESMFold, Foldseek); biases in any individual tool propagate to the final output.
- Future directions include conditional generation (with control labels), integration of structural constraints (active-site motif injection), and the use of larger backbone models.

## Related Work & Insights
- **vs. RFdiffusion (structure-guided design)**: RFdiffusion requires detailed structural knowledge of catalytic sites, making it impractical for TPS; TpsGPT requires only sequence data.
- **vs. ProGEN (large-scale PLM fine-tuning)**: ProGEN uses 280M parameters and requires large datasets with conditioning labels; TpsGPT employs a 38.9M distilled model without conditioning labels, making it substantially more lightweight.
- **vs. HMMER database mining**: Database mining can only retrieve naturally occurring sequences; TpsGPT generates novel sequences that are evolutionarily distant (<60% identity) from known TPS.
- **Insight**: PLM fine-tuning is particularly valuable for "small-data enzyme families"—as few as ~1,000 seed sequences, expanded through computational mining, are sufficient to train an effective generative model.

## Rating
- **Novelty**: ⭐⭐⭐ Fine-tuning PLMs for protein generation is not novel per se, but the complete data-generation-filtering-experimental pipeline tailored to the TPS family is new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Computational validation is comprehensive (6 validation metrics) and supported by wet-lab experiments (2/7 active), though the sample size is small.
- **Writing Quality**: ⭐⭐⭐⭐ The pipeline is described clearly, filtering criteria are well justified, and results are presented intuitively.
- **Value**: ⭐⭐⭐⭐ Low-cost functional enzyme de novo design has direct applications in synthetic biology, and the pipeline is generalizable to other enzyme families.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICCV 2025\] Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines](../../ICCV2025/medical_imaging/integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[NeurIPS 2025\] Amortized Active Generation of Pareto Sets](amortized_active_generation_of_pareto_sets.md)

</div>

<!-- RELATED:END -->
