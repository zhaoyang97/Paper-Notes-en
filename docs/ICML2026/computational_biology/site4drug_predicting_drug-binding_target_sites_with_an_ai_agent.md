---
title: >-
  [Paper Note] Site4Drug: Predicting Drug-Binding Target Sites with an AI Agent
description: >-
  [ICML 2026][Computational Biology][Drug Targets] Site4Drug reframes the upstream bottleneck of "where to target a drug on a protein" as a constraint-first evidence integration problem. An LLM Agent derives feasibility si…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Drug Targets"
  - "Epitope Discovery"
  - "Pocket Discovery"
  - "LLM Agent"
  - "Auditability"
date: 2026-05-08
content_hash: 220b0d23fc792141
---

# Site4Drug: Predicting Drug-Binding Target Sites with an AI Agent

**Conference**: ICML 2026  
**arXiv**: [2606.01816](https://arxiv.org/abs/2606.01816)  
**Code**: https://github.com/winterrykim/Site4Drug_Demo  
**Area**: Scientific Computing / Drug Discovery / LLM Agent  
**Keywords**: Drug Targets, Epitope Discovery, Pocket Discovery, LLM Agent, Auditability  

## TL;DR
Site4Drug reframes the upstream bottleneck of "where to target a drug on a protein" as a constraint-first evidence integration problem. An LLM Agent derives feasibility signals such as topology, PTMs, Motifs, and Cysteine networks from sequences, outputting ranked candidate sites with scores, risk labels, and traceable logs, while automatically recommending the appropriate modality (antibody/peptide vs. small molecule).

## Background & Motivation

**Background**: Existing drug design pipelines mostly assume "binding sites are known," focusing primarily on docking, virtual screening, or binder generation (e.g., BoltzGen, DrugCLIP, BindCLIP). Sites are either taken from existing co-crystal structures (residues within $\le 4\text{Å}$ of the ligand) or extracted from structures using geometric tools (fpocket, RAPID-Net).

**Limitations of Prior Work**: The early "site selection before binder selection" stage often fails in real-world scenarios. For membrane proteins, only certain regions are physically accessible; topology predictions can be contradictory, and PTMs like glycosylation can mask or disrupt candidate epitopes. When downstream screening fails, teams often struggle to distinguish whether the issue lies with the binder model or the site selection because the reasoning for site selection is rarely recorded. Geometric methods only find canonical pockets, cannot incorporate heterogeneous metadata, and do not cover non-small molecule modalities.

**Key Challenge**: Feasibility evidence (topology, PTMs, cysteine networks, Motif context) is discrete, heterogeneous, and cross-modal, whereas downstream tools require a single, comparable, and interpretable site ranking.

**Goal**: To output (i) recommended binding modality, (ii) ranked candidate sites, and (iii) evidence summaries, risk labels, and decision logs for each candidate from any protein sequence without relying on fixed ground truth.

**Key Insight**: LLMs are naturally capable of fused unstructured evidence and generating traceable reasoning chains. Allowing an LLM to score small molecule pockets and antibody epitopes simultaneously on unified evidence avoids false positives that are "chemically plausible but biologically masked."

**Core Idea**: Replace single geometric/learned scoring with "constraint-first + evidence aggregation + Agent-based multi-review," transforming site selection into an auditable multi-agent decision process.

## Method

### Overall Architecture
The input is a protein amino acid sequence $x_{1:L}$, and the output is a structured report including: modality recommendation $\hat{m}\in\{\text{epitope}, \text{pocket}, \text{other}\}$, $K$ candidate regions $\{r_k\}$ with scores $S(r_k)$, accessibility/topology labels, evidence summaries, and typed risk labels. The pipeline consists of two modules: Module 1 handles evidence extraction → candidate generation → scoring/ranking → Agent review; Module 2 passes high-scoring candidates to modality-matched downstream design tools (Epitope → BoltzGen, Pocket → DrugCLIP/BindCLIP).

### Key Designs

1.  **Three-way Feasibility Evidence Extraction**:
    - **Function**: Extracts multi-source signals regarding "site feasibility" from pure sequences to feed into the LLM for candidate nomination and subsequent auditing.
    - **Mechanism**: (a) Topology/Hydrophobicity: Uses Kyte–Doolittle sliding window hydrophobicity values plus heuristic TM detection to provide coarse labels like `tmd/restricted` or `outside/exposed`, with confidence determined by the margin between hydrophobicity and the TM threshold; (b) PTM Risks: Uses MusiteDeep to predict sites like phosphorylation and glycosylation, expanded into typed local masks (e.g., phosphorylation at 211 expands to 208–214), recording overlaps and type counts with candidates; (c) Motifs and Cysteines: Uses ScanProsite to identify Motif hits labeled as `motif-overlap`, using local cysteine counts as a lightweight proxy for disulfide bond constraints.
    - **Design Motivation**: Explicitly enumerates multi-source biological constraints ("why this site cannot be chosen") rather than burying them in LLM implicit priors, enabling the Agent to provide auditable critiques in a claim→evidence→impact format.

2.  **LLM Nomination + Specialist Multi-Agent Review**:
    - **Function**: Uses an LLM to generate ranked candidate JSONs based on unified evidence, followed by specialized Agents reviewing and producing final modality decisions and adjusted rankings.
    - **Mechanism**: The LLM receives the sequence plus a compressed evidence summary and outputs a list of candidate regions. After filtering invalid entries, each candidate is tagged with topology labels, PTM/Motif overlaps, cysteine counts, risk labels, and heuristic scores. Subsequently, BioAgent, ChemAgent, and RiskAgent return critiques in a claim→evidence→impact format based on the same evidence summary. The DecisionAgent synthesizes all critiques for the final modality determination and re-ranking, restricted to only citing evidence already present in the context.
    - **Design Motivation**: A single LLM is prone to "self-persuasion." Multi-agent adversarial reviews using consistent evidence suppress hallucinations. Restricting the DecisionAgent to existing evidence builds traceability into the training paradigm.

3.  **Modality-Aware Scoring Function and Risk Vectors**:
    - **Function**: Expresses candidate scoring logic as a modality-dependent penalty formula while maintaining an independent vector of typed risk labels.
    - **Mechanism**: Candidate scores are conceptually written as $S_0(r) = s_{\text{mode}}(r) - p_{\text{TM}}(r) - p_{\text{PTM}}(r) - p_{\text{motif}}(r)$, where $s_{\text{mode}}$ represents modality base preference—epitopes prefer polar, non-TM, low-PTM windows, while pockets prefer hydrophobic cores with weaker PTM penalties. Simultaneously, an independent vector $g(r)$ outputs typed risk labels such as `TM-overlap`, `PTM-overlap`, `glyco-mask-overlap`, `PTM-dense`, `disulfide-constrained`, `hydrophobic-core`, and `motif-overlap`.
    - **Design Motivation**: Decoupling scores from risks makes rankings comparable while keeping failure reasons explainable—candidates with the same score can be clearly distinguished by operators (e.g., "no TM contact" vs. "buried in a glycosylation cluster").

### Loss & Training
The authors initially attempted SFT using structured demonstrations (Qwen3-235B Instruct) but found that while SFT improved output formatting, it introduced shortcut behaviors like "repeatedly picking N-terminal similar windows." Consequently, the main report is based on base model inference. The authors note that "biologically grounded reward or preference signals" are needed for future post-training.

## Key Experimental Results

### Main Results
| Dataset | Setting | Site4Drug Top-1 | Site4Drug Top-5 | Baseline |
|---------|---------|-----------------|-----------------|----------|
| RCSB Co-crystal Pockets (n=63) | $p<0.05$ Significance Rate | 20/63 | 18/63 | fpocket+AlphaFold3: 20/63; fpocket+RCSB w/ Ligand: 62/63 |
| ABCD Antibody Epitopes (n=26) | $p<0.05$ Significance Rate | 8/26 | 11/26 | — |

Without direct access to structures, Site4Drug ties with "feeding AlphaFold3 structures to fpocket." fpocket achieves 62/63 on ligand-containing RCSB structures because the ligand orientation leaks the answer to the geometric detector. GO enrichment shows that Top-1 significant targets primarily cluster in the kinase family, aligning with the biological prior that kinases possess recurring small-molecule-accessible pockets.

### Ablation Study
| Configuration | Top-1 Significance Rate | Top-5 Significance Rate | Description |
|---------------|-------------------------|-------------------------|-------------|
| Site4Drug Full Pipeline | 20/63 | 18/63 | Topology/PTM/Motif/Cys Evidence + Specialists |
| Sequence Only + $k=1$ (No Evidence) | 3/63 | 3/63 | Only Generic ID and Sequence |
| Sequence Only + $k=3$ Self-Consistency | 7/63 | 6/63 | Three Nomination Votes |
| Structural Confidence (AF3 pLDDT) | — | — | Top-1 Avg pLDDT > Top-5 Set Avg; only 9 inverse cases |

### Key Findings
- The explicit evidence pipeline is an order of magnitude better than "prompt engineering + sequence input" (20 vs. 3–7), proving that Site4Drug's gain comes from enforced feasibility constraints rather than general LLM sequence patterns.
- Despite not using 3D structures directly, the average pLDDT of predicted sites is higher than the average of the Top-5 regions, suggesting that aggregated sequence evidence implicitly recovers "structurally reliable regions."
- End-to-end demo on EGFR: Top-1 pocket fed to DrugCLIP retrieved small molecules with a hypergeometric test $p < 10^{-11}$ for overlap with the lapatinib binding site. Top-1 epitope fed to BoltzGen generated peptide binders where only Rank-1 calculated LIS at the PAE $<12$ threshold; Rank-2 fell into Domain III, a known antibody target.
- Kinases dominate Top-1 significant samples (e.g., pralsetinib targets 11 kinases including DDR1, FGFRs, RET, etc.), matching the biological prior of recurring kinase pockets and suggesting Site4Drug captures family-level geometric/sequence features rather than random hits.
- Automated modality recommendation identifies mixed-modality targets like EGFR and HER2 (where both small molecules and antibodies exist), avoiding pitfalls caused by manually preset modalities.

## Highlights & Insights
- "Auditability" is upgraded from a post-hoc report to a design goal: the scoring function, risk labels, and Agent critiques are all forced to be based on the explicit evidence list, allowing errors to be traced to specific evidence lines. This style is crucial for strictly regulated drug discovery.
- Skepticism towards ground truth: The authors explicitly state that IEDB epitopes and RCSB "4Å ligand neighborhoods" are task-dependent rather than exhaustive labels, thus using hypergeometric tests for relative comparison rather than absolute regression—a paradigm shift valuable for scientific discovery tasks lacking gold standards.
- Modality before site: Instead of requiring users to specify "antibody or small molecule," the system allows the same evidence to recommend the modality, avoiding typical errors like designing antibodies for the intracellular domain of membrane proteins.

## Limitations & Future Work
- Limited data scale (63 pockets, 26 epitopes) and potential data leakage risks with structural models like RAPID-Net (trained on scPDB) prevent fair head-to-head comparisons.
- Processes only single sequences and cannot handle quaternary structures—though many channel protein drugs target multi-subunit assemblies. The authors argue LLM-Agents are more easily extended to "multi-chain metadata" than single-sequence geometric tools.
- SFT post-training shows N-terminal window shortcuts, suggesting a need for biological reward signals. Risks of topology inversion under concentration dependence or partial sequence input were not covered in the current evaluation.

## Related Work & Insights
- **vs. fpocket / RAPID-Net**: Geometric/structural methods approach the ceiling on ligand-containing structures but cannot incorporate heterogeneous metadata like PTMs or Motifs, nor can they perform epitope discovery directly. Site4Drug offers a unified framework and traceable logs, though its absolute significance upper bound is limited by sequence evidence.
- **vs. DrugCLIP / BindCLIP / BoltzGen**: These works assume known sites for binder scoring or generation. Site4Drug is orthogonal and upstream, with its Module 2 interface demonstrated in the paper.
- **vs. AI Scientist / Virtual Lab Agents**: Also falls under "LLM Agents for scientific decision-making," but Site4Drug defines domain feasibility constraints most granularly, representing a specialized application of Agentic science in drug targeting.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefines site selection as an auditable Agent decision problem with a clear framework.
- Experimental Thoroughness: ⭐⭐⭐ Dataset size is limited; only one target has a full end-to-end demo.
- Writing Quality: ⭐⭐⭐⭐ Fluent narrative on evidence flow, modality recommendation, and auditability; ample appendix details.
- Value: ⭐⭐⭐⭐ Directly provides interface and log standards landing value for real-world drug discovery pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Holo Pockets to Electron Density: GPT-style Drug Design with Density](from_holo_pockets_to_electron_density_gpt-style_drug_design_with_density.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/computational_biology/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection](../../NeurIPS2025/computational_biology/unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)

</div>

<!-- RELATED:END -->
