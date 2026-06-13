---
title: >-
  [Paper Note] Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators
description: >-
  [ICML 2026][Computational Biology][Protein Language Models] This paper incorporates residue embeddings from pre-trained protein language models (pLM) directly into Transferable Implicit Transfer Operators (TITO). The res…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Protein Language Models"
  - "Implicit Transfer Operators"
  - "Flow Matching"
  - "Coarse-grained MD"
  - "Cross-system Generalization"
date: 2026-05-08
content_hash: 3dad6818c88e2da1
---

# Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators

**Conference**: ICML 2026  
**arXiv**: [2602.11216](https://arxiv.org/abs/2602.11216)  
**Code**: https://github.com/PanosAntoniadis/platito (Available)  
**Area**: Scientific Computing / Molecular Dynamics / Generative Models / Protein Representation  
**Keywords**: Protein Language Models, Implicit Transfer Operators, Flow Matching, Coarse-grained MD, Cross-system Generalization

## TL;DR
This paper incorporates residue embeddings from pre-trained protein language models (pLM) directly into Transferable Implicit Transfer Operators (TITO). The resulting model, PLaTITO, trained on mdCATH with only 56 ms of trajectories and 1100 GPU hours, allows a coarse-grained $C_\alpha$ model with as few as 19M parameters to comprehensively outperform BioEmu in equilibrium sampling of outlier systems such as fast-folding proteins.

## Background & Motivation
**Background**: Molecular Dynamics (MD) is a core tool in computational biology, requiring sampling from the Boltzmann distribution $\mu(x)\propto\exp(-\beta U(x))$ to estimate thermodynamic and kinetic observables. Classical MD is limited by femtosecond integration steps, making it nearly impossible to bridge to microsecond or second-scale relaxation times. Recent Generative MD (GenMD) follows two paths: Boltzmann Generators/Emulators (BG/BE, e.g., BioEmu) directly learn $p(x|S,T)$, while Implicit Transfer Operators (ITO/TITO) learn the long-time transition density $p(x_{t+\Delta t}|x_t,\Delta t,S,T)$.

**Limitations of Prior Work**: BG/BE requires compressing the entire monomeric folding space into a neural network, necessitating massive trajectories and protein structure supervision for cross-system generalization (BioEmu used 31M parameters, 216 ms of trajectories, and 9216 GPU hours, with additional supervision from 131k AFDB structures and 502k experimental $\Delta G$). TITO is more data-efficient by utilizing the autocorrelation structure of MD, but small models focusing only on coordinates still struggle with novel protein sequences, hindered by the lack of priors on "what a residue looks like."

**Key Challenge**: Generalization capability vs. Training cost—covering the vast protein sequence space requires either massive data/parameters or an efficient inductive bias injection channel.

**Goal**: (i) Verify if the TITO framework itself is more data-efficient than BE; (ii) Identify an auxiliary conditioning signal that significantly improves outlier generalization without increasing test-time computational costs; (iii) Check if temperature conditioning allows the model to spontaneously learn authentic non-Arrhenius folding kinetics.

**Key Insight**: pLMs (e.g., ESM Cambrian), pre-trained on billions of unlabeled sequences, have been repeatedly proven to implicitly encode evolutionary, structural, stability, and even thermodynamic information in their residue embeddings. This provides the "physical prior" missing in TITO, and since embeddings can be pre-computed offline, they incur almost zero cost.

**Core Idea**: Injecting pLM residue embeddings $e_{\text{seq}}=\phi_{\text{pLM}}(S)$ as conditions into the TITO transition density network is equivalent to letting the coarse-grained $C_\alpha$ model "integrate dynamics while observing evolutionary priors," thereby supporting stronger cross-protein generalization with less trajectory data.

## Method
PLaTITO performs conditional extension within the ITO framework: given the current $C_\alpha$ backbone $x_t\in\mathbb{R}^{3L}$, amino acid sequence $S$, temperature $T$, and time step $\Delta t$, it learns the long-time transition density $p(x_{t+\Delta t}|x_t,\Delta t,S,T)$, augmented by three pluggable auxiliary embeddings.

### Overall Architecture
Input: Paired coarse-grained conformations $(x_t,x_{t+\Delta t})$ from mdCATH, sequence $S$, temperature $T$, and time interval $\Delta t$.
Output: $x_{t+\Delta t}$ is obtained by integrating an ODE with a learned velocity field starting from $z_0 \sim \mathcal{N}(0,I)$ via iterative roll-outs; multiple roll-outs are concatenated to form long trajectories.
Mechanism: The process involves two stages. First, a conditioning network $f_c$ extracts a residue-level conditional representation $c\in\mathbb{R}^{L\times d}$ from $(x_t,\Delta t,S,T,e_{\text{seq}},e_{\text{struct}},A_{LLM})$. Second, a velocity network $f_v(z_s,s,c)$ predicts the velocity $v\in\mathbb{R}^{3L}$ at flow-matching time $s\in[0,1]$. Both networks are Proteina-style non-equivariant Transformers, with residue-pair representations used as attention biases.

Training Target: Using rectified-flow linear interpolation $z_s=s\,z_1+(1-s)\,z_0$, where $z_0\sim\mathcal{N}(0,I)$ and $z_1=x_{t+\Delta t}$. The conditional flow matching loss is:
$$\mathcal{L}(\theta)=\mathbb{E}_{x_t,x_{t+\Delta t},s,z_0}\|v^{\theta}(z_s;s,x_t,\Delta t,S,T)-(x_{t+\Delta t}-z_0)\|^2$$
Inference: An ODE pushes noise to the next conformation. 1000 steps with $\Delta t=1\,\mathrm{ns}$ generate a 1 µs trajectory.

### Key Designs

1. **Coarse-grained $C_\alpha$-only TITO Backbone + Flow Matching Long-step Integration**:
    - **Function**: Folds monomeric proteins into $L\times 3$ residue coordinate sequences, directly learning long-step transition densities over nanoseconds or tens of nanoseconds, bypassing femtosecond integration.
    - **Mechanism**: Eliminates side chains to retain only $C_\alpha$, allowing all 4482 mdCATH domains to fit into a single architecture. The time step $\Delta t$ is treated as a learnable embedding, shared with sequence/temperature, allowing a single network to share parameters across multiple time scales. Sampling uses the constant velocity field of rectified-flow $u_s=z_1-z_0$, where the ODE step is controlled by $s$ rather than physical time, converting kinetic sampling into a generation problem with low noise steps.
    - **Design Motivation**: BE directly sampling $p(x|S,T)$ discards the temporal autocorrelation of MD, wasting expensive trajectory signals. TITO leverages correlations between adjacent frames to extract many $(x_t,x_{t+\Delta t})$ training pairs, allowing a 3M parameter model to match BE performance with equivalent data.

2. **pLM Residue Embeddings as Zero-Overhead Sequence Priors**:
    - **Function**: Residue-level embeddings $e_{\text{seq}}\in\mathbb{R}^{L\times d_{\text{pLM}}}$ from ESM Cambrian (300M for PLaTITO, 6B for PLaTITO-19M) are injected into the conditioning network $f_c$.
    - **Mechanism**: Embeddings are forward-computed once and cached before training; the pLM is not called during training or inference. Embeddings are concatenated into the residue representation of $f_c$ and fused with coordinate information via attention. This allows the network to "know" which residues favor $\beta$-sheets, which are prone to exposure, or which segments are disordered loops before seeing any MD trajectories for that protein.
    - **Design Motivation**: pLM embeddings have been shown (e.g., by Meier/Frazer) to encode variant stability and functional information, precisely the chemical/evolutionary priors missing in coarse-grained $C_\alpha$ models. Implementing this as a pluggable conditional channel rather than replacing the backbone retains TITO's lightweight nature, resulting in almost no test-time overhead ($\le 5\%$) while reducing MAE for outlier proteins from 1.068 to 0.949.

3. **Structural Embeddings and LLM Annotations as Control Experiments**:
    - **Function**: Two additional auxiliary conditions are attached—structural embeddings $e_{\text{struct}}=\phi_{\text{PSM}}(x_t)$ from Proteina 60M, and LLM-based binary judgments $S_{LLM}$ and confidence $C_{LLM}$ regarding the feasibility of simulating the protein.
    - **Mechanism**: Structural embeddings must be computed online (dependent on current $x_t$), so a relatively light 60M model was chosen. LLM annotations prompt an LLM with dataset metadata (domain length, sub-cellular location, known functions, presence of binding partners) and use the responses as additional condition embeddings.
    - **Design Motivation**: The authors sought to isolate gains from "sequence priors," "geometric priors," and "data quality priors." Results indicate that the first two are complementary, while LLM annotations unexpectedly degraded performance, suggesting that current prompt engineering cannot extract signals stronger than the model's own inductive bias from metadata.

### Loss & Training
The sole training objective is the conditional flow matching regression loss. Training data consists of 4471 domains from mdCATH with $\le 200$ residues (filtered at 40% sequence similarity to exclude test-set overlaps), totaling approximately 56 ms of aggregate trajectories. Random uniform sampling across temperatures and time steps allows the model to implicitly learn temperature-dependent kinetics. All models were trained on a single A100 80GB for 1100 GPU hours. PLaTITO-19M uses larger ESM Cambrian 6B embeddings and 19M backbone parameters but maintains the same data and compute budget.

## Key Experimental Results

### Main Results
The evaluation set consists of 12 fast-folding proteins (Lindorff-Larsen et al. 2011), all strictly non-overlapping in sequence with the training set. Metrics include MAE, RMSE, and Coverage (percentage of reference bins covered by the model) against reference MD free energy landscapes.

| Model | MAE ↓ | RMSE ↓ | Coverage ↑ | Params | GPU Hours | MD Data |
|------|------|------|------|------|------|------|
| TITO (baseline) | 1.068 | 1.382 | 0.590 | 3 M | 1100 | 56 ms |
| TITO + Struct | 1.004 | 1.310 | 0.560 | 3 M | 1100 | 56 ms |
| **PLaTITO** | 0.949 | 1.228 | 0.651 | 3 M | 1100 | 56 ms |
| PLaTITO + Struct | 0.938 | 1.213 | 0.655 | 3 M | 1100 | 56 ms |
| PLaTITO + Struct + LLM | 1.066 | 1.346 | 0.570 | 3 M | 1100 | 56 ms |
| **PLaTITO-19M** | **0.824** | **1.099** | **0.666** | 19 M | 1100 | 56 ms |
| Emu (BE arch) | 1.305 | 1.639 | 0.529 | 3 M | 1100 | 56 ms |
| BioEmu | 1.110 | 1.389 | 0.594 | 31 M | 9216 | 216 ms + structures/$\Delta G$ |

PLaTITO-19M achieves a 26% lower MAE than BioEmu, using ~4x less training data and 8x fewer GPU hours.

### Ablation Study

| Configuration | Key Observation |
|------|---------|
| TITO vs. Emu (Same arch & compute) | MAE 1.068 vs. 1.305; proves leveraging MD temporal autocorrelation is more data-efficient than pure Boltzmann learning. |
| +Struct (Proteina 60M embeddings) | Consistently slight improvements across all metrics; structural and sequence priors are complementary, though online compute adds overhead. |
| +Seq (ESM Cambrian 300M) | MAE 1.068→0.949, Cov 0.590→0.651; pLM embeddings are the largest single-point contributor with zero test-time cost. |
| +LLM Annotations | MAE regressed to 1.066; current prompt+metadata is insufficient for useful conditioning and introduces noise. |
| Cambrian 300M→6B + 3M→19M backbone | MAE further dropped to 0.824; scaling representation and model capacity provides additive benefit. |

### Key Findings
- **pLM embeddings are the single largest point of leverage**: Reverting PLaTITO to TITO caused the sharpest performance drop (MAE +0.119, Cov −0.061). Given the zero inference overhead, this suggests "cheap pre-trained priors + physical kinetic backbones" is a sweet spot for GenMD.
- **Structural and sequence embeddings are complementary**: Combining both yields better results than using either alone, but structural embeddings require re-computation at each step, making them less cost-effective than pLMs.
- **LLM annotation counter-example**: Adding DeepSeek's stability judgments within the same framework degraded performance. This major negative result warns that prompt engineering currently cannot reliably translate protein context into kinetic inductive biases.
- **Physical temperature dependence**: PLaTITO-19M's predicted folding/unfolding timescales across 320–440 K deviate from the Arrhenius curve, matching the "rugged landscape" of real folding kinetics (e.g., BBA, Villin), showing temperature conditioning is physically utilized.
- **Cryptic pocket formation**: On 4 cryptic pocket benchmarks, starting from holo states, the model sampled apo-like conformations, successfully recovering 1 case where BioEmu failed.

## Highlights & Insights
- **"Cheap Priors + Lean Backbone > Giant End-to-End"**: Using existing ESM Cambrian embeddings as zero-overhead conditions allows a 19M $C_\alpha$ TITO to outperform a 31M all-atom BioEmu with an order of magnitude less data and compute. This provides a practical paradigm: freeze a large protein foundation model and attach a task-specific TITO.
- **MD Autocorrelation $\neq$ Boltzmann Distribution**: The Emu vs. TITO comparison (same architecture, same data) is a clean experimental design, proving that discarding temporal correlation significantly wastes supervision signals—a point often overlooked in BE studies.
- **Valuable Negative Results**: Clearly reporting that LLM annotations degrade performance serves as a warning to the community attempting "LLM-as-conditioner" approaches; one must first verify the information density of the metadata itself.
- **Flow Matching + Multi-timestep Conditioning**: Treating $\Delta t$ as a learnable embedding rather than a fixed step allows a single TITO network to serve nanosecond to microsecond scales, a paradigm applicable to trajectory generation and video diffusion.

## Limitations & Future Work
- $C_\alpha$ coarse-graining ignores side chains, inherently limiting tasks like ligand specificity or allosteric regulation; all-atom extension is necessary.
- The ITO framework lacks formal guarantees for detailed balance, Chapman-Kolmogorov consistency, or stability; MFPT tends to be systematically underestimated (as suggested by the variational principle).
- Training data is limited to monomeric mdCATH; generalization across chemical spaces (complexes, small molecule interactions) and extreme thermodynamic conditions remains unverified.
- The failure of LLM annotations suggests a bottleneck in prompts and metadata; future work needs either finer prompt engineering or structured physical features (e.g., secondary structure abundance, SASA).
- Dependency on external pre-training (pLM, Proteina) makes this a "modular" pipeline. End-to-end self-training would require re-designing pre-training objectives to produce reusable embeddings.

## Related Work & Insights
- **vs. BioEmu (Lewis et al., 2025)**: BioEmu is a pure BE that learns $p(x|S,T)$, relying on 31M parameters and extra supervision to achieve generalization. PLaTITO uses the ITO route with pLM conditioning to be smaller, more efficient, and more accurate, but lacks BE's reweighting theoretical properties.
- **vs. TITO (Diez et al., 2026)**: While the original TITO backbone generalized across small peptides and molecules, this work proves significant untapped generalization space exists by injecting appropriate pre-trained representations.
- **vs. Boltzmann Generator (Noé et al., 2019) & Timewarp (Klein et al., 2023)**: Early BG/flow methods worked only on single systems or small peptides. This work demonstrates that "large cheap priors + compact kinetic networks" is a viable path for protein-scale applications.
- **vs. ESMFold/AlphaFold-like Static Prediction**: Static models provide only energy minima, whereas this work provides kinetic samples. These could be synergistic: using AF-like models to find minima and PLaTITO to sample the surrounding free energy landscape.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple concept of pLM injection into ITO, but the first systematic validation of three auxiliary conditions with clear controls.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five models, multiple pLM families, varying sizes, cross-temperature, and cryptic pocket tests.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of formulas and structure, though some appendices are heavily referenced.
- Value: ⭐⭐⭐⭐⭐ Outperforming BioEmu with 1/8th the compute and identifying "LLM annotation failure" provides strong guidance for the GenMD community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](../../ICLR2026/computational_biology/reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[ICML 2026\] Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design](cross-chirality_generalization_by_axial_vectors_for_hetero-chiral_protein-peptid.md)
- [\[ICLR 2026\] How to Make the Most of Your Masked Language Model for Protein Engineering](../../ICLR2026/computational_biology/how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](../../ICLR2026/computational_biology/protein_as_a_second_language_for_llms.md)

</div>

<!-- RELATED:END -->
