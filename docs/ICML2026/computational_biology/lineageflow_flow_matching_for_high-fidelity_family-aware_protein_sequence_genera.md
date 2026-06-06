---
title: >-
  [Paper Note] LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation
description: >-
  [ICML 2026][Computational Biology][Flow Matching] The authors replace generic uniform/mask noise priors with family-specific Dirichlet priors derived from Ancestral Sequence Reconstruction (ASR). This allows Dirichlet fl…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Flow Matching"
  - "Dirichlet Prior"
  - "Ancestral Sequence Reconstruction (ASR)"
  - "Protein Families"
  - "Directed Evolution"
date: 2026-05-08
content_hash: 3e807811eb5c8282
---

# LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation

**Conference**: ICML 2026  
**arXiv**: [2605.22252](https://arxiv.org/abs/2605.22252)  
**Code**: https://github.com/Jinx-byebye/LineageFlow (Available)  
**Area**: Protein Generation / Flow Matching / Phylogenetic Priors  
**Keywords**: Flow Matching, Dirichlet Prior, Ancestral Sequence Reconstruction (ASR), Protein Families, Directed Evolution

## TL;DR
The authors replace generic uniform/mask noise priors with family-specific Dirichlet priors derived from Ancestral Sequence Reconstruction (ASR). This allows Dirichlet flow matching to initiate structured mutations from an "evolved scaffold." By inserting a mutate–select–amplify rerouting step at an intermediate time, the model achieves a family recognition accuracy close to natural sequences (95.3% vs 96.6%) across 8,886 Pfam families, while maintaining high novelty and folding confidence.

## Background & Motivation

**Background**: Protein sequence generation is currently dominated by two approaches: large-scale protein language models (e.g., ESM, ProtT5, ProGen) and discrete diffusion/flow matching models on the amino acid simplex (e.g., EvoDiff, DFM, ProtBFN). When the goal is to generate new sequences for a specific Pfam family, the standard practice is to condition the denoiser with a family label or an MSA prompt.

**Limitations of Prior Work**: Existing discrete generative models typically default to a "universal prior"—either starting from a uniform distribution on the simplex or masking all positions. However, protein families are characterized by **site-specific** evolutionary constraints: some sites are highly conserved to maintain structure/catalysis, while others vary to accommodate functional diversity. Universal priors collapse this structure, forcing the denoiser to synthesize every conserved residue "from scratch" from a nearly random state, which places extreme pressure on early time steps. This contradiction is evident in experiments: even with explicit family labels, DFM and EvoDiff achieve **0%** family recognition accuracy using profile-HMMs, with pLDDT scores around 45. The MSA-prompt-based model PoET also yields 0% accuracy.

**Key Challenge**: The absence of evolutionary structure in the prior $\leftrightarrow$ family information exists only as a label or prompt and does not penetrate the generative trajectory. The conditioning signal "stays outside," while the denoiser performs from-scratch synthesis internally.

**Goal**: To "bake" family conditions directly into the prior itself, ensuring that at $t=0$, the process already starts on the family manifold. The model then learns the mutation process from "ancestor $\to$ extant sequence" rather than a "noise $\to$ protein" synthesis process.

**Key Insight**: Molecular evolution provides existing tools: building MSAs for homologous sequences, inferring phylogenetic trees with IQ-TREE, and performing Ancestral Sequence Reconstruction (ASR) at the root using PAML. This yields a site-specific amino acid posterior that is near one-hot at conserved sites and maintains high entropy at variable sites—effectively a "family scaffold."

**Core Idea**: Use the ASR root posterior as a family-specific Dirichlet prior $q_0^{(h)}$ at the start of the flow matching trajectory on the simplex toward extant sequences. An intermediate mutate–select–amplify step, mimicking directed evolution, is added for goal-oriented sampling.

## Method

### Overall Architecture
LineageFlow interprets the generative trajectory $t \in [0, 1]$ as an evolutionary timeline from family ancestors to extant leaf nodes:

- **Preprocessing**: For each of the 8,886 Pfam families $h$, an MSA is constructed, a maximum likelihood tree is inferred via IQ-TREE, and marginal ASR is performed at the root via PAML to obtain the amino acid posterior for each site $l$. This is encoded into Dirichlet concentration parameters $\boldsymbol{\alpha}^{(h,l)} \in \mathbb{R}^K_{>0}$ ($K=20$).
- **Loss & Training**: All families share a single denoiser $\hat{p}_\theta(\mathbf{X}_1 \mid \mathbf{X}_t, t)$. For each training sequence, $t \sim \mathcal{U}[0,1]$ is sampled, and intermediate states are drawn from the family's Dirichlet path $p_t^{(h,l)}(\mathbf{x} \mid \mathbf{e}_i) = \mathrm{Dir}(\mathbf{x}; \boldsymbol{\alpha}^{(h,l)} + t_{\max} t \cdot \mathbf{e}_i)$. The model learns endpoint classification via sequence-averaged cross-entropy. $t_{\max}=6$.
- **Mechanism (Sampling)**: ① Base flow ($t \in [0, t_{\mathrm{int}}]$): Start from $q_0^{(h)}$ and integrate along the learned vector field; ② Rerouting ($t = t_{\mathrm{int}}=0.5$): Maintain a swarm of particles and perform several rounds of mutate $\to$ select $\to$ amplify; ③ Refinement ($t \in [t_{\mathrm{int}}, 1]$): Continue integration from the selected particles to the endpoint.

### Key Designs

1. **ASR Ancestral Dirichlet Prior**:
    - **Function**: Directly embeds family evolutionary constraints into the starting point $q_0$ of the generation process.
    - **Mechanism**: For each family $h$, site $l$ maintains Dirichlet concentrations $\boldsymbol{\alpha}^{(h,l)}$ encoding the ASR root posterior. The sequence-level family prior is a product of site-independent Dirichlets $q_0^{(h)}(\mathbf{X}) = \prod_l \mathrm{Dir}(\mathbf{x}^{(l)}; \boldsymbol{\alpha}^{(h,l)})$, and the global prior is a mixture of families $q_0 = \sum_h \pi_h q_0^{(h)}$. The conditional path is no longer the DFM path $\mathrm{Dir}(\mathbf{x}; \mathbf{1} + t_{\max} t \cdot \mathbf{e}_i)$ but $\mathrm{Dir}(\mathbf{x}; \boldsymbol{\alpha}^{(h,l)} + t_{\max} t \cdot \mathbf{e}_i)$. Consequently, the transport velocity $c_h^{(l)}(z, t)$ from the continuity equation becomes family- and target-residue-specific, with a closed-form solution involving the regularized incomplete Beta function.
    - **Design Motivation**: Compared to MSA column frequencies, ASR root posteriors are corrected by the phylogenetic tree, removing sampling redundancy. Compared to picking an extant MSA sequence as a starting point, the root posterior preserves uncertainty at truly variable sites, preventing the generation from being locked to training samples. Experiment §6.3 uses a Bayes-oracle to prove that in the "difficulty zone" ($t \le 0.2$), the upper bound of recoverable signals under the ASR prior is significantly higher than under a uniform prior, raising the performance ceiling for any denoiser.

2. **Classifier-Parameterized Family-Specific Vector Field**:
    - **Function**: Uses a shared denoising classifier to reconstruct specific continuous transport fields for every site in every family.
    - **Mechanism**: The training objective is $\mathcal{L}(\theta) = \mathbb{E}[-\frac{1}{|\mathcal{V}|}\sum_l \log \hat{p}_\theta(\mathbf{x}_1^{(l)} \mid \mathbf{X}_t, t)]$, which is standard categorical cross-entropy. During inference, the drift is combined as $\hat{\mathbf{v}}^{(h,l)} = \sum_i \mathbf{u}_t^{(h,l)}(\mathbf{x}^{(l)} \mid \mathbf{e}_i) \cdot \hat{p}_\theta(\mathbf{x}_1^{(l)} = \mathbf{e}_i \mid \mathbf{X}, t)$. Gap columns in MSAs are masked (excluded from the alphabet and loss), and variable lengths are handled via empirical gap rate resampling.
    - **Design Motivation**: By delegating "family specificity" to the analytical $\boldsymbol{\alpha}^{(h,l)}$ and $c_h^{(l)}$, the denoiser itself does not require family labels or family-specific heads. This allows training on the full Pfam set on a single machine (4×4090 for 26 hours per epoch). It also locates quality bottlenecks at specific time intervals where the classifier performs poorly (confirmed as $t \le 0.2$ in §6.3).

3. **Rerouting: Intermediate Mutate–Select–Amplify**:
    - **Function**: Pushes the particle swarm towards user-specified fitness goals at $t_{\mathrm{int}}$ without requiring gradients.
    - **Mechanism**: The ODE is paused at $t_{\mathrm{int}}=0.5$. A swarm of particles undergoes (i) mutate: injecting diversity via a proposal kernel $\mathcal{K}$, (ii) select: reweighting by $\exp(\beta J)$, and (iii) amplify: resampling based on weights. The target distribution is the exponentially tilted $p^{\mathrm{sel}} \propto (p_{t_{\mathrm{int}}} \mathcal{K})(\mathbf{X}) \exp(\beta J(\mathbf{X}))$. Proposition 5.2 proves this is the optimal solution for the KL-constrained optimization $\max_q \mathbb{E}_q[J] - \frac{1}{\beta} \mathrm{KL}(q \| p^{\mathrm{mut}})$. After selection, particles continue integration to $t=1$.
    - **Design Motivation**: Continuous guidance (classifier guidance, SMC) requires computing gradients or resampling at every Euler step, which is expensive and may push samples off the manifold. Inserting "artificial selection" only at the intermediate stage preserves the family trajectory while injecting target bias. $t_{\mathrm{int}}$ serves as a critical knob: too early, and samples are too noisy for meaningful selection; too late, and samples are near one-hot with no room for change. 0.5 offers the best compromise.

### Loss & Training
The model uses sequence-averaged cross-entropy $\mathcal{L}(\theta)$ (Eq. 6) with $t$ sampled uniformly and $t_{\max}=6$. Training is conducted on Pfam-A RP35 (8,886 families, 8.94M aligned sequences) with a 5% within-family hold-out. Training took ~26 hours for 1 epoch on 4×RTX 4090 with a learning rate of $10^{-5}$ and an equivalent batch size of 128.

## Key Experimental Results

### Main Results (Pfam unconditional, 1,024 sequences/method)

| Method | $\mathrm{Acc}_{\mathrm{fam}}$↑ | pLDDT↑ | scPPL↓ | Novelty@0.6↑ | Diversity↑ |
|------|------|------|------|------|------|
| Pfam held-out (Ceiling) | 96.6 | 86.4 | 5.02 | 12.6 | 806 |
| DFM (Uniform Prior) | 0.0 | 46.2 | 12.62 | — | 90 |
| EvoDiff (Mask Prior) | 0.0 | 45.4 | 12.60 | — | 54 |
| PoET (MSA prompt) | 0.0 | 52.0 | 13.76 | — | 47 |
| ProtBFN† (8× Larger Corpus) | — | 71.9 | 5.91 | 64.0 | 604 |
| ASR-PSSM iid (Prior only) | 92.8 | 70.8 | 7.08 | 32.0 | 378 |
| LineageFlow w/o rerouting | 93.0 | 69.6 | 7.96 | 52.0 | 440 |
| **LineageFlow (Full)** | **95.3** | **76.6** | **6.67** | **48.9** | **587** |

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| DFM / EvoDiff + Family Label | $\mathrm{Acc}_{\mathrm{fam}}=0$ | Family signals exist in labels but not priors $\to$ complete failure. |
| ASR-PSSM iid | $\mathrm{Acc}_{\mathrm{fam}}=92.8$, pLDDT 70.8 | The prior alone carries strong family signals. |
| w/o rerouting | pLDDT 69.6, Novelty@0.6 52.0 | Base flow maintains family identity but offers higher novelty/diversity than the pure prior. |
| + rerouting | pLDDT 76.6 (+7), Acc 95.3 (+2.3) | Rerouting is the primary driver of plausibility. |
| Token acc in difficulty zone (§6.3) | LF ≫ DFM | ASR prior raises the Bayes upper bound and denoising accuracy. |

### Key Findings
- **Prior is Paramount**: DFM/EvoDiff fail at 0% family recognition even with labels. Simply replacing the prior with the ASR root posterior—without learning any flow—jumps recognition to 92.8%. This supports the core thesis: "Evolutionary structure in the prior is an irreplaceable conditioning signal."
- **Rerouting as a Plausibility Lever**: While the base flow inherits family signals and improves novelty, it lacks folding confidence. Inserting mutate–select–amplify raises pLDDT by 7 points. PCA visualizations show the particle swarm shifting toward true sequence clusters.
- **Sweet Spot at $t_{\mathrm{int}}=0.5$**: Earlier selection is meaningless due to unformed samples; later selection is restricted by one-hot convergence. 0.5 provides the best "plasticity $\times$ structure" trade-off.
- **Zero-shot Enzyme Generation**: On three held-out enzyme families (2OG-FeII_Oxy, Trp_syntA, RNase_HII), $q_0^{(h)}$ was recalculated using MSAs without retraining $\theta$. Generated sequences retained catalytic motifs. Rerouting using ESM-2 pseudo-likelihood improved proxies for solubility and thermal stability (DeepSol/Meltome) even though rerouting did not optimize these specifically.

## Highlights & Insights
- **"Changing the Prior" is more fundamental than "Adding Guidance"**: For conditions with strong structure (like family-level conservation), embedding them into the prior is more effective than feeding them into the denoiser. This logic applies to any task where the prior itself has structure (e.g., codon usage, chemical reaction templates).
- **Leveraging Classical Biological Tools**: IQ-TREE and PAML are mature phylogenetic packages. Instead of reinventing ancestral inference, the authors cleverly package the root posterior as Dirichlet concentrations—a great case of "importing external inductive bias into the prior."
- **Formalizing Directed Evolution via Rerouting**: The one-time mutate–select–amplify step is equivalent to exponentially tilted sampling with KL regularization (Proposition 5.2). This provides both a theoretical framework and a practical particle filter algorithm for intermediate intervention.
- **Diagnosing the Difficulty Zone**: §6.3 attributes generation bottlenecks to the Bayes upper bound at early time steps, which may inspire other diffusion models to focus on "prior engineering" rather than just "increasing network capacity."

## Limitations & Future Work
- **Dependency on MSA Quality**: Since all $\boldsymbol{\alpha}^{(h,l)}$ are derived from MSAs and phylogenetic trees, this method cannot handle single sequences, novel protein families, or orphan proteins. Open-world generation requires addressing the "no MSA" prerequisite.
- **Fixed Alignment Coordinates**: Generation occurs within the coordinates of family-aligned columns, meaning length variations and indels are not explicitly modeled, making it unsuitable for domain recombination scenarios.
- **Reliance on Computational Proxies**: Foldability is evaluated via OmegaFold pLDDT, and stability/solubility via ESM-2-based models, without wet-lab verification. Thermal stability proxy accuracy is cited at only ~50%, so zero-shot results should be interpreted cautiously.
- **Rerouting Fitness Function Dependency**: Currently, ESM-2 pseudo-likelihood is used as $J$. While effective, this is a tautology for plausibility. Efficiency for more complex fitness functions (e.g., docking or binding specificity) remains to be seen.
- **Sampling Overhead**: Rerouting increases the sampling time for 512 sequences from 759s to 1047s. While within the same order of magnitude, the swarm size and iterations are not yet optimized.

## Related Work & Insights
- **vs DFM (Stark et al., 2024)**: DFM uses a uniform $\mathrm{Dir}(\mathbf{1})$ prior. This paper proves the prior is the bottleneck, as switching to $\mathrm{Dir}(\boldsymbol{\alpha}^{(h,l)})$ jumps family recognition from 0% to 95%.
- **vs EvoDiff**: Mask-based discrete diffusion. Both suffer from the inability of labels to penetrate the prior. This work highlights the extreme importance of noise prior design in discrete generation.
- **vs ProtBFN (Atkinson et al., 2025)**: BFN uses 71M sequences to achieve folding quality. LineageFlow achieves higher pLDDT on an 8× smaller corpus, proving "correct prior" > "more data."
- **vs PoET (Truong & Bepler, 2023)**: MSA prompt method. Experiments show prompts don't necessarily translate to family recognition, whereas LineageFlow achieves this via the prior.
- **vs Classifier Guidance / RL Fine-tuning**: Instead of step-by-step guidance or RL, rerouting offers a "single intermediate KL-constrained selection" paradigm that is both lightweight and theoretically grounded.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Changing the paradigm by using ASR root posteriors as Dirichlet priors is a brilliant move; rerouting provides a clean non-gradient guidance framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full Pfam training and comprehensive ablation studies across 5 research questions. Zero-shot enzyme cases are meaningful; lacks wet-lab validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear reasoning chain (noise prior collapses structure $\to$ ASR prior preserves it $\to$ difficulty zone ceiling raised), integrating method, theory, and experiments.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for protein engineering, with insights on prior-based conditioning applicable to all discrete generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/computational_biology/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/computational_biology/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[ICML 2026\] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](temporal_score_rescaling_for_temperature_sampling_in_diffusion_and_flow_models.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)

</div>

<!-- RELATED:END -->
