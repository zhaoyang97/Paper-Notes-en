---
title: >-
  [Paper Note] Towards A Generative Protein Evolution Machine with DPLM-Evo
description: >-
  [ICML 2026][Computational Biology][Protein Language Models] This paper proposes DPLM-Evo, extending discrete diffusion in protein language models from "mask-replacement only" to explicitly modeling three evolutionary edi…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Protein Language Models"
  - "Discrete Diffusion"
  - "Evolutionary Modeling"
  - "Variable-length Generation"
  - "Insertion/Deletion"
date: 2026-05-08
content_hash: 88db1deda1aa5fd2
---

# Towards A Generative Protein Evolution Machine with DPLM-Evo

**Conference**: ICML 2026  
**arXiv**: [2605.00182](https://arxiv.org/abs/2605.00182)  
**Code**: None  
**Area**: Protein Generation / Discrete Diffusion / Biomedicine  
**Keywords**: Protein Language Models, Discrete Diffusion, Evolutionary Modeling, Variable-length Generation, Insertion/Deletion  

## TL;DR
This paper proposes DPLM-Evo, extending discrete diffusion in protein language models from "mask-replacement only" to explicitly modeling three evolutionary edits: substitution, insertion, and deletion. By decoupling variable-length observed sequences into an upsampled latent alignment space combined with a contextualized evolutionary noise kernel, it achieves variable-length evolutionary generation and trajectory-based protein post-editing, while reaching SOTA on ProteinGym single-sequence variant effect prediction.

## Background & Motivation

**Background**: Protein Language Models (PLMs, e.g., ESM, ProGen, DPLM, DPLM-2) learn evolutionary constraints from large-scale sequence databases for applications such as zero-shot variant effect prediction, structure prediction, and sequence generation. Among these, discrete diffusion PLMs (DPLM series) outperform autoregressive PLMs in representation and generation due to their bidirectional receptive fields and long-range dependency modeling.

**Limitations of Prior Work**: Existing DPLMs use an absorbing-state (mask) as the forward noise kernel, simplifying generation to "iterative mask recovery." This fails to reflect biological reality—proteins do not emerge from masks but evolve through accumulated discrete edits (substitutions, insertions, deletions). Indels are crucial for reshaping loops, adjusting linker lengths, and generating or removing short motifs. Masked diffusion lacks native indel operations, uses a clumsy fixed-length framework, and cannot represent variable-length evolutionary trajectories or perform authentic post-editing on existing proteins.

**Key Challenge**: Standard discrete diffusion is defined on a fixed-dimensional categorical state space, whereas indels inherently change sequence length. These two mathematical structures are fundamentally incompatible.

**Goal**: Construct a unified discrete diffusion framework where both forward noise and reverse denoising explicitly express substitution, insertion, and deletion, supporting variable-length generation, evolutionary post-editing, and directed optimization of existing proteins.

**Key Insight**: Leverage latent alignment ideas similar to CTC or EditFlow. Decouple the variable-length observed sequence space $\mathcal{X}$ into an upsampled latent alignment space $\mathcal{Z}$ with length $2L$. The latter transforms the variable-length problem into a fixed-length problem by inserting gap tokens $\phi$. The diffusion process is defined on $\mathcal{Z}$, while the neural network only processes collapsed sequences in $\mathcal{X}$.

**Core Idea**: Use a unified transition matrix $\mathbf{Q}_{\mathrm{noise}}$ in the latent alignment space to encode three types of transitions ($\mathcal{A}\leftrightarrow\phi$) representing substitution, insertion, and deletion. This is supplemented by a "contextualized evolutionary noise kernel"—replacing uniform noise with conditional distributions predicted by the model itself to ensure the corruption process aligns with evolutionary preferences. During decoding, three independent heads (substitution, deletion, insertion) sequentially perform delete→insert→substitute→renoise operations to achieve variable-length denoising.

## Method

### Overall Architecture
The framework operates across two spaces: the observation space $\mathcal{X}=\mathcal{V}^L$ (where $\mathcal{V}=\mathcal{A}\cup\{\mathbf{m}\}$ includes masks) and the latent alignment space $\mathcal{Z}=(\mathcal{V}\cup\{\phi\})^{2L}$ (where $\phi$ is a gap). A collapse function $\Gamma^{-1}(\mathbf{z})$ removes all $\phi$ to restore $\mathbf{x}$ from $\mathbf{z}$, where the set $\Gamma(\mathbf{x})$ contains all valid alignments of $\mathbf{x}$. Forward diffusion $q_t(\mathbf{z}_t|\mathbf{z}_0)=\bar\alpha_t\delta_{\mathbf{z}_0}+(1-\bar\alpha_t)\pi(\mathbf{z}_0)$ occurs in the latent space. The network $f_\theta$ operates on $\mathbf{x}_t=\Gamma^{-1}(\mathbf{z}_t)$, with three heads simultaneously predicting the substitution distribution, deletion probability, and right-side insertion probability for each token. The ELBO objective is: $$\log p_\theta(\mathbf{x}_0)\geq\mathbb{E}_{\mathbf{z}_0\in\Gamma(\mathbf{x}_0)}[\mathbb{E}_{q_t}[\log p_\theta(\mathbf{z}_0|\mathbf{z}_t)]]$$.

### Key Designs

1.  **Latent Alignment Space for Variable-length Indels**:
    - **Function**: Models variable-length indel evolution within a fixed-dimensional computational framework while allowing the network to process compact observed sequences.
    - **Mechanism**: Upsamples the sequence by a factor of 2 and inserts gap placeholders $\phi$ (e.g., $[A,B,C]\mapsto[A,\phi,\phi,B,\phi,C]$). The forward diffusion process transitions symbols in $\mathbf{z}$, while reverse denoising projects back to the observation space via $\Gamma^{-1}$. The transition matrix $\mathbf{Q}_{\mathrm{noise}}$ is controlled by three hyperparameters $(\omega_{\mathrm{del}},\omega_{\mathrm{ins}},\rho_{\mathrm{mask}})$: a state in $\mathcal{A}$ transitions to another amino acid with probability $1-\omega_{\mathrm{del}}$ (or becomes a mask with $\rho_{\mathrm{mask}}$ ratio) and transitions to $\phi$ (deletion) with probability $\omega_{\mathrm{del}}$; a $\phi$ state transitions to an amino acid (insertion) with probability $\omega_{\mathrm{ins}}$.
    - **Design Motivation**: Defining a Markov process directly on variable-length sequences is extremely complex. Latent alignment encodes indels as simple token replacement problems, reusing established fixed-length diffusion toolchains. The $2x$ upsampling ensures net insertion does not exceed the original length $L$, covering typical protein engineering scenarios (loop/linker adjustment). This trick also allows DPLM-Evo to initialize from existing masked DPLM weights.

2.  **Contextualized Evolutionary Noise Kernel**:
    - **Function**: Ensures the forward noise reflects real evolutionary preferences (biologically "reasonable" substitutions at each site) rather than uniform randomness.
    - **Mechanism**: The substitution matrix $\mathcal{T}_{\mathrm{sub}}$ offers three choices: (i) uniform $\mathbf{U}_K=\tfrac{1}{K}\mathbf{1}\mathbf{1}^\top$; (ii) static biological prior $\mathbf{M}_{\mathrm{BLOSUM}}$; or (iii) contextualized $\mathcal{T}_{\mathrm{sub}}^{(j)}=\mathbb{E}_{q'_t(\mathbf{z}'_t|\mathbf{z}_0)}[p_\theta(\cdot|\mathbf{z}'^{\setminus j}_t,\mathbf{m})]$. The latter forces a mask at target site $j$, letting the model predict its identity based on the partially-masked context. After a warmup phase with simple mask noise, the system switches to self-prediction noise. At $t=1$, this degrades to a learnable prior $p_\theta(\cdot|\mathbf{m}^L)$.
    - **Design Motivation**: Uniform noise treats rare mutations like "Lys → Trp" with the same weight as common ones like "Lys → Arg," wasting capacity. Contextualized noise ensures the corruption encountered by the model resembles mutations seen in real evolution (conservative substitutions, contextual preferences), improving training efficiency and encouraging the model to capture homologous dependencies.

3.  **Tri-head Decoupling + Binary Classification Indel Training**:
    - **Function**: Prevents interference between substitution, deletion, and insertion tasks and resolves mode collapse caused by indel class imbalance.
    - **Mechanism**: An Index Mapping Function $\mathcal{I}:\{1,\dots,L_t\}\to\{1,\dots,N\}$ maps observed tokens back to latent positions. Three mutually exclusive losses are defined based on $(\mathbf{z}_t, \mathbf{z}_0)$ token combinations: $\mathcal{L}_{\mathrm{sub}}^{(k)}$ applies only when both are amino acids and differ (standard CE); indel losses are formatted as binary classifications: $\mathcal{L}_{\mathrm{del}}^{(k)}=\mathrm{BCE}(\mathbb{I}_{\mathbf{z}_0^{(\mathcal{I}(k))}=\phi},p_\theta^{\mathrm{del}})$ and $\mathcal{L}_{\mathrm{ins}}^{(k)}=\mathrm{BCE}(\mathbb{I}_{v_{\mathrm{next}}^{(k)}\neq\emptyset},p_\theta^{\mathrm{ins}})$. The total loss is $\mathcal{L}_t=\mathbb{E}[\sum_k\lambda_{t-1}(\gamma_{\mathrm{sub}}\mathcal{L}_{\mathrm{sub}}+\gamma_{\mathrm{del}}\mathcal{L}_{\mathrm{del}}+\gamma_{\mathrm{ins}}\mathcal{L}_{\mathrm{ins}})]$.
    - **Design Motivation**: Original multinomial indel losses led to deletion mode collapse and unstable insertion training because substitutions vastly outnumber indels in biological sequences. Decoupling "whether to delete/insert" from "what to substitute/insert" via binary classification stabilizes training.

### Loss & Training
- $\mathcal{L}_t=\mathbb{E}_{\mathbf{x}_0,\mathbf{z}_0,\mathbf{z}_t}[\sum_k\lambda_{t-1}(\gamma_{\mathrm{sub}}\mathcal{L}_{\mathrm{sub}}^{(k)}+\gamma_{\mathrm{del}}\mathcal{L}_{\mathrm{del}}^{(k)}+\gamma_{\mathrm{ins}}\mathcal{L}_{\mathrm{ins}}^{(k)})]$, with three $\gamma$ weights adjusting preferences for different evolutionary operations.
- Initialization from pretrained DPLM → Warmup phase with mask noise → Switch to contextualized noise kernel.
- Sampling: Localize a noisy index set $\mathcal{N}_t$, then sequentially: (i) delete sites where $p_\theta^{\mathrm{del}}>\tau_{\mathrm{del}}$; (ii) insert $\mathbf{m}$ to the right of sites where $p_\theta^{\mathrm{ins}}>\tau_{\mathrm{ins}}$; (iii) fill all noisy and masked sites with the substitution head; (iv) re-noise the least confident positions using the evolutionary noise kernel.

## Key Experimental Results

### Main Results
The authors evaluated multiple tasks (abstracted conclusions; detailed figures in the appendix):

| Task | Metric | DPLM-Evo Performance | vs Prev. SOTA |
| :--- | :--- | :--- | :--- |
| ProteinGym Variant Prediction (Single-seq) | Spearman Correlation | **SOTA** | Better than masked-scoring DPLM/ESM |
| Unconditional substitution-only generation | Foldability / Diversity | Comparable or better than DPLM | On par at same dimensions |
| Full edit operations (incl. indel) generation | Variable-length viability | Native support | Impossible for masked diffusion |
| Motif scaffolding (Conditional) | Scaffold Success / Adjustable length | Dynamically adjusts length via ins/del heads | Impossible for fixed-length methods |
| GFP Directed Evolution | Explicit edit trajectory | Improved fluorescence via iterative editing | No trajectory in masked diffusion |

Notably, DPLM-Evo bypasses the standard "mask residue → read logits" scoring loop; it directly inputs the wild-type and evaluates the substitution distribution, a capability unique to substitution-based models.

### Ablation Study

| Configuration | Key Metric Impact | Description |
| :--- | :--- | :--- |
| Full DPLM-Evo | Best ProteinGym Performance | Full model |
| $\mathcal{T}_{\mathrm{sub}}=\mathbf{U}_K$ (Uniform kernel) | Significant drop | Uninformative noise slows learning |
| $\mathcal{T}_{\mathrm{sub}}=\mathbf{M}_{\mathrm{BLOSUM}}$ (Static prior) | Intermediate | Better than uniform, worse than self-conditional |
| Original multinomial indel loss (No BCE) | Mode collapse | Predictions collapsed to deletion; unstable |
| $\omega_{\mathrm{del}}=\omega_{\mathrm{ins}}=0$ (Indels off) | Degenerates to DPLM | No variable-length capability |
| $\rho_{\mathrm{mask}}=1$ (Pure mask) | Degenerates to absorbing diffusion | Classical DPLM/MaskedDiff |

### Key Findings
- **Contextualized Evolutionary Noise > Static BLOSUM > Uniform**: Model-predicted corruption distributions closer to real evolutionary preferences yield higher training efficiency.
- **Binarized Indel Loss is Critical**: Original multinomial forms caused deletion mode collapse; BCE maintains theoretical consistency while stabilizing training.
- **Framework Degeneracy**: Adjusting $\omega_{\mathrm{del}},\omega_{\mathrm{ins}},\rho_{\mathrm{mask}}$ allows strict degradation to masked or uniform diffusion, facilitating warm-starts from existing PLMs.
- **Single-sequence ProteinGym SOTA** suggests that explicit substitution modeling is more natural for variant scoring than mask-and-recover, as it directly reads substitution preferences for every position in the unmasked wild-type.

## Highlights & Insights
- **Decoupled Latent Alignment**: Mapping "variable-length indel evolution" to "token replacement within a fixed-length alignment space" is an elegant mathematical transformation previously seen in CTC or EditFlow, now systematically applied to proteins.
- **Biological Prior as Learnable Noise**: Self-prediction as evolutionary noise is a novel concept—moving from fixed BLOSUM/PAM matrices to a contextualized prior that defines "reasonable transitions" based on the current environment.
- **Breaking the Fixed-length Assumption**: By moving beyond "diffusion = fixed-length generator," DPLM-Evo aligns diffusion models with the reality that protein engineering is fundamentally an editing process (loop remodeling, adjustable scaffolding, directed evolution trajectories).

## Limitations & Future Work
- The $2L$ upsampling imposes a hard limit on net insertion; it cannot handle extreme length expansions like domain duplications.
- Contextualized noise kernels increase training costs and complexity (self-bootstrapping requires careful handling).
- Scalability for very long proteins (>500 residues) regarding diffusion steps and computational cost remains to be fully explored.

## Related Work & Insights
- **vs DPLM/DPLM-2**: DPLM is a subset; DPLM-Evo generalizes the noise kernel to include substitutions and indels.
- **vs ESM-2/3**: DPLM-Evo's substitution distribution provides a more direct and natural scoring mechanism for variants.
- **vs RFdiffusion**: While RFdiffusion focuses on 3D coordinates, DPLM-Evo addresses sequence-level evolutionary trajectories and variable-length editing.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Explicitly integrating variable-length edits into discrete diffusion is a major paradigm shift).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Covers understanding, generation, and optimization, though specific metrics for some tasks are in the appendix).
- **Writing Quality**: ⭐⭐⭐⭐ (Mathematical derivations are clean and self-consistent).
- **Value**: ⭐⭐⭐⭐⭐ (Provides the first diffusion PLM specifically for edited-based protein engineering).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CoSiNE: Conditional Site-Independent Neural Evolution Model for Antibody Sequences](conditionally_site-independent_neural_evolution_of_antibody_sequences.md)
- [\[ICLR 2026\] DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](../../ICLR2026/computational_biology/distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2026\] On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering](on_the_collapse_of_generative_paths_a_criterion_and_correction_for_diffusion_ste.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)

</div>

<!-- RELATED:END -->
