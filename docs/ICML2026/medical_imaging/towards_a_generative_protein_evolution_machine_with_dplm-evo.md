---
title: >-
  [Paper Note] Towards A Generative Protein Evolution Machine with DPLM-Evo
description: >-
  [ICML 2026][Medical Imaging][Protein Language Model] This work proposes DPLM-Evo, extending the discrete diffusion in protein language models from "mask substitution only" to "explicit modeling of substitution + insertio…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Protein Language Model"
  - "Discrete Diffusion"
  - "Evolutionary Modeling"
  - "Variable-Length Generation"
  - "Insertion Deletion"
date: 2026-05-08
content_hash: 874181b5fb64ee19
---

# Towards A Generative Protein Evolution Machine with DPLM-Evo

**Conference**: ICML 2026  
**arXiv**: [2605.00182](https://arxiv.org/abs/2605.00182)  
**Code**: None  
**Area**: Protein Generation / Discrete Diffusion / Biomedicine  
**Keywords**: Protein Language Model, Discrete Diffusion, Evolutionary Modeling, Variable-Length Generation, Insertion Deletion

## TL;DR
This work proposes DPLM-Evo, extending the discrete diffusion in protein language models from "mask substitution only" to "explicit modeling of substitution + insertion + deletion evolutionary edits." By decoupling variable-length observed sequences into an upsampled-length latent alignment space plus a context-aware evolutionary noise kernel, it enables variable-length evolutionary generation and trajectory-based protein post-editing/optimization, achieving SOTA on ProteinGym single-sequence variant effect prediction.

## Background & Motivation

**Background**: Protein language models (PLMs, e.g., ESM, ProGen, DPLM, DPLM-2) learn evolutionary constraints from large-scale sequence databases, with applications in zero-shot variant effect prediction, structure prediction, and sequence generation. Discrete diffusion PLMs (DPLM series) outperform autoregressive PLMs in both representation and generation due to their bidirectional receptive field and long-range dependency modeling.

**Limitations of Prior Work**: Existing DPLMs use an absorbing-state (mask) as the forward noise kernel, reducing generation to "iterative mask recovery." This is biologically unrealistic—protein evolution does not emerge from masks but through cumulative discrete edits (substitution, insertion, deletion), with indels being crucial for loop reshaping, linker length tuning, and motif generation/removal. Mask diffusion lacks native indel actions, and the fixed-length generation framework is cumbersome, making it difficult to express variable-length evolutionary trajectories or perform authentic post-editing on existing proteins.

**Key Challenge**: Standard discrete diffusion is defined on a fixed-dimensional categorical state space, whereas indels necessarily change sequence length—these two mathematical structures are fundamentally incompatible.

**Goal**: To construct a unified discrete diffusion framework where both forward noise and reverse denoising explicitly express substitution/insertion/deletion evolutionary edits, supporting variable-length generation, evolutionary post-editing, and targeted optimization of existing proteins.

**Key Insight**: Drawing from CTC/EditFlow-style latent alignment—decouple the variable-length observed sequence space $\mathcal{X}$ into an upsampled-length ($2L$) latent alignment space $\mathcal{Z}$, where insertion of gap symbol $\phi$ transforms the variable-length problem into a fixed-length one; the diffusion process is defined on $\mathcal{Z}$, while the neural network only sees the collapsed sequence on $\mathcal{X}$.

**Core Idea**: In the latent alignment space, use a unified transition matrix $\mathbf{Q}_{\mathrm{noise}}$ to encode three types of $\mathcal{A}\leftrightarrow\phi$ transitions (substitution/insertion/deletion), supplemented by a "context-aware evolutionary noise kernel"—replace substitution noise with the model's own predicted conditional distribution, making the corruption process evolutionarily plausible; during decoding, use three independent heads for substitution/deletion/insertion, sequentially executing delete→insert→substitute→renoise to achieve variable-length denoising.

## Method

### Overall Architecture
Two spaces: observation space $\mathcal{X}=\mathcal{V}^L$ ($\mathcal{V}=\mathcal{A}\cup\{\mathbf{m}\}$ includes mask) and latent alignment space $\mathcal{Z}=(\mathcal{V}\cup\{\phi\})^{2L}$ ($\phi$ is gap); the collapse function $\Gamma^{-1}(\mathbf{z})$ removes all $\phi$ from $\mathbf{z}$ to recover $\mathbf{x}$, and the set $\Gamma(\mathbf{x})$ contains all valid alignments of $\mathbf{x}$. Forward diffusion $q_t(\mathbf{z}_t|\mathbf{z}_0)=\bar\alpha_t\delta_{\mathbf{z}_0}+(1-\bar\alpha_t)\pi(\mathbf{z}_0)$ operates in the latent space; the network $f_\theta$ acts on $\mathbf{x}_t=\Gamma^{-1}(\mathbf{z}_t)$, with three heads simultaneously predicting the substitution distribution, deletion probability, and right-side insertion probability for each token. The ELBO form: $\log p_\theta(\mathbf{x}_0)\geq\mathbb{E}_{\mathbf{z}_0\in\Gamma(\mathbf{x}_0)}[\mathbb{E}_{q_t}[\log p_\theta(\mathbf{z}_0|\mathbf{z}_t)]]$.

### Key Designs

1. **Latent Alignment Space Decouples Variable-Length Indels**:

    - **Function**: Models variable-length indel evolution within a fixed-dimensional computational framework, while allowing the network to process only compact observed sequences.
    - **Mechanism**: Upsample the sequence by 2x and insert gap placeholders $\phi$ (e.g., $[A,B,C]\mapsto[A,\phi,\phi,B,\phi,C]$); the forward diffusion process transitions on $\mathbf{z}$, and reverse denoising projects back to observation space via $\Gamma^{-1}$. The transition matrix $\mathbf{Q}_{\mathrm{noise}}$ is controlled by three hyperparameters $(\omega_{\mathrm{del}},\omega_{\mathrm{ins}},\rho_{\mathrm{mask}})$: $\mathcal{A}$ states are replaced with another amino acid with probability $1-\omega_{\mathrm{del}}$ (or with mask at ratio $\rho_{\mathrm{mask}}$), or become $\phi$ (deletion) with probability $\omega_{\mathrm{del}}$; $\phi$ states become amino acids (insertion) with probability $\omega_{\mathrm{ins}}$.
    - **Design Motivation**: Defining a Markov process directly on variable-length sequences is extremely complex (each step must jointly sample length and content); latent alignment encodes indels as simple token replacements, leveraging the mature fixed-length diffusion toolkit; 2x upsampling ensures net insertion does not exceed original length $L$, covering typical protein engineering scenarios (loop/linker length adjustment). This trick also allows DPLM-Evo to initialize from existing masked DPLM weights, effectively extending capability with minimal changes.

2. **Context-Aware Evolutionary Noise Kernel**:

    - **Function**: Ensures forward noise reflects true evolutionary preferences (contextually "reasonable" substitutions) rather than uniform random.
    - **Mechanism**: The substitution matrix $\mathcal{T}_{\mathrm{sub}}$ has three options: (i) uniform $\mathbf{U}_K=\tfrac{1}{K}\mathbf{1}\mathbf{1}^\top$; (ii) static biological prior $\mathbf{M}_{\mathrm{BLOSUM}}$; (iii) context-aware $\mathcal{T}_{\mathrm{sub}}^{(j)}=\mathbb{E}_{q'_t(\mathbf{z}'_t|\mathbf{z}_0)}[p_\theta(\cdot|\mathbf{z}'^{\setminus j}_t,\mathbf{m})]$—force target position $j$ to mask and let the model predict it in a partial-masked context. The warmup phase uses simple mask noise for training, then switches to self-prediction noise; when $t=1$ (full noise), it degenerates to $p_\theta(\cdot|\mathbf{m}^L)$, providing a learnable prior reflecting natural amino acid statistics.
    - **Design Motivation**: Uniform noise trains diffusion models to treat biologically rare substitutions (e.g., "Lys → Trp") as equally important as common ones ("Lys → Arg"), wasting capacity; context-aware noise exposes the model to corruptions closer to real evolutionary mutations (conservative substitutions, context preferences), improving training efficiency and encouraging explicit modeling of evolutionary and homology dependencies.

3. **Three-Head Decoupling + Binary Classification Indel Training**:

    - **Function**: Ensures substitution/deletion/insertion tasks do not interfere and addresses mode collapse due to class imbalance in indels.
    - **Mechanism**: Use an Index Mapping Function $\mathcal{I}:\{1,\dots,L_t\}\to\{1,\dots,N\}$ to map observed tokens back to latent sequence positions; define three mutually exclusive losses based on $(\mathbf{z}_t,\mathbf{z}_0)$ token class combinations—$\mathcal{L}_{\mathrm{sub}}^{(k)}$ applies only when both are amino acids and differ (standard CE); indel losses are binary classification: $\mathcal{L}_{\mathrm{del}}^{(k)}=\mathrm{BCE}(\mathbb{I}_{\mathbf{z}_0^{(\mathcal{I}(k))}=\phi},p_\theta^{\mathrm{del}})$, $\mathcal{L}_{\mathrm{ins}}^{(k)}=\mathrm{BCE}(\mathbb{I}_{v_{\mathrm{next}}^{(k)}\neq\emptyset},p_\theta^{\mathrm{ins}})$; total loss $\mathcal{L}_t=\mathbb{E}[\sum_k\lambda_{t-1}(\gamma_{\mathrm{sub}}\mathcal{L}_{\mathrm{sub}}+\gamma_{\mathrm{del}}\mathcal{L}_{\mathrm{del}}+\gamma_{\mathrm{ins}}\mathcal{L}_{\mathrm{ins}})]$.
    - **Design Motivation**: The original multinomial indel loss (treating $\phi$ as a token in the extended vocabulary) led to deletion mode collapse (model predicts all deletions) and unstable insertion training, as substitutions are much more frequent than indels in biological sequences; switching to binary classification decouples "whether to delete/insert" from "what to insert/replace," containing class imbalance within BCE and stabilizing training.

### Loss & Training

- $\mathcal{L}_t=\mathbb{E}_{\mathbf{x}_0,\mathbf{z}_0,\mathbf{z}_t}[\sum_k\lambda_{t-1}(\gamma_{\mathrm{sub}}\mathcal{L}_{\mathrm{sub}}^{(k)}+\gamma_{\mathrm{del}}\mathcal{L}_{\mathrm{del}}^{(k)}+\gamma_{\mathrm{ins}}\mathcal{L}_{\mathrm{ins}}^{(k)})]$, with three $\gamma$'s tuning preferences for different evolutionary operations.
- Initialize from pretrained DPLM → warmup phase with mask noise → switch to context-aware noise kernel for continued training.
- Sampling: maintain a noisy index set $\mathcal{N}_t$, and at each step: (i) delete positions with $p_\theta^{\mathrm{del}}>\tau_{\mathrm{del}}$; (ii) insert $\mathbf{m}$ to the right of positions with $p_\theta^{\mathrm{ins}}>\tau_{\mathrm{ins}}$; (iii) fill all noisy and mask positions using the substitution head; (iv) re-noise the least confident positions using the evolutionary noise kernel.

## Key Experimental Results

### Main Results
The authors evaluated multiple tasks (main conclusions in the abstract; detailed numbers in the appendix):

| Task | Metric | DPLM-Evo Performance | vs Prev. SOTA |
|------|--------|---------------------|---------------|
| ProteinGym Variant Effect Prediction (Single Sequence) | Spearman Correlation | **SOTA** | Outperforms masked-scoring DPLM/ESM |
| Unconditional Substitution-Only Generation | Foldability / Diversity | Comparable or better than DPLM | On par |
| Full Edit Operations (with indel) Generation | Variable-Length Feasibility | Natively supported | Mask diffusion cannot achieve |
| Motif Scaffolding (Conditional Generation) | Scaffold Success Rate / Adjustable Length | Dynamically adjusts scaffold length via ins/del head | Fixed-length methods cannot do this |
| GFP Directed Evolution Optimization | Explicit Edit Trajectory | Iterative substitution+indel improves fluorescence | Mask diffusion lacks trajectory |

DPLM-Evo no longer follows the conventional "mask residues to predict → read logits" variant scoring process, but directly inputs the wild-type and evaluates the substitution distribution, a unique capability of substitution-based models.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full DPLM-Evo (Contextual Kernel + Three-Head BCE) | Best ProteinGym Performance | full model |
| $\mathcal{T}_{\mathrm{sub}}=\mathbf{U}_K$ (Uniform Kernel) | Significant Drop | Uninformative noise slows learning |
| $\mathcal{T}_{\mathrm{sub}}=\mathbf{M}_{\mathrm{BLOSUM}}$ (Static Prior) | Intermediate | Better than uniform, not as good as self-conditional |
| Original Multinomial Indel Loss (No BCE) | Mode Collapse | All deletions, training diverges |
| $\omega_{\mathrm{del}}=\omega_{\mathrm{ins}}=0$ (No indel) | Degenerates to DPLM | No variable-length capability |
| $\rho_{\mathrm{mask}}=1$ (Pure mask) | Degenerates to absorbing diffusion | Classic DPLM/MaskedDiff |
| $\rho_{\mathrm{mask}}=0$ (Pure uniform) | Degenerates to uniform diffusion | Austin et al. 2021 |

### Key Findings
- **Context-aware evolutionary noise kernel > static BLOSUM > uniform**: The model's self-predicted corruption distribution better matches true evolutionary preferences; preliminary experiments confirm (iii) is optimal and thus set as default.
- **Binary indel loss is crucial for training stability**: The original multinomial form led to deletion mode collapse and unstable insertion; BCE form maintains theoretical consistency and stabilizes training.
- **Unified framework is fully degradable**: By tuning $\omega_{\mathrm{del}},\omega_{\mathrm{ins}},\rho_{\mathrm{mask}}$, the model can strictly degenerate to masked diffusion, uniform diffusion, or any mixture, facilitating hot-start from existing masked PLMs.
- **Single-sequence ProteinGym SOTA** demonstrates that explicit modeling of substitution (rather than mask-and-recover) is more natural for variant scoring—directly reading the substitution distribution at each position of the unmasked wild-type, saving the mask loop and better matching the task definition of "evaluating substitution preference at a site."
- **2L upsampling design** means net insertion cannot exceed the original length, which suffices for engineering scenarios (loop adjustment of 5–30 residues), but is not suitable for extreme length expansion (domain duplication).

## Highlights & Insights
- **Decoupled design of latent alignment space**: Mapping "variable-length indel evolution" to "token replacement in fixed-length alignment space" is an elegant mathematical transformation. This "upsampling + gap + collapse" pattern appears in CTC, EditFlow, DreamOn, etc., but DPLM-Evo is the first to systematically apply it to protein diffusion.
- **Biological prior as learnable noise kernel**: Using "self-prediction as evolutionary noise" is novel—no longer relying on hand-crafted BLOSUM/PAM matrices, but letting the model define "what transformations are reasonable in the current context," upgrading evolutionary preference from fixed prior to learnable & contextual prior.
- **Unified + degradable framework**: A single set of transition matrix hyperparameters covers all existing discrete diffusion variants (masked / uniform / mixed), providing a unified perspective and practical convenience for initializing from any checkpoint. This "general → specialized" design philosophy is worth emulating.
- **Variable-length generation unlocks three major protein engineering scenarios** (loop remodeling, motif scaffold length tuning, directed evolution trajectory)—breaking the long-standing assumption that "diffusion models = fixed-length generators," and aligning diffusion models with the essence of "protein engineering as editing."

## Limitations & Future Work
- 2L upsampling strictly limits net insertion to $L$; scenarios requiring domain duplication (e.g., tandem repeats, dimerization) are not supported. Dynamic upsampling ratios or multi-stage cascaded generation could be considered.
- Context-aware noise kernel requires model-generated noise after warmup, increasing training cost and instability (self-bootstrapping often needs tricks like stop-gradient, target network); the paper's "warmup→switch" strategy lacks ablation on warmup duration.
- The experiments lack specific quantitative results visible in this note (cache truncated at ProteinGym introduction), only qualitative conclusion "single-sequence SOTA"; concrete differences with SOTA models like ESM-2, SaProt, and key metrics such as TM-score/pLDDT for generated structures need to be supplemented from the appendix.
- Motif scaffolding introduces an "additional structure encoder" for motif coordinates, but the reliability of dynamic scaffold length via ins/del head vs fixed prediction is not fully compared; baseline comparisons for scaffold diversity and design feasibility are lacking.
- Diffusion steps and computational cost for long proteins (>500 residues) are not discussed; industrial applicability requires further validation.

## Related Work & Insights
- **vs DPLM/DPLM-2 (masked diffusion series)**: DPLM can only do mask-predict, fixed-length; DPLM-Evo generalizes the noise kernel from mask to mask+sub+ins+del, making it a strict superset in expressive power.
- **vs ESM-2/ESM-3 (autoencoding PLM)**: ESM requires "mask residues → read logits" for variant scoring; DPLM-Evo natively scores with substitution distributions, better matching task semantics and achieving SOTA in ProteinGym single-sequence setting.
- **vs EditFlow / DreamOn (variable-length text diffusion)**: Also uses latent alignment + gap symbol for variable-length generation, but in text; DPLM-Evo adapts this mechanism to proteins and adds biological prior noise kernel for greater specificity.
- **vs ProGen/RFdiffusion (autoregressive PLM / structure diffusion)**: Autoregressive models generate only left-to-right, not suited for post-editing; structure diffusion focuses on 3D coordinates, not evolutionary trajectories; DPLM-Evo fills the gap for "sequence + variable-length + edit-based generation."
- **vs Discrete diffusion family (Austin et al. 2021, SEDD, etc.)**: DPLM-Evo's unified transition matrix subsumes this family and highlights indel as a long-neglected dimension, offering methodological insights for discrete diffusion research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Explicitly integrating variable-length evolutionary edits into discrete diffusion is a true paradigm extension; the context-aware noise kernel is also a novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers understanding (ProteinGym), unconditional generation, conditional generation, and directed optimization, but specific numbers require checking the appendix and quantitative comparisons need improvement.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations (ELBO, $\mathbf{Q}_{\mathrm{noise}}$, ELBO decomposition, degradation relations with existing methods) are clean and self-consistent; some notation density requires diffusion modeling background.
- Value: ⭐⭐⭐⭐⭐ Provides the protein engineering community with the first diffusion PLM supporting edit-based generation, variable-length, and evolutionary priors, with direct potential for directed evolution, scaffold engineering, and loop optimization.

## Related Papers

- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/medical_imaging/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[ICLR 2026\] DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](../../ICLR2026/medical_imaging/distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)
- [\[ICML 2026\] Protein Circuit Tracing via Cross-layer Transcoders](protein_circuit_tracing_via_cross-layer_transcoders.md)

</div>

<!-- RELATED:END -->
