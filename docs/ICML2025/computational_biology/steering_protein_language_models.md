---
title: >-
  [Paper Note] Steering Protein Language Models
description: >-
  [ICML 2025][Computational Biology][Protein Language Models] This work migrates the Activation Steering technique from the LLM domain to protein language models (PLMs) for the first time. By editing internal model activations during inference, the proposed method guides protein sequence generation and optimization toward target properties (e.g., thermostability, solubility) completely training-free. Additionally, it introduces an Activation Steering-based Protein Optimization…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Protein Language Models"
  - "Activation Steering"
  - "Protein Optimization"
  - "Mutation Site Identification"
  - "Training-Free Control"
date: 2026-05-08
content_hash: 97953441807ad3c6
---

# Steering Protein Language Models

**Conference**: ICML 2025  
**arXiv**: [2509.07983](https://arxiv.org/abs/2509.07983)  
**Code**: None (Tencent AI Lab)  
**Area**: Protein Engineering / Computational Biology  
**Keywords**: Protein Language Models, Activation Steering, Protein Optimization, Mutation Site Identification, Training-Free Control

## TL;DR

This work migrates the Activation Steering technique from the LLM domain to protein language models (PLMs) for the first time. By editing internal model activations during inference, the proposed method guides protein sequence generation and optimization toward target properties (e.g., thermostability, solubility) completely training-free. Additionally, it introduces an Activation Steering-based Protein Optimization (ASPO) algorithm for mutation site identification using steering vector dissimilarity, significantly outperforming traditional methods in lysozyme and GFP optimization tasks.

## Background & Motivation

**Background**: Pretrained on billions of evolutionary sequences, protein language models (PLMs) have emerged as core tools for protein design. PLMs are categorized into autoencoding (AE) models like ESM2/ESM3 and autoregressive (AR) models like ProLLaMA. While they excel in mutation effect prediction and structure inference, precisely controlling output sequences to express specific functional properties remains challenging.

**Limitations of Prior Work**: Existing methods for controlling PLM outputs suffer from significant drawbacks: (1) fine-tuning requires massive annotated data and computational resources, running the risk of catastrophic forgetting of pretrained knowledge; (2) prompt-based control using keyword tags lacks flexibility and is constrained by the tags used during pretraining; (3) search- or sampling-based methods (e.g., AdaLead, PEX) are highly inefficient, requiring screening over a large number of generated sequences and relying heavily on fitness predictors.

**Key Challenge**: Intrinsic knowledge of protein properties (such as thermostability and solubility) is already encoded within PLMs, yet this knowledge does not always manifest in the output. How can this implicit knowledge be precisely "released" without modifying model weights?

**Goal**: To explore an inference-time intervention method that controls PLMs to generate protein sequences with target properties without requiring training or weight updates, and to extend this approach to mutation site identification and directed mutation in protein optimization scenarios.

**Key Insight**: Drawing inspiration from the Activation Steering techniques successfully applied in LLMs, this work adapts them to PLMs. Since protein sequences reflect biophysical functions rather than linguistic semantics and their activation space is shaped by evolutionary and structural constraints, specialized handling is required.

**Core Idea**: Calculate the mean difference in internal PLM representations between proteins with and without the target property to serve as the steering vector. Adding this vector to activations during inference steers the generation direction. Additionally, utilize the cosine dissimilarity between the steering vector and token representations to identify potential mutation sites.

## Method

### Overall Architecture

The method comprises two levels: (1) Activation Steering for protein generation control—adding a steering vector to the activations of each PLM layer to bias the generation direction, applicable to both AR-PLMs (autoregressive generation) and AE-PLMs (masked position prediction); (2) ASPO for protein optimization—identifying mutation-dependent positions via the steering vector, and then guiding mutation predictions through activation steering across multi-round iterations to progressively enhance target properties.

### Key Designs

1. **Steering Vector Calculation and Activation Editing**
    - **Function**: Calculate vectors representing the direction of the target property and inject them into the activations of each model layer during inference.
    - **Mechanism**: Collect a positive set $\mathcal{P}$ and a negative set $\mathcal{N}$ for the target property (100 sequences each). For each layer $l$, calculate $\mathbf{v}_l = \frac{1}{|\mathcal{P}|}\sum_{x_p \in \mathcal{P}} \mathbf{h}_l^{avg}(x_p) - \frac{1}{|\mathcal{N}|}\sum_{x_n \in \mathcal{N}} \mathbf{h}_l^{avg}(x_n)$ (using average token representations for AE-PLM and the last token for AR-PLM). During inference, perform $\tilde{\mathbf{h}}_l = \mathbf{h}_l + \alpha \mathbf{v}_l$ at each layer, and rescale the result to the original norm.
    - **Design Motivation**: The mean difference extracts the linear direction of the property within the representation space, while rescaling prevents activation magnitude shifts from disrupting internal model dynamics.

2. **Mutation Site Identification Based on Steering Vector Dissimilarity**
    - **Function**: Automatically identify amino acid sites in the protein sequence that are least correlated with the target property as candidates for mutation.
    - **Mechanism**: Calculate the cosine similarity between each token representation and the steering vector as $s^k = \frac{\mathbf{v}_l^\top \mathbf{h}_l^k}{||\mathbf{v}_l|| \cdot ||\mathbf{h}_l^k||}$, selecting the $T$ sites with the lowest scores. The optimal layer for score calculation is selected by evaluating discriminant ability across layers using linear classifiers trained on positive/negative sets.
    - **Design Motivation**: Low similarity suggests that the current amino acid at the position contradicts the target property direction, making it the most suitable candidate for modification. This approach is more targeted than random selection and eliminates the need for an external fitness predictor.

3. **Multi-Round Iterative Optimization via ASPO**
    - **Function**: Gradually optimize the protein toward the target property through a multi-round mask-then-predict pipeline.
    - **Mechanism**: Each round executes: calculate dissimilarity scores of all tokens $\rightarrow$ select the $T$ lowest-scoring sites $\rightarrow$ mask these sites $\rightarrow$ use activation steering to guide the PLM in predicting new amino acids. Repeat for $R$ rounds ($R=8, T=4$ for thermostability experiments; $R=4, T=2$ for solubility/GFP experiments).
    - **Design Motivation**: Iterative optimization avoids structural collapse caused by changing too many sites at once; recalculating scores each round ensures adaptive updates of mutation choices.

### Loss & Training

This method does not involve model training or weight updates. All operations occur during inference. Fine-tuning baselines: AE-PLM fine-tunes the final layer; AR-PLM uses LoRA (rank=4, alpha=16) across all layers. ASPO baseline methods (AdaLead, PEX, GWG) require training a fitness predictor.

## Key Experimental Results

### Protein Generation Experiments (Lysozyme, 1000 Sequences)

| Base Model | Method | Thermostability↑ | Diversity↑ | Novelty↑ | Solubility↑ |
|---------|------|----------|--------|--------|--------|
| ProLLaMA | Original | 56.18 (8.05) | 0.931 | 0.767 | 0.230 |
| ProLLaMA | Fine-tuning | 57.24 (8.64) | 0.958 | 0.798 | 0.241 |
| ProLLaMA | **Act. Steering** | **67.68 (12.86)** | 0.927 | **0.807** | **0.276** |
| ESM2 | Original | 56.48 (12.04) | 0.954 | 0.591 | 0.289 |
| ESM2 | Fine-tuning | 63.56 (14.87) | 0.953 | 0.585 | 0.356 |
| ESM2 | **Act. Steering** | **82.20 (12.92)** | **0.971** | **0.739** | **0.494** |
| ESM3 | Original | 55.20 (11.14) | 0.952 | 0.573 | 0.257 |
| ESM3 | Fine-tuning | 62.82 (14.72) | 0.949 | 0.568 | 0.318 |
| ESM3 | **Act. Steering** | **82.06 (12.06)** | 0.954 | **0.614** | **0.582** |

### Protein Optimization Experiments (Thermostability + GFP Fluorescence)

| Method | Thermostability-Medium↑ | Thermostability-Hard↑ | GFP-Medium↑ | GFP-Hard↑ |
|------|----------------|--------------|-------------|-----------|
| AdaLead | 63.56 | 55.16 | 1.179 | 1.255 |
| PEX | 66.80 | 48.95 | 1.426 | 1.320 |
| GWG | 68.25 | 47.73 | 1.683 | 1.510 |
| **ESM2+ASPO** | **84.34** | **74.69** | **3.862** | **3.907** |
| **ESM3+ASPO** | **88.42** | **86.43** | 3.739 | 3.687 |

### Key Findings

- Activation Steering on ESM2 improves thermostability from 56.48 to 82.20 (+46%), far exceeding the 63.56 achieved via fine-tuning.
- ESM3+ASPO reaches 86.43 in hard thermostability optimization, which is 1.8 times better than GWG (47.73).
- In GFP fluorescence optimization, ESM2+ASPO achieves 3.862, more than doubling the performance of the best baseline GWG (1.683).
- With positive/negative set sizes of 100, AE-PLMs achieve $\ge$ 95% of peak performance, while AR-PLMs reach optimality with as few as 10 samples.
- A default setting of $\alpha=1.0$ is recommended, covering 90-98% of peak performance, whereas performance on the solubility task collapses for $\alpha > 5$.
- The generated sequences maintain or even increase diversity and novelty, indicating that the method is not merely memorizing the positive set.

## Highlights & Insights

- **Elegant Success of Cross-Domain Transfer**: The migration from NLP to protein engineering is highly natural. t-SNE and linear probes confirm that PLM activation spaces already encode property information, and activation steering merely unlocks this knowledge.
- **Enormous Practical Value of Zero Training Cost**: Requiring only 100 positive/negative sample pairs and a single hyperparameter $\alpha$, attribute-guided generation can be achieved on any PLM training-free.
- **Ingenuity of the Mutation Site Identification Algorithm**: Utilizing the steering vector itself to locate mutation sites removes the dependency on a fitness predictor, eliminating an entire pipeline component.
- **Stunning Performance of ASPO on GFP**: Protein fluorescence is optimized from 1.3-1.5 to 3.7-3.9 (more than doubled), demonstrating substantial potential in real-world protein engineering tasks.

## Limitations & Future Work

- The steering vector is computed as a mean difference, which is a linear direction and may fail to capture complex non-linear property-representation relationships.
- Attribute evaluation relies on computational predictors rather than wet-lab experiments; predictor inaccuracies could affect the reliability of the conclusions.
- Steering multiple attributes simultaneously may trigger vector conflicts, which is only briefly discussed in the appendix.
- The method is validated on only two protein families (lysozyme and GFP); performance across broader protein spaces remains to be tested.
- AR-PLM (ProLLaMA) benefits less from larger contrastive sets, implying limitations of using the last-token representation.

## Related Work & Insights

- **Activation Addition (Turner et al., 2023)**: The direct inspiration for this work, which uses contrastive prompts to compute steering vectors for controlling LLMs.
- **CAA (Panickssery et al., 2023)**: Aggregates steering vectors from hundreds of contrastive pairs to reduce noise.
- **GGS (Kirjner et al., 2023)**: Employs an energy-based model to smooth the fitness landscape combined with Gibbs sampling for protein optimization; a primary comparison baseline for ASPO.
- **ESM2/ESM3/ProLLaMA**: Three baselines of PLMs undergoing steering, representing AE and AR architectures, respectively.
- **Insights**: The linear structure of PLM activation spaces is more orderly than expected; simple linear operations are sufficient to achieve precise control.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First cross-domain transfer; the mutation site identification algorithm is a protein-specific innovation)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (3 architectures $\times$ 3 properties $\times$ 2 tasks $\times$ 2 difficulties, with exhaustive hyperparameter sensitivity analysis)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear methodology, intuitive diagrams, and complete pseudocode/workflow algorithms)
- **Value**: ⭐⭐⭐⭐ (Provides a plug-and-play, zero-cost control scheme for protein design, offering high clinical/industrial utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/computational_biology/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)

</div>

<!-- RELATED:END -->
