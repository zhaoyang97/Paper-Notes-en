---
title: >-
  [Paper Note] ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders
description: >-
  [AAAI 2026][Medical Imaging][Protein language models] This paper proposes ProtSAE, which incorporates semantic annotations and domain ontology knowledge as guidance signals during sparse autoencoder training to address t…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Protein language models"
  - "sparse autoencoders"
  - "semantic guidance"
  - "interpretability"
  - "feature disentanglement"
  - "ontology embeddings"
date: 2026-05-08
content_hash: fddde8573bf652e5
---

# ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders

**Conference**: AAAI 2026  
**arXiv**: [2509.05309](https://arxiv.org/abs/2509.05309)  
**Code**: [nju-websoft/ProtSAE](https://github.com/nju-websoft/ProtSAE)  
**Area**: Bioinformatics  
**Keywords**: Protein language models, sparse autoencoders, semantic guidance, interpretability, feature disentanglement, ontology embeddings

## TL;DR

This paper proposes ProtSAE, which incorporates semantic annotations and domain ontology knowledge as guidance signals during sparse autoencoder training to address the semantic entanglement problem of conventional SAEs. The method aligns latent features of protein language models with biological concepts (molecular function, biological process, ion binding sites, etc.) with high precision, while maintaining high reconstruction fidelity and supporting concept-level generation steering.

## Background & Motivation

**Rapid development of protein language models with opaque internal mechanisms**: PLMs such as ESM2 are widely used for function prediction, structure modeling, and protein design, yet how their latent features map to biological concepts such as binding pockets, post-translational modifications, and fold families remains largely unknown.

**Sparse autoencoders as a powerful tool for understanding LLM internal representations**: SAEs decompose high-dimensional latent representations into sparse features based on the linear feature superposition hypothesis, and subsequently interpret each feature via correlation analysis with annotated data. This approach has been successfully applied to interpretability studies of Claude 3 Sonnet and GPT-4.

**Severe semantic entanglement in conventional SAEs**: Individual neurons tend to conflate multiple semantically unrelated biological concepts simultaneously (e.g., a single activation associated with both iron ion binding and sodium transport), resulting in ambiguous feature interpretations and unreliable generation control.

**Existing methods perform annotation-based interpretation only post-training, failing to resolve entanglement at the source**: The standard pipeline trains SAEs in an unsupervised manner and then analyzes feature–concept correlations post hoc. This posterior approach cannot constrain the training process, leaving the entanglement problem unaddressed.

**The protein domain provides rich structured prior knowledge**: Ontologies such as Gene Ontology define logical relations among concepts, including is-a, part-of, and regulates. This expert knowledge can provide additional semantic constraints for feature learning, yet has not been exploited in prior work.

**A semantically-guided SAE approach that incorporates guidance during training is therefore needed**: This paper proposes ProtSAE, which integrates annotated data and ontology knowledge into SAE training simultaneously, constraining feature–concept correspondence from the outset to achieve genuine semantic disentanglement.

## Method

### Overall Architecture

ProtSAE is built on the TopK-SAE architecture and partitions latent activations into two groups: $m$ *defined activations* $\mathbf{z}_{\text{def}}$ that are bound one-to-one to predefined biological concepts, and the remaining $n-m$ *unknown activations* $\mathbf{z}_{\text{unk}}$ that freely capture unknown semantics. The training objective consists of three components: reconstruction loss $\mathcal{L}_{\text{rec}}$, semantic annotation loss $\mathcal{L}_{\text{annot}}$, and ontology axiom loss $\mathcal{L}_{\text{axiom}}$. Forced activation and feature rescaling mechanisms ensure that defined activations effectively participate in reconstruction.

### Key Design 1: Annotation-Based Semantic Disentanglement

- **Function**: A concept predictor $\pi_{\text{pred}} = \sigma(\mathbf{W}_{\text{pred}}(\mathbf{x} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{pred}})$ is introduced and trained on annotated data via binary cross-entropy loss $\mathcal{L}_{\text{annot}}$, enabling each defined neuron to learn to detect the presence of a specific biological concept.
- **Mechanism**: Through weight binding $\mathbf{W}_{\text{def}} = \mathbf{W}_{\text{pred}}^{\text{detach}} \cdot \exp(\mathbf{r}_{\text{pred}})$, the defined weights of the encoder share the semantic direction of the predictor while independently adjusting magnitude via a learnable scaling factor $\mathbf{r}_{\text{pred}}$. The detach operation prevents reconstruction gradients from corrupting the semantic direction of the predictor.
- **Design Motivation**: Allowing reconstruction loss to directly update $\mathbf{W}_{\text{pred}}$ would distort feature directions toward the reconstruction objective, causing semantic drift. The detach + exponential scaling design preserves semantic purity while permitting magnitude adaptation to reconstruction demands.

### Key Design 2: Forced Activation

- **Function**: When the concept predictor determines that a concept is present ($\pi_{\text{pred}} > 0.5$) but the corresponding defined activation falls below the mean of $\mathbf{z}_{\text{unk}}$, a semantic bias $\mathbf{z}_{\text{bias}}$ is applied to raise the activation above the mean.
- **Mechanism**: $\mathbf{z}_{\text{bias}} = \mathbb{1}_{\pi_{\text{pred}}>0.5} \cdot \text{ReLU}(\text{mean}(\mathbf{z}_{\text{unk}}) - \hat{\mathbf{z}}_{\text{def}})$, ensuring that semantically predicted active features are not neglected during decoding.
- **Design Motivation**: Empirical observations show that the reconstruction process relies more heavily on the entangled unsupervised activations $\mathbf{z}_{\text{unk}}$, causing defined activations to be marginalized. Forced activation ensures that semantic features actively participate in reconstruction, providing a foundation for downstream generation steering.

### Key Design 3: Ontology Constraints from Domain Knowledge

- **Function**: Using the ELEmbeddings method, the four classes of normalized axioms in Gene Ontology (subclass NF1, conjunctive subclass NF2, existential inclusion NF3, existential restriction NF4) are encoded as geometric constraints in the predictor weight space, yielding total loss $\mathcal{L}_{\text{axiom}} = \mathcal{L}_{\text{NF1}} + \mathcal{L}_{\text{NF2}} + \mathcal{L}_{\text{NF3}} + \mathcal{L}_{\text{NF4}}$.
- **Mechanism**: The paper demonstrates that the predictor weights $\mathbf{W}_{\text{pred}}$ are structurally equivalent to ontology embeddings in ELEmbeddings, allowing ontology axiom constraints to be applied directly in this space without requiring a separate embedding space.
- **Design Motivation**: Biological concepts are not mutually independent (e.g., "lytic vacuole" is-a "vacuole"), and annotation data alone cannot fully capture these hierarchical relations. Ontology constraints ensure that the geometric structure of the feature space faithfully reflects the logical relations among concepts, improving semantic consistency.

### Key Design 4: Training Strategy

- **Function**: The total loss is $\mathcal{L} = \|\hat{\mathbf{x}} - \mathbf{x}\|_2^2 + \lambda_{\text{annot}} \mathcal{L}_{\text{annot}} + \lambda_{\text{axiom}} \mathcal{L}_{\text{axiom}}$, where $\lambda_{\text{annot}} = \lambda_{\text{axiom}} = 1$.
- **Mechanism**: The three loss terms serve distinct roles — the reconstruction loss maintains fidelity, the annotation loss guides semantic alignment, and the axiom loss models inter-concept relations. Sparsity is controlled by $K \in \{50, 100, 500, 1000\}$.
- **Design Motivation**: Equal-weight combination simplifies hyperparameter tuning, and experiments show that this setting achieves a favorable balance between fidelity and interpretability.

## Key Experimental Results

### Table 1: Probe-Based Protein Function Prediction (Average over Three Ontologies)

| Method | $F_{\max}\uparrow$ | $S_{\min}\downarrow$ | AUPR $\uparrow$ | AUC $\uparrow$ |
|------|------|------|------|------|
| SpLiCE | .417 | 23.4 | .360 | .329 |
| Naive SAE | .421 | 23.3 | .340 | .511 |
| Gated SAE | .441 | 22.7 | .368 | .533 |
| TopK SAE | .444 | 22.7 | .379 | .565 |
| Linear Probe (PLM) | .537 | 20.9 | .522 | .751 |
| **ProtSAE** | **.579** | **20.9** | **.487** | **.797** |

ProtSAE substantially outperforms all SAE baselines across all metrics. AUC improves from 0.565 (TopK SAE) to 0.797 (+41%), surpassing even a linear probe applied directly to PLM hidden states (AUC 0.751).

### Table 2: Key Ablation Comparisons (BPO Dataset, Trend at K=100)

| Variant | AUC Change | Reconstruction Fidelity Change | Defined Activation Participation |
|------|---------|-------------|-------------|
| Full ProtSAE | Baseline | Baseline | ~100% |
| w/o detach | Sharp AUC drop | Slight improvement | — |
| w/o $\mathcal{L}_{\text{axiom}}$ | Notable AUC drop | Notable drop | — |
| w/o $\mathbf{z}_{\text{bias}}$ | Slight AUC drop | Slight drop | Notable drop |
| w/o $\mathbf{r}_{\text{pred}}$ | AUC drop | Drop | — |

Ablation experiments confirm that each component is indispensable: detach is the key to semantic purity, ontology constraints are critical for modeling complex conceptual relations, and forced activation ensures that defined features participate in reconstruction.

## Key Findings

1. **Semantically-guided training significantly improves feature–concept alignment quality**: In relevance-based F1 evaluation, ProtSAE achieves substantially higher mean and maximum scores among top-10 features compared to all SAE baselines, demonstrating that incorporating annotation constraints during training is more effective than post-hoc annotation.

2. **ProtSAE maintains its advantage across varying sparsity levels**: As $K$ varies from 50 to 1000, ProtSAE consistently leads all other methods in AUC, while its Loss Recovered remains comparable to TopK SAE, indicating that semantic guidance does not sacrifice reconstruction quality.

3. **Learned features can be visualized as mapping to functional regions in protein structure**: The iron ion binding feature precisely activates at TonB-dependent receptor regions, the sodium transport feature localizes to transmembrane α-helical segments, and the metal ion binding feature accurately marks specific binding sites.

4. **Concept-level generation steering experiments demonstrate the causal significance of features**: Intervening on specific concept activations yields generated proteins with significantly improved TM-score, reduced RMSD, and higher pLDDT scores, confirming that ProtSAE's defined features can effectively guide PLMs toward generating proteins with target functions.

5. **Generated proteins are structurally similar to natural proteins yet sequence-novel**: For example, a protein generated via steering with a DNA transcription repressor concept achieves a TM-score of 0.829 with A0A346G484 while exhibiting only 30% sequence identity, demonstrating genuine function-oriented design capability.

## Highlights & Insights

- **Upgrading interpretability tools from "post-hoc explanation" to "training guidance"**: This represents the most fundamental conceptual shift in the paper — rather than passively analyzing what an SAE has learned, the approach actively specifies what the SAE should learn, resolving entanglement at the source.
- **Elegant weight binding via detach + scaling**: This design ensures that the encoder and predictor share semantic directions while gradient isolation prevents reconstruction objectives from interfering with semantics, reflecting a deep understanding of gradient conflicts in multi-task learning.
- **Seamless integration of ontology knowledge**: By demonstrating the equivalence between predictor weights and ELEmbeddings, the method eliminates the need for a separate embedding space and applies axiom constraints directly on SAE parameters — an elegant and efficient solution.
- **Cross-domain methodological value**: The semantic guidance paradigm is not limited to proteins; it carries potential transfer value for LLM interpretability research in any domain with annotated ontologies, such as medical ontologies or legal classification systems.

## Limitations & Future Work

1. **Concept coverage is limited by annotation data**: The $m$ defined activations depend on existing GO annotations and cannot model unannotated novel functions or rare concepts. Semi-supervised or active learning approaches could be explored to extend concept coverage.
2. **Validation restricted to ESM2-15B**: Applicability to other PLMs (e.g., ProtTrans, Ankh) has not been tested, leaving generalizability an open question.
3. **Non-trivial computational cost**: Training the SAE on ESM2-15B requires 4 A800 GPUs with activation width up to 40,000, imposing significant overhead for large-scale applications.
4. **Fixed concept granularity**: Current concept granularity is determined by GO terms, lacking flexible hierarchical or multi-granularity interpretation capabilities.
5. **Limited scale of generation steering experiments**: Steering experiments were conducted for only 7 concepts under a specific 50% masking setting; validation at larger scale and under more diverse configurations is needed.

## Related Work & Insights

- **SAE for LLM interpretability**: Anthropic (Paulo et al. 2024) and OpenAI (Gao et al. 2024) have validated the scalability of SAEs on Claude and GPT-4, respectively; ProtSAE extends this to the protein domain while addressing semantic entanglement.
- **Protein interpretability**: CB-pLM (Ismail et al. 2025) achieves controllable generation via concept bottleneck layers — a complementary approach. CB-pLM modifies model architecture, whereas ProtSAE modifies the interpretability tool.
- **Ontology embedding ELEmbeddings**: The ontology representation learning method of Kulmanov et al. (2019) is ingeniously embedded into SAE training, establishing a new paradigm for incorporating knowledge graphs into deep learning interpretability.
- **Broader inspiration**: The semantic guidance approach could extend to SAE analysis of vision models (using ImageNet class hierarchies as ontologies) and concept-level steering in multimodal models.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing semantic guidance into SAE training constitutes a clear methodological innovation; the weight binding and ontology constraint designs are particularly elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Interpretability visualization, probe evaluation, ablation studies, and steering experiments provide comprehensive coverage with sufficient biological validation
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated, method derivation is rigorous, and the core ideas are accessible to cross-domain readers
- Value: ⭐⭐⭐⭐ Represents an important contribution to protein AI interpretability, with broadly transferable semantic guidance ideas

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease](../../NeurIPS2025/medical_imaging/from_black_box_to_biomarker_sparse_autoencoders_for_interpreting_speech_models_o.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](../../ICLR2026/medical_imaging/controlling_repetition_in_protein_language_models.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](../../ICLR2026/medical_imaging/protein_as_a_second_language_for_llms.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[AAAI 2026\] Investigating Data Pruning for Pretraining Biological Foundation Models at Scale](investigating_data_pruning_for_pretraining_biological_foundation_models_at_scale.md)

</div>

<!-- RELATED:END -->
