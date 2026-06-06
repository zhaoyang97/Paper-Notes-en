---
title: >-
  [Paper Note] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining
description: >-
  [ACL 2026][Interpretability][sparse crosscoder] By training a shared feature dictionary across multiple pretraining checkpoints of the same LLM using a sparse crosscoder…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "sparse crosscoder"
  - "pretraining dynamics"
  - "RelIE"
  - "causal attribution"
  - "emergence of syntactic concepts"
date: 2026-05-08
content_hash: 9964c0b82af0a4e4
---

# Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining

**Conference**: ACL 2026  
**arXiv**: [2509.05291](https://arxiv.org/abs/2509.05291)  
**Code**: [github.com/bayazitdeniz/crosscoding-through-time](https://github.com/bayazitdeniz/crosscoding-through-time)  
**Area**: LLM Interpretability / Training Dynamics / Representation Learning  
**Keywords**: sparse crosscoder, pretraining dynamics, RelIE, causal attribution, emergence of syntactic concepts

## TL;DR
By training a shared feature dictionary across multiple pretraining checkpoints of the same LLM using a sparse crosscoder, this work proposes Relative Indirect Effect (RelIE) to measure the "emergence, maintenance, or disappearance" of per-feature causal importance as token counts increase. This approach enables the first observation of the concept-level evolutionary trajectory in Pythia/OLMo/BLOOM, where LLMs transition from "specific subword detectors" to "internalized abstract syntactic/cross-lingual detectors."

## Background & Motivation

**Background**: Understanding "when an LLM learns specific capabilities" primarily relies on two methodologies: (a) observing phase transitions in accuracy curves on proxy tasks like BLiMP; and (b) analyzing similarity changes in activation or parameter spaces. Recently, Sparse Autoencoders (SAEs) have been used to decompose dense activations of a single checkpoint into interpretable sparse feature dictionaries, providing fine-grained characterizations of mechanisms such as subject-verb agreement or parenthesis matching.

**Limitations of Prior Work**:

1. Accuracy or activation similarity only indicates *when* a change occurs, without explaining the concept-level shifts occurring inside the model.
2. Training separate SAEs for each checkpoint results in non-commensurable feature spaces, making it impossible to determine if "feature 17 in checkpoint A" is the same concept as "feature 17 in checkpoint B."
3. While the crosscoder framework has been proposed to learn shared feature spaces across models, it has prioritized post-training comparisons (e.g., pretrained vs. instruction-tuned) and has not been applied to the temporal dimension of pretraining.

**Key Challenge**: To track when concepts emerge and stabilize during pretraining, an interpreter capable of direct cross-checkpoint feature comparison is required. Furthermore, a "shared feature dictionary" only reveals which checkpoint has a larger decoder norm for a feature (structural similarity) but does not indicate its actual contribution to a task (causal importance)—these two types of information must be obtained orthogonally.

**Goal**:

- Sub-problem 1: Can a shared sparse feature dictionary be trained across multiple pretraining checkpoints (remaining robust even for early, near-random checkpoints)?
- Sub-problem 2: Within this shared dictionary, how can the causal contribution of each feature be independently measured for each checkpoint to track the "emergence-maintenance-extinction" trajectory?

**Key Insight**: This work extends the crosscoder from Lindsey et al. (2024)—originally for two post-trained models—to a triplet of checkpoints from the same training run. It introduces a new normalized ratio metric, RelIE, based on existing integrated-gradients-based indirect effect tools in the SAE community, to completely separate "structural relativity" from "causal relativity."

**Core Idea**: **Crosscoder + RelIE = Concept-level causal attribution over time**. The crosscoder solves "feature alignment across checkpoints," while RelIE identifies "at which checkpoint a feature's contribution actually occurs."

## Method

### Overall Architecture
The pipeline consists of three steps (see Fig. 1 in the paper): (1) **Phase Transition Identification**: Plot accuracy curves on target tasks (e.g., BLiMP/MultiBLiMP/CLAMS subject-verb agreement) and pairwise cosine similarity heatmaps of mid-layer activations to identify 3-4 representative checkpoints (e.g., 128M/1B/4B/286B tokens for Pythia-1B); (2) **Crosscoder Training**: Jointly train a shared sparse dictionary ($2^{14}$ features) on these checkpoint triplets, where all checkpoints share the same feature indices but have unique encoder/decoder parameters; (3) **Feature Attribution & Annotation**: Calculate the Indirect Effect (IE) for each feature using integrated gradients per checkpoint, apply RelIE for normalized comparison, and manually annotate linguistic functions of top-IE features via "max-activating sequences."

### Key Designs

1.  **Sparse Crosscoder: Shared Sparse Feature Dictionary Across Checkpoints**:
    - **Function**: Encodes mid-layer activations $\mathbf{x}_c$ from multiple checkpoints into a unified sparse feature space $\mathbf{f}$, ensuring "feature $i$ in checkpoint A" naturally aligns with "feature $i$ in checkpoint B."
    - **Mechanism**: Each checkpoint $c$ has unique encoder/decoder weights $W_{\text{enc}}^c, W_{\text{dec}}^c$ but shares a single feature activation vector. Encoding follows $\mathbf{f}=\text{ReLU}(\sum_c W_{\text{enc}}^c \mathbf{x}_c + \mathbf{b}_{\text{enc}})$. The loss is the sum of reconstruction errors across checkpoints plus an aggregated sparsity penalty $\sum_c \sum_i \mathbf{f}_i \lVert W_{\text{dec},i}^c \rVert_2$. This aggregated sparsity encourages the dictionary to contain "shared features" used by all checkpoints and "unique features" important only to specific ones.
    - **Design Motivation**: Traditional SAE feature spaces are single-point and non-commensurable. The crosscoder translates "shared vs. unique" into readable signals (decoder norm magnitudes), making temporal feature evolution comparable for the first time. It remains robust for very early, near-random checkpoints (validated in §6.2 with $\Delta \text{CE} < 0.35$).

2.  **Relative Indirect Effect (RelIE): Normalized Causal Importance Ratio Between Checkpoints**:
    - **Function**: Quantifies when a specific feature actually influences task performance as a relative value (0-1), independent of "structural relativity" (RelDec).
    - **Mechanism**: For each feature $f_i$ and checkpoint $c$, integrated gradients are used to approximate the indirect effect on the task metric $m(x) = \log p(t_{\text{wrong}}|x) - \log p(t_{\text{correct}}|x)$ (the change in $m$ upon zero-ablation). The normalized ratio is defined as $\text{RelIE}_{2\text{-way},i} = |\hat{\text{IE}}_{ig,i}^{c_2}| / (|\hat{\text{IE}}_{ig,i}^{c_1}| + |\hat{\text{IE}}_{ig,i}^{c_2}|)$, extended to a one-vs-all vector for triplets. A RelIE near 1 indicates the feature's causal role is exclusive to $c_2$, while 0.5 suggests contributions in both (shared causal feature).
    - **Design Motivation**: The original RelDec (decoder norm ratio) from Lindsey et al. only reflects if a feature structure is strongly utilized; a feature might be strongly learned yet irrelevant to the task. RelIE directly attributes "task performance," filtering out task-agnostic noise and retaining features that drive behavior—ablation in Appendix E confirms RelIE exposes more task-specific features than RelDec.

3.  **Phase Transition Identification + Checkpoint Triplet Selection Strategy**:
    - **Function**: Automatically selects "key nodes where concepts significantly shift" from a long sequence of pretraining checkpoints.
    - **Mechanism**: Two signals are tracked in parallel: (a) accuracy curves on target benchmarks (to find jumps in performance); (b) pairwise cosine similarity heatmaps of **mid-layer** activations (to find jumps in representational structure). Mid-layers are chosen as they capture high-level linguistic/cross-lingual abstractions, whereas earlier/later layers are tied to input/output. These signals are often asynchronous—for instance, OLMo's accuracy stabilizes after 33B tokens, but activation similarity continues to change until 3T tokens—making "saturated accuracy but evolving representations" a high-value area for crosscoder research.
    - **Design Motivation**: Training crosscoders on every checkpoint pair is computationally expensive. This "dual-signal phase transition" strategy ensures selected triplets contain both behavioral and representational shifts.

### Loss & Training
The crosscoder loss combines reconstruction error and aggregated sparsity. The dictionary size is fixed at $2^{14}$. Training data consists of 400M tokens sampled from Pile/Dolma/mC4 multilingual subsets for respective models. IE is approximated via integrated gradients with zero-ablation as the patching method. Three model families (Pythia-1B, OLMo-1B, BLOOM-1B) provide different perspectives: Pythia for dense early emergence, OLMo for long-term maintenance, and BLOOM for cross-lingual abstraction.

## Key Experimental Results

### Main Results

The table below summarizes core qualitative results (Table 1) from the Pythia-1B [1B↔4B↔286B] triplet crosscoder, classifying features by RelIE:

| Category | RelIE [1B, 4B, 286B] | Feature Example | Meaning |
|---|---|---|---|
| 1B-4B Shared | [0.53, 0.33, 0.15] | Subword detector: -ans (vans/cans...) | Early specific token detector |
| 1B-286B Shared | [0.52, 0.01, 0.46] | Detector for 'man' (singular noun) | Lexical feature across stages |
| 4B Unique | [0.00, 1.00, 0.00] | Multi-word scientific compound noun | Transient abstract concept |
| 4B Unique | [0.02, 0.68, 0.30] | Plural person nouns (people, students) | Emergence of collective abstraction |
| 286B Unique | [0.00, 0.18, 0.82] | Nominalized nouns (reactions, inclusion) | Late-stage stable syntactic abstraction |
| 286B Unique | [0.00, 0.01, 0.99] | Preposition detector | Function word detection after accuracy plateaus |

Multilingual expansion (BLOOM-1B on CLAMS): At 6B, there are language-specific detectors for "év" subwords (fra/por/spa). By the 55B-341B stage, these evolve into a shared "multilingual relative pronoun detector" (que/that/who/aladhi), demonstrating a "language-specific → cross-lingual abstraction" consolidation trajectory.

Quantitatively, Pythia-1B shows a jump in BLiMP accuracy from ~50% to >90% between 128M and 4B tokens. Subsequently, while accuracy is stable from 4B to 286B, mid-layer activation similiarities continue to evolve, proving RelIE captures conceptual evolution during accuracy plateaus.

### Ablation Study

| Configuration | ΔCE / Key Observation | Meaning |
|---|---|---|
| Crosscoder on trained triplet | $\Delta$CE < 0.2 | High reconstruction quality for mid-late checkpoints |
| Crosscoder on early + late mix | $\Delta$CE ≈ 0.35 (slight increase + more dead features) | Early, near-random checkpoints can still learn sparse dictionaries |
| RelDec only (Appendix E) | Exposes many task-irrelevant features | Limitations of structural-only relativity |
| RelIE (Ours) | Significantly narrows to task-driving features | Validates necessity of causal normalization |

### Key Findings

- **Concept evolution follows a "concrete → abstract" trajectory**: 1B features are subword/irregular form detectors; 4B features include collective nouns; 286B features are function words like prepositions and nominalizations. This aligns with the "memorize tokens first, learn structure later" hypothesis.
- **Accuracy plateaus $\neq$ representational stasis**: OLMo accuracy saturates at 33B, but mid-layer representations evolve until 3T. New features (e.g., "plural nouns for occupations/skills") prove LLMs continue reorganizing internal representations even after benchmark scores peak.
- **Cross-lingual abstractions merge late in pretraining**: BLOOM's early language-specific detectors for "év"-style subwords merge into cross-lingual relative pronoun detectors between 55B-341B.
- **Early checkpoints are coverable by sparse dictionaries**: Unlike traditional SAEs that fail on random models, the crosscoder's aggregated sparsity learns interpretable dictionaries even at 128M tokens, expanding the scope of mechanistic interpretability.

## Highlights & Insights

- **The decoupling of RelDec and RelIE** is a significant methodological contribution, separating "structural strength" (how strongly a feature is encoded) from "causal role" (how much it affects behavior).
- **Dual-signal Phase Transition Identification** (accuracy + mid-layer activation similarity) ensures crosscoder training resources are focused on intervals where representations shift, even if behavioral metrics are stagnant.
- **The trajectory from subword to preposition detectors** provides a vivid picture of AI language acquisition: the model doesn't learn grammar all at once but starts with frequency-based token patterns before reorganizing them into syntactic categories. This mirrors the "item-based → schema-based" theories in cognitive science.

## Limitations & Future Work

- The dictionary size is fixed at $2^{14}$; whether this needs to scale for much larger models remains unverified, and computational costs for every triplet are a factor.
- Tasks are focused on subject-verb agreement; whether other linguistic phenomena (anaphora resolution, long-range dependencies) follow the same "concrete → abstract" trajectory requires expansion.
- RelIE relies on zero-ablation as a replacement; since zero-activation isn't always in-distribution, future work could test mean-ablation or distribution-matched ablation.

## Related Work & Insights

- **vs. Lindsey et al. 2024**: While they applied crosscoders to post-training (pretrained vs. instruction-tuned), this work is the first to extend them to multiple pretraining checkpoints within a single run.
- **vs. Kangaslahti et al. 2025 (POLCA)**: POLCA identifies *when* phase transitions occur; this work identifies *what* specific representational roles emerge.
- **vs. Marks et al. 2025 / Hanna & Mueller 2025**: These works focus on circuit analysis of a final checkpoint; this work adds a "when are mechanisms formed" perspective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives](do_llms_capture_embodied_cognition_and_cultural_variation_cross-linguistic_evide.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICML 2026\] Memorization Dynamics of Fill-in-the-Middle Pretraining](../../ICML2026/interpretability/memorization_dynamics_of_fill-in-the-middle_pretraining.md)

</div>

<!-- RELATED:END -->
