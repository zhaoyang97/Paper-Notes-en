---
title: >-
  [Paper Note] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations
description: >-
  [ACL 2026][Interpretability][Sparse Autoencoder] AdaptiveK proposes a Sparse Autoencoder driven by input semantic complexity, allowing simple text to activate fewer features and complex text to activate more. Across 8 au…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Sparse Autoencoder"
  - "Mechanistic Interpretability"
  - "Linear Probe"
  - "Adaptive Sparsity"
  - "Representational Complexity"
date: 2026-05-08
content_hash: edab8a92ef3cc159
---

# AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations

**Conference**: ACL 2026  
**arXiv**: [2508.17320](https://arxiv.org/abs/2508.17320)  
**Code**: https://github.com/hiyukie/adaptiveK  
**Area**: Mechanistic Interpretability / Sparse Autoencoder  
**Keywords**: Sparse Autoencoder, Mechanistic Interpretability, Linear Probe, Adaptive Sparsity, Representational Complexity  

## TL;DR
AdaptiveK proposes a Sparse Autoencoder driven by input semantic complexity, allowing simple text to activate fewer features and complex text to activate more. Across 8 autoregressive LLMs and additional architectural experiments, it improves reconstruction quality, concept disentanglement, and training efficiency while reducing the need for repetitive hyperparameter tuning of fixed TopK values.

## Background & Motivation
**Background**: Sparse Autoencoders (SAEs) have become essential tools for interpreting the internal representations of LLMs. By decomposing model activations into a higher-dimensional but sparse latent dictionary, they aim to ensure each latent corresponds to a more monosemantic and understandable concept, thereby mitigating interpretation difficulties caused by polysemanticity and superposition. Recent methods like TopK SAE, BatchTopK, Gated SAE, JumpReLU, P-anneal, and Matryoshka SAE primarily focus on improving the balance between reconstruction fidelity and sparsity.

**Limitations of Prior Work**: Most existing SAEs impose uniform sparsity constraints on all inputs. For instance, TopK fixes the number of active features for every sample, and methods like L1, Gated, or P-anneal exert similar sparsity pressure across different inputs. However, textual complexity is not uniform: a simple, short concept may require only a few features to explain, whereas specialized long contexts with multiple entities and complex logic demand more capacity. Uniform constraints cause wasted features for simple samples and under-representation for complex ones.

**Key Challenge**: SAE training usually treats sparsity as a global hyperparameter, but the real demand for representational capacity varies locally. Fixing $k$ to accommodate complex samples sacrifices the sparsity of simple ones; maintaining overall sparsity leads to insufficient reconstruction of complex samples. Furthermore, researchers often must train multiple models with different $k$ or sparsity penalty settings to find the optimal Pareto trade-off.

**Goal**: The authors aim to address three specific questions: First, is "contextual complexity" linearly encoded in LLM activations? Second, can this complexity signal dynamically determine the number of active features in an SAE? Third, does this adaptive sparsity outperform fixed-sparsity baselines in terms of reconstruction, interpretability, and training efficiency?

**Key Insight**: The paper draws inspiration from linear probe research, which observes that many high-level attributes—such as sentiment, political stance, spatio-temporal information, and truthfulness—can be extracted linearly from LLM activation space. The authors further hypothesize that although text complexity is multi-dimensional, it is also linearly encoded by LLM representations. If complexity can be read out using a low-cost probe, it can be converted into a dynamic $k$ for the SAE.

**Core Idea**: Use a linear probe to predict input complexity and map it to a sample-specific TopK value. This replaces a "one-size-fits-all" fixed sparsity with a strategy where "complex inputs receive more features, simple inputs receive fewer."

## Method
AdaptiveK can be viewed as a standard TopK SAE augmented with a "complexity controller." This controller does not read the raw text directly but instead processes the LLM hidden activations. It outputs a continuous complexity score, which is mapped via a sigmoid function to determine how many SAE latents to retain for the current sample. Consequently, the SAE dictionary, encoder, and decoder maintain familiar architectures, but the activation function changes from fixed TopK to sample-adaptive TopK.

### Overall Architecture
The pipeline consists of four steps. First, the input context is fed into the target LLM to extract the last token hidden state of a selected layer as the context-level activation $x$. Second, a pre-trained linear probe predicts the complexity score $c$. Third, $c$ is mapped to an adaptive sparsity level $k_{adp}$ within a range defined by $k_{min}$ and $k_{max}$. Fourth, the SAE encoder generates latent pre-activations, retains only the top-$k_{adp}$ activations for the current sample, and the decoder reconstructs the original activation.

A key intuition is that the last token representation aggregates prior contextual information, serving as a readout point for "contextual complexity." The main experiments utilize a dataset constructed from pile-uncopyrighted with 250,000 training contexts and 10,000 test contexts, with complexity scores labeled by GPT-4.1-mini.

### Key Designs
1.  **Linear Probe for Contextual Complexity**:
    - **Function**: Predicts a complexity score (0 to 10) for each input activation to serve as the control signal for AdaptiveK.
    - **Mechanism**: The authors use GPT-4.1-mini to label text complexity across dimensions such as lexical complexity, syntactic complexity, conceptual density, specialized domain knowledge, and logical structure. Ridge regression is then used to predict this score from LLM hidden states. The training objective is defined as $L(w,b)=\frac{1}{n}\sum_i(y_i-(w^Tx_i+b))^2+\frac{\lambda}{2}\|w\|_2^2$.
    - **Design Motivation**: If complexity prediction required complex non-linear models, AdaptiveK would introduce an additional black box. A linear probe is compatible with the "linear representation" hypothesis in mechanistic interpretability and makes the complexity signal more interpretable.

2.  **Mapping Complexity to Adaptive TopK**:
    - **Function**: Assigns an individual active feature budget to each input rather than sharing a global $k$.
    - **Mechanism**: The complexity $c$ is transformed via a sigmoid function to fall within the range $[k_{min}, k_{max}]$: $k_{adp}=k_{min}+\sigma(s((c-c_{min})/(c_{max}-c_{min})-0.5))(k_{max}-k_{min})$. Here, $s$ controls the steepness of the curve; the default configuration uses $k_{min}=20$, base $k=80$, and $k_{max}=320$.
    - **Design Motivation**: Linear mapping might be over-sensitive to extreme complexity, while fixed TopK ignores it. Sigmoid mapping provides a smooth but bounded capacity allocation, preventing simple samples from activating too many latents while avoiding infinite expansion for complex samples.

3.  **Three-Stage Training to Stabilize Probe-SAE Coupling**:
    - **Function**: Prevents the SAE reconstruction objective from distorting the complexity probe while allowing for collaborative optimization.
    - **Mechanism**: Phase 1 trains only the complexity probe. Phase 2 freezes the probe and trains the SAE with $L_{SAE}=L_{recon}+\alpha L_{sparsity}+\beta L_{aux}$ ($\alpha=0.005$, $\beta=1/32$). Phase 3 performs joint fine-tuning of the probe and SAE, using a deviation penalty to ensure the probe does not drift too far from its pre-trained parameters.
    - **Design Motivation**: Immediate joint training might cause the probe to deviate from semantic complexity to satisfy SAE reconstruction. Permanently freezing the probe risks missing opportunities for mutual adaptation. Three-stage training balances semantic stability and task adaptation.

### Loss & Training
Training proceeds through probe pre-training, frozen-probe SAE training, and joint fine-tuning. The probe uses ridge regression with 5-fold cross-validation to select the regularization strength from $\{0.001,0.01,0.1,1.0,10.0,100.0,1000.0\}$, ultimately choosing $\lambda=100.0$.

In the SAE phase, the reconstruction goal is to minimize the difference between the decoder output $\hat{x}$ and the original activation $x$. Sparsity regularization uses a normalized $L_1$ term $\|z\|_1/\|x\|_2$, with an auxiliary loss to handle dead features. The joint fine-tuning objective is $L_{joint}=L_{SAE}+\gamma(L_{probe}+\delta L_{deviation})$, where $\gamma=0.9$, and $\delta$ is dynamically adjusted between 0.01 and 0.5 (starting at 0.2) to prevent probe parameters from deviating excessively from pre-trained values. Adam is used as the optimizer with a learning rate of $1e^{-3}$, 15 warm-up steps, and linear decay after 70% of training progress.

## Key Experimental Results

### Main Results
The main experiments cover 8 decoder-only LLMs: Pythia-70M, Pythia-160M, Gemma-2-2B, Gemma-2-9B, Llama-3.1-8B, Qwen-3-8B, Qwen-3-14B, and Phi-4-14B. All SAEs use a dictionary size of 16,384, compared against 7 baselines: ReLU, ReLU_new, TopK, BatchTopK, Gated, P-anneal, and Matryoshka. Pareto curves show that AdaptiveK achieves lower L2 loss, lower unexplained variance, and lower $1-\cos$ at equivalent sparsity levels across the board.

The linear complexity probe results on Pythia-70M (250,000 training, 10,000 test) support the premise that complexity can be read out linearly.

| Complexity Predictor | RMSE ↓ | Pearson ↑ | Spearman ↑ | Conclusion |
|--------------|--------|-----------|------------|------|
| Linear probe | 1.41 | 0.72 | 0.76 | Close to non-linear models with better interpretability |
| MLP | 1.37 | 0.74 | 0.77 | Slightly better numerically but introduces a non-linear black box |
| XGBoost | 1.42 | 0.71 | 0.74 | Does not outperform the linear probe |

Adaptive capacity allocation is evident: while fixed TopK maintains the same $k$ for all samples, AdaptiveK increases active features with complexity. In one example, where TopK is fixed at 80, AdaptiveK scales from approximately 103 to 394 features based on complexity.

### Ablation Study
The authors ablated the $k$ mapping range, sigmoid steepness, probe weights, and mapping functions. A core finding is that increasing $k_{max}$ consistently improves reconstruction, but the average activation count also rises, indicating that AdaptiveK still follows the standard capacity-sparsity trade-off.

| $k$ Range Setting | Test Set min $k$ | Test Set max $k$ | Average $k$ | Explained Var ↑ | Cosine Sim ↑ | L2 Ratio |
|--------------|----------------|----------------|----------|-----------------|--------------|----------|
| $k_{min}=20,k_{max}=320$ | 96 | 291 | 214 | 0.743 | 0.909 | 0.921 |
| $k_{min}=20,k_{max}=480$ | 132 | 435 | 313 | 0.768 | 0.919 | 0.926 |
| $k_{min}=20,k_{max}=640$ | 170 | 579 | 415 | 0.789 | 0.926 | 0.935 |

Ablation of the sigmoid steepness $s$ shows the method is robust to this hyperparameter. For Gemma-2-2B, changing $s$ from 2.0 to 12.0 expanded the dynamic $k$ range from [143, 225] to [58, 315], yet explained variance and cosine similarity remained nearly constant.

| $s$ | min $k$ | max $k$ | avg $k$ | Explained Var ↑ | Cosine Sim ↑ | L2 Ratio |
|-----|---------|---------|---------|-----------------|--------------|----------|
| 2.0 | 143 | 225 | 188 | 0.738 | 0.908 | 0.911 |
| 6.0 | 96 | 291 | 214 | 0.743 | 0.909 | 0.921 |
| 12.0 | 58 | 315 | 232 | 0.742 | 0.909 | 0.917 |

### Key Findings
- **Complexity is linearly readable**: On Pythia-70M, the linear probe achieved RMSE 1.41 and Pearson 0.72. On Gemma-2-2B, layer-wise analysis showed the probe reaching Pearson 0.814 at layer 22.
- **AdaptiveK outperforms fixed sparsity on the Pareto frontier**: AdaptiveK consistently delivers better results in terms of L2 loss, unexplained variance, and cosine similarity. Some baselines require 10x higher sparsity to match its reconstruction performance.
- **Cross-architecture scalability**: High performance was observed for BERT-340M (cosine 0.89) and T5-small (encoder explained variance 0.97).
- **Practical training efficiency**: On Gemma-2-2B, AdaptiveK took 11,084 minutes total; a single fixed-TopK run took 13,955 minutes. If traditional methods require training 6 sparsity levels for tuning, AdaptiveK is significantly faster.
- **Interpretability metrics support AdaptiveK**: RAVEL scores for disentanglement/cause/isolation were approximately 0.62/0.60/0.65. Sparse Probing indicated that the Top-1 latent captures about 82% of the information available in the full SAE representation.

## Highlights & Insights
- **Transforming "input complexity" from an intuition into a training signal**: While many SAE works assume varying sample difficulty, they still use global sparsity hyperparameters. AdaptiveK's value lies in operationalizing this intuition via a measurable linear probe.
- **Adaptive sparsity is more elegant than simply expanding the dictionary**: To help complex samples, fixed TopK must increase overall $k$, which forces simple samples to activate unnecessary features. AdaptiveK makes capacity follow the sample, aligning closer to the goal of "activating only necessary concepts."
- **Restraint in using a linear probe**: Using ridge regression instead of a complex controller network keeps the method compatible with mechanistic interpretability's linear assumptions and avoids the issue of "explaining a black box with another black box."
- **Separating reconstruction from human-understandable interpretability**: The paper evaluates both mathematical fidelity (L2, variance) and semantic focus (MaxAct, VocabProj). AdaptiveK's tendency to focus on specialized terms in complex biomedical examples—where TopK might include irrelevant functional words—provides strong qualitative evidence.

## Limitations & Future Work
- **Dependency on GPT-4.1-mini for complexity labeling**: The labeling cost limits the training data to 250,000 contexts, significantly smaller than the 500M tokens used in some SAE benchmarks. The stability of scores across languages, code, and mathematics remains to be verified.
- **Context length inconsistency**: There are minor inconsistencies in the paper regarding context length (2048 vs. 1024 tokens), which may affect replication regarding buffer management and token-level evaluation.
- **Heavy reliance on curves over tables**: Pareto frontier graphs demonstrate clear advantages, but detailed numerical tables for every model and sparsity level are lacking, making exact comparisons difficult without source logs.
- **Complexity vs. all interpretive difficulties**: While complexity explains much of the "how many features" question, some concepts might be semantically simple but mechanically complex (e.g., safety refusals, rare token behaviors). Future work could expand capacity signals beyond a single complexity score.
- **Scaling to massive training runs**: While AdaptiveK performs well on smaller datasets, its stability during full token-level training on frontier LLMs requires further validation.

## Related Work & Insights
- **vs. TopK SAE / BatchTopK SAE**: While TopK is stable and simple, its fixed $k$ is its weakness. AdaptiveK preserves the TopK mechanism but shifts $k$ from a global hyperparameter to a sample-level variable.
- **vs. Gated SAE / P-anneal SAE**: These improve sparsity control via gating or annealing but still rely on global pressure. AdaptiveK differs by letting the input content itself determine the required sparsity.
- **vs. Matryoshka SAE**: Matryoshka focuses on feature hierarchies and dictionary levels. AdaptiveK is orthogonal to this and focuses on per-sample budgets; the two approaches could potentially be combined.
- **vs. Linear Probing**: Building on work showing linear encoding of time and space, this paper treats "complexity" as a synthesizable readable attribute. It suggests that many SAE hyperparameters could be automatically controlled by linear properties of model activations.
- **Future Inspiration**: AdaptiveK highlights that interpretability depends not just on the dictionary, but on activation budget allocation. This could extend to multimodal models or agent trajectories using task density or uncertainty as control signals.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using input complexity to adaptively determine SAE sparsity is a clear and useful innovation, even if the core components (linear probe + TopK SAE) are existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 8 LLMs and 7 baselines with diverse metrics; however, the lack of full numerical tables and minor descriptive inconsistencies are noted.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation is clear and method description is thorough; however, some repetitive dimensions and inconsistent context length labels slightly hinder replication.
- Value: ⭐⭐⭐⭐⭐ Extremely insightful for SAE practice, particularly for reducing tuning costs while maintaining semantic focus in latents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](../../NeurIPS2025/interpretability/transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](../../ICML2026/interpretability/prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)

</div>

<!-- RELATED:END -->
