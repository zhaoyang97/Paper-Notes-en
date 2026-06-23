---
title: >-
  [Paper Note] ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior
description: >-
  [ICML 2026][Interpretability][Grokking] ExPLAIND unifies three traditionally separate interpretability strands—"model component attribution, data attribution, and training trajectory attribution"—into a single theoretical framework. By strictly rewriting models trained with AdamW as kernel machines (an extension of the Exact Path Kernel, EPK), it derives add
tags:
  - ICML 2026
  - Interpretability
  - Grokking
date: 2026-05-08
content_hash: 58505f5610538eb4
---
# ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior

**Conference**: ICML2026  
**arXiv**: [2505.20076](https://arxiv.org/abs/2505.20076)  
**Code**: https://github.com/mainlp/explaind  
**Area**: Interpretability  
**Keywords**: Path Kernel, Attribution, Training Dynamics, Grokking, Influence Scores

## TL;DR
ExPLAIND unifies three traditionally separate interpretability strands—"model component attribution, data attribution, and training trajectory attribution"—into a single theoretical framework. By strictly rewriting models trained with AdamW as kernel machines (an extension of the Exact Path Kernel, EPK), it derives additive influence scores indexed by parameter/sample/training step. These can be accumulated along any dimension to explain model behavior at any granularity, newly characterizing the learning phases of Grokking and the two-stage dynamics of EuroLLM pre-training.

## Background & Motivation
**Background**: Post-hoc interpretability generally attributes model behavior along three main axes: attribution to **model components** (which layer/neuron), **training data** (which sample influenced the prediction), or **training dynamics** (how the training process evolved). Simultaneously, each method is often tied to a specific granularity on the "local-global" spectrum: providing either fine-grained local explanations or coarse-grained global characterizations.

**Limitations of Prior Work**: These three perspectives and different granularities are almost always **used in isolation**. Component-only explanations ignore the influence of individual training samples and fail to show how components evolve during optimization; data-centric explanations obscure how different model parts internalize those samples; and single-granularity analysis misses patterns that only emerge at another level. This fragmentation leads to a lack of a unified view on how "data, components, and training dynamics **jointly** shape behavior," leaving critical interactions overlooked. Existing work probing training dynamics (e.g., probing or circuit finding at different checkpoints) mostly treats each training step as a discrete event, **lacking a theory to connect checkpoints**.

**Key Challenge**: Interpretability requires a unified framework that spans multiple "perspectives" (components/data/steps) and "granularities" (local ↔ global) with strict theoretical connections between checkpoints. Existing methods either lack theoretical grounding or isolate these dimensions.

**Goal**: ① Provide a theoretically grounded framework integrating components, data, and training trajectories that supports arbitrary granularity; ② Adapt it to modern real-world training (AdamW, weight decay, dynamic learning rates, mini-batches) rather than idealized gradient descent; ③ Use it to re-explain emergent phenomena such as Grokking and LLM pre-training.

**Key Insight**: The authors start from the **Exact Path Kernel (EPK, Bell et al. 2023)**. EPK strictly (not approximately) rewrites a model trained via gradient descent as a kernel machine, centered on the step-wise comparison of training sample gradients and test sample gradients via dot products. However, the original EPK does not cover real-world components like first/second-order momentum estimation, weight decay, dynamic learning rates, or mini-batches.

**Core Idea**: By generalizing EPK to AdamW, an **exact additive decomposition** of final predictions along the "data × parameter × training step" axes is obtained. An "influence tensor" is then defined; accumulating it along different axes yields parameter-level, data-level, or step-level attributions at any granularity. This unifies the three perspectives using the same set of scores. ExPLAIND stands for *Exact Path-Level Attribution Integrating Network and Data*.

## Method

### Overall Architecture
ExPLAIND is a "decomposition-accumulation" analytical framework rather than a pipeline: it first **exactly decomposes** the final prediction of an AdamW-trained model into countless atomic influence terms (each tied to a training step $s$, a parameter $\theta^{(i)}$, a training sample $x_k$, and an output dimension $j$). These terms are stored in a multi-dimensional "influence tensor $\Gamma$." By **selecting which axes to retain and which to sum over**, one can extract attributions from the parameter, data, or step perspectives at any granularity—from a single parameter to an entire layer, or from a single step to a whole training phase. The framework is validated on CNNs, Transformers, and small LLMs to ensure the "decomposition is exact," its utility is confirmed via parameter pruning, and it is finally applied to case studies on Grokking and EuroLLM.

### Key Designs

**1. Extending AdamW-trained models to Exact Path Kernels: Adapting decomposition to modern optimizers**

The original EPK only covers naive gradient descent and cannot represent momentum, weight decay, dynamic learning rates, and mini-batches in AdamW, causing it to fail on real training. Theorem 3.1 generalizes EPK to AdamW: starting from AdamW parameter updates and using indicator variables for mini-batches, the prediction change at each step is written as "parameter gradients induced by training samples" plus "additional terms from decoupled weight decay." The final prediction is exactly decomposed as:

$$f_{\theta_N}(x)=f_{\theta_0}(x)-\sum_{k=1}^{M}\sum_{s=0}^{N-1}\phi_s^{test}(x)\,\phi_s^{train}(x_k)-\sum_{s=0}^{N-1}\phi_s^{test}(x)\,\mathbf{r}_s$$

where the test feature map $\phi_s^{test}(x):=\int_0^1\nabla_\theta f_{\theta_s(t)}(x)\,dt$ is integrated between adjacent parameters (using 100 integration steps), the training feature map $\phi_s^{train}(x_k)$ accumulates gradients weighted by learning rates $\alpha_{s,i}$ and normalized by second moments $\sqrt{\hat v_{s+1}}+\epsilon$, and $\mathbf{r}_s:=\alpha_s\lambda\theta_s$ captures weight decay effects. Corollary 3.2 provides the version for gradient descent with momentum, and Corollary 3.3 further generalizes the decomposition to **intermediate activations** and **differentiable functions of the output (e.g., loss)**.

**2. Influence Tensor and Axis-wise Accumulation: Supporting "Any Perspective × Any Granularity" with a single set of additive scores**

With the exact decomposition, the atomic influence of "training step $s$, parameter $\theta^{(i)}$, training sample $x_k$, and output dimension $j$ for prediction $x$" is defined as:

$$\psi(s,\theta^{(i)},x_k,x)_j:=\phi_s^{test}(x)_{j,i}\cdot\phi_s^{train}(x_k)_i$$

The influence of the regularization term $\psi^{reg}$ is defined similarly. A key property is that these scores are **additive and their sum exactly equals the model prediction** (Eq. 2). By storing atomic scores in the influence tensor $\Gamma(\mathcal S,\Theta,\mathcal X,\mathcal X_{pred})_{\mathcal J}$ and defining accumulated influence $\Psi:=\mathrm{sum}(\Gamma)$, the perspective and granularity are determined by which axes are summed. For example, summing over the training set yields $\Psi(\theta^{(i)},x)$ for a single parameter's effect; summing over all parameters in a layer yields layer-level attribution $\Psi_s(\Theta_{L},x)$. Absolute importance $\bar\Psi$ and the difference $D$ between $\Psi$ and $\Psi^{reg}$ are also defined. This design integrates previously fragmented perspectives into one framework.

**3. Validating score utility via pruning and scaling to LLMs via approximations**

Exact decomposition does not guarantee "useful" scores. The authors perform validation and efficiency optimization. **Validation**: Ranking CNN parameters by kernel importance $\Psi_S(\theta)$ and zeroing all but the TOP-$cD$ parameters (excluding the output layer) performs comparably to magnitude pruning (Li et al., 2017) at 70%-99% sparsity without retraining, while maintaining lower KL divergence. This proves the influence scores faithfully quantify parameter contributions. **Efficiency**: A naive implementation has a memory complexity of $\mathcal O(NDMO)$ ($N$ steps, $D$ parameters, $M$ samples, $O$ outputs). Two solutions: ① "Early accumulation" of training gradients along the data axis before dot products if only component-wise influence is needed; ② Step sub-sampling by ranking training steps by absolute loss change. On MNIST, reconstructed predictions remain accurate even with ~60% step sparsity.

## Key Experimental Results

### Main Results: Exactness of the EPK Representation
The foundation of ExPLAIND is that the "decomposition must be exact." The authors verify the consistency between EPK reconstruction and original model predictions across three models/tasks.

| Model / Data | Integration Steps | EPK Acc. | KL Divergence |
| :--- | :--- | :--- | :--- |
| ResNet9 / CIFAR-2 | 100 | 1.0 | 0.0 |
| Transformer / MOD-113 | 100 | 1.0 | 0.0 |
| CNN / MNIST | 100 | 1.0 | 0.0 |
| Transformer / MOD-113 | 10 | 0.748 | 0.885 |

(Standard 100 steps are used as 10 steps introduce bias.)

### Case Studies: Key Findings

| Case | Structure Revealed by ExPLAIND | Significance |
| :--- | :--- | :--- |
| Grokking (Modular addition) | Three phases: Decoder dominates memorization → Intermediate layers alternate to form circuits → Outer layers (Embedding+Decoder) align to the representation pipeline under high regularization. | Refines Nanda et al.'s three-stage theory, showing late-stage outer layers "align and reuse" rather than create new mechanisms. |
| Causal Verification | Replacing Random Attention/Linear-1 with grokked versions leads to immediate generalization in 200 steps, skipping the memorization phase. | Proves the alignment of outer layers around intermediate representations is causal. |
| EuroLLM-1.7B Pre-training | Two phases: Initially driven by outer MLPs (input-side then output-side); after a phase change (~60K steps), the relative influence of intermediate/lower layers and Attention layers increases. | First parameter-level unified attribution for LLM pre-training dynamics. |

### Key Findings
- **Grokking is "Aligning and Reusing" rather than "Generating New Mechanisms"**: ExPLAIND characterizes the memorization, circuit formation, and cleanup phases. Causal evidence via layer replacement confirms that outer layers align around existing representation pipelines, while regularization suppresses inefficient memorization.
- **Scalable to Real LLMs**: Using early accumulation and step sub-sampling, a single H100 can decompose a batch's loss trajectory from 37 off-the-shelf checkpoints in ~15 minutes with a reconstruction error mean of $4.46\times10^{-8}$.
- **Attribution Shifts Significantly Over Time**: In both Grokking and EuroLLM, data/component attributions differ wildly across phases. This suggests future interpretability should actively expose these critical stages rather than focusing only on the final state.

## Highlights & Insights
- **Strictly rewriting trained models as kernel machines is a powerful move**: The extension of EPK to AdamW makes the additive decomposition exact. This provides a theoretical base where "prediction = sum of influences" for data/components/steps.
- **"Influence Tensor + Axis Accumulation" is an elegant interface**: A single set of atomic scores serves as a "decode-once, slice-anywhere" tool, unifying three previously separate interpretability strands.
- **Pruning as "Validation over SOTA" is rigorous**: Training-free pruning maintaining lower KL divergence at high sparsity proves the scores capture real parameter contributions.
- **Reinterpretation of Grokking**: The "outer layer alignment" explanation combined with causal layer substitution provides a new, falsifiable perspective on a well-studied phenomenon.

## Limitations & Future Work
- **Parameter-level and Non-causal**: ExPLAIND does not intrinsically provide causal explanations. Its naive data influence scores are comparable to TracIn but inferior to TRAK for causal data attribution.
- **Computational Overhead**: Despite optimizations, the costs are significant. Memory for full scores follows $\mathcal O(NDMO)$; large models are limited by the availability of intermediate checkpoints.
- **Limited Scale of Case Studies**: Studies were conduct on small Transformers and a 1.7B EuroLLM (first phase only). Whether conclusions on modular addition generalize to larger models or complex tasks remains to be seen.
- **Future Directions**: Identifying mechanistic circuits at the activation level, learning approximate accumulated scores for efficiency, and broader comparisons with non-unified attribution methods.

## Related Work & Insights
- **vs. EPK (Bell et al. 2023)**: EPK rewrites gradient descent models for data attribution. ExPLAIND extends this to AdamW and generalizes it to parameter and training step dimensions.
- **vs. LCA / POLCA (Loss Change Allocation)**: Similar concept of decomposing loss, but those are approximate and lack links to training data. ExPLAIND is exact and provides the data-parameter-step link.
- **vs. TracIn / TRAK (Data Attribution)**: TracIn uses step-wise gradient products but lacks a theoretical link to model output. TRAK remains stronger for causal data attribution.
- **vs. DualXDA (Yolcu et al. 2025)**: Unifies data and input features. ExPLAIND unifies parameters, steps, and data; the two are complementary.
- **vs. Probing/Circuit Finding**: These often treat checkpoints as independent. ExPLAIND uses the Path Kernel to connect them theoretically.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extending EPK to AdamW and unifying three perspectives is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong on exactness and validation, but case studies are limited in scale and causal data attribution is a relative weakness.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theory and clear case studies, though the high density of formulas creates a barrier for entry.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretically grounded, unified interpretability toolbox that scales to real-world LLMs.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICML 2026\] OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization](omnisapiens_a_foundation_model_for_social_behavior_processing_via_heterogeneity-.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](../../NeurIPS2025/interpretability/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](discovering_implicit_large_language_model_alignment_objectives.md)

</div>

<!-- RELATED:END -->
