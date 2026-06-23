---
title: >-
  [Paper Note] Comparing the learning dynamics of in-context learning and fine-tuning in language models
description: >-
  [ICLR 2026][Interpretability][Paper Note] The authors treat In-Context Learning (ICL) and Supervised Fine-Tuning (SFT) as two "learning algorithms" and compare their learning trajectories and internal representations shot-by-shot on a geometrically controllable 2D linear classification toy task. They find that while both achieve similar generalization accuracy
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: 855b57a102c09010
---
# Comparing the learning dynamics of in-context learning and fine-tuning in language models

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=cJAtzOcAnd](https://openreview.net/forum?id=cJAtzOcAnd)
**Code**: https://github.com/basile6/ICLvsSFT  
**Area**: Interpretability / Mechanistic Analysis  
**Keywords**: In-context learning, supervised fine-tuning, learning dynamics, inductive bias, representation collapse

## TL;DR
The authors treat In-Context Learning (ICL) and Supervised Fine-Tuning (SFT) as two "learning algorithms" and compare their learning trajectories and internal representations shot-by-shot on a geometrically controllable 2D linear classification toy task. They find that while both achieve similar generalization accuracy, their mechanisms differ significantly: ICL preserves rich input representations but carries stronger pre-training priors (numerical comparison, pattern matching), whereas SFT collapses representations along the label axis, resulting in higher confidence but greater brittleness.

## Background & Motivation

**Background**: Large language models (LLMs) acquire new tasks via two paths: ICL (inserting exemplar-label pairs into the prompt at inference time without weight updates) and SFT (using labeled data for gradient updates to modify weights). Both can achieve comparable performance, but increasing evidence suggests systematic differences in their inductive biases, order sensitivity, and out-of-distribution (OOD) behavior, where ICL often generalizes more robustly than SFT even when trained on identical data.

**Limitations of Prior Work**: The origins of these differences remain unclear. Most previous comparisons were conducted on natural language tasks where task semantics, vocabulary priors, and data geometry are entangled. This makes it difficult to determine whether observed differences stem from the algorithms themselves or semantic confounders. Furthermore, most studies focus on final scalar metrics (accuracy) rather than "learning trajectories"—how model behavior and representations evolve as the number of exemplars increases.

**Key Challenge**: To cleanly compare ICL and SFT as learning algorithms, one must strip away linguistic priors, precisely control task geometry, and align both trajectories using the **exact same training samples, in the same order, at the same shot count**. Otherwise, any observed difference could be attributed to confounding variables.

**Goal**: Construct a controlled task with minimal confounding to perform a head-to-head comparison of ICL and SFT on matched trajectories, answering two questions: (1) How do their inductive biases differ? (2) Do these differences correspond to distinct internal representation geometries?

**Key Insight**: Both ICL and SFT can be viewed as "learning algorithms." For ICL, different shot counts correspond to independent forward passes with prompts of different lengths (no weight updates); for SFT, each shot count corresponds to a model fine-tuned from scratch on the cumulative training set. By describing both using the same language of "shot count $\rightarrow$ performance/representation evolution," they can be compared as alignable trajectories.

**Core Idea**: Use a 2D linear classification task with controllable angles and label-agnostic semantics as a "microscope." By tracking decision fields, four quantitative metrics, and layer-wise Representational Similarity Analysis (RSA) shot-by-shot, the mechanistic differences between "context-driven" and "weight-driven" learning are illuminated.

## Method

### Overall Architecture
This is a mechanistic analysis paper that does not propose a new model; its "method" is a carefully designed controlled experimental setup. The approach is to construct a 2D linear classification task where the decision boundary angle $\theta$ can be precisely controlled and labels are semantically neutral. A single pre-trained model (primarily Llama3-8B) is tasked with learning via both ICL and SFT. Under **fully matched training samples and sequences**, "K shots per class" is treated as the time axis. The evolution of decision fields, four quantitative metrics (accuracy, smoothness, confidence, inferred angle), and RSA of layer-wise activations are recorded at each shot. Differences in inductive bias and representation geometry are then located by systematically varying task angles, injecting periodic ordering patterns, and replicating across models and tasks.

The task (Fig.1A): Input is a pair of ordered integers $x=(n_1,n_2)$, where $n_1,n_2\in\{0,\dots,99\}$, represented by single tokens. Outputs are two semantically neutral labels " Baz" and " Rud". A task instance is determined by a single parameter $\theta\in[0,180^\circ]$, representing the angle of the ground-truth linear decision boundary relative to the first feature $n_1$. The dataset is class-balanced at every shot count. All $100\times100=10{,}000$ possible inputs can be sampled, allowing the model's "decision field" to be fully mapped.

### Key Designs

**1. Angle-controllable, semantically neutral 2D toy task: Turning off confounders**
To cleanly compare learning algorithms, the main enemy is confounding factors—in NLP tasks, label semantics, vocabulary priors, and data geometry are deeply intertwined. By using 2D integer pairs as input and compressing the decision boundary into a single angular parameter $\theta$, the authors decouple "task difficulty" from "task geometry." Theoretically, tasks with different $\theta$ have similar difficulty, but if a model carries biases (e.g., "generalizing along rows/columns" or "comparison"), it will perform better at specific angles, exposing biases as measurable angular dependencies. Labels like " Baz"/" Rud" are chosen over "Foo"/"Bar" because they are single tokens in most tokenizers and rare in pre-training corpora, minimizing verbalizer priors.

**2. Matching trajectory protocol: Aligning ICL and SFT on the same scale**
Since ICL and SFT mechanisms differ fundamentally, comparing only endpoints is meaningless. The authors enforce that both process the **same training samples in the same intra-shot order**. For ICL, K exemplars per class are randomly sampled without replacement to form the prompt; the performance of the same ordered stream is studied across shot counts. For SFT, a new model is **fine-tuned from the base model from scratch** for each shot count using the same cumulative dataset and order (using AdamW + cosine LR). While this is a liberal use of "learning dynamics," this alignment allows "shot count" to serve as a comparable common time axis.

**3. Four quantitative metrics + Angle/Order probes: Quantifying bias**
Accuracy alone does not reveal bias. The authors track four metrics over each trajectory: (i) **Accuracy** across all $10{,}000$ inputs; (ii) **Smoothness**, defined as $1$ minus the fraction of grid points where the predicted label differs from at least two of its four neighbors; (iii) **Confidence**, taking the maximum softmax probability; (iv) **Inferred angle**, derived by fitting a linear classifier to the model's grid predictions. Furthermore, two probes are used: **Varying task angles**—if a "seen feature bias" exists (generalization along rows/columns), performance should be better at $\theta=0^\circ/90^\circ$. If a "comparison bias" exists (preference for diagonals), $\theta=45^\circ$ should be better, with $\theta=30^\circ$ being overestimated and $\theta=60^\circ$ underestimated (diagonal pulling). **Injecting periodic sequences**—arranging in-context exemplars in periods like "1212" or "1221" to see if the model switches from linear classification to pure pattern matching.

**4. Layer-wise Representational Similarity Analysis (RSA): Revealing "Collapse vs. Preservation"**
Differences in inductive bias must reflect in representations. For every input, authors take the activation of the final query token after each MLP layer (32 layers in Llama3-8B) and compute a cosine similarity matrix ($10{,}000\times10{,}000$) between all pairs. They plot similarity histograms and $400\times400$ sub-matrices (sorted by label) to inspect structure. A key observation: while early layers are similar, SFT exhibits **representation collapse** along the label axis in middle layers (activations clusters into two groups), whereas ICL preserves rich input-specific structures. Three controls (LoRA, freezing unembedding, and tracking SFT dynamics) confirm that collapse is an inherent feature of SFT bound to performance rather than a specific strategy or unembedding effect.

## Key Experimental Results

### Main Results: Similar Generalization, Divergent Biases
In matched data and order settings, Llama3-8B solves the task using both ICL and SFT, with comparable hold-out accuracy and learning speeds. However, SFT consistently maintains higher confidence than ICL at the same shot count, indicating stronger logit-label alignment. Decision fields (especially at low shots) reveal two ICL biases: **Seen feature bias** (extrapolation along rows/columns reusing in-context values) and **Comparison bias** (preferring boundaries near $\theta\approx45^\circ$). These biases remain detectable even after global accuracy converges (200 shots/class).

| Dimension | ICL | SFT | Notes |
| :--- | :--- | :--- | :--- |
| Hold-out Accuracy / Learning Speed | Similar | Similar | Both can solve the task |
| Confidence | Lower | **Higher** | SFT has stronger logit-label alignment |
| Angular Dependency | $\theta=0/45/90^\circ$ better; $\theta=30^\circ$ overestimated | $\theta=0/45/90^\circ$ also better, but weaker diagonal pull | Biases manifest as angular dependencies |
| Order Sensitivity | Period 2 triggers pure pattern matching | — | Short periods have high impact; long periods minimal |
| Mid-layer Representations | Preserves input structure | **Collapses along label axis** | Core difference revealed by RSA |

### Ablation Study: Representation Analysis and Controls

| Configuration | Representation Collapse | Notes |
| :--- | :--- | :--- |
| ICL | None; preserves structure across layers | Exemplars are more similar regardless of class |
| Standard SFT | Obvious collapse from mid-layers | Activations cluster into class-based groups |
| LoRA Fine-tuning | Collapse is mitigated | RSA matrix still closer to SFT than ICL |
| SFT w/ Frozen Unembedding | Collapse persists | Indicates collapse is bound to task performance |

### Key Findings
- **Similar Accuracy $\neq$ Same Mechanism**: ICL and SFT achieve almost identical generalization accuracy on this controlled task, yet their inductive biases and representation geometries are vastly different.
- **ICL Performs "In-context Algorithm Selection"**: A period-2 label arrangement can cause the model to abandon linear classification for pure pattern matching, supporting theoretical predictions that transformers can select algorithms within context.
- **Collapse is an Inherent Feature of SFT**: Qualitative patterns of ICL bias and SFT collapse (mitigated by LoRA) replicate across Qwen3-8B, Gemma3-12B/27B, showing these are not model-specific artifacts.
- **Robustness Across Tasks**: Changing integers to adjectives (semantic version) or performing an XOR of two linear tasks (non-linear version) preserves main trends, though learning is slower and angular differences are subtler.

## Highlights & Insights
- **The "Learning Algorithm as Object" perspective**: By matching training samples, order, and shot counts, the authors successfully align two heterogeneous mechanisms on a common axis, revealing that many "algorithmic differences" are actually hidden confounders.
- **Turning bias into measurable angular dependency**: Projecting inductive bias onto a continuous geometric parameter ($\theta$) transforms vague intuitions (e.g., "the model likes comparing numbers") into falsifiable predictions.
- **RSA + Controlled Experiments lock in the cause of collapse**: Instead of just observing collapse, the authors use LoRA, frozen unembeddings, and fixed-shot dynamics to rule out alternative explanations, establishing collapse as an inherent SFT trait tied to performance.
- **Transferable Insights**: The authors predict that SFT may harm transfer learning due to representation collapse—a "collapse $\rightarrow$ OOD fragility" chain that can explain why fine-tuned models often fail on out-of-distribution data.

## Limitations & Future Work
- **Single Task Family**: The focus on geometrically controllable 2D classification might not capture the complexity of hierarchical or multi-step reasoning.
- **Limited Model Scale**: Experiments focused on mid-sized models ($\ge 8B$).
- **Incomplete Hyperparameter Coverage**: While multiple SFT sweeps were done, the study did not exhaustively test all regularization strategies (e.g., specific weight decay schedules) or calibration techniques that might mitigate collapse.
- **Correlational Evidence**: RSA focuses on the final token activation, and the link between representation and bias is correlational; causal interventions remain a key future direction.

## Related Work & Insights
- **vs. Implicit Optimization/Bayesian views (Von Oswald 2023, Akyürek 2022)**: While those works interpret ICL as literal gradient descent in simple settings, this paper's results on mid-sized LMs suggest ICL is better viewed as conditional inference using pre-training priors rather than literal GD.
- **vs. Doimo 2024 (Representations on MMLU)**: While they report SFT is more task-aligned than ICL, they did not analyze learning dynamics. This paper adds the shot-by-shot trajectory and links collapse to inductive bias and order sensitivity.
- **vs. Representation Compression (Kumar 2022)**: It is known that SFT compresses representations toward task-relevant directions; this work visualizes this as label-axis collapse in a controlled setting and proves it is inherent to SFT.

## Rating
- Novelty: ⭐⭐⭐⭐ The shot-by-shot + layer-wise mechanistic comparison of ICL/SFT as matched learning algorithms is a highly effective perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid multi-dimensional probes (angle/order/cross-model/cross-task) and RSA controls, though limited to mid-sized models and specific task families.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, excellent visual correspondence, and strong connection to existing theory.
- Value: ⭐⭐⭐⭐ The "similar accuracy, different mechanism" and "SFT representation collapse" findings provide practical insights into fine-tuning fragility and ICL vs. SFT selection.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs](reinforcement_learning_fine-tuning_enhances_activation_intensity_and_diversity_i.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/interpretability/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[ICLR 2026\] InputDSA: Demixing, then comparing recurrent and externally driven dynamics](inputdsa_demixing_then_comparing_recurrent_and_externally_driven_dynamics.md)
- [\[ICLR 2026\] Learning to Interpret Weight Differences in Language Models](learning_to_interpret_weight_differences_in_language_models.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](../../ICML2026/interpretability/dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)

</div>

<!-- RELATED:END -->
