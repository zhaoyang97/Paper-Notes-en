---
title: >-
  [Paper Note] Knowledge Vector of Logical Reasoning in Large Language Models
description: >-
  [ACL 2026][Interpretability][Knowledge Vector] The authors demonstrate that the internal capabilities of LLMs for deductive, inductive…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Knowledge Vector"
  - "Logical Reasoning"
  - "Sparse Autoencoder"
  - "Complementary Subspace"
  - "Activation Steering"
date: 2026-05-08
content_hash: 6b1981d5b54cf43a
---

# Knowledge Vector of Logical Reasoning in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.23877](https://arxiv.org/abs/2604.23877)  
**Code**: https://github.com/lei-nlp-lab/knowledge_vector_acl_2026  
**Area**: Interpretability / Logical Reasoning / Activation Steering  
**Keywords**: Knowledge Vector, Logical Reasoning, Sparse Autoencoder, Complementary Subspace, Activation Steering

## TL;DR
The authors demonstrate that the internal capabilities of LLMs for deductive, inductive, and abductive reasoning can be linearly represented as three nearly orthogonal "knowledge vectors." They propose a complementary refinement framework based on SAE subspace constraints, allowing these vectors to learn from each other while preserving their unique features, thereby consistently improving the performance of the three types of reasoning in steering settings.

## Background & Motivation

**Background**: Knowledge vectors and activation steering have been proven to linearly represent high-level concepts like truthfulness and instruction following in LLMs, allowing model behavior to be regulated by superimposing these vectors onto the residual stream. However, existing work almost exclusively targets specific behaviors (e.g., truthfulness, backtracking); the systematic study of whether "general logical reasoning ability" can be linearly represented remains unexplored.

**Limitations of Prior Work**: Classical logic categorizes reasoning into deduction, induction, and abduction. However, how LLMs internally represent these categories—and whether they are independent or collaborative—remains a black box. Without precise localization, controllable intervention is impossible, and the nature of reasoning within chain-of-thought remains unexplainable.

**Key Challenge**: The authors first extracted naive vectors for each reasoning type using contrastive probes and found that the pairwise cosine similarity between these three vectors was near zero. This suggests that LLMs represent the three types of reasoning in nearly orthogonal subspaces. This contradicts cognitive science, where the three types of reasoning share underlying cognitive operations (e.g., new premises from induction are immediately used in deduction). Thus, "geometric independence" is likely a byproduct of suboptimal LLM representation rather than an optimal one.

**Goal**: (1) Verify whether the three types of logical reasoning satisfy the linear representation hypothesis; (2) Design a mechanism to allow the three vectors to "share what should be shared and remain independent where necessary," validating this as a superior representation through steering performance; (3) Analyze the internal circuits before and after refinement using SAE and activation patching for mechanistic interpretability.

**Key Insight**: Sparse Autoencoders (SAEs) can decompose the LLM residual stream into sparse, interpretable features. By filtering the most discriminative SAE features for each reasoning type and performing QR orthogonalization on their decoder directions, a set of subspace bases preserving the reasoning "fingerprint" is obtained. Jointly optimizing "complementary attraction" and "subspace retention" allows reasoning vectors to learn from each other without overlapping.

**Core Idea**: Use a complementary cosine loss to "pull" the three reasoning vectors closer while using an SAE subspace projection loss to "drag" them back to their respective feature structures, creating complementary reasoning vectors that possess both commonality and uniqueness.

## Method

### Overall Architecture
The method consists of two stages: **(A) Naive Knowledge Vector Extraction**: For deduction, induction, and abduction tasks, pairs are sampled using "strong prompt vs. weak prompt," retaining only samples where the strong prompt succeeds and the weak prompt fails. The residual activations of the generated tokens at layer $l$ are averaged to obtain positive activations $\bar a^+$ and negative activations $\bar a^-$. A BCE linear probe $p_r=\sigma(\theta_r^\top x+b_r)$ is trained, with the probe weight $\theta_r$ serving as the "knowledge vector." Layer 13 is selected for Llama-3.1-8B-it and Gemma-2-9B-it. **(B) Complementary Refinement**: Two additional losses (complementary cosine loss + SAE subspace constraint loss) are added to the probe BCE loss to optimize the refined vectors $\theta_r^{\text{ref}}$. During inference, $\theta_r^{\text{ref}}$ is added to the generated tokens at layer 13 for steering to amplify the corresponding reasoning capability.

### Key Designs

1.  **Complementary Cosine Loss**:
    - **Function**: Pull the directions of different reasoning vectors closer, forcing each vector to absorb "complementary knowledge" from the others.
    - **Mechanism**: For all different types $r \neq s$, minimize $\mathcal{L}_{\text{com}}=-\sum_{r\neq s}\frac{\theta_r^\top\theta_s}{\|\theta_r\|\|\theta_s\|}$; i.e., directly maximize the sum of pairwise cosine similarities.
    - **Design Motivation**: The near-zero cosine similarity of naive vectors implies that the three types of reasoning are "isolated." Cognitive science and the authors' observations (e.g., intermediate products of one reasoning type being used by another) suggest the existence of shared substructures. However, using this loss alone would cause the vectors to collapse into a single direction, losing specificity—hence the need for the second term.

2.  **SAE Subspace Constraint Loss**:
    - **Function**: Anchor each reasoning vector near its specific SAE feature subspace to prevent the complementary loss from pulling it away from its core characteristics.
    - **Mechanism**: Residual activations are passed through a pre-trained SAE (Llama Scope / Gemma Scope) to obtain sparse codes $z$. For each hidden unit $j$, the ratio of mean square activations on positive/negative samples is calculated as $\rho_r(j)=\mu_r^+(j)/(\mu_r^-(j)+\varepsilon)$. Features above the $\alpha=0.9$ quantile are selected, and the Top-$K=3000$ by activation strength form the feature set $\mathcal{F}_r$. The corresponding SAE decoder directions are stacked into $V_r$ and QR-orthogonalized to obtain the basis $U_r$. The projection component of $\theta_r$ outside $U_r$ is penalized: $\mathcal{L}_{\text{sub}}^{(r)}=\|(I-U_rU_r^\top)\theta_r\|_2^2$.
    - **Design Motivation**: The SAE subspace acts as a "fingerprint" that must be preserved for that reasoning type. As long as $\theta_r$ remains within the subspace, it can rotate freely to absorb knowledge from other reasoning types without losing its identity.

3.  **From Refined Vectors to Steering**:
    - **Function**: Use the refined $\theta_r^{\text{ref}}$ for directional addition at layer 13 to improve reasoning quality.
    - **Mechanism**: During generation, $c \cdot \theta_r^{\text{ref}}$ is added to the residual stream at layer 13 for every new token position after the prompt ends. The total loss is $\mathcal{L}=\sum_r\mathcal{L}_{\text{probe}}^{(r)}+\lambda_{\text{com}}\mathcal{L}_{\text{com}}+\lambda_{\text{sub}}\sum_r\mathcal{L}_{\text{sub}}^{(r)}$, with default $\lambda_{\text{com}}=0.1, \lambda_{\text{sub}}=0.01$.
    - **Design Motivation**: The authors found that refined vectors achieve higher peak performance with smaller steering coefficients, indicating they capture "cleaner and more aligned" reasoning directions, consistent with the Linear Representation Hypothesis (LRT).

### Loss & Training
The total objective is shown above; optimized via Adam, lr=$10^{-3}$, batch size 16. SAE subspace threshold $\tau=0.9$, $K=3000$, $\varepsilon=10^{-6}$. All three reasoning types share a joint training loop. Intervention occurs at layer 13 for both Llama-3.1-8B-it and Gemma-2-9B-it. The same paradigm is directly transferred to GPT-OSS-20B.

## Key Experimental Results

### Main Results
Three datasets correspond to three reasoning types: JustLogic (Deduction, acc), DEER (Induction, METEOR), and ART (Abduction, acc). Results for Llama-3.1-8B-it under Greedy decoding:

| Reasoning Type | Dataset | Unsteered | Mono Steering | Complementary | Gain vs Unsteered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Deduction | JustLogic | 48.95 | 55.22 | **56.46** | +7.51 |
| Induction | DEER | 26.36 | 27.13 | **27.55** | +1.19 |
| Abduction | ART | 32.27 | 39.19 | **40.95** | +8.68 |

On Gemma-2-9B-it, deduction improved from 56.86 to 59.05, and abduction from 54.67 to 58.20. On GPT-OSS-20B (MoE), abduction improved from 45.09 to 50.50, proving the method is architecture-agnostic. Trends remain consistent under Sampling@5 decoding, with complementary steering always outperforming mono steering. In cross-task transfer to GSM8K, performance improved from 76.26 (Unsteered) to 79.37 (Complementary steering with deduction vector), indicating that the learned vectors capture more than just dataset bias.

### Ablation Study
Llama-3.1-8B-it full model vs. removing individual components (relative $\Delta$):

| Configuration | Deduction | Induction | Abduction | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full (Comp. + Subspace) | 56.46 | 27.55 | 40.95 | Full method |
| w/o Complementary Enhancement | −2.81 | −0.46 | −3.33 | Reverts to mono; significant drops in deduction and abduction |
| w/o SAE Subspace Constraint | −4.58 | −0.29 | −2.09 | Unconstrained complementarity hurts deduction the most |

### Key Findings
- Both losses are indispensable: removing the complementary loss is equivalent to returning to mono steering; removing the subspace constraint is worse than removing the complementary loss (Deduction −4.58 vs −2.81), showing that "unrestricted complementarity" causes vector collapse, which is more dangerous than forgoing shared knowledge.
- After refinement, the co-activation similarity of SAE features between deduction and induction increased from 0.600 to 0.655, while their similarity with abduction continued to decrease (0.474 to 0.425). This aligns with cognitive patterns where deduction/induction are evidence-based, while abduction focuses on hypothesis selection.
- Activation patching shows core attention heads (e.g., layer 31 head 14) maintain high activation before and after refinement, proving that refinements do not destroy original circuits; new activation heads also appear (e.g., layer 17 head 24 for deduction), with overall activations becoming more concentrated and sparse.
- Text span analysis: Deduction vectors tend to amplify causal connectors like "therefore/since," induction vectors favor quantifiers and "statistical regularity" phrases, and abduction vectors amplify hypothesis-preference expressions like "more plausible/likely," proving the vectors encode semantic cues consistent with human linguistic intuition.

## Highlights & Insights
- "Geometric similarity near 0 $\neq$ optimal representation" is a counter-intuitive entry point. Most steering work assumes independent vectors should be as independent as possible; this paper treats it as a symptom to be cured.
- Using SAE subspaces for "soft anchoring" is a clever engineering solution: traditional regularization uses either hard constraints (e.g., freezing dimensions) or L2 penalties towards the origin. Here, the SAE decoder directions construct a "semantically homogeneous subspace," allowing vectors to rotate freely while retaining identity.
- The dual design of Complementarity + Subspace can be directly transferred to other "shared yet specialized" scenarios: Multi-task LoRA, multilingual knowledge neurons, multimodal heads, etc.
- While the absolute performance gains are modest (max +8), the authors honestly position the value as providing a "controllable, analyzable reasoning handle" rather than simply pushing SOTA, reflecting the restraint typical of mechanistic interpretability work.

## Limitations & Future Work
- Only covers three types of classical logical reasoning; whether analogical reasoning, counterfactual reasoning, or planning can be linearly represented is unverified.
- The complementary loss is only pairwise cosine maximization; as the number of tasks $|\mathcal{P}| > 3$, there is a risk of "centralized collapse," which is not discussed.
- Interventions are performed only at a single location (layer 13/14); multi-layer cascades or dynamic layer selection might yield further improvements.
- Steering coefficients still require manual grid search; learning the steering intensity would be more practical.
- Evaluation is primarily in-domain; OOD generalization beyond GSM8K (e.g., code reasoning, theorem proving) is not tested.

## Related Work & Insights
- **vs. Rimsky et al. 2024 (CAA)**: CAA uses difference vectors from contrastive pairs for steering; this paper upgrades "difference vectors" to "subspace-constrained vectors after complementary refinement," which is more robust and naturally supports multi-task synergy.
- **vs. Venhoff et al. 2025**: They perform steering on specific reasoning patterns like backtracking; this paper raises the granularity to "general logical reasoning categories" and addresses multi-vector collaboration.
- **vs. Wang et al. 2025a (Adaptive Activation Steering)**: That paper focuses on adaptive steering for truthfulness; the method here can be combined orthogonally—using SAE subspaces for refinement followed by adaptive intensity.
- **vs. Cai et al. 2025 (deductive/inductive in LLMs)**: They analyze the interdependence of reasoning types at the behavioral level; this paper provides corresponding geometric evidence and controllable intervention methods at the representational level.

## Rating
- Novelty: ⭐⭐⭐⭐ Treating "geometric independence of reasoning vectors" as an entry point and using SAE subspaces for orthogonalization is a novel and natural combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three reasoning types × three models × two decoding methods + GSM8K cross-task + activation patching + SAE feature analysis provides broad coverage; lacks more OOD tasks.
- Writing Quality: ⭐⭐⭐⭐ The connection between motivation, geometric observation, method, and analysis is clear; Figure 2 is intuitive; some notation could be more compact.
- Value: ⭐⭐⭐⭐ Provides a paradigm for "controllable representation of reasoning types," offering insights for both mechanistic interpretability and multi-property steering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)

</div>

<!-- RELATED:END -->
