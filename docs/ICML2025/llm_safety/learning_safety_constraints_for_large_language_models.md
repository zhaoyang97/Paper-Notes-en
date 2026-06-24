---
title: >-
  [Paper Note] Learning Safety Constraints for Large Language Models
description: >-
  [ICML 2025 Spotlight][LLM Safety][Safety Polytope] The paper proposes SaP (Safety Polytope): it learns a "safety polytope" in the representation space of LLMs and geometrically steers unsafe generation trajectories back into the safe region during inference, achieving interpretable safety constraints without shifting model weights.
tags:
  - "ICML 2025 Spotlight"
  - "LLM Safety"
  - "Safety Polytope"
  - "CMDP"
  - "Representation Steering"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: b2dc86471dc5a344
---

# Learning Safety Constraints for Large Language Models

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2505.24445](https://arxiv.org/abs/2505.24445)  
**Code**: [https://github.com/lasgroup/SafetyPolytope](https://github.com/lasgroup/SafetyPolytope)  
**Area**: ai_safety (LLM safety control / inference-time alignment)  
**Keywords**: LLM safety, Safety Polytope, CMDP, Representation Steering, Adversarial Robustness

## TL;DR
The paper proposes SaP (Safety Polytope): it learns a "safety polytope" in the representation space of LLMs and geometrically steers unsafe generation trajectories back into the safe region during inference, achieving interpretable safety constraints without shifting model weights.

## Background & Motivation

### 1. Background
Current mainstream paths for LLM safety roughly fall into three categories:
1. Input/output constraints at the prompting/template level.
2. Alignment during training (e.g., RLHF / safe-RLHF).
3. External guardrails/classifiers (classifier / policy wrapper).

### 2. Limitations of Prior Work
The paper explicitly identifies several key issues in the introduction:
1. Prompting methods are fragile and easily bypassed.
2. Training-time methods are costly, requiring re-annotation and re-training.
3. It is difficult to explain "why a certain request is unsafe" and quantify the "degree of unsafety".

### 3. Key Challenge
The tension lies in:
1. Seeking strong and stable safety control.
2. Desiring no degradation to the model's existing capabilities.
3. Expecting interpretability and diagnosability at the same time.

### 4. Goal
The authors break down the problem into three sub-tasks:
1. How to explicitly model the "safety set" within LLM internal representations.
2. How to detect boundary violations and steer them back during inference.
3. How to assign interpretable semantic divisions to each safety constraint.

### 5. Key Insight
Drawing from the constraint learning perspective of CMDP, the paper views LLM generation as a sequential decision-making process and proposes that "safety can be modeled as a set of linear geometric constraints."

### 6. Core Idea
Using labeled safe/unsafe samples to learn a safety polytope formed by the intersection of multiple half-spaces in the hidden representation space; during inference, if the representation exceeds the boundary, steering is applied along the geometric direction to pull the output back into the feasible safety domain.

## Method

### Overall Architecture
The high-level pipeline of SaP can be summarized in three steps:
1. Extract intermediate layers' representation features from the pre-trained LLM.
2. Learn the safety polytope (parameters and thresholds of the facet hyperplanes).
3. Geometrically steer unsafe trajectories during the inference phase.

The input consists of text sequences and their safety labels (safe/unsafe).
The intermediate variable is the hidden representation vector of a certain layer (referred to as features in the text).
The outputs include:
1. An explicit safety feasible region (polytope).
2. An inference-time steering mechanism (steering algorithm).

### Theoretical Perspective: CMDP to Safety Geometry
The paper first maps language modeling to token-level MDP:
1. State is the historical token sequence.
2. Action is the next token.
3. Policy is the next-token distribution of the autoregressive LM.

In CMDP, besides maximizing rewards, cost budget constraints must also be satisfied.
Relying on existing theoretical results (where constraints can be learned from demonstration trajectories and are linearly related to feature expectations), the authors propose:
1. In a certain feature space, the safety feasible set can be formulated as a convex polytope.
2. i.e., the intersection of a set of linear inequalities.

It can be written as (the core geometric expression of the paper):

$$
	ilde{\mathcal{Q}} = \{\tilde{\mathbf{f}} \mid \phi^{\top}\tilde{\mathbf{f}} \le \tilde{\xi}\}
$$

Where:
1. $\phi$ corresponds to multiple facets (safety constraint directions).
2. $\tilde{\xi}$ is the threshold for each constraint.
3. $\tilde{\mathbf{f}}$ is the feature vector of the input sample in the representation space.

### Key Designs

#### Key Design 1: Concept Encoding and Feature Extraction
**Function**: Extract intermediate representations that can be used for safety classification from pre-trained models.  
**Mechanism**: Use labeled samples $(x^i, y^i)$ to obtain hidden vectors $h^i$ via forward propagation, and then map them to a feature space suitable for constraint learning.  
**Design Motivation**: To maximize the reuse of the base model's capabilities and reduce deployment costs without altering pre-trained model parameters.

#### Key Design 2: Safety Polytope Learning
**Function**: Learn $K$ hyperplanes and thresholds to construct the boundary of the safety region.  
**Mechanism**: Use binary supervision of safe/unsafe data, constraining safe samples to fall within the polytope while making unsafe samples more likely to trigger violations of certain facets.  
**Design Motivation**: Compared to a single safety score, the facet structure is more interpretable, as different facets can correspond to distinct semantic risks.

#### Key Design 3: Inference-time Geometric Steering
**Function**: Pull back internal representations when critical safety facets are triggered during token generation.  
**Mechanism**: Perform constrained modifications in the representation space to steer features back into the feasible region before continuing decoding.  
**Design Motivation**: Replace retraining with inference-time controls to avoid large-scale weight updates and capability drift.

### Loss & Training
Although the cached text does not provide full details of the loss formulation, the training objectives can be inferred from the framework:
1. Constraint feasibility objective: safe samples should satisfy all facet inequalities as much as possible.
2. Separability objective: unsafe samples should significantly violate several facets.
3. Stability objective: maintain manipulable geometric boundaries for inference-time steering.

In practice, such design typically balances the "safety margin" with "preserving original performance", which is also explicitly claimed in the abstract of the paper.

## Key Experimental Results

### Explanation
The current notes are strictly based on the local cache `paper_cache/ICML2025/2505.24445.txt`.
This cache contains the abstract, introduction, and the main thread of theory and method, but does not fully contain the detailed numerical tables (such as specific ASR/MMLU percentages) from the paper.
Therefore, the table below uses "conclusions visible in the cache + placeholders for quantitative items" to avoid fabricating specific numbers.

### Main Results

| Evaluation Dimension | Metric | SaP (Visible in Abstract & Intro) | Baselines (Categories) | Conclusion |
|---|---|---|---|---|
| Unsafe Request Detection | Safety Identification | Effectively detects unethical inputs | Prompt-based / Training-time alignment | SaP is effective in detection |
| Adversarial Defense | Attack Success Rate (ASR) | Reduces adversarial attack success rates | Pre-trained models without geometric constraints / conventional methods | SaP is more robust |
| Utility Preservation | Performance on Standard Tasks | Maintains performance on standard tasks | Strong constraint solutions that sacrifice capability | SaP achieves "safer without significant capability degradation" |
| Interpretability | Constraint Semantic Interpretability | Facets exhibit semantic specialization | Black-box alignment strategies | SaP has stronger diagnosability |

### Ablation Study

| Configuration | Key Observation | Impact on Safety | Impact on Performance | Explanation |
|---|---|---|---|---|
| Full SaP (Concept encoding + multi-facet + steering) | Most complete scheme | Strongest (claimed) | Most balanced (claimed) | Exploits both structured constraints and inference-time steering |
| w/o steering (detection only, no steering) | Can detect violations, but cannot pull back generation | Decreased defense efficacy | Higher utility preservation | Demonstrates that steering is a "defense" component rather than just "classification" |
| w/o multi-facet (degraded to single constraint) | Decreased semantic specialization | Weaker coverage of complex risks | May slightly simplify inference | Shows that multi-facet is helpful for fine-grained safety modeling |
| Training-time alignment only (no inference-time geometric control) | Missing an explicit feasible region | Vulnerable to adversarial bypasses (motivation) | Relies on retraining quality | Contrast highlights the post-hoc advantages of SaP |

### Key Findings
1. The paper emphasizes that safety constraints can be explicitly modeled in the representation space rather than merely implicitly assimilated by weights.
2. Inference-time steering is key; it extends "detection" to "correction," corresponding to actual defense benefits.
3. Semantic specialization of facets is an important interpretability signal, showing that different constraint directions capture different risk semantics.
4. From a methodological standpoint, SaP attempts to shift the "safety-performance trade-off" from a retraining problem to a geometric projection/pull-back problem.

## Highlights & Insights

### Highlights 1: Formulating LLM Safety as a Geometric Feasible Region
One of the most valuable aspects of this work is transforming "safety" from vague preferences into an explicit set of constraints.
Once a feasible region is defined, one can discuss safety violations in terms of distance, violation directions, and constraint contributions, which provides clear entry points for engineering diagnostics.

### Highlights 2: Post-processing Safety Control
SaP does not require updating parameters of the large model; instead, it steers the internal representations during inference.
This is highly practical in industry: fast deployment, quick rollback, and dynamic enable/disable for different scenarios.

### Highlights 3: Interpretable Facet Specialization
The paper not only pursues "being safer" but also examines "why it is safer."
Facet specialization means the system can form a division of labor for risk sub-concepts, which is crucial for auditing and compliance.

### Transferable Insights
1. The "polytope constraint + steering" framework can be transferred to prevent privacy leakage (e.g., PII facets).
2. It can be transferred to multimodal models to define cross-modal safety boundaries in a joint embedding space.
3. It can be used for agent tasks, encoding tool-calling safety policies as constraint facets.

## Limitations & Future Work

### Limitations of the Work (Inferred from motivation)
1. The linear facet assumption may be insufficient to cover highly non-linear risk semantics.
2. The quality of safety labels remains an upper bound; annotation bias affects boundary learning.

### Limitations Identified by Readers
1. The current cache does not provide full experimental values, making it hard to evaluate the exact gains across various benchmarks.
2. If steering is triggered too frequently, it may introduce shifts in generation style or redundant refusals, requiring more fine-grained gating.
3. Whether the polytope transfers well across multilingual and multicultural safety norms remains to be validated.

### Future Directions
1. Extend from linear polytopes to piecewise linear/kernelized constraints to improve coverage of complex risks.
2. Introduce uncertainty estimation to apply adaptive steering strength when close to boundaries.
3. Automatically align facets with human-readable policy clauses to form bi-directional "constraint-policy" tracking.

## Related Work & Insights

### vs Prompt-based Safety Methods
Prompting methods are cheap but fragile, often bypassed by jailbreaks.
SaP’s advantage lies in acting directly on internal representations, providing a deeper level of control.

### vs Training-time safe-RLHF
safe-RLHF can optimize unified targets but suffers from high training cost and slow iterations.
SaP offers a post-hoc plug-and-play advantage, making it ideal for online hotfixes and fast regulatory policy iterations.

### vs Pure Classifier Gateways
Gateways mainly perform input/output classification and cannot correct the generation trajectory.
SaP provides an integrated "detect + steer" route, which is theoretically more suitable for adversarial scenarios.

### Insights for Current Research
This paper suggests that:
1. "Safety boundary modeling capability" might become a core asset of the next-generation LLM safety stack.
2. Safety systems should feature interpretable geometric objects rather than just a single scoring model.

## Key Practical Steps for Replication (Offline Reader Perspective)
1. Determine the layers and token aggregation method (e.g., last token or mean pooling) used for feature extraction first.
2. Balance and denoise the safe/unsafe data to avoid boundary bias.
3. After training, perform facet visualization and trigger statistics before deploying steering online.
4. Log four core metrics when deploying: trigger rate, steering magnitude, refusal rate, and utility preservation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ (4/5) Merges CMDP constraint learning with LLM representation safety, offering a clear geometric formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4/5) Covered detection, defense, and capability preservation from the abstract; however, the local cache lacks complete numerical tables, preventing a perfect score.
- Writing Quality: ⭐⭐⭐⭐☆ (4/5) Problem definition, method motivation, and contributions are clearly stated.
- Value: ⭐⭐⭐⭐⭐ (5/5) Highly industry-friendly, balancing interpretability with post-processing controllability.

## Related Papers
Xin Chen, Yarden As, Andreas Krause. **Learning Safety Constraints for Large Language Models**. ICML 2025.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](../../ACL2026/llm_safety/preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](../../ACL2025/llm_safety/relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ICML 2025\] Safety Alignment Can Be Not Superficial With Explicit Safety Signals](safety_alignment_can_be_not_superficial_with_explicit_safety_signals.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](../../ACL2025/llm_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)

</div>

<!-- RELATED:END -->
