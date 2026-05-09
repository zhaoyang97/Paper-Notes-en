---
title: >-
  [Paper Note] V-CECE: Visual Counterfactual Explanations via Conceptual Edits
description: >-
  [NeurIPS 2025][Image Generation][Counterfactual explanation] V-CECE proposes the first black-box visual counterfactual explanation framework that systematically reveals the *explanatory gap* between human and neural network semantic understanding. It guarantees edit-set optimality via WordNet knowledge graphs and the Hungarian algorithm, and executes concept-level edits using Stable Diffusion. The key finding is that CNN classifiers are severely misaligned with human semantic reasoning (requiring 5+ edit steps), whereas LVLMs (Claude 3.5 Sonnet) are highly aligned with humans (requiring only 2–3 steps).
tags:
  - NeurIPS 2025
  - Image Generation
  - Counterfactual explanation
  - concept editing
  - black-box
  - knowledge graph
  - diffusion models
date: 2026-05-08
content_hash: fb76ce0781b35999
---

# V-CECE: Visual Counterfactual Explanations via Conceptual Edits

**Conference**: NeurIPS 2025
**arXiv**: [2509.16567](https://arxiv.org/abs/2509.16567)
**Code**: [Project Page](https://nickspanos55.github.io/vcece)
**Area**: Explainable AI / Counterfactual Explanation / Diffusion Models
**Keywords**: Counterfactual explanation, concept editing, black-box, knowledge graph, diffusion models

## TL;DR

V-CECE proposes the first black-box visual counterfactual explanation framework that systematically reveals the *explanatory gap* between human and neural network semantic understanding. It guarantees edit-set optimality via WordNet knowledge graphs and the Hungarian algorithm, and executes concept-level edits using Stable Diffusion. The key finding is that CNN classifiers are severely misaligned with human semantic reasoning (requiring 5+ edit steps), whereas LVLMs (Claude 3.5 Sonnet) are highly aligned with humans (requiring only 2–3 steps).

## Background & Motivation

**State of the Field**: Counterfactual explanation is an important tool in explainable AI—revealing model decision rationale through the lens of "how would the classification change if X were modified?" Existing methods are divided into white-box (requiring gradient access) and black-box (requiring no internal access) approaches.

**Limitations of Prior Work**: Existing counterfactual image generation methods suffer from three major issues: (1) edits are dispersed and uninterpretable (ACE, DiME, etc. produce pixel-level changes that humans cannot comprehend); (2) over-reliance on training to guide generation (white-box methods require days of training); (3) most critically, all semantic counterfactual methods assume classifiers reason at a human semantic level—an assumption that has never been validated.

**Root Cause**: Do humans and neural network classifiers understand semantics in the same way? If not, using human-interpretable concept edits to explain CNN classification decisions is inherently misleading—more dangerous than uninterpretable adversarial edits, as it introduces a false sense of interpretability.

**Paper Goals**: Two progressive questions: (1) Can the decision process of a classifier be explained using human-level semantics? (2) If so, what is the minimum set of semantic edits required to flip the classification label?

**Starting Point**: Decompose counterfactual explanation into two independent stages—first compute the optimal semantic edit set on a knowledge graph (model-agnostic), then execute edits using a frozen diffusion model (avoiding training bias), and finally verify effectiveness via classification outcomes.

**Core Idea**: Use knowledge graphs to guarantee edit optimality and frozen diffusion models to ensure evaluation fairness, thereby systematically measuring the semantic understanding gap between humans and models.

## Method

### Overall Architecture

A two-stage pipeline: (1) **Explanation stage**: Given concept sets for source class $L$ and target class $L^*$, solve for the minimum-cost edit set $E$ (comprising insertions $I$, deletions $D$, and substitutions $S$) on the WordNet knowledge graph using the Hungarian algorithm. (2) **Generation stage**: Order edits according to one of three strategies, localize target regions using GroundingDINO+SAM, execute edits using Stable Diffusion v1.5 Inpainting, and check after each step whether the classifier flips its label.

### Key Designs

1. **Optimal Edit Guarantee (Knowledge Graph + Hungarian Algorithm)**:

    - Function: Compute the minimum semantic edit set from class $L$ to class $L^*$.
    - Mechanism: Substitution cost = shortest-path distance between two concepts on WordNet, computed via Dijkstra's algorithm. The problem is formulated as a bipartite graph matching—source and target concepts as nodes on each side, edge weights as substitution costs, with virtual nodes added to model insertions/deletions (cost = distance to root node). The Hungarian algorithm solves for the minimum-weight matching with time complexity $O(mn\log n)$.
    - Design Motivation: Provides a deterministic optimality guarantee, unlike the heuristic edit selection of prior methods. Non-executable edits can be automatically excluded by assigning infinite cost.

2. **Three Edit Ordering Strategies**:

    - Function: Determine execution order within the optimal edit set $E$ to flip the label as early as possible.
    - Mechanism: (1) **Local Edits**: An LVLM observes the current image and remaining edits at each step and selects the next operation (image updated each step to prevent logical inconsistencies). (2) **Global Edits**: Aggregates edit frequencies across all images and ranks edits by an Importance score—defined as $(|I_{s_j^*}| - |D_{s_i}| + |S_{s_i \to s_j^*}| - |S_{s_j^* \to s_i}|) / |e \in E|$—to capture systematic classifier biases. (3) **Local-Global**: Selects a local edit subset for a specific image and orders it by global importance.
    - Design Motivation: Local leverages image context but ignores classifier bias; Global captures bias but ignores scene details; Local-Global balances both.

3. **Frozen Diffusion Model for Edit Execution**:

    - Function: Execute concept-level image edits while maintaining fair evaluation.
    - Mechanism: Uses Stable Diffusion v1.5 Inpainting (frozen weights, zero training), DPM++ 2M SDE sampler with 40 steps, GroundingDINO+SAM to generate concept masks, and inpaints only within masked regions. An LVLM (Claude 3.5 Sonnet) determines optimal placement and background fill.
    - Design Motivation: Deliberately avoids fine-tuning the diffusion model on the target dataset, as training would introduce data bias and yield spuriously favorable counterfactual images. A frozen model ensures consistent bias, so differences in evaluation results reflect solely classifier behavior.

### Loss & Training

No training—V-CECE is a fully plug-and-play framework. All modules (classifier, diffusion model, LVLM) are used in a black-box manner with zero training.

## Key Experimental Results

### Main Results

BDD100K (autonomous driving scene classification, Stop/Move):

| Method | FID↓ | CMMD↓ | SR↑ | Avg\|E\|↓ | Training |
|------|------|-------|------|-----------|------|
| ACE l1 (white-box) | 1.02 | - | 99.9% | - | Days |
| TIME (black-box) | 51.5 | - | 81.8% | - | Hours |
| V-CECE+DenseNet Local | 90.42 | 1.101 | 88.9% | 4.77 | N/A |
| V-CECE+Claude3.5 Global | **45.22** | **0.427** | 97.8% | **2.65** | N/A |
| V-CECE+Claude3.5 L-G | 42.76 | 0.364 | **98.1%** | 2.44 | N/A |

### Ablation Study

Human evaluation—edit steps required by the model vs. steps deemed reasonable by humans:

| Classifier | Avg\|E\| Model | Avg\|E\| Human | Visual Correctness (%) |
|--------|-------------|-------------|-------------|
| DenseNet | 5.22 | 2.21 | 59.71 |
| ConvNext | 7.35 | 2.27 | 34.24 |
| EfficientNet | 5.96 | 2.66 | 30.17 |
| Claude 3 Haiku | 2.91 | 1.88 | 69.58 |
| Claude 3.5 Sonnet | 2.19 | 1.33 | 81.20 |
| Claude 3.7 Sonnet | 2.50 | 1.37 | 79.98 |

### Key Findings

- **CNNs exhibit a significant semantic gap with humans**: DenseNet requires 5.22 edit steps to flip the label, whereas humans consider 2.21 steps sufficient. Moreover, 59.7% of flipped DenseNet images already exhibit visual artifacts, indicating that the classifier relies on pixel distribution shifts rather than semantic changes.
- **LVLMs are highly aligned with humans**: Claude 3.5 Sonnet requires only 2.19 steps (close to the human baseline of 1.33 steps), with 81.2% of images visually correct, demonstrating near-human semantic understanding.
- **CNN decisions lack consistency**: The highest-importance concept scores only 0.16–0.23, and 35–55 important concepts are identified, indicating no consistent semantic dependency pattern. LVLMs show the opposite: top concept importance reaches 0.37–0.40, with only 27–31 important concepts.
- **Chain-of-thought reasoning is counterproductive**: Enabling thinking in Claude 3.7 increases the required edit steps (3.78 vs. 3.03) and worsens FID, corroborating prior findings that CoT can be detrimental for visual tasks.

## Highlights & Insights

- **Precise problem formulation**: Decomposing counterfactual explanation into "semantic alignment verification" and "minimum edit computation" as two progressive questions is more fundamental than prior work that addresses only the second step. If a classifier does not reason at the semantic level, explaining it with semantic counterfactuals is misleading.
- **Frozen models ensure evaluation fairness**: The deliberate choice not to fine-tune the diffusion model is elegant—it prevents data bias from contaminating evaluation results, enabling genuine comparisons of semantic understanding across classifiers.
- **LVLM-as-classifier as a new paradigm**: Using LVLMs as classifiers is not only feasible but yields high semantic alignment, offering a new paradigm for explaining opaque commercial models.

## Limitations & Future Work

- **Limited scale of human evaluation**: The current human study is small in scale, limiting statistical power and precision; results should be treated as preliminary insights.
- **Generation quality constraints of diffusion models**: Visual artifacts accumulating after multiple Stable Diffusion v1.5 inpainting steps are unavoidable, potentially conflating "classifier semantic misalignment" with "classification changes caused by image quality degradation."
- **Semantic granularity of the knowledge graph**: WordNet has fixed and incomplete concept granularity, potentially missing visually important concepts that are relevant to classification.
- **Evaluation limited to BDD100K and Visual Genome**: Generalizability to high-stakes domains such as medical imaging requires further validation.
- Future directions: incorporating white-box generative models for comparison, scaling human evaluation with inter-annotator agreement assessment, and testing next-generation diffusion models (SD3/Flux).

## Related Work & Insights

- **vs. ACE/DiME (white-box counterfactuals)**: White-box methods achieve SR up to 99.9% but require gradients and training, producing uninterpretable edits; V-CECE is black-box and training-free, with human-understandable edits.
- **vs. Dervakos/Dimitriou (semantic counterfactuals)**: Prior work requires 12+ edit steps and does not generate images; V-CECE requires only 2–3 steps and produces visualizations.
- **vs. TIME (black-box counterfactuals)**: TIME requires training and does not provide semantic edits; V-CECE with an LVLM classifier achieves superior FID and SR compared to TIME.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of the semantic understanding gap between humans and models; the problem formulation is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers diverse classifiers (CNNs, ViTs, LVLMs) and includes human evaluation, though the human study scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, with precise problem motivation and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ The revealed explanatory gap carries fundamental significance for the XAI field and reshapes the understanding of counterfactual explanation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)
- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[NeurIPS 2025\] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[ICCV 2025\] Looking in the Mirror: A Faithful Counterfactual Explanation Method for Interpreting Deep Image Classification Models](../../ICCV2025/image_generation/looking_in_the_mirror_a_faithful_counterfactual_explanation_method_for_interpret.md)

<!-- RELATED:END -->
