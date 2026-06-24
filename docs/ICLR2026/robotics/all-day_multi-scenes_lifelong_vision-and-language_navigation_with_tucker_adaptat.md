---
title: >-
  [Paper Note] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation
description: >-
  [ICLR 2026][Robotics][Lifelong Vision-and-Language Navigation] The authors propose Tucker Adaptation (TuKA), which represents multi-level navigation knowledge across various scenes and environments as high-order tensors. Using Tucker decomposition, the method decouples navigation knowledge into a shared subspace (core tensor + encoders/decoders) and scene/environment-specific expert vectors. Combined with a Decoupled Knowledge Incremental Learning strategy…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Lifelong Vision-and-Language Navigation"
  - "Tucker Decomposition"
  - "Parameter-Efficient Fine-Tuning"
  - "Catastrophic Forgetting"
  - "Multi-level Knowledge Decoupling"
date: 2026-05-08
content_hash: ed402c4e705abf9d
---

# All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation

**Conference**: ICLR 2026  
**arXiv**: [2603.14276](https://arxiv.org/abs/2603.14276)  
**Code**: [https://ganvin-li.github.io/AlldayWalker/](https://ganvin-li.github.io/AlldayWalker/)  
**Area**: Robotics  
**Keywords**: Lifelong Vision-and-Language Navigation, Tucker Decomposition, Parameter-Efficient Fine-Tuning, Catastrophic Forgetting, Multi-level Knowledge Decoupling  

## TL;DR
The authors propose Tucker Adaptation (TuKA), which represents multi-level navigation knowledge across various scenes and environments as high-order tensors. Using Tucker decomposition, the method decouples navigation knowledge into a shared subspace (core tensor + encoders/decoders) and scene/environment-specific expert vectors. Combined with a Decoupled Knowledge Incremental Learning strategy, TuKA achieves all-day multi-scene lifelong VLN, outperforming LoRA variants in SR and forgetting rates across 24 navigation scenarios.

## Background & Motivation

**Background**: VLN agents have evolved from discrete graph navigation to low-level action navigation in continuous environments. However, in practical deployments, agents encounter diverse scenes (bedrooms, living rooms, etc.) and varying environmental conditions (normal, low-light, overexposed, scattering), requiring continuous adaptation through lifelong learning.

**Limitations of Prior Work**: VLN agents fine-tuned on a specific scene suffer from catastrophic forgetting of previous scenes when switching to new ones. Existing LoRA/MoE-LoRA methods can only represent a two-level knowledge structure ("shared matrix + specific matrix"), failing to decouple orthogonal dimensions such as "scene knowledge" and "environmental knowledge."

**Key Challenge**: Navigation knowledge possesses a multi-level structure: core navigation skills (shared across all tasks), scene-specific knowledge (e.g., indoor layouts), and environment-specific knowledge (e.g., visual adaptation to low light). These three layers of knowledge must be learned independently yet shared across tasks where applicable.

**Goal**: To formalize the "All-day Multi-scene Lifelong VLN" (AML-VLN) problem and design a parameter-efficient adaptation method capable of decoupling multi-level knowledge.

**Key Insight**: Utilize the inherent multi-modal decomposition capability of Tucker tensor decomposition—where a core tensor captures shared knowledge and the rows of factor matrices encode scene/environment experts respectively.

**Core Idea**: Employ Tucker decomposition of a fourth-order tensor to simultaneously encode shared core navigation skills, scene experts, and environment experts. Achieving zero-forgetting lifelong navigation through decoupled incremental learning.

## Method

### Overall Architecture
TuKA addresses the "All-day Multi-scene Lifelong VLN" problem, where a navigation agent must continuously learn as it switches between scenes (bedroom, living room...) and environmental conditions (normal, low-light, overexposed, scattering) without losing previously acquired skills. The approach attaches a fourth-order tensor $\mathcal{X}^l \in \mathbb{R}^{a_l \times b_l \times M \times N}$ to each layer of the LLM backbone (Qwen2-7B). This tensor is decomposed via Tucker decomposition into shared and expert components: the core tensor $\mathcal{G}$ stores shared navigation skills, $U^1, U^2$ serve as shared encoders/decoders, each row of $U^3 \in \mathbb{R}^{M \times r_3}$ represents a scene expert, and each row of $U^4 \in \mathbb{R}^{N \times r_4}$ represent an environment expert. When learning the $t$-th scene, only the corresponding scene expert row $U^3[s,:]$ and environment expert row $U^4[e,:]$ are extracted and combined with the shared components to form the adaptation weight $\Delta W_t$. The backbone remains frozen while only these tensor factors are trained. During training, Decoupled Knowledge Incremental Learning (DKIL) constrains the update of the three knowledge types; during testing, no task-id is required as CLIP visual features automatically route to the appropriate expert combination.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}%%
flowchart TD
    DATA["Allday-Habitat Platform<br/>Physical Imaging Synthesis of Degraded Environments<br/>24 Navigation Scenarios"] --> TASK["New Scene Task T_t<br/>s-th Scene + e-th Environment"]
    TASK --> TUKA
    subgraph TUKA["Tucker Adaptation (4th-order Tensor per Layer)"]
        direction TB
        G["Shared Core Tensor G<br/>+ Encoders U¹,U²"] --> DW["Adaptation Increment ΔW_t"]
        U3["Scene Expert U³[s,:]"] --> DW
        U4["Environment Expert U⁴[e,:]"] --> DW
    end
    DW --> ADD["Superimposed on Frozen Backbone<br/>(Qwen2-7B / StreamVLN)"]
    ADD -->|During Training| DKIL["Decoupled Knowledge Incremental Learning (DKIL)<br/>Shared EWC + Expert Consistency + Expert Orthogonality"]
    ADD -->|Test (No Task-ID)| SEARCH["Task Expert Inference Search<br/>CLIP Features Cosine Matching for Experts"]
    DKIL --> OUT["All-day Multi-scene<br/>Forget-free Navigation Actions"]
    SEARCH --> OUT
```

### Key Designs

**1. Tucker Adaptation Architecture: Replacing LoRA's Low-Rank Matrix Decomposition with High-Order Tensor Decomposition**

LoRA and MoE-LoRA compress all knowledge into 2D matrices (one shared matrix + several specific matrices), which cannot independently model orthogonal dimensions like "scene" and "environment." TuKA instead uses Tucker decomposition of a fourth-order tensor. The adaptation increment for each layer is generated as:

$$\Delta W_t = U^1 \cdot (\mathcal{G} \times_3 U^3[s,:] \times_4 U^4[e,:]) \cdot (U^2)^T$$

The scene expert is selected via the 3rd mode (selecting the $s$-th from $M$), and the environment expert via the 4th mode (selecting the $e$-th from $N$), naturally spanning a 2D combinatorial space of "scene $\times$ environment." The dimension alignment is achieved by contracting the tensor using only a single row from the factor matrices, reducing the high-order tensor back to a 2D weight matrix matching the LLM backbone's structure.

**2. Decoupled Knowledge Incremental Learning (DKIL): Different Update Cadences for Shared, Old, and New Knowledge**

The difficulty of lifelong learning lies in the conflicting needs of different knowledge types. DKIL uses three losses:
- **Shared Knowledge EWC ($\mathcal{L}_{ewc}$)**: Imposes Fisher information-weighted quadratic constraints on the core tensor and encoders to prevent shared components from being drastically shifted by new tasks.
- **Expert Consistency ($\mathcal{L}_{co}$)**: Imposes $L_2$ constraints on previously learned scene/environment experts to prevent forgetting.
- **Expert Orthogonality ($\mathcal{L}_{es}$)**: Forces new experts to be orthogonal to existing ones, pushing new knowledge into independent subspaces to avoid interference.

**3. Task Expert Inference Search: Automatic Routing via Visual Features**

During deployment, the agent does not know its current scene or environment index. TuKA stores CLIP visual feature prototypes for each scene and environment during training. At test time, it extracts the CLIP feature of the current observation and uses cosine similarity to match the nearest scene and environment experts.

**4. Allday-Habitat Simulation Platform: Evaluating Physical Degradation**

The authors extended Habitat into Allday-Habitat, using three imaging models (atmospheric scattering, low-light noise, and overexposure clipping) to synthesize degraded environments from normal ones. The final dataset includes 24 scenarios (5 simulated scenes $\times$ 4 environments + 2 real-world scenes $\times$ 2 environments).

## Key Experimental Results

### Main Results (Average SR% over 24 Scenarios)

| Method | Avg SR↑ | Avg F-SR↓ | Note |
|------|---------|----------|------|
| Seq-FT | 11% | High | Sequential Fine-tuning, severe forgetting |
| EWC-LoRA | 15% | - | LoRA + EWC |
| HydraLoRA | ~17% | - | MoE-LoRA variant |
| BranchLoRA | ~18% | - | Branching LoRA |
| **AlldayWalker (TuKA)** | **Best** | **Lowest** | Tucker Adaptation |

TuKA consistently outperforms LoRA-based baselines in SR and SPL across all scenarios with significantly lower forgetting rates.

### Ablation Study

| Configuration | Avg SR | Note |
|------|--------|------|
| w/o Core Tensor Sharing | Decrease | Shared knowledge cannot transfer across tasks |
| w/o EWC Constraint | Major Decrease | Shared knowledge overwritten by new tasks |
| w/o Orthogonality Constraint | Decrease | Interference between new and old experts |
| w/o Expert Consistency | Decrease | Learned experts modified causing forgetting |
| Full TuKA | **Best** | Complete framework |

### Key Findings
- Sequential fine-tuning (Seq-FT) success rates drop to nearly 0 on early scenes (0% for T1-T6), indicating extreme catastrophic forgetting.
- The factor matrices of Tucker decomposition naturally support combinatorial generalization—trained scenes exhibit some robustness in unseen environmental conditions.
- Orthogonality constraints are simple yet crucial for independent learning of new experts.

## Highlights & Insights
- **Hierarchical Knowledge Modeling via Tensors**: The mapping between Tucker decomposition's structure and the "shared skill + scene expert + environment expert" hierarchy is highly intuitive.
- **Dimensionality Alignment**: The technique of selecting single rows for tensor contraction to match LLM weight dimensions is an elegant solution to the high-order to 2D weight mapping problem.
- **DKIL Toolbox**: The combination of EWC, consistency, and orthogonality constraints provides a comprehensive strategy for "consolidation, maintenance, and exploration" in lifelong learning.

## Limitations & Future Work
- **Scalability**: Currently tested on 7 scenes and 4 environments; scalability to hundreds of scenes remains unknown.
- **Fixed Experts**: The number of experts $M$ and $N$ must be predefined, whereas true lifelong learning should support open-ended growth.
- **Visual Routing Dependency**: Inference relies on CLIP feature matching; failure may occur if a new environment differs drastically from the prototypes.
- **Complexity of Environments**: While physically grounded, the current four environments are simpler than real-world conditions (e.g., rain, fog, motion blur).

## Related Work & Insights
- **vs. LoRA**: LoRA's 2D decomposition cannot decouple multi-dimensional knowledge; TuKA extends this to high-order tensors.
- **vs. MoE-LoRA (HydraLoRA/BranchLoRA)**: These methods provide only a two-level "shared + specific" structure, whereas TuKA introduces a third dimension for environment-specific adaptation.
- **vs. Traditional Lifelong Learning**: Methods like EWC/LwF do not consider the hierarchical structure of navigation knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lifelong Embodied Navigation Learning](lifelong_embodied_navigation_learning.md)
- [\[ICLR 2026\] M³E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts](m3e_continual_vision-and-language_navigation_via_mixture_of_macro_and_micro_expe.md)
- [\[ICLR 2026\] VITA: Zero-Shot Value Functions via Test-Time Adaptation of Vision–Language Models](vita_zero-shot_value_functions_via_test-time_adaptation_of_visionlanguage_models.md)
- [\[ICML 2026\] Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation](../../ICML2026/robotics/turning_adaptation_into_assets_cross-domain_bridging_for_online_vision-language_.md)
- [\[ICLR 2026\] On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations](on_robustness_of_vision-language-action_model_against_multi-modal_perturbations.md)

</div>

<!-- RELATED:END -->
