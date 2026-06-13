---
title: >-
  [Paper Note] AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs
description: >-
  [NeurIPS 2025][Dialogue Systems][Access Control] AC-LoRA is an end-to-end system that trains independent LoRA adapters for datasets with different permission levels. At inference time…
tags:
  - "NeurIPS 2025"
  - "Dialogue Systems"
  - "Access Control"
  - "LoRA Adapter"
  - "Information Isolation"
  - "Enterprise LLM"
  - "Multi-Modal"
date: 2026-05-08
content_hash: 6dd9134abea2e62b
---

# AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2505.11557](https://arxiv.org/abs/2505.11557)  
**Code**: [huawei-csl/AC-LoRA](https://github.com/huawei-csl/AC-LoRA)  
**Area**: Dialogue Systems
**Keywords**: Access Control, LoRA Adapter, Information Isolation, Enterprise LLM, Multi-Modal

## TL;DR

AC-LoRA is an end-to-end system that trains independent LoRA adapters for datasets with different permission levels. At inference time, it dynamically retrieves and training-freely merges multiple LoRA outputs based on cosine similarity and user permissions, achieving strong information isolation while matching or surpassing SOTA LoRA mixture methods in response quality.

## Background & Motivation

**Background**: Enterprise-level LLMs are increasingly deployed for internal knowledge management, handling sensitive information such as email records, meeting minutes, and project documents. Such information typically follows organizational hierarchical access control policies, where different employees can only access data within their authorized scope.

**Limitations of Prior Work**: Current LLMs are prone to leaking sensitive information seen during training through memorization, making deployment in scenarios requiring strict access control difficult. The naive solution of training a separate model for each permission group is impractical, as an organization with $n$ permission domains can have up to $2^n$ permission combinations, leading to exponentially growing maintenance costs. Existing LoRA mixture methods (e.g., LoRA Router) require additional training of routing networks and provide no information isolation guarantees.

**Key Challenge**: How can strict information isolation (preventing any user from accessing information beyond their authorization through the LLM) be guaranteed while still leveraging knowledge from multiple permission domains to provide high-quality responses?

**Goal**: (1) A training-free LoRA selection and merging mechanism; (2) An architecture that guarantees information isolation—a LoRA is loaded only when the user has the corresponding permission; (3) Generalizability to multi-modal settings (text + image).

**Key Insight**: Each permission dataset is used to independently train a LoRA adapter and a corresponding document embedding vector store. At query time, relevant LoRAs are retrieved within the user's permission scope via vector similarity, and the similarity scores are directly used as merging weights—serving simultaneously as retrieval criteria and routing weights, eliminating the need to train an additional router.

**Core Idea**: Combine independent LoRAs, permission-filtered retrieval, and similarity-weighted merging to realize a zero-extra-training access control-aware LLM.

## Method

### Overall Architecture

The AC-LoRA system comprises three stages: (1) **Offline preparation**—fine-tuning LoRA adapters separately for each permission dataset and constructing corresponding FAISS vector stores (storing document embeddings with their associated LoRA labels); (2) **Online retrieval**—retrieving the top-k relevant documents and their corresponding LoRAs from the vector store based on the user query and permission scope; (3) **Inference merging**—loading the retrieved LoRAs, generating responses from each independently, and performing weighted merging in the hidden layer using softmax-normalized cosine similarity scores as gating weights.

### Key Designs

1. **Permission-Aware LoRA Retrieval**:

    - Function: Selects relevant LoRAs based on user permission constraints and query semantics.
    - Mechanism: Document embeddings from all permission domains are stored in a unified FAISS vector store, with each record carrying source metadata identifying its associated LoRA. At query time, records are first filtered by user permissions (`filter={"source": {"$in": permission}}`), followed by top-k similarity retrieval. Similarity scores for multiple documents returned from the same LoRA are averaged, and LoRAs falling below a threshold are discarded.
    - Design Motivation: Permission filtering is hard-coded at the retrieval layer, providing an architectural guarantee that LoRAs associated with unauthorized data are never loaded, thus ensuring strong information isolation.

2. **Training-Free Similarity-Weighted Merging (Gate Mechanism)**:

    - Function: Merges outputs from multiple LoRAs weighted by their relevance.
    - Mechanism: The cosine similarity scores of retrieved LoRAs are softmax-normalized to serve as gating weights. During the forward pass, the $B \cdot A \cdot x$ outputs of each LoRA are computed separately and then merged via `torch.einsum("blnd,n->bld", loras, gate)`. This requires no additional training of a routing module.
    - Design Motivation: Cosine similarity is itself a reliable measure of query-dataset relevance. Directly reusing it as a merging weight is both simple and effective. Compared to methods such as LoRA Router that require training a routing module, AC-LoRA achieves "almost zero training."

3. **Cross-Modal Extension and Hint Mechanism**:

    - Function: Supports multi-modal tasks including text-based QA and image generation.
    - Mechanism: The text modality uses TextModel (based on PEFT), while the image modality uses Stable Diffusion's LoRA (the MultiModalSD class); the retrieval and merging pipeline is identical across modalities. Additionally, a "hint" mechanism is provided—when the most relevant LoRA retrieved under full-permission search falls outside the user's permission scope, the system notifies the user that "there may be more relevant information you do not have access to."
    - Design Motivation: The unified vector retrieval and LoRA merging architecture naturally supports multi-modal extension, requiring only a replacement of the underlying model class.

## Key Experimental Results

### Main Results

Evaluated on RepLiQA (enterprise knowledge QA) and FlanV2 datasets.

| Method | RepLiQA Accuracy | FlanV2 Accuracy | Information Isolation | Extra Training |
|--------|-----------------|----------------|----------------------|---------------|
| AC-LoRA | Matches/exceeds SOTA | Matches/exceeds SOTA | ✓ Strong guarantee | ✗ Not required |
| LoRA Router | SOTA | SOTA | ✗ No guarantee | ✓ Router training required |
| Single LoRA (uniform merge) | Lower | Lower | ✓ | ✗ |
| Base Model (no LoRA) | Lowest | Lowest | ✓ | ✗ |

### Ablation Study

| Configuration | Key Observation |
|--------------|----------------|
| k=1 (single LoRA) | Effective when retrieval is correct; lacks complementary knowledge |
| k=3 (multi-LoRA merging) | Optimal balance between retrieval and merging |
| No threshold filtering | Introduces irrelevant LoRAs, degrading quality |
| Uniform merge vs. similarity-weighted | Similarity-weighted significantly outperforms uniform merging |

### Key Findings

- Using cosine similarity directly as merging weights achieves performance comparable to or better than trained routers.
- Information isolation incurs almost no loss in response quality—retrieval within the permission scope is sufficient.
- The "hint" mechanism improves user experience in permission-restricted scenarios.
- The method transfers directly to WikiArts image generation and MMSci multi-modal tasks, validating cross-modal generalizability.

## Highlights & Insights

- **Elegant zero-training routing design**: Reusing retrieval-time cosine similarity as merging weights naturally delegates the routing role to the retrieval process, avoiding additional training and architectural complexity. This design philosophy—"similarity serves as both retrieval criterion and routing signal"—is transferable to other scenarios requiring dynamic combination of experts or modules.
- **Security by design**: Information isolation guarantees do not rely on model behavior but are enforced through architectural-level permission filtering. Even under adversarial conditions, unauthorized LoRAs remain inaccessible.
- **An $n$-complexity solution to the $2^n$ problem**: $n$ permission domains require only $n$ LoRAs; combinatorial retrieval covers all $2^n$ permission scenarios.

## Limitations & Future Work

- Experiments are conducted only on relatively small-scale datasets; scalability to real enterprise-scale deployments (hundreds of permission domains, tens of millions of documents) remains unvalidated.
- Independent fine-tuning of each LoRA may lead to knowledge fragmentation, limiting the ability to integrate cross-domain knowledge.
- Permission changes require retraining the corresponding LoRA and updating the vector store, incurring non-trivial operational costs.
- Similarity-weighted merging may produce incoherent outputs when merging LoRAs from highly divergent domains.
- The security analysis does not address adversarial attacks (e.g., crafting specific queries to bypass permission filtering).

## Related Work & Insights

- **vs. LoRA Router/Retriever (Styx 2024)**: These methods require training an additional router to determine LoRA selection; AC-LoRA is entirely training-free and provides information isolation guarantees.
- **vs. RAG approaches**: RAG enforces access control at the retrieval layer, but the LLM may still leak information through prompt injection or memorization; AC-LoRA isolates information at the model parameter level.
- **vs. multi-model approaches**: Deploying independent models for each permission group incurs $2^n$ cost; AC-LoRA addresses this with $n$ lightweight LoRAs.

## Rating

- Novelty: ⭐⭐⭐⭐ The zero-training design of reusing retrieval similarity as routing weights is elegant.
- Experimental Thoroughness: ⭐⭐⭐ Dataset scale is limited; large-scale and adversarial testing are absent.
- Writing Quality: ⭐⭐⭐⭐ System design is clearly described; code is open-sourced.
- Value: ⭐⭐⭐⭐ Offers direct practical value for secure enterprise-level LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] KL Penalty Control via Perturbation for Direct Preference Optimization](kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](../../ACL2026/dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[NeurIPS 2025\] Agentic Persona Control and Task State Tracking for Realistic User Simulation](agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2026\] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](../../ACL2026/dialogue/codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)

</div>

<!-- RELATED:END -->
