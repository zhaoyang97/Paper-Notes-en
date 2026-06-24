---
title: >-
  [Paper Note] Access Controls Will Solve the Dual-Use Dilemma
description: >-
  [ICML 2025][Dual-use dilemma] Proposes a conceptual framework based on access control to address the dual-use dilemma in AI safety. By obtaining real-world context through user verification and combining it with content classification, the framework achieves fine-grained permission management, simultaneously mitigating over-refusal and under-refusal.
tags:
  - "ICML 2025"
  - "Dual-use dilemma"
  - "access control"
  - "AI safety"
  - "over-refusal"
  - "gradient routing"
  - "technical AI governance"
date: 2026-05-08
content_hash: a5597fbef517e451
---

# Access Controls Will Solve the Dual-Use Dilemma

**Conference**: ICML 2025  
**arXiv**: [2505.09341](https://arxiv.org/abs/2505.09341)  
**Code**: None  
**Area**: Other  
**Keywords**: Dual-use dilemma, access control, AI safety, over-refusal, gradient routing, technical AI governance

## TL;DR

Proposes a conceptual framework based on access control to address the dual-use dilemma in AI safety. By obtaining real-world context through user verification and combining it with content classification, the framework achieves fine-grained permission management, simultaneously mitigating over-refusal and under-refusal.

## Background & Motivation

**Background**: Current LLM safety systems primarily rely on content analysis to determine if requests are harmful. Systems decide whether to refuse a response by analyzing the user input text and conversation history. Common methods include output monitoring, unlearning, and system prompts.

**Limitations of Prior Work**: A large number of requests fall into a "grey zone," where their harmfulness depends on the requester's identity and intent rather than the content itself. For example, the query "What features of viral surface proteins are recognized by human antibodies?" has entirely different implications when asked by a vaccine researcher versus a bioweapons developer. Pure content analysis cannot distinguish between these two cases.

**Key Challenge**: Safety systems face an irreconcilable binary choice—either refuse grey-zone requests (leading to over-refusal, harming legitimate users) or allow them (leading to under-refusal, enabling malicious users). This is not an issue that can be solved by improving jailbreak defenses, as the root of the problem is not adversarial attacks, but rather the lack of real-world context.

**Goal**
   - How to obtain reliable, hard-to-fabricate real-world context?
   - How to integrate context information into the safety decision-making workflow?
   - How to simultaneously address over-refusal and under-refusal?

**Key Insight**: The authors observe that the traditional computer security field has long solved similar problems through access control mechanisms. Operating systems do not block all file access; instead, they determine who can access what based on user identity and permissions. This design philosophy can be migrated to AI safety.

**Core Idea**: Leverage user verification (ID checks, institutional authentication, etc.) to obtain real-world context, and combine this with content classification to build a tiered access control framework, granting different levels of AI capability to users with different credentials.

## Method

### Overall Architecture

This framework is a conceptual safety architecture whose core idea is to transform the AI safety problem into a classic access control problem. The overall pipeline consists of three phases:

1. **Content Classification**: Maps model outputs to different content categories, each corresponding to a different sensitivity level.
2. **User Verification**: Obtains user real-world identity information through identity validation, institutional authentication, government licensing, etc.
3. **Access Decision**: Checks whether the user possesses the credentials required to access the detected content category, making an allow/deny decision.

The input consists of the user request and user credentials, and the output is the allow/deny decision. Unlike traditional pure content analysis, the decision considers both "what is said" and "who is asking."

### Key Designs

1. **Grey-Zone Detection**:

    - **Function**: Identifies which requests belong to the grey zone and require contextual judgment.
    - **Mechanism**: Classifies requests into three categories—clearly benign (directly permitted), clearly harmful (directly refused), and grey zone (enters the access control workflow). The determination of the grey zone is based on whether the content exhibits dual-use characteristics.
    - **Design Motivation**: Avoids requiring identity verification for all requests, thereby minimizing friction for normal usage.

2. **Verification-Based Context**:

    - **Function**: Obtains user real identity and status information through external verification mechanisms.
    - **Mechanism**: Utilizes ID checks, institutional affiliations, and government-issued credentials to acquire hard-to-fabricate contextual information. Unlike inferring context from conversation history, this information originates from independent third-party verification.
    - **Design Motivation**: Resolves the vulnerability in existing methods where context can be easily fabricated by adversaries. Opponents can fabricate conversation history, but they cannot easily forge institutional verifications.

3. **Content Category Classification via Gradient Routing**:

    - **Function**: Maps model outputs to predefined content categories to determine which sensitivity level the output belongs to.
    - **Mechanism**: A technical scheme based on UNDO (a robust unlearning method) and gradient routing. Gradient routing organizes knowledge modularly by routing different categories of knowledge to different subnetworks (sub-modules) of the model during training. During inference, the content category is determined by detecting which subnetwork is activated.
    - **Design Motivation**: Avoids the "incapable monitor" issue in traditional output monitoring. If a small model monitors a large model, a capability gap exists, and the large model may generate harmful content that the monitor cannot comprehend. Gradient routing implements classification directly within the model architecture, bypassing this issue.
    - **Novelty**: Traditional methods use external classifiers for content moderation, whereas gradient routing embeds classification capability directly into the model architecture.

4. **Tiered Access Policy**:

    - **Function**: Defines the mapping relationship between different content categories and user credentials.
    - **Mechanism**: Similar to an OS Access Control List (ACL), it defines the minimum credentials required for each content category. For example, biosecurity-related content might require researcher credentials in a relevant field, while cybersecurity tool-related content might require security practitioner credentials.
    - **Design Motivation**: Enables a more fine-grained safety policy than binary "all-deny" or "all-permit" schemes, allowing regulators to establish targeted policies.

### Analysis of Existing Safety Methods

The authors systematically analyze why three types of existing methods fail to resolve the dual-use dilemma:

| Method Category | Core Mechanism | Handling of Dual-Use | Fundamental Limitation |
|----------|----------|----------------|----------|
| Unlearning | Permanently deletes specific knowledge from the model | Completely non-contextual, cannot differentiate users | Treats all users identically, denying even legitimate researchers access to deleted knowledge |
| System Prompts | Guides model behavior via instructions | Can infer conversation context, but source is untrusted | Context relies on user input, which can be easily forged by adversaries |
| Output Monitoring | External model reviews output content | Analyzes content only, does not consider user identity | Suffers from capability gap and lacks real-world context |

### How the Framework Resolves Over-Refusal and Under-Refusal

- **Addressing Over-Refusal**: Verified legitimate users (e.g., researchers with verified credentials in relevant fields) can access grey-zone content and are no longer refused outright.
- **Addressing Under-Refusal**: Unverified users cannot access sensitive content categories. Even if they break down harmful requests into seemingly benign sub-queries through decomposition attacks, the system still demands verification based on the content category classification.

## Key Experimental Results

This is a conceptual framework paper and does not contain traditional empirical experiments. However, the authors support their perspective through analytical comparisons and feasibility arguments.

### Comparative Analysis of Safety Methods

| Method | Context-Aware? | Context Reliability | Resolves Over-Refusal? | Resolves Under-Refusal? | Requires External Verification? |
|------|---------------|-----------------|----------------------|----------------------|-----------------|
| Unlearning | ✗ | N/A | ✗ | ✗ | ✗ |
| System Prompts | Partial | Low (User-fabricated) | Partial | ✗ | ✗ |
| Output Monitoring | ✗ | N/A | ✗ | ✗ | ✗ |
| **Ours** | **✓** | **High (Third-party verification)** | **✓** | **✓** | **✓** |

### Comparison of Content Classification Technical Routes

| Technical Route | Implementation | Robustness | Capability Gap Issue | Modularity Level |
|----------|----------|--------|-------------|-----------|
| External Classifier | Independent model reviews output | Low | Exists | Low |
| Moderation API | Calls safety auditing interfaces | Medium | Exists | Low |
| Gradient Routing | Routes to subnetworks during training | High | Does not exist | High |
| UNDO + Routing | Distillation robustification + Gradient routing | High | Does not exist | High |

### Key Findings

- The dual-use dilemma is not a jailbreak defense problem, but an information omission problem—safety systems lack real-world context.
- Decomposition attacks essentially exploit under-refusal by breaking down clearly harmful queries into a sequence of grey-zone sub-queries.
- Gradient routing avoids the capability gap by nesting classification within the model, offering a core advantage over external monitoring.
- This framework is orthogonal to and can complement existing jailbreak defense methods.

## Highlights & Insights

- **Analogizing AI safety to OS access control**: A highly natural but previously overlooked analogy. Operating systems do not prohibit all file operations; they manage permissions via ACLs. Similarly, AI safety should not flatly refuse all sensitive topics, but rather grant tiered access based on user credentials. This design philosophy is simple yet powerful.
- **Distinguishing "content-dimensional" and "user-dimensional" safety**: Traditional AI safety research almost exclusively focuses on content (what should and shouldn't be said). This paper points out that the user dimension (who is asking) is equally critical, providing a new conceptual framework for the domain.
- **Transferable concept of gradient routing for content classification**: Routing knowledge modularly to different subnetworks and using activation pattern detection at inference to determine content categories represents a technical approach transferable to other scenarios requiring fine-grained content control (e.g., personalized content filtering, domain-specific knowledge gating).

## Limitations & Future Work

- **Purely conceptual framework without empirical validation**: The paper does not implement any prototype system, leaving all discussions at a theoretical level. There is a lack of quantitative data supporting key metrics such as content classification accuracy, the practical feasibility of user verification, and system latency overhead.
- **Privacy and surveillance risks**: Requiring users to provide verification information (IDs, institutional credentials, etc.) introduces significant privacy risks. The paper lacks sufficient discussion on this aspect—anonymous AI usage is an important right in certain countries and scenarios.
- **Subjectivity in content category definitions**: The framework assumes that "content categories" and their corresponding "required credentials" can be clearly defined, but in reality, this mapping is highly complex and culturally dependent. Who defines these categories, and what are the standards?
- **Technical challenges in gradient routing implementation**: Although the paper proposes gradient-routing-based content classification, its feasibility has yet to be verified in large-scale LLMs (e.g., GPT-4 level).
- **Adversarial adaptability**: The framework assumes credentials are hard to fabricate, but in practice, credential theft, spoofing, and other attack vectors remain viable threats.
- **Potential to widen the digital divide**: Granting more powerful AI capabilities to credentialed/certified users might further widen the gap between those with resources and those without.

## Related Work & Insights

- **vs. Traditional RLHF/DPO Alignment**: Methods like RLHF train models to learn the global boundary of "what should and shouldn't be said." This paper suggests that boundaries should vary by user, which is a more flexible paradigm.
- **vs. Jailbreak Defense Research**: Mainstream AI safety studies focus on defending against jailbreak attacks (e.g., GCG, PAIR), but this paper shows that the dual-use dilemma is orthogonal to jailbreaks. Even with perfect jailbreak defenses, over-refusal and under-refusal persist.
- **vs. Original Gradient Routing Work**: Cloud et al. proposed gradient routing for model modularization; this paper extends it to safe content classification, offering an interesting application direction.
- **vs. UNDO (Robust Unlearning)**: Lee et al.'s UNDO method improves unlearning robustness through distillation. This paper combines it with gradient routing to solve robustness issues in content classification.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing access control to AI safety is a natural yet neglected perspective; the formal analysis of the dual-use dilemma is insightful.
- Experimental Thoroughness: ⭐⭐ A purely conceptual paper lacking empirical validation, leaving all arguments at a theoretical level.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation, complete logical chain, and the virology example of the dual-use dilemma is highly intuitive.
- Value: ⭐⭐⭐⭐ Provides a new thinking dimension for the AI safety field, but lacks a clear path to actual deployment, serving more as a position paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do not Abstain! Identify and Solve the Uncertainty](../../ACL2025/others/do_not_abstain_identify_and_solve_the_uncertainty.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](../../ACL2025/others/using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ICML 2025\] Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena](position_solve_layerwise_linear_models_first_to_understand_neural_dynamical_phen.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](../../ACL2025/others/proxann_topic_model_eval.md)
- [\[ICML 2025\] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)

</div>

<!-- RELATED:END -->
