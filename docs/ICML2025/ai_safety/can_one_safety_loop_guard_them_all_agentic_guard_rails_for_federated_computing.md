---
title: >-
  [Paper Note] Can One Safety Loop Guard Them All? Agentic Guard Rails for Federated Computing
description: >-
  [ICML 2025][AI Safety][federated computing] Proposes Guardian-FC—the first backend-agnostic unified security framework for federated computing. By employing a finite-state safety loop (Sense→Predict→Act→Prove) on an Agentic-AI control plane, Guardian-FC uniformly regulates heterogeneous privacy mechanisms such as FHE, DP, and MPC, achieving consistent execution of a single set of guard-rail policies across all privacy backends.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "federated computing"
  - "guard rails"
  - "privacy-preserving"
  - "FHE"
  - "DP"
  - "MPC"
  - "finite-state machine"
date: 2026-05-08
content_hash: 60a5d5c80be03606
---

# Can One Safety Loop Guard Them All? Agentic Guard Rails for Federated Computing

**Conference**: ICML 2025  
**Authors**: Narasimha Raghavan Veeraragavan, Jan Franz Nygård  
**arXiv**: [2506.20000](https://arxiv.org/abs/2506.20000)  
**Code**: Not released  
**Area**: AI Safety, Federated Computing, Privacy Preservation  
**Keywords**: federated computing, guard rails, privacy-preserving, FHE, DP, MPC, finite-state machine  

## TL;DR

Proposes Guardian-FC—the first backend-agnostic unified security framework for federated computing. By employing a finite-state safety loop (Sense→Predict→Act→Prove) on an Agentic-AI control plane, Guardian-FC uniformly regulates heterogeneous privacy mechanisms such as FHE, DP, and MPC, achieving consistent execution of a single set of guard-rail policies across all privacy backends.

## Background & Motivation

**Background**: Federated computing allows multiple institutions to collaboratively train and analyze models without sharing raw data, serving as a core paradigm for privacy preservation. Commonly used privacy-preserving technologies include Fully Homomorphic Encryption (FHE, tracking noise budget), Differential Privacy (DP, tracking $\varepsilon$ budget), and Secure Multi-Party Computation (MPC, tracking share integrity), each having its own independent safety metrics and checking mechanisms.

**Limitations of Prior Work**: Currently, each privacy backend requires customized safety monitoring, control logic, and auditing tools, leading to: (1) **Fragmentation**—development teams must build and maintain separate safety frameworks for each backend, resulting in significant duplication of effort; (2) **Poor Interoperability**—workflows involving hybrid privacy backends (such as federated Kaplan-Meier survival curve computations using FHE→MPC→DP) lack a unified safety framework to coordinate safety policies across stages; (3) **Consistency Risks**—inconsistent enforcement of safety policies in heterogeneous systems can result in privacy leaks or system failures.

**Key Challenge**: The diversity of privacy mechanisms demands flexibility, whereas safety enforcement requires consistency. The challenge lies in supporting new privacy backends without modifying the core safety logic.

**Key Insight**: Decoupling safety guard-rails from privacy mechanisms by inserting a backend-agnostic Domain-Specific Language (DSL) between them, allowing the same safety loop logic to run across all privacy backends.

**Core Idea**: Abstracting privacy operations with a DSL and executing a unified guard-rail via an Agentic-AI safety loop to achieve the vision of "one safety loop protecting all privacy backends."

## Method

### Overall Architecture

Guardian-FC adopts a two-layer architecture: the upper layer is the **Agentic-AI Control Plane** (which only processes signed metadata and does not touch raw data), and the lower layer is the **Federated Data Plane** (where nodes and the central aggregator execute privacy-preserving computations). The two layers interact via authenticated channels, transmitting signed telemetry frames upstream and signed control commands downstream. The federated computing logic is written as a plug-in using a backend-agnostic DSL, which is dynamically bound to interchangeable Execution Providers (EPs) at runtime.

### Key Designs

1. **Backend-Agnostic Domain-Specific Language (DSL) and Execution Providers (EP)**:

    - **Function**: Decoupling computational logic from privacy implementations.
    - **Mechanism**: Data scientists write plug-ins (modular computational units) using the DSL to define "what to do" without involving the details of privacy mechanisms. EPs implement the concrete semantics of DSL operations—FHE-EP performs encrypted computation, DP-EP injects noise, and MPC-EP handles secret sharing. The same plug-in code can run on different privacy backends simply by switching the EP without any modification. At compile time, a manifest (a structured JSON description file) is generated to specify the plug-in name, DSL operations, selected EPs, and enabled guard-rail predicates.
    - **Design Motivation**: Backend agnosticism is a prerequisite for achieving a unified safety loop—only when the computation description is decoupled from the privacy implementation can the safety logic remain indifferent to low-level technical details.

2. **Agentic-AI Finite-State Safety Loop**:

    - **Function**: Continuously monitoring and executing safety policies at a fixed frequency (e.g., 1Hz).
    - **Mechanism**: The safety loop consists of four stages executed periodically:
        - **Sense**: Telemetry collectors gather signed metric frames (noise budget, privacy loss, latency, etc.) from all nodes and aggregators.
        - **Predict**: Sentinels align the telemetry streams and evaluate guard-rail predicates (boolean functions $p_i: \{\mathbf{m}, \hat{\mathbf{m}}\} \to \{true, false\}$) on current and predicted metrics to forecast impending safety violations.
        - **Act**: The control engine issues signed A-commands (such as A-BOOTSTRAP to reset the noise budget, A-ABORT_JOB to terminate the job, and A-ISOLATE_PARTY to isolate faulty nodes) through a cryptographic orchestrator.
        - **Prove**: The audit engine appends all commands and acknowledgments to an append-only Merkle ledger, providing tamper-proof evidence for compliance auditing.
    - **Design Motivation**: A finite-state safety loop serves as the foundation for formal verification—the state space is sufficiently small to allow for model checking, ensuring that safety invariants can be proved.

3. **Manifest-Driven Admission Verification**:

    - **Function**: Fast-failing and rejecting incompatible configurations before job execution.
    - **Mechanism**: When a job is submitted with its manifest, the control engine performs a fail-fast admission check: it verifies that each DSL opcode in the plug-in is implemented by the selected EP, and ensures that all metric keys referenced by the enabled guard-rail predicates are provided by the EP. Once verified, the manifest is broadcast to all nodes, and the telemetry collectors reject any telemetry frame that diverges from the manifest's specifications.
    - **Design Motivation**: Pre-verification prevents resource waste and runtime safety violations; the manifest acts as an authoritative contract to ensure that all components share a consistent configuration.

### Loss & Training

This work focuses on system framework design and does not involve model training.

## Experiments

### Main Results: Three Qualitative Safety Scenarios

| Scenario | Privacy Backend | Trigger Predicate | Safety Action | Outcome |
|------|---------|---------|---------|------|
| A: CKKS Noise Depletion | FHE | $p_1$: noiseBits < $\theta_{fhe}$ | A-BOOTSTRAP (Reset noise budget) | Local remediation, computation continues |
| B: DP Budget Overflow | DP | $p_2$: $\varepsilon_{spent}$ > $\varepsilon_{max}$ | A-ABORT_JOB (Abort job) | Securely clear seeds and shares |
| C: Malicious MPC Share | MPC | $p_3$: shareAuthFail ≥ 2 | A-ISOLATE_PARTY (Isolate faulty party) | Remaining shares merged to complete execution |

### Formal Safety Property Verification

| Property | Formal Formulation | Meaning |
|------|-----------|------|
| Safety | Aggregator=FINALIZE ⟹ (∀i: Node[i]∈$S_{ok}$) ∧ (∀p∈P: ¬p) | All nodes are safe and no predicates are violated when the job is completed |
| Liveness | ◇(μ=0), ranking function μ is monotonically decreasing | The system will not remain stuck in intermediate states indefinitely, ensuring eventuality |

### Key Findings

- The core logic of the safety loop (Sense→Predict→Act→Prove), component boundaries, and auditing semantics remain completely unchanged across all three scenarios—only the triggered predicates and corresponding A-commands differ.
- Manifest-driven admission verification enables compile-time safety checks, shifting incompatibility detection upstream prior to execution.
- The state space of the finite-state machine is deliberately restricted to a small scale, creating feasible conditions for automated model checking.
- Extending support to a new privacy backend only requires implementing a new EP module, without any modification to the control plane or the guard-rail logic.

## Highlights & Insights

- Proposes the first unified safety framework across FHE/DP/MPC, addressing the fundamental issue of fragmented safety enforcement in federated computing.
- Two-layer architecture design: the control plane only processes signed metadata and never accesses raw data, inherently satisfying privacy constraints.
- The decoupled design of DSL + EP provides both flexibility and consistency, paving a clean extensibility path for integrating new privacy technologies (e.g., functional encryption, trusted execution environments).
- Formal safety properties (Safety + Liveness) lay a solid theoretical foundation for future model-checking verifications.

## Limitations & Future Work

- Currently validated only through qualitative scenario demonstrations rather than quantitative experiments, lacking performance evaluation data of a prototype system.
- Whether a 1Hz telemetry frequency is sufficient to capture all safety events remains unverified; high-throughput scenarios may require higher frequencies.
- The expressiveness and concrete specification of the DSL have not yet been fully defined, remaining in a conceptual phase.
- Assumes that all nodes honestly transmit signed telemetry; the issue of malicious nodes forging telemetry is left unaddressed.
- Thresholds for guard-rail predicates (e.g., $\theta_{fhe}$, $\varepsilon_{max}$) are fixed as static values, lacking adaptive adjustment strategies.
- No runtime overhead estimation or comparison with existing safety frameworks is provided.

## Related Work & Insights

- **Federated Learning Safety**: Most existing works focus on safety within a single privacy backend (e.g., privacy accounting for DP, noise management for FHE). Guardian-FC attempts to unify management at a higher level of abstraction.
- **Agentic AI Safety**: Applying the "sense-make-act" loop of AI agents to system safety monitoring is a novel direction.
- **Insights**: The DSL design of the framework could influence the standardization trajectory of federated computing. If the community can unify DSL specifications, it would significantly lower development costs for federated computing systems.
- **Open Research Directions**: The paper proposes valuable future directions such as RL-based adaptive threshold tuning, typed (ε,δ,λ)-calculus for multi-EP composition, and human-in-the-loop overrides.

## Rating

| Dimension | Rating | Reason |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Proposes the first backend-agnostic unified safety framework for federated computing |
| Technical Depth | ⭐⭐⭐ | Formal model is grounded but has no implementation validation, and the DSL specification is incomplete |
| Experimental Thoroughness | ⭐⭐ | Only qualitative scenario demonstrations, without quantitative performance evaluations |
| Writing Quality | ⭐⭐⭐⭐ | Clear architectural description and standardized formal formulations |
| Value | ⭐⭐⭐ | Conceptually forward-looking but far from practical deployment, requiring prototype validation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](../../ICML2026/ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ICML 2026\] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](../../ICML2026/ai_safety/omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)
- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)
- [\[ICML 2025\] Generalization in Federated Learning: A Conditional Mutual Information Framework](generalization_in_federated_learning_a_conditional_mutual_information_framework.md)
- [\[ICML 2025\] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)

</div>

<!-- RELATED:END -->
