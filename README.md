# OpenCog Central: Towards the CogPrime Vision for AGI

Welcome to `opencog-central`. This repository serves as a central point of information and a conceptual guide for understanding the **OpenCog** project and, specifically, the **CogPrime** architecture for Artificial General Intelligence (AGI).

## Introduction to OpenCog and CogPrime

**OpenCog** is an open-source software framework and a collaborative research project aimed at developing Artificial General Intelligence. It provides a diverse set of tools, libraries, and cognitive algorithms designed to support the creation of sophisticated AI systems. At its core, OpenCog features the **AtomSpace**, a weighted, labeled hypergraph database that serves as a flexible knowledge representation store.

**CogPrime** is a comprehensive AGI architecture designed by Dr. Ben Goertzel. It outlines a theoretical and practical blueprint for achieving human-level (and potentially beyond) general intelligence. The fundamental idea is that:

`OpenCogPrime = OpenCog + CogPrime`

This means CogPrime is a *design* intended to be implemented *within* the OpenCog framework, leveraging its existing components and extending them to realize a fully integrated cognitive system. The ultimate implementation of CogPrime within OpenCog is referred to as **OpenCogPrime**.

The CogPrime architecture is detailed in the paper: "[CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence](docs/CogPrime_Integrative_Architecture_AGI.md)". (Context: `CogPrime - An Integrative Architecture for Embodied Artificial General Intelligence.md`)

## The Essence of CogPrime: Cognitive Synergy

A core hypothesis underlying the CogPrime approach is **cognitive synergy**. This refers to the idea that robust, human-level intelligence can emerge from the rich, dynamic, and cooperative interaction of multiple symbolic and subsymbolic learning and memory components, all integrated within an appropriate cognitive architecture and environment. (Context: `CogPrime - An Integrative Architecture for Embodied Artificial General Intelligence.md`, Section 1.2)

CogPrime emphasizes that simply having different AI algorithms is not enough; their careful integration and interoperation are crucial for achieving the emergent structures and dynamics characteristic of general intelligence.

## Key Architectural Components of CogPrime

The CogPrime design specifies several key types of memory and associated cognitive processes:

(Context: `CogPrime - An Integrative Architecture for Embodied Artificial General Intelligence.md`, Sections 1.4, 5, 6, 7 and Table 1)

1.  **Knowledge Representation:**
    *   **AtomSpace:** Utilizes OpenCog's AtomSpace for storing various forms of knowledge as weighted, labeled hypergraphs.
    *   **Glocal Memory:** A hybrid approach where knowledge is stored both locally (explicitly) and globally (as distributed patterns or "maps"), allowing for robust and flexible memory retrieval and reconstruction.

2.  **Memory Types & Cognitive Processes:**
    *   **Declarative Memory:**
        *   Handled by **Probabilistic Logic Networks (PLN)**, an uncertain inference framework integrating fuzzy logic and imprecise probabilities.
    *   **Procedural Memory:**
        *   Handled by algorithms like **MOSES (Meta-Optimizing Semantic Evolutionary Search)**, a probabilistic evolutionary program learning algorithm.
    *   **Episodic Memory:**
        *   Managed by an internal **Simulation Engine**, allowing the agent to replay and learn from past experiences or simulate future scenarios.
    *   **Attentional Memory:**
        *   Regulated by **Economic Attention Networks (ECAN)**, which allocate system resources (ShortTermImportance and LongTermImportance) based on artificial economics principles.
    *   **Intentional Memory (Goals):**
        *   Represented declaratively (e.g., within PLN) and managed economically (via ECAN), often structured in a goal hierarchy.
    *   **Sensory Memory:**
        *   Intended to be handled by systems like **DeSTIN (Deep Spatio-Temporal Inference Network)** or similar hierarchical temporal memory systems for processing low-level sensorimotor data.

3.  **Goal-Oriented Dynamics:**
    *   Driven by "cognitive schematics" of the form: `Context -> Procedure -> Goal`, which represent the system's understanding of how actions lead to outcomes in specific situations.

4.  **Embodiment and Development:**
    *   CogPrime emphasizes the importance of embodiment (virtual or physical) and a developmental approach, where the AGI system learns and grows through experience, potentially mirroring human child development stages.

## Project Structure (Conceptual for CogPrime within `opencog-central`)

While `opencog-central` currently serves as an organizational profile, a repository aiming to implement or coordinate the CogPrime vision might adopt a structure like this:

```
.
├── docs/
│   └── CogPrime_Integrative_Architecture_AGI.md  # The core CogPrime paper
│   └── (other design documents, whitepapers)
├── src/                                          # Conceptual location for CogPrime implementation
│   ├── atomspace_integration/                    # Interfacing with and extending AtomSpace
│   ├── memory_systems/                           # Implementations for declarative, procedural, etc.
│   │   ├── declarative/ (PLN logic and truth values)
│   │   ├── procedural/ (MOSES, program execution)
│   │   ├── episodic/ (Simulation engine interface)
│   │   └── ...
│   ├── cognitive_processes/                      # Core algorithms (PLN, MOSES, ECAN)
│   │   ├── pln/
│   │   ├── moses/
│   │   └── ecan/
│   ├── agent_core/                               # Main agent loop, goal management, cognitive cycle
│   └── perception_action/                        # Interface to sensory (e.g., DeSTIN) and motor systems
├── examples/                                     # Sample agent configurations and scenarios
│   └── virtual_agent_minecraft_like/
│   └── robot_preschool/
├── tests/                                        # Unit, integration, and scenario-based tests
└── README.md                                     # This file
```

## Current Status & Future Vision

`opencog-central` currently acts as a high-level pointer to the broader OpenCog ecosystem. Many components foundational to CogPrime exist within various OpenCog repositories:

*   **Active and Stable:**
    *   [OpenCog AtomSpace](https://github.com/opencog/atomspace): The core hypergraph database.
    *   [Link Grammar](https://github.com/opencog/link-grammar): For natural language parsing.
*   **Research & Development:**
    *   [Learn](https://github.com/opencog/learn): Symbolic learning.
    *   [Agents](https://github.com/opencog/agents): Refactoring learning for interactive environments.
*   **Historical Components ("Fossils"):**
    *   As noted in the [OpenCog Project Profile](https://github.com/opencog/.github/blob/main/profile/README.md), earlier implementations of some systems envisioned by CogPrime (like specific versions of PLN, MOSES, Attention mechanisms) are now considered "fossils" and are not actively maintained in their original forms. (Context: `opencog-central` `profile/README.md`)
*   **Hyperon:**
    *   A newer initiative, OpenCog Hyperon, is being developed by [SingularityNET](https://singularitynet.io) and represents a next-generation approach, evolving from the original OpenCog/CogPrime ideas.

The vision for `opencog-central` is to serve as a beacon for the CogPrime architecture, potentially guiding future efforts to integrate existing OpenCog tools and develop new components in a way that holistically realizes the CogPrime design.

## Getting Started & Contributing

To get involved or learn more:

1.  **Study the CogPrime Architecture:** Read the "[CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence](docs/CogPrime_Integrative_Architecture_AGI.md)" paper thoroughly.
2.  **Explore the OpenCog Wiki:** The [OpenCog Wiki](https://wiki.opencog.org/w/The_Open_Cognition_Project) is a primary resource for documentation on various OpenCog components and concepts.
3.  **Investigate Key Repositories:**
    *   [opencog/atomspace](https://github.com/opencog/atomspace)
    *   [opencog/learn](https://github.com/opencog/learn)
    *   [opencog/agents](https://github.com/opencog/agents)
    *   [opencog/link-grammar](https://github.com/opencog/link-grammar)
    *   Explore other repositories listed on the [OpenCog GitHub organization page](https://github.com/opencog).
4.  **Follow Hyperon Development:** Check out [SingularityNET](https://singularitynet.io) for information on OpenCog Hyperon.
5.  **Join the Community:** Look for OpenCog mailing lists, forums, or other community channels to engage in discussions.
6.  **Contribute:** Contributions can range from theoretical discussions, documentation improvements, to code contributions to the various active OpenCog projects.

## Resources

*   **OpenCog Wiki:** [https://wiki.opencog.org/w/The_Open_Cognition_Project](https://wiki.opencog.org/w/The_Open_Cognition_Project)
*   **CogPrime Architecture Paper:** "[CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence](docs/CogPrime_Integrative_Architecture_AGI.md)" (Context: `CogPrime - An Integrative Architecture for Embodied Artificial General Intelligence.md`)
*   **OpenCog GitHub Organization:** [https://github.com/opencog](https://github.com/opencog)
*   **SingularityNET (for Hyperon):** [https://singularitynet.io](https://singularitynet.io)
*   **Original OpenCog Project Overview:** [profile/README.md](profile/README.md) (Context: `opencog-central` `profile/README.md`)

This README aims to bridge the comprehensive vision of CogPrime with the existing OpenCog landscape, providing a clearer path for those interested in understanding and contributing to this ambitious AGI endeavor.
