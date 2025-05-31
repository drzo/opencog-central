# CogPrime Architecture Diagram

This document provides a visual representation of the CogPrime architecture, as described in Dr. Ben Goertzel's paper, "CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence." The diagram illustrates the key components, their interactions, data flows, and the principle of cognitive synergy.

```mermaid
graph TD
    subgraph ExternalWorld["External World / Embodiment"]
        direction LR
        Perception["Perception<br>(e.g., DeSTIN, Sensory Input)"]
        Action["Action<br>(Motor Output, Effectors)"]
    end

    AtomSpace["<strong>AtomSpace (Central Hub)</strong><br/><i>Weighted, Labeled Hypergraph</i><br/>- Glocal Memory<br/>- Declarative Knowledge (TVs)<br/>- Procedural Knowledge (Programs/Links)<br/>- Episodic Traces (Simulations)<br/>- Attentional Info (AVs)<br/>- Goals & Schematics"]

    subgraph CognitiveProcesses ["Cognitive Processes (MindAgents)"]
        direction LR
        PLN["<strong>PLN (Declarative)</strong><br/>Probabilistic Logic Networks<br/><i>Reasoning, Inference</i>"]
        MOSES["<strong>MOSES (Procedural)</strong><br/>Evolutionary Program Learning<br/><i>Procedure Learning & Execution</i>"]
        ECAN["<strong>ECAN (Attentional)</strong><br/>Economic Attention Networks<br/><i>Resource (STI/LTI) Allocation</i>"]
        SimulationEngine["<strong>Simulation (Episodic)</strong><br/>Internal World Simulation<br/><i>Experience Replay/Prediction</i>"]
        GoalSystem["<strong>Goal System (Intentional)</strong><br/><i>Manages Goals (Atoms)</i><br/>Cognitive Schematics (C→P→G)"]
        PatternMining["<strong>Pattern Mining / Map Formation</strong><br/><i>(Cognitive Equation)</i><br/>Recognizes & Embodies Patterns"]
    end

    %% Core Data Flows
    Perception -- "Sensory Atoms" --> AtomSpace
    AtomSpace -- "Action Schemas / Procedures" --> MOSES
    MOSES -- "Motor Commands" --> Action
    Action -. "Environmental Feedback" .-> Perception

    %% AtomSpace as the Central Medium for all Cognitive Processes
    PLN <-->|Reads/Writes Declarative Knowledge (TVs)| AtomSpace
    MOSES <-->|Reads Context, Goal Info; Writes Procedures| AtomSpace
    ECAN <-->|Reads/Writes AttentionValues (AVs)| AtomSpace
    SimulationEngine <-->|Reads Scenarios; Writes Episodic Atoms| AtomSpace
    GoalSystem <-->|Reads Environment/Self State; Writes Goals/Subgoals| AtomSpace
    PatternMining <-->|Analyzes AtomSpace; Creates New Pattern Atoms| AtomSpace

    %% Goal System Driving Action & Reasoning
    GoalSystem -- "Requests Procedure for Goal" --> MOSES
    GoalSystem -- "Requests Reasoning for Goal Elaboration/Feasibility" --> PLN
    PLN -- "Provides Inferences to" --> GoalSystem
    ECAN -- "Allocates Attention to Goals/Atoms, Influencing" --> GoalSystem
    ECAN -- "Allocates Attention to Goals/Atoms, Influencing" --> PLN
    ECAN -- "Allocates Attention to Goals/Atoms, Influencing" --> MOSES
    ECAN -- "Allocates Attention to Goals/Atoms, Influencing" --> SimulationEngine
    ECAN -- "Allocates Attention to Goals/Atoms, Influencing" --> PatternMining


    %% Key Cognitive Synergy Interactions (as described in CogPrime paper Sec 8.8)
    %% These are often mediated via AtomSpace or direct calls
    PLN -. "Synergy: Inference for Learning" .-> MOSES
    MOSES -. "Synergy: Procedures for Reasoning" .-> PLN

    PLN -. "Synergy: Inference for Attention" .-> ECAN
    ECAN -. "Synergy: Attention for Inference" .-> PLN

    MOSES -. "Synergy: Learning for Attention" .-> ECAN
    ECAN -. "Synergy: Attention for Learning" .-> MOSES

    SimulationEngine -. "Synergy: Simulated Data for" .-> PLN
    SimulationEngine -. "Synergy: Simulated Data for" .-> MOSES

    PatternMining -. "Synergy: New Concepts/Patterns for" .-> PLN
    PatternMining -. "Synergy: New Concepts/Patterns for" .-> MOSES


    %% Styling
    classDef atomspace fill:#D1E8FF,stroke:#007bff,stroke-width:3px,color:black,font-weight:bold;
    classDef process fill:#E8D1FF,stroke:#8A2BE2,stroke-width:2px,color:black;
    classDef external fill:#D1FFD1,stroke:#28a745,stroke-width:2px,color:black;
    classDef goal fill:#FFFACD,stroke:#FFD700,stroke-width:2px,color:black;
    classDef pattern fill:#FFD1D1,stroke:#dc3545,stroke-width:2px,color:black;

    class AtomSpace atomspace;
    class PLN,MOSES,ECAN,SimulationEngine process;
    class GoalSystem goal;
    class Perception,Action external;
    class PatternMining pattern;

    linkStyle default interpolate basis
```

**Diagram Legend and Notes:**

*   **AtomSpace (Central Hub):** The core knowledge repository where all information (declarative, procedural, episodic, attentional, goals) is stored as a weighted, labeled hypergraph. It supports Glocal Memory. TruthValues (TVs) are associated with declarative knowledge, and AttentionValues (AVs) with attentional status.
*   **Cognitive Processes (MindAgents):** These are the primary algorithmic components that operate on and interact via the AtomSpace.
    *   **PLN (Probabilistic Logic Networks):** Handles declarative knowledge, performing uncertain reasoning and inference.
    *   **MOSES (Meta-Optimizing Semantic Evolutionary Search):** Learns and executes procedures (procedural knowledge).
    *   **ECAN (Economic Attention Networks):** Manages attentional memory by allocating resources (ShortTermImportance - STI, LongTermImportance - LTI) to Atoms, guiding the focus of other processes.
    *   **Simulation Engine:** Manages episodic memory, allowing the system to replay past experiences or simulate future scenarios.
    *   **Goal System:** Manages intentional memory, representing and processing goals, often structured as `Context -> Procedure -> Goal` (Cognitive Schematics).
    *   **Pattern Mining / Map Formation:** Implements the "Cognitive Equation" by recognizing large-scale patterns in the AtomSpace and embodying them as new, localized knowledge items (Atoms).
*   **External World / Embodiment:**
    *   **Perception:** Interface for sensory input (e.g., from systems like DeSTIN), which is converted into Atoms.
    *   **Action:** Interface for motor output, executing procedures learned/selected by the system.
*   **Arrows:**
    *   **Solid Arrows (`-->` or `<-->`):** Represent primary data flow or control paths. Bidirectional arrows indicate strong two-way interaction with AtomSpace.
    *   **Dotted Arrows (`-.->`):** Represent "Cognitive Synergy" – where processes dynamically support, inform, or call upon each other, often to overcome bottlenecks or enhance capabilities. This is a key principle of CogPrime.
*   **Cognitive Schematics (C→P→G):** Managed by the Goal System, these structures (Context → Procedure → Goal) guide the system's actions by linking situations, actions, and desired outcomes. The Goal System requests procedures from MOSES and reasoning from PLN based on these schematics.
*   **ECAN's Role:** ECAN pervasively influences all other cognitive processes by modulating the attention (STI/LTI) associated with Atoms in the AtomSpace, thereby guiding their processing priority and resource allocation.
