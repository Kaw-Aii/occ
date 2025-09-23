# OpenCog Collection Architecture

The OpenCog Collection (OCC) is designed as a modular and extensible architecture for building and experimenting with Artificial General Intelligence (AGI) systems. The core of the OCC is the **AtomSpace**, a hypergraph database that serves as a flexible and powerful knowledge representation.

## Core Components

-   **AtomSpace**: The central knowledge store of the OCC. It is a weighted, labeled hypergraph that can represent complex relationships between concepts.
-   **Scheme**: The primary language for interacting with the AtomSpace and implementing cognitive algorithms.
-   **Cognitive Processes**: A collection of AI algorithms that operate on the AtomSpace, including:
    -   **Pattern Miner**: Discovers patterns and relationships in the AtomSpace.
    -   **Unsupervised Language Learning**: Learns the structure of language from raw text.
    -   **Probabilistic Logic Networks (PLN)**: A reasoning system that can handle uncertainty.

## Cognitive Synergy

The OCC is designed to foster **cognitive synergy**, where the interaction of different cognitive processes leads to emergent intelligence. This is achieved through:

-   **A shared knowledge representation**: The AtomSpace provides a common language for all cognitive processes to communicate and share information.
-   **A flexible and extensible architecture**: The OCC can be easily extended with new cognitive processes and integrations.
-   **A focus on interaction and collaboration**: The OCC is designed to encourage the interaction and collaboration of different cognitive processes.

## Guix and the Devcontainer

The OCC uses **GNU Guix** to create a reproducible and isolated development environment. The devcontainer provides a complete development environment with all the necessary dependencies, so you can start building and experimenting with the OCC right away.

