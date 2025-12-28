# A4: Generating and Correcting Design

Design diagrams (UML class diagrams, sequence diagrams, state machines) are essential for software engineering but often missing or outdated. LLMs can help generate diagrams from text or correct existing diagrams.

Research Questions

- How accurate are LLMs at generating formal design diagrams from natural
  language?
- Can multi-agent pipelines improve diagram consistency?
- How effective are LLMs at detecting and correcting structural errors in diagrams?

Minimum requirements

- Design Artifacts: Choose 3–5 software components (e.g., small systems, class hierarchies, workflows) from:
  - past course assignments
  - open-source documentation
  - your own designed examples
- System Implementation. Include:
  - One single-agent baseline→ An LLM generates or corrects diagrams directly from text.
  - One multi-agent workflow with ≥2 roles
- Evaluation. Use at least one of the following evaluation methods:
  - structural accuracy against a small human-created gold standard
  - consistency checking between diagram elements (e.g., missing methods, invalid relations)
  - error-detection and correction effectiveness

## Prerequisites

Before running the project, ensure you have the following components set up:

### 1. LLM Backend (LM Studio)

The system is configured to use **LM Studio** as the LLM provider.

- Download and install [LM Studio](https://lmstudio.ai/).
- Load a model (e.g., `qwen2.5-coder-14b-instruct`).
- Start the Local Server on `http://localhost:1234`.

### 2. PlantUML Server

A local PlantUML server is required for syntax validation and diagram rendering.

- The easiest way is using Docker:
  ```bash
  docker run -d -p 8080:8080 plantuml/plantuml-server:jetty
  ```
- Alternatively, if you want to run the server locally, you can follow the official guide at `https://plantuml.com/starting`.

## Setup Instructions

1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- [langgraph.ipynb](langgraph.ipynb): The main Jupyter notebook containing the multi-agent workflow.
- [data/](data/):
  - [complete_shots.json](data/complete_shots.json): Few-shot examples for the LLM.
  - [test_exercises.json](data/test_exercises.json): Test cases with requirements and gold standard solutions.

## How to Run

1. Open [langgraph.ipynb](langgraph.ipynb) in VS Code or Jupyter Lab.
2. Ensure LM Studio and the PlantUML server are running.
3. Run the cells sequentially to:
   - Initialize the system components (LLM, PlantUML tool, Memory Manager).
   - Define the LangGraph workflow.
   - Execute a single test or batch evaluation.

## Evaluation

The system evaluates generated diagrams against a human-created gold standard using:

- **Structural Accuracy**: Precision, Recall, and F1 score for Classes, Attributes, and Relationships.
- **Consistency Checking**: Validating that all requirements are met without unnecessary additions.
