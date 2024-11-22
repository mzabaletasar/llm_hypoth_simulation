# Simulating Tabular Datasets through LLMs to Rapidly Explore Hypotheses about Real-World Entities

This repository contains **`main.py`**, a script designed to simulate tabular datasets using large language models (LLMs). It facilitates rapid exploration of hypotheses about real-world entities through two distinct modes of operation.

## Modes of Operation

1. **Hypothesis Mode (`hypothesis_mode`)**  
   - In this mode, users provide a hypothesis description and a list of entities. The script simulates data based on the given hypothesis.
   
2. **Feature Mode (`feature_mode`)**  
   - Here, users explicitly define the features to be simulated. The script generates data aligned with the provided feature definitions.

## Requirements

The script requires Python 3.8+ and the following Python packages:  
(*Replace the list below with actual dependencies. I recommend generating a `requirements.txt` file using `pip freeze` if needed.*)

- `openai`
- `pandas`
- `numpy`
- `python-dotenv`  
(*Add more as applicable.*)

### Installing Dependencies

To install the required dependencies, you can create a `requirements.txt` file and use:

```bash
pip install -r requirements.txt
```

## How to Use
1. Clone the repository:
```bash
git clone https://github.com/mzabaletasar/llm_hypoth_simulation.git
cd llm_hypoth_simulation
```
2. Set up your .env file with the necessary API key(s) for OpenAI, Replicate, or Groq. Example:

```makefile
OPENAI_API_KEY=your_openai_api_key
REPLICATE_API_KEY=your_replicate_api_key
GROQ_API_KEY=your_groq_api_key
```

3. Run the script via the command line using one of the two modes:

* Hypothesis Mode:
```bash
python main.py --run_mode hypothesis --hypothesis "Your hypothesis" --entities entity1,entity2,entity3
```

* Feature Mode:
```bash
python main.py --run_mode feature --hypothesis "Your hypothesis" --entities entity1,entity2,entity3
```


## Output
The script generates the following output files:

1. CSV File: Contains the simulated tabular dataset.
2. JSON File: Includes metadata for the features, such as:
  * feature_name
  * feature_description
  * feature_values

## Audience
This script is designed for anyone interested in quickly exploring hypotheses about real-world entities, including researchers, AI enthusiasts, and domain experts.
