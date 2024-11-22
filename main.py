import argparse
import json
import os

import pandas as pd
from groq import Groq
from openai import OpenAI
from tqdm import tqdm

# Define the clients and model choices
CLIENTS = {"gpt": OpenAI(), "gemma": Groq(), "llama3": Groq()}
MODEL_CHOICES = ["gpt-4o-mini", "gpt-3.5-turbo-1106", "gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192"]


def completions(**kwargs):
    global MODEL_NAME
    client = CLIENTS[f"{MODEL_NAME.split('-')[0]}"]
    chat_completion = client.chat.completions.create(model=MODEL_NAME, temperature=0.5, seed=42, **kwargs)
    return chat_completion.choices[0].message.content.strip().lower()


def prompts_for_features_llm(hypothesis_description, num_features):
    # Main system prompt for the LLM
    sys_prompt = f"""You are an assistant to a scientist exploring hypotheses using a Large Language Model. Your task is to generate specific, measurable, and fact-checkable features related to a given hypothesis. All features must be based on measurable data, avoid speculation, and focus on a single factor at a time. Ensure that one primary response variable is provided, with all other features explaining this variable. Limit the total number of features to {num_features}, with {num_features - 1} independent features that can explain the response variable. Respond in a JSON format with the keys 'sys_prompt' and 'user_prompt'."""

    # User prompt example
    one_shot_user_prompt = """I am interested in what makes Nobel-winning scientists more successful. Are there factors that determine scientists achieving more awards in their lives?
    I'm particularly interested in factors about their childhood and upbringing, such as psychological or socio-cultural factors."""

    # Example response format for the assistant with dynamic feature count and JSON response format.
    one_shot_assistant_prompt = f"""{{'sys_prompt': 'You are an expert in developmental psychology, scientific research, and sociology. Your role is to generate a list of specific, measurable, and fact-checkable features related to childhood factors influencing scientists' success. Avoid speculative or general factors and focus on verifiable data from biographies, interviews, or demographic studies. Each feature must measure one thing only, and the total number of features should be limited to {num_features}, with {num_features - 1} independent features explaining one response variable. Respond in JSON format with each feature listed using 'feature_name', 'feature_description', and 'possible_values' as keys.',
    'user_prompt': 'For the hypothesis: "What personal or environmental factors in childhood are most common among scientists who made significant breakthroughs?" provide a list of key features to explore. Each feature must be specific, measurable, and verifiable using data such as biographies or public records. Ensure there are {num_features - 1} independent features that explain one response variable related to scientific success.'}}"""

    # Define messages for the LLM
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": one_shot_user_prompt},
        {"role": "assistant", "content": one_shot_assistant_prompt},
        {"role": "user", "content": hypothesis_description},
    ]

    # Invoke the LLM with the constructed messages and specify the JSON response format.
    response = completions(messages=messages, response_format={"type": "json_object"})
    return json.loads(response)


def extract_features(features_text):
    sys_prompt = """You are an expert in feature and response variable extraction. Your task is to extract specific features and a response variable from a structured text. For each variable, extract the name, description, and any measurable range or possible values (if available). Respond in JSON format, where each variable is a key, and the value is a list with two elements: 
    1) A short description of what the variable measures (from the 'description' field in the text) 
    2) The possible values or ranges for the variable (from the 'range' or 'measurement' fields, if applicable).
    
    If no specific range or values are mentioned, leave the second list element empty. The format should be:
    {variable_name: [Short Description of what it measures, Possible values or ranges (if applicable)]}."""

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": features_text}]
    extraction = completions(messages=messages, response_format={"type": "json_object"})
    return json.loads(extraction)


def simulate_response_variable(response_variable_info, simulations):
    response_variable = list(response_variable_info.keys())[0]
    print(f"{response_variable}")
    simulations[response_variable] = []
    for entity in tqdm(simulations["entities"]):
        extracted_value = get_corrected_simulation(response_variable_info, entity)
        simulations[response_variable].append(extracted_value)
    return simulations


def get_corrected_simulation(feature_info, entity):
    feature_name = list(feature_info.keys())[0]
    description, possible_values = list(feature_info.values())[0]
    # Step 1: Generate the simulated value
    sys_prompt = (
        f"You are tasked with predicting a feature for an entity based on the description and possible values."
        f"Respond only the estimated value for the feature '{feature_name}' for the entity '{entity}', nothing else."
    )
    user_prompt = f"What is the value of '{feature_name}' for the entity '{entity}'? Description: {description}. Possible values: {possible_values}. Entity: {entity}, {feature_name}: "
    sim_value = completions(messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])

    # Step 2: Assess the simulated value and get reasoning + conclusion
    sys_prompt = """You are an expert evaluator tasked with assessing the accuracy of a value for a specific feature. Based on the provided information, analyze and compare the proposed value with relevant data, providing logical reasoning in your response. Your output should be in JSON format with two keys: "reasoning" and "conclusion".

    - In "reasoning", explain whether the provided value is accurate based on the evidence, and if not, suggest the correct value.
    - In "conclusion", provide only the value that you believe to be correct. If the provided value is correct, repeat it as the answer. If it is incorrect, provide your suggested corrected value as the answer. Avoid any other text or symbols in the "conclusion" field, using only the value (e.g., 0.75, high, or 1965)."""
    q = f"Is the {feature_name} of {entity} accurately described as {sim_value}? Consider the description: '{description}' and possible values: '{possible_values}'"

    # Generate the assessment and reasoning
    correction = completions(
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}],
        response_format={"type": "json_object"},
    )

    # Step 3: Extract corrected simulation (concise value extraction)
    # correction = extract_corrected_simulation(correction_text)
    correction = json.loads(correction)["conclusion"]
    return correction


# The function for running the experiment
def run_experiment(data_output_path, features_output_path, entities, hypothesis_description, num_features, features=None):
    os.makedirs(os.path.dirname(data_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_output_path), exist_ok=True)

    if hypothesis_description is None:
        if features is None:
            raise ValueError("You must provide features if hypothesis_description is not provided.")
    else:
        prompts = prompts_for_features_llm(hypothesis_description, num_features)
        features_text = completions(
            messages=[{"role": "system", "content": prompts["sys_prompt"]}, {"role": "user", "content": prompts["user_prompt"]}],
            response_format={"type": "json_object"} if "json" in prompts["sys_prompt"] + prompts["user_prompt"] else None,
        )
        features = extract_features(features_text)

    print("---FEATURES---")
    print(json.dumps(features, indent=2))
    with open(features_output_path, "w") as json_file:
        json.dump(features, json_file, indent=2)

    response_variable = completions(
        messages=[
            {
                "role": "user",
                "content": f"Given the hypothesis description '{hypothesis_description}' and the features '{list(features.keys())}', which feature is the most appropriate response_variable? Respond only with the name of the feature",
            }
        ]
    )
    response_variable_info = {response_variable: features[response_variable]}
    features_info = [{k: features[k]} for k in features if k != response_variable]
    simulations = {"entities": entities}

    for feature_info in features_info:
        corrected_simulations_cached = []
        for entity in tqdm(entities):
            correction = get_corrected_simulation(feature_info, entity)
            corrected_simulations_cached.append(correction)

        simulations[list(feature_info.keys())[0]] = corrected_simulations_cached

    simulations = simulate_response_variable(response_variable_info, simulations)
    simulated_df = pd.DataFrame.from_dict(simulations)
    simulated_df.to_csv(data_output_path, index=False)


# Argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate simulations based on a hypothesis and list of entities.")

    # Model name argument
    parser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES[0],
        help=f"Specify the model to use (default: {MODEL_CHOICES[0]})",
    )

    # Run mode argument
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["hypothesis", "features"],
        default="hypothesis",
        help="Specify the run mode: 'hypothesis' to generate features, 'features' to use existing features",
    )

    # Hypothesis description
    parser.add_argument(
        "--hypothesis_description",
        type=str,
        help="Provide the hypothesis description in natural language (required if run_mode is 'hypothesis')",
        default="I am interested in exploring the hypothesis that the type of sport (team or individual) and total amount of major injuries (that lasted more than 2 months), has an effect on the athlete's peak performance age.",
    )

    # Number of features (relevant if run_mode is 'hypothesis')
    parser.add_argument(
        "--num_features",
        type=int,
        help="Specify the number of features (including response variable) for hypothesis generation",
        default=3,
    )

    # Entities list (can be a JSON file or list)
    parser.add_argument(
        "--entities",
        type=str,
        help="Path to the entities JSON file or a JSON string representing the list of entities",
        default=[
            "Lionel Messi",
            "Cristiano Ronaldo",
            "Neymar Jr.",
            "Kylian Mbappé",
            "Erling Haaland",
            "Mohamed Salah",
            "Kevin De Bruyne",
            "Luka Modrić",
            "Harry Kane",
            "Robert Lewandowski",
            "Sergio Ramos",
            "Virgil van Dijk",
            "Paul Pogba",
            "Gareth Bale",
            "Sadio Mané",
            "Karim Benzema",
            "Ruben Dias",
            "Jack Grealish",
            "Vinícius Júnior",
            "Marc-André ter Stegen",
            # Tennis player names
            "Novak Djokovic",
            "Rafael Nadal",
            "Roger Federer",
            "Serena Williams",
            "Venus Williams",
            "Simona Halep",
            "Naomi Osaka",
            "Ashleigh Barty",
            "Maria Sharapova",
            "Kim Clijsters",
            "Andre Agassi",
            "Pete Sampras",
            "Steffi Graf",
            "Martina Navratilova",
            "Belinda Bencic",
            "Stan Wawrinka",
            "Dominic Thiem",
            "Svetlana Kuznetsova",
            "Andy Murray",
            "Caroline Wozniacki",
        ],
    )

    # Features (if providing them directly)
    parser.add_argument(
        "--features",
        type=str,
        help="A JSON string representing the features in the format {'feature_name': [<feature_description>, <feature_range>]} (required if run_mode is 'features')",
    )

    # Output paths
    parser.add_argument(
        "--data_output_path", type=str, help="Path to the output CSV file where the simulations will be saved", default="data/data.csv"
    )

    parser.add_argument(
        "--features_output_path",
        type=str,
        help="Path to the output JSON file where the features will be saved",
        default="features/features.json",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    MODEL_NAME = args.model_name

    # Parse features if run_mode is 'features' (optional)
    features = None
    if args.run_mode == "features" and args.features:
        features = json.loads(args.features)

    # Validate hypothesis_description for 'hypothesis' mode
    if args.run_mode == "hypothesis" and not args.hypothesis_description:
        print("Error: 'hypothesis_description' is required when 'run_mode' is 'hypothesis'.")
    else:
        run_experiment(
            data_output_path=args.data_output_path,
            features_output_path=args.features_output_path,
            entities=args.entities,
            hypothesis_description=args.hypothesis_description,
            num_features=args.num_features,
            features=features,
        )
