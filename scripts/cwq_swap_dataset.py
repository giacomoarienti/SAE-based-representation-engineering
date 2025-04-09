import random
import copy
import sys
import resource
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

# Model and generation parameters
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
RESPONSE_LEN = 22
WORKERS = 8
BATCH_SIZE = 4
MAX_RAM_MEMORY_GB = 40
MAX_GRAPH_HOPS = 2
MAX_GRAPH_INSTANCES = 20


def set_memory_limit(max_memory_gb):
    """
    Set a maximum memory usage limit (in GB) for the current process.
    """
    max_memory_bytes = max_memory_gb * 1024 ** 3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard))


def initialize_model():
    """
    Initialize and return the LLM model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device


def filter_graph(graph, q_entities, a_entities, max_hops=MAX_GRAPH_HOPS, max_instances=MAX_GRAPH_INSTANCES):
    """
    Filter the graph (list of triples) to keep only relevant paths between question and answer entities.
    
    Args:
        graph: List of triples representing the knowledge graph
        q_entities: Set of entities mentioned in the question
        a_entities: Set of entities in the answer
        max_hops: Maximum path length between entities to consider
        max_instances: Maximum number of triples to keep
        
    Returns:
        List of filtered triples
    """
    # Keep only triples that contain either question or answer entities
    filtered = [
        triple for triple in graph 
        if (triple[0] in q_entities or triple[2] in q_entities or 
            triple[0] in a_entities or triple[2] in a_entities)
    ]
    
    # Build an adjacency dictionary for path finding
    adj = {}
    for triple in filtered:
        src, _, dst = triple
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    
    def path_exists(a_entity, q_entity):
        """Find if a path exists between answer and question entities using BFS."""
        queue = [(a_entity, 0)]
        visited = {a_entity}
        
        while queue:
            current, depth = queue.pop(0)
            if current == q_entity:
                return True
                
            if depth < max_hops:
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return False

    # Keep only triples that are part of valid paths
    valid_triples = []
    for triple in filtered:
        src, _, dst = triple
        # Check if any answer entity connects to any question entity
        if any(path_exists(a, q) for a in a_entities for q in q_entities):
            valid_triples.append(triple)
    
    # Sample if we have too many triples
    if len(valid_triples) > max_instances:
        valid_triples = random.sample(valid_triples, max_instances)
    
    return valid_triples


def substitute_graph(graph, a_entities, sub_entities_map):
    """
    Replace original answer entities with their substitutes in the graph.
    
    Args:
        graph: Original graph (list of triples)
        a_entities: Set of answer entities
        sub_entities_map: Mapping from original entities to substitutes
        
    Returns:
        New graph with substituted entities
    """
    new_graph = []
    for triple in graph:
        new_triple = []
        for node in triple:
            # Replace node if it's an answer entity, otherwise keep it
            new_node = sub_entities_map.get(node, node) if node in a_entities else node
            new_triple.append(new_node)
        new_graph.append(new_triple)
    return new_graph


def create_substitution_prompt(question, answer):
    """
    Create a prompt for the LLM to generate answer substitutions.
    
    Args:
        question: The original question
        answer: The original answer
        
    Returns:
        Formatted prompt string
    """
    prompt = (
        "Please rephrase the answer below so that its overall meaning remains unchanged while replacing specific details—"
        "such as dates, numbers, names, and locations—with new, plausible alternatives. "
        "Ensure that the new answer is coherent, contextually appropriate for the accompanying question, and maintains the original semantic intent. "
        "The New Answer should only contain the rephrased answer, without any additional commentary or explanation.\n\n"
        "Question: What country sharing borders with Spain does the SetÃºbal District belong to?\nOriginal Answer: Portugal\nNew Answer: England\n"
        "Question: What language is spoken in the country that has Southern Peninsular?\nOriginal Answer: Icelandic Language\nNew Answer: Norwegian Language\n"
        "Question: What is the capital of the country that has the city of La Paz?\nOriginal Answer: Bolivia\nNew Answer: Peru\n"
        f"Question: {question}\n"
        f"Original Answer: {answer[0] if answer else 'default'}\n"
        "New Answer:"
    )
    return prompt


def transform_entry(entry):
    """
    Process one dataset entry: filter the graph and create an LLM prompt.
    
    Args:
        entry: Dictionary containing question, graph, entities, and answer
        
    Returns:
        Processed entry dictionary or None if invalid
    """
    org_answer = entry["answer"]
    original_graph = entry["graph"]
    q_entities = entry["q_entity"]
    a_entities = entry["a_entity"]
    
    # Filter graph to keep only relevant triples
    org_context = filter_graph(original_graph, q_entities, a_entities)
    if not org_context:
        return None

    # Create prompt for answer substitution
    prompt = create_substitution_prompt(entry["question"], org_answer)
    
    return {
        "question": entry["question"],
        "org_context": org_context,
        "org_answer": org_answer,
        "prompt": prompt,
        "a_entities": a_entities
    }


class CustomGraphDataset(TorchDataset):
    """
    A custom PyTorch dataset that applies transformations to a Hugging Face dataset.
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return transform_entry(self.hf_dataset[idx])


def collate_fn(batch):
    """
    Custom collate function to filter out None values from batch.
    """
    return [item for item in batch if item is not None]


def batch_generate(model, tokenizer, device, prompts, batch_size=BATCH_SIZE):
    """
    Generate outputs in batches for a list of prompts.
    
    Args:
        model: The language model
        tokenizer: The model's tokenizer
        device: Device to run inference on
        prompts: List of prompts to generate responses for
        batch_size: Number of prompts to process at once
        
    Returns:
        List of generated answers
    """
    generated_answers = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating substitute answers", total=total_batches):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch with padding
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=inputs.input_ids.shape[1] + RESPONSE_LEN,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract answers from full outputs
        for prompt, full_text in zip(batch_prompts, decoded_outputs):
            sub_ans = full_text[len(prompt):].strip()
            
            # Clean up the answer
            if "\n" in sub_ans:
                sub_ans = sub_ans.split("\n")[0]
                
            if not sub_ans:
                sub_ans = "default variant"
                
            generated_answers.append(sub_ans)
            
    return generated_answers


def main():
    """
    Main execution function.
    """
    # Initialize model and tokenizer
    model, tokenizer, device = initialize_model()
    
    # Load the dataset
    hf_dataset = load_dataset("rmanluo/RoG-cwq", split="train")
    custom_dataset = CustomGraphDataset(hf_dataset)
    
    # Create DataLoader with multiple workers for preprocessing
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=WORKERS, 
        collate_fn=collate_fn
    )
    
    # Process all entries
    transformed_entries = []
    for batch in tqdm(dataloader, desc="Preprocessing data"):
        transformed_entries.extend(batch)
    
    if not transformed_entries:
        print("No valid entries found after transformation.")
        sys.exit(1)
    
    # Extract prompts and generate substitutions
    prompts = [entry["prompt"] for entry in transformed_entries]
    print(f"Generating substitutions for {len(prompts)} entries...")
    sub_answers = batch_generate(model, tokenizer, device, prompts)
    
    # Create the final dataset with substitutions
    print("Creating final dataset...")
    final_entries = []
    for entry, sub_ans in zip(transformed_entries, sub_answers):
        # Create mapping from original answers to substitutes
        sub_entities_map = {orig: sub_ans for orig in entry["a_entities"]}
        
        # Substitute entities in the graph
        sub_context = substitute_graph(
            copy.deepcopy(entry["org_context"]), 
            entry["a_entities"], 
            sub_entities_map
        )
        
        final_entries.append({
            "question": entry["question"],
            "org_context": entry["org_context"],
            "org_answer": entry["org_answer"],
            "sub_context": sub_context,
            "sub_answer": [sub_ans]
        })
    
    # Save the final dataset
    new_dataset = Dataset.from_list(final_entries)
    new_dataset.save_to_disk("cwq-swap")
    print("Transformation complete. Dataset saved to 'cwq-swap'.")


if __name__ == '__main__':
    try:
        set_memory_limit(MAX_RAM_MEMORY_GB)
        main()
    except MemoryError:
        sys.stderr.write('Memory limit exceeded.\n')
        sys.exit(1)