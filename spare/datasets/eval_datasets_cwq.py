import numpy as np
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
import copy
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CWQDataset(Dataset):
    def __init__(self, k_shot: int, seed: int, tokenizer: LlamaTokenizer, demonstrations_org_context,
                 demonstrations_org_answer, rog_method: str = "arrow", num_examples: int = None):
        super(CWQDataset, self).__init__()
        self.k_shot = k_shot
        self.seed = seed
        self.tokenizer = tokenizer

        self.demonstrations_org_context = demonstrations_org_context
        self.demonstrations_org_answer = demonstrations_org_answer
        self.rog_method = rog_method.lower()
        assert self.rog_method in ["arrow", "triplet", "lookup"], f"Unknown rog_method: {self.rog_method}"

        self.data = datasets.load_dataset("rmanluo/RoG-cwq")["train"]
        self.data = list(self.data)

        self.demonstration_pool = copy.deepcopy(self.data[-256:])
        self.rng = np.random.RandomState(self.seed)
        self.rng.shuffle(self.demonstration_pool)
        self.demonstrations = self.demonstration_pool[:self.k_shot]

        if num_examples is not None:
            self.data = self.data[:num_examples]

        self.with_info_prompt, self.without_info_prompt = self.verbalise_demonstrations()

    def verbalise_graph(self, graph):
        def arrow_graph(graph):
            lines = ["The entities are presented as a graph in the following section:\n"]
            for path in graph:
                if len(path) == 3:
                    lines.append(f"{path[0]} -- {path[1]} --> {path[2]}")
                elif len(path) == 2:
                    lines.append(f"{path[0]} --> {path[1]}")
            return "\n".join(lines)
        
        def tuple_graph(graph):
            return "The entities are presented as knowledge graph triplets (head, relation, tail):\n" + \
                "\n".join(
                    f"({t[0]}, {t[1]}, {t[2]})" if len(t) == 3 else f"({t[0]}, {t[1]})"
                    for t in graph
                )
            
        def lookup_table_graph(graph):
            def index_to_alpha(n):
                import string
                n += 26  # Start from 'aa' instead of 'a'
                result = ''
                while n >= 0:
                    result = chr(ord('a') + (n % 26)) + result
                    n = n // 26 - 1
                    if n < 0:
                        break
                return result

            entity_map = {}
            rel_map = {}
            entity_ids = {}
            rel_ids = {}
            entity_counter = 0
            rel_counter = 0
            triples = []

            for t in graph:
                if len(t) != 3:
                    continue
                head, rel, tail = t
                for e in [head, tail]:
                    if e not in entity_ids:
                        key = chr(ord('A') + entity_counter)
                        entity_ids[e] = key
                        entity_map[key] = e
                        entity_counter += 1
                if rel not in rel_ids:
                    key = index_to_alpha(rel_counter)
                    rel_ids[rel] = key
                    rel_map[key] = rel
                    rel_counter += 1
                triples.append(f"{entity_ids[head]}, {rel_ids[rel]}, {entity_ids[tail]}")

            lines = []
            for k, v in entity_map.items():
                lines.append("The entities are assigned symbolic keys as follows:\n\n")
                lines.append(f"{k}: {v}")
            for k, v in rel_map.items():
                lines.append("\nThe relations are assigned symbolic keys as follows:\n\n")
                lines.append(f"{k}: {v}")
            lines.append("\nThe graph is defined with the symbolic references:\n\n")
            lines.extend(triples)
            return "" + "\n".join(lines)
        
        if self.rog_method == "arrow":
            return arrow_graph(graph)

        elif self.rog_method == "tuple":
            return tuple_graph(graph)

        elif self.rog_method == "lookup":
            return lookup_table_graph(graph)
        
        raise ValueError("Invaid RoG method =", self.rog_method)

    def verbalise_one_example(self, example, is_test: bool = False):
        graph_context = self.verbalise_graph(example["graph"]) + "\n" if self.demonstrations_org_context else ""
        prompt = graph_context
        prompt += "question: " + example["question"] + "\n"
        if is_test:
            prompt += "answer:"
        else:
            prompt += "answer: " + example["answer"][0] + "\n\n"
        return prompt

    def verbalise_close_book_example(self, example, is_test: bool = False):
        prompt = "question: " + example["question"] + "\n"
        if is_test:
            prompt += "answer:"
        else:
            prompt += "answer: " + example["answer"][0] + "\n\n"
        return prompt

    def verbalise_demonstrations(self):
        with_info_prompt = ""
        without_info_prompt = ""
        for demo in self.demonstrations:
            with_info_prompt += self.verbalise_one_example(demo, is_test=False)
            without_info_prompt += self.verbalise_close_book_example(demo, is_test=False)
        return with_info_prompt, without_info_prompt

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, batch_size, num_workers=4, shuffle=False, rog_method="arrow"):
        """
        Creates a DataLoader for the CWQ dataset, formatting prompts with and without context graphs.
        
        :param batch_size: Batch size for the DataLoader.
        :param num_workers: Number of worker processes.
        :param shuffle: Whether to shuffle the dataset.
        :param rog_method: One of "arrow", "triplet", "lookup" — controls how graph context is formatted.
        """

        def collate_fn(batch):
            with_ctx_inputs_str = []
            without_ctx_inputs_str = []
            answers = []
            questions = []

            for item in batch:
                with_ctx_prompt = self.with_info_prompt
                without_ctx_prompt = self.without_info_prompt

                # Add prompt with graph context (based on rog_method)
                with_ctx_prompt += self.verbalise_one_example(item, rog_method=rog_method, is_test=True)
                with_ctx_inputs_str.append(with_ctx_prompt)

                # Add prompt without graph context
                without_ctx_prompt += self.verbalise_close_book_example(item, is_test=True)
                without_ctx_inputs_str.append(without_ctx_prompt)

                answers.append(item["answer"])
                questions.append(item["question"])

            w_inputs = self.tokenizer(with_ctx_inputs_str, return_tensors="pt", padding=True, truncation=True)
            wo_inputs = self.tokenizer(without_ctx_inputs_str, return_tensors="pt", padding=True, truncation=True)

            return {
                "with_ctx_input_ids": w_inputs["input_ids"],
                "with_ctx_attention_mask": w_inputs["attention_mask"],
                "with_ctx_inputs_str": with_ctx_inputs_str,

                "without_ctx_input_ids": wo_inputs["input_ids"],
                "without_ctx_attention_mask": wo_inputs["attention_mask"],
                "without_ctx_inputs_str": without_ctx_inputs_str,

                "answers": answers,
                "questions": questions,
            }

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
