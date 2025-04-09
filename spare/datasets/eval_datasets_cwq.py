import numpy as np
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
import copy
import logging
from spare.datasets.graph_utils import arrow_graph, tuple_graph, lookup_table_graph

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CWQSwap(Dataset):
    def __init__(self, k_shot: int, seed: int, tokenizer: LlamaTokenizer,
                 demonstrations_org_context, demonstrations_org_answer,
                 rog_method: str = "arrow",
                 num_examples=None, test_example_org_context=False):
        super(CWQSwap, self).__init__()
        self.k_shot = k_shot
        self.seed = seed
        self.tokenizer = tokenizer
        self.rog_method = rog_method  # default is "arrow"

        self.demonstrations_org_context = demonstrations_org_context
        self.demonstrations_org_answer = demonstrations_org_answer

        self.test_example_org_context = test_example_org_context
        if self.test_example_org_context:
            logger.info("no KC")

        self.data = datasets.load_dataset("giacomoarienti/rog-swap-cwq", split="train")
        self.data = [_ for _ in self.data]
        # Make a deep copy of the last 256 examples to form a demonstration pool
        self.demonstration_pool = copy.deepcopy(self.data[-256:])
        self.rng = np.random.RandomState(self.seed)
        self.rng.shuffle(self.demonstration_pool)
        self.demonstrations = self.demonstration_pool[:self.k_shot]

        if num_examples is not None:
            self.data = self.data[:num_examples]
        self.with_ctx_prompt, self.without_ctx_prompt = self.verbalise_demonstrations()

    def verbalise_graph(self, graph):
        if self.rog_method == "arrow":
            return arrow_graph(graph)
        elif self.rog_method == "tuple":
            return tuple_graph(graph)
        elif self.rog_method == "lookup":
            return lookup_table_graph(graph)
        else:
            raise ValueError(f"Unknown rog_method: {self.rog_method}")

    def verbalise_one_example(self, example, ctx_key, ans_key, is_test=False):
        # Here we assume that example[ctx_key] contains a graph that needs to be verbalised.
        context_str = self.verbalise_graph(example[ctx_key])
        prompt = "context: " + context_str + "\n\n"
        prompt = prompt + "question: " + example["question"] + "\n"
        if is_test:
            prompt = prompt + "answer:"
        else:
            prompt = prompt + "answer: " + example[ans_key][0] + "\n\n"
        return prompt

    def verbalise_close_book_example(self, example, is_test=False):
        prompt = "question: " + example["question"] + "\n"
        if is_test:
            prompt += "answer:"
        else:
            prompt += "answer: " + example["org_answer"][0] + "\n\n"
        return prompt

    def verbalise_demonstrations(self, demonstrations=None):
        if demonstrations is None:
            demonstrations = self.demonstrations
        with_ctx_prompt = ""
        without_ctx_prompt = ""
        ctx_key = "org_context" if self.demonstrations_org_context else "sub_context"
        ans_key = "org_answer" if self.demonstrations_org_answer else "sub_answer"
        for demonstration in demonstrations:
            with_ctx_prompt = with_ctx_prompt + self.verbalise_one_example(demonstration, ctx_key, ans_key)
            without_ctx_prompt = without_ctx_prompt + self.verbalise_close_book_example(demonstration)
        return with_ctx_prompt, without_ctx_prompt

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, batch_size, num_workers=4, shuffle=False):
        test_ctx_key = "org_context" if self.test_example_org_context else "sub_context"

        def collate_fn(batch):
            with_ctx_inputs_str = []
            without_ctx_inputs_str = []
            sub_answers = []
            org_answers = []
            questions = []
            for item in batch:
                with_ctx_prompt = self.with_ctx_prompt
                without_ctx_prompt = self.without_ctx_prompt

                with_ctx_prompt += self.verbalise_one_example(item, test_ctx_key, None, is_test=True)
                with_ctx_inputs_str.append(with_ctx_prompt)

                without_ctx_prompt = without_ctx_prompt + "question: " + item["question"] + "\n"
                without_ctx_prompt = without_ctx_prompt + "answer:"
                without_ctx_inputs_str.append(without_ctx_prompt)

                sub_answers.append(item["sub_answer"])
                org_answers.append(item["org_answer"])
                questions.append(item["question"])

            w_inputs = self.tokenizer(with_ctx_inputs_str, return_tensors="pt", padding=True)
            wo_inputs = self.tokenizer(without_ctx_inputs_str, return_tensors="pt", padding=True)

            return {"with_ctx_input_ids": w_inputs["input_ids"],
                    "with_ctx_attention_mask": w_inputs["attention_mask"],
                    "with_ctx_inputs_str": with_ctx_inputs_str,  # list[list]

                    "without_ctx_input_ids": wo_inputs["input_ids"],
                    "without_ctx_attention_mask": wo_inputs["attention_mask"],
                    "without_ctx_inputs_str": without_ctx_inputs_str,  # list[list]

                    "sub_answers": sub_answers,  # list[list[srt]]
                    "org_answers": org_answers,  # list[list[srt]]
                    "questions": questions
                    }

        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=collate_fn)
