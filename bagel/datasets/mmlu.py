import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 3
CHOICES = ["A", "B", "C", "D", "E", "F", "G"]


def load_data():
    """MMLU training split."""
    data = []
    logger.info("Loading MMLU train split...")
    for item in tqdm(load_dataset("cais/mmlu", "all", split="auxiliary_train")):
        if len(item["choices"]) > len(CHOICES):
            continue
        question = "\n".join(
            [
                item["question"],
                "\n".join(
                    [
                        f"{CHOICES[idx]}. {item['choices'][idx]}"
                        for idx in range(len(item["choices"]))
                    ]
                ),
            ]
        )
        answer = CHOICES[item["answer"]] + " " + item["choices"][item["answer"]]
        if random.random() <= 0.5:
            question = "Answer with only the letter of the answer.\n" + question
            answer = CHOICES[item["answer"]]
        data.append(as_conversation(question, answer))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
