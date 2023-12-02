from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 3


def load_data():
    """BoolQ train split."""
    data = []
    logger.info("Loading BoolQ train split...")
    for item in tqdm(load_dataset("boolq", split="train")):
        question = "\n".join(
            [
                "Read the passage of text provided below, then answer the question.",
                "",
                item["passage"],
                "",
                "True or false - " + item["question"],
            ]
        )
        answer = str(item["answer"]).lower()
        data.append(as_conversation(question, answer))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
