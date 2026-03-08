from tensor.instruct import Instruct
from tensor.instruct.data import HubSource

run = Instruct(
    base="/root/testing/output/my-model/model",
    data=HubSource("HuggingFaceH4/ultrachat_200k", split="train_sft"),
    output="/root/testing/output/my-instruct-model",
)

run.train()