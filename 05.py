from langchain_ollama import OllamaLLM
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

model = OllamaLLM(model="deepseek-r1:1.5b")

example_template = PromptTemplate.from_template(
    "单词:{word}, 反义词:{antonym}"
)
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "前", "antonym": "后"}
]

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="请给出单词的反义词",
    suffix="单词:{input}, 反义词:",
    input_variables=["input"],
)

res = few_shot_prompt.stream({"input": "左"})
print(next(res))
