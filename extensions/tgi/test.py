from langchain import HuggingFaceTextGenInference
from langchain.llms.loading import load_llm

llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8010/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

llm.max_new_tokens = max_new_tokens
llm.generate()

llm.save("llm.json")

llm = load_llm("llm.json")

print(llm("Hello"))
