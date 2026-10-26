from sanic import Sanic
from sanic.response import json, text

from llama_cpp import Llama

MODEL_PATH="/models/meta-llama-3.1-8b-instruct-q4_k_m.gguf"

print("[INFO] Loading local LLaMA model . . .")
llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=8192,
        device_map="0:cuda",
        verbose=False)


app = Sanic("SanicLLM")

@app.post("/infer")
async def infer(request):
    prompt = request.json.get("query")

    resp = llm(prompt, max_tokens=512)

    return json({"answer": str((resp['choices'][0])['text'])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
