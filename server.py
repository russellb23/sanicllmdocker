from sanic import Sanic, response, text

app = Sanic("SanicLLM")

@app.get("/")
async def hellow(request):
    return text("OK")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
