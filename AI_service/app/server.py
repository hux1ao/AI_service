from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_community.chat_models import ChatOllama
from .embedding.bge import bge_chain

llm = ChatOllama(model="llama2")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")




add_routes(
    app,
    llm,
    path="/ollama",
)
add_routes(
    app,
    bge_chain,
    path="/embedding",
)


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
