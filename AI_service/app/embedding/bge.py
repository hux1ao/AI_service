from langchain_core.runnables import chain
from langchain.embeddings import HuggingFaceBgeEmbeddings

@chain
def bge_chain(text):
    # embedding
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    output = model.embed_query(text)
    return output
