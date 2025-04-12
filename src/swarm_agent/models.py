import os

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_API_KEY"] = ...
os.environ["OPENAI_API_KEY"] = ...
os.environ["DEEPSEEK_API_KEY"] = ...

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm_4o = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm_4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

llm_deepseek = ChatDeepSeek(model="deepseek-chat", temperature=0.0)
deepseek_context_length = 64000
gpt_4o_context_length = 128000

llm_large = llm_4o
llm_small = llm_4o_mini
context_length = gpt_4o_context_length
