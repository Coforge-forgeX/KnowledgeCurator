import os
from sys import api_version
import nest_asyncio
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from openai import AzureOpenAI
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.azure_openai import azure_openai_complete
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
import aiohttp
from configparser import ConfigParser
from kbcurator.server.server import mcp

# nest_asyncio.apply()

config = ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), r'../config.ini')
config_path = os.path.abspath(config_path)
print("Loading config from:", config_path)
read_files = config.read(config_path)
if not read_files:
    raise FileNotFoundError(f"Could not find config.ini at {config_path}")

azure_api_key = os.getenv('AZURE_OPENAI_LLM_MODEL_API_KEY')
azure_api_base = os.getenv('AZURE_OPENAI_LLM_MODEL_API_BASE')
azure_api_version = os.getenv('AZURE_OPENAI_LLM_MODEL_API_VERSION')
azure_deployement_name = os.getenv('AZURE_OPENAI_LLM_MODEL_LLM_MODEL')

os.environ["NEO4J_URI"] = os.getenv("NEO4J_DATABASE_NEO4J_BOLT_URI", "bolt://localhost:7687") or "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_DATABASE_NEO4J_USER") or ""
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_DATABASE_NEO4J_PASSWORD") or ""
 
embedding_dim = int(os.getenv("OLLAMA_MODEL_EMBEDDING_MODEL_DIMS", "3072"))
max_token_size = int(os.getenv("OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS", "8192"))
base_url = os.getenv("OLLAMA_MODEL_BASE_URL")
embedding_model = os.getenv("OLLAMA_MODEL_EMBEDDING_MODEL")

embedding_dim = int(os.getenv("OLLAMA_MODEL_EMBEDDING_MODEL_DIMS", "3072"))
max_token_size = int(os.getenv("OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS", "8192"))
base_url = os.getenv("OLLAMA_MODEL_BASE_URL")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key,
    }
    endpoint = f"{azure_api_base}openai/deployments/{azure_deployement_name}/chat/completions?api-version={azure_api_version}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0),
        "top_p": kwargs.get("top_p", 1),
        "n": kwargs.get("n", 1),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            return result["choices"][0]["message"]["content"]

async def initialize_rag(working_dir: str = "./srivastava"):
    
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embedding_model,
                host=base_url,
            ),
        ),
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
        chunk_token_size=1000,
        chunk_overlap_token_size=200,
    )

    await rag.initialize_storages()
    initialize_share_data()
    await initialize_pipeline_status()

    return rag