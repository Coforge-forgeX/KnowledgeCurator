from fastmcp import FastMCP
from contextlib import asynccontextmanager
from typing import AsyncIterator
from agent_search.utils.mongodb_singleton import get_mongodb_client
import logging
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    mongo_client = get_mongodb_client()
    logging.info("✅ MongoDB singleton initialized in lifespan")

    try:
        yield
    finally:
        logging.info("🔧 Shutting down lifespan, closing MongoDB...")
        mongo_client.close()
        logging.info("✅ Lifespan cleanup complete")

mcp = FastMCP("kbCuratorAdapter", lifespan=lifespan)
