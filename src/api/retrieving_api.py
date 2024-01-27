from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from database import ChromaDBAgent
from model import CohereAgent


router = APIRouter()


class RetrievingQueryRequest(BaseModel):
    vector_db_host: str
    key: str
    question: str


class RetrievingQueryResponse(BaseModel):
    sql: str


@router.post("/retreive_query", response_model=RetrievingQueryResponse)
def get_results(request: RetrievingQueryRequest) -> RetrievingQueryResponse:
    logger.info(f"Retrieving query with request: {request}")

    try:
        db_agent = ChromaDBAgent(request.vector_db_host)
        related_questions = db_agent.get_related_questions(request.question)
        related_ddls = db_agent.get_related_ddls(request.question)
        related_docs = db_agent.get_related_docs(request.question)

        cohere_agent = CohereAgent(request.key)
        generated_sql = cohere_agent.generate_sql(request.question, related_questions, related_ddls, related_docs)

        return RetrievingQueryResponse(sql=generated_sql)

    except Exception as e:
        logger.critical(e)
        raise e
