from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel
from sqlai.database import ChromaDBAgent
from sqlai.model import CohereAgent, GeminiAgent


router = APIRouter()


class RetrievingQueryRequest(BaseModel):
    vector_db_host: str
    llm_api_key: str
    question: str
    llm_service_id: str = "gemini"


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

        if request.llm_service_id == "gemini":
            agent = GeminiAgent(request.llm_api_key)
        elif request.llm_service_id == "cohere":
            agent = CohereAgent(request.llm_api_key)
        else:
            raise ValueError(f"Service id {request.llm_service_id} is not supported.")

        generated_sql = agent.generate_sql(request.question, related_questions, related_ddls, related_docs)

        return RetrievingQueryResponse(sql=generated_sql)

    except Exception as e:
        logger.critical(e)
        raise e
