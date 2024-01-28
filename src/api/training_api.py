from fastapi import APIRouter, Response, status
from loguru import logger
from pydantic import BaseModel

from database import ChromaDBAgent


router = APIRouter()


class TrainingModelRequest(BaseModel):
    vector_db_host: str
    sql: str | None = None
    question: str | None = None
    ddl: str | None = None
    doc: str | None = None


@router.post("/train_model")
def train_model(request: TrainingModelRequest) -> Response:
    logger.info(f"Training model with request: {request}")

    try:
        db_agent = ChromaDBAgent(request.vector_db_host)
        if request.sql:
            logger.info(f"Adding sql: {request.sql}")
            if not request.question:
                raise ValueError("Question is not provided.")

            logger.info(f"Adding question: {request.question}")
            db_agent.add_sql_question(request.sql, request.question)

        if request.doc:
            logger.info(f"Adding doc: {request.doc}")
            db_agent.add_doc(request.doc)

        if request.ddl:
            logger.info(f"Adding ddl: {request.ddl}")
            db_agent.add_ddl(request.ddl)

        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        logger.critical(e)
        raise e
