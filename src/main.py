from fastapi import FastAPI

from .api import training_api, retrieving_api


app = FastAPI()

app.include_router(training_api.router)
app.include_router(retrieving_api.router)
