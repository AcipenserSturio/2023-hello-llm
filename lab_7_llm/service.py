"""
Web service for model inference.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pandas import DataFrame
from pydantic.dataclasses import dataclass

from config.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

root_path = Path(__file__).parent


def init_application() -> tuple[FastAPI, LLMPipeline]:
    """
    #doctest: +SKIP
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload

    Returns:
        tuple[fastapi.FastAPI, LLMPipeline]: instance of server and pipeline
    """
    settings = LabSettings(root_path / "settings.json")
    device = "cpu"
    batch_size = 1
    max_length = 120

    empty_dataset = TaskDataset(DataFrame())

    model_pipeline = LLMPipeline(settings.parameters.model, empty_dataset, max_length, batch_size,
                                 device)

    application = FastAPI()
    application.mount("/assets", StaticFiles(directory=root_path / "assets"), name="assets")

    return application, model_pipeline


app, pipeline = init_application()
templates = Jinja2Templates(directory=str(root_path / "assets"))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> templates.TemplateResponse:
    """
    Root endpoint.
    Returns:
        templateResponse: An index.html template
    """
    return templates.TemplateResponse("index.html", {"request": request})


@dataclass
class Query:
    """
    A model reqeust dataclass.
    """

    question: str

@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer endpoint.
    Args:
        query (Query): Query from the user
    Returns:
        dict: A dictionary with the prediction
    """
    prediction = pipeline.infer_sample(query.question)
    return {"infer": prediction}
