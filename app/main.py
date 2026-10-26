from fastapi import FastAPI
from app.routers import router

app = FastAPI(
    title="API de Manutenção Preditiva — Indústria 4.0",
    description="API REST para predição de falhas em maquinários industriais.",
    version="1.0.0",
)

app.include_router(router)
