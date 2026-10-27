import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import router
from app.services import carregar_modelo_disco

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manutencao_preditiva")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de ciclo de vida da aplicação (Lifespan - Fase 3).
    
    Carrega o modelo serializado na memória RAM uma única vez no startup.
    O pipeline fica acessível através de `app.state.model`, eliminando leituras
    de disco durante o processamento das requisições HTTP (evita training-serving I/O overhead).
    """
    logger.info("=== INICIANDO APLICAÇÃO — CARREGANDO MODELO PREDITIVO ===")
    try:
        app.state.model = carregar_modelo_disco()
        print(">>> [LIFESPAN] Pipeline de Machine Learning carregado com sucesso na memória RAM! <<<")
        logger.info("Modelo de falha carregado na memória RAM (app.state.model).")
    except Exception as exc:
        logger.error(f"Erro ao carregar modelo no lifespan: {exc}")
        app.state.model = None

    yield

    logger.info("=== ENCERRANDO APLICAÇÃO ===")
    app.state.model = None
    print(">>> [LIFESPAN] Aplicação finalizada. Memória liberada. <<<")


app = FastAPI(
    title="API de Manutenção Preditiva — Indústria 4.0",
    description="API REST de alta performance para inferência em tempo real de falhas mecânicas.",
    version="1.1.0",
    lifespan=lifespan,
)

app.include_router(router)
