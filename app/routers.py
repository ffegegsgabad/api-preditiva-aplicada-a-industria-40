from fastapi import APIRouter, Request, HTTPException, status
from app.schemas import PredictInput, PredictOutput, HealthOutput
from app.services import (
    executar_inferencia,
    RegraNegocioException,
    ModeloNaoCarregadoException,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthOutput,
    status_code=status.HTTP_200_OK,
    summary="Verificação de integridade operacional (Health Check)",
    tags=["Monitoramento"],
)
async def health_check(request: Request):
    """
    Endpoint de monitoramento de integridade operacional (Fase 5).
    
    Informa se o serviço FastAPI está ativo e se o modelo de Machine Learning
    reside corretamente na memória RAM (carregado via Lifespan).
    """
    modelo = getattr(request.app.state, "model", None)
    modelo_carregado = modelo is not None

    return HealthOutput(
        status="operacional",
        modelo_carregado=modelo_carregado,
        versao_api="1.0.0",
    )


@router.post(
    "/v1/predict",
    response_model=PredictOutput,
    status_code=status.HTTP_200_OK,
    summary="Predição de risco de falha mecânica",
    tags=["Inferência Preditiva"],
)
async def predict(dados: PredictInput, request: Request):
    """
    Endpoint de inferência para manutenção preditiva (Fase 5).
    
    Consome o payload de sinais telemétricos validado pelo Pydantic (Fase 4),
    acessa o modelo em RAM (Fase 3) e delega a inferência à camada de serviços (Fase 2).
    
    * **200 OK:** Predição calculada com sucesso, retornando diagnóstico e probabilidade.
    * **400 Bad Request:** Violação de restrição física do processo industrial.
    * **422 Unprocessable Entity:** Validação de esquema do Pydantic (tipos incorretos ou fora das faixas).
    * **503 Service Unavailable:** Modelo preditivo indisponível em memória RAM.
    * **500 Internal Server Error:** Falha inesperada durante o processamento.
    """
    modelo = getattr(request.app.state, "model", None)

    try:
        resultado = executar_inferencia(dados, modelo)
        return resultado
    except RegraNegocioException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ModeloNaoCarregadoException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a inferência preditiva: {str(exc)}",
        )
