from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class PredictInput(BaseModel):
    """
    Esquema de entrada para inferência preditiva (Fase 4).
    
    Recebe os sinais dos sensores da máquina industrial com validações estritas
    de limites físicos operacionais, descrições detalhadas e exemplos representativos.

    - strict=True: texto em campo numérico é recusado (inclusive "42.8" entre aspas).
    - extra="forbid": campos que o modelo não conhece são recusados com 422.
    """
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "tipo": "M",
                "temperatura_ar_k": 298.1,
                "temperatura_processo_k": 308.6,
                "velocidade_rotacao_rpm": 1551.0,
                "torque_nm": 42.8,
                "desgaste_ferramenta_min": 0,
            }
        }
    )

    tipo: Literal["L", "M", "H"] = Field(
        ...,
        description="Variante de qualidade do produto: 'L' (Low - 60%), 'M' (Medium - 30%) ou 'H' (High - 10%)",
        examples=["M"],
    )
    temperatura_ar_k: float = Field(
        ...,
        ge=250.0,
        le=350.0,
        description="Temperatura do ar ambiente em Kelvin [faixa física aceitável: 250.0 a 350.0 K]",
        examples=[298.1],
    )
    temperatura_processo_k: float = Field(
        ...,
        ge=250.0,
        le=350.0,
        description="Temperatura de operação do processo em Kelvin [faixa física aceitável: 250.0 a 350.0 K]",
        examples=[308.6],
    )
    velocidade_rotacao_rpm: float = Field(
        ...,
        gt=0.0,
        le=5000.0,
        description="Velocidade de rotação do fuso em RPM [faixa física aceitável: > 0 a 5000.0 RPM]",
        examples=[1551.0],
    )
    torque_nm: float = Field(
        ...,
        gt=0.0,
        le=200.0,
        description="Torque aplicado pelo motor em Newton-metro [faixa física aceitável: > 0 a 200.0 Nm]",
        examples=[42.8],
    )
    desgaste_ferramenta_min: int = Field(
        ...,
        ge=0,
        le=500,
        description="Tempo de desgaste acumulado da ferramenta de corte em minutos [faixa física aceitável: 0 a 500 min]",
        examples=[0],
    )


class PredictOutput(BaseModel):
    """
    Esquema de saída para inferência preditiva (Fase 4).
    
    Retorna o diagnóstico preditivo, a probabilidade calculada e o nível de risco.
    """
    falha_prevista: bool = Field(
        ...,
        description="Indicação binária de falha iminente (True = Risco de Falha, False = Operação Normal)",
        examples=[False],
    )
    probabilidade_falha: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada de ocorrência de falha (entre 0.0 e 1.0)",
        examples=[0.0523],
    )
    nivel_risco: Literal["Baixo", "Médio", "Alto"] = Field(
        ...,
        description="Classificação categórica de severidade operacional com base na probabilidade calculada",
        examples=["Baixo"],
    )
    versao_modelo: str = Field(
        ...,
        description="Identificador da versão do modelo em produção",
        examples=["1.0.0"],
    )


class HealthOutput(BaseModel):
    """
    Esquema de saída do monitoramento de integridade operacional (Health Check).
    """
    status: str = Field(
        ...,
        description="Status operacional da aplicação",
        examples=["operacional"],
    )
    modelo_carregado: bool = Field(
        ...,
        description="Status de carregamento do pipeline preditivo em memória RAM",
        examples=[True],
    )
    versao_api: str = Field(
        ...,
        description="Versão da API",
        examples=["1.0.0"],
    )
