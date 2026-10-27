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
    # Limites = faixa vista no treino (min/max do dataset) com folga.
    # O modelo interpola bem dentro do que já viu, mas não extrapola com segurança.
    temperatura_ar_k: float = Field(
        ...,
        ge=290.0,
        le=310.0,
        description="Temperatura do ar ambiente em Kelvin [aceito: 290.0 a 310.0 K; no treino: 295.3 a 304.5 K]",
        examples=[298.1],
    )
    temperatura_processo_k: float = Field(
        ...,
        ge=300.0,
        le=320.0,
        description="Temperatura de operação do processo em Kelvin [aceito: 300.0 a 320.0 K; no treino: 305.7 a 313.8 K]",
        examples=[308.6],
    )
    velocidade_rotacao_rpm: float = Field(
        ...,
        ge=1000.0,
        le=3000.0,
        description="Velocidade de rotação do fuso em RPM [aceito: 1000 a 3000 RPM; no treino: 1168 a 2886 RPM]",
        examples=[1551.0],
    )
    torque_nm: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description="Torque aplicado pelo motor em Newton-metro [aceito: maior que 0 até 100 Nm; no treino: 3.8 a 76.6 Nm]",
        examples=[42.8],
    )
    desgaste_ferramenta_min: int = Field(
        ...,
        ge=0,
        le=300,
        description="Tempo de desgaste acumulado da ferramenta de corte em minutos [aceito: 0 a 300 min; no treino: 0 a 253 min]",
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
        examples=["1.1.0"],
    )


class HealthOutput(BaseModel):
    """
    Esquema de saída do monitoramento de integridade operacional (Health Check).
    """
    status: Literal["operacional", "degradado"] = Field(
        ...,
        description="'operacional' com o modelo em memória; 'degradado' se a API subiu sem o modelo",
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
        examples=["1.1.0"],
    )
