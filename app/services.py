import os
import joblib
import pandas as pd
from app.schemas import PredictInput, PredictOutput

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_falha.joblib")
MODEL_VERSION = "1.1.0"  # 1.1.0 = Árvore de Decisão (max_depth=5)


class RegraNegocioException(Exception):
    """Exceção levantada quando dados violam restrições físicas ou regras de negócio."""
    pass


class ModeloNaoCarregadoException(Exception):
    """Exceção levantada quando o modelo não está disponível em memória."""
    pass


def carregar_modelo_disco():
    """
    Carrega o artefato binário do pipeline treinado do disco.
    
    Esta função deve ser chamada exclusivamente durante o ciclo de vida
    da aplicação (startup via lifespan), garantindo que o modelo resida
    em memória RAM e evitando gargalos de I/O em tempo de requisição.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado no caminho: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def validar_regras_negocio(dados: PredictInput) -> None:
    """
    Valida restrições físicas de operação da máquina industrial (Fase 5 - Erro 400).
    
    Regra de negócio: Na usinagem industrial, o atrito mecânico e o trabalho elétrico
    geram calor continuo, de modo que a temperatura do processo é invariavelmente
    superior à temperatura ambiente do ar (no dataset histórico de 10.000 amostras,
    não existe nenhum registro onde temperatura_processo_k < temperatura_ar_k).
    Uma leitura contrária caracteriza inconsistência nos sensores ou dado inválido.
    """
    if dados.temperatura_processo_k < dados.temperatura_ar_k:
        raise RegraNegocioException(
            f"Inconsistência termodinâmica no processo: a temperatura do processo "
            f"({dados.temperatura_processo_k} K) não pode ser inferior à temperatura "
            f"do ar ambiente ({dados.temperatura_ar_k} K)."
        )


def calcular_nivel_risco(probabilidade: float) -> str:
    """Classifica o nível de risco operacional com base na probabilidade de falha."""
    if probabilidade >= 0.70:
        return "Alto"
    elif probabilidade >= 0.35:
        return "Médio"
    return "Baixo"


def executar_inferencia(dados: PredictInput, modelo) -> PredictOutput:
    """
    Executa a predição através do pipeline de Machine Learning carregado.
    
    Desacopla totalmente a regra matemática e de ML do protocolo HTTP (Fase 2).
    """
    if modelo is None:
        raise ModeloNaoCarregadoException("O modelo preditivo não está carregado na memória RAM.")

    # 1. Validação de regras de negócio físicas (gera erro 400 se incoerente)
    validar_regras_negocio(dados)

    # 2. Monta o DataFrame com os tipos e colunas esperadas pelo ColumnTransformer
    df_input = pd.DataFrame([{
        "tipo": dados.tipo,
        "temperatura_ar_k": dados.temperatura_ar_k,
        "temperatura_processo_k": dados.temperatura_processo_k,
        "velocidade_rotacao_rpm": dados.velocidade_rotacao_rpm,
        "torque_nm": dados.torque_nm,
        "desgaste_ferramenta_min": dados.desgaste_ferramenta_min,
    }])

    # 3. Predição através do pipeline serializado
    pred_classe = int(modelo.predict(df_input)[0])
    pred_proba = float(modelo.predict_proba(df_input)[0][1])

    nivel_risco = calcular_nivel_risco(pred_proba)

    return PredictOutput(
        falha_prevista=bool(pred_classe == 1),
        probabilidade_falha=round(pred_proba, 4),
        nivel_risco=nivel_risco,
        versao_modelo=MODEL_VERSION,
    )
