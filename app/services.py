import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_falha.joblib")


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
