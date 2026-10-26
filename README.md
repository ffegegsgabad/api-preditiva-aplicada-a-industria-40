# Manutenção Preditiva API — Indústria 4.0

> **SCTEC — Desenvolvimento de IA para Análise Preditiva [T1 e T2]**  
> **Projeto Avaliativo Final do Módulo 2 (Semana 13)**  
> **Aluno:** [Seu Nome Completo]  
> **Repositório GitHub:** [[Link do Repositório GitHub]](https://github.com/[seu-usuario]/manutencao-preditiva-api)  
> **Vídeo de Demonstração (Google Drive):** [[Link do Vídeo no Google Drive]](https://drive.google.com/file/d/SEU_ID_DO_VIDEO/view?usp=sharing)  

---

## 1. Nome e Resumo do Projeto

A **Manutenção Preditiva API** é um microsserviço RESTful construído em **FastAPI** voltado ao paradigma da **Indústria 4.0**. Seu propósito fundamental é antecipar falhas catastróficas em equipamentos mecânicos (fuso, motor e ferramentas de corte) a partir de sinais telemétricos contínuos de sensores operacionais.

### Problema de Negócio
Na manufatura moderna, paradas não programadas geram perdas financeiras astronômicas, desperdício de lotes de produção e riscos à segurança operacional. Ao substituir manutenções corretivas (reativas) ou preventivas em intervalos fixos por um **sistema preditivo acionado por dados**, a indústria consegue intervir no equipamento no momento exato antes da quebra, otimizando a vida útil dos componentes e assegurando máxima disponibilidade fabril.

* **Objetivo do Modelo:** Classificação binária preditiva para determinar a probabilidade de falha mecânica (`falha_maquina`).
* **Abordagem Técnica:** Pipeline scikit-learn serializado com pré-processamento robusto, balanceamento com SMOTE e classificador Random Forest servido em arquitetura assíncrona modular com FastAPI.

---

## 2. Origem do Modelo e Evolução Técnica

Este projeto reaproveita e aprimora substancialmente o modelo construído no **Projeto Final do Módulo 1** (`pipeline-preditivo-completo-aplicado-a-industria-40-v2`).

### O que mudou em relação à versão anterior (V2):

| Dimensão Técnica | Versão Anterior (M1 / V2) | Versão Atual de Produção (M2 / Final) | Justificativa de Engenharia e MLOps |
|---|---|---|---|
| **Estratégia de Particionamento** | Split 80/20 (Treino / Teste) | **Split Estratificado 60/20/20** (Treino / Validação / Teste) | O conjunto de validação permite comparar algoritmos e hiperparâmetros sem contaminar os dados de teste (avaliados uma única vez). |
| **Pipeline de Pré-processamento** | Tratamento manual e disperso com Pandas | **`ColumnTransformer` + `Pipeline` integrado** | Evita *training-serving skew*: a API consome o dado cru e o pipeline executa todas as transformações de forma atômica. |
| **Imputação de Nulos** | Mediana aplicada de forma global | **`SimpleImputer(strategy='median')`** | Os 500 nulos por sensor têm a mediana aprendida estritamente no treino e persistida no artefato. |
| **Variável Categórica (`tipo`)** | Tratamento implícito / manual | **`OneHotEncoder(handle_unknown='ignore')`** | Lida de maneira resiliente e estrita com as variantes L, M e H, sem quebrar com novas categorias. |
| **Balanceamento de Classes** | SMOTE aplicado manualmente fora do fluxo | **SMOTE dentro do `imblearn.pipeline.Pipeline`** | O SMOTE atua exclusivamente durante o `fit()` de treino e é ignorado no `predict()` em produção. |
| **Métricas de Avaliação** | Acurácia e F1 pontuais | **Acurácia, Recall, F1, ROC-AUC e Matriz de Confusão** | Em manutenção preditiva, o **Recall** e o **ROC-AUC** são vitais para mitigar falsos negativos (máquinas que quebram sem alerta). |
| **Deployment / Servir Modelo** | Execução restrita a script/notebook | **API REST FastAPI com Lifespan** | Modelo carregado uma única vez na memória RAM (`app.state`), eliminando gargalos de I/O por requisição. |

---

## 3. Tecnologias Utilizadas

O ecossistema do projeto foi construído sobre Python 3 com versões fixadas no `requirements.txt`:

* **[FastAPI](https://fastapi.tiangolo.com/) (v0.162.0):** Framework assíncrono para construção de APIs REST de alta performance.
* **[Uvicorn](https://www.uvicorn.org/) (v0.54.0):** Servidor ASGI leve e de baixa latência para produção Python.
* **[Pydantic](https://docs.pydantic.dev/) (v2.14.1):** Modelagem de dados, tipagem estrita e validação de contratos com `Field`.
* **[Scikit-Learn](https://scikit-learn.org/) (v1.9.1):** Construção dos pipelines de transformação e estimador Random Forest.
* **[Imbalanced-Learn](https://imbalanced-learn.org/) (v0.14.2):** Implementação de SMOTE dentro do pipeline de treinamento.
* **[Pandas](https://pandas.pydata.org/) (v3.1.0) & [NumPy](https://numpy.org/) (v2.5.4):** Manipulação e computação de estruturas de dados.
* **[Joblib](https://joblib.readthedocs.io/) (v1.6.0):** Serialização e desserialização de artefatos de Machine Learning.
* **[Matplotlib](https://matplotlib.org/) (v3.11.2) & [Seaborn](https://seaborn.pydata.org/) (v0.13.2):** Visualização gráfica de métricas e curvas ROC no notebook.

---

## 4. Instruções de Instalação e Execução

### Pré-requisitos
* Python 3.10 ou superior
* Git instalado

### Passo a Passo

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/[seu-usuario]/manutencao-preditiva-api.git
   cd manutencao-preditiva-api
   ```

2. **Criar e Ativar o Ambiente Virtual:**
   ```bash
   # No macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # No Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Instalar as Dependências:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Inicializar o Servidor ASGI com Uvicorn:**
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Acessar a Aplicação:**
   * **Swagger UI (Documentação Interativa):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * **Redoc UI:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
   * **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 5. Arquitetura Modular em Camadas

O projeto obedece à separação estrita de responsabilidades em camadas desacopladas:

```
manutencao-preditiva-api/
├── app/
│   ├── __init__.py           # Inicializador do pacote
│   ├── main.py               # Instanciação FastAPI, Lifespan e registro de routers
│   ├── routers.py            # Endpoints HTTP (/health e /v1/predict) isolados
│   ├── schemas.py            # Modelos Pydantic (BaseModel, Field) de entrada e saída
│   └── services.py           # Carregamento do modelo e lógica de inferência desacoplada
├── data/
│   └── manutencao_preditiva.csv # Dataset histórico de 10.000 amostras
├── models/
│   ├── modelo_falha.joblib   # Pipeline completo serializado
│   └── metricas.json         # Métricas registradas de validação e teste
├── notebooks/
│   └── treinamento_modelo.ipynb # Pipeline de Ciência de Dados documentado (Fase 1)
├── requirements.txt          # Dependências com versões travadas
└── README.md                 # Documentação técnica central
```

### Otimização de Ciclo de Vida (`Lifespan`)
Através do `@asynccontextmanager`, o artefato `modelo_falha.joblib` é carregado no evento de inicialização do Uvicorn e vinculado a `app.state.model`. Isso garante que requisições HTTP façam inferências exclusivamente em memória RAM, **sem leituras concorrentes em disco a cada chamada**.

---

## 6. Documentação da API e Exemplos de Payloads

### 6.1. Endpoint de Monitoramento: `GET /health`

**Resposta de Sucesso (`HTTP 200 OK`):**
```json
{
  "status": "operacional",
  "modelo_carregado": true,
  "versao_api": "1.0.0"
}
```

---

### 6.2. Endpoint de Inferência: `POST /v1/predict`

#### Caso 1: Sucesso — Máquina em Operação Normal (`HTTP 200 OK`)
**Requisição:**
```json
{
  "tipo": "M",
  "temperatura_ar_k": 298.1,
  "temperatura_processo_k": 308.6,
  "velocidade_rotacao_rpm": 1551.0,
  "torque_nm": 42.8,
  "desgaste_ferramenta_min": 0
}
```

**Resposta:**
```json
{
  "falha_prevista": false,
  "probabilidade_falha": 0.0445,
  "nivel_risco": "Baixo",
  "versao_modelo": "1.0.0"
}
```

---

#### Caso 2: Sucesso — Máquina sob Estresse com Risco de Falha (`HTTP 200 OK`)
**Requisição:**
```json
{
  "tipo": "H",
  "temperatura_ar_k": 304.0,
  "temperatura_processo_k": 313.0,
  "velocidade_rotacao_rpm": 1200.0,
  "torque_nm": 75.0,
  "desgaste_ferramenta_min": 240
}
```

**Resposta:**
```json
{
  "falha_prevista": true,
  "probabilidade_falha": 0.573,
  "nivel_risco": "Médio",
  "versao_modelo": "1.0.0"
}
```

---

#### Caso 3: Violação de Regra de Negócio Física (`HTTP 400 Bad Request`)
Se a temperatura do processo for inferior à temperatura do ar ambiente, a camada de serviço levanta uma exceção semântica:

**Requisição (Inconsistente):**
```json
{
  "tipo": "M",
  "temperatura_ar_k": 310.0,
  "temperatura_processo_k": 295.0,
  "velocidade_rotacao_rpm": 1500.0,
  "torque_nm": 40.0,
  "desgaste_ferramenta_min": 50
}
```

**Resposta:**
```json
{
  "detail": "Inconsistência termodinâmica no processo: a temperatura do processo (295.0 K) não pode ser inferior à temperatura do ar ambiente (310.0 K)."
}
```

---

#### Caso 4: Erro de Validação de Esquema Pydantic (`HTTP 422 Unprocessable Entity`)
Interceptado automaticamente antes de alcançar o pipeline de Machine Learning:

* **Categoria Inválida (`tipo: "X"`):**
  ```json
  {
    "detail": [
      {
        "type": "literal_error",
        "loc": ["body", "tipo"],
        "msg": "Input should be 'L', 'M' or 'H'",
        "input": "X"
      }
    ]
  }
  ```
* **Valor Fora do Limite Físico (`torque_nm: -10.0`):**
  ```json
  {
    "detail": [
      {
        "type": "greater_than",
        "loc": ["body", "torque_nm"],
        "msg": "Input should be greater than 0",
        "input": -10.0
      }
    ]
  }
  ```

---

## 7. Desempenho e Métricas do Modelo

Conforme detalhado no notebook `notebooks/treinamento_modelo.ipynb` e no relatório persistido `models/metricas.json`:

### Comparativo na Validação (Seleção de Modelos):
| Algoritmo Candidato | Acurácia | Recall (Falha) | Precisão | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Regressão Logística** | 81,80% | 82,35% | 13,73% | 0,2353 | 0,8776 |
| **Árvore de Decisão (`max_depth=5`)** | 92,15% | **91,18%** | 29,11% | 0,4413 | 0,9286 |
| **Random Forest (`max_depth=6`) [Vencedor]** | **91,90%** | **79,41%** | **26,73%** | **0,4000** | **0,9561** |

### Desempenho Final no Conjunto de Teste (Random Forest):
* **Acurácia:** 93,85%
* **Recall (Sensibilidade para Falhas):** 76,47%
* **Precisão:** 32,70%
* **F1-Score:** 0,4581
* **ROC-AUC:** **0,9338**

**Matriz de Confusão no Teste (2.000 amostras):**
* Verdadeiros Negativos: 1.825
* Falsos Positivos: 107
* Falsos Negativos: 16 (apenas 16 falhas não detectadas em 2.000 amostras)
* Verdadeiros Positivos: 52

---

## 8. Estrutura de Branches Git (GitFlow)

O projeto cumpre integralmente as boas práticas de versionamento exigidas no edital. Nenhuma feature branch foi excluída:

* `main`: Código de produção estável e validado.
* `develop`: Branch de integração de todas as funcionalidades concluídas.
* `feat/pipeline-treinamento`: Notebook com carga, limpeza de leakage, split estratificado e pipeline.
* `feat/serializacao-modelo`: Exportação de `modelo_falha.joblib` e `metricas.json`.
* `feat/estrutura-api`: Arquitetura em camadas no pacote `app/`.
* `feat/lifespan-modelo`: Gerenciamento assíncrono de inicialização e carga em RAM.
* `feat/schemas-pydantic`: Validação estrita de contratos de dados com `Field`.
* `feat/endpoints-predict-health`: Rotas `/health`, `/v1/predict` e códigos semânticos (200, 400, 422, 503).
* `docs/readme`: Documentação técnica completa e evidências de entrega.

---

## 9. Evidências e Gravação do Vídeo

* **Link do Repositório GitHub:** `https://github.com/[seu-usuario]/manutencao-preditiva-api`
* **Link da Apresentação em Vídeo:** `https://drive.google.com/file/d/SEU_ID_DO_VIDEO/view?usp=sharing` *(Configurado em modo leitor para qualquer pessoa com o link)*

### Roteiro Seguido no Vídeo (Duração $\le$ 10 minutos):
1. **[0:00–1:00] Objetivo do Sistema:** Contexto da Indústria 4.0 e problema de paradas não programadas.
2. **[1:00–4:00] Demonstração no `/docs`:** Chamada do `/health`, `/v1/predict` com caso de sucesso, demonstração de erro 422 (Pydantic) e erro 400 (regra física) + log do modelo sendo carregado uma única vez no startup.
3. **[4:00–5:00] Instruções de Execução:** Demonstração do `venv`, instalação do `requirements.txt` e comando `uvicorn`.
4. **[5:00–6:00] Organização Prévia:** Planejamento de tarefas e checklist de critérios antes do código.
5. **[6:00–7:30] Branches no GitHub:** Apresentação da árvore de branches preservadas a partir da `develop`.
6. **[7:30–9:00] Argumentação Técnica:** Justificativas do pipeline unificado, SMOTE no treino, Lifespan em memória e priorização de Recall/ROC-AUC.
7. **[9:00–10:00] Oportunidades de Melhoria:** Conteinerização com Docker, testes automatizados contínuos (CI com Pytest), observabilidade de drift de dados e deploy em nuvem.
