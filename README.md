# Manutenção Preditiva API — Indústria 4.0

> **SCTEC — Desenvolvimento de IA para Análise Preditiva [T1 e T2]**  
> **Projeto Avaliativo Final do Módulo 2 (Semana 13)**  
> **Aluno:** Rian Gomes  
> **Repositório GitHub:** [github.com/ffegegsgabad/api-preditiva-aplicada-a-industria-40](https://github.com/ffegegsgabad/api-preditiva-aplicada-a-industria-40)  
> **Vídeo de Demonstração (Google Drive):** [[Link do Vídeo no Google Drive]](https://drive.google.com/file/d/SEU_ID_DO_VIDEO/view?usp=sharing)  

---

## 1. Nome e Resumo do Projeto

A **Manutenção Preditiva API** é um microsserviço RESTful construído em **FastAPI** voltado ao paradigma da **Indústria 4.0**. Seu propósito é antecipar falhas em equipamentos mecânicos (fuso, motor e ferramentas de corte) a partir das leituras dos sensores de operação.

### Problema de Negócio
Na manufatura, uma parada não programada interrompe a linha de produção, gera manutenção emergencial, desperdício de lotes e risco para quem opera a máquina. Ao substituir manutenções corretivas (reativas) ou preventivas em intervalos fixos por um **sistema preditivo acionado por dados**, a indústria consegue intervir no equipamento antes da quebra, aproveitando melhor a vida útil dos componentes e mantendo a fábrica disponível.

* **Objetivo do Modelo:** Classificação binária para estimar a probabilidade de falha mecânica (`falha_maquina`).
* **Abordagem Técnica:** Pipeline scikit-learn serializado com pré-processamento, balanceamento com SMOTE e classificador **Árvore de Decisão (`max_depth=5`)**, servido em uma API FastAPI organizada em camadas.

---

## 2. Origem do Modelo e Evolução Técnica

Este projeto reaproveita e aprimora o modelo construído no **Projeto Final do Módulo 1** (`pipeline-preditivo-completo-aplicado-a-industria-40-v2`) — **Opção A** do enunciado.

### O que mudou em relação à versão anterior (V2):

| Dimensão Técnica | Versão Anterior (M1 / V2) | Versão Atual de Produção (M2 / Final) | Justificativa de Engenharia e MLOps |
|---|---|---|---|
| **Estratégia de Particionamento** | Split 80/20 (Treino / Teste) | **Split Estratificado 60/20/20** (Treino / Validação / Teste) | O conjunto de validação permite comparar algoritmos e hiperparâmetros sem contaminar os dados de teste (avaliados uma única vez). |
| **Pipeline de Pré-processamento** | Tratamento manual e disperso com Pandas | **`ColumnTransformer` + `Pipeline` integrado** | Evita *training-serving skew*: a API consome o dado cru e o pipeline executa todas as transformações de forma atômica. |
| **Imputação de Nulos** | Mediana aplicada de forma global, antes do split | **`SimpleImputer(strategy='median')`** | Os 500 nulos por sensor têm a mediana aprendida estritamente no treino e persistida no artefato. |
| **Variável Categórica (`tipo`)** | Tratamento implícito / manual | **`OneHotEncoder(handle_unknown='ignore')`** | Lida com as variantes L, M e H sem quebrar com novas categorias. |
| **Balanceamento de Classes** | SMOTE aplicado manualmente fora do fluxo | **SMOTE dentro do `imblearn.pipeline.Pipeline`** | O SMOTE atua exclusivamente durante o `fit()` de treino e é ignorado no `predict()` em produção. |
| **Seleção do Modelo** | Árvore `max_depth=10` escolhida comparando no teste | **Árvore `max_depth=5` escolhida na validação e confirmada na validação cruzada (5 folds)** | O teste só mede o modelo já escolhido; a validação cruzada mostra se o resultado é estável. |
| **Métricas de Avaliação** | Acurácia, recall e F1 no teste | **Acurácia, Recall, Precisão, F1, ROC-AUC, Matriz de Confusão e importância das variáveis** | O **Recall** é o critério principal: um falso negativo é uma máquina que quebra sem alerta. |
| **Deployment / Servir Modelo** | Execução restrita a script/notebook | **API REST FastAPI com Lifespan** | Modelo carregado uma única vez na memória RAM (`app.state`), eliminando gargalos de I/O por requisição. |

---

## 3. Tecnologias Utilizadas

O ecossistema do projeto foi construído sobre Python 3.14 com versões fixadas no `requirements.txt`:

* **[FastAPI](https://fastapi.tiangolo.com/) (v0.162.0):** Framework para construção de APIs REST com Swagger automático.
* **[Uvicorn](https://www.uvicorn.org/) (v0.54.0):** Servidor ASGI que executa a aplicação.
* **[Pydantic](https://docs.pydantic.dev/) (v2.14.1):** Modelagem de dados, tipagem estrita e validação de contratos com `Field`.
* **[Scikit-Learn](https://scikit-learn.org/) (v1.9.1):** Pipelines de transformação, modelos candidatos (Regressão Logística, Árvore de Decisão e Random Forest), validação cruzada e métricas.
* **[Imbalanced-Learn](https://imbalanced-learn.org/) (v0.14.2):** Implementação de SMOTE dentro do pipeline de treinamento.
* **[Pandas](https://pandas.pydata.org/) (v3.1.0) & [NumPy](https://numpy.org/) (v2.5.4):** Manipulação e computação de estruturas de dados.
* **[Joblib](https://joblib.readthedocs.io/) (v1.6.0):** Serialização e desserialização de artefatos de Machine Learning.
* **[Matplotlib](https://matplotlib.org/) (v3.11.2) & [Seaborn](https://seaborn.pydata.org/) (v0.13.2):** Matriz de confusão, curva ROC e importância das variáveis no notebook.
* **[Jupyter](https://jupyter.org/) (v1.1.1):** Ambiente do notebook de treinamento.

---

## 4. Instruções de Instalação e Execução

### Pré-requisitos
* **Python 3.12 ou superior** (testado no 3.14.5 — o NumPy 2.5 exige 3.12+)
* Git instalado

### Passo a Passo

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/ffegegsgabad/api-preditiva-aplicada-a-industria-40.git
   cd api-preditiva-aplicada-a-industria-40
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

4. **(Opcional) Retreinar o Modelo:** o artefato já está em `models/`. Para reproduzir o treinamento, abra o notebook e execute todas as células (*Kernel → Restart & Run All*). Com `random_state=42`, as métricas saem iguais.
   ```bash
   jupyter notebook notebooks/treinamento_modelo.ipynb
   ```

5. **Inicializar o Servidor ASGI com Uvicorn:**
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

6. **Acessar a Aplicação:**
   * **Swagger UI (Documentação Interativa):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * **Redoc UI:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
   * **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 5. Arquitetura Modular em Camadas

O projeto obedece à separação de responsabilidades em camadas desacopladas:

```
api-preditiva-aplicada-a-industria-40/
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
Através do `@asynccontextmanager`, o artefato `modelo_falha.joblib` é carregado no evento de inicialização do Uvicorn e vinculado a `app.state.model`. Isso garante que requisições HTTP façam inferências exclusivamente em memória RAM, **sem leituras em disco a cada chamada**.

A rota de predição é declarada com `def` (e não `async def`): a inferência do scikit-learn é cálculo de CPU, e o FastAPI executa rotas `def` num pool de threads, sem travar o servidor para as demais requisições.

---

## 6. Documentação da API e Exemplos de Payloads

### 6.1. Endpoint de Monitoramento: `GET /health`

**Resposta de Sucesso (`HTTP 200 OK`):**
```json
{
  "status": "operacional",
  "modelo_carregado": true,
  "versao_api": "1.1.0"
}
```

Se a API subir sem conseguir carregar o modelo, o `/health` continua respondendo `200 OK`, mas com `"status": "degradado"` e `"modelo_carregado": false` — e o `/v1/predict` passa a responder `503 Service Unavailable`.

---

### 6.2. Endpoint de Inferência: `POST /v1/predict`

#### Limites aceitos na entrada

Os limites seguem a faixa vista no treino, com folga. O modelo é bom em **interpolar** (prever dentro do que já viu) e não é confiável para **extrapolar** — por isso um valor muito fora dessa faixa é recusado com `422` em vez de gerar uma previsão sem base.

| Campo | Tipo | Aceito | Visto no treino |
|---|---|---|---|
| `tipo` | texto | `"L"`, `"M"` ou `"H"` | L, M, H |
| `temperatura_ar_k` | número | 290 a 310 K | 295,3 a 304,5 K |
| `temperatura_processo_k` | número | 300 a 320 K | 305,7 a 313,8 K |
| `velocidade_rotacao_rpm` | número | 1.000 a 3.000 rpm | 1.168 a 2.886 rpm |
| `torque_nm` | número | maior que 0 até 100 Nm | 3,8 a 76,6 Nm |
| `desgaste_ferramenta_min` | inteiro | 0 a 300 min | 0 a 253 min |

O schema é **estrito**: texto em campo numérico (mesmo `"42.8"` entre aspas) e campos que o modelo não conhece são recusados com `422`.

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
  "probabilidade_falha": 0.0585,
  "nivel_risco": "Baixo",
  "versao_modelo": "1.1.0"
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
  "probabilidade_falha": 0.9492,
  "nivel_risco": "Alto",
  "versao_modelo": "1.1.0"
}
```

O `nivel_risco` traduz a probabilidade para a equipe de manutenção: **Baixo** abaixo de 0,35, **Médio** de 0,35 a 0,70 e **Alto** a partir de 0,70.

---

#### Caso 3: Violação de Regra de Negócio Física (`HTTP 400 Bad Request`)
Se a temperatura do processo for inferior à temperatura do ar ambiente, a camada de serviço levanta uma exceção de regra de negócio. Nos 10.000 registros do dataset isso nunca acontece (a menor temperatura do processo é 305,7 K e a maior do ar é 304,5 K), então uma leitura assim indica sensor com defeito ou dado inválido.

**Requisição (Inconsistente):**
```json
{
  "tipo": "M",
  "temperatura_ar_k": 308.0,
  "temperatura_processo_k": 303.0,
  "velocidade_rotacao_rpm": 1500.0,
  "torque_nm": 40.0,
  "desgaste_ferramenta_min": 50
}
```

**Resposta:**
```json
{
  "detail": "Inconsistência termodinâmica no processo: a temperatura do processo (303.0 K) não pode ser inferior à temperatura do ar ambiente (308.0 K)."
}
```

**Por que 400 e não 422?** O `422` é erro de formato: cada campo, sozinho, está errado (tipo, faixa, campo faltando). O `400` é erro de regra de negócio: cada campo está válido, mas a combinação deles não faz sentido.

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
* **Texto em Campo Numérico (`torque_nm: "42.8"`):**
  ```json
  {
    "detail": [
      {
        "type": "float_type",
        "loc": ["body", "torque_nm"],
        "msg": "Input should be a valid number",
        "input": "42.8"
      }
    ]
  }
  ```
* **Campo Desconhecido (`"umidade": 50`):**
  ```json
  {
    "detail": [
      {
        "type": "extra_forbidden",
        "loc": ["body", "umidade"],
        "msg": "Extra inputs are not permitted",
        "input": 50
      }
    ]
  }
  ```

---

#### Caso 5: Erros do Servidor (`HTTP 503` e `HTTP 500`)
* **`503 Service Unavailable`:** o modelo não está carregado na memória (ex.: arquivo `.joblib` ausente no startup).
* **`500 Internal Server Error`:** falha inesperada durante a inferência. O cliente recebe uma mensagem genérica; o detalhe técnico fica só no log do servidor.

---

## 7. Desempenho e Métricas do Modelo

Conforme detalhado no notebook `notebooks/treinamento_modelo.ipynb` e no relatório persistido `models/metricas.json`.

**Critério de escolha:** **Recall** da classe falha em primeiro lugar (deixar passar uma falha é o erro mais caro), **F1** como desempate e **ROC-AUC** como complemento.

### Comparativo na Validação (2.000 amostras, limiar 0,5):
| Algoritmo Candidato | Acurácia | Recall (Falha) | Precisão | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Regressão Logística | 81,80% | 82,35% | 13,73% | 0,2353 | 0,8776 |
| **Árvore de Decisão (`max_depth=5`) — escolhida** | **92,15%** | **91,18%** | **29,11%** | **0,4413** | 0,9286 |
| Random Forest (`max_depth=6`) | 91,90% | 79,41% | 26,73% | 0,4000 | **0,9561** |

### Validação Cruzada Estratificada (5 folds, no treino):
| Algoritmo Candidato | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| Regressão Logística | 0,784 ± 0,095 | 0,231 ± 0,018 | 0,884 ± 0,039 |
| **Árvore de Decisão (`max_depth=5`) — escolhida** | **0,863 ± 0,049** | **0,485 ± 0,050** | 0,902 ± 0,031 |
| Random Forest (`max_depth=6`) | 0,779 ± 0,059 | 0,455 ± 0,059 | **0,942 ± 0,028** |

**Por que a Árvore:** ela detecta mais falhas nas duas formas de validação. O Random Forest tem ROC-AUC maior — ordena melhor as máquinas por risco —, mas no limiar padrão de 0,5 deixa passar mais falhas. Além disso, a Árvore é interpretável: suas regras podem ser explicadas para a equipe de manutenção.

### Desempenho Final no Conjunto de Teste (Árvore de Decisão, avaliada uma única vez):
* **Acurácia:** 93,05%
* **Recall (Sensibilidade para Falhas):** 79,41%
* **Precisão:** 30,17%
* **F1-Score:** 0,4372
* **ROC-AUC:** 0,8762

**Matriz de Confusão no Teste (2.000 amostras):**
* Verdadeiros Negativos: 1.807
* Falsos Positivos: 125 (inspeções que se mostram desnecessárias)
* Falsos Negativos: 14 (falhas que passam sem alerta)
* Verdadeiros Positivos: 54 (54 das 68 falhas detectadas)

O recall caiu de 0,91 na validação para 0,79 no teste. Com apenas 68 falhas em cada conjunto, poucas máquinas mudam bastante o percentual — por isso a validação cruzada (0,86 ± 0,05) é a estimativa mais confiável do desempenho.

### Importância das Variáveis (Árvore de Decisão):
| Variável | Importância |
|---|---|
| `velocidade_rotacao_rpm` | 39,1% |
| `torque_nm` | 35,4% |
| `desgaste_ferramenta_min` | 20,0% |
| `temperatura_ar_k` | 3,6% |
| `temperatura_processo_k` | 0,8% |
| `tipo` (L, M, H) | 1,0% |

Rotação e torque juntos explicam quase 75% das decisões — o esforço mecânico da máquina —, seguidos pelo desgaste da ferramenta.

---

## 8. Estrutura de Branches Git (GitFlow)

O projeto segue o fluxo `feature → develop → main`. Nenhuma branch foi excluída:

* `main`: Código de produção estável e validado.
* `develop`: Branch de integração de todas as funcionalidades concluídas.
* `feat/pipeline-treinamento`: Notebook com carga, limpeza de leakage, split estratificado e pipeline.
* `feat/serializacao-modelo`: Exportação de `modelo_falha.joblib` e `metricas.json`.
* `feat/estrutura-api`: Arquitetura em camadas no pacote `app/`.
* `feat/lifespan-modelo`: Inicialização com carga única do modelo em RAM.
* `feat/schemas-pydantic`: Contratos de entrada e saída com `Field`.
* `feat/endpoints-predict-health`: Rotas `/health`, `/v1/predict` e códigos semânticos (200, 400, 422, 500, 503).
* `docs/readme`: Documentação técnica — primeira versão e revisão final com o modelo definitivo.
* `feat/validacao-cruzada`: Validação cruzada estratificada (5 folds) e correção do reuso do pré-processador entre candidatos.
* `feat/modelo-arvore-decisao`: Troca do modelo final para a Árvore de Decisão, importância das variáveis e reexportação dos artefatos.
* `fix/validacao-estrita-api`: Schema estrito, faixas do domínio de treino, inferência em threadpool, erro 500 sem detalhes internos e status degradado no `/health`.

---

## 9. Evidências e Gravação do Vídeo

* **Link do Repositório GitHub:** [https://github.com/ffegegsgabad/api-preditiva-aplicada-a-industria-40](https://github.com/ffegegsgabad/api-preditiva-aplicada-a-industria-40)
* **Link da Apresentação em Vídeo:** `https://drive.google.com/file/d/SEU_ID_DO_VIDEO/view?usp=sharing` *(Configurado em modo leitor para qualquer pessoa com o link)*

### Roteiro Seguido no Vídeo (Duração ≤ 10 minutos):
1. **[0:00–0:40] Objetivo do Sistema:** Problema de paradas não programadas e o que a API entrega.
2. **[0:40–1:30] Origem do Modelo e Dados:** Projeto Final do Módulo 1 e o que os dados exigiram.
3. **[1:30–4:00] Demonstração no `/docs`:** `/health`, predição normal e de risco, erros 422 e 400 e o log do modelo carregado uma única vez no startup.
4. **[4:00–4:45] Instruções de Execução:** `venv`, `requirements.txt` e comando `uvicorn`.
5. **[4:45–5:45] Organização Prévia:** Plano de ação, cronograma e quadro de tarefas antes do código.
6. **[5:45–6:45] Branches no GitHub:** Fluxo `feature → develop → main` e objetivo de cada branch.
7. **[6:45–8:45] Argumentação Técnica:** Pipeline unificado, SMOTE só no treino, validação cruzada, escolha da Árvore pelo recall, Lifespan e validação estrita.
8. **[8:45–9:30] Oportunidades de Melhoria:** Ajuste do limiar de decisão, testes automatizados, Docker e registro de experimentos com MLflow.
9. **[9:30–9:45] Encerramento.**

### Próximos Passos (Oportunidades de Melhoria)
* **Limiar de decisão:** testar o Random Forest com limiar escolhido na validação (ex.: 0,35), que aumenta o recall mantendo o ROC-AUC mais alto.
* **Ajuste de hiperparâmetros:** `GridSearchCV` com validação cruzada para `max_depth` e `min_samples_leaf`.
* **Testes automatizados:** `pytest` + `TestClient` cobrindo os códigos 200, 400, 422, 500 e 503.
* **Docker:** empacotar a API em um contêiner para rodar igual em qualquer máquina.
* **MLflow:** registrar experimentos, métricas e versões do modelo.
* **Monitoramento de drift:** acompanhar se os dados dos sensores em produção se afastam da faixa do treino.
