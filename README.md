# 📊 ERP-OFDI Model - Investimentos Chineses

## 🌟 Sobre o Projeto
Pipeline para análise de investimentos chineses no exterior (OFDI) e impacto da política **ERP (Ecological Redlines Policy)** implementada em 2017.

**Fonte dos dados**: 🎓 Dr. Derek Scissors, PhD - Stanford University / American Enterprise Institute

**Documentação:** Consulte `docs.pdf` para detalhes de implementação, metodologia e especificações técnicas.

---

## 🔄 Estrutura do Projeto

### 📁 `data/` → 📥 **ETAPA 1: Carga de Dados**
**Arquivos**: `data.csv` (transações 2005-2024) + `naturalearth_lowres.zip` (mapas)
- **Processo**: Carrega CSV → Limpa dados → Converte colunas numéricas → Filtra anos válidos
- **Resultado**: DataFrame limpo com 20+ colunas processadas

### 📁 `code/` → 🚀 **Execução Principal**
**Arquivos**: `main.py` (orquestração) + `requirements.txt` (dependências)
- **Comando**: `python .\code\main.py`
- **Função**: Executa todas as 11 etapas automaticamente

### 📁 `results/` → 📊 **Saídas Geradas**

#### 🎨 `r_style_plots/` → 🎨 **ETAPA 3: Visualizações**
**Gráficos gerados**:
- `r_style_world_maps.png` → Mapas mundiais coloridos por investimento
- `r_style_investment_by_sector.png` → Evolução setorial em 3 fases
- `r_style_greenfield_donuts.png` → Donut charts Greenfield vs M&A
- `python_style_boxplot_greenfield.png` → Análise estatística comparativa

#### 📁`eda/` → 📊 **ETAPA 4: Análise Exploratória**
**Arquivos**:
- `summary_statistics.csv` → Médias, medianas, desvios padrão
- `eda_target_distribution_log_vs_raw.png` → Distribuição dos valores
- `eda_total_investment_over_time.png` → Série temporal completa
- `eda_h1_sector_shift_summary.csv` → Mudanças setoriais pós-ERP

#### 📁 `phase_analysis/` → 🔬 **ETAPA 5: Análise por Fase GG**
**Arquivos**:
- `summary_interaction_ols_by_phase.csv` → Efeitos de interação estatística
- `summary_mediation_ols_by_phase.csv` → Efeitos de mediação causal
- **Fases**: GG 1.0 (2005-2012) | GG 2.0 (2013-2016) | GG 3.0+ERP (2017-2024)

#### 📁 `models/regression/` → 🤖 **ETAPA 6: Modelos Preditivos**
**Arquivos**:
- `regression_models_summary_otimizada.csv` → Performance XGBoost vs LightGBM
- **Métricas**: MAPE (Mean Absolute Percentage Error) | RMSE logarítmico
- **Target**: Prever `log_Valor_USD` (valor do investimento)

#### 📁 `models/classification/` → 🎯 **ETAPA 7: Classificação Binária**
**Arquivos**:
- `classification_models_summary_otimizada.csv` → AUC-ROC | Acurácia
- **Target**: `Alvo_Adaptativo` (1=alto valor, 0=baixo valor)
- **Modelos**: XGBoost, LightGBM, CatBoost com otimização de hiperparâmetros

#### 📁 `models/timeseries/` → ⏱️ **ETAPA 8: Séries Temporais**
**Arquivos**:
- `arima_forecast.png` → Previsões ARIMA
- `prophet_forecast.png` → Previsões Facebook Prophet
- `prophet_forecast_data.csv` → Dados de projeção 5 anos
- **Objetivo**: Prever tendências futuras de investimento

#### 📁 `models/causal/` → 🔬 **ETAPA 9: Análise Causal**
**Arquivos**:
- `dml_summary_att.csv` → Efeito causal da ERP usando Double Machine Learning
- `markov_prob.png` → Probabilidades de mudança de regime Markov
- `markov_switching_summary.txt` → Detalhes do modelo de regimes
- **Pergunta**: "A política ERP realmente causou mudanças nos investimentos?"

#### 📁 `models/shap/` → 🔍 **ETAPA 10: Interpretabilidade**
**Arquivos**:
- `shap_summary_plot.png` → Impacto de cada variável no modelo
- `shap_importance_plot.png` → Importância relativa das features
- **Variáveis chave**: `valor_roll_mean_3` (média móvel) | `Sector` | `Region`

---

## 🛠️ **ETAPA 2: Engenharia de Features**
**Features criadas**:
- `valor_roll_mean_2/3` → Médias móveis de 2/3 períodos
- `post_ERP` → Flag pós-2017 (dummy policy)
- `Fase_GG` → Categorização em 3 fases "Going Global"
- `Alvo_Binario` → Classificação alto/baixo valor
- `policy_interaction` → Interação tempo × política

---

## 📋 **Resumo das 11 Etapas Automáticas**

1. **📥 Carga Dados** → Limpeza e validação
2. **🛠️ Engenharia Features** → Criação variáveis ML
3. **🎨 Visualizações R-style** → 10+ gráficos profissionais
4. **📊 EDA** → Análise exploratória estatística
5. **🔬 Análise Fase GG** → Modelos OLS por período
6. **🤖 Regressão** → Previsão valor investimento (XGBoost/LightGBM)
7. **🎯 Classificação** → Identificação alto valor (AUC-ROC)
8. **⏱️ Séries Temporais** → ARIMA & Prophet
9. **🔬 Modelos Causais** → Double ML & Markov Switching
10. **🔍 SHAP** → Interpretabilidade do modelo
11. **💾 Consolidação** → Salva todos resultados

---

## 🚀 **Execução**

```bash
# 1. Instalar dependências
python -m pip install -r requirements.txt

# 2. Executar pipeline
python .\code\main.py

# 3. Ver resultados na pasta /results/