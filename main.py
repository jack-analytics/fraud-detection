# ==============================================================================
# PIPELINE CENTRAL DE DETECÇÃO DE FRAUDE (EXECUÇÃO DE PRODUÇÃO)
# Autor: Jackelinne Rodrigues
# Propósito: Executar a carga de dados e avaliar os três modelos implementados
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report

def executar_pipeline():
    print("\n" + "="*60)
    print(" INICIANDO PIPELINE DE INTELIGÊNCIA DE MERCADO & RISCO ")
    print("="*60)

    # 1. CARGA DE DADOS
    print("\n[Passo 1/5] Carregando dados históricos da API do TensorFlow...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    print(f"-> Base carregada com sucesso! Volume total: {df.shape[0]} transações.")

    # Preparação das variáveis comuns
    X = df.drop(columns=['Class'])
    y = df['Class']
    contamination_rate = df['Class'].value_counts()[1] / len(df)

    # 2. AVALIAÇÃO DO ISOLATION FOREST
    print("\n[Passo 2/5] Rodando algoritmo Isolation Forest (Não Supervisionado)...")
    clf_if = IsolationForest(contamination=contamination_rate, random_state=42)
    pred_if = clf_if.fit_predict(X)
    y_pred_if = np.where(pred_if == -1, 1, 0)
    
    print("\n--- Desempenho do Isolation Forest ---")
    print(classification_report(y, y_pred_if))

    # 3. AVALIAÇÃO DO LOCAL OUTLIER FACTOR (LOF)
    print("\n[Passo 3/5] Rodando algoritmo Local Outlier Factor (Não Supervisionado)...")
    clf_lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate)
    pred_lof = clf_lof.fit_predict(X)
    y_pred_lof = np.where(pred_lof == -1, 1, 0)
    
    print("\n--- Desempenho do Local Outlier Factor ---")
    print(classification_report(y, y_pred_lof))

    # 4. AVALIAÇÃO DA RANDOM FOREST
    print("\n[Passo 4/5] Separando dados e treinando Random Forest (Supervisionado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    print("\n--- Desempenho da Random Forest (Dados Inéditos de Teste) ---")
    print(classification_report(y_test, y_pred_rf))

    print("\n" + "="*60)
    print(" PIPELINE EXECUTADO COM SUCESSO DE PONTA A PONTA! ")
    print("="*60 + "\n")

if __name__ == "__main__":
    executar_pipeline()