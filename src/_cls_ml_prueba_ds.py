"""
Created By Emerson Aguilar Cruz
"""

import pandas as pd
import os
import numpy as np
import warnings
from pathlib import Path
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
path_home = str(Path.home()) 
current_folder = os.path.dirname(os.path.abspath(__file__)) 
project_folder = os.path.dirname(current_folder) 
print(project_folder)

class MlPruebaDS:

    def __init__(self, archivo=None):

        self.archivo = os.path.join(project_folder, 'data', 'PruebaDS.xlsx')
        self.df = None

    def load_datos(self):
        self.df = pd.read_excel(self.archivo)
        print("Archivo cargado correctamente")
        return self.df

    def data_cleaning(self):
        if self.df is None:
            raise ValueError(f"No hay datos en {self.archivo}")

        if self.df['mes'].dtype == 'object':
            self.df['mes'] = pd.to_datetime(
                self.df['mes'],
                format='%Y-%m',
                errors='coerce'
            )

        self.df = (
            self.df
            .sort_values(['identificacion', 'mes'])
            .reset_index(drop=True)
        )

        if 'genero' in self.df.columns:
            self.df['genero'] = (
                self.df['genero']
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({
                    'HOMBRE': 'M',
                    'H': 'M',
                    'MASCULINO': 'M',
                    'MUJER': 'F',
                    'FEMENINO': 'F'
                })
            )

            self.df['genero'] = self.df['genero'].where(
                self.df['genero'].isin(['M', 'F']),
                'sin_info'
            )

        if 'antiguedad_deuda' in self.df.columns:
            self.df['antiguedad_deuda'] = pd.to_datetime(
                self.df['antiguedad_deuda'],
                errors='coerce'
            )

            fecha_hoy = pd.Timestamp.today().normalize()

            self.df['antiguedad_deuda_dias'] = (
                fecha_hoy - self.df['antiguedad_deuda']
            ).dt.days

            self.df.loc[
                self.df['antiguedad_deuda_dias'] < 0,
                'antiguedad_deuda_dias'
            ] = 0

        var_categ = self.df.select_dtypes(include='object').columns
        self.df[var_categ] = self.df[var_categ].fillna('sin_info')

        print("Datos limpiados y preparados")
        return self.df

    def crear_features_nulos(self):

        self.df['sin_historial_pago'] = self.df['meses_desde_ultimo_pago'].isna().astype(int)
        self.df['sin_antiguedad_conocida'] = self.df['antiguedad_deuda'].isna().astype(int)

        self.df['meses_desde_ultimo_pago'] = self.df['meses_desde_ultimo_pago'].fillna(-1)
        self.df['antiguedad_deuda'] = self.df['antiguedad_deuda'].fillna(-1)

        print(" Features de información faltante creados")
        return self.df

    def crear_features_negocio(self):
        
        self.df['ratio_mora_saldo'] = self.df['dias_mora'] / (self.df['saldo_capital'] + 1)
        
        self.df['contacto_total'] = (
            self.df['contacto_mes_actual'] + 
            self.df['contacto_mes_anterior'] + 
            self.df['contacto_ultimos_6meses']
        )
        
        self.df['pago_reciente'] = (
            (self.df['pago_mes_anterior'] == 1) | 
            (self.df['sin_pago_previo'] == 0)
        ).astype(int)
        
        self.df['saldo_x_mora'] = self.df['saldo_capital'] * self.df['dias_mora']
        
        print(" Features de negocio creados")
        return self.df

    def crear_intervalos(self):

        self.df['saldo_capital_rango'] = pd.cut(
            self.df['saldo_capital'],
            bins=[0, 500000, 2000000, 10000000, np.inf],
            labels=['bajo', 'medio', 'alto', 'muy_alto']
        )

        self.df['dias_mora_categoria'] = pd.cut(
            self.df['dias_mora'],
            bins=[-1, 0, 30, 90, np.inf],
            labels=['al_dia', 'mora_leve', 'mora_grave', 'critico']
        )

        self.df['antiguedad_deuda_rango'] = pd.cut(
            self.df['antiguedad_deuda_dias'],
            bins=[0, 30, 90, 180, 365, 730, 1500, np.inf],
            labels=[
                '0-30d',
                '31-90d',
                '91-180d',
                '181-365d',
                '1-2a',
                '2-4a',
                '4a+'
            ],
            include_lowest=True
        )

        print(" Intervalos aplicados")
        return self.df

    def crear_features_temporales(self):
        
        print("Creando desfase temporal")
        
        for lag in [1, 2, 3]:
            self.df[f'pago_lag{lag}'] = self.df.groupby('identificacion')['pago'].shift(lag)
            self.df[f'saldo_capital_lag{lag}'] = self.df.groupby('identificacion')['saldo_capital'].shift(lag)
            self.df[f'dias_mora_lag{lag}'] = self.df.groupby('identificacion')['dias_mora'].shift(lag)
        
        self.df['contacto_total_lag1'] = self.df.groupby('identificacion')['contacto_total'].shift(1)
        
        print(" Creando rolling features...")
        self.df['saldo_avg_3m'] = self.df.groupby('identificacion')['saldo_capital'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        
        self.df['mora_avg_3m'] = self.df.groupby('identificacion')['dias_mora'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        
        self.df['total_pagos_historicos'] = self.df.groupby('identificacion')['pago'].shift(1).fillna(0).groupby(self.df['identificacion']).cumsum()
        
        print(" Creando deltas...")
        self.df['cambio_saldo'] = self.df['saldo_capital'] - self.df['saldo_capital_lag1']
        self.df['cambio_mora'] = self.df['dias_mora'] - self.df['dias_mora_lag1']
        self.df['mejora_mora'] = (self.df['cambio_mora'] < 0).astype(int)
        
        print(" Creando features de contexto...")
        self.df['es_cliente_nuevo'] = (self.df.groupby('identificacion').cumcount() == 0).astype(int)
        self.df['meses_en_cartera'] = self.df.groupby('identificacion').cumcount() + 1
        
        self.df['ratio_pagos_exitosos'] = self.df['total_pagos_historicos'] / (self.df['meses_en_cartera'] - 1 + 0.001)
        
        def calcular_meses_sin_pagar(grupo):
            pagos_pasados = grupo.shift(1).fillna(0)
            return (pagos_pasados == 0).groupby((pagos_pasados != 0).cumsum()).cumsum()
        
        self.df['meses_sin_pagar_consecutivos'] = self.df.groupby('identificacion')['pago'].transform(calcular_meses_sin_pagar)
        
        print(" Creando features del mes...")
        if pd.api.types.is_datetime64_any_dtype(self.df['mes']):
            self.df['mes_numero'] = self.df['mes'].dt.month
            self.df['trimestre'] = self.df['mes'].dt.quarter
        else:
            self.df['mes_numero'] = self.df['mes'].astype(str).str[-2:].astype(int)
            self.df['trimestre'] = ((self.df['mes_numero'] - 1) // 3) + 1
        
        self.df['es_mes_alto'] = self.df['mes_numero'].isin([8, 9, 10]).astype(int)
        self.df['es_fin_anio'] = self.df['mes_numero'].isin([11, 12]).astype(int)
        
        lag_cols = [col for col in self.df.columns if 'lag' in col or 'avg_' in col]
        self.df[lag_cols] = self.df[lag_cols].fillna(0)
        
        print(" Features temporales creados")
        return self.df

    def split_temporal(self, train_end='202509', val_end='202510'):
        
        if pd.api.types.is_datetime64_any_dtype(self.df['mes']):
            self.df['mes_int'] = self.df['mes'].dt.year * 100 + self.df['mes'].dt.month
            self.df['mes_str'] = self.df['mes'].dt.strftime('%Y%m')
        else:
            self.df['mes_str'] = self.df['mes'].astype(str).str.replace('-', '')
            self.df['mes_int'] = pd.to_numeric(self.df['mes_str'], errors='coerce')
        
        self.df = self.df.dropna(subset=['mes_int'])
        self.df['mes_int'] = self.df['mes_int'].astype(int)
        
        train_end_int = int(train_end)
        val_end_int = int(val_end)
        
        train = self.df[self.df['mes_int'] <= train_end_int].copy()
        val = self.df[(self.df['mes_int'] > train_end_int) & (self.df['mes_int'] <= val_end_int)].copy()
        test = self.df[self.df['mes_int'] > val_end_int].copy()
        
        print(f" Split temporal:")
        print(f" TRAIN: {len(train)} registros | Tasa pago: {train['pago'].mean()*100:.2f}%")
        print(f" VAL:   {len(val)} registros | Tasa pago: {val['pago'].mean()*100:.2f}%")
        print(f" TEST:  {len(test)} registros | Tasa pago: {test['pago'].mean()*100:.2f}%")
        
        return train, val, test

    def aplicar_encoding(self, train, val, test):
        
        exclude_before = ['identificacion', 'mes', 'mes_str', 'mes_int', 'pago']
        
        cat_bajas = ['genero', 'rango_edad_probable', 'saldo_capital_rango', 'dias_mora_categoria', 'antiguedad_deuda_rango']
        cat_altas = ['departamento', 'banco', 'tipo_documento']
        
        train = pd.get_dummies(train, columns=cat_bajas, drop_first=True, dtype=int)
        val = pd.get_dummies(val, columns=cat_bajas, drop_first=True, dtype=int)
        test = pd.get_dummies(test, columns=cat_bajas, drop_first=True, dtype=int)
        
        train_cols = train.columns
        for col in train_cols:
            if col not in val.columns:
                val[col] = 0
            if col not in test.columns:
                test[col] = 0
        
        val = val[train_cols]
        test = test[train_cols]
        
        encoder = TargetEncoder(cols=cat_altas)
        train[cat_altas] = encoder.fit_transform(train[cat_altas], train['pago'])
        val[cat_altas] = encoder.transform(val[cat_altas])
        test[cat_altas] = encoder.transform(test[cat_altas])
        
        for col in ['mes', 'mes_str', 'mes_int']:
            if col not in train.columns and col in exclude_before:
                pass
        
        print(" Encoding aplicado")
        
        return train, val, test, encoder

    def preparar_X_y(self, train, val, test, target='pago'):
        
        exclude_cols = ['identificacion', 'mes', 'mes_str', 'mes_int', target]
        
        datetime_cols = train.select_dtypes(include=['datetime64']).columns.tolist()
        exclude_cols.extend(datetime_cols)
        
        exclude_cols = list(set(exclude_cols))
        
        feature_cols = [col for col in train.columns if col not in exclude_cols]
        
        X_train = train[feature_cols].copy()
        y_train = train[target].copy()
        
        X_val = val[feature_cols].copy()
        y_val = val[target].copy()
        
        X_test = test[feature_cols].copy()
        y_test = test[target].copy()
        
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f" Features preparadas: {len(feature_cols)} variables")
        print(f" Train: {X_train.shape}")
        print(f" Val:   {X_val.shape}")
        print(f" Test:  {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    def aplicar_balanceo(self, X_train, y_train, strategy=0.3):
        
        print(f" Antes de SMOTE:")
        print(f" Clase 0: {(y_train == 0).sum()}")
        print(f" Clase 1: {(y_train == 1).sum()}")
        
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f" Después de SMOTE:")
        print(f" Clase 0: {(y_train_balanced == 0).sum()}")
        print(f" Clase 1: {(y_train_balanced == 1).sum()}")
        
        return X_train_balanced, y_train_balanced

    def entrenar_baseline(self, X_train, y_train, X_val, y_val):
        print(" Entrenando Regresión Logística")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(X_train_scaled, y_train)
        
        y_pred_proba_val = lr.predict_proba(X_val_scaled)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred_proba_val)
        
        print(f" Baseline AUC en Validación: {auc:.4f}")
        
        return lr, scaler, auc

    def entrenar_xgboost(self, X_train, y_train, X_val, y_val):
        print("Entrenando XGBoost")
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20
        )
        
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        y_pred_proba_val = xgb.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred_proba_val)
        
        print(f" XGBoost AUC en Validación: {auc:.4f}")
        
        return xgb, auc

    def evaluar_modelo_completo(self, modelo, X_test, y_test, test_df, threshold=None):

        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        threshold = np.percentile(y_pred_proba, 90)
        y_pred = (y_pred_proba >= threshold).astype(int)

        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"AUC-ROC:    {auc:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-Score:   {f1:.4f}")
        print(f"Accuracy:   {accuracy:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print(" Matriz de Confusión:")
        print(cm)

        print("\n Reporte de Clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\n Estabilidad Temporal (AUC por mes):")

        test_df = test_df.copy()
        test_df['pred_proba'] = y_pred_proba

        meses_unicos = sorted(test_df['mes_str'].unique())
        aucs_por_mes = {}

        for mes in meses_unicos:
            mes_data = test_df[test_df['mes_str'] == mes]
            if len(mes_data) > 0 and mes_data['pago'].nunique() > 1:
                auc_mes = roc_auc_score(mes_data['pago'], mes_data['pred_proba'])
                aucs_por_mes[mes] = auc_mes
                print(f"  Mes {mes}: AUC = {auc_mes:.4f} | Registros: {len(mes_data)}")

        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'aucs_por_mes': aucs_por_mes
        }

    def optimizar_threshold(self, y_true, y_pred_proba, metrica='f1'):

        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)

            if metrica == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metrica == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metrica == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)

            scores.append(score)

        optimal_idx = np.argmax(scores)

        return thresholds[optimal_idx], scores[optimal_idx]

    def mostrar_feature_importance(self, modelo, feature_cols, top_n=20):

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': modelo.feature_importances_
        }).sort_values('importance', ascending=False)

        print(importance.head(top_n))
        return importance

    def analizar_clientes_alta_probabilidad(self, test_df, y_pred_proba, top_percentil=10):

        df = test_df.copy()
        df['pred_proba'] = y_pred_proba

        umbral = np.percentile(y_pred_proba, 100 - top_percentil)
        clientes_top = df[df['pred_proba'] >= umbral]

        print(f" Total clientes: {len(clientes_top)}")
        print(f" Tasa pago real: {clientes_top['pago'].mean()*100:.2f}%")

        return clientes_top

    def exportar_predicciones(self, test_df, y_pred_proba, y_pred, filename='predicciones_test.csv'):

        output = test_df[['identificacion', 'mes', 'pago']].copy()
        output['probabilidad_pago'] = y_pred_proba
        output['prediccion'] = y_pred

        output = output.sort_values('probabilidad_pago', ascending=False)

        ruta_salida = os.path.join(project_folder, 'data', filename)
        output.to_csv(ruta_salida, index=False)

        print(f"Predicciones exportadas a {ruta_salida}")
        return output


if __name__ == "__main__":

    ds = MlPruebaDS()

    df = ds.load_datos()
    df = ds.data_cleaning()
    df = ds.crear_features_nulos()
    df = ds.crear_features_negocio()
    df = ds.crear_intervalos()
    df = ds.crear_features_temporales()

    train, val, test = ds.split_temporal(
        train_end='202509',
        val_end='202510'
    )

    train, val, test, encoder = ds.aplicar_encoding(
        train, val, test
    )

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = ds.preparar_X_y(
        train, val, test, target='pago'
    )

    X_train_bal, y_train_bal = ds.aplicar_balanceo(
        X_train, y_train, strategy=0.3
    )

    lr_model, scaler_lr, auc_lr = ds.entrenar_baseline(
        X_train_bal,
        y_train_bal,
        X_val,
        y_val
    )

    print(f"\n AUC Validación Regresión Logística (Baseline): {auc_lr:.4f}")

    xgb_model, auc_xgb = ds.entrenar_xgboost(
        X_train_bal,
        y_train_bal,
        X_val,
        y_val
    )

    print(f"\n AUC Validación XGBoost: {auc_xgb:.4f}")

    y_val_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

    optimal_thresh_f1, best_f1 = ds.optimizar_threshold(
        y_val,
        y_val_proba_xgb,
        metrica='f1'
    )

    print(f"\n Threshold óptimo (F1): {optimal_thresh_f1:.2f}")
    print(f" Mejor F1 en validación: {best_f1:.4f}")

    resultados_xgb = ds.evaluar_modelo_completo(
        xgb_model,
        X_test,
        y_test,
        test,
        threshold=optimal_thresh_f1
    )

    importance = ds.mostrar_feature_importance(
        xgb_model,
        feature_cols,
        top_n=20
    )

    clientes_top = ds.analizar_clientes_alta_probabilidad(
        test,
        resultados_xgb['y_pred_proba'],
        top_percentil=10
    )

    predicciones = ds.exportar_predicciones(
        test,
        resultados_xgb['y_pred_proba'],
        resultados_xgb['y_pred'],
        filename='predicciones_test.csv'
    )

    print(f" AUC Regresión Logística (Baseline): {auc_lr:.4f}")
    print(f" AUC XGBoost (Final):               {auc_xgb:.4f}")

    print("\n Pipeline ejecutado correctamente")