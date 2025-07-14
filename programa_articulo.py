#!/usr/bin/env python3
"""
Gráficos Individuales para Artículo Científico de PM2.5
Cada gráfico se puede usar independientemente en el paper

Autor: Carlos Paolo Fontanil Barrera
Fecha: Enero 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, Ridge, ElasticNet, LinearRegression, lasso_path
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo profesional para artículo científico
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

class GraficosArticuloCientifico:
    """
    Clase para generar gráficos individuales para artículo científico
    """
    
    def __init__(self, archivo_datos="datos_pm25_lima.csv"):
        """Inicializa con los datos"""
        self.archivo_datos = archivo_datos
        self.df = None
        self.promedios_mensuales = None
        self.estandares = {
            'OMS_24H': 15,
            'ECA_PERU_24H': 50
        }
        self.cargar_datos()
    
    def cargar_datos(self):
        """Carga los datos"""
        print("📊 Cargando datos para gráficos...")
        try:
            self.df = pd.read_csv(self.archivo_datos, parse_dates=["FECHA"])
            self.df.dropna(subset=["PM2_5"], inplace=True)
            self.df["MES"] = self.df["FECHA"].dt.to_period("M")
            self.df["ANIO"] = self.df["FECHA"].dt.year
            self.df["NUM_MES"] = self.df["FECHA"].dt.month
            self.promedios_mensuales = self.df.groupby("MES")["PM2_5"].mean().reset_index()
            print(f"✅ Datos cargados: {len(self.df):,} registros")
            return True
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False
    
    def grafico_1_serie_temporal(self):
        """GRÁFICO 1: Serie Temporal de PM2.5"""
        print("📈 Creando Gráfico 1: Serie Temporal...")
        
        plt.figure(figsize=(12, 6))
        
        fechas = range(len(self.promedios_mensuales))
        valores = self.promedios_mensuales['PM2_5']
        
        plt.plot(fechas, valores, color='#2E8B57', linewidth=2.5, 
                marker='o', markersize=5, alpha=0.9, label='PM2.5 Mensual')
        
        plt.axhline(y=self.estandares['OMS_24H'], color='#DC143C', 
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'WHO Guidelines (24h): {self.estandares["OMS_24H"]} μg/m³')
        plt.axhline(y=self.estandares['ECA_PERU_24H'], color='#FF8C00', 
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Peru ECA (24h): {self.estandares["ECA_PERU_24H"]} μg/m³')
        
        z = np.polyfit(fechas, valores, 1)
        p = np.poly1d(z)
        plt.plot(fechas, p(fechas), color='red', linestyle=':', 
                linewidth=2, alpha=0.7, label='Trend Line')
        
        plt.title('Monthly Average PM2.5 Concentrations in Lima Metropolitan Area\n(2015-2024)', 
                 fontweight='bold', pad=20)
        plt.xlabel('Time Period (Monthly)', fontweight='bold')
        plt.ylabel('PM2.5 Concentration (μg/m³)', fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.legend(loc='upper right', framealpha=0.9)
        
        promedio = valores.mean()
        plt.text(0.02, 0.98, f'Overall Mean: {promedio:.1f} μg/m³\nData Points: {len(valores)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        step = max(1, len(fechas) // 8)
        plt.xticks(range(0, len(fechas), step), 
                  [str(self.promedios_mensuales['MES'].iloc[i])[:7] 
                   for i in range(0, len(fechas), step)], rotation=45)
        
        plt.tight_layout()
        plt.savefig('Figura_1_Serie_Temporal_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_1_Serie_Temporal_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 1 guardada: Figura_1_Serie_Temporal_PM25.png/.pdf")
    
    def grafico_2_patron_estacional(self):
        """GRÁFICO 2: Patrón Estacional"""
        print("📈 Creando Gráfico 2: Patrón Estacional...")
        
        plt.figure(figsize=(10, 6))
        
        patron_mensual = self.df.groupby("NUM_MES")["PM2_5"].mean()
        error_std = self.df.groupby("NUM_MES")["PM2_5"].std()
        
        meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        bars = plt.bar(range(1, 13), patron_mensual.values, 
                      color='#4682B4', alpha=0.8, edgecolor='black', linewidth=1,
                      yerr=error_std.values, capsize=5, error_kw={'color': 'black', 'alpha': 0.7})
        
        promedio_anual = patron_mensual.mean()
        plt.axhline(y=promedio_anual, color='red', linestyle=':', 
                   linewidth=2, alpha=0.8, label=f'Annual Average: {promedio_anual:.1f} μg/m³')
        
        plt.title('Seasonal Pattern of PM2.5 Concentrations\nLima Metropolitan Area (2015-2024)', 
                 fontweight='bold', pad=20)
        plt.xlabel('Month', fontweight='bold')
        plt.ylabel('PM2.5 Concentration (μg/m³)', fontweight='bold')
        plt.xticks(range(1, 13), meses)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        for i, (bar, valor) in enumerate(zip(bars, patron_mensual.values)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{valor:.1f}', ha='center', va='bottom', fontweight='bold')
        
        mes_max = patron_mensual.idxmax()
        mes_min = patron_mensual.idxmin()
        plt.text(0.02, 0.98, f'Highest: {meses[mes_max-1]} ({patron_mensual[mes_max]:.1f} μg/m³)\n'
                            f'Lowest: {meses[mes_min-1]} ({patron_mensual[mes_min]:.1f} μg/m³)', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('Figura_2_Patron_Estacional_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_2_Patron_Estacional_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 2 guardada: Figura_2_Patron_Estacional_PM25.png/.pdf")
    
    def grafico_3_percentiles(self):
        """GRÁFICO 3: Distribución de Percentiles"""
        print("📈 Creando Gráfico 3: Percentiles...")
        
        plt.figure(figsize=(10, 6))
        
        datos_pm25 = self.df["PM2_5"]
        percentiles = [10, 25, 50, 75, 90, 95]
        valores_perc = [np.percentile(datos_pm25, p) for p in percentiles]
        etiquetas = [f'P{p}' for p in percentiles]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(percentiles)))
        
        bars = plt.bar(etiquetas, valores_perc, color=colors, 
                      edgecolor='black', linewidth=1, alpha=0.8)
        
        plt.axhline(y=self.estandares['OMS_24H'], color='red', 
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'WHO Guidelines: {self.estandares["OMS_24H"]} μg/m³')
        plt.axhline(y=self.estandares['ECA_PERU_24H'], color='orange', 
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Peru ECA: {self.estandares["ECA_PERU_24H"]} μg/m³')
        
        plt.title('PM2.5 Concentration Percentiles Distribution\nLima Metropolitan Area (2015-2024)', 
                 fontweight='bold', pad=20)
        plt.xlabel('Percentiles', fontweight='bold')
        plt.ylabel('PM2.5 Concentration (μg/m³)', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        for bar, valor in zip(bars, valores_perc):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{valor:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.text(0.02, 0.98, f'Total Observations: {len(datos_pm25):,}\n'
                            f'Mean: {datos_pm25.mean():.1f} μg/m³\n'
                            f'Std Dev: {datos_pm25.std():.1f} μg/m³', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('Figura_3_Percentiles_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_3_Percentiles_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 3 guardada: Figura_3_Percentiles_PM25.png/.pdf")
    
    def grafico_4_comparacion_temporal(self):
        """GRÁFICO 4: Comparación Pre/Post 2020"""
        print("📈 Creando Gráfico 4: Comparación Temporal...")
        
        plt.figure(figsize=(10, 6))
        
        pre_2020 = self.df[self.df["ANIO"] < 2020]["PM2_5"]
        post_2020 = self.df[self.df["ANIO"] >= 2020]["PM2_5"]
        
        periodos = ['Pre-2020\n(2015-2019)', 'Post-2020\n(2020-2024)']
        medias = [pre_2020.mean(), post_2020.mean()]
        errores = [pre_2020.std()/np.sqrt(len(pre_2020)), 
                  post_2020.std()/np.sqrt(len(post_2020))]
        
        t_stat, p_valor = stats.ttest_ind(pre_2020.dropna(), post_2020.dropna())
        
        colors = ['#87CEEB', '#F08080']
        
        bars = plt.bar(periodos, medias, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5,
                      yerr=errores, capsize=8, error_kw={'color': 'black', 'linewidth': 2})
        
        plt.title('Temporal Comparison of PM2.5 Concentrations\nPre vs Post-2020 Period', 
                 fontweight='bold', pad=20)
        plt.ylabel('PM2.5 Concentration (μg/m³)', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, valor) in enumerate(zip(bars, medias)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + errores[i] + 0.5,
                    f'{valor:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        diferencia = post_2020.mean() - pre_2020.mean()
        significancia = "***" if p_valor < 0.001 else "**" if p_valor < 0.01 else "*" if p_valor < 0.05 else "ns"
        
        y_max = max(medias) + max(errores) + 2
        plt.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        plt.plot([0, 0], [y_max-0.5, y_max], 'k-', linewidth=1.5)
        plt.plot([1, 1], [y_max-0.5, y_max], 'k-', linewidth=1.5)
        plt.text(0.5, y_max + 0.5, f'{significancia}', ha='center', fontweight='bold', fontsize=14)
        
        plt.text(0.02, 0.98, f'Difference: {diferencia:+.1f} μg/m³\n'
                            f't-statistic: {t_stat:.3f}\n'
                            f'p-value: {p_valor:.4f}\n'
                            f'Effect size: {abs(diferencia)/np.sqrt((pre_2020.var()+post_2020.var())/2):.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('Figura_4_Comparacion_Temporal_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_4_Comparacion_Temporal_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 4 guardada: Figura_4_Comparacion_Temporal_PM25.png/.pdf")
    
    def grafico_5_superaciones(self):
        """GRÁFICO 5: Porcentaje de Superaciones"""
        print("📈 Creando Gráfico 5: Superaciones...")
        
        plt.figure(figsize=(10, 6))
        
        datos_pm25 = self.df["PM2_5"]
        superaciones = {}
        
        for nombre, umbral in self.estandares.items():
            superacion = (datos_pm25 > umbral).sum()
            porcentaje = (superacion / len(datos_pm25)) * 100
            superaciones[nombre] = porcentaje
        
        etiquetas = ['WHO Guidelines\n(24h limit)', 'Peru ECA\n(24h limit)']
        valores = [superaciones['OMS_24H'], superaciones['ECA_PERU_24H']]
        colors = ['#DC143C', '#FF8C00']
        
        bars = plt.bar(etiquetas, valores, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        plt.title('Exceedance Frequency of Air Quality Standards\nLima Metropolitan Area (2015-2024)', 
                 fontweight='bold', pad=20)
        plt.ylabel('Percentage of Exceedances (%)', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, valor in zip(bars, valores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.text(0.02, 0.98, f'Total Observations: {len(datos_pm25):,}\n'
                            f'WHO Exceedances: {int(len(datos_pm25)*valores[0]/100):,} records\n'
                            f'Peru ECA Exceedances: {int(len(datos_pm25)*valores[1]/100):,} records\n'
                            f'Analysis Period: 2015-2024', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('Figura_5_Superaciones_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_5_Superaciones_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 5 guardada: Figura_5_Superaciones_PM25.png/.pdf")
    
    def grafico_6_caracteristicas_dispersas(self):
        """GRÁFICO 6: Características Seleccionadas (Optimización Dispersa)"""
        print("📈 Creando Gráfico 6: Características Dispersas...")
        
        df_features = self.df.copy()
        df_features['hora'] = df_features['FECHA'].dt.hour
        df_features['dia_semana'] = df_features['FECHA'].dt.dayofweek
        df_features['mes'] = df_features['FECHA'].dt.month
        df_features['trimestre'] = df_features['FECHA'].dt.quarter
        
        df_features['sin_mes'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['cos_mes'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        df_features['sin_hora'] = np.sin(2 * np.pi * df_features['hora'] / 24)
        df_features['cos_hora'] = np.cos(2 * np.pi * df_features['hora'] / 24)
        
        for lag in [1, 6, 12]:
            df_features[f'pm25_lag_{lag}'] = df_features['PM2_5'].shift(lag)
        
        df_features['fin_semana'] = (df_features['dia_semana'] >= 5).astype(int)
        df_features['hora_pico'] = ((df_features['hora'] >= 7) & 
                                  (df_features['hora'] <= 9) | 
                                  (df_features['hora'] >= 17) & 
                                  (df_features['hora'] <= 19)).astype(int)
        
        feature_cols = ['sin_mes', 'cos_mes', 'sin_hora', 'cos_hora',
                       'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_12',
                       'fin_semana', 'hora_pico', 'trimestre']
        
        try:
            df_clean = df_features[feature_cols + ['PM2_5']].dropna()
            X = df_clean[feature_cols].values
            y = df_clean['PM2_5'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            tscv = TimeSeriesSplit(n_splits=5)
            lasso = LassoCV(cv=tscv, random_state=42, max_iter=2000)
            lasso.fit(X_scaled, y)
            
            caracteristicas_seleccionadas = np.abs(lasso.coef_) > 1e-4
            nombres_seleccionados = [feature_cols[i] for i in range(len(feature_cols)) 
                                   if caracteristicas_seleccionadas[i]]
            coefs_seleccionados = lasso.coef_[caracteristicas_seleccionadas]
            
            plt.figure(figsize=(12, 6))
            
            nombres_mejorados = {
                'pm25_lag_1': 'PM2.5 (t-1)',
                'pm25_lag_6': 'PM2.5 (t-6h)', 
                'pm25_lag_12': 'PM2.5 (t-12h)',
                'sin_mes': 'Month (sin)',
                'cos_mes': 'Month (cos)',
                'sin_hora': 'Hour (sin)',
                'cos_hora': 'Hour (cos)',
                'fin_semana': 'Weekend',
                'hora_pico': 'Rush Hour',
                'trimestre': 'Quarter'
            }
            
            nombres_graf = [nombres_mejorados.get(nombre, nombre) for nombre in nombres_seleccionados]
            importancias = np.abs(coefs_seleccionados)
            
            indices_orden = np.argsort(importancias)[::-1]
            nombres_ordenados = [nombres_graf[i] for i in indices_orden]
            importancias_ordenadas = importancias[indices_orden]
            
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(nombres_ordenados)))
            
            y_pos = np.arange(len(nombres_ordenados))
            bars = plt.barh(y_pos, importancias_ordenadas, color=colors, 
                           edgecolor='black', linewidth=1, alpha=0.8)
            
            plt.title('Selected Features from Sparse Optimization\n(Proximal Gradient Method - LASSO)', 
                     fontweight='bold', pad=20)
            plt.xlabel('Feature Importance (|Coefficient|)', fontweight='bold')
            plt.ylabel('Selected Features', fontweight='bold')
            plt.yticks(y_pos, nombres_ordenados)
            plt.grid(True, alpha=0.3, axis='x')
            
            for bar, valor in zip(bars, importancias_ordenadas):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{valor:.3f}', va='center', fontweight='bold')
            
            r2 = r2_score(y, lasso.predict(X_scaled))
            
            plt.text(0.98, 0.02, f'Model Performance:\n'
                                f'R² Score: {r2:.3f}\n'
                                f'Selected: {len(nombres_seleccionados)}/{len(feature_cols)} features\n'
                                f'Regularization: α = {lasso.alpha_:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))
            
        except Exception as e:
            print(f"⚠️ Error en optimización LASSO: {e}")
            plt.figure(figsize=(12, 6))
            correlaciones = df_features[feature_cols + ['PM2_5']].corr()['PM2_5'].abs().sort_values(ascending=True)
            correlaciones = correlaciones[:-1]
            
            nombres_mejorados = {
                'pm25_lag_1': 'PM2.5 (t-1)',
                'pm25_lag_6': 'PM2.5 (t-6h)', 
                'pm25_lag_12': 'PM2.5 (t-12h)',
                'sin_mes': 'Month (sin)',
                'cos_mes': 'Month (cos)',
                'sin_hora': 'Hour (sin)',
                'cos_hora': 'Hour (cos)',
                'fin_semana': 'Weekend',
                'hora_pico': 'Rush Hour',
                'trimestre': 'Quarter'
            }
            
            nombres_graf = [nombres_mejorados.get(nombre, nombre) for nombre in correlaciones.index]
            
            y_pos = np.arange(len(nombres_graf))
            bars = plt.barh(y_pos, correlaciones.values, color='skyblue', 
                           edgecolor='black', linewidth=1, alpha=0.8)
            
            plt.title('Feature Correlation with PM2.5\n(Alternative Analysis)', 
                     fontweight='bold', pad=20)
            plt.xlabel('Absolute Correlation with PM2.5', fontweight='bold')
            plt.ylabel('Features', fontweight='bold')
            plt.yticks(y_pos, nombres_graf)
            plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('Figura_6_Caracteristicas_Dispersas_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_6_Caracteristicas_Dispersas_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 6 guardada: Figura_6_Caracteristicas_Dispersas_PM25.png/.pdf")
    
    def grafico_7_convergencia_algoritmo(self):
        """GRÁFICO 7: Convergencia del Algoritmo FISTA/LASSO"""
        print("📈 Creando Gráfico 7: Convergencia del Algoritmo...")
        
        df_features = self.df.copy()
        df_features['hora'] = df_features['FECHA'].dt.hour
        df_features['mes'] = df_features['FECHA'].dt.month
        df_features['sin_mes'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['cos_mes'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        for lag in [1, 6, 12]:
            df_features[f'pm25_lag_{lag}'] = df_features['PM2_5'].shift(lag)
        
        feature_cols = ['sin_mes', 'cos_mes', 'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_12']
        df_clean = df_features[feature_cols + ['PM2_5']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['PM2_5'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y, alphas=None, eps=0.001, n_alphas=50)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for i, feature in enumerate(feature_cols):
            plt.plot(alphas_lasso, coefs_lasso[i, :], linewidth=2, label=feature)
        
        plt.xlabel('Regularization Parameter (α)', fontweight='bold')
        plt.ylabel('Coefficient Value', fontweight='bold')
        plt.title('LASSO Regularization Path\n(Proximal Gradient Method)', fontweight='bold')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        n_features_selected = [np.sum(np.abs(coefs_lasso[:, i]) > 1e-4) 
                              for i in range(len(alphas_lasso))]
        
        plt.plot(alphas_lasso, n_features_selected, 'b-', linewidth=3, marker='o', markersize=4)
        plt.xlabel('Regularization Parameter (α)', fontweight='bold')
        plt.ylabel('Number of Selected Features', fontweight='bold')
        plt.title('Feature Selection vs Regularization\n(Sparsity Pattern)', fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        lasso_cv = LassoCV(cv=5, random_state=42)
        lasso_cv.fit(X_scaled, y)
        optimal_alpha = lasso_cv.alpha_
        
        plt.axvline(x=optimal_alpha, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Optimal α = {optimal_alpha:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('Figura_7_Convergencia_Algoritmo_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_7_Convergencia_Algoritmo_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 7 guardada: Figura_7_Convergencia_Algoritmo_PM25.png/.pdf")
    
    def grafico_8_comparacion_metodos(self):
        """GRÁFICO 8: Comparación de Métodos de Optimización"""
        print("📈 Creando Gráfico 8: Comparación de Métodos...")
        
        df_features = self.df.copy()
        df_features['mes'] = df_features['FECHA'].dt.month
        df_features['sin_mes'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['cos_mes'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        for lag in [1, 6, 12]:
            df_features[f'pm25_lag_{lag}'] = df_features['PM2_5'].shift(lag)
        
        feature_cols = ['sin_mes', 'cos_mes', 'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_12']
        df_clean = df_features[feature_cols + ['PM2_5']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['PM2_5'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        metodos = {
            'LASSO (L1)': LassoCV(cv=5, random_state=42),
            'Ridge (L2)': Ridge(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        resultados = {}
        
        for nombre, modelo in metodos.items():
            cv_scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='r2')
            modelo.fit(X_scaled, y)
            y_pred = modelo.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            if hasattr(modelo, 'coef_'):
                n_features = np.sum(np.abs(modelo.coef_) > 1e-4) if nombre != 'Linear Regression' else len(modelo.coef_)
            else:
                n_features = len(feature_cols)
            
            resultados[nombre] = {
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std(),
                'rmse': rmse,
                'n_features': n_features
            }
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        nombres = list(resultados.keys())
        r2_means = [resultados[m]['r2_mean'] for m in nombres]
        r2_stds = [resultados[m]['r2_std'] for m in nombres]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = plt.bar(nombres, r2_means, yerr=r2_stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Model Performance\n(R² Score)', fontweight='bold')
        plt.ylabel('R² Score', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, valor in zip(bars, r2_means):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 3, 2)
        rmse_values = [resultados[m]['rmse'] for m in nombres]
        bars = plt.bar(nombres, rmse_values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Model Error\n(RMSE)', fontweight='bold')
        plt.ylabel('RMSE (μg/m³)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, valor in zip(bars, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{valor:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 3, 3)
        n_features_values = [resultados[m]['n_features'] for m in nombres]
        bars = plt.bar(nombres, n_features_values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Model Complexity\n(Selected Features)', fontweight='bold')
        plt.ylabel('Number of Features', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, valor in zip(bars, n_features_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{valor}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('Figura_8_Comparacion_Metodos_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_8_Comparacion_Metodos_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 8 guardada: Figura_8_Comparacion_Metodos_PM25.png/.pdf")
    
    def grafico_9_mapa_correlaciones(self):
        """GRÁFICO 9: Mapa de Calor de Correlaciones"""
        print("📈 Creando Gráfico 9: Mapa de Correlaciones...")
        
        df_corr = self.df.copy()
        df_corr['hora'] = df_corr['FECHA'].dt.hour
        df_corr['dia_semana'] = df_corr['FECHA'].dt.dayofweek
        df_corr['mes'] = df_corr['FECHA'].dt.month
        df_corr['trimestre'] = df_corr['FECHA'].dt.quarter
        
        df_corr['sin_mes'] = np.sin(2 * np.pi * df_corr['mes'] / 12)
        df_corr['cos_mes'] = np.cos(2 * np.pi * df_corr['mes'] / 12)
        df_corr['sin_hora'] = np.sin(2 * np.pi * df_corr['hora'] / 24)
        df_corr['cos_hora'] = np.cos(2 * np.pi * df_corr['hora'] / 24)
        
        for lag in [1, 6, 12, 24]:
            df_corr[f'PM25_lag_{lag}h'] = df_corr['PM2_5'].shift(lag)
        
        df_corr['Weekend'] = (df_corr['dia_semana'] >= 5).astype(int)
        df_corr['Rush_Hour'] = ((df_corr['hora'] >= 7) & (df_corr['hora'] <= 9) | 
                               (df_corr['hora'] >= 17) & (df_corr['hora'] <= 19)).astype(int)
        df_corr['Winter_Months'] = ((df_corr['mes'] >= 6) & (df_corr['mes'] <= 8)).astype(int)
        
        vars_corr = ['PM2_5', 'PM25_lag_1h', 'PM25_lag_6h', 'PM25_lag_12h', 'PM25_lag_24h',
                    'sin_mes', 'cos_mes', 'sin_hora', 'cos_hora',
                    'Weekend', 'Rush_Hour', 'Winter_Months', 'trimestre']
        
        df_clean_corr = df_corr[vars_corr].dropna()
        corr_matrix = df_clean_corr.corr()
        
        plt.figure(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   fmt='.3f', annot_kws={'size': 9})
        
        plt.title('Correlation Matrix of PM2.5 and Related Variables\nLima Metropolitan Area (2015-2024)', 
                 fontweight='bold', pad=20, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.figtext(0.02, 0.02, f'Sample Size: {len(df_clean_corr):,} observations\n'
                               f'Analysis Period: 2015-2024\n'
                               f'Variables: {len(vars_corr)} features', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('Figura_9_Mapa_Correlaciones_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_9_Mapa_Correlaciones_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 9 guardada: Figura_9_Mapa_Correlaciones_PM25.png/.pdf")
    
    def grafico_10_residuos_modelo(self):
        """GRÁFICO 10: Distribución de Residuos del Modelo"""
        print("📈 Creando Gráfico 10: Análisis de Residuos...")
        
        df_features = self.df.copy()
        df_features['mes'] = df_features['FECHA'].dt.month
        df_features['sin_mes'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['cos_mes'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        for lag in [1, 6, 12]:
            df_features[f'pm25_lag_{lag}'] = df_features['PM2_5'].shift(lag)
        
        feature_cols = ['sin_mes', 'cos_mes', 'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_12']
        df_clean = df_features[feature_cols + ['PM2_5']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['PM2_5'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_scaled, y)
        y_pred = lasso.predict(X_scaled)
        residuos = y - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].scatter(y_pred, residuos, alpha=0.6, color='steelblue', s=20)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted PM2.5 (μg/m³)', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals (μg/m³)', fontweight='bold')
        axes[0, 0].set_title('Residuals vs Fitted Values', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        stats.probplot(residuos, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Test)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(residuos, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
        
        mu, sigma = stats.norm.fit(residuos)
        x_norm = np.linspace(residuos.min(), residuos.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        axes[1, 0].plot(x_norm, y_norm, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        axes[1, 0].set_xlabel('Residuals (μg/m³)', fontweight='bold')
        axes[1, 0].set_ylabel('Density', fontweight='bold')
        axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(range(len(residuos)), residuos, alpha=0.7, color='green', linewidth=0.5)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Observation Order', fontweight='bold')
        axes[1, 1].set_ylabel('Residuals (μg/m³)', fontweight='bold')
        axes[1, 1].set_title('Residuals vs Order', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        from scipy.stats import jarque_bera, shapiro
        jb_stat, jb_pval = jarque_bera(residuos)
        sw_stat, sw_pval = shapiro(residuos[:5000] if len(residuos) > 5000 else residuos)
        
        stats_text = f'''Model Diagnostics:
R² Score: {lasso.score(X_scaled, y):.4f}
RMSE: {np.sqrt(np.mean(residuos**2)):.3f} μg/m³
Mean Residual: {np.mean(residuos):.4f}
Std Residual: {np.std(residuos):.3f}

Normality Tests:
Jarque-Bera: {jb_pval:.4f}
Shapiro-Wilk: {sw_pval:.4f}
'''
        
        plt.figtext(0.02, 0.02, stats_text, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                   fontsize=10)
        
        plt.suptitle('LASSO Model Residual Analysis\nProximal Gradient Method for PM2.5 Prediction', 
                    fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig('Figura_10_Residuos_Modelo_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_10_Residuos_Modelo_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 10 guardada: Figura_10_Residuos_Modelo_PM25.png/.pdf")
    
    def grafico_11_analisis_sensibilidad(self):
        """GRÁFICO 11: Análisis de Sensibilidad (Lambda vs Características)"""
        print("📈 Creando Gráfico 11: Análisis de Sensibilidad...")
        
        df_features = self.df.copy()
        df_features['mes'] = df_features['FECHA'].dt.month
        df_features['sin_mes'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['cos_mes'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        df_features['hora'] = df_features['FECHA'].dt.hour
        df_features['sin_hora'] = np.sin(2 * np.pi * df_features['hora'] / 24)
        
        for lag in [1, 6, 12, 24]:
            df_features[f'pm25_lag_{lag}'] = df_features['PM2_5'].shift(lag)
        
        feature_cols = ['sin_mes', 'cos_mes', 'sin_hora', 'pm25_lag_1', 
                       'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24']
        df_clean = df_features[feature_cols + ['PM2_5']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['PM2_5'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        alphas = np.logspace(-4, 1, 50)
        
        r2_scores = []
        n_features_selected = []
        rmse_scores = []
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=2000)
            lasso.fit(X_train, y_train)
            
            y_pred = lasso.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            n_features = np.sum(np.abs(lasso.coef_) > 1e-4)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            n_features_selected.append(n_features)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].semilogx(alphas, r2_scores, 'b-', linewidth=3, marker='o', markersize=4)
        axes[0].set_xlabel('Regularization Parameter (λ)', fontweight='bold')
        axes[0].set_ylabel('R² Score', fontweight='bold')
        axes[0].set_title('Model Performance vs Regularization', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        best_idx = np.argmax(r2_scores)
        best_alpha = alphas[best_idx]
        best_r2 = r2_scores[best_idx]
        axes[0].scatter(best_alpha, best_r2, color='red', s=100, zorder=5)
        axes[0].annotate(f'Best λ = {best_alpha:.4f}\nR² = {best_r2:.3f}', 
                        xy=(best_alpha, best_r2), xytext=(best_alpha*10, best_r2-0.05),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        axes[1].semilogx(alphas, rmse_scores, 'r-', linewidth=3, marker='s', markersize=4)
        axes[1].set_xlabel('Regularization Parameter (λ)', fontweight='bold')
        axes[1].set_ylabel('RMSE (μg/m³)', fontweight='bold')
        axes[1].set_title('Model Error vs Regularization', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].semilogx(alphas, n_features_selected, 'g-', linewidth=3, marker='^', markersize=4)
        axes[2].set_xlabel('Regularization Parameter (λ)', fontweight='bold')
        axes[2].set_ylabel('Number of Selected Features', fontweight='bold')
        axes[2].set_title('Sparsity vs Regularization', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        axes[2].axvspan(best_alpha/2, best_alpha*2, alpha=0.2, color='yellow', 
                       label='Optimal Range')
        axes[2].legend()
        
        plt.suptitle('Sensitivity Analysis of LASSO Regularization\nProximal Gradient Method Parameter Tuning', 
                    fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig('Figura_11_Analisis_Sensibilidad_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_11_Analisis_Sensibilidad_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 11 guardada: Figura_11_Analisis_Sensibilidad_PM25.png/.pdf")
    
    def grafico_12_prediccion_vs_observado(self):
        """GRÁFICO 12: Predicción vs Observado"""
        print("📈 Creando Gráfico 12: Predicción vs Observado...")
        
        df_features = self.df.copy()
        df_features['mes'] = df_features['FECHA'].dt.month
        df_features['sin_mes'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['cos_mes'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        for lag in [1, 6, 12]:
            df_features[f'pm25_lag_{lag}'] = df_features['PM2_5'].shift(lag)
        
        feature_cols = ['sin_mes', 'cos_mes', 'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_12']
        df_clean = df_features[feature_cols + ['PM2_5']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['PM2_5'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train, y_train)
        
        y_train_pred = lasso.predict(X_train)
        y_test_pred = lasso.predict(X_test)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=20, label='Training Data')
        
        min_val = min(y_train.min(), y_train_pred.min())
        max_val = max(y_train.max(), y_train_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        z = np.polyfit(y_train, y_train_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_train, p(y_train), 'g-', linewidth=2, alpha=0.8, label='Fitted Line')
        
        plt.xlabel('Observed PM2.5 (μg/m³)', fontweight='bold')
        plt.ylabel('Predicted PM2.5 (μg/m³)', fontweight='bold')
        plt.title(f'Training Set Performance\nR² = {r2_train:.3f}, RMSE = {rmse_train:.2f}', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, alpha=0.6, color='red', s=20, label='Test Data')
        
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        z = np.polyfit(y_test, y_test_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_test, p(y_test), 'g-', linewidth=2, alpha=0.8, label='Fitted Line')
        
        plt.xlabel('Observed PM2.5 (μg/m³)', fontweight='bold')
        plt.ylabel('Predicted PM2.5 (μg/m³)', fontweight='bold')
        plt.title(f'Test Set Performance\nR² = {r2_test:.3f}, RMSE = {rmse_test:.2f}', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        info_text = f'''LASSO Model Performance Summary:

Training Set:
• R² Score: {r2_train:.4f}
• RMSE: {rmse_train:.3f} μg/m³
• MAE: {mae_train:.3f} μg/m³
• Samples: {len(y_train):,}

Test Set:
• R² Score: {r2_test:.4f}
• RMSE: {rmse_test:.3f} μg/m³
• MAE: {mae_test:.3f} μg/m³
• Samples: {len(y_test):,}

Model Configuration:
• Method: Proximal Gradient (LASSO)
• Features: {len(feature_cols)}
• Selected: {np.sum(np.abs(lasso.coef_) > 1e-4)}
• Alpha: {lasso.alpha_:.4f}
'''
        
        plt.figtext(0.02, 0.02, info_text, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
                   fontsize=10)
        
        plt.suptitle('Model Prediction Quality Assessment\nObserved vs Predicted PM2.5 Concentrations', 
                    fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig('Figura_12_Prediccion_vs_Observado_PM25.png', dpi=300, bbox_inches='tight')
        plt.savefig('Figura_12_Prediccion_vs_Observado_PM25.pdf', bbox_inches='tight')
        plt.show()
        print("✅ Figura 12 guardada: Figura_12_Prediccion_vs_Observado_PM25.png/.pdf")
    
    def generar_todos_los_graficos(self):
        """Genera todos los 12 gráficos para artículo científico completo"""
        print("🚀 Generando TODOS los gráficos para artículo científico...")
        print("="*70)
        
        print("📊 GENERANDO GRÁFICOS BÁSICOS...")
        self.grafico_1_serie_temporal()
        print()
        self.grafico_2_patron_estacional()
        print()
        self.grafico_3_percentiles()
        print()
        self.grafico_4_comparacion_temporal()
        print()
        self.grafico_5_superaciones()
        print()
        self.grafico_6_caracteristicas_dispersas()
        print()
        
        print("🧮 GENERANDO GRÁFICOS AVANZADOS...")
        self.grafico_7_convergencia_algoritmo()
        print()
        self.grafico_8_comparacion_metodos()
        print()
        self.grafico_9_mapa_correlaciones()
        print()
        self.grafico_10_residuos_modelo()
        print()
        self.grafico_11_analisis_sensibilidad()
        print()
        self.grafico_12_prediccion_vs_observado()
        
        print("\n🎉 ¡TODOS LOS 12 GRÁFICOS GENERADOS!")
        print("="*70)
        print("Archivos generados (PNG y PDF):")
        print("\n📊 GRÁFICOS BÁSICOS:")
        print("• Figura_1_Serie_Temporal_PM25")
        print("• Figura_2_Patron_Estacional_PM25") 
        print("• Figura_3_Percentiles_PM25")
        print("• Figura_4_Comparacion_Temporal_PM25")
        print("• Figura_5_Superaciones_PM25")
        print("• Figura_6_Caracteristicas_Dispersas_PM25")
        print("\n🧮 GRÁFICOS AVANZADOS:")
        print("• Figura_7_Convergencia_Algoritmo_PM25")
        print("• Figura_8_Comparacion_Metodos_PM25")
        print("• Figura_9_Mapa_Correlaciones_PM25")
        print("• Figura_10_Residuos_Modelo_PM25")
        print("• Figura_11_Analisis_Sensibilidad_PM25")
        print("• Figura_12_Prediccion_vs_Observado_PM25")
        print("\n🏆 ¡ARTÍCULO CIENTÍFICO COMPLETO Y ROBUSTO!")
        print("💡 Cada gráfico está listo para usar en las secciones correspondientes")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para generar gráficos"""
    try:
        # Crear generador de gráficos
        generador = GraficosArticuloCientifico()
        
        # Preguntar qué gráfico generar
        print("\n🎯 ¿Qué gráfico deseas generar?")
        print("\n📊 GRÁFICOS BÁSICOS:")
        print("1. Serie Temporal (Figura 1)")
        print("2. Patrón Estacional (Figura 2)")
        print("3. Percentiles (Figura 3)")
        print("4. Comparación Temporal (Figura 4)")
        print("5. Superaciones (Figura 5)")
        print("6. Características Dispersas (Figura 6)")
        print("\n🧮 GRÁFICOS AVANZADOS:")
        print("7. Convergencia del Algoritmo (Figura 7)")
        print("8. Comparación de Métodos (Figura 8)")
        print("9. Mapa de Correlaciones (Figura 9)")
        print("10. Análisis de Residuos (Figura 10)")
        print("11. Análisis de Sensibilidad (Figura 11)")
        print("12. Predicción vs Observado (Figura 12)")
        print("\n0. TODOS los 12 gráficos")
        
        opcion = input("\nSelecciona una opción (0-12): ").strip()
        
        if opcion == "1":
            generador.grafico_1_serie_temporal()
        elif opcion == "2":
            generador.grafico_2_patron_estacional()
        elif opcion == "3":
            generador.grafico_3_percentiles()
        elif opcion == "4":
            generador.grafico_4_comparacion_temporal()
        elif opcion == "5":
            generador.grafico_5_superaciones()
        elif opcion == "6":
            generador.grafico_6_caracteristicas_dispersas()
        elif opcion == "7":
            generador.grafico_7_convergencia_algoritmo()
        elif opcion == "8":
            generador.grafico_8_comparacion_metodos()
        elif opcion == "9":
            generador.grafico_9_mapa_correlaciones()
        elif opcion == "10":
            generador.grafico_10_residuos_modelo()
        elif opcion == "11":
            generador.grafico_11_analisis_sensibilidad()
        elif opcion == "12":
            generador.grafico_12_prediccion_vs_observado()
        elif opcion == "0":
            generador.generar_todos_los_graficos()
        else:
            print("❌ Opción inválida. Generando todos los gráficos...")
            generador.generar_todos_los_graficos()
        
        return generador
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("📊 GENERADOR DE GRÁFICOS PARA ARTÍCULO CIENTÍFICO")
    print("🔬 Métodos de Gradiente Proximal para Optimización Dispersa")
    print("📍 Análisis de Calidad del Aire PM2.5 - Lima Metropolitana")
    print("="*60)
    generador = main()