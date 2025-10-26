# --- Imports e Configurações Iniciais ---
import matplotlib
matplotlib.use('Agg')
import os
import warnings
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import ruptures as rpt
import traceback
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, roc_auc_score, f1_score,
    average_precision_score, precision_recall_curve,
    classification_report, make_scorer
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from econml.dml import LinearDML
from econml.metalearners import XLearner
from econml.cate_interpreter import SingleTreeCateInterpreter
import shap
import geopandas as gpd
import requests
import pingouin as pg
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.io as pio
import csv
import io

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
pio.templates.default = "plotly_white"

# --- Paths ---
BASE_DIR = Path('c:/Users/PC GAMER/Downloads/erp-ofdi-model')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
RESULTS_DIR = BASE_DIR / 'results'
R_STYLE_DIR = RESULTS_DIR / 'r_style_plots'
NAT_EARTH_ZIP = BASE_DIR / "data" / "naturalearth_lowres.zip"

# --- Diretórios ---
DIRS = {
    "BASE": RESULTS_DIR, "R_STYLE": R_STYLE_DIR, 
    "EDA": RESULTS_DIR / 'eda', "PHASE_ANALYSIS": RESULTS_DIR / 'phase_analysis', 
    "REGRESSION": RESULTS_DIR / 'models' / 'regression', 
    "CLASSIFICATION": RESULTS_DIR / 'models' / 'classification', 
    "TIMESERIES": RESULTS_DIR / 'models' / 'timeseries', 
    "CAUSAL": RESULTS_DIR / 'models' / 'causal', "SHAP": RESULTS_DIR / 'models' / 'shap'
}

# --- Funções Auxiliares ---
def create_dir(path): Path(path).mkdir(parents=True, exist_ok=True)

def save_plot(fig, folder_path, filename_base, dpi=150):
    try:
        create_dir(Path(folder_path))
        filepath = Path(folder_path) / f"{filename_base}.png"
        if isinstance(fig, go.Figure):
            fig.write_image(filepath, scale=2)
        else:
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"       ✓ Gráfico salvo: {filename_base}.png")
    except Exception as e:
        print(f"       ✗ Erro ao salvar gráfico {filename_base}: {e}")
    finally:
        if not isinstance(fig, go.Figure):
            plt.close(fig)

def clean_numeric_column(series: pd.Series) -> pd.Series:
    if series is None: return pd.Series(dtype=float)
    cleaned = series.astype(str).str.replace(r'[\[\]$,%\s]', '', regex=True).str.strip()
    numeric_direct = pd.to_numeric(cleaned, errors='coerce')
    failed_mask = numeric_direct.isna()
    if failed_mask.any():
        NUMBER_REGEX = r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        extracted = cleaned[failed_mask].str.extract(NUMBER_REGEX, flags=re.IGNORECASE).iloc[:, 0]
        numeric_extracted = pd.to_numeric(extracted, errors='coerce')
        numeric_direct.loc[failed_mask] = numeric_extracted
    return numeric_direct

def print_header(title): print("\n" + "="*80); print(f" {title}"); print("="*80)
def print_status(message, status="✓"): print(f"   {status} {message}")

# =============================================================================
# 1.5 GERAÇÃO DE GRÁFICOS (R-style)
# =============================================================================
def plot_transactions_by_investor(df, output_dir):
    print_status("Gerando gráfico: Transações por Investidor (3 Fases)...")
    phases = {
        'Going Global 1.0\nfrom 2005 to 2012.': (2005, 2012),
        'Going Global 2.0\nfrom 2013 to 2016.': (2013, 2016),
        'Going Global 3.0\nfrom 2017 to 2024.': (2017, 2024)
    }
    top_n = 20
    counts_by_phase = {}
    global_min, global_max = np.inf, -np.inf
    for label, (start, end) in phases.items():
        df_phase = df[(df['Year'] >= start) & (df['Year'] <= end)]
        counts = df_phase['Natureza_Investidor'].value_counts().nlargest(top_n).sort_values(ascending=True)
        counts_by_phase[label] = counts
        if not counts.empty:
            global_min = min(global_min, counts.min())
            global_max = max(global_max, counts.max())

    if not np.isfinite(global_min):
        print_status("Sem dados para transações por investidor.", "⚠")
        return

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=global_min, vmax=global_max)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), facecolor="#f3f3f3")
    fig.subplots_adjust(wspace=0.12, bottom=0.24)

    for ax, (phase_label, counts) in zip(axes, counts_by_phase.items()):
        ax.set_facecolor("#f3f3f3")
        if counts.empty:
            ax.text(0.5, 0.5, "Sem dados suficientes", ha="center", va="center", fontsize=12)
            continue

        positions = np.arange(len(counts))
        colors = cmap(norm(counts.values))
        ax.bar(positions, counts.values, color=colors, edgecolor="#4d4d4d", linewidth=0.6)
        ax.scatter(positions, counts.values, color="#111111", s=35, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(counts.index, rotation=75, ha="right", fontsize=9)
        ax.set_ylabel("Number of Transactions", fontsize=11)
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_title(phase_label, loc="left", fontsize=17, color="#d82616", fontweight="bold")

        ax.grid(axis='y', linestyle='--', color='#9e9e9e', linewidth=0.6, alpha=0.6)
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

    save_plot(fig, output_dir, 'r_style_transactions_by_investor')

def plot_investment_by_sector(df, output_dir):
    print_status("Gerando gráfico: Investimento por Setor (3 Fases)...")
    phases = {
        'Going Global 1.0\nfrom 2005 to 2012.': (2005, 2012),
        'Going Global 2.0\nfrom 2013 to 2016.': (2013, 2016),
        'Going Global 3.0\nfrom 2017 to 2024.': (2017, 2024)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), facecolor='white')
    fig.subplots_adjust(wspace=0.3)
    cmap = plt.get_cmap('viridis')

    for ax, (title, (start, end)) in zip(axes, phases.items()):
        df_phase = df[(df['Year'] >= start) & (df['Year'] <= end)]
        sector_sum = df_phase.groupby('Sector')['Valor_USD'].sum().sort_values(ascending=True)
        
        colors = cmap(np.linspace(0.1, 0.9, len(sector_sum)))
        ax.barh(sector_sum.index, sector_sum.values, color=colors)
        
        ax.set_title(title, loc='left', fontsize=16, color='red', weight='bold')
        ax.set_xlabel('Amount of Investment in US$', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0e'))
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.set_facecolor('white')
        ax.grid(False)

    plt.tight_layout()
    save_plot(fig, output_dir, 'r_style_investment_by_sector')

def plot_greenfield_donuts(df, output_dir):
    print_status("Gerando gráfico: Donut Greenfield vs M&A (3 Fases)...")
    phases = {
        'Going Global 1.0\nfrom 2005 to 2012.': (2005, 2012),
        'Going Global 2.0\nfrom 2013 to 2016.': (2013, 2016),
        'Going Global 3.0\nfrom 2017 to 2024.': (2017, 2024)
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')
    fig.subplots_adjust(wspace=0.18)

    for ax, (title, (start, end)) in zip(axes, phases.items()):
        df_phase = df[(df['Year'] >= start) & (df['Year'] <= end)]
        gf_sum = df_phase[df_phase['Tipo_Investimento'] == 'Greenfield']['Valor_USD'].sum()
        ma_sum = df_phase[df_phase['Tipo_Investimento'] == 'M&A']['Valor_USD'].sum()
        total = gf_sum + ma_sum

        if total == 0:
            ax.text(0.5, 0.5, "Sem dados", ha='center', va='center')
            continue

        sizes = [gf_sum, ma_sum]
        labels = ['Greenfield', 'Other']
        colors = ['#440154', '#fde725']

        wedges, _ = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            radius=1.05,
            counterclock=False,
            wedgeprops=dict(width=0.35, edgecolor='white')
        )

        ax.add_artist(plt.Circle((0, 0), 0.65, fc='white'))
        ax.set_title(title, loc='left', fontsize=17, color="#d82616", fontweight="bold")

        for wedge, label, value in zip(wedges, labels, sizes):
            if value == 0:
                continue
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            ax.annotate(
                f"{label}\n{value / total * 100:.1f}%\n${value / 1e3:.1f}B",
                xy=(x * 0.9, y * 0.9),
                xytext=(x * 1.35, y * 1.35),
                ha='center',
                va='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#4d4d4d", lw=0.8),
                arrowprops=dict(arrowstyle="-", color="#4d4d4d", lw=0.8)
            )

        ax.text(0, 0, f"Total\n${total / 1e3:.1f}B", ha='center', va='center', fontsize=13, fontweight='bold', color='#333333')
        ax.axis('equal')

    save_plot(fig, output_dir, 'r_style_greenfield_donuts')

def plot_world_maps(df, output_dir):
    print_status("Gerando gráfico: Choropleth Mundial (3 Fases)...")
    try:
        if not NAT_EARTH_ZIP.exists():
            NAT_EARTH_ZIP.parent.mkdir(parents=True, exist_ok=True)
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            with requests.get(url, timeout=30) as resp:
                resp.raise_for_status()
                NAT_EARTH_ZIP.write_bytes(resp.content)
        world = gpd.read_file(f"zip://{NAT_EARTH_ZIP}!ne_110m_admin_0_countries.shp")
        
        country_map = {
            "United States of America": "United States", "United Kingdom": "United Kingdom",
            "South Korea": "Korea, Rep.", "Russia": "Russian Federation",
            "Czechia": "Czech Republic", "Hong Kong SAR, China": "Hong Kong",
            "Iran": "Iran, Islamic Rep.", "Egypt": "Egypt, Arab Rep.",
            "Slovakia": "Slovak Republic"
        }
        df['country_mapped'] = df['Country'].replace(country_map)
        
        phases = {
            'Going Global 1.0\nfrom 2005 to 2012.': (2005, 2012),
            'Going Global 2.0\nfrom 2013 to 2016.': (2013, 2016),
            'Going Global 3.0\nfrom 2017 to 2024.': (2017, 2024)
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 10), facecolor='#f0f0f0')
        fig.subplots_adjust(wspace=0.05, top=0.9, bottom=0.1)

        vmax = df.groupby(['country_mapped', pd.cut(df['Year'], [2004, 2012, 2016, 2024])])['Valor_USD'].sum().max()
        vmin = df['Valor_USD'].min()

        for i, (ax, (title, (start, end))) in enumerate(zip(axes, phases.items())):
            df_phase = df[(df['Year'] >= start) & (df['Year'] <= end)]
            country_sum = df_phase.groupby('country_mapped')['Valor_USD'].sum().reset_index()
            
            merged = world.set_index('NAME').join(country_sum.set_index('country_mapped'))
            
            ax.set_facecolor('#f0f0f0')
            merged.plot(column='Valor_USD', cmap='plasma', linewidth=0.5, ax=ax, edgecolor='0.5', 
                        missing_kwds={"color": "grey", "label": "No data"},
                        vmax=vmax, vmin=vmin)
            
            ax.set_title(title, loc='center', fontsize=20, color='red', weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        cax = fig.add_axes([0.25, 0.15, 0.5, 0.03])
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Amount of Investment (USD Millions)', fontsize=12)
        cbar.outline.set_visible(False)

        save_plot(fig, output_dir, 'r_style_world_maps')
    except Exception as e:
        print_status(f"Erro ao gerar mapas: {e}. Verifique a instalação do geopandas e seus dados.", "✗")
        traceback.print_exc()

def plot_sector_sunburst(df, output_dir):
    print_status("Gerando novo gráfico: Sunburst de Setor/Subsetor por Fase...")
    phases = {
        'Going Global 1.0 (2005-2012)': (2005, 2012),
        'Going Global 2.0 (2013-2016)': (2013, 2016),
        'Going Global 3.0 (2017-2024)': (2017, 2024)
    }

    for i, (title, (start, end)) in enumerate(phases.items()):
        df_phase = df[(df['Year'] >= start) & (df['Year'] <= end)]
        if df_phase.empty:
            print_status(f"Sem dados para Sunburst na fase {title}", "⚠")
            continue

        df_grouped = df_phase.groupby(['Sector', 'Subsector'])['Valor_USD'].sum().reset_index()
        df_grouped = df_grouped[df_grouped['Valor_USD'] > 0]

        if df_grouped.empty:
            print_status(f"Sem dados agrupados para Sunburst na fase {title}", "⚠")
            continue

        fig = go.Figure(go.Sunburst(
            labels=df_grouped['Subsector'].tolist() + df_grouped['Sector'].unique().tolist(),
            parents=[f"{row['Sector']}" for _, row in df_grouped.iterrows()] + [''] * df_grouped['Sector'].nunique(),
            values=df_grouped['Valor_USD'].tolist() + df_grouped.groupby('Sector')['Valor_USD'].sum().tolist(),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Total Investment: $%{value:,.2f}M<br>Parent: %{parent}<extra></extra>',
            marker=dict(colorscale='viridis')
        ))

        fig.update_layout(
            title_text=f"Investment by Sector/Subsector<br><sup>{title}</sup>",
            title_font_size=20,
            margin=dict(t=80, l=0, r=0, b=0)
        )
        
        filename = f'new_sunburst_phase_{i+1}'
        save_plot(fig, output_dir, filename)

def plot_investment_distribution_over_time(df, output_dir):
    print_status("Gerando novo gráfico: Ridgeline de Distribuição de Investimentos...")
    try:
        from joypy import joyplot
        
        df_plot = df[df['Valor_USD'] > 0].copy()
        df_plot['log_Valor_USD'] = np.log10(df_plot['Valor_USD'])
        
        if df_plot.empty or df_plot['Year'].nunique() < 2:
            print_status("Dados insuficientes para o gráfico Ridgeline.", "⚠")
            return

        fig, axes = joyplot(
            data=df_plot[['log_Valor_USD', 'Year']], 
            by='Year',
            figsize=(12, 10),
            colormap=plt.get_cmap('plasma'),
            alpha=0.8,
            linewidth=1,
            linecolor='w',
            overlap=2.5,
            grid='y',
            legend=False
        )
        
        plt.title('Distribution of Investment Value Over Time (Log Scale)', fontsize=20, loc='left', pad=40)
        plt.xlabel("Log10(Investment Value in USD Millions)")
        
        def log_to_orig(x, pos):
            return f"${10**x / 1e3:.1f}B" if x >= 3 else f"${10**x:.0f}M"
        
        formatter = mticker.FuncFormatter(log_to_orig)
        axes[-1].xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)

        save_plot(fig, output_dir, 'new_ridgeline_distribution')

    except ImportError:
        print_status("Pacote 'joypy' não encontrado. Pulando gráfico Ridgeline.", "⚠")
        print_status("Para instalar, use: pip install joypy", "ℹ")
    except Exception as e:
        print_status(f"Erro ao gerar gráfico Ridgeline: {e}", "✗")
        traceback.print_exc()

def plot_investment_by_region_over_time(df, output_dir):
    print_status("Gerando novo gráfico: Investimento por Região ao Longo do Tempo...")
    
    top_n_regions = df.groupby('Region')['Valor_USD'].sum().nlargest(7).index
    df_top_regions = df[df['Region'].isin(top_n_regions)]
    
    region_ts = df_top_regions.groupby(['Year', 'Region'])['Valor_USD'].sum().unstack().fillna(0)
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    
    palette = sns.color_palette("viridis", len(top_n_regions))
    region_ts.plot(kind='line', marker='o', ax=ax, color=palette, linewidth=2.5)
    
    ax.set_title('Investment Evolution for Top Regions', fontsize=18, fontweight='bold', loc='left')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Investment (USD Millions)', fontsize=12)
    ax.legend(title='Region', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', visible=False)
    ax.set_facecolor('white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    save_plot(fig, output_dir, 'new_investment_by_region_over_time')

def plot_sector_concentration_hhi(df, output_dir):
    print_status("Gerando novo gráfico: Concentração de Setor por Fase (HHI)...")
    
    phases = {
        'Going Global 1.0 (2005-2012)': (2005, 2012),
        'Going Global 2.0 (2013-2016)': (2013, 2016),
        'Going Global 3.0 (2017-2024)': (2017, 2024)
    }
    
    hhi_scores = {}
    for phase, (start, end) in phases.items():
        df_phase = df[(df['Year'] >= start) & (df['Year'] <= end)]
        if df_phase.empty:
            hhi_scores[phase] = 0
            continue
        
        sector_totals = df_phase.groupby('Sector')['Valor_USD'].sum()
        total_investment = sector_totals.sum()
        
        if total_investment == 0:
            hhi_scores[phase] = 0
            continue
            
        market_shares = (sector_totals / total_investment) * 100
        hhi = (market_shares ** 2).sum()
        hhi_scores[phase] = hhi
        
    hhi_series = pd.Series(hhi_scores)
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    colors = sns.color_palette("viridis", len(hhi_series))
    bars = ax.bar(hhi_series.index, hhi_series.values, color=colors)
    
    ax.set_title('Sector Investment Concentration (Herfindahl-Hirschman Index)', fontsize=16, fontweight='bold', loc='left')
    ax.set_ylabel('HHI Score (0-10,000)', fontsize=12)
    ax.set_xlabel('Phase', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:.0f}', ha='center', va='bottom', fontsize=11)
        
    ax.axhline(y=1500, color='gray', linestyle='--', lw=1)
    ax.text(ax.get_xlim()[1], 1500, ' Unconcentrated < 1500', va='center', ha='left', color='gray')
    ax.axhline(y=2500, color='gray', linestyle='--', lw=1)
    ax.text(ax.get_xlim()[1], 2500, ' Moderately Concentrated < 2500', va='center', ha='left', color='gray')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False)
    ax.set_facecolor('white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    save_plot(fig, output_dir, 'new_sector_concentration_hhi')

def plot_investment_by_share_size(df, output_dir):
    print_status("Gerando gráfico: Investimento por Faixa de Market Share (3 Fases)...")
    
    if 'Share_Size' not in df.columns or df['Share_Size'].isnull().all():
        print_status("Coluna 'Share_Size' não encontrada ou vazia. Pulando gráfico.", "⚠")
        return

    phases = {
        'Going Global 1.0': (2005, 2012),
        'Going Global 2.0': (2013, 2016),
        'Going Global 3.0': (2017, 2024)
    }
    
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-19', '20-39', '40-59', '60-79', '80-99']
    
    df_plot = df.copy()
    df_plot['Share_Size'] = pd.to_numeric(df_plot['Share_Size'], errors='coerce')
    df_plot.dropna(subset=['Share_Size'], inplace=True)
    
    df_plot['Share_Bin'] = pd.cut(df_plot['Share_Size'], bins=bins, labels=labels, right=False, include_lowest=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="#f3f3f3")
    fig.subplots_adjust(wspace=0.2, bottom=0.15)

    for ax, (title, (start, end)) in zip(axes, phases.items()):
        df_phase = df_plot[(df_plot['Year'] >= start) & (df_plot['Year'] <= end)]
        
        ax.set_facecolor("#f3f3f3")
        ax.set_title(title, loc='center', fontsize=17, color="#d82616", fontweight="bold", pad=20)
        
        if df_phase.empty or df_phase['Share_Bin'].isnull().all():
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            ax.set_xlabel("Market Share (%)", fontsize=11)
            ax.set_ylabel("Amount of Investment", fontsize=11)
            ax.grid(False)
            for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_visible(False)
            continue

        grouped = df_phase.groupby('Share_Bin', observed=False)['Valor_USD'].sum()
        total_investment = grouped.sum()

        bars = ax.bar(grouped.index, grouped.values, color='#555555', width=0.6)
        
        ax.set_ylabel("Amount of Investment", fontsize=11)
        ax.set_xlabel("Market Share (%)", fontsize=11)
        
        if total_investment > 0:
            for bar in bars:
                height = bar.get_height()
                percentage = (height / total_investment) * 100
                y_offset = ax.get_ylim()[1] * 0.01
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + y_offset, f'{percentage:.1f}%',
                        ha='center', va='bottom', fontsize=10, color='black')

        ax.grid(axis='y', linestyle='--', color='#9e9e9e', linewidth=0.6, alpha=0.7)
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        
        if not grouped.empty:
            ax.set_ylim(0, grouped.max() * 1.20)

    save_plot(fig, output_dir, 'r_style_investment_by_share_size')

def _plot_ggstats_style(df_plot, x_var, y_var, output_dir, filename, title_prefix):
    """Função auxiliar para criar gráficos no estilo ggbetweenstats."""
    if df_plot[x_var].nunique() < 2:
        print_status(f"Dados insuficientes para o gráfico {filename} (menos de 2 grupos em '{x_var}').", "⚠")
        return

    # Preparação do plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Paleta de cores similar ao R
    palette = {"Greenfield": "#0072B2", "M&A": "#E69F00", "BRI": "#0072B2", "Others": "#E69F00"}
    
    # 1. Violin Plot (fundo)
    sns.violinplot(data=df_plot, x=x_var, y=y_var, ax=ax, palette=palette,
                   inner=None, saturation=0.4, alpha=0.3, linewidth=0)

    # 2. Box Plot (dentro do violino)
    sns.boxplot(data=df_plot, x=x_var, y=y_var, ax=ax, palette=palette,
                width=0.2, boxprops={'zorder': 2, 'facecolor': 'none', 'edgecolor': 'black'},
                whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                medianprops={'color': 'red', 'linewidth': 2, 'solid_capstyle': 'round'})

    # 3. Pontos de dados (jitter)
    sns.stripplot(data=df_plot, x=x_var, y=y_var, ax=ax, palette=palette,
                  jitter=0.15, size=5, alpha=0.5, edgecolor='black', linewidth=0.5)

    # Análise estatística com Pingouin
    try:
        stats = pg.ttest(df_plot[y_var], df_plot[x_var], correction='auto')
        t_val = stats['T'].iloc[0]
        p_val = stats['p-val'].iloc[0]
        dof = stats['dof'].iloc[0]
        g_hedges = stats['hedges-g'].iloc[0]
        ci95 = stats['CI95%'].iloc[0]
        n_obs = len(df_plot)
        
        stats_title = (f"$t_\\mathrm{{Welch}}$({dof:.2f}) = {t_val:.2f}, $p$ = {p_val:.3f}, "
                       f"$\\hat{{g}}_\\mathrm{{Hedges}}$ = {g_hedges:.2f}, CI$_{{95\\%}}$ [{ci95[0]:.2f}, {ci95[1]:.2f}], "
                       f"$n_\\mathrm{{obs}}$ = {n_obs}")
        ax.set_title(stats_title, loc='left', fontsize=11, family='serif')
    except Exception as e:
        print_status(f"Erro na anotação estatística para {filename}: {e}", "⚠")
        ax.set_title(f"Análise de {title_prefix}", loc='left', fontsize=14, fontweight='bold')

    # Anotações de média
    means = df_plot.groupby(x_var)[y_var].mean()
    for i, grp in enumerate(means.index):
        mean_val = means[grp]
        ax.plot([i - 0.25, i + 0.25], [mean_val, mean_val], 'k--', lw=1, zorder=3)
        ax.text(i + 0.3, mean_val, f"$\\hat{{\\mu}}_\\mathrm{{mean}}$ = {mean_val:.2f}",
                ha='left', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

    # Anotações de outliers (top 5 por grupo)
    outliers = df_plot.groupby(x_var).apply(lambda g: g.nlargest(5, y_var)).reset_index(drop=True)
    for _, row in outliers.iterrows():
        x_pos = list(df_plot[x_var].unique()).index(row[x_var])
        ax.annotate(f"{row[y_var]:.0f}", (x_pos, row[y_var]),
                    textcoords="offset points", xytext=(0, 5), ha='center',
                    fontsize=8, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.5))

    # Estilo final (Tufte)
    ax.set_xlabel(title_prefix, fontsize=12, fontweight='bold')
    ax.set_ylabel("Quantity in Millions", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    sns.despine(ax=ax, offset=10, trim=True)
    ax.grid(False)

    save_plot(fig, output_dir, filename)

def plot_ggstats_greenfield(df, output_dir):
    print_status("Gerando gráfico estilo R: Greenfield vs M&A...")
    df_plot = df.copy()
    df_plot = df_plot.rename(columns={'Tipo_Investimento': 'Greenfield'})
    df_plot['Greenfield'] = df_plot['Greenfield'].replace({'Other': 'M&A'})
    df_plot = df_plot[df_plot['Valor_USD'] <= 1100]
    
    _plot_ggstats_style(df_plot, 'Greenfield', 'Valor_USD', output_dir, 
                        'python_style_boxplot_greenfield', 'Greenfield')

def plot_ggstats_bri(df, output_dir):
    print_status("Gerando gráfico estilo R: BRI vs Others...")
    if 'BRI' not in df.columns:
        print_status("Coluna 'BRI' não encontrada. Pulando gráfico.", "⚠")
        return
        
    df_plot = df.copy()
    df_plot['bri'] = np.where(
        df_plot['BRI'].astype(str).str.strip().str.lower().isin(['1', 'bri', 'y', 'yes', 'sim', 'true']), 
        'BRI', 'Others'
    )
    df_plot = df_plot[df_plot['Valor_USD'] <= 1100]
    
    _plot_ggstats_style(df_plot, 'bri', 'Valor_USD', output_dir, 
                        'python_style_boxplot_bri', 'BRI')

# =============================================================================
# 2. CARREGAMENTO E LIMPEZA DE DADOS
# =============================================================================
def load_and_clean_data(data_path):
    print_header("[ETAPA 2/11] 📥 Carregamento e Limpeza de Dados")
    df = None
    try:
        df = pd.read_csv(data_path, sep=',', header=0, encoding='utf-8', engine='c', low_memory=False, skipinitialspace=True)
        print_status("Dados brutos carregados com engine 'c' (utf-8).")
    except Exception as e:
        print_status(f"Falha com engine 'c': {e}. Tentando parser manual...", "⚠")
        try:
            with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            header_line = lines[0].strip()
            header = [h.strip().replace('/', '_') for h in header_line.split(',')]
            num_columns = len(header)
            
            data = []
            for i, line in enumerate(lines[1:]):
                if not line.strip(): continue
                
                reader = csv.reader([line], skipinitialspace=True)
                fields = next(reader)
                
                if len(fields) > num_columns:
                    investor_end_index = len(fields) - (num_columns - 3)
                    fields[2] = ','.join(fields[2:investor_end_index + 1])
                    del fields[3:investor_end_index + 1]

                while len(fields) < num_columns:
                    fields.append(None)
                
                data.append(fields[:num_columns])

            df = pd.DataFrame(data, columns=header)
            print_status(f"Dados carregados com parser manual: {df.shape[0]} linhas x {df.shape[1]} colunas")

        except Exception as e_manual:
            print_status(f"ERRO no parser manual: {e_manual}", status="✗")
            traceback.print_exc()
            return None

    if df is None:
        print_status("ERRO: DataFrame não foi carregado.", status="✗")
        return None

    df.columns = df.columns.str.strip().str.replace(' ', '_')
    print_status(f"Colunas limpas: {df.columns.tolist()}")

    rename_map = {
        'Quantity_in_Millions': 'Valor_USD',
        'Investor_Contractor': 'Natureza_Investidor',
        'Greenfield': 'Tipo_Investimento'
    }
    df = df.rename(columns=rename_map)
    
    if 'Valor_USD' not in df.columns:
        print_status("Coluna 'Valor_USD' não encontrada após renomear.", "✗")
        return None

    print_status("Limpando 'Valor_USD'..."); df['Valor_USD'] = clean_numeric_column(df.get('Valor_USD'))
    print_status("Limpando 'Year'..."); df['Year'] = pd.to_numeric(df.get('Year'), errors='coerce')
    if 'Share_Size' in df.columns:
        print_status("Limpando 'Share_Size'..."); df['Share_Size'] = clean_numeric_column(df.get('Share_Size'))
    else:
        print_status("Coluna 'Share_Size' não encontrada, pulando limpeza.", "⚠")

    initial_rows = df.shape[0]
    df.dropna(subset=['Valor_USD', 'Year'], inplace=True)
    df = df[df['Valor_USD'] > 0]
    df['Year'] = df['Year'].astype(int)
    df = df[df['Year'] <= 2024]
    print_status(f"Linhas removidas por NaN/Zero/Ano>2024: {initial_rows - df.shape[0]}")

    if df.empty: print_status("Não restaram dados.", "✗"); return None
    print_status(f"Dados limpos: {df.shape[0]} linhas x {df.shape[1]} (Anos: {df['Year'].min()}-{df['Year'].max()})")
    return df

# ========================================================================
# 3. ENGENHARIA DE FEATURES OTIMIZADA
# ========================================================================
def criar_features_avancadas(df):
    print_status("Criando features avançadas...")
    
    df = df.sort_values('Year')
    
    # Features temporais
    for window in [2, 3]:
        df[f'valor_roll_mean_{window}'] = df.groupby('Sector')['Valor_USD']\
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'valor_roll_std_{window}'] = df.groupby('Sector')['Valor_USD']\
            .transform(lambda x: x.rolling(window, min_periods=1).std())
    
    # Features de política
    df['policy_cumulative'] = (df['Year'] >= 2017).cumsum()
    df['policy_interaction'] = df['post_ERP'] * df['Year']
    
    # Features hierárquicas
    for col in ['Sector', 'Region']:
        if col in df.columns:
            sector_mean = df.groupby('Sector')['log_Valor_USD'].transform('mean')
            global_mean = df['log_Valor_USD'].mean()
            df[f'{col}_hierarchical'] = 0.7 * sector_mean + 0.3 * global_mean
    
    # Períodos
    df['phase_1'] = ((df['Year'] >= 2005) & (df['Year'] <= 2012)).astype(int)
    df['phase_2'] = ((df['Year'] >= 2013) & (df['Year'] <= 2016)).astype(int)
    df['phase_3'] = (df['Year'] >= 2017).astype(int)
    
    # Interações
    if 'Sector_Num' in df.columns:
        df['sector_year_int'] = df['Sector_Num'] * df['Year']
    if 'Region_Num' in df.columns:
        df['region_policy_int'] = df['Region_Num'] * df['post_ERP']
    
    return df

def melhorar_target(df):
    print_status("Melhorando target...")
    
    # Target mais discriminativo
    high_threshold = df['Valor_USD'].quantile(0.70)
    df['Alvo_Binario_Melhor'] = (df['Valor_USD'] > high_threshold).astype(int)
    
    # Target adaptativo por ano
    def target_adaptativo(group):
        return (group > group.quantile(0.70)).astype(int)
    
    df['Alvo_Adaptativo'] = df.groupby('Year')['Valor_USD'].transform(target_adaptativo)
    
    return df

def feature_engineer(df):
    print_header("[ETAPA 3/11] 🛠️ Engenharia de Features")
    df = df.sort_values(by=['Year']).reset_index(drop=True)
    print_status("Dados ordenados por 'Year' para features temporais.")
    
    df['log_Valor_USD'] = np.log(df['Valor_USD'])
    print_status("Alvo 'log_Valor_USD' criado.")

    df['post_ERP'] = (df['Year'] >= 2017).astype(int)

    def assign_gg_phase(year):
        if 2005 <= year <= 2012: return 'GG_1.0'
        if 2013 <= year <= 2016: return 'GG_2.0'
        if 2017 <= year <= 2024: return 'GG_3.0_ERP'
        return f'Fora ({year})'

    df['Fase_GG'] = df['Year'].apply(assign_gg_phase).astype('category')
    print_status(f"Fases GG corrigidas: {list(df['Fase_GG'].cat.categories)}")

    if 'Tipo_Investimento' in df.columns: 
        df['Tipo_Investimento'] = np.where(
            df['Tipo_Investimento'].astype(str).str.upper().str.strip().isin(['G', 'GREENFIELD', '1', 'Y', 'YES', 'SIM', 'TRUE']), 
            'Greenfield', 'M&A'
        )
    else: 
        df['Tipo_Investimento'] = 'M&A'
    df['Tipo_Investimento'] = df['Tipo_Investimento'].astype('category')

    mediana_valor = df['Valor_USD'].median()
    df['Alvo_Binario'] = (df['Valor_USD'] > mediana_valor).astype(int)
    print_status(f"Alvo binário 'Alvo_Binario' criado (1 se > {mediana_valor:.2f} USDM)")
    print_status(f"Distribuição: {df['Alvo_Binario'].mean()*100:.1f}% 'Alto Valor'")

    # Processamento Categórico
    cols_to_process = [
        'Sector', 'Region', 'Natureza_Investidor', 'Tipo_Investimento', 
        'Fase_GG', 'Subsector', 'Country'
    ]
    
    cols_high_cardinality = ['Natureza_Investidor', 'Country']
    cols_low_medium_cardinality = [c for c in cols_to_process if c not in cols_high_cardinality]
    
    print_status("Iniciando limpeza robusta de NaNs...")
    for col in cols_to_process:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'None', '', ' '], 'Missing').str.strip()
                df[col] = df[col].fillna('Missing')
                df[col] = df[col].astype('category')
                print_status(f"Coluna '{col}' limpa e convertida para categoria.")

                if col in cols_low_medium_cardinality:
                    df[f'{col}_Num'] = df[col].cat.codes
                    print_status(f"   Feature '{col}_Num' criada.")
                else:
                    print_status(f"   Coluna '{col}' (alta cardinalidade) será tratada com Target Encoding.")
            
            except Exception as e:
                 print_status(f"Erro ao processar coluna '{col}': {e}", "✗")
        else:
            print_status(f"Coluna '{col}' não encontrada.", status="⚠")

    # Features avançadas e target melhorado
    df = criar_features_avancadas(df)
    df = melhorar_target(df)

    return df

# =============================================================================
# 4. ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================================
def run_eda(df, eda_dir):
    print_header("[ETAPA 4/11] 📊 Análise Exploratória de Dados (EDA)")
    create_dir(eda_dir)
    try:
        
        df.describe(include='all').to_csv(eda_dir / 'summary_statistics.csv')

        print_status("Plot: Distribuição do Alvo...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(df['Valor_USD'], kde=True, ax=ax1, bins=100); ax1.set_title(f"Valor_USD\nMediana: {df['Valor_USD'].median():.0f}"); ax1.set_xlabel("Valor (USD Milhões)")
        sns.histplot(df['log_Valor_USD'], kde=True, ax=ax2, bins=50); ax2.set_title(f"log_Valor_USD\nMediana: {df['log_Valor_USD'].median():.2f}"); ax2.set_xlabel("Log(Valor)")
        save_plot(fig, eda_dir, 'eda_target_distribution_log_vs_raw')

        print_status("Plot: Evolução Temporal Total...")
        ts_total = df.groupby('Year')['Valor_USD'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(14, 7)); ax.plot(ts_total['Year'], ts_total['Valor_USD'], marker='o'); ax.set_title('Evolução Total por Ano'); ax.set_xlabel('Ano'); ax.set_ylabel('Total (USD Milhões)')
        if not ts_total.empty: ax.axvspan(2017, ts_total['Year'].max(), color='red', alpha=0.1, label='Pós-ERP (2017+)')
        ax.legend(); save_plot(fig, eda_dir, 'eda_total_investment_over_time')

        print_status("Plot: H1 - Análise Setorial pré/pós-ERP...")
        if 'Sector' in df.columns and 'post_ERP' in df.columns and df['post_ERP'].nunique() >= 2:
            try:
                sector_summary = df.groupby(['post_ERP', 'Sector'])['Valor_USD'].sum().unstack(fill_value=0)
                sector_share = sector_summary.apply(lambda x: 100 * x / x.sum(), axis=0).rename(columns={0: 'Pre-ERP Share (%)', 1: 'Post-ERP Share (%)'})
                sector_share['Change (p.p.)'] = sector_share['Post-ERP Share (%)'] - sector_share['Pre-ERP Share (%)']
                sector_share = sector_share.sort_values('Change (p.p.)', ascending=False)
                sector_share.to_csv(eda_dir / 'eda_h1_sector_shift_summary.csv')

                fig, ax = plt.subplots(figsize=(14, 10)); top_n = 7
                plot_data = pd.concat([sector_share.nlargest(top_n, 'Change (p.p.)'), sector_share.nsmallest(top_n, 'Change (p.p.)')]).sort_values('Change (p.p.)')
                plot_data = plot_data[~plot_data.index.duplicated(keep='first')]
                cols_to_plot = [col for col in ['Pre-ERP Share (%)', 'Post-ERP Share (%)'] if col in plot_data.columns]
                if not cols_to_plot or plot_data.empty: print_status("       Erro plot H1.", "✗")
                else: plot_data.plot(kind='barh', y=cols_to_plot, ax=ax); ax.set_title(f'Mudança Setorial (Top {top_n} Aumentos/Reduções)'); ax.set_xlabel('Share (%)'); ax.set_ylabel('Setor'); plt.tight_layout(); save_plot(fig, eda_dir, 'eda_h1_sector_shift_plot')
            except Exception as e: print(f"       ✗ Erro H1: {e}")
        else: print_status("Dados insuficientes/colunas faltando para H1.", "⚠")

        print_status("Plot: H3 - Greenfield vs M&A...")
        tipo_summary = df.groupby(['Year', 'Tipo_Investimento'])['Valor_USD'].sum().unstack(fill_value=0)
        tipo_summary.to_csv(eda_dir / 'eda_h3_greenfield_vs_ma.csv')
        fig, ax = plt.subplots(figsize=(14, 7)); tipo_summary.plot(kind='area', stacked=True, ax=ax, alpha=0.7); ax.set_title('Valor por Tipo (M&A vs Greenfield)'); ax.set_ylabel('Valor (USD Milhões)')
        if not ts_total.empty: ax.axvspan(2017, ts_total['Year'].max(), color='red', alpha=0.1, label='Pós-ERP (2017+)')
        ax.legend(); save_plot(fig, eda_dir, 'eda_h3_greenfield_vs_ma_over_time')
    except Exception as e:
        print_status(f"Erro na EDA: {e}", "✗")

# =============================================================================
# 4.5 ANÁLISE DE MEDIAÇÃO E MODERAÇÃO POR FASE GG
# =============================================================================
def run_ols_interaction_phase(df_phase: pd.DataFrame, phase_name: str, y_var: str, x_var: str, z_var: str, results_dir: Path):
    
    print(f"       - Rodando Interação OLS: {y_var} ~ {x_var} * C({z_var})")
    results_list = []
    try:
        if df_phase[z_var].nunique() < 2:
            print(f"         ⚠ Pulando OLS: Coluna '{z_var}' tem menos de 2 níveis na fase '{phase_name}'.")
            return results_list
        
        # Criar dummies manualmente para melhor performance
        dummies = pd.get_dummies(df_phase[z_var], prefix=z_var, drop_first=True)
        df_temp = pd.concat([df_phase[[y_var, x_var]], dummies], axis=1)
        
        # Construir fórmula dinamicamente
        dummy_cols = [f"Q('{col}')" for col in dummies.columns]
        formula = f"Q('{y_var}') ~ Q('{x_var}') * ({' + '.join(dummy_cols)})"
        
        model = ols(formula, data=df_temp).fit()
        
        # Resultados mais completos
        result_summary = {
            'Phase': phase_name,
            'Interaction': f'{x_var} * {z_var}',
            'R2': model.rsquared,
            'R2_adj': model.rsquared_adj,
            'F_pvalue': model.f_pvalue,
            'N_obs': len(df_phase),
            'AIC': model.aic,
            'BIC': model.bic
        }
        
        # Adicionar coeficientes significativos
        coef_table = model.summary2().tables[1]
        significant_coefs = coef_table[coef_table['P>|t|'] < 0.1]
        result_summary['N_Significant'] = len(significant_coefs)
        
        results_list.append(result_summary)
        print(f"         ✓ Interação OLS concluída (R²={model.rsquared:.3f}, R²_adj={model.rsquared_adj:.3f}).")
        
    except Exception as e: 
        print(f"         ✗ Erro OLS para '{z_var}': {e}")
    return results_list

def run_mediation_phase(df_phase: pd.DataFrame, phase_name: str, y_var: str, x_var: str, m_var: str, results_dir: Path):
    
    print(f"       - Rodando Mediação OLS: {x_var} -> {m_var} -> {y_var}")
    results = {'Phase': phase_name, 'Model': f'{x_var} -> {m_var} -> {y_var}'}
    try:
        # Verificar variância e tamanho amostral
        if (df_phase[x_var].nunique() < 2 or df_phase[m_var].nunique() < 2 or 
            df_phase[y_var].nunique() < 2 or len(df_phase) < 50):
            print(f"         ⚠ Pulando Mediação: Dados insuficientes para '{m_var}'.")
            return None

        # Modelo M ~ X
        model_m = ols(f'Q("{m_var}") ~ Q("{x_var}")', data=df_phase).fit()
        a_coef = model_m.params.get(f'Q("{x_var}")', np.nan)
        a_pval = model_m.pvalues.get(f'Q("{x_var}")', np.nan)
        a_sig = a_pval < 0.1
        
        # Modelo Y ~ X + M
        model_y = ols(f'Q("{y_var}") ~ Q("{x_var}") + Q("{m_var}")', data=df_phase).fit()
        b_coef = model_y.params.get(f'Q("{m_var}")', np.nan)
        b_pval = model_y.pvalues.get(f'Q("{m_var}")', np.nan)
        b_sig = b_pval < 0.1
        c_prime_coef = model_y.params.get(f'Q("{x_var}")', np.nan)
        c_prime_pval = model_y.pvalues.get(f'Q("{x_var}")', np.nan)
        
        # Modelo total Y ~ X
        model_total = ols(f'Q("{y_var}") ~ Q("{x_var}")', data=df_phase).fit()
        c_coef = model_total.params.get(f'Q("{x_var}")', np.nan)
        c_pval = model_total.pvalues.get(f'Q("{x_var}")', np.nan)

        # Efeitos de mediação
        acme = a_coef * b_coef
        ade = c_prime_coef
        total_effect = c_coef
        prop_mediated = (acme / total_effect) if total_effect != 0 else np.nan
        
        # Apenas retornar se efeitos forem relevantes
        if abs(acme) < 0.001 and not (a_sig and b_sig):
            return None
            
        results.update({
            'a_coef': a_coef, 'a_pval': a_pval, 'a_sig': a_sig,
            'b_coef': b_coef, 'b_pval': b_pval, 'b_sig': b_sig,
            'c_prime_coef': c_prime_coef, 'c_prime_pval': c_prime_pval,
            'c_coef': c_coef, 'c_pval': c_pval,
            'ACME': acme, 'ADE': ade, 'Total_Effect': total_effect, 
            'Prop_Mediated': prop_mediated,
            'Mediation_Significant': a_sig and b_sig,
            'N_obs': len(df_phase)
        })
        
        sig_status = "SIGNIFICATIVA" if (a_sig and b_sig) else "não significativa"
        print(f"         ✓ Mediação OLS concluída. ACME={acme:.3f} ({sig_status})")
        return results
        
    except Exception as e: 
        print(f"         ✗ Erro Mediação para '{m_var}': {e}")
        return None

def run_phase_analysis(df, phase_col='Fase_GG', results_dir=DIRS["PHASE_ANALYSIS"]):
    print_header("[ETAPA 4.5/11] 🔬 Análise por Fase GG (Interação e Mediação)")
    create_dir(results_dir)
    
    # Focar nas variáveis mais promissoras
    mediator_vars_num = ['Sector_Num', 'Region_Num']  # Remover Subsector se tiver muitos níveis
    mediator_vars_num = [v for v in mediator_vars_num if v in df.columns]
    print_status(f"Mediadores (Num) para teste: {mediator_vars_num}")
    
    interaction_vars_cat = ['Sector', 'Region']  # Focar nas mais importantes
    interaction_vars_cat = [v for v in interaction_vars_cat if v in df.columns]
    print_status(f"Interações (Cat) para teste: {interaction_vars_cat}")

    outcome_var = 'log_Valor_USD'
    predictor_var = 'Year'
    phases = df[phase_col].unique()
    all_interaction_results = []
    all_mediation_results = []

    for phase in phases:
        if isinstance(phase, str) and phase.startswith('Fora'): 
            continue
            
        print(f"   - Analisando Fase: {phase}")
        df_phase = df[df[phase_col] == phase].copy()
        
        # Aumentar mínimo de observações
        if df_phase.shape[0] < 50: 
            print(f"       ⚠ Dados insuficientes fase {phase}")
            continue
        
        # Interações
        for z_var_cat in interaction_vars_cat:
            results = run_ols_interaction_phase(df_phase, phase, outcome_var, predictor_var, z_var_cat, results_dir)
            all_interaction_results.extend(results)

        # Mediação apenas se houver dados suficientes
        if len(df_phase) > 80:
            for m_var_num in mediator_vars_num:
                if m_var_num in df_phase.columns:
                    result_med = run_mediation_phase(df_phase, phase, outcome_var, predictor_var, m_var_num, results_dir)
                    if result_med:
                        all_mediation_results.append(result_med)
    
    
    if all_interaction_results:
        df_int = pd.DataFrame(all_interaction_results)
        df_int = df_int.sort_values('R2', ascending=False)
        df_int.to_csv(results_dir / 'summary_interaction_ols_by_phase.csv', index=False)
        
        best_r2 = df_int['R2'].max() if not df_int.empty else 0
        print_status(f"Sumário Interação salvo. Melhor R²: {best_r2:.3f}")
        
    else: 
        print_status("Nenhum resultado de Interação OLS.", "⚠")
        
    if all_mediation_results:
        df_med = pd.DataFrame(all_mediation_results)
        # Filtrar apenas mediações com efeitos relevantes
        df_med_significant = df_med[df_med['Mediation_Significant'] == True]
        
        if not df_med_significant.empty:
            df_med_significant.to_csv(results_dir / 'summary_mediation_ols_by_phase.csv', index=False)
            print_status(f"Sumário Mediação salvo. {len(df_med_significant)} mediações significativas")
        else:
            df_med.to_csv(results_dir / 'summary_mediation_ols_by_phase.csv', index=False)
            print_status("Sumário Mediação salvo (nenhuma significativa)", "⚠")
    else: 
        print_status("Nenhum resultado de Mediação OLS.", "⚠")

# =============================================================================
# 5. DEFINIÇÃO DE PIPELINES DE ML
# =============================================================================
def get_ml_pipelines(numeric_features, categorical_features):
    print_header("[ETAPA 5/11] 🔬 Definindo Pipelines de Pré-processamento")
    
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []

    numeric_transformer_linear = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    numeric_transformer_tree = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) 
    ])
    
    categorical_transformer_ohe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    categorical_transformer_ord = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    transformers_linear = []
    transformers_tree = []

    if numeric_features:
        transformers_linear.append(('num', numeric_transformer_linear, numeric_features))
        transformers_tree.append(('num', numeric_transformer_tree, numeric_features))
    else:
        print_status("Nenhuma feature numérica para o pipeline.", "⚠")

    if categorical_features:
        transformers_linear.append(('cat', categorical_transformer_ohe, categorical_features))
        transformers_tree.append(('cat', categorical_transformer_ord, categorical_features))
    else:
        print_status("Nenhuma feature categórica para o pipeline.", "⚠")

    if not transformers_linear and not transformers_tree:
        print_status("Nenhuma feature numérica ou categórica para criar pipelines.", "✗")
        return None, None

    preprocessor_linear = ColumnTransformer(transformers_linear, remainder='drop')
    preprocessor_tree = ColumnTransformer(transformers_tree, remainder='drop')

    preprocessor_linear.set_output(transform="pandas")
    preprocessor_tree.set_output(transform="pandas")
    print_status("Pipelines 'linear' (OHE) e 'tree' (Ordinal) criadas.")
    return preprocessor_linear, preprocessor_tree

# =============================================================================
# 6. MODELOS DE REGRESSÃO
# =============================================================================
def mape_scorer(y_log, y_pred_log):
    y_orig = np.exp(y_log)
    y_pred_orig = np.exp(y_pred_log)
    y_pred_orig[y_pred_orig < 0] = 0
    y_orig_safe = np.maximum(y_orig, 1)
    mape = np.mean(np.abs((y_orig - y_pred_orig) / y_orig_safe)) * 100
    return -mape

neg_mape_scorer = make_scorer(mape_scorer, greater_is_better=False)

def run_regression_otimizada(X, y_log, preprocessor_linear, preprocessor_tree, reg_dir):
    print_header("[ETAPA 6/11] 🤖 Modelos de Regressão Otimizados")
    create_dir(reg_dir)
    tscv = TimeSeriesSplit(n_splits=5)

    if X.empty or preprocessor_linear is None or preprocessor_tree is None:
        print_status("X (features) ou preprocessors vazios. Pulando Regressão.", "✗")
        return None, None

    # --- MODELOS ---
    pipelines = {
        "XGB_Ord": Pipeline([('pp', preprocessor_tree), ('m', xgb.XGBRegressor(random_state=42, n_jobs=-1))]),
        "LGBM_Ord": Pipeline([('pp', preprocessor_tree), ('m', lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))])
    }
    
    param_grids = {
        "XGB_Ord": {
            'm__n_estimators': [800, 1000],
            'm__learning_rate': [0.005, 0.01],
            'm__max_depth': [4, 6],
            'm__subsample': [0.7, 0.8],
            'm__colsample_bytree': [0.7, 0.8],
            'm__reg_alpha': [0.1, 0.5],
            'm__reg_lambda': [1, 1.5]
        },
        "LGBM_Ord": {
            'm__n_estimators': [800, 1000],
            'm__learning_rate': [0.005, 0.01],
            'm__num_leaves': [15, 31],
            'm__max_depth': [6, 8],
            'm__subsample': [0.7, 0.8],
            'm__colsample_bytree': [0.7, 0.8],
            'm__reg_alpha': [0.1, 0.5],
            'm__reg_lambda': [0.1, 0.5]
        }
    }

    results = {}
    best_model_name = None
    best_mape = np.inf

    for name in ['XGB_Ord', 'LGBM_Ord']:  
        print(f"   - Treinando {name}...")
        pipeline = pipelines[name]
        param_grid = param_grids[name]
        
        try:
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=tscv, 
                scoring=neg_mape_scorer, refit=True, 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X, y_log)
            
            # Calcular métricas
            best_model = grid_search.best_estimator_
            
            # Validação 
            y_preds_log, y_tests_log = [], []
            for train_idx, test_idx in tscv.split(X):
                if len(train_idx) < 20: 
                    continue
                    
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
                
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                
                y_preds_log.extend(y_pred)
                y_tests_log.extend(y_test)
            
            # Calcular MAPE 
            y_test_orig = np.exp(y_tests_log)
            y_pred_orig = np.exp(y_preds_log)
            y_pred_orig = np.maximum(y_pred_orig, 0.1)
            
            mape = np.mean(np.abs((y_test_orig - y_pred_orig) / np.maximum(y_test_orig, 1))) * 100
            rmse_log = np.sqrt(mean_squared_error(y_tests_log, y_preds_log))
            
            results[name] = {
                'RMSE(log)': rmse_log,
                'MAPE(Original)': mape,
                'Best_Params': grid_search.best_params_
            }
            
            print(f"       ✓ {name}: RMSE(log)={rmse_log:.3f}, MAPE(Original)={mape:.2f}%")
            
            if mape < best_mape:
                best_mape = mape
                best_model_name = name
                
        except Exception as e:
            print(f"       ✗ Erro em {name}: {e}")

    if results:
        summary_df = pd.DataFrame(results).T.sort_values('MAPE(Original)')
        summary_df.to_csv(reg_dir / 'regression_models_summary_otimizada.csv')
        
        
        print_status(f"Melhor modelo: {best_model_name} (MAPE: {best_mape:.1f}%)")
    
    return best_model_name, results[best_model_name]['Best_Params'] if best_model_name else None

# =============================================================================
# 7. MODELOS DE CLASSIFICAÇÃO
# =============================================================================
def run_classification_otimizada(X, y, preprocessor_linear, preprocessor_tree, clf_dir):
    print_header("[ETAPA 7/11] 🎯 Modelos de Classificação Otimizados")
    create_dir(clf_dir)

    if y.nunique() < 2: print_status("Alvo com 1 classe.", "⚠"); return None, None
    if X.empty or preprocessor_linear is None or preprocessor_tree is None:
        print_status("X (features) ou preprocessors vazios. Pulando Classificação.", "✗"); return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print_status(f"Treino: {X_train.shape[0]} / Teste: {X_test.shape[0]}")

    pipelines = {
        "XGB_Ord": Pipeline([('pp', preprocessor_tree), ('m', xgb.XGBClassifier(random_state=42, n_jobs=-1))]),
        "LGBM_Ord": Pipeline([('pp', preprocessor_tree), ('m', lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))]),
        "CatBoost_Ord": Pipeline([('pp', preprocessor_tree), ('m', cb.CatBoostClassifier(random_state=42, verbose=0))])
    }
    
    param_grids = {
        "XGB_Ord": {
            'm__n_estimators': [300, 500],
            'm__learning_rate': [0.01, 0.05],
            'm__max_depth': [6, 8],
            'm__scale_pos_weight': [1, 2]
        },
        "LGBM_Ord": {
            'm__n_estimators': [300, 500],
            'm__learning_rate': [0.01, 0.05],
            'm__num_leaves': [31, 63],
            'm__max_depth': [8, 12],
            'm__scale_pos_weight': [1, 2]
        },
        "CatBoost_Ord": {
            'm__iterations': [300, 500],
            'm__learning_rate': [0.01, 0.05],
            'm__depth': [6, 8],
            'm__auto_class_weights': ['Balanced', None]
        }
    }

    results = {}
    best_model_name = None
    best_roc_auc = 0.0

    for name in pipelines:
        print(f"   - Treinando {name}...")
        pipeline = pipelines[name]
        param_grid = param_grids[name]
        
        try:
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=skf, 
                scoring='roc_auc', refit=True, 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            accuracy = accuracy_score(y_test, best_model.predict(X_test))

            results[name] = {
                'Accuracy': accuracy,
                'ROC-AUC': roc_auc,
                'Best_Params': grid_search.best_params_
            }
            
            print(f"       ✓ {name}: AUC={roc_auc:.3f}, Acc={accuracy:.3f}")
            print(f"       ✓ Melhores Parâmetros: {grid_search.best_params_}")

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = name

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
            ax1.plot([0, 1], [0, 1], "k:", label="Perfeita")
            ax1.plot(mean_pred, fraction_pos, "s-", label=f"{name} (Best)")
            ax1.set_title("Calibração")
            ax1.legend()
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ax2.plot(recall, precision, label=f"{name} (AUC = {roc_auc:.3f})")
            ax2.set_title("PR Curve")
            ax2.legend()
            save_plot(fig, clf_dir, f'{name}_best_calibration_pr_curve')

        except Exception as e:
            print(f"       ✗ Erro em {name}: {e}")

    if results:
        summary_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
        summary_df.to_csv(clf_dir / 'classification_models_summary_otimizada.csv')
        print_status(f"Resumo classificação salvo. Melhor: {best_model_name} (AUC: {best_roc_auc:.3f})")

    return best_model_name, results[best_model_name]['Best_Params'] if best_model_name else None

# =============================================================================
# 8. MODELOS DE SÉRIES TEMPORAIS
# =============================================================================
def run_time_series(df, ts_dir):
    print_header("[ETAPA 8/11] ⏱️ Modelos de Séries Temporais")
    create_dir(ts_dir)
    ts_data = df.groupby('Year')['Valor_USD'].sum().reset_index().rename(columns={'Year': 'ds', 'Valor_USD': 'y'})
    ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')
    if len(ts_data) < 10: print_status("Dados insuficientes.", "⚠"); return

    # ARIMA
    print("   - Treinando ARIMA...");
    try:
        ts_log = np.log(ts_data.set_index('ds')['y'])
        model_arima = ARIMA(ts_log, order=(1, 1, 1))
        fit_arima = model_arima.fit()
        with open(ts_dir / 'arima_summary.txt', 'w') as f:
            f.write(fit_arima.summary().as_text())
        fc_log = fit_arima.get_forecast(steps=5)
        fc_mean = np.exp(fc_log.predicted_mean)
        ci = np.exp(fc_log.conf_int())
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(ts_data['ds'], ts_data['y'], label='Observado')
        ax.plot(fc_mean.index, fc_mean.values, label='ARIMA')
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.3)
        ax.legend()
        ax.set_title('ARIMA')
        save_plot(fig, ts_dir, 'arima_forecast')
        print_status("ARIMA salvo.")
    except Exception as e:
        print(f"       ✗ Erro ARIMA: {e}")

    # Prophet
    print("   - Treinando Prophet...");
    try:
        prophet_df = ts_data.copy()
        prophet_df['y'] = np.log(prophet_df['y'])
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=5, freq='Y')
        fc = m.predict(future)
        for col in ['yhat','yhat_lower','yhat_upper','trend']:
            fc[col] = np.exp(fc[col])
        fc.to_csv(ts_dir / 'prophet_forecast_data.csv', index=False)
        fig1 = m.plot(fc)
        ax = fig1.gca()
        ax.plot(ts_data['ds'], ts_data['y'], 'k.', label='Observado')
        ax.set_title('Prophet')
        ax.legend()
        save_plot(fig1, ts_dir, 'prophet_forecast')
        fig2 = m.plot_components(fc)
        save_plot(fig2, ts_dir, 'prophet_components')
        print_status("Prophet salvo.")
    except Exception as e:
        print(f"       ✗ Erro Prophet: {e}")

# =============================================================================
# 9. MODELOS CAUSAIS E DE REGIME
# =============================================================================
def run_causal_models(df_ml, preprocessor_tree, numeric_features, categorical_features, causal_dir):
    print_header("[ETAPA 9/11] 🔬 Modelos Causais e de Regime")
    create_dir(causal_dir)
    
    all_features = numeric_features + categorical_features
    if not all_features or preprocessor_tree is None:
        print_status("Nenhuma feature ou preprocessor definido para modelos causais. Pulando.", "⚠")
        return

    # Markov Switching
    print("   - Treinando Markov Switching...");
    try:
        ts_data = df_ml.groupby('Year')['Valor_USD'].sum().reset_index().set_index('Year')
        if len(ts_data) < 10: raise ValueError("Dados insuficientes")
        mod_ms = MarkovRegression(endog=ts_data['Valor_USD'], k_regimes=2, trend='c', switching_variance=True)
        res_ms = mod_ms.fit(search_reps=10)
        with open(causal_dir / 'markov_switching_summary.txt', 'w') as f:
            f.write(res_ms.summary().as_text())
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(ts_data.index, res_ms.smoothed_marginal_probabilities[0])
        axes[1].plot(ts_data.index, res_ms.smoothed_marginal_probabilities[1])
        fig.suptitle('Markov Switching Probs')
        save_plot(fig, causal_dir, 'markov_prob')
        print_status("Markov Switching salvo.")
    except Exception as e:
        print(f"       ✗ Erro Markov: {e}")

    # DML
    Y_causal = df_ml['log_Valor_USD']
    T_causal = df_ml['post_ERP']
    X_causal = df_ml[all_features]
    
    valid_idx = X_causal.notna().all(axis=1) & Y_causal.notna()
    Y_causal = Y_causal[valid_idx]
    T_causal = T_causal[valid_idx]
    X_causal = X_causal[valid_idx]

    if X_causal.empty:
        print_status("Não há dados para DML/XLearner após remover NaNs.", "⚠")
        return

    print("   - Processando controles causais...");
    try:
        from sklearn.base import clone
        preprocessor_causal = clone(preprocessor_tree)
        preprocessor_causal.set_output(transform="default")
        X_processed = preprocessor_causal.fit_transform(X_causal)
        if not isinstance(X_processed, np.ndarray) or not X_processed.flags['C_CONTIGUOUS']:
            X_processed = np.ascontiguousarray(X_processed)
    except Exception as e:
        print(f"       ✗ Erro processamento: {e}")
        return

    print("   - Treinando DML...");
    try:
        model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
        model_t = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
        dml_est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, random_state=42)
        dml_est.fit(Y_causal, T_causal, X=X_processed, W=None)
        summary_dml_df = pd.DataFrame({
            'effect (ATT_log)': dml_est.ate_, 
            'stderr(log)': dml_est.ate_stderr_, 
            'p_value': dml_est.pvalue_, 
            'conf_lower(log)': dml_est.ate_interval()[0], 
            'conf_upper(log)': dml_est.ate_interval()[1]
        }, index=['post_ERP'])
        summary_dml_df.to_csv(causal_dir / 'dml_summary_att.csv')
        effect_pct = (np.exp(dml_est.ate_[0]) - 1) * 100
        print_status(f"DML salvo. Efeito ERP: {dml_est.ate_[0]:.3f} (log) (~{effect_pct:.2f}%)")
    except Exception as e:
        print(f"       ✗ Erro DML: {e}")

# =============================================================================
# 10. INTERPRETABILIDADE (SHAP)
# =============================================================================
def fit_final_model_and_run_shap(model_name, best_params, preprocessor_linear, preprocessor_tree, X, y_log, feature_names_in, shap_dir):
    print_header("[ETAPA 10/11] 🔍 Interpretabilidade (SHAP)")
    create_dir(shap_dir)

    if model_name is None:
        print_status("Nenhum modelo de regressão foi selecionado. Pulando SHAP.", "⚠")
        return
    if X.empty or preprocessor_linear is None or preprocessor_tree is None:
        print_status("X (features) ou preprocessors vazios para SHAP. Pulando.", "⚠")
        return

    print_status(f"Retreinando o melhor modelo ({model_name}) no dataset completo para o SHAP...")

    model_map = {
        "XGB_Ord": (preprocessor_tree, xgb.XGBRegressor(random_state=42, n_jobs=-1)),
        "LGBM_Ord": (preprocessor_tree, lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
        "CatBoost_Ord": (preprocessor_tree, cb.CatBoostRegressor(random_state=42, verbose=0)),
        "RandomForest_Ord": (preprocessor_tree, RandomForestRegressor(random_state=42, n_jobs=-1))
    }

    preprocessor_to_use, base_model = model_map.get(model_name, (None, None))

    if preprocessor_to_use is None:
        print_status(f"Nome do modelo '{model_name}' não encontrado no mapa SHAP. Pulando.", "✗")
        return

    final_pipeline = Pipeline(steps=[('pp', preprocessor_to_use), ('m', base_model)])

    if best_params:
        try:
            model_params = {key.replace('m__', ''): val for key, val in best_params.items() if key.startswith('m__')}
            if model_params:
                final_pipeline.named_steps['m'].set_params(**model_params)
                print_status(f"Parâmetros definidos para {model_name}: {model_params}")
        except Exception as e:
            print(f"       ⚠ Aviso: Falha ao definir parâmetros {best_params} para {model_name}. Usando defaults. Erro: {e}")

    try:
        valid_idx_shap = X.notna().all(axis=1) & y_log.notna()
        X_shap = X[valid_idx_shap]
        y_log_shap = y_log[valid_idx_shap]
        
        if X_shap.empty:
            print_status("Não há dados para SHAP após remover NaNs.", "⚠")
            return

        final_pipeline.fit(X_shap, y_log_shap)
        print_status("Modelo final treinado.")

        fitted_preprocessor = final_pipeline.named_steps['pp']
        fitted_model = final_pipeline.named_steps['m']

        print_status("Processando dados para SHAP...")
        
        feature_names_out = []
        try:
            feature_names_out = list(fitted_preprocessor.get_feature_names_out(feature_names_in))
        except Exception as e_names:
            print(f"       ⚠ Aviso: Falha ao obter nomes detalhados das features ({e_names}). Tentando fallback.")
            try:
                n_features_out = fitted_preprocessor.transform(X_shap.iloc[[0]]).shape[1]
                feature_names_out = [f'feature_{i}' for i in range(n_features_out)]
                print("       ... Usando nomes genéricos para SHAP.")
            except Exception as e_fallback:
                print(f"       ✗ Erro Crítico: Falha ao determinar número/nomes de features pós-transformação. Pulando SHAP. Erro: {e_fallback}")
                return

        X_processed = fitted_preprocessor.transform(X_shap)
        if len(feature_names_out) != X_processed.shape[1]:
             print(f"       ✗ Erro Crítico SHAP: Mismatch! Nomes={len(feature_names_out)}, Colunas={X_processed.shape[1]}. Pulando SHAP.")
             return

        shap_input_data = None
        is_dataframe = False
        try:
             X_processed_df = pd.DataFrame(X_processed, columns=feature_names_out).astype(float)
             X_processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
             shap_input_data = X_processed_df
             is_dataframe = True
             print_status("Usando DataFrame para SHAP.")
        except Exception as e_df:
             print(f"       ⚠ Aviso: Falha ao criar DataFrame para SHAP ({e_df}). Usando array numpy.")
             shap_input_data = X_processed

        print_status(f"Calculando valores SHAP... (Modelo: {type(fitted_model)}, Dados: {shap_input_data.shape})")

        shap_values = None
        if isinstance(fitted_model, (xgb.XGBRegressor, lgb.LGBMRegressor, cb.CatBoostRegressor, RandomForestRegressor)):
            explainer = shap.TreeExplainer(fitted_model)
            shap_values = explainer(shap_input_data)
        else:
            print(f"       ⚠ Modelo SHAP não suportado: {type(fitted_model)}")
            return

        if shap_values is not None:
            plot_kwargs = {'feature_names': feature_names_out} if not is_dataframe and hasattr(shap_values, 'feature_names') and shap_values.feature_names is None else {}
            print_status("Salvando plots SHAP...")
            plt.figure()
            shap.summary_plot(shap_values, shap_input_data, plot_type="dot", show=False, **plot_kwargs)
            plt.tight_layout()
            save_plot(plt.gcf(), shap_dir, 'shap_summary_plot')
            plt.figure()
            shap.summary_plot(shap_values, shap_input_data, plot_type="bar", show=False, **plot_kwargs)
            plt.tight_layout()
            save_plot(plt.gcf(), shap_dir, 'shap_importance_plot')
            print_status("SHAP salvo.")
        else:
            print_status("SHAP não calculado.", "⚠")
    
    except Exception as e:
        print(f"       ✗ Erro no SHAP: {e}")

# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================
def main():
    print_header("INICIANDO PIPELINE COMPLETA: ANÁLISE COFDI vs ERP")
    [create_dir(d) for d in DIRS.values()]
    print_status(f"Diretórios em: {DIRS['BASE']}")

    # ETAPAS 2 & 3
    df = load_and_clean_data(DATA_PATH)
    if df is None:
        print_header("FALHA: Carga/Limpeza de dados.")
        return
    df = feature_engineer(df)

    # ETAPA 1.5: Geração de Gráficos Estilo R
    print_header("[ETAPA 1.5/11] 🎨 Geração de Gráficos (Estilo R)")
    plot_transactions_by_investor(df, DIRS['R_STYLE'])
    plot_investment_by_sector(df, DIRS['R_STYLE'])
    plot_greenfield_donuts(df, DIRS['R_STYLE'])
    plot_world_maps(df, DIRS['R_STYLE'])
    plot_investment_by_region_over_time(df, DIRS['R_STYLE'])
    plot_sector_concentration_hhi(df, DIRS['R_STYLE'])
    plot_sector_sunburst(df, DIRS['R_STYLE'])
    plot_investment_distribution_over_time(df, DIRS['R_STYLE'])
    plot_investment_by_share_size(df, DIRS['R_STYLE'])
  
    plot_ggstats_greenfield(df, DIRS['R_STYLE'])
    plot_ggstats_bri(df, DIRS['R_STYLE'])

    # ETAPA 4 EDA & 4.5 Análise por Fase
    run_eda(df, DIRS['EDA'])
    run_phase_analysis(df)

    # Preparação ML
    TARGET_REG_LOG = 'log_Valor_USD'
    TARGET_CLF = 'Alvo_Adaptativo'
    
    # Features
    FEATURES_NUM_BASE = ['Year']
    FEATURES_CAT_BASE = ['Sector', 'Region', 'Tipo_Investimento', 'Fase_GG', 'post_ERP']
    
    
    FEATURES_NUM_AVANCADAS = [
        'valor_roll_mean_2', 'valor_roll_mean_3', 
        'valor_roll_std_2', 'valor_roll_std_3',
        'policy_cumulative', 'policy_interaction',
        'Sector_hierarchical', 'Region_hierarchical',
        'phase_1', 'phase_2', 'phase_3'
    ]
    
    if 'sector_year_int' in df.columns:
        FEATURES_NUM_AVANCADAS.append('sector_year_int')
    if 'region_policy_int' in df.columns:
        FEATURES_NUM_AVANCADAS.append('region_policy_int')

    # Combinar features
    all_num_features = FEATURES_NUM_BASE + [f for f in FEATURES_NUM_AVANCADAS if f in df.columns]
    all_cat_features = [f for f in FEATURES_CAT_BASE if f in df.columns]
    
    # Criar dataset ML
    cols_ml = list(set(
        all_num_features + all_cat_features + 
        [TARGET_REG_LOG, TARGET_CLF, 'Valor_USD']
    ))
    cols_ml = [col for col in cols_ml if col in df.columns]
    
    df_ml = df[cols_ml].copy()
    df_ml = df_ml.sort_values('Year').reset_index(drop=True)
    df_ml.dropna(subset=[TARGET_REG_LOG, TARGET_CLF], inplace=True)

    if df_ml.empty:
        print_status("Sem dados para ML", "✗")
        return

    # Preparar dados
    X_reg = df_ml[all_num_features + all_cat_features]
    y_reg_log = df_ml[TARGET_REG_LOG]
    
    X_clf = df_ml[all_num_features + all_cat_features]  
    y_clf = df_ml[TARGET_CLF]

    # Pipelines
    preprocessor_linear, preprocessor_tree = get_ml_pipelines(all_num_features, all_cat_features)

    if preprocessor_linear is None or preprocessor_tree is None:
        print_status("Erro pipelines", "✗")
        return

    # Modelos otimizados
    best_reg_model, best_reg_params = run_regression_otimizada(
        X_reg, y_reg_log, preprocessor_linear, preprocessor_tree, DIRS['REGRESSION']
    )
    
    best_clf_model, best_clf_params = run_classification_otimizada(
        X_clf, y_clf, preprocessor_linear, preprocessor_tree, DIRS['CLASSIFICATION']
    )

    # Séries temporais
    run_time_series(df, DIRS['TIMESERIES'])

    # Modelos causais
    run_causal_models(df_ml, preprocessor_tree, all_num_features, all_cat_features, DIRS['CAUSAL'])

    # SHAP
    fit_final_model_and_run_shap(
        best_reg_model, best_reg_params, 
        preprocessor_linear, preprocessor_tree, 
        X_reg, y_reg_log, 
        all_num_features + all_cat_features, 
        DIRS['SHAP']
    )

    print_header("✅ PIPELINE CONCLUÍDA!")

if __name__ == "__main__":
    main()