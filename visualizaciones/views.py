import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pandas import DataFrame


def graficas(request):
    # 1️⃣ Cargar el dataset
    df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')

    # 2️⃣ Mostrar df.head(10) (permitiendo editar o no)
    head_html = df.head(10).to_html(classes="table table-striped", border=0)

    # 3️⃣ Mostrar df.describe()
    describe_html = df.describe().to_html(classes="table table-bordered", border=0)

    # 4️⃣ Mostrar df.info()
    buffer_info = io.StringIO()
    df.info(buf=buffer_info)
    info_str = buffer_info.getvalue().replace("\n", "<br>")

    # 5️⃣ Longitud y número de características
    longitud = len(df)
    num_caract = len(df.columns)

    # 6️⃣ Conteo de clases
    conteo_clases = df["calss"].value_counts().to_frame().to_html(classes="table table-sm", border=0)

    # 7️⃣ Separar variables independientes y dependiente
    X = df.drop("calss", axis=1)
    y = df["calss"]

    # 8️⃣ Calcular correlaciones
    corr_matrix = X.corr()

    # Añadir columna de clase al final para calcular correlación (aunque no numérica)
    df_corr = df.copy()
    df_corr["calss"] = pd.factorize(df_corr["calss"])[0]
    corr_matrix_full = df_corr.corr()

    # Correlación con la clase
    corr_with_class = corr_matrix_full["calss"].sort_values(ascending=False).to_frame().to_html(classes='table table-bordered')

    # Variables con correlación > 0.05
    corr_pos = corr_matrix_full[corr_matrix_full["calss"] > 0.05]["calss"].dropna().to_frame().to_html(classes='table table-bordered')

    # 9️⃣ Preparar train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔟 Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_train_scaled_head = X_train_scaled.head(5).to_html(classes='table table-striped')

    # 1️⃣1️⃣ Modelos RandomForestClassifier
    clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    y_pred_train = clf_rnd.predict(X_train)
    y_pred_val = clf_rnd.predict(X_test)

    f1_train = f1_score(y_train, y_pred_train, average="weighted", zero_division=1)
    f1_val = f1_score(y_test, y_pred_val, average="weighted", zero_division=1)

    # 1️⃣2️⃣ Modelo RandomForestRegressor
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    rf_regressor.fit(X_train, pd.factorize(y_train)[0])
    y_pred_reg = rf_regressor.predict(X_test)

    mse = mean_squared_error(pd.factorize(y_test)[0], y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(pd.factorize(y_test)[0], y_pred_reg)

    # 1️⃣3️⃣ Gráfica valores reales vs predichos
    plt.figure(figsize=(7, 6))
    plt.scatter(pd.factorize(y_test)[0], y_pred_reg, color='blue', alpha=0.7)
    plt.plot([pd.factorize(y_test)[0].min(), pd.factorize(y_test)[0].max()],
             [pd.factorize(y_test)[0].min(), pd.factorize(y_test)[0].max()],
             'r--')
    plt.xlabel("Valores Reales (factorizados)")
    plt.ylabel("Predicciones")
    plt.title("Random Forest Regressor - Valores Reales vs Predichos")

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    grafica_base64 = base64.b64encode(image_png).decode('utf-8')

    # 🧠 Renderizar todo en HTML
    contexto = {
        "head_html": head_html,
        "describe_html": describe_html,
        "info_str": info_str,
        "longitud": longitud,
        "num_caract": num_caract,
        "conteo_clases": conteo_clases,
        "corr_with_class": corr_with_class,
        "corr_pos": corr_pos,
        "X_train_scaled_head": X_train_scaled_head,
        "f1_train": f1_train,
        "f1_val": f1_val,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "grafica_base64": grafica_base64,
    }

    return render(request, 'visualizaciones/graficas_modelo.html', contexto)
