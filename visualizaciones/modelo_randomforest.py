import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def entrenar_y_graficar():
    # ===============================
    # Cargar datos
    # ===============================
    ruta = "visualizaciones/TotalFeatures-ISCXFlowMeter.csv"
    df = pd.read_csv(ruta)

    print(df.head(10))
    print(df.describe())
    print("Longitud del DataSet:", len(df))
    print("Número de características del DataSet:", len(df.columns))

    # ===============================
    # Limpiar y preparar datos
    # ===============================
    if "calss" not in df.columns:
        raise ValueError("❌ La columna 'calss' no se encuentra en el dataset.")

    # Si la columna objetivo es categórica (ej. 'benign', 'attack'), la codificamos
    if df["calss"].dtype == "object":
        le = LabelEncoder()
        df["calss"] = le.fit_transform(df["calss"])

    # Reemplazar valores infinitos y NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Verificamos que el dataset tenga registros
    if df.empty:
        raise ValueError("❌ El dataset está vacío después de limpiar los datos.")

    # ===============================
    # Separar variables
    # ===============================
    X = df.drop("calss", axis=1)
    y = df["calss"]

    # ===============================
    # División de datos
    # ===============================
    if len(df) < 10:
        test_size = 0.5  # si hay pocos registros
    else:
        test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ===============================
    # Escalado
    # ===============================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===============================
    # Entrenamiento del modelo
    # ===============================
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # ===============================
    # Métricas
    # ===============================
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # ===============================
    # Gráfica
    # ===============================
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, color="blue", label="Predicciones", alpha=0.6)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Línea ideal"
    )
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Predicciones")
    ax.set_title("Random Forest Regressor - Valores Reales vs Predichos")
    ax.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    imagen_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    plt.close(fig)

    # ===============================
    # Retornar resultados
    # ===============================
    resultados = {
        "df_head": df.head(10).to_html(classes="table table-striped", border=0),
        "df_describe": df.describe().to_html(classes="table table-bordered", border=0),
        "info_dataset": {
            "longitud": len(df),
            "caracteristicas": len(df.columns),
        },
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "grafica_valores_reales_predichos": imagen_base64,
    }

    return resultados
