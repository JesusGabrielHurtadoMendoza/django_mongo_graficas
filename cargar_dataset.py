# ejecutar: python cargar_dataset.py
import pandas as pd
from pymongo import MongoClient

# Cambia estos valores según tu instalación (Atlas, local o Docker)
# Conexión local sin autenticación
client = MongoClient("mongodb://localhost:27017/")

db = client["mi_basedatos"]
coleccion = db["dataset"]

# Cargar CSV (ajusta el nombre del archivo)
df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')   # o 'datos.xlsx' con pd.read_excel()

# Opcional: limpiar duplicados / NaNs antes
records = df.to_dict('records')

# Insertar (si ya insertaste antes, evita duplicados; aquí borramos coleccion y cargamos nueva)
coleccion.delete_many({})
coleccion.insert_many(records)
print("✅ Dataset cargado en MongoDB. Registros:", coleccion.count_documents({}))
