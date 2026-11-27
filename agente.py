import os
from pathlib import Path

from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from google.adk.agents import Agent

# Carga las credenciales del archivo .env (está en esta misma carpeta)
load_dotenv(Path(__file__).parent / ".env")

# Lee los valores del .env
proveedor = os.getenv("PROVEEDOR", "gemini")  # gemini, openai u ollama
clave_openai = os.getenv("OPEN_API_KEY")
clave_ollama = os.getenv("OLLAMA_API_KEY")

if proveedor == "openai":
    from google.adk.models.lite_llm import LiteLlm

    modelo = LiteLlm(model="openai/gpt-5.6-luna", api_key=clave_openai)
elif proveedor == "ollama":
    from google.adk.models.lite_llm import LiteLlm

    # Ollama usa un formato compatible con OpenAI,
    # por eso el modelo empieza con "openai/"
    modelo = LiteLlm(
        model="openai/gemma4:31b",
        api_base="https://ollama.com/v1",  # dirección de la nube de Ollama
        api_key=clave_ollama,
    )
else:
    # Gemini lee la clave GEMINI_API_KEY que cargó load_dotenv
    modelo = "gemini-flash-lite-latest"

# Carpeta donde están los archivos CSV (junto a este archivo)
carpeta = Path(__file__).parent / "Datos"


# Herramientas: funciones normales de Python.
# Cada una lee el CSV que necesita al empezar y, si cambia algo, lo guarda al terminar.
def viajes_disponibles() -> dict:
    """Lista los viajes publicados que todavía tienen asientos libres."""
    viajes = pd.read_csv(carpeta / "viajes.csv")
    viajes_libres = viajes[
        (viajes["estado"] == "Publicado") & (viajes["asientos_disponibles"] > 0)
    ]
    return {
        "cantidad": len(viajes_libres),
        "viajes": viajes_libres.head(10).to_dict(orient="records"),
    }


def buscar_usuario(nombre: str) -> dict:
    """Busca usuarios por nombre o parte del nombre. Ejemplo: 'Ana'."""
    usuarios = pd.read_csv(carpeta / "usuarios.csv")
    usuarios_encontrados = usuarios[usuarios["nombre"].str.contains(nombre, case=False)]
    return {"usuarios": usuarios_encontrados.fillna("").to_dict(orient="records")}


def reservas_de_pasajero(id_pasajero: str) -> dict:
    """Devuelve las reservas de un pasajero. Ejemplo de id: 'U002'."""
    reservas = pd.read_csv(carpeta / "reservas.csv")
    reservas_del_pasajero = reservas[reservas["id_pasajero"] == id_pasajero]
    return {"reservas": reservas_del_pasajero.fillna("").to_dict(orient="records")}


def detalle_viaje(id_viaje: str) -> dict:
    """Muestra el detalle de un viaje por su código. Ejemplo: 'VJ001'."""
    viajes = pd.read_csv(carpeta / "viajes.csv")
    usuarios = pd.read_csv(carpeta / "usuarios.csv")
    vehiculos = pd.read_csv(carpeta / "vehiculos.csv")

    viaje = viajes[viajes["id_viaje"] == id_viaje]
    if viaje.empty:
        return {"error": "No existe ese viaje"}

    # El conductor trae el id de su vehículo
    conductor = usuarios[usuarios["id_usuario"] == viaje.iloc[0]["id_conductor"]]
    vehiculo = vehiculos[vehiculos["id_vehiculo"] == conductor.iloc[0]["id_vehiculo"]]

    return {
        "viaje": viaje.to_dict(orient="records"),
        "conductor": conductor.fillna("").to_dict(orient="records"),
        "vehiculo": vehiculo.to_dict(orient="records"),
    }


def detalle_reserva(id_reserva: str) -> dict:
    """Muestra el detalle de una reserva y su pago. Ejemplo: 'RES001'."""
    reservas = pd.read_csv(carpeta / "reservas.csv")
    viajes = pd.read_csv(carpeta / "viajes.csv")
    pagos = pd.read_csv(carpeta / "pagos.csv")

    reserva = reservas[reservas["id_reserva"] == id_reserva]
    if reserva.empty:
        return {"error": "No existe esa reserva"}

    # El monto a pagar es el costo por persona del viaje
    viaje = viajes[viajes["id_viaje"] == reserva.iloc[0]["id_viaje"]]
    monto_a_pagar = float(viaje.iloc[0]["costo_persona"])

    return {
        "reserva": reserva.fillna("").to_dict(orient="records"),
        "monto_a_pagar_soles": monto_a_pagar,
        "pagos": pagos[pagos["id_reserva"] == id_reserva].to_dict(orient="records"),
    }


def resumen_ingresos() -> dict:
    """Calcula el total cobrado y el total pendiente de cobro."""
    pagos = pd.read_csv(carpeta / "pagos.csv")
    total_cobrado = pagos[pagos["estado"] == "Completado"]["monto"].sum()
    total_pendiente = pagos[pagos["estado"] == "Pendiente"]["monto"].sum()
    return {"total_cobrado": float(total_cobrado), "total_pendiente": float(total_pendiente)}


def ingresos_por_metodo() -> dict:
    """Suma lo cobrado agrupado por método de pago (útil para graficar)."""
    pagos = pd.read_csv(carpeta / "pagos.csv")
    pagos_realizados = pagos[pagos["estado"] == "Completado"]
    total_por_metodo = pagos_realizados.groupby("tipo_pago")["monto"].sum()
    return {
        "etiquetas": total_por_metodo.index.tolist(),
        "valores": total_por_metodo.tolist(),
    }


def reservas_por_estado() -> dict:
    """Cuenta las reservas según su estado (útil para graficar)."""
    reservas = pd.read_csv(carpeta / "reservas.csv")
    total = reservas.groupby("estado")["id_reserva"].count()
    return {
        "etiquetas": total.index.tolist(),
        "valores": total.tolist(),
    }


def crear_grafico(
    titulo: str, tipo: str, etiquetas: list[str], valores: list[float]
) -> dict:
    """Muestra un gráfico al usuario. Úsala cuando pidan graficar algo.

    Args:
        titulo: título del gráfico.
        tipo: 'bar' (barras), 'line' (líneas) o 'pie' (torta).
        etiquetas: nombres de cada categoría.
        valores: número de cada categoría, en el mismo orden.
    """
    return {
        "titulo": titulo,
        "tipo": tipo,
        "etiquetas": etiquetas,
        "valores": valores,
    }


def registrar_reserva(id_pasajero: str, id_viaje: str) -> dict:
    """Registra una reserva nueva.
    Ejemplo: id_pasajero 'U002', id_viaje 'VJ001'."""
    # 1. Leer las tablas
    reservas = pd.read_csv(carpeta / "reservas.csv")
    viajes = pd.read_csv(carpeta / "viajes.csv")

    # 2. Armar la fila nueva
    reserva_nueva = {
        "id_reserva": "RES" + str(len(reservas) + 1),   # por ejemplo RES1404
        "id_viaje": id_viaje,
        "id_pasajero": id_pasajero,
        "estado": "Confirmada",
        "calificacion": "",
    }

    # 3. Pegarla al final de la tabla y guardar
    reservas = pd.concat([reservas, pd.DataFrame([reserva_nueva])])
    reservas.to_csv(carpeta / "reservas.csv", index=False)

    # 4. Quitar un asiento al viaje
    viajes.loc[viajes["id_viaje"] == id_viaje, "asientos_disponibles"] -= 1
    viajes.to_csv(carpeta / "viajes.csv", index=False)
    return {"mensaje": "Reserva registrada", "id_reserva": reserva_nueva["id_reserva"]}


def registrar_pago(id_reserva: str, monto: float, metodo: str) -> dict:
    """Registra el pago de una reserva. Métodos: Yape, Plin o Efectivo.
    Ejemplo: id_reserva 'RES1404', monto 4.5, metodo 'Yape'."""
    # 1. Leer la tabla y quitar el pago anterior de esta reserva (si había uno pendiente)
    pagos = pd.read_csv(carpeta / "pagos.csv")
    pagos = pagos[pagos["id_reserva"] != id_reserva]

    # 2. Armar la fila nueva
    pago_nuevo = {
        "id_pago": "PAG" + id_reserva[3:],   # el pago de RES1404 es PAG1404
        "id_reserva": id_reserva,
        "monto": monto,
        "tipo_pago": metodo,
        "estado": "Completado",
        "fecha_pago": str(datetime.now())[:16],
    }

    # 3. Pegarla al final de la tabla y guardar
    pagos = pd.concat([pagos, pd.DataFrame([pago_nuevo])])
    pagos.to_csv(carpeta / "pagos.csv", index=False)
    return {"mensaje": "Pago registrado"}


# Manual de comportamiento del agente, dividido en secciones con reglas cortas
INSTRUCCION = """
ROL
Eres un agente de un servicio de carpooling universitario.
Respondes en español, con frases cortas, sin listas ni símbolos,
porque tu respuesta puede leerse en voz alta.

REGLAS GENERALES
1. Usa siempre las herramientas para consultar los datos.
2. No inventes información.
3. Los montos están en soles (S/).

USUARIOS
- Nunca le pidas el id al usuario: él te da el nombre.
- Con el nombre, usa buscar_usuario para obtener el id y úsalo en las demás herramientas.
- Solo los usuarios de tipo Pasajero pueden hacer reservas.

RESERVAS
- Para registrar una reserva necesitas: pasajero y viaje.
  Pregunta lo que falte.
- Antes de registrar, comprueba con viajes_disponibles que el viaje tenga asientos libres.

PAGOS
- Para ver el monto a pagar o si una reserva ya está pagada, usa detalle_reserva.
- Métodos de pago válidos: Yape, Plin y Efectivo.
  Escríbelos exactamente así.

CONFIRMACIÓN
- Antes de registrar una reserva o un pago, repite los datos y pregunta: ¿Confirmas?
- Registra solo si el usuario responde que sí.

GRÁFICOS
- Obtén los datos con una herramienta y luego llama a crear_grafico.
"""


root_agent = Agent(
    model=modelo,  # el modelo de lenguaje
    name="agente_carpooling",
    instruction=INSTRUCCION,  # el manual de arriba
    tools=[
        viajes_disponibles,
        buscar_usuario,
        reservas_de_pasajero,
        detalle_viaje,
        detalle_reserva,
        resumen_ingresos,
        ingresos_por_metodo,
        reservas_por_estado,
        crear_grafico,
        registrar_reserva,
        registrar_pago,
    ],
)