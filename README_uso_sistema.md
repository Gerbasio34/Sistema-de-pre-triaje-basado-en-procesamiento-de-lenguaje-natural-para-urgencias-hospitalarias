# Sistema de Pre-Triaje ESI

## Arquitectura

El sistema integra tres componentes:

- **BioLORD-2023** — autocomplete semántico sobre vocabulario UMLS para estandarizar la descripción del paciente
- **BioBERT + CRF** — extracción de síntomas en lenguaje coloquial mediante NER
- **SapBERT fine-tuned** — clasificador ESI con fusión de texto y variables clínicas mediante cross-attention

## Requisitos

- Python 3.10 o superior
- GPU recomendada (CUDA). Funciona en CPU pero el arranque es más lento.
- ~6 GB de RAM mínimo para cargar los modelos

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Gerbasio34/NOMBRE_REPO
cd NOMBRE_REPO
```

### 2. Crear un entorno virtual e instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate        

pip install -r requirements.txt
```

### 3. Descargar los modelos

Los modelos están alojados en Hugging Face Hub. Ejecuta el script de descarga una sola vez **desde la raíz del repositorio**:

```bash
python3 preparar_modelos.py
```

Esto descargará automáticamente los tres modelos en la carpeta `FRONTEND_DEMO/`. Puede tardar varios minutos dependiendo de la conexión (~4 GB en total).

### 4. Arrancar el sistema

```bash
cd FRONTEND_DEMO
python3 API.py
```

El primer arranque tarda unos segundos mientras carga los modelos en memoria.

### 5. Abrir en el navegador

```
http://localhost:8000
```

La pantalla de inicio permite elegir entre la vista de paciente y la vista de enfermera. Se pueden abrir ambas en pestañas separadas simultáneamente.

## Uso

**Vista paciente** (`http://localhost:8000/patient`): el paciente introduce sus síntomas mediante autocomplete UMLS o texto libre, y sus constantes vitales subjetivas en escala 1-5.

**Vista enfermera** (`http://localhost:8000/nurse`): muestra la cola de pacientes ordenada por urgency score con el nivel ESI predicho, síntomas detectados y probabilidades por clase. Se actualiza automáticamente cada 3 segundos.

## Notas

- El sistema está diseñado para adultos (18-91 años) que acuden por sus propios medios
- Solo acepta entrada en inglés
- Las constantes vitales son subjetivas y autorreportadas mediante escala discreta 1-5
- El enfermero retiene siempre la autoridad clínica final
