# 🧠 Inspector de Completitud de Kit

## ✨ Descripción del Proyecto

Este repositorio alberga el proyecto **Inspector de Completitud de Kit**, una aplicación de **Visión Artificial** implementada con la librería **YOLOv8**.

El objetivo es detectar la presencia de un conjunto de objetos predefinidos (un "kit") en tiempo real a través de una webcam, o en imágenes estáticas. El sistema realiza una verificación comparando la detección con una lista de objetos esperados y genera un reporte de tipo *Pass/Fail* (pasa/falla).

Los objetos detectados por el modelo entrenado incluyen: `enchufe`, `headset`, `lentes`, `llave`, `lápiz` y `tarjeta`.

---

## 📖 Tabla de Contenidos

* [1. Requisitos Previos](#1-requisitos-previos)
* [2. Configuración del Entorno (Anaconda)](#2-configuración-del-entorno-anaconda)
    * [2.1 Creación del Entorno](#21-creación-del-entorno)
    * [2.2 Activación del Entorno](#22-activación-del-entorno)
    * [2.3 Instalación de Dependencias](#23-instalación-de-dependencias)
* [3. Ejecución del Código](#3-ejecución-del-código)
    * [3.1 Rutas Importantes](#31-rutas-importantes-model_path-y-image_path)
    * [3.2 Prueba con Imagen Estática](#32-prueba-con-imagen-estática)
    * [3.3 Inspección con Webcam](#33-inspección-con-webcam)
* [4. Video de Presentación](#4-video-de-presentación)
* [5. Integrantes](#5-integrantes)
* [6. Licencia](#6-licencia)

---

## 1. Requisitos Previos

Antes de clonar y ejecutar este repositorio, asegúrate de contar con lo siguiente:

### 💻 Sistema operativo

- Windows 10/11, macOS o alguna distribución de Linux.
- Se recomienda un equipo con al menos **8 GB de RAM** para trabajar cómodamente con modelos de visión artificial.

### 🐍 Python y Anaconda

- **Python 3.10 o 3.11** (el entorno se creará con esta versión).
- **Anaconda o Miniconda** instalado, para gestionar entornos y dependencias de forma aislada.  
  > Si nunca has usado Anaconda, en el siguiente apartado se explica cómo crear un entorno desde cero.

### 📦 Herramientas adicionales

- (Opcional pero recomendado) **Git** para clonar el repositorio y gestionar versiones.
- Un editor de código a elección:
  - Visual Studio Code, PyCharm, o similar.

### 🎥 Hardware para las pruebas

- Para el script de inspección con **webcam**:
  - Cámara web funcional (integrada o externa).
- Para el script de prueba con **imagen estática**:
  - Al menos una imagen de prueba donde aparezcan los objetos del kit.

### 🤖 Modelo entrenado

- Archivo de modelo YOLOv8 entrenado (por ejemplo: `best.pt`), resultado del entrenamiento del proyecto.
- Este archivo puede estar ubicado en **cualquier carpeta** de tu equipo; más adelante se explicará cómo configurar la ruta (`MODEL_PATH`) para que el código lo encuentre correctamente.

---

## 2. Configuración del Entorno (Anaconda)

Para asegurar una ejecución estable y evitar conflictos entre versiones de librerías, este proyecto utiliza un entorno independiente creado con **Anaconda** o **Miniconda**.

A continuación se describe el proceso para crear, activar e instalar las dependencias del entorno.

### 2.1 Creación del Entorno

Puedes crear un entorno con el nombre que tú quieras.  
A modo de ejemplo, aquí se crea uno llamado **kit-inspector**, pero puedes reemplazarlo por cualquier otro nombre.

```bash
conda create -n kit-inspector python=3.11
```

- `-n kit-inspector` → nombre del entorno (puedes cambiarlo).
- `python=3.11` → versión recomendada para compatibilidad con Ultralytics y OpenCV.

### 2.2 Activación del Entorno

Una vez creado el entorno, actívalo con:

```bash
conda activate kit-inspector
```

Cada vez que desees ejecutar los scripts del proyecto, debes asegurarte de tener este entorno activado.

### 2.3 Instalación de Dependencias

Con el entorno ya activado, instala las librerías necesarias:

```bash
pip install ultralytics opencv-python
```

Dependiendo de tu proyecto, también puedes instalar:

```bash
pip install numpy matplotlib
```

**Nota:**
- Ultralytics descarga automáticamente los componentes necesarios de YOLOv8 y administra internamente las dependencias del modelo.
- Si el usuario no tiene una webcam o no desea instalar OpenCV completo, puede omitirlo; sin embargo, es obligatorio para ejecutar el script de inspección con cámara.

---

## 3. Ejecución del Código

Este proyecto incluye dos formas de ejecutar el inspector de completitud:

1. **Prueba con imagen estática** → [`inspector_foto.py`](inspector_foto.py)
2. **Inspección en tiempo real con webcam** → [`inspector webcam.py`](inspector webcam.py)

Antes de ejecutar cualquier script, asegúrate de:

- Tener el entorno activado (`conda activate kit-inspector`)
- Colocar la ruta correcta del modelo (`MODEL_PATH`)
- En el caso del script por imagen, definir la ruta de la imagen (`IMAGE_PATH`)

### 3.1 Rutas Importantes (MODEL_PATH y IMAGE_PATH)

Ambas rutas son totalmente **personalizables**.  
El modelo **NO necesita estar en una carpeta fija**, y la imagen **puede estar en cualquier ubicación** de tu computador.

Ejemplo de configuración:

```python
MODEL_PATH = r"C:\ruta\hacia\tu\modelo\best.pt"
IMAGE_PATH = r"C:\ruta\de\la\imagen\foto1.jpg"
```

### 3.2 Prueba con Imagen Estática

Ejecuta el script:

```bash
python inspector_foto.py
```

El script:
- Carga el modelo YOLOv8.
- Procesa la imagen indicada en `IMAGE_PATH`.
- Muestra detecciones y el reporte de objetos faltantes.

### 3.3 Inspección con Webcam

Ejecuta:

```bash
python inspector_webcam.py
```

El script:
- Abre la cámara web.
- Detecta objetos en tiempo real.
- Compara con la lista esperada del kit.
- Muestra un estado PASS / FAIL en pantalla.

**Para cerrar la ventana, presiona Q.**

---

## 4. Video de Presentación

El siguiente video resume la metodología, funcionamiento del sistema de inspección y principales resultados obtenidos:

🔗 **Video de presentación:**  
[Haz clic aquí para ver el video](URL_DEL_VIDEO)

---

## 5. Integrantes

| Nombre | Email |
|--------|-------|
| Bastián Gálvez | bastian.galvez@mayor.cl |
| Bruno Meza | bruno.meza@mayor.cl |
| Miguel Retamal | miguel.retamal@mayor.cl |
| Diego Villalón | diego.villalong@mayor.cl |

---
