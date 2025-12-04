# inspector_kit_refinado.py
# Usa el modelo refinado del kit para revisar completitud con la webcam

from ultralytics import YOLO
import cv2
from collections import Counter

# 1) Ruta a tu modelo refinado (best.pt NUEVO)
MODEL_PATH = r"C:\Users\Ultrabook\Documents\Universidad\Semestre 8\Vision artifical\kit_model_refinado\weights\best.pt"

# 2) Definición del kit esperado
#    Claves = nombres de clase EXACTOS que usa tu modelo
#    Valores = cantidad mínima esperada
KIT_DEFINITION = {
    "enchufe": 1,   # cargador / enchufe
    "headset": 1,   # audífonos tipo casco
    "lentes": 1,    # lentes
    "llave": 1,     # llave de casa
    "objects": 1,   # lápiz (lo etiquetaste como 'objects')
    "tarjeta": 1,   # tarjeta débito
    # la clase '0' la ignoramos, no es un objeto del kit
}

CONF_THRESHOLD = 0.4  # puedes jugar con este valor (0.25–0.5)


def analizar_frame(model, frame, kit_definition, conf_threshold=0.4):
    """
    Corre el modelo sobre un frame, dibuja cajas y calcula completitud del kit.
    Devuelve:
      - frame_annotated: imagen con cajas y textos
      - counts: dict con conteo por clase detectada
      - missing: lista de clases faltantes
    """
    # Ejecutar modelo
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    # Acceder a nombres de clases del modelo
    names = results.names  # dict: id -> nombre

    # Contador de objetos detectados por nombre de clase
    counts = Counter()

    # Dibujar detecciones
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_name = names.get(cls_id, str(cls_id))

            # Actualizar conteo solo si la clase está definida en el kit o es interesante
            counts[class_name] += 1

            # Dibujar caja
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Texto: nombre + confianza
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Determinar qué falta según KIT_DEFINITION
    missing = []
    for cls_name, expected_count in kit_definition.items():
        if counts[cls_name] < expected_count:
            missing.append(cls_name)

    # Escribir resumen en la parte superior
    status_text = ""
    if len(missing) == 0:
        status_text = "KIT COMPLETO"
        color_status = (0, 200, 0)  # verde
    else:
        status_text = "KIT INCOMPLETO: falta(n) " + ", ".join(missing)
        color_status = (0, 0, 255)  # rojo

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

    # También podemos mostrar los conteos
    y0 = 50
    for i, (cls_name, expected) in enumerate(kit_definition.items()):
        detected = counts.get(cls_name, 0)
        txt = f"{cls_name}: {detected}/{expected}"
        cv2.putText(frame, txt, (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, counts, missing


def demo_webcam():
    """Demo en tiempo real con la webcam."""
    print("Cargando modelo refinado...")
    model = YOLO(MODEL_PATH)
    print("Modelo cargado.")

    cap = cv2.VideoCapture(0)  # si no funciona, prueba con 1 o 2

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        frame_annotated, counts, missing = analizar_frame(
            model, frame, KIT_DEFINITION, CONF_THRESHOLD
        )

        cv2.imshow("Inspector de Completitud - Modelo Refinado", frame_annotated)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada.")


if __name__ == "__main__":
    demo_webcam()

