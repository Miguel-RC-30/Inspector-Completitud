# probar_modelo_refinado.py
# Probar el modelo refinado con una sola imagen

from ultralytics import YOLO
import cv2

# 1) Ruta al modelo refinado (best.pt NUEVO)
MODEL_PATH = r"C:\Users\Ultrabook\Documents\Universidad\Semestre 8\Vision artifical\kit_model_refinado\weights\best.pt"

# 2) Ruta a una imagen de prueba (por ejemplo, una de images/val)
IMAGE_PATH = r"C:\Users\Ultrabook\Documents\Universidad\Semestre 8\Vision artifical\kits_dataset\images\val\IMG-20251201-WA0088_jpg.rf.5df733576d2908050c0aa2fd70839154.jpg"

# 3) Carpeta donde se guardará la imagen anotada
OUTPUT_PROJECT = r"C:\Users\Ultrabook\Documents\Universidad\Semestre 8\Vision artifical"
OUTPUT_NAME = "pred_refinado_test"  # se creará una carpeta con este nombre


def main():
    # --- Cargar modelo ---
    print("Cargando modelo refinado...")
    model = YOLO(MODEL_PATH)

    # --- Hacer predicción y guardar imagen anotada ---
    print("Haciendo predicción sobre la imagen...")
    results = model.predict(
        source=IMAGE_PATH,
        save=True,                 # guarda la imagen anotada en disco
        project=OUTPUT_PROJECT,    # carpeta base de salida
        name=OUTPUT_NAME,          # subcarpeta de salida
        conf=0.4                   # umbral de confianza (puedes probar 0.25–0.5)
    )

    # --- Mostrar la imagen en una ventana (opcional) ---
    # results[0] es la primera (y única) imagen procesada
    annotated = results[0].plot()  # devuelve el frame con las cajas dibujadas

    cv2.imshow("Prediccion modelo refinado", annotated)
    print("Presiona cualquier tecla en la ventana de la imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nListo.")
    print("La imagen anotada se guardó en:")
    print(fr"{OUTPUT_PROJECT}\{OUTPUT_NAME}")


if __name__ == "__main__":
    main()
