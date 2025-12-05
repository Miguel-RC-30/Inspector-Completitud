# probar_modelo_refinado_completo.py
# Probar el modelo refinado con una sola imagen + análisis de completitud del kit

from ultralytics import YOLO
import cv2
from collections import Counter

# 1) Ruta al modelo refinado (best.pt NUEVO)
MODEL_PATH = r"C:\Users\HP\OneDrive\Escritorio\UNIVERSIDAD\8vo Semestre\Visión Artificial\Parcial 3\best\best.pt"

# 2) Ruta a una imagen de prueba (por ejemplo, una de images/val)
#IMAGE_PATH = r"C:\Users\HP\OneDrive\Imágenes\kits_dataset\Imagen 1.jpg"
#IMAGE_PATH = r"C:\Users\HP\OneDrive\Imágenes\kits_dataset\Imagen 2.jpg"
IMAGE_PATH = r"C:\Users\HP\OneDrive\Imágenes\kits_dataset\Imagen 3.jpg"

# 3) Carpeta donde se guardará la imagen anotada
OUTPUT_PROJECT = r"C:\Users\HP\OneDrive\Escritorio\UNIVERSIDAD\8vo Semestre\Visión Artificial\Segunda Imagen"
OUTPUT_NAME = "pred_refinado_test"  # se creará una carpeta con este nombre

# 4) Definición del kit esperado con colores únicos (BGR format)
KIT_DEFINITION = {
    "enchufe": {"cantidad": 1, "color": (0, 255, 255)},    # amarillo/cyan
    "headset": {"cantidad": 1, "color": (255, 0, 255)},    # magenta
    "lentes": {"cantidad": 1, "color": (0, 165, 255)},     # naranja
    "llave": {"cantidad": 1, "color": (255, 255, 0)},      # cyan
    "objects": {"cantidad": 1, "color": (0, 255, 0)},      # verde
    "tarjeta": {"cantidad": 1, "color": (255, 0, 0)},      # azul
}

CONF_THRESHOLD = 0.4


def analizar_imagen(model, image_path, kit_definition, conf_threshold=0.4):
    """
    Corre el modelo sobre una imagen, dibuja cajas y calcula completitud del kit.
    Devuelve:
      - frame_annotated: imagen con cajas, textos y estado del kit
      - counts: dict con conteo por clase detectada
      - missing: lista de clases faltantes
    """
    # Leer imagen
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
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
            
            # Actualizar conteo
            counts[class_name] += 1
            
            # Obtener color específico para esta clase
            color = (0, 255, 0)  # verde por defecto
            if class_name in kit_definition:
                color = kit_definition[class_name]["color"]
            
            # Dibujar caja
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Preparar el texto
            label = f"{class_name} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Decidir si poner el texto arriba o abajo de la caja
            if y1 - text_height - 10 < 0:
                # Si está muy arriba, poner el texto DENTRO de la caja (arriba)
                text_y1 = y1 + text_height + 5
                text_y2 = y1
                text_pos_y = y1 + text_height
            else:
                # Poner el texto ARRIBA de la caja (normal)
                text_y1 = y1 - text_height - 10
                text_y2 = y1
                text_pos_y = y1 - 5
            
            # Fondo para el texto
            cv2.rectangle(frame, (x1, text_y1), 
                         (x1 + text_width, text_y2), color, -1)
            
            # Texto: nombre + confianza en negro para mejor contraste
            cv2.putText(frame, label, (x1, text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Determinar qué falta según KIT_DEFINITION
    missing = []
    for cls_name, info in kit_definition.items():
        expected_count = info["cantidad"]
        if counts[cls_name] < expected_count:
            missing.append(cls_name)
    
    # Obtener dimensiones de la imagen
    height, width = frame.shape[:2]
    
    # Crear panel lateral derecho
    panel_width = 280
    panel_height = height
    
    # Expandir el canvas para agregar el panel lateral
    frame_with_panel = cv2.copyMakeBorder(
        frame, 0, 0, 0, panel_width,
        cv2.BORDER_CONSTANT, value=(40, 40, 40)
    )
    
    # Posición inicial del texto en el panel
    panel_x = width + 15
    y_current = 40
    
    # Escribir estado del kit
    if len(missing) == 0:
        status_text = "KIT COMPLETO"
        color_status = (0, 255, 0)  # verde
        cv2.circle(frame_with_panel, (panel_x - 5, y_current - 5), 8, color_status, -1)
    else:
        status_text = "KIT INCOMPLETO"
        color_status = (0, 0, 255)  # rojo
        cv2.circle(frame_with_panel, (panel_x - 5, y_current - 5), 8, color_status, -1)
    
    cv2.putText(frame_with_panel, status_text, (panel_x + 15, y_current),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_current += 50
    
    # Línea separadora
    cv2.line(frame_with_panel, (panel_x - 10, y_current - 20), 
             (panel_x + panel_width - 20, y_current - 20), (100, 100, 100), 1)
    
    # Título de la sección
    cv2.putText(frame_with_panel, "OBJETOS:", (panel_x, y_current),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_current += 35
    
    # Mostrar los conteos individuales
    for cls_name, info in kit_definition.items():
        expected = info["cantidad"]
        color_obj = info["color"]
        detected = counts.get(cls_name, 0)
        
        # Dibujar círculo con el color del objeto
        cv2.circle(frame_with_panel, (panel_x, y_current - 5), 7, color_obj, -1)
        cv2.circle(frame_with_panel, (panel_x, y_current - 5), 7, (255, 255, 255), 1)
        
        # Texto del objeto
        txt = f"{cls_name}: {detected}/{expected}"
        cv2.putText(frame_with_panel, txt, (panel_x + 20, y_current),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        y_current += 30
    
    # Si hay objetos faltantes, listarlos
    if len(missing) > 0:
        y_current += 20
        cv2.line(frame_with_panel, (panel_x - 10, y_current - 15), 
                 (panel_x + panel_width - 20, y_current - 15), (100, 100, 100), 1)
        y_current += 10
        
        cv2.putText(frame_with_panel, "FALTANTES:", (panel_x, y_current),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y_current += 30
        
        for item in missing:
            cv2.putText(frame_with_panel, f"- {item}", (panel_x + 5, y_current),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            y_current += 25
    
    return frame_with_panel, counts, missing
    
    return frame, counts, missing


def main():
    # --- Cargar modelo ---
    print("Cargando modelo refinado...")
    model = YOLO(MODEL_PATH)
    print("Modelo cargado.")
    
    # --- Analizar imagen con completitud del kit ---
    print("\nAnalizando imagen...")
    frame_annotated, counts, missing = analizar_imagen(
        model, IMAGE_PATH, KIT_DEFINITION, CONF_THRESHOLD
    )
    
    # --- Mostrar resultados en consola ---
    print("\n" + "="*50)
    print("RESULTADOS DEL ANÁLISIS")
    print("="*50)
    print("\nObjetos detectados:")
    for cls_name, info in KIT_DEFINITION.items():
        expected = info["cantidad"]
        detected = counts.get(cls_name, 0)
        status = "✓" if detected >= expected else "✗"
        print(f"  {status} {cls_name}: {detected}/{expected}")
    
    if len(missing) == 0:
        print("\n✓ KIT COMPLETO - Todos los objetos están presentes")
    else:
        print(f"\n✗ KIT INCOMPLETO - Faltan: {', '.join(missing)}")
    print("="*50)
    
    # --- Guardar imagen anotada ---
    import os
    
    # Buscar el siguiente número disponible
    base_name = OUTPUT_NAME
    counter = 0
    output_dir = os.path.join(OUTPUT_PROJECT, base_name)
    
    # Si el directorio ya existe, buscar el siguiente número
    while os.path.exists(output_dir):
        counter += 1
        output_dir = os.path.join(OUTPUT_PROJECT, f"{base_name}{counter}")
    
    # Crear el directorio
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "imagen_analizada.jpg")
    cv2.imwrite(output_path, frame_annotated)
    
    # Nombre del directorio para mostrar
    folder_name = base_name if counter == 0 else f"{base_name}{counter}"
    print(f"\nImagen anotada guardada en: {output_path}")
    print(f"Carpeta creada: {folder_name}")
    
    # --- Mostrar imagen en ventana ---
    cv2.imshow("Análisis de Completitud - Modelo Refinado", frame_annotated)
    print("\nPresiona cualquier tecla en la ventana de la imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nListo.")


if __name__ == "__main__":
    main()
