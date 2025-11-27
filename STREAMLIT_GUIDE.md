# 🚀 Guía de Uso - Aplicación Streamlit

## ¿Qué acabas de crear?

Acabas de agregar una **interfaz web** a tu cotizador de préstamos PyME. En lugar de usar la terminal, ahora tus usuarios pueden:
- Llenar un formulario visual
- Ver resultados en tiempo real
- Obtener gráficos y métricas interactivas

## 📋 Pasos para usar la app de Streamlit

### 1. Instalar dependencias

```bash
pip install streamlit
```

O instala todo junto:
```bash
pip install -r requirements.txt
pip install streamlit
```

### 2. Asegúrate de tener los modelos entrenados

Si aún no has entrenado los modelos:
```bash
python train.py
```

Esto creará el archivo `sba_mexico_model.pkl` que la app necesita.

### 3. Ejecutar la aplicación web

```bash
streamlit run app.py
```

Esto abrirá automáticamente tu navegador en `http://localhost:8501`

### 4. Usar la interfaz

1. Llena los campos del formulario:
   - Monto del préstamo
   - Plazo en meses
   - Número de empleados
   - Sector SCIAN
   - Estado
   - Tasa de interés
   - Opciones adicionales (garantía, recesión)

2. Presiona el botón "Calcular Cotización"

3. Verás:
   - Categoría GPS (Ultra-Oro, Oro, Estándar, o Rechazo)
   - Probabilidad de Default
   - Pérdida Esperada
   - Pago Mensual
   - Detalles financieros completos

## 🌐 Desplegar en Streamlit Cloud (GRATIS)

### Opción 1: Streamlit Community Cloud

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona tu repositorio
5. Define:
   - Main file: `app.py`
   - Python version: 3.9+
6. Click en "Deploy"

**¡Listo!** Tu app estará disponible en una URL pública como:
`https://tu-usuario-nombre-repo.streamlit.app`

### Opción 2: Deployment Local (Red Local)

Para compartir en tu red local:
```bash
streamlit run app.py --server.address 0.0.0.0
```

Otros en tu red podrán acceder usando tu IP local.

## 📁 Estructura de archivos necesaria

```
tu-repositorio/
├── app.py                    # ← Nueva app de Streamlit
├── quoter.py                 # Tu lógica de cotización
├── features.py               # Procesamiento de features
├── models.py                 # Modelos ML
├── train.py                  # Entrenamiento
├── requirements.txt          # Dependencias
├── sba_mexico_model.pkl      # Modelos entrenados
└── data/
    └── sba_mexico_sintetico.csv  # Dataset
```

## 🎨 Personalización

Puedes personalizar la app editando `app.py`:

- **Colores y estilos**: Modifica la sección `st.markdown()` con CSS
- **Logo**: Agrega `st.image("tu-logo.png")` en el header
- **Más métricas**: Agrega gráficos con `st.line_chart()`, `st.bar_chart()`, etc.

## 🔧 Troubleshooting

### Error: "Modelos no encontrados"
**Solución:** Ejecuta `python train.py` primero

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solución:** `pip install streamlit`

### La app se ve diferente en el navegador
**Solución:** Presiona `Ctrl+Shift+R` para recargar sin caché

### Cambios en el código no se reflejan
**Solución:** Streamlit auto-detecta cambios. Si no, presiona "R" en la app o "Always rerun"

## 📊 Comparación: Terminal vs Web

| Característica | Terminal (`quoter.py`) | Streamlit (`app.py`) |
|----------------|----------------------|---------------------|
| **Interfaz** | Texto | Visual e interactiva |
| **Input** | Escribir valores | Formularios y sliders |
| **Output** | Texto plano | Gráficos y métricas |
| **Acceso** | Solo local | Puede ser web pública |
| **Usuarios** | Técnicos | Cualquiera |

## 🚀 Próximos pasos

1. **Agrega gráficos**: Usa `plotly` o `matplotlib` para visualizar la distribución de riesgo
2. **Historial**: Guarda cotizaciones previas en una base de datos
3. **Exportar PDF**: Permite descargar la cotización en PDF
4. **Comparación**: Compara múltiples escenarios lado a lado
5. **Dashboard**: Crea un dashboard de todas las cotizaciones

## 💡 Tips

- **Desarrollo**: Usa `streamlit run app.py --server.runOnSave true` para auto-reload
- **Debug**: Agrega `st.write()` para inspeccionar variables
- **Performance**: Usa `@st.cache_data` para cachear datos pesados
- **Secretos**: Usa `st.secrets` para API keys (no hardcodear)

---

**¿Necesitas ayuda?** Revisa la [documentación oficial de Streamlit](https://docs.streamlit.io)
