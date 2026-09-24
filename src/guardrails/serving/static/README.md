# `serving/static` — frontend de la consola de gobernanza

Assets servidos por `guardrails.serving.backend`. Sin build step: HTML, CSS y JS planos.

| Página | Ruta | Archivos |
|---|---|---|
| Análisis de prompts | `/` | `index.html`, `app.js`, `styles.css` |
| Panel de entrenamiento | `/train` | `train.html`, `train.js`, `train.css` |
| Panel de evaluación | `/eval` | `eval.html`, `eval.js`, `eval.css` |

`app.js` mantiene en memoria del navegador la latencia media de las peticiones de la sesión.
`train.js` y `eval.js` consultan periódicamente `/api/train/status` y `/api/eval/golden-status`.

No hay pruebas de frontend: la superficie es un panel de operación interno.
