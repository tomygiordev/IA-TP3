# Política de commits por feature

Este documento amplía la regla general definida en `AGENTS.md`.

## Objetivo

Mantener un historial Git que sirva para revisar, explicar, defender y reconstruir el trabajo realizado en el repositorio.

La idea no es solo "guardar cambios", sino dejar evidencia útil para cualquier integrante o revisor.

## Principio central

Cada feature implementada debe cerrarse con un commit independiente, escrito completamente en español y con el mayor nivel de detalle razonable.

## Qué debe incluir un buen commit

Un buen commit de este repositorio debe responder, sin obligar a abrir el diff completo, estas preguntas:

1. Qué problema se resolvió.
2. Qué solución se implementó.
3. Qué partes del proyecto se tocaron.
4. Qué comprobaciones se hicieron.
5. Qué límites o deudas quedaron.

## Plantilla recomendada

```text
tipo: resumen breve de la feature

Detalle:
- Qué se agregó, cambió o eliminó.
- Archivos principales tocados.
- Decisiones técnicas importantes.
- Motivo de la implementación.

Verificación:
- Tests, comandos o pruebas manuales ejecutadas.
- Resultado de esas verificaciones.
- Si algo no se pudo verificar, por qué.

Notas:
- Riesgos.
- Supuestos.
- Mejoras futuras o límites actuales.
```

## Casos concretos

### Nueva funcionalidad

Usar normalmente `feat`.

Ejemplo:

```text
feat: agregar entrenamiento completo de experimentos MLP

Detalle:
- Se implementó el flujo de entrenamiento para múltiples configuraciones de hiperparámetros.
- Se agregó selección automática del mejor modelo por F1 de validación.
- Se generaron archivos de resultados y gráficos de curvas de aprendizaje.

Verificación:
- Se ejecutó el entrenamiento completo con python src\train.py.
- Se revisó la generación de metrics.csv y los gráficos en results/.

Notas:
- El dataset se descarga desde Hugging Face en tiempo de ejecución.
```

### Documentación

Usar normalmente `docs`.

### Corrección

Usar normalmente `fix`.

### Refactor

Usar `refactor` cuando no cambie el comportamiento visible o cuando el foco principal sea ordenar el código.

## Qué no hacer

- No agrupar varias features ajenas en un mismo commit.
- No mezclar documentación, refactor y funcionalidad si pueden separarse de forma razonable.
- No hacer commits con descripciones vacías o genéricas.
- No incluir secretos, caches, `.venv`, binarios irrelevantes o archivos temporales.
- No seleccionar archivos sin revisar qué entra.

## Criterio de separación

Si un reviewer pudiera comentar una parte sin comentar la otra, probablemente convenga separarlas en commits distintos.

## Relación con agentes

`AGENTS.md` es la fuente general y portátil para cualquier agente.

Este archivo es apoyo adicional para agentes o personas que necesiten más contexto sobre la política de commits del repositorio.
