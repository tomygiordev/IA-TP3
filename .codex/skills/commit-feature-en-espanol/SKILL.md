---
name: commit-feature-en-espanol
description: Hacer commits detallados en español después de cada feature implementada. Usar cuando el usuario pida desarrollar, modificar, refactorizar, documentar, corregir o agregar funcionalidad en un repositorio Git y quiera que cada feature quede registrada con un commit individual, claro y exhaustivo.
---

# Commit por feature en español

## Regla central

Después de implementar cada feature o unidad funcional terminada, crear un commit Git independiente con mensaje y descripción en español.

El commit debe explicar con detalle qué se hizo, por qué se hizo y qué se verificó. Cuanto más detallado y útil sea para revisar el historial, mejor.

## Qué cuenta como feature

Considerar feature cualquier unidad de trabajo que pueda revisarse por separado:

- Nueva funcionalidad.
- Corrección de bug.
- Refactor significativo.
- Documentación nueva o reestructurada.
- Cambio visual o de interfaz.
- Scripts, tests, configuración o resultados generados que acompañan una mejora.

Si una tarea grande incluye varias features separables, hacer un commit por cada una.

## Flujo obligatorio

1. Implementar la feature.
2. Revisar cambios con `git status --short` y, si hace falta, `git diff`.
3. Verificar lo implementado con tests, ejecución manual, smoke test o revisión razonable.
4. Asegurar que no se incluyan secretos, archivos temporales, caches, entornos virtuales ni cambios ajenos no relacionados.
5. Preparar un commit solo con los archivos de esa feature.
6. Crear un commit en español con asunto claro y cuerpo detallado.
7. Informar al usuario el hash corto y un resumen del commit.

## Formato del commit

Usar este estilo:

```text
tipo: resumen breve de la feature

Detalle:
- Qué se agregó, cambió o eliminó.
- Archivos principales tocados.
- Decisiones técnicas importantes.
- Motivo de la implementación.

Verificación:
- Comandos ejecutados o validaciones realizadas.
- Si algo no se pudo verificar, explicar por qué.

Notas:
- Riesgos, limitaciones o próximos pasos relevantes.
```

Tipos sugeridos:

- `feat`: nueva funcionalidad.
- `fix`: corrección.
- `docs`: documentación.
- `refactor`: reorganización sin cambio funcional.
- `test`: pruebas.
- `chore`: configuración, limpieza o mantenimiento.

Ejemplo:

```text
docs: documentar arquitectura y flujo del TP

Detalle:
- Se agregó documentación de arquitectura, experimentos y resultados.
- Se movieron materiales teóricos a docs/materials para limpiar la raíz.
- Se agregó una guía de defensa ante observaciones críticas.

Verificación:
- Se revisó la estructura con git status --short.
- Se validó que los enlaces relativos del README apunten a archivos existentes.

Notas:
- El informe final debe redactarlo el grupo para respetar la consigna.
```

## Reglas de seguridad y prolijidad

- No hacer commit si el usuario pidió explícitamente no commitear.
- No commitear cambios de otra persona sin entenderlos y sin que correspondan a la feature actual.
- No usar `git add .` de forma automática si hay cambios no relacionados; preferir agregar rutas específicas.
- No incluir `.venv`, `__pycache__`, credenciales, tokens, caches ni archivos generados irrelevantes.
- Si el repositorio no tiene Git inicializado, informar que no se puede commitear hasta ejecutar `git init` o hasta que el usuario indique cómo proceder.
- Si no hay identidad Git configurada y el commit falla, informar el bloqueo y los comandos necesarios para configurar `user.name` y `user.email`.
- Si la feature queda incompleta o sin verificar, no hacer commit salvo que el usuario pida explícitamente registrar trabajo parcial.

## Español obligatorio

Todo debe estar en español:

- Asunto del commit.
- Cuerpo del commit.
- Resumen final al usuario.
- Notas de verificación.

Evitar mensajes vagos como `actualizar archivos`, `cambios varios` o `fix cosas`.
