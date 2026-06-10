# Instrucciones generales para agentes

Estas instrucciones aplican a cualquier agente que trabaje dentro de este repositorio.

## Regla obligatoria de commits

Después de implementar cada feature o unidad funcional terminada, crear un commit Git independiente, en español y con una descripción detallada.

El commit debe explicar:

- Qué se hizo.
- Por qué se hizo.
- Qué archivos principales se tocaron.
- Qué verificación se ejecutó.
- Qué limitaciones o notas quedaron pendientes, si corresponde.

## Qué cuenta como feature

Considerar como feature cualquier unidad de trabajo revisable por separado:

- Nueva funcionalidad.
- Corrección de bug.
- Refactor significativo.
- Cambio visual o de interfaz.
- Documentación nueva o reestructurada.
- Tests, scripts o configuración asociados a una mejora concreta.

Si una tarea grande contiene varias features separables, hacer un commit por cada una.

## Flujo obligatorio

1. Implementar la feature.
2. Revisar cambios con `git status --short` y, si hace falta, `git diff`.
3. Verificar lo implementado con tests, ejecución manual, smoke test o revisión razonable.
4. Evitar incluir secretos, archivos temporales, caches, entornos virtuales o cambios ajenos no relacionados.
5. Preparar el commit solo con los archivos de esa feature.
6. Crear el commit en español con asunto claro y cuerpo detallado.
7. Informar al usuario el hash corto y un resumen del commit.

## Formato esperado del commit

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

- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`

## Restricciones

- No hacer commit si el usuario pidió explícitamente no commitear.
- No commitear cambios ajenos sin entenderlos y sin que correspondan a la feature actual.
- No usar mensajes vagos como `cambios varios`, `update`, `fix cosas`.
- No usar `git add .` si hay riesgo de incluir cambios no relacionados.
- No registrar trabajo parcial sin aclararlo, salvo que el usuario lo pida.
