# `src/smoke_test.py`

## Responsabilidad

Ejecuta una prueba rapida del pipeline completo:

- Carga un subconjunto chico.
- Adapta vectorizador.
- Entrena una epoca.
- Calcula metricas.

## Cuando usarlo

Antes de correr todos los experimentos o despues de instalar dependencias en una maquina nueva.

```powershell
python src\smoke_test.py
```

Si este archivo corre, el entorno esta listo para ejecutar `src\train.py`.
