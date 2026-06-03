import os, json, time, asyncio, sqlite3, httpx, re
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ── Config ────────────────────────────────────────────────────────────────────
app          = FastAPI()
OLLAMA_URL   = "http://127.0.0.1:11434"
DIR_SESIONES = "sesiones_guardadas"
DIR_PROMPTS  = "prompts"
DB_FILE      = "experimentos.db"

os.makedirs(DIR_SESIONES, exist_ok=True)
os.makedirs(DIR_PROMPTS,  exist_ok=True)

if not os.listdir(DIR_PROMPTS):
    with open(os.path.join(DIR_PROMPTS, "experto_ciber.txt"), "w", encoding="utf-8") as f:
        f.write("Eres un experto senior en ciberseguridad. Respondes con precision tecnica, sin saludos y directo al punto.")

# Kill-switch por sesion activa
_abort_events: Dict[str, asyncio.Event] = {}

# ── DB ────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sesiones (
        id_sesion        TEXT PRIMARY KEY,
        titulo_ia        TEXT,
        subtitulo_humano TEXT,
        reacciones       TEXT,
        fecha            TEXT,
        duracion_total_s REAL    DEFAULT 0,
        es_resesion      INTEGER DEFAULT 0,
        sesion_origen    TEXT    DEFAULT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS interacciones (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        id_sesion           TEXT,
        turno               INTEGER,
        modelo              TEXT,
        prompt_usuario      TEXT,
        respuesta_modelo    TEXT,
        tiempo_escritura_s  REAL    DEFAULT 0,
        tiempo_inferencia_s REAL    DEFAULT 0,
        tps                 REAL    DEFAULT 0,
        prompt_tokens       INTEGER DEFAULT 0,
        output_tokens       INTEGER DEFAULT 0,
        editado             INTEGER DEFAULT 0,
        FOREIGN KEY(id_sesion) REFERENCES sesiones(id_sesion)
    )""")
    # Migraciones no destructivas
    for tabla, col, defn in [
        ("sesiones",      "duracion_total_s",   "REAL DEFAULT 0"),
        ("sesiones",      "es_resesion",        "INTEGER DEFAULT 0"),
        ("sesiones",      "sesion_origen",      "TEXT DEFAULT NULL"),
        ("interacciones", "tiempo_escritura_s", "REAL DEFAULT 0"),
        ("interacciones", "prompt_tokens",      "INTEGER DEFAULT 0"),
        ("interacciones", "output_tokens",      "INTEGER DEFAULT 0"),
        ("interacciones", "editado",            "INTEGER DEFAULT 0"),
    ]:
        try:
            c.execute(f"ALTER TABLE {tabla} ADD COLUMN {col} {defn}")
        except Exception:
            pass
    conn.commit()
    conn.close()

init_db()

# ── Modelos Pydantic ──────────────────────────────────────────────────────────
class IniciarSesionPayload(BaseModel):
    id_sesion:     str
    modelo:        str
    system_prompt: str
    parametros:    Dict[str, Any]
    es_resesion:   bool = False
    sesion_origen: Optional[str] = None

class InferenciaPayload(BaseModel):
    id_sesion:          str
    modelo:             str
    system_prompt:      str
    origen_prompt:      str
    mensajes_historial: List[Dict[str, str]]
    prompt_actual:      str
    temperatura:        float
    top_p:              float
    top_k:              int
    max_tokens:         int
    timeout_s:          int   = 120
    tiempo_escritura_s: float = 0.0

class GuardarTurnoPayload(BaseModel):
    id_sesion:           str
    turno:               int
    modelo:              str
    prompt_usuario:      str
    respuesta_modelo:    str
    tiempo_escritura_s:  float
    tiempo_inferencia_s: float
    tps:                 float
    prompt_tokens:       int
    output_tokens:       int

class EditarTurnoPayload(BaseModel):
    id_sesion: str
    turno:     int
    campo:     str      # "prompt_usuario" | "respuesta_modelo"
    contenido: str

class EliminarTurnosPayload(BaseModel):
    id_sesion: str
    turnos:    List[int]

class AbortPayload(BaseModel):
    id_sesion: str

class FinalizarPayload(BaseModel):
    id_sesion:             str
    subtitulo_humano:      str
    reacciones_viscerales: str
    historial_completo:    List[Dict[str, Any]]
    parametros_finales:    Dict[str, Any]
    metadatos_modelo:      Dict[str, Any]
    duracion_total_s:      float = 0.0
    es_resesion:           bool  = False
    sesion_origen:         Optional[str] = None

# ── Utilidades ────────────────────────────────────────────────────────────────
def safe_prompt_path(nombre: str) -> str:
    limpio = os.path.basename(nombre)
    ruta   = os.path.realpath(os.path.join(DIR_PROMPTS, limpio))
    base   = os.path.realpath(DIR_PROMPTS)
    if not ruta.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="Nombre invalido")
    return ruta

def extraer_meta(model_id: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"cuantizacion":"—","tamano":"—","familia":"—","contexto_max":4096,"size_gb":0}
    q = re.search(r'(q\d[_a-z0-9]*)', model_id, re.IGNORECASE)
    if q: meta["cuantizacion"] = q.group(1).upper()
    b = re.search(r'(\d+(?:\.\d+)?b)', model_id, re.IGNORECASE)
    if b: meta["tamano"] = b.group(1).lower()
    meta["familia"] = model_id.split(":")[0].split("/")[-1]
    return meta

def json_sesion_ruta(id_sesion: str) -> str:
    return os.path.join(DIR_SESIONES, f"{id_sesion}.json")

def leer_json_sesion(id_sesion: str) -> Optional[Dict]:
    ruta = json_sesion_ruta(id_sesion)
    if not os.path.isfile(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

def escribir_json_sesion(id_sesion: str, data: Dict):
    with open(json_sesion_ruta(id_sesion), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ── Rutas ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/modelos")
async def get_modelos():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            data = resp.json()
            modelos = []
            for m in data.get("models", []):
                mid  = m.get("name", "")
                det  = m.get("details", {})
                base = extraer_meta(mid)
                modelos.append({"id": mid, "metadatos": {
                    "cuantizacion": det.get("quantization_level", base["cuantizacion"]),
                    "tamano":       det.get("parameter_size",     base["tamano"]),
                    "familia":      det.get("family",             base["familia"]),
                    "contexto_max": m.get("context_length", 4096),
                    "size_gb":      round(m.get("size", 0) / 1e9, 1),
                }})
            return {"data": modelos}
        except Exception as e:
            return {"data": [], "error": str(e)}

@app.get("/api/prompts")
async def listar_prompts():
    return {"prompts": [f for f in sorted(os.listdir(DIR_PROMPTS)) if f.endswith(".txt")]}

@app.get("/api/prompts/{nombre}")
async def cargar_prompt(nombre: str):
    ruta = safe_prompt_path(nombre)
    if not os.path.isfile(ruta):
        raise HTTPException(status_code=404, detail="No encontrado")
    with open(ruta, "r", encoding="utf-8") as f:
        return {"contenido": f.read()}

# ── Sesiones ──────────────────────────────────────────────────────────────────
@app.post("/api/sesiones/iniciar")
async def iniciar_sesion(payload: IniciarSesionPayload):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""INSERT OR IGNORE INTO sesiones
                 (id_sesion,titulo_ia,subtitulo_humano,reacciones,fecha,
                  duracion_total_s,es_resesion,sesion_origen)
                 VALUES (?,?,?,?,?,?,?,?)""",
              (payload.id_sesion, f"Sesion {payload.modelo}", "", "",
               datetime.now().isoformat(), 0,
               int(payload.es_resesion), payload.sesion_origen))
    conn.commit(); conn.close()

    data = {
        "id_sesion":           payload.id_sesion,
        "fecha":               datetime.now().isoformat(),
        "titulo_ia":           f"Sesion {payload.modelo}",
        "es_resesion":         payload.es_resesion,
        "sesion_origen":       payload.sesion_origen,
        "modelo_activo":       {"id": payload.modelo},
        "parametros_iniciales": payload.parametros,
        "system_prompt":       payload.system_prompt,
        "interacciones":       []
    }
    escribir_json_sesion(payload.id_sesion, data)
    return {"status": "ok"}

@app.post("/api/sesiones/turno")
async def guardar_turno(payload: GuardarTurnoPayload):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""INSERT INTO interacciones
                 (id_sesion,turno,modelo,prompt_usuario,respuesta_modelo,
                  tiempo_escritura_s,tiempo_inferencia_s,tps,prompt_tokens,output_tokens)
                 VALUES (?,?,?,?,?,?,?,?,?,?)""",
              (payload.id_sesion, payload.turno, payload.modelo,
               payload.prompt_usuario, payload.respuesta_modelo,
               payload.tiempo_escritura_s, payload.tiempo_inferencia_s,
               payload.tps, payload.prompt_tokens, payload.output_tokens))
    c.execute("""UPDATE sesiones SET duracion_total_s=(
                     SELECT COALESCE(SUM(tiempo_escritura_s+tiempo_inferencia_s),0)
                     FROM interacciones WHERE id_sesion=?)
                 WHERE id_sesion=?""", (payload.id_sesion, payload.id_sesion))
    conn.commit(); conn.close()

    data = leer_json_sesion(payload.id_sesion)
    if data:
        data["interacciones"].append({
            "turno":          payload.turno,
            "role_user":      payload.prompt_usuario,
            "role_assistant": payload.respuesta_modelo,
            "metricas": {
                "tiempo_escritura_s":  payload.tiempo_escritura_s,
                "tiempo_inferencia_s": payload.tiempo_inferencia_s,
                "tps":                 payload.tps,
                "prompt_tokens":       payload.prompt_tokens,
                "output_tokens":       payload.output_tokens,
            }
        })
        escribir_json_sesion(payload.id_sesion, data)
    return {"status": "ok"}

@app.put("/api/sesiones/turno/editar")
async def editar_turno(payload: EditarTurnoPayload):
    if payload.campo not in ("prompt_usuario", "respuesta_modelo"):
        raise HTTPException(status_code=400, detail="Campo invalido")
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute(f"UPDATE interacciones SET {payload.campo}=?,editado=1 WHERE id_sesion=? AND turno=?",
              (payload.contenido, payload.id_sesion, payload.turno))
    conn.commit(); conn.close()

    data = leer_json_sesion(payload.id_sesion)
    if data:
        campo_j = "role_user" if payload.campo == "prompt_usuario" else "role_assistant"
        for inter in data["interacciones"]:
            if inter.get("turno") == payload.turno:
                inter[campo_j]    = payload.contenido
                inter["editado"]  = True
        escribir_json_sesion(payload.id_sesion, data)
    return {"status": "ok"}

@app.delete("/api/sesiones/{id_sesion}/turnos")
async def eliminar_turnos(id_sesion: str, payload: EliminarTurnosPayload):
    if not re.match(r'^[a-zA-Z0-9_\-]+$', id_sesion):
        raise HTTPException(status_code=400, detail="ID invalido")
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    for t in payload.turnos:
        c.execute("DELETE FROM interacciones WHERE id_sesion=? AND turno=?", (id_sesion, t))
    conn.commit(); conn.close()

    data = leer_json_sesion(id_sesion)
    if data:
        data["interacciones"] = [i for i in data["interacciones"]
                                 if i.get("turno") not in payload.turnos]
        escribir_json_sesion(id_sesion, data)
    return {"status": "ok", "eliminados": len(payload.turnos)}

# ── Abort ─────────────────────────────────────────────────────────────────────
@app.post("/api/inferencia/abortar")
async def abortar_inferencia(payload: AbortPayload):
    ev = _abort_events.get(payload.id_sesion)
    if ev:
        ev.set()
    return {"status": "ok"}

# ── Inferencia (streaming) ────────────────────────────────────────────────────
@app.post("/api/inferencia")
async def inferencia(payload: InferenciaPayload, request: Request):
    mensajes = [{"role": "system", "content": payload.system_prompt}]
    mensajes.extend(payload.mensajes_historial)
    mensajes.append({"role": "user", "content": payload.prompt_actual})

    ollama_payload = {
        "model":   payload.modelo,
        "messages": mensajes,
        "stream":  True,
        "options": {
            "temperature": payload.temperatura,
            "top_p":       payload.top_p,
            "top_k":       payload.top_k,
            "num_predict": payload.max_tokens,
        }
    }

    abort_event = asyncio.Event()
    _abort_events[payload.id_sesion] = abort_event
    timeout = httpx.Timeout(connect=5.0, read=float(payload.timeout_s),
                            write=10.0, pool=5.0)

    async def stream_generator():
        start         = time.time()
        prompt_tokens = 0
        output_tokens = 0
        got_prompt    = False

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat",
                                         json=ollama_payload) as resp:
                    async for line in resp.aiter_lines():

                        # Kill desde botón stop
                        if abort_event.is_set():
                            await resp.aclose()
                            yield "[ABORTED]||\n"
                            return

                        # Kill si el browser se desconectó
                        if await request.is_disconnected():
                            await resp.aclose()
                            return

                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue

                        # Metadata del prompt (llega en el chunk final de Ollama)
                        if not got_prompt and "prompt_eval_count" in data:
                            prompt_tokens = data["prompt_eval_count"]
                            pt_dur        = data.get("prompt_eval_duration", 0)
                            pt_tps        = round(prompt_tokens/(pt_dur/1e9), 1) if pt_dur > 0 else 0
                            got_prompt    = True
                            yield f"[PROMPT_DONE]||{prompt_tokens}||{pt_tps}\n"

                        if "eval_count" in data:
                            output_tokens = data["eval_count"]

                        token = data.get("message", {}).get("content", "")
                        done  = data.get("done", False)

                        if token:
                            elapsed  = time.time() - start
                            tps_live = round(output_tokens / elapsed, 1) if elapsed > 0 else 0
                            # Escapar saltos de línea para no romper el protocolo línea-a-línea
                            tok_enc = token.replace("\\", "\\\\").replace("\n", "\\n")
                            yield f"[TK]||{tok_enc}||{output_tokens}||{tps_live}\n"

                        if done:
                            total = round(time.time() - start, 2)
                            tps   = round(output_tokens / total, 2) if total > 0 else 0
                            yield f"[DONE]||{total}||{tps}||{prompt_tokens}||{output_tokens}\n"
                            return

        except httpx.TimeoutException:
            yield "[ERROR]||timeout\n"
        except Exception as e:
            yield f"[ERROR]||{str(e)[:120]}\n"
        finally:
            _abort_events.pop(payload.id_sesion, None)

    return StreamingResponse(stream_generator(), media_type="text/plain")

# ── Listar / Cargar ───────────────────────────────────────────────────────────
@app.get("/api/sesiones")
async def listar_sesiones():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""SELECT id_sesion,titulo_ia,subtitulo_humano,fecha,
                        duracion_total_s,es_resesion,sesion_origen
                 FROM sesiones ORDER BY fecha DESC LIMIT 100""")
    rows = c.fetchall(); conn.close()
    return {"sesiones": [
        {"id_sesion": r[0], "titulo": r[1] or r[0], "subtitulo": r[2] or "",
         "fecha": r[3], "duracion_total_s": r[4] or 0,
         "es_resesion": bool(r[5]), "sesion_origen": r[6]}
        for r in rows
    ]}

@app.get("/api/sesiones/{id_sesion}")
async def cargar_sesion(id_sesion: str):
    if not re.match(r'^[a-zA-Z0-9_\-]+$', id_sesion):
        raise HTTPException(status_code=400, detail="ID invalido")
    data = leer_json_sesion(id_sesion)
    if data is None:
        raise HTTPException(status_code=404, detail="No encontrada")
    return data

# ── Finalizar ─────────────────────────────────────────────────────────────────
@app.post("/api/sesiones/finalizar")
async def finalizar_sesion(payload: FinalizarPayload):
    titulo_ia = payload.parametros_finales.get("modelo", "Sesion")
    try:
        msgs = [{"role":"system","content":"Titulador tecnico. Responde SOLO el titulo, max 5 palabras, sin puntuacion final."}]
        for m in payload.historial_completo:
            if m.get("role") in ("user","assistant"):
                msgs.append({"role": m["role"], "content": str(m.get("content",""))[:400]})
        msgs.append({"role":"user","content":"Titulo tecnico de esta conversacion."})
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model":   payload.parametros_finales.get("modelo",""),
                "messages": msgs, "stream": False,
                "options": {"temperature": 0.3, "num_predict": 20}
            })
            titulo_ia = r.json().get("message",{}).get("content", titulo_ia).strip().strip('"\'')
    except Exception:
        pass

    fecha = datetime.now().isoformat()
    sesion_data = {
        "id_sesion":             payload.id_sesion,
        "fecha":                 fecha,
        "titulo_ia":             titulo_ia,
        "subtitulo_humano":      payload.subtitulo_humano,
        "reacciones_viscerales": payload.reacciones_viscerales,
        "duracion_total_s":      payload.duracion_total_s,
        "es_resesion":           payload.es_resesion,
        "sesion_origen":         payload.sesion_origen,
        "modelo_activo":         {"id": payload.parametros_finales.get("modelo"),
                                  "metadata": payload.metadatos_modelo},
        "parametros_finales":    payload.parametros_finales,
        "interacciones":         payload.historial_completo,
    }
    escribir_json_sesion(payload.id_sesion, sesion_data)

    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("INSERT OR REPLACE INTO sesiones VALUES (?,?,?,?,?,?,?,?)",
              (payload.id_sesion, titulo_ia, payload.subtitulo_humano,
               payload.reacciones_viscerales, fecha, payload.duracion_total_s,
               int(payload.es_resesion), payload.sesion_origen))
    conn.commit(); conn.close()
    return {"status": "ok", "titulo_ia": titulo_ia}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
