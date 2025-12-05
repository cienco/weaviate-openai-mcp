# serve.py
import os
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import mcp.types as types

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

# --- Weaviate client imports (v4) ---
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# OpenAI client per descrizioni immagini
from openai import OpenAI

_OPENAI_CLIENT = None
if os.environ.get("OPENAI_API_KEY"):
    _OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    print("[query-caption] WARNING: OPENAI_API_KEY non impostata, niente descrizioni testuali per le query.")

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_PROMPT_PATH = _BASE_DIR / "prompts" / "instructions.md"
_DEFAULT_DESCRIPTION_PATH = _BASE_DIR / "prompts" / "description.txt"
_BASE_URL = os.environ.get("BASE_URL", "https://weaviate-mcp-render-z4im.onrender.com")


def _get_weaviate_url() -> str:
    url = os.environ.get("WEAVIATE_CLUSTER_URL") or os.environ.get("WEAVIATE_URL")
    if not url:
        raise RuntimeError("Please set WEAVIATE_URL or WEAVIATE_CLUSTER_URL.")
    return url


def _get_weaviate_api_key() -> str:
    api_key = os.environ.get("WEAVIATE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set WEAVIATE_API_KEY.")
    return api_key


def _connect():
    """
    Connessione a Weaviate Cloud usando:
    - API key del cluster (WEAVIATE_API_KEY)
    - API key OpenAI per text2vec-openai (OPENAI_API_KEY)
    """
    url = _get_weaviate_url()
    key = _get_weaviate_api_key()

    # Costruisci headers (REST)
    headers: Dict[str, str] = {}
    
    # OpenAI per text2vec-openai
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if openai_key:
        headers["X-OpenAI-Api-Key"] = openai_key
        print(f"[openai] using OpenAI API key for text2vec-openai (prefix: {openai_key[:10]}...)")
    else:
        print("[openai] WARNING: OPENAI_API_KEY not set, text2vec-openai may not work")

    # Crea client Weaviate con header
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(key),
        headers=headers or None,
    )

    return client


def _load_text_source(env_keys, file_path):
    if isinstance(env_keys, str):
        env_keys = [env_keys]
    path = Path(file_path) if file_path else None
    if path and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as exc:
            print(f"[mcp] warning: cannot read instructions file '{path}': {exc}")
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val.strip()
    return None


_MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "weaviate-mcp-http")
_MCP_INSTRUCTIONS_FILE = os.environ.get("MCP_PROMPT_FILE") or os.environ.get(
    "MCP_INSTRUCTIONS_FILE"
)
if not _MCP_INSTRUCTIONS_FILE and _DEFAULT_PROMPT_PATH.exists():
    _MCP_INSTRUCTIONS_FILE = str(_DEFAULT_PROMPT_PATH)
_MCP_DESCRIPTION_FILE = os.environ.get("MCP_DESCRIPTION_FILE")
if not _MCP_DESCRIPTION_FILE and _DEFAULT_DESCRIPTION_PATH.exists():
    _MCP_DESCRIPTION_FILE = str(_DEFAULT_DESCRIPTION_PATH)

_MCP_INSTRUCTIONS = _load_text_source(
    ["MCP_PROMPT", "MCP_INSTRUCTIONS"], _MCP_INSTRUCTIONS_FILE
)
_MCP_DESCRIPTION = _load_text_source("MCP_DESCRIPTION", _MCP_DESCRIPTION_FILE)

# Porta e host per FastMCP / uvicorn (per Render)
SERVER_PORT = int(os.environ.get("PORT", "10000"))
os.environ.setdefault("FASTMCP_PORT", str(SERVER_PORT))
os.environ.setdefault("FASTMCP_HOST", "0.0.0.0")

# Host esterno esposto da Render (se disponibile)
render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")

allowed_hosts = [
    "localhost",
    "127.0.0.1:*",  # utile per sviluppo locale
]

# Aggiungi host di Render, se definito
if render_host:
    allowed_hosts.append(render_host)
    allowed_hosts.append(f"{render_host}:*")
else:
    # fallback hard-coded per il tuo servizio attuale
    allowed_hosts.append("weaviate-text2vec-mcp.onrender.com")
    allowed_hosts.append("weaviate-text2vec-mcp.onrender.com:*")

transport_security = TransportSecuritySettings(
    # Manteniamo la protezione DNS rebinding ma permettiamo il tuo dominio
    enable_dns_rebinding_protection=True,
    allowed_hosts=allowed_hosts,
    # Lasciamo vuoto allowed_origins per evitare rogne con l'header Origin
    allowed_origins=[],
)

# Non passiamo host/port direttamente, lasciamo che FastMCP usi le env FASTMCP_*
mcp = FastMCP(
    _MCP_SERVER_NAME,
    stateless_http=True,
    transport_security=transport_security,
)


def _apply_mcp_metadata():
    try:
        if hasattr(mcp, "set_server_info"):
            server_info: Dict[str, Any] = {}
            if _MCP_DESCRIPTION:
                server_info["description"] = _MCP_DESCRIPTION
            if _MCP_INSTRUCTIONS:
                server_info["instructions"] = _MCP_INSTRUCTIONS

            # 🔍 LOG DI DEBUG
            print("[mcp] server_info.description presente:", bool(_MCP_DESCRIPTION))
            print("[mcp] server_info.instructions presente:", bool(_MCP_INSTRUCTIONS))
            if _MCP_INSTRUCTIONS_FILE:
                print("[mcp] instructions file:", _MCP_INSTRUCTIONS_FILE)
            if _MCP_DESCRIPTION_FILE:
                print("[mcp] description file:", _MCP_DESCRIPTION_FILE)

            if server_info:
                mcp.set_server_info(**server_info)
            else:
                print("[mcp] WARNING: server_info is empty!")
    except Exception as e:
        print(f"[mcp] ERROR in _apply_mcp_metadata: {e}")


_apply_mcp_metadata()


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({"status": "ok", "service": "weaviate-mcp-http"})


@mcp.tool()
def ping() -> str:
    """
    Tool neutro usato solo per evitare che il server venga interpretato
    come "solo motore di ricerca semantica".
    """
    return "pong"


@mcp.tool()
def get_instructions() -> Dict[str, Any]:
    return {
        "instructions": _MCP_INSTRUCTIONS,
        "description": _MCP_DESCRIPTION,
        "server_name": _MCP_SERVER_NAME,
        "prompt_file": _MCP_INSTRUCTIONS_FILE,
        "description_file": _MCP_DESCRIPTION_FILE,
    }


@mcp.tool()
def reload_instructions() -> Dict[str, Any]:
    global _MCP_INSTRUCTIONS, _MCP_DESCRIPTION, _MCP_INSTRUCTIONS_FILE, _MCP_DESCRIPTION_FILE
    _MCP_INSTRUCTIONS_FILE = os.environ.get("MCP_PROMPT_FILE") or os.environ.get(
        "MCP_INSTRUCTIONS_FILE"
    )
    if not _MCP_INSTRUCTIONS_FILE and _DEFAULT_PROMPT_PATH.exists():
        _MCP_INSTRUCTIONS_FILE = str(_DEFAULT_PROMPT_PATH)
    _MCP_DESCRIPTION_FILE = os.environ.get("MCP_DESCRIPTION_FILE")
    if not _MCP_DESCRIPTION_FILE and _DEFAULT_DESCRIPTION_PATH.exists():
        _MCP_DESCRIPTION_FILE = str(_DEFAULT_DESCRIPTION_PATH)
    _MCP_INSTRUCTIONS = _load_text_source(
        ["MCP_PROMPT", "MCP_INSTRUCTIONS"], _MCP_INSTRUCTIONS_FILE
    )
    _MCP_DESCRIPTION = _load_text_source("MCP_DESCRIPTION", _MCP_DESCRIPTION_FILE)
    _apply_mcp_metadata()
    return get_instructions()


@mcp.tool()
def get_config() -> Dict[str, Any]:
    return {
        "weaviate_url": os.environ.get("WEAVIATE_CLUSTER_URL")
        or os.environ.get("WEAVIATE_URL"),
        "weaviate_api_key_set": bool(os.environ.get("WEAVIATE_API_KEY")),
        "openai_api_key_set": bool(
            os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        ),
        "cohere_api_key_set": bool(os.environ.get("COHERE_API_KEY")),
    }


@mcp.tool()
def check_connection() -> Dict[str, Any]:
    client = _connect()
    try:
        ready = client.is_ready()
        return {"ready": bool(ready)}
    finally:
        client.close()


@mcp.tool()
def list_collections() -> List[str]:
    client = _connect()
    try:
        colls = client.collections.list_all()
        if isinstance(colls, dict):
            names = list(colls.keys())
        else:
            try:
                names = [getattr(c, "name", str(c)) for c in colls]
            except Exception:
                names = list(colls)
        return sorted(set(names))
    finally:
        client.close()


@mcp.tool()
def get_schema(collection: str) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        try:
            cfg = coll.config.get()
        except Exception:
            try:
                cfg = coll.config.get_class()
            except Exception:
                cfg = {"info": "config API not available in this client version"}
        return {"collection": collection, "config": cfg}
    finally:
        client.close()


@mcp.tool()
def keyword_search(collection: str, query: str, limit: int = 10) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.bm25(
            query=query,
            return_metadata=MetadataQuery(score=True),
            limit=limit,
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "bm25_score": getattr(getattr(o, "metadata", None), "score", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def semantic_search(collection: str, query: str, limit: int = 10) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "distance": getattr(getattr(o, "metadata", None), "distance", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def hybrid_search(
    collection: str,
    query: str,
    limit: int = 10,
    alpha: float = 0.5,
    query_properties: Optional[Any] = None,
) -> Dict[str, Any]:
    if collection and collection != "Chunk":
        print(
            f"[hybrid_search] warning: collection '{collection}' requested, but using 'Chunk' as per instructions"
        )
        collection = "Chunk"

    if query_properties and isinstance(query_properties, str):
        try:
            query_properties = json.loads(query_properties)
        except (json.JSONDecodeError, TypeError):
            pass

    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}

        hybrid_params = {
            "query": query,
            "alpha": alpha,
            "limit": limit,
            "return_properties": ["content", "file_name", "absolute_path", "commessa_code", "commessa_name"],
            "return_metadata": MetadataQuery(score=True, distance=True),
        }
        if query_properties:
            hybrid_params["query_properties"] = query_properties
        resp = coll.query.hybrid(**hybrid_params)

        # Log dei risultati nel formato Colab
        print("[DEBUG] Risultati hybrid search:")
        for o in getattr(resp, "objects", []) or []:
            file_name = getattr(o, "properties", {}).get("file_name", "N/A")
            md = getattr(o, "metadata", None)
            score = getattr(md, "score", None)
            if score is not None:
                print(f"{file_name}  score={score:.4f}")
            else:
                print(f"{file_name}  score=N/A")

        out = []
        for o in getattr(resp, "objects", []) or []:
            md = getattr(o, "metadata", None)
            score = getattr(md, "score", None)
            distance = getattr(md, "distance", None)
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "bm25_score": score,
                    "distance": distance,
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


# Registry dei tool normali che vuoi esporre alla App
TOOL_REGISTRY: Dict[str, Any] = {
    "ping": ping,
    "get_instructions": get_instructions,
    "reload_instructions": reload_instructions,
    "get_config": get_config,
    "check_connection": check_connection,
    "list_collections": list_collections,
    "get_schema": get_schema,
    "keyword_search": keyword_search,
    "semantic_search": semantic_search,
    "hybrid_search": hybrid_search,
}

# Tool nascosti (non esposti all'LLM ma ancora disponibili internamente)
_HIDDEN_TOOLS: set[str] = {
    "semantic_search",
    "keyword_search",
    "list_collections",
    "get_schema",
    # "get_instructions",  # RIMOSSO: deve essere visibile per debug
    "reload_instructions",
    "get_config",
    "check_connection",
}


# --- Alias /mcp senza slash finale, se serve --------------------------------
try:
    from starlette.routing import Route

    _starlette_app = getattr(mcp, "app", None) or getattr(mcp, "_app", None)

    if _starlette_app is not None:

        async def _mcp_alias(request):
            scope = dict(request.scope)
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"
            return await _starlette_app(scope, request.receive, request.send)

        _starlette_app.router.routes.insert(
            0,
            Route(
                "/mcp",
                endpoint=_mcp_alias,
                methods=["GET", "HEAD", "POST", "OPTIONS"],
            ),
        )
except Exception as _route_err:
    print("[mcp] warning: cannot register MCP alias route:", _route_err)

@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    """Espone tutti i tool normali a ChatGPT."""
    tools: List[types.Tool] = []

    # Tutti i tool normali (escludendo quelli nascosti)
    for name in TOOL_REGISTRY.keys():
        # Salta i tool nascosti
        if name in _HIDDEN_TOOLS:
            continue
        # Schema di default: argomenti liberi
        input_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        }

        tool_title = name
        tool_description = name
        annotations = {
            "destructiveHint": False,
            "openWorldHint": True,
            "readOnlyHint": False,
        }

        # ✅ Schema specifico per hybrid_search con istruzioni incluse
        if name == "hybrid_search":
            input_schema = {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Lascia vuoto oppure usa il valore predefinito per cercare nella documentazione E&G.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Testo della domanda o delle parole chiave da cercare nella documentazione E&G.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Numero massimo di risultati da usare come base per la risposta.",
                        "default": 10,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Parametro interno di ricerca (lascia il valore predefinito salvo casi particolari).",
                        "default": 0.5,
                    },
                    "query_properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Campi testuali interni su cui cercare (di solito non serve modificarli).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            }
            tool_title = "Ricerca nella documentazione E&G"
            tool_description = (
                "Usa questo tool per cercare nella documentazione tecnica E&G. "
                "È il modo principale per trovare informazioni su commesse, documenti, progetti e contenuti tecnici. "
                "Prima esegui una ricerca con questo tool, poi costruisci la risposta usando i documenti trovati."
            )

        tools.append(
            types.Tool(
                name=name,
                title=tool_title,
                description=tool_description,
                inputSchema=input_schema,
                annotations=annotations,
            )
        )

    return tools


@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    return []


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    return []


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    return types.ServerResult(
        types.ReadResourceResult(
            contents=[],
            _meta={"error": f"Unknown resource: {req.params.uri}"},
        )
    )


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    name = req.params.name
    args = req.params.arguments or {}

    # LOG DI DEBUG: vediamo quali tool vengono chiamati
    print(f"[call_tool] name={name}, args={json.dumps(args, ensure_ascii=False)}")

    # Tool normali (quelli del registry)
    if name in TOOL_REGISTRY:
        fn = TOOL_REGISTRY[name]

        # Caso speciale: hybrid_search → ripuliamo gli argomenti (niente return_properties)
        if name == "hybrid_search":
            print("[call_tool] hybrid_search called with:", args)

            clean_args: Dict[str, Any] = {}

            # collection con default "Chunk"
            clean_args["collection"] = args.get("collection") or "Chunk"

            # query obbligatoria
            q = args.get("query")
            if not q:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text="Errore: parametro obbligatorio 'query' mancante per hybrid_search.",
                            )
                        ],
                        isError=True,
                    )
                )
            clean_args["query"] = q

            # parametri opzionali
            if "limit" in args:
                clean_args["limit"] = args["limit"]
            if "alpha" in args:
                clean_args["alpha"] = args["alpha"]
            if "query_properties" in args:
                clean_args["query_properties"] = args["query_properties"]

            # 🔴 QUI LA COSA IMPORTANTE:
            # sovrascriviamo args con la versione ripulita
            # (così return_properties e qualsiasi altro extra SPARISCONO)
            args = clean_args

        # Tutti gli altri tool normali rimangono come prima
        try:
            # Proviamo a passare gli argomenti così come sono
            result = fn(**args)
            # Se la funzione è async, await
            if hasattr(result, "__await__"):
                result = await result
        except TypeError as e:
            # Se la firma non combacia (ad es. tool senza parametri), riproviamo senza args
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    result = await result
            except Exception as e2:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"Errore chiamando tool {name}: {e2}",
                            )
                        ],
                        isError=True,
                    )
                )
        except Exception as e:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Errore chiamando tool {name}: {e}",
                        )
                    ],
                    isError=True,
                )
            )

        # A questo punto abbiamo "result" (sincrono o async già risolto)

        import json as _json

        # Normalizziamo il risultato in una forma strutturata
        if isinstance(result, (dict, list)):
            structured = result
        else:
            structured = {"result": result}

        # Creiamo una rappresentazione testuale generica (JSON pretty-print)
        try:
            text_repr = _json.dumps(structured, ensure_ascii=False, indent=2)
        except Exception:
            text_repr = str(structured)

        # Per evitare risposte enormi, tronchiamo solo il testo (NON structuredContent)
        max_chars = 6000
        if len(text_repr) > max_chars:
            text_repr = (
                text_repr[:max_chars]
                + "\n\n[Output troncato per lunghezza. I dati completi sono in structuredContent.]"
            )

        text_msg = (
            f"Risultato del tool {name} in formato strutturato (JSON leggibile):\n\n"
            f"{text_repr}"
        )

        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=text_msg,
                    )
                ],
                structuredContent=structured,
            )
        )

    # 3) Tool sconosciuto
    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ],
            isError=True,
        )
    )


# Registra i request handler sul server MCP
mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


# ==== Esponi l'app ASGI per uvicorn (per uso diretto nello start command) ====
# Puoi usare: uvicorn serve:app --host 0.0.0.0 --port $PORT
# Come nell'esempio Pizzaz, usiamo semplicemente mcp.streamable_http_app()
try:
    app = mcp.streamable_http_app()
    if app is None:
        raise ValueError("streamable_http_app() returned None")
    print("[mcp] app obtained via streamable_http_app()")
except Exception as e:
    print(f"[mcp] error getting app via streamable_http_app(): {e}")
    # Fallback: prova a ottenere l'app in altri modi
    from starlette.applications import Starlette
    app = None
    for attr_name in ["app", "_app", "asgi_app", "_asgi_app"]:
        app = getattr(mcp, attr_name, None)
        if app and isinstance(app, Starlette):
            print(f"[mcp] found app via mcp.{attr_name} (fallback)")
            break
    if app is None:
        raise RuntimeError("Cannot get FastMCP app - streamable_http_app() failed and no app found")

# Aggiungi CORS middleware se disponibile (opzionale)
try:
    from starlette.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    pass

# ==== main: avvia il server con uvicorn ==================
if __name__ == "__main__":
    import uvicorn
    
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "10000"))
    
    # Usa direttamente l'oggetto app che hai creato sopra
    uvicorn.run(app, host=host, port=port)

