# Weaviate MCP Server (HTTP) — Render-ready

Server MCP HTTP (Streamable HTTP) per collegare **Weaviate Cloud** a client MCP remoti (es. Claude).

## Deploy rapido su Render

1. **Builda l'app React localmente** (vedi [BUILD.md](BUILD.md)):
   ```bash
   cd weaviate-image-app
   npm install
   npm run build
   git add weaviate-image-app/dist/
   git commit -m "Add built React app"
   git push
   ```

2. Crea un nuovo **Web Service** su Render da questo repo/cartella.
3. Render userà automaticamente Python (non Docker).
4. Imposta le variabili d'ambiente:
   - `WEAVIATE_URL` (oppure `WEAVIATE_CLUSTER_URL`) - **Obbligatorio**
   - `WEAVIATE_API_KEY` - **Obbligatorio**
   - (opz) `MCP_PATH` (default `/mcp/`)
   - (opz) `PORT` (default `10000`)
4. Deploy.
5. Verifica: `GET https://<service>.onrender.com/health` → `{"status":"ok",...}`.

## Collegamento da Claude (Remote MCP)

Aggiungi un **Custom/Remote MCP server** con URL:
```
https://<service>.onrender.com/mcp/
```
oppure (senza slash finale):
```
https://<service>.onrender.com/mcp
```

### Strumenti disponibili

**Configurazione e diagnostica:**
- `get_config()` - Mostra la configurazione Weaviate e lo stato delle API keys
- `check_connection()` - Verifica la connessione a Weaviate
- `get_instructions()` - Restituisce le istruzioni/prompt configurati per il server
- `reload_instructions()` - Ricarica istruzioni da variabili d'ambiente o file
- `diagnose_vertex()` - Report sullo stato dell'autenticazione Vertex AI

**Gestione collection:**
- `list_collections()` - Elenca tutte le collection disponibili
- `get_schema(collection)` - Ottiene lo schema di una collection specifica

**Ricerca:**
- `keyword_search(collection, query, limit=10)` - Ricerca keyword-based (BM25)
- `semantic_search(collection, query, limit=10)` - Ricerca vettoriale semantica (near_text)
- `hybrid_search(collection, query, limit=10, alpha=0.8, query_properties=None, image_id=None, image_url=None)` - Ricerca ibrida (BM25 + vettoriale)
  - **Nota**: Se configurato per l'assistente WindBilance, forza automaticamente l'uso della collection "WindBilance"
  - Supporta ricerca per immagini tramite `image_id` (preferito) o `image_url`
  - La conversione in base64 viene gestita automaticamente dal server
- `image_search_vertex(collection, image_id=None, image_url=None, caption=None, limit=10)` - Ricerca vettoriale per immagini usando Vertex AI
  - **Nota**: Se configurato per l'assistente WindBilance, forza automaticamente l'uso della collection "WindBilance"
  - Supporta `image_id` (preferito) o `image_url`
  - La conversione in base64 viene gestita automaticamente dal server

**Gestione immagini:**
- `upload_image(image_url=None, image_path=None)` - Carica un'immagine e restituisce un `image_id` valido per 1 ora
  - `image_url`: URL pubblico dell'immagine (preferito)
  - `image_path`: Path locale del file sul server
  - **Nota**: Non accetta `image_b64` direttamente - usa l'endpoint HTTP per upload diretto
- `insert_image_vertex(collection, image_id=None, image_url=None, caption=None, id=None)` - Inserisce un'immagine con embedding Vertex AI in una collection
  - `image_id`: ID ottenuto da `upload_image` o `/upload-image` (preferito)
  - `image_url`: URL pubblico dell'immagine (verrà scaricata automaticamente)
  - La conversione in base64 viene gestita automaticamente dal server
  - **Nota**: Non passare direttamente stringhe base64 - usa `image_id` o `image_url`

## Upload Immagini

Il server supporta l'upload di immagini in due modi:

1. **Tool MCP `upload_image`**: 
   - Accetta `image_url` (URL pubblico - preferito) o `image_path` (file locale sul server)
   - Il server scarica/converte automaticamente in base64
   - Restituisce un `image_id` valido per 1 ora
   - **Non accetta `image_b64` direttamente** - usa l'endpoint HTTP per upload diretto

2. **Endpoint HTTP `POST /upload-image`**: Per upload diretto di file binari:
   - Accetta `multipart/form-data` con campo `image` (file binario)
   - Oppure JSON con `image_b64` (stringa base64)
   - Restituisce `{"image_id": "...", "expires_in": 3600}`
   
   Esempio con curl:
   ```bash
   curl -X POST https://<service>.onrender.com/upload-image \
     -F "image=@/path/to/image.jpg"
   ```
   
   Oppure con JSON:
   ```bash
   curl -X POST https://<service>.onrender.com/upload-image \
     -H "Content-Type: application/json" \
     -d '{"image_b64": "base64_string_here"}'
   ```

L'`image_id` restituito può essere usato in `hybrid_search` o `image_search_vertex` per evitare di dover passare l'immagine ogni volta.

**Formati supportati**: JPEG, PNG, GIF, WEBP  
**Dimensione massima**: 10MB  
**Validità**: 1 ora (pulizia automatica delle immagini scadute)

## Note

- Per Weaviate Cloud bastano **URL + API key**.
- Il server ascolta su `0.0.0.0:$PORT` (compatibile Render, default porta 10000).
- Health-check disponibile su `/health`.
- Supporto per embedding OpenAI: imposta `OPENAI_API_KEY` o `OPENAI_APIKEY` per usare `text2vec-openai` in Weaviate.
- Puoi personalizzare nome/descrizione/prompt del server con:
  - `MCP_SERVER_NAME` (default `weaviate-mcp-http`)
  - `MCP_DESCRIPTION` per una descrizione breve
  - `MCP_PROMPT` / `MCP_INSTRUCTIONS` per un prompt testuale condiviso con il client
- In alternativa puoi versionare i messaggi in file e puntarli con:
  - `MCP_PROMPT_FILE` o `MCP_INSTRUCTIONS_FILE` (es. `prompts/instructions.md`)
  - `MCP_DESCRIPTION_FILE` (es. `prompts/description.txt`)
- Se non configuri nulla, il server carica automaticamente il prompt predefinito in `prompts/instructions.md` (se presente).
- Lo strumento `get_instructions` restituisce in ogni momento il prompt attivo.
- Usa `reload_instructions` per rileggere i file senza riavviare il server.

## Autenticazione Vertex AI

Il server supporta tre metodi di autenticazione per Vertex AI:

1. **API Key statica**: Imposta `VERTEX_APIKEY` (senza refresh automatico)
2. **Bearer token**: Imposta `VERTEX_BEARER_TOKEN` con un token OAuth già ottenuto esternamente
3. **OAuth con refresh automatico**: Imposta `VERTEX_USE_OAUTH=true` e fornisci un **service account**:
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON` con il JSON in chiaro **oppure**
   - `GOOGLE_APPLICATION_CREDENTIALS` con il path del file **oppure**
   - `VERTEX_SA_PATH` (default `/etc/secrets/weaviate-sa.json`, ideale su Render)
   - Il server rileva automaticamente il `project_id` dal service account
   - Il token viene rigenerato ogni ~55 minuti (o prima della scadenza) e inserito sia negli header REST (`X-Goog-Vertex-Api-Key`, `X-Goog-User-Project`) sia nei metadata gRPC

**Nota**: Per OAuth, il server supporta anche la discovery automatica del progetto GCP tramite Application Default Credentials (ADC).

**Location Vertex AI**: Configurabile con `VERTEX_LOCATION` (default: `us-central1`)

## Configurazione Assistente WindBilance

Se configurato con il prompt predefinito (`prompts/instructions.md`), il server forza automaticamente:
- L'uso della collection "WindBilance" per `hybrid_search` e `image_search_vertex`
- L'uso di `hybrid_search` come metodo di ricerca principale
- Parametri predefiniti ottimizzati per ricerche multimodali (immagini + testo)
