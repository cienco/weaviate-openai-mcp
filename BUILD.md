# Istruzioni per il Build

## Build dell'app React

Prima di fare il deploy, devi buildare l'app React localmente:

```bash
cd weaviate-image-app
npm install
npm run build
```

Questo creerà la cartella `weaviate-image-app/dist/` con i file compilati.

## Commit della cartella dist/

Dopo il build, committa la cartella `dist/` nel repository:

```bash
git add weaviate-image-app/dist/
git commit -m "Add built React app"
git push
```

## Deploy su Render.com

Render.com userà automaticamente:
- `buildCommand: pip install -r requirements.txt` (solo dipendenze Python)
- `startCommand: python serve.py` (avvia il server)

**Nota**: La cartella `dist/` deve essere presente nel repository perché Render.com non builda l'app React (non ha Node.js installato).

## Aggiornare l'app React

Se modifichi l'app React:

1. Modifica i file in `weaviate-image-app/src/`
2. Builda: `cd weaviate-image-app && npm run build`
3. Committa: `git add weaviate-image-app/dist/ && git commit -m "Update React app"`
4. Push: `git push`
5. Render.com farà il deploy automaticamente

