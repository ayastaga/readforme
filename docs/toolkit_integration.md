# cohere-toolkit integration

Verified against cohere-toolkit's `main` (last upstream commit March 2025). The toolkit stores
uploaded files as extracted **text** only (`File.file_content`) and rejects image extensions, so
the integration has two halves:

## 1. Upload hook (`src/backend/services/readforme_upload.py`)

`services/file.py::get_file_content` gains a branch: image extensions are POSTed to the ReadForMe
service (`tools.read_this_for_me.url`, default `http://readforme:8090`) and the JSON reading is
stored as Markdown + a fenced JSON block marked `<!-- readforme:v1 -->`. From that point the photo
behaves like any document: `read_file` and `search_file` work, and the chat model sees the summary,
the facts with their `(NOT CONFIRMED ON PAGE)` tags, the warnings, and the transcription.

The target language for the summary at upload time is `tools.read_this_for_me.default_language`.
The chat model re-explains in whatever language the user writes in, using only the listed facts.

## 2. Tool (`src/backend/tools/read_this_for_me.py`)

`read_this_for_me(file=(filename, file_id))` returns one tool-result document per fact, titled e.g.
`Amount due — verified` or `Payment deadline — NOT CONFIRMED ON PAGE`, plus overview, actions,
warnings and transcription. Command cites these individually, so the UI's citation chips show the
verification state. A default preamble is registered instructing the model never to state a number
that isn't in the results.

## What the installer patches

| file | change |
|---|---|
| `src/backend/tools/__init__.py` | import + `__all__` entry |
| `src/backend/config/tools.py` | `Read_This_For_Me = ReadThisForMeTool` in the `Tool` enum |
| `src/backend/config/settings.py` | `ReadThisForMeSettings` (url, default_language, timeout_s); field on `ToolSettings` |
| `src/backend/services/file.py` | image branch in `get_file_content` |
| `src/backend/config/configuration.template.yaml` | `tools.read_this_for_me` block |
| `src/interfaces/assistants_web/src/constants/conversation.ts` | `image/jpeg`, `image/png`, `image/webp` accepted |
| `src/interfaces/assistants_web/src/utils/file.ts` | jpg/jpeg/webp → mime |
| `docker-compose.readforme.yml` | new `readforme` service on `proxynet`; `READFORME_URL` on backend |

Env var equivalents: `READFORME_URL`, `READFORME_DEFAULT_LANGUAGE`, `READFORME_TIMEOUT_S`.

## Manual fallback

If an anchor has moved upstream, apply the table above by hand; each change is 1–12 lines and the
installer prints which anchor failed.

## Known limitations

* Upload-time reading means a long prefill (a full-page photo is ~4k visual tokens); the toolkit's
  upload request waits for it. On a single GPU expect several seconds per page. The service
  timeout defaults to 300 s.
* The toolkit's default agent model (`command-r-plus`) may no longer exist on the Cohere API
  (toolkit issue #972); set a current Command model in the agent settings.
* Only one image per upload is read; multi-page documents should be uploaded as separate photos.
