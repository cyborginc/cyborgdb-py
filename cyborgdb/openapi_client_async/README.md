# cyborgdb/openapi_client_async

Async httpx transport for the CyborgDB Python SDK, generated from `openapi.json`
and pruned to retain only transport-specific files (models are shared with
`cyborgdb.openapi_client`).

## Manual generation command

```bash
openapi-generator-cli generate \
  -i openapi.json \
  -g python \
  -o . \
  --package-name cyborgdb.openapi_client_async \
  --additional-properties=generateSourceCodeOnly=true,library=httpx
```

Generator version: **7.22.0** (pinned in `openapitools.json`).

After generation, retain only:

- `api/default_api.py`
- `rest.py`
- `api_client.py`
- `configuration.py`
- `__init__.py`

Delete `models/`, `exceptions.py`, and `api_response.py` from the generated
output.  Rewrite all imports in the retained files to point at
`cyborgdb.openapi_client` equivalents (models, exceptions, api_response).

> **Note:** Extension of `update-openapi-client.sh` to automate this pruning
> step is deferred to a human maintainer (expertise-guarded path).  Until that
> extension lands, run the command above and follow the pruning steps manually
> whenever `openapi.json` changes.
