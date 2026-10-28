import os

import pytest

from plantops import config
from plantops.tools import runtime


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    """One offline environment for the whole run: hash embeddings, temp var dir, ingested data."""
    var = tmp_path_factory.mktemp("var")
    os.environ.update({"PLANTOPS_EMBEDDER": "hash", "PLANTOPS_VAR_DIR": str(var), "PLANTOPS_TOOL_TRANSPORT": "inprocess"})
    config.get_settings.cache_clear()
    runtime.reset()
    s = config.get_settings()

    from plantops.ingest.embeddings import make_embedder
    from plantops.ingest.pipeline import run_ingest

    run_ingest(s, make_embedder(s))
    return s


@pytest.fixture(scope="session")
def service(settings):
    from google.adk.sessions import InMemorySessionService

    from plantops.service import PlantOpsService
    from plantops.testing import ScriptedLlm

    llm = ScriptedLlm({})
    svc = PlantOpsService(settings, worker=llm, strong=llm, session_service=InMemorySessionService(), synchronous_tracing=True)
    svc.llm = llm
    return svc
