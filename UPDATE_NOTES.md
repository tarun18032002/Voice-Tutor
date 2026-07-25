# Update notes

Replace the old files with the files in this package. Remove the old copied
production graph from `agent/test.py`; the included `agent/test.py` is only a
manual smoke test.

The following old imports must not remain:

```python
from agent.test import get_tutor_graph
from agent.tutor import tools
```

The production entrypoint now imports the graph only during worker prewarm,
after environment validation. The tutor and graph both obtain tools from
`agent/tools.py`.

The filename `agent/retriver.py` is intentionally preserved because that is
the spelling used by the existing project. Renaming it to `retriever.py` can
be done later as a separate refactor after updating every import.
