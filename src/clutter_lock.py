"""Cross-launch writer exclusion using Modal Dict's atomic skip-if-exists put.

A lease deliberately survives an interrupted full coordinator. Recovery requires
the operator to stop its app/workers first; elapsed time is never proof of death.
"""

import time
import uuid
from contextlib import contextmanager


@contextmanager
def writer_lease(store, analysis_name, operation, call_id):
    value = {"token": uuid.uuid4().hex, "analysis_name": analysis_name, "operation": operation,
             "function_call_id": call_id, "started_epoch": time.time()}
    if not store.put("writer", value, skip_if_exists=True):
        raise RuntimeError(f"Another Phase-6 writer holds the lease: {store.get('writer')}. Use status; after stopping the old app and all workers, use recover --confirm-stopped.")
    try:
        yield value["token"]
    except BaseException:
        # Prepare/pilot/approval have no spawned GPU children. Full coordinators may.
        if operation != "full" and store.get("writer", {}).get("token") == value["token"]:
            store.pop("writer")
        raise
    else:
        if store.get("writer", {}).get("token") == value["token"]:
            store.pop("writer")


def require_writer(store, token):
    if not token or store.get("writer", {}).get("token") != token:
        raise RuntimeError("Worker belongs to an inactive launch; refusing to write")


def recover_writer(store, analysis_name, confirm_stopped=False):
    if not confirm_stopped:
        raise ValueError("Stop the old Modal app and every worker, then pass --confirm-stopped")
    value = store.get("writer")
    if value is None:
        return {"state": "no outstanding writer lease"}
    if value["analysis_name"] != analysis_name:
        raise ValueError(f"Lease belongs to analysis {value['analysis_name']}; use that analysis config")
    store.pop("writer")
    return {"state": "released after operator confirmed all old workers stopped", "previous_writer": value}
