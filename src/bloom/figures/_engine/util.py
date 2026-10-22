from __future__ import annotations
import json, os, hashlib, tempfile, re, time
from pathlib import Path

TRANS_MODELS = [
    "Autoformer",
    "Crossformer",
    "DLinear",
    "FEDformer",
    "FiLM",
    "FreTS",
    "Informer",
    "iTransformer",
    "LightTS",
    "Mamba",
    "Nonstationary_Transformer",
    "PatchTST",
    "PAttn",
    "Reformer",
    "SegRNN",
    "TiDE",
    "TimeFilter",
    "TimesNet",
    "TimeXer",
    "Transformer",
    "TSMixer",
    "WPMixer",
]
FM_MODELS = ["v6_200_new", "v6_author", "v7_200", "v7_author"]
ALIASES = {
    "v6_200_new": "unidirectional BLOOM",
    "v6_author": "unidirectional BLOOM",
    "v7_200": "bidirectional BLOOM",
    "v7_author": "bidirectional BLOOM",
    "v6_HVG": "unidirectional BLOOM",
    "v7_HVG": "bidirectional BLOOM",
    "Nonstationary_Transformer": "Nonstationary Transformer",
}


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_")[:150] or "unnamed"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, default=str, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if st.st_size > 1024 * 1024:
            f.seek(max(0, st.st_size - 1024 * 1024))
            h.update(f.read(1024 * 1024))
    return {
        "path": str(p.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "edge_sha256": h.hexdigest(),
    }


def config_read(path):
    path = Path(path).resolve()
    c = read_json(path)
    base = path.parent
    for key in ["v5_roots", "legacy_roots"]:
        c[key] = [
            str((base / Path(p)).resolve()) if not Path(p).is_absolute() else str(Path(p))
            for p in c.get(key, [])
        ]
    for key in ["output_root", "cache_root"]:
        p = Path(c.get(key, "results" if key == "output_root" else "cache"))
        c[key] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    for collection in ("sources", "saved_diagnostic_sources"):
        for source in c.get(collection, []):
            if source.get("path"):
                p = Path(source["path"]).expanduser()
                source["path"] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    for spec in c.get("datasets", {}).values():
        for key in [
            "obs_path",
            "programs_path",
            "edges_path",
            "author_genes_path",
            "gene_annotations_path",
            "gene_evidence_path",
            "author_graph_path",
            "graph_path",
            "graph_nodes_path",
            "graph_cell_map_path",
        ]:
            if spec.get(key):
                p = Path(spec[key])
                spec[key] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    c["config_path"] = str(path)
    c.setdefault("aliases", ALIASES.copy())
    return c


def resources(requested="auto", reserve=0, gb_per_worker=1.0):
    import psutil

    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    for f in ["/sys/fs/cgroup/cpu.max"]:
        try:
            q, period = Path(f).read_text().split()
            if q != "max":
                cpu = min(cpu, max(1, int(int(q) / int(period))))
        except (OSError, ValueError):
            pass
    mem = psutil.virtual_memory().available
    try:
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if lim != "max":
            stats = dict(
                line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            )
            reclaim = int(stats.get("inactive_file", 0))
            mem = min(mem, max(256 * 2**20, int(lim) - used + reclaim))
    except (OSError, ValueError):
        pass
    n = max(1, cpu - int(reserve)) if str(requested) == "auto" else max(1, min(cpu, int(requested)))
    mem_workers = max(1, int(mem * 0.7 / max(gb_per_worker * 2**30, 1)))
    return min(n, mem_workers), dict(
        available_cpus=cpu,
        reserve=reserve,
        available_memory_gb=round(mem / 2**30, 2),
        estimated_gb_per_worker=gb_per_worker,
        cpu_workers=n,
        memory_workers=mem_workers,
    )


def code_digest():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


from contextlib import contextmanager


@contextmanager
def process_pool(max_workers):
    """Spawn workers safely after numerical imports; Ctrl+C stops active jobs."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))
    try:
        yield pool
    except BaseException:
        processes = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=0.5)
        raise
    else:
        pool.shutdown(wait=True)
