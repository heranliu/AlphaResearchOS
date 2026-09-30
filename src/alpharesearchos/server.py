"""Loopback research workspace with real factors, backtests, and model settings."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import json
import math
import mimetypes
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .config import ResearchConfig
from .data import FIELDS, inspect_csv, load_csv
from .engine import create_run, run_research
from .independent_backtest import create_backtest, list_backtests, run_backtest
from .library import FactorLibrary
from .model_settings import SettingsStore
from .proposals import llm_configured, provider_context
from .store import atomic_json, clean, read_json

RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
ARTIFACTS = {"report.html", "report.md", "report.json", "trials.csv", "selected_factor.py", "selected_candidate.json", "research_graph.json", "training_audit.json", "holdout_curve.csv", "config.json", "manifest.json"}
BACKTEST_ARTIFACTS = {"report.json", "curve.csv", "factors.json", "config.json"}
DATASET_MAX_BYTES = 20_000_000


def dataset_summary(path, panel):
    close = panel["close"]
    return {"id": path.name, "label": path.stem, "source": "csv", "rows": len(close),
            "assets": len(close.columns), "start": str(close.index[0].date()), "end": str(close.index[-1].date())}


def dataset_error(exc):
    message = str(exc)
    if "400–20000" in message:
        return "数据需要 3–100 个资产、400–20,000 个共同交易日。"
    if "duplicate date/symbol" in message:
        return "symbol 不能为空；同一 date、symbol 只能有一行。"
    if "missing or non-finite" in message or "inconsistent dates/assets" in message:
        return "所有资产必须覆盖相同交易日期；请补齐源数据中的缺失记录，并移除空值、NaN 和无穷值。"
    if "nonpositive prices or negative volume" in message:
        return "open、high、low、close 必须大于 0，volume 必须大于等于 0。"
    if "High must cover" in message:
        return "high 必须大于等于同一行的 open 和 close。"
    if "Low must cover" in message:
        return "low 必须小于等于同一行的 open 和 close。"
    if "date" in message.lower() or "time" in message.lower() or "day" in message.lower():
        return "date 必须是有效日期，格式为 YYYY-MM-DD，不含时分秒。"
    return "CSV 无法读取；请检查英文列名、逗号分隔、UTF-8 编码和数值字段。"


def dataset_upload(body):
    """Validate the shared upload envelope without touching the filesystem."""
    if set(body) != {"name", "content"} or not isinstance(body.get("content"), str):
        raise ValueError("请提供 CSV 文件名 name 和 UTF-8 文本 content。")
    name = body.get("name")
    if (not isinstance(name, str) or len(name.encode("utf-8")) > 180 or not name.lower().endswith(".csv")
            or name.startswith(".") or any(not (char.isalnum() or char in "._- ") for char in name)):
        raise ValueError("文件名需以 .csv 结尾，只能包含文字、数字、空格、点、下划线或短横线，不能包含目录。")
    name = name[:-4] + ".csv"
    content = body["content"]
    if not content.strip():
        raise ValueError("CSV 文件为空，请选择包含表头和行情数据的文件。")
    if len(content.encode("utf-8")) > DATASET_MAX_BYTES:
        raise OverflowError("CSV 文件不能超过 20 MB。")
    return name, content


def inspect_dataset(body):
    name, content = dataset_upload(body)
    return {"name": name, **inspect_csv(content)}


def import_dataset(dataset_root, body):
    name, content = dataset_upload(body)
    destination = dataset_root / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("已有同名数据集，请重命名 CSV 后重新导入。")
    # Bound cardinality before pandas pivots long rows into a rectangular panel.
    symbols, dates = set(), set()
    try:
        reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")), strict=True)
        header = reader.fieldnames or []
        required = ["date", "symbol", *FIELDS]
        missing = [field for field in required if field not in header]
        if missing:
            raise ValueError("缺少必需列：" + ", ".join(missing) + "。")
        if any(header.count(field) != 1 for field in required):
            raise ValueError("date、symbol 和 OHLCV 列名不能重复。")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in required):
                raise ValueError(f"第 {row_number} 行的列数与表头不一致。")
            if not row["symbol"].strip():
                raise ValueError(f"第 {row_number} 行 symbol 不能为空。")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]):
                raise ValueError(f"第 {row_number} 行 date 需为 YYYY-MM-DD 日期。")
            try:
                date.fromisoformat(row["date"])
            except ValueError:
                raise ValueError(f"第 {row_number} 行 date 不是有效日期，请使用 YYYY-MM-DD。") from None
            for field in FIELDS:
                try:
                    value = float(row[field])
                except ValueError:
                    raise ValueError(f"第 {row_number} 行 {field} 必须是有效数字。") from None
                if not math.isfinite(value):
                    raise ValueError(f"第 {row_number} 行 {field} 必须是有限数字，不能为空、NaN 或无穷值。")
            symbols.add(row["symbol"])
            dates.add(row["date"])
            if len(symbols) > 100 or len(dates) > 20000:
                raise ValueError("最多支持 100 个资产和 20,000 个交易日。")
    except csv.Error:
        raise ValueError("CSV 格式有误，请检查引号配对、逗号分隔和单元格长度。") from None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dataset_root,
                                         prefix=".upload-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            panel = load_csv(temporary)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(dataset_error(exc)) from None
        # Hard-link publication is atomic and refuses an existing file even across processes.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError("已有同名数据集，请重命名 CSV 后重新导入。") from None
        return dataset_summary(destination, panel)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def available_run_lock(directory: Path, *, create=False):
    """Probe the shared flock; optionally hold it while recording a worker failure."""
    try:
        lock = (directory / ".lock").open("a" if create else "r")
    except FileNotFoundError:
        if create:
            raise
        yield True
        return
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def externally_active(directory: Path):
    with available_run_lock(directory) as available:
        return not available


def make_server(runs_root: Path, dataset_root: Path, port=8765):
    runs_root, dataset_root = runs_root.resolve(), dataset_root.resolve()
    backtests_root = runs_root.parent / "backtests"
    for directory in (runs_root, dataset_root, backtests_root):
        directory.mkdir(parents=True, exist_ok=True)
    static = Path(__file__).parent / "static"
    settings = SettingsStore(runs_root.parent / ".alphaos")
    library = FactorLibrary(runs_root, runs_root.parent / ".alphaos")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alphaos-research")
    active = {}
    mutex = threading.RLock()

    def job_info(job_id, directory):
        with mutex:
            task = active.get(job_id)
            external = task is None and externally_active(directory)
            return {"active": task is not None or external, "externally_active": external,
                    "pause_requested": bool(task and task["pause"].is_set())}

    def launch(directory, kind="research", provider=None):
        # Callers hold this lock through create+submit, so concurrent requests cannot create orphan runs.
        with mutex:
            if active:
                raise ValueError("已有任务正在运行")
            pause = threading.Event()
            active[directory.name] = {"kind": kind, "pause": pause}

        def work():
            try:
                if kind == "backtest":
                    run_backtest(directory)
                else:
                    with provider_context(provider):
                        run_research(directory, should_pause=pause.is_set)
            except Exception as exc:
                # A CLI can win the lock after a resume request's preflight check.
                # Its checkpoint remains authoritative, even if it already released the lock.
                if isinstance(exc, RuntimeError) and str(exc) == "This run is already active in another process":
                    return
                # A preflight failure (fingerprint/snapshot/etc.) must not leave a queued job forever.
                # Keep ownership through read+write so a newly resumed CLI cannot be overwritten.
                with available_run_lock(directory, create=True) as available:
                    if not available:
                        return
                    record = read_json(directory / "report.json")
                    if record.get("status") not in {"completed", "failed"}:
                        record["status"] = "failed"
                        message = str(exc)[:500] if isinstance(exc, ValueError) else f"{type(exc).__name__}: 任务执行失败"
                        record["error"] = message
                        atomic_json(directory / "report.json", record)
                print(f"Run {directory.name} failed: {type(exc).__name__}", flush=True)
            finally:
                with mutex:
                    active.pop(directory.name, None)
        try:
            executor.submit(work)
        except Exception:
            with mutex:
                active.pop(directory.name, None)
            raise

    def run_directory(run_id, root=runs_root):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run ID")
        path = root / run_id
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError("Run not found")
        return path

    def registered_csv(name):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("请先导入并选择一个 CSV 数据集。")
        file = dataset_root / name
        if not file.is_file() or file.is_symlink():
            raise ValueError("数据集不存在，请重新导入 CSV。")
        return file

    def public_report(directory, files):
        record = read_json(directory / "report.json")
        record.pop("_state", None)
        decorate_job(record, directory)
        record["artifacts"] = sorted(name for name in files if (directory / name).is_file() and not (directory / name).is_symlink())
        return record

    def decorate_job(record, directory):
        record.update(job_info(record["id"], directory))
        if record.get("status") in {"running", "queued"} and not record["active"]:
            record["status"] = "interrupted"
            record["error"] = "服务已中断，可恢复研究或重新运行回测"
        return record

    class Handler(BaseHTTPRequestHandler):
        server_version = "AlphaResearchOS/0.2"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args) > 1 else "") != "200":
                super().log_message(fmt, *args)

        def send_bytes(self, payload, content_type="application/json; charset=utf-8", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, payload, status=200):
            self.send_bytes(json.dumps(clean(payload), ensure_ascii=False, allow_nan=False).encode(), status=status)

        def trusted(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host", "") in allowed and (
                not self.headers.get("Origin") or self.headers["Origin"] in {f"http://{host}" for host in allowed}
            )

        def do_GET(self):
            if not self.trusted():
                return self.json({"error": "Loopback origin required"}, 403)
            path = urlparse(self.path).path
            try:
                if path == "/api/health":
                    with mutex:
                        jobs = [{"id": key, "kind": task["kind"], "pause_requested": task["pause"].is_set()} for key, task in active.items()]
                    return self.json({"ok": True, "version": __version__, "llm_configured": llm_configured(settings.resolve()), "jobs": jobs})
                if path == "/api/settings":
                    return self.json(settings.public_config())
                if path == "/api/factors":
                    return self.json({"factors": library.list_factors()})
                if path == "/api/backtests":
                    return self.json({"backtests": [decorate_job(record, backtests_root / record["id"]) for record in list_backtests(backtests_root)]})
                if path.startswith("/api/backtests/"):
                    if len(path.split("/")) != 4:
                        raise FileNotFoundError("Not found")
                    return self.json(public_report(run_directory(path.split("/")[3], backtests_root), BACKTEST_ARTIFACTS))
                if path == "/api/runs":
                    runs = []
                    for file in sorted(runs_root.glob("*/report.json"), reverse=True):
                        if not RUN_ID.fullmatch(file.parent.name) or file.parent.is_symlink() or file.is_symlink():
                            continue
                        try:
                            record = read_json(file)
                            runs.append(decorate_job({"id": record["id"], "status": record["status"], "created_at": record["created_at"],
                                         "direction": record["config"]["direction"], "source": record["source"],
                                         "best_score": (record.get("selected") or {}).get("score"),
                                         **record["progress"]}, file.parent))
                        except (ValueError, KeyError):
                            continue
                    return self.json({"runs": runs})
                if path.startswith("/api/runs/"):
                    if len(path.split("/")) != 4:
                        raise FileNotFoundError("Not found")
                    return self.json(public_report(run_directory(path.split("/")[3]), ARTIFACTS))
                if path == "/api/datasets":
                    datasets = []
                    for file in sorted(dataset_root.glob("*.csv")):
                        if file.is_symlink():
                            continue
                        try:
                            datasets.append(dataset_summary(file, load_csv(file)))
                        except (ValueError, KeyError):
                            continue
                    return self.json({"datasets": datasets})
                if path.startswith("/artifacts/"):
                    parts = path.split("/")
                    if len(parts) == 5 and parts[2] == "backtests" and parts[4] in BACKTEST_ARTIFACTS:
                        directory, name = run_directory(parts[3], backtests_root), parts[4]
                    elif len(parts) == 4 and parts[3] in ARTIFACTS:
                        directory, name = run_directory(parts[2]), parts[3]
                    else:
                        raise FileNotFoundError("Unknown artifact")
                    target = directory / name
                    if not target.is_file() or target.is_symlink():
                        raise FileNotFoundError("Artifact is not available yet")
                    return self.send_bytes(target.read_bytes(), (mimetypes.guess_type(target)[0] or "text/plain") + "; charset=utf-8")
                name = "index.html" if path == "/" else path.removeprefix("/static/").lstrip("/")
                if name not in {"index.html", "app.js", "theme.js", "style.css"}:
                    raise FileNotFoundError("Not found")
                return self.send_bytes((static / name).read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain") + "; charset=utf-8")
            except FileNotFoundError as exc:
                self.json({"error": str(exc)}, 404)
            except (ValueError, KeyError) as exc:
                self.json({"error": str(exc)}, 400)
            except Exception:
                self.json({"error": "读取失败，请检查本地服务日志"}, 500)

        def do_POST(self):
            if not self.trusted():
                return self.json({"error": "Loopback origin required"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.json({"error": "application/json required"}, 415)
            try:
                path = urlparse(self.path).path
                size = int(self.headers.get("Content-Length", "0"))
                dataset_upload_path = path in {"/api/datasets", "/api/datasets/import", "/api/datasets/inspect"}
                limit = DATASET_MAX_BYTES * 2 + 4096 if dataset_upload_path else 256000 if path == "/api/factors/import" else 16000
                if not 0 <= size <= limit:
                    return self.json({"error": "CSV 文件不能超过 20 MB。" if dataset_upload_path else "Request too large"}, 413)
                body = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if path == "/api/datasets/inspect":
                    return self.json({"inspection": inspect_dataset(body)})
                if path in {"/api/datasets", "/api/datasets/import"}:
                    return self.json({"dataset": import_dataset(dataset_root, body)}, 201)
                if path == "/api/factors/import":
                    return self.json(library.import_factors(body), 201)
                if path == "/api/settings":
                    with mutex:
                        if active:
                            return self.json({"error": "任务运行期间不能更改模型配置"}, 409)
                        return self.json(settings.update(body))
                if path in {"/api/settings/test", "/api/settings/jev/test"}:
                    if body:
                        raise ValueError("Save configuration before testing")
                    with mutex:
                        if active:
                            return self.json({"error": "请等待当前任务结束后测试连接"}, 409)
                        active["connection-test"] = {"kind": "connection", "pause": threading.Event()}
                    try:
                        probe = settings.test_jev_connection if path == "/api/settings/jev/test" else settings.test_connection
                        return self.json(probe())
                    finally:
                        with mutex:
                            active.pop("connection-test", None)
                if path == "/api/backtests":
                    with mutex:
                        if active:
                            return self.json({"error": "已有任务正在运行，请等待完成"}, 409)
                        registered_csv(body.get("dataset"))
                        directory = create_backtest(backtests_root, dataset_root, body, library.list_factors())
                        launch(directory, "backtest")
                    return self.json({"id": directory.name, "status": "running"}, 201)
                if path == "/api/runs":
                    with mutex:
                        if active:
                            return self.json({"error": "已有任务正在运行，请等待完成"}, 409)
                        allowed = {"direction", "mode", "trials", "seed", "cost_bps", "top_k", "rebalance_every", "dataset", "max_seconds", "max_llm_calls", "max_llm_tokens", "factor_ids", "constraints"}
                        if set(body) - allowed:
                            raise ValueError("Unknown configuration fields")
                        factor_ids = body.pop("factor_ids", [])
                        body.setdefault("mode", "agent")
                        if body["mode"] == "agent":
                            for key, default in {"trials": 6, "max_seconds": 900, "max_llm_calls": 12, "max_llm_tokens": 786432, "max_complexity": 150, "warmup": 252}.items():
                                body.setdefault(key, default)
                        config = ResearchConfig(**body)
                        if not isinstance(factor_ids, list) or len(factor_ids) > min(50, config.trials) or any(not isinstance(value, str) for value in factor_ids) or len(factor_ids) != len(set(factor_ids)):
                            raise ValueError("Choose unique factors within the trial budget (at most 50)")
                        factors = {factor["id"]: factor for factor in library.list_factors()} if factor_ids else {}
                        candidates = []
                        for factor_id in factor_ids:
                            if factor_id not in factors:
                                raise ValueError("Factor not found")
                            factor = factors[factor_id]
                            candidates.append({key: factor[key] for key in ("name", "expression", "hypothesis")})
                            candidates[-1].update(origin="library_import", parents=[factor_id])
                        provider = settings.resolve()
                        with provider_context(provider):
                            directory = create_run(runs_root, config, registered_csv(config.dataset), candidates)
                        launch(directory, provider=provider)
                    return self.json({"id": directory.name, "status": "running"}, 201)
                if path.startswith("/api/runs/"):
                    parts = path.split("/")
                    if len(parts) != 5 or parts[4] not in {"resume", "pause"}:
                        raise FileNotFoundError("Not found")
                    if body:
                        raise ValueError("No configuration changes allowed on resume/pause")
                    directory = run_directory(parts[3])
                    with mutex:
                        if parts[4] == "pause":
                            task = active.get(directory.name)
                            if not task or task["kind"] != "research":
                                return self.json({"error": "该研究当前未运行"}, 409)
                            task["pause"].set()
                            return self.json({"id": directory.name, "status": "pause_requested", "active": True})
                        record = read_json(directory / "report.json")
                        if record["status"] == "completed":
                            return self.json({"id": directory.name, "status": "completed"})
                        if active:
                            return self.json({"error": "已有任务正在运行，请等待完成"}, 409)
                        if externally_active(directory):
                            return self.json({"error": "该研究正在其他进程中运行，请等待完成"}, 409)
                        provider = settings.resolve()
                        if record["config"]["mode"] in {"llm", "agent"} and not llm_configured(provider):
                            raise ValueError("请先配置模型连接")
                        launch(directory, provider=provider)
                    return self.json({"id": directory.name, "status": "running"})
                self.json({"error": "Not found"}, 404)
            except FileNotFoundError as exc:
                self.json({"error": str(exc)}, 404)
            except FileExistsError as exc:
                self.json({"error": str(exc)}, 409)
            except OverflowError as exc:
                self.json({"error": str(exc)}, 413)
            except UnicodeError:
                self.json({"error": "请将 CSV 保存为 UTF-8 编码后重新导入。"}, 400)
            except (ValueError, TypeError, KeyError) as exc:
                self.json({"error": str(exc)}, 400)
            except Exception:
                self.json({"error": "操作失败，请检查本地服务日志"}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.research_executor = executor
    return server


def serve(runs_root, dataset_root, port=8765):
    server = make_server(runs_root, dataset_root, port)
    print(f"AlphaResearchOS: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.research_executor.shutdown(wait=True)
