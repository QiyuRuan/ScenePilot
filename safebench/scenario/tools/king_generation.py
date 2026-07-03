import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _vendored_king_root(root_dir: Path) -> Path:
    return (root_dir / "safebench" / "king").resolve()


def _log(logger, msg, color="green"):
    if logger is None:
        print(msg)
    else:
        logger.log(msg, color=color)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def _filter_source_entries(entries: List[dict], scenario_id: Optional[int], route_id: Optional[int]) -> List[dict]:
    out = entries
    if scenario_id is not None:
        out = [e for e in out if int(e["scenario_id"]) == int(scenario_id)]
    if route_id is not None:
        out = [e for e in out if int(e["route_id"]) == int(route_id)]
    return out


def _dedup_pairs(entries: List[dict]) -> List[Tuple[int, int, str]]:
    seen = OrderedDict()
    for item in entries:
        sid = int(item["scenario_id"])
        rid = int(item["route_id"])
        key = (sid, rid)
        if key not in seen:
            seen[key] = str(item.get("scenario_folder", "adv_behavior_single"))
    return [(sid, rid, folder) for (sid, rid), folder in seen.items()]


def _route_file_path(root_dir: Path, route_dir: str, scenario_id: int, route_id: int) -> Path:
    rel = Path(route_dir) / f"scenario_{scenario_id:02d}_routes" / f"scenario_{scenario_id:02d}_route_{route_id:02d}.xml"
    return root_dir / rel


def _load_route_xml(route_file: Path) -> ET.Element:
    tree = ET.parse(route_file)
    route = tree.getroot().find("route")
    if route is None:
        raise RuntimeError(f"No <route> found in {route_file}")
    return route


def _root_relative(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return str(path)


def _build_king_routes_xml(
    root_dir: Path,
    route_dir: str,
    selected_pairs: List[Tuple[int, int, str]],
    routes_xml_path: Path,
    mapping_json_path: Path,
    samples_per_route: int = 1,
):
    root = ET.Element("routes")
    mapping = {}
    expanded_index = 0
    num_samples = max(1, int(samples_per_route))

    for scenario_id, route_id, scenario_folder in selected_pairs:
        src_file = _route_file_path(root_dir, route_dir, scenario_id, route_id)
        if not src_file.is_file():
            raise FileNotFoundError(f"Route XML not found: {src_file}")

        route_elem = _load_route_xml(src_file)
        for sample_id in range(num_samples):
            route_copy = ET.fromstring(ET.tostring(route_elem))
            route_copy.set("id", str(expanded_index))
            root.append(route_copy)

            mapping[str(expanded_index)] = {
                "scenario_id": scenario_id,
                "route_id": route_id,
                "sample_id": sample_id,
                "scenario_folder": scenario_folder,
                "source_route_xml": _root_relative(src_file, root_dir),
            }
            expanded_index += 1

    routes_xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(routes_xml_path, encoding="UTF-8", xml_declaration=True)
    _write_json(mapping_json_path, mapping)


def _find_generation_outputs(king_output_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for results_file in king_output_dir.rglob("results.json"):
        scenario_records = results_file.parent / "scenario_records.json"
        if scenario_records.is_file():
            yield results_file, scenario_records


def _build_safebench_scenario_json(
    root_dir: Path,
    source_entries: List[dict],
    mapping: Dict[str, dict],
    king_output_dir: Path,
    output_json: Path,
    logger=None,
):
    records_by_route_index: Dict[str, Path] = {}

    for results_file, scenario_records in _find_generation_outputs(king_output_dir):
        result_payload = _read_json(results_file)
        meta = result_payload.get("meta_data", {})
        route_index = str(meta.get("index", ""))
        if route_index not in mapping:
            continue
        records_by_route_index[route_index] = scenario_records

    pair_to_source: Dict[Tuple[int, int], dict] = {}
    for item in source_entries:
        sid = int(item["scenario_id"])
        rid = int(item["route_id"])
        pair_to_source.setdefault((sid, rid), item)

    converted = []
    data_id = 0
    missing = []
    for route_index in sorted(mapping.keys(), key=int):
        if route_index not in records_by_route_index:
            info = mapping[route_index]
            missing.append((int(info["scenario_id"]), int(info["route_id"]), int(info.get("sample_id", 0))))
            continue

        info = mapping[route_index]
        sid = int(info["scenario_id"])
        rid = int(info["route_id"])
        pair = (sid, rid)
        source_item = pair_to_source.get(pair)
        if source_item is None:
            missing.append((sid, rid, int(info.get("sample_id", 0))))
            continue

        records_path = records_by_route_index[route_index]
        rel_records = os.path.relpath(records_path, root_dir)
        converted.append(
            {
                "data_id": data_id,
                "scenario_folder": source_item.get("scenario_folder", "adv_behavior_single"),
                "scenario_id": sid,
                "route_id": rid,
                "sample_id": int(info.get("sample_id", 0)),
                "risk_level": source_item.get("risk_level", None),
                "parameters": rel_records,
            }
        )
        data_id += 1

    _write_json(output_json, converted)

    unique_missing = sorted(set(missing))
    if unique_missing:
        _log(logger, "[WARN] Missing KING outputs for route samples:", color="yellow")
        for sid, rid, sample_id in unique_missing:
            _log(logger, f"  - scenario_id={sid}, route_id={rid}, sample_id={sample_id}", color="yellow")

    _log(logger, f">> KING scenario index written: {output_json} ({len(converted)} entries)")


def _cast_extra_arg(old_v, new_v: str):
    if isinstance(old_v, bool):
        return new_v.lower() in {"1", "true", "yes", "y"}
    if isinstance(old_v, int):
        return int(new_v)
    if isinstance(old_v, float):
        return float(new_v)
    return new_v


def _normalize_king_extra(raw_extra: Any) -> List[str]:
    if raw_extra is None:
        return []
    if isinstance(raw_extra, list):
        return [str(x) for x in raw_extra]
    if isinstance(raw_extra, str):
        tokens = raw_extra.strip().split()
        return [str(x) for x in tokens]
    return [str(raw_extra)]


def _resolve_path_from_root(root_dir: Path, maybe_path: Any) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(str(maybe_path))
    if p.is_absolute():
        return p
    return (root_dir / p).resolve()


def _pick_existing_path(*candidates: Optional[Path]) -> Optional[Path]:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def _resolve_generation_profile(config: Dict[str, Any], root_dir: Path, logger=None) -> Dict[str, str]:
    """
    Resolve KING generation ego setup from SafeBench agent config.
    Returns dict with keys: ego_agent, ego_agent_ckpt, renderer_class, init_root, routes_file_adv.
    """
    profile = {
        "ego_agent": str(config.get("king_ego_agent", "aim-bev")),
        "ego_agent_ckpt": str(
            config.get("king_ego_agent_ckpt", "driving_agents/king/aim_bev/model_checkpoints/regular")
        ),
        "renderer_class": str(config.get("king_renderer_class", "STN")),
        "init_root": str(
            config.get(
                "king_init_root",
                "driving_agents/king/aim_bev/king_initializations/initializations_subset/",
            )
        ),
        "routes_file_adv": str(config.get("king_routes_file_adv", "data/routes/adv_all.xml")),
    }

    if not bool(config.get("king_follow_safebench_agent", True)):
        return profile

    agent_type = str(config.get("agent_policy_type", "")).strip().lower()

    if agent_type == "aim_bev":
        profile["ego_agent"] = "aim-bev"
        profile["renderer_class"] = str(config.get("king_renderer_class_for_aim_bev", "STN"))
        profile["init_root"] = str(
            config.get(
                "king_init_root_for_aim_bev",
                "driving_agents/king/aim_bev/king_initializations/initializations_subset/",
            )
        )
        sb_ckpt = _resolve_path_from_root(root_dir, config.get("safebench_agent_ckpt_dir_aim_bev"))
        fallback_ckpt = _resolve_path_from_root(root_dir, "safebench/agent/model_ckpt/aim_bev/regular")
        chosen = _pick_existing_path(sb_ckpt, fallback_ckpt)
        if chosen is not None:
            profile["ego_agent_ckpt"] = str(chosen)
        _log(logger, ">> KING generation follows SafeBench agent: aim_bev -> aim-bev")
        return profile

    if agent_type == "transfuser":
        profile["ego_agent"] = "transfuser"
        profile["renderer_class"] = str(config.get("king_renderer_class_for_transfuser", "CARLA"))
        profile["init_root"] = str(
            config.get(
                "king_init_root_for_transfuser",
                "driving_agents/king/transfuser/king_initializations/initializations_subset",
            )
        )
        sb_ckpt = _resolve_path_from_root(root_dir, config.get("safebench_agent_ckpt_dir_transfuser"))
        fallback_ckpt = _resolve_path_from_root(root_dir, "safebench/agent/model_ckpt/transfuser/regular/transfuser")
        chosen = _pick_existing_path(sb_ckpt, fallback_ckpt)
        if chosen is not None:
            profile["ego_agent_ckpt"] = str(chosen)
        _log(logger, ">> KING generation follows SafeBench agent: transfuser -> transfuser")
        return profile

    # SafeBench expert-like agents are mapped to KING AutoPilot in generation.
    if agent_type in {"autopilot", "basic", "behavior"}:
        profile["ego_agent"] = "safebench_expert"
        profile["renderer_class"] = str(config.get("king_renderer_class_for_expert", "STN"))
        profile["init_root"] = str(
            config.get(
                "king_init_root_for_expert",
                "driving_agents/king/aim_bev/king_initializations/initializations_subset/",
            )
        )
        _log(
            logger,
            f">> KING generation follows SafeBench agent: {agent_type} -> KING AutoPilot expert",
            color="yellow",
        )
        return profile

    _log(
        logger,
        f">> KING generation cannot map SafeBench agent '{agent_type}', use configured king_ego_agent.",
        color="yellow",
    )
    return profile


def _run_king_generation_inprocess(args, routes_xml_path: Path):
    king_root = Path(args.king_root).resolve()
    king_script = king_root / "generate_scenarios.py"
    if not king_script.is_file():
        raise FileNotFoundError(f"KING script not found: {king_script}")

    added_paths = []
    for p in [king_root, king_root / "leaderboard", king_root / "scenario_runner"]:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
            added_paths.append(p_str)

    prev_cwd = os.getcwd()
    try:
        os.chdir(str(king_root))
        spec = importlib.util.spec_from_file_location("king_generate_scenarios", str(king_script))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load KING module from {king_script}")
        king_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(king_mod)

        resolved_ego_agent = str(args.ego_agent)
        if resolved_ego_agent == "safebench_expert":
            # Reuse KING's AutoPilot as the generation ego policy.
            king_mod.AimBEVAgent = king_mod.AutoPilot
            resolved_ego_agent = "aim-bev"

        king_args = SimpleNamespace(
            device=str(args.device),
            save_path=str(Path(args.king_output_dir).resolve()),
            seed=int(args.seed),
            batch_size=1,
            max_num_routes=-1,
            opt_iters=int(args.opt_iters),
            learning_rate=float(args.learning_rate),
            num_agents=int(args.num_agents),
            sim_tickrate=4,
            sim_horizon=int(args.sim_horizon),
            renderer_class=str(args.renderer_class),
            port=int(args.port),
            routes_file=str(routes_xml_path),
            routes_file_adv=str(args.routes_file_adv),
            ego_agent=resolved_ego_agent,
            ego_agent_ckpt=str(args.ego_agent_ckpt),
            gradient_clip=0.0,
            detach_ego_path=int(args.detach_ego_path) if args.detach_ego_path is not None else 1,
            w_ego_col=float(args.w_ego_col),
            w_adv_col=float(args.w_adv_col),
            adv_col_thresh=1.25,
            w_adv_rd=float(args.w_adv_rd),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            king_data_fps=2,
            init_root=str(args.init_root),
        )

        extra = _normalize_king_extra(args.king_extra)
        if len(extra) % 2 != 0:
            raise ValueError("king_extra should use '--key value' pairs.")
        for i in range(0, len(extra), 2):
            k = extra[i].lstrip("-").replace("-", "_")
            v = extra[i + 1]
            old_v = getattr(king_args, k, None)
            setattr(king_args, k, _cast_extra_arg(old_v, v) if old_v is not None else v)

        import random
        import numpy as np
        import torch

        np.random.seed(king_args.seed)
        torch.manual_seed(king_args.seed)
        random.seed(king_args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        Path(king_args.save_path).mkdir(parents=True, exist_ok=True)
        if hasattr(king_mod, "save_args"):
            king_mod.save_args(king_args, king_args.save_path)

        king_mod.args = king_args
        king_mod.main(king_args)
    finally:
        os.chdir(prev_cwd)
        for p in added_paths:
            if p in sys.path:
                sys.path.remove(p)


def _run_king_generation_subprocess(args, routes_xml_path: Path):
    king_root = Path(args.king_root).resolve()
    king_script = king_root / "generate_scenarios.py"
    if not king_script.is_file():
        raise FileNotFoundError(f"KING script not found: {king_script}")

    if str(args.ego_agent) == "safebench_expert":
        raise RuntimeError("safebench_expert ego mapping is only supported in king_run_mode=inprocess.")

    cmd = [
        str(args.python_bin),
        str(king_script),
        "--routes_file",
        str(routes_xml_path),
        "--save_path",
        str(Path(args.king_output_dir).resolve()),
        "--batch_size",
        "1",
        "--num_agents",
        str(args.num_agents),
        "--opt_iters",
        str(args.opt_iters),
        "--learning_rate",
        str(args.learning_rate),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--w_ego_col",
        str(args.w_ego_col),
        "--w_adv_col",
        str(args.w_adv_col),
        "--w_adv_rd",
        str(args.w_adv_rd),
        "--seed",
        str(args.seed),
        "--port",
        str(args.port),
        "--sim_horizon",
        str(args.sim_horizon),
        "--renderer_class",
        str(args.renderer_class),
        "--ego_agent",
        str(args.ego_agent),
        "--ego_agent_ckpt",
        str(args.ego_agent_ckpt),
        "--routes_file_adv",
        str(args.routes_file_adv),
        "--init_root",
        str(args.init_root),
        "--device",
        str(args.device),
    ]
    if args.detach_ego_path is not None:
        cmd += ["--detach_ego_path", str(args.detach_ego_path)]

    for extra in _normalize_king_extra(args.king_extra):
        cmd.append(extra)

    env = os.environ.copy()
    py_paths = [
        str(king_root),
        str(king_root / "leaderboard"),
        str(king_root / "scenario_runner"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([env.get("PYTHONPATH", "")] if env.get("PYTHONPATH") else []))

    subprocess.run(cmd, cwd=str(king_root), env=env, check=True)


def _should_generate(config: Dict[str, Any]) -> bool:
    mode = str(config.get("mode", "eval"))
    on_train = config.get("king_auto_generate_on_train", None)
    on_eval = config.get("king_auto_generate_on_eval", None)

    if mode == "train_scenario":
        if on_train is not None:
            return bool(on_train)
        return True

    if mode == "eval":
        if on_eval is not None:
            return bool(on_eval)
        return False

    return False


def _resolve_output_json(root_dir: Path, config: Dict[str, Any]) -> Path:
    scenario_type = str(config.get("scenario_type", "king_generated.json"))
    path = Path(scenario_type)
    if path.is_absolute():
        return path
    return root_dir / config["scenario_type_dir"] / scenario_type


def maybe_prepare_king_scenarios(config: Dict[str, Any], logger=None):
    if config.get("policy_type") != "king":
        return
    if not _should_generate(config):
        return

    root_dir = Path(config["ROOT_DIR"]).resolve()
    output_json = _resolve_output_json(root_dir, config)

    force_regen = bool(config.get("king_force_regenerate", config.get("mode") == "train_scenario"))
    if output_json.is_file() and not force_regen:
        _log(logger, f">> Reusing existing KING scenario index: {output_json}")
        return

    source_json = Path(config.get("king_source_scenario_json", "safebench/scenario/config/scenario_type/scenario_gen.json"))
    if not source_json.is_absolute():
        source_json = root_dir / source_json
    source_entries = _read_json(source_json)
    source_entries = _filter_source_entries(source_entries, config.get("scenario_id"), config.get("route_id"))
    if len(source_entries) == 0:
        raise RuntimeError("No source KING entries left after filtering scenario_id/route_id.")

    selected_pairs = _dedup_pairs(source_entries)
    work_dir = Path(config.get("king_work_dir", "safebench/scenario/scenario_data/king_bridge"))
    if not work_dir.is_absolute():
        work_dir = root_dir / work_dir
    routes_xml = work_dir / str(config.get("king_routes_xml_name", "routes_for_king.xml"))
    route_mapping = work_dir / str(config.get("king_route_mapping_name", "route_mapping.json"))

    _build_king_routes_xml(
        root_dir=root_dir,
        route_dir=str(config.get("route_dir", "safebench/scenario/scenario_data/route")),
        selected_pairs=selected_pairs,
        routes_xml_path=routes_xml,
        mapping_json_path=route_mapping,
        samples_per_route=int(config.get("king_samples_per_route", 1)),
    )
    _log(logger, f">> KING routes prepared: {routes_xml}")

    skip_generation = bool(config.get("king_skip_generation", False))
    if not skip_generation:
        gen_profile = _resolve_generation_profile(config, root_dir, logger=logger)
        run_args = SimpleNamespace(
            king_root=Path(config.get("king_root", str(_vendored_king_root(root_dir)))),
            python_bin=str(config.get("python_bin", "python")),
            king_run_mode=str(config.get("king_run_mode", "inprocess")),
            king_output_dir=Path(config.get("king_output_dir", "safebench/scenario/scenario_data/king_outputs")),
            num_agents=int(config.get("king_num_agents", 4)),
            opt_iters=int(config.get("king_opt_iters", 100)),
            learning_rate=float(config.get("king_learning_rate", 0.005)),
            beta1=float(config.get("king_beta1", 0.8)),
            beta2=float(config.get("king_beta2", 0.99)),
            w_ego_col=float(config.get("king_w_ego_col", 1.0)),
            w_adv_col=float(config.get("king_w_adv_col", 3.0)),
            w_adv_rd=float(config.get("king_w_adv_rd", 20.0)),
            seed=int(config.get("king_seed", config.get("seed", 0))),
            port=int(config.get("port", 2000)),
            sim_horizon=int(config.get("king_sim_horizon", 80)),
            ego_agent=str(gen_profile["ego_agent"]),
            ego_agent_ckpt=str(gen_profile["ego_agent_ckpt"]),
            renderer_class=str(gen_profile["renderer_class"]),
            init_root=str(gen_profile["init_root"]),
            routes_file_adv=str(gen_profile["routes_file_adv"]),
            device=str(config.get("device", "cuda")),
            detach_ego_path=int(config.get("king_detach_ego_path", 1)),
            king_extra=config.get("king_extra", []),
        )
        if not Path(run_args.king_output_dir).is_absolute():
            run_args.king_output_dir = root_dir / run_args.king_output_dir

        if run_args.ego_agent == "safebench_expert" and run_args.king_run_mode == "subprocess":
            _log(
                logger,
                ">> safebench_expert mapping requires inprocess mode; overriding king_run_mode to inprocess.",
                color="yellow",
            )
            run_args.king_run_mode = "inprocess"

        _log(logger, f">> KING generation started ({run_args.king_run_mode}) ...")
        if run_args.king_run_mode == "subprocess":
            _run_king_generation_subprocess(run_args, routes_xml)
        else:
            _run_king_generation_inprocess(run_args, routes_xml)
        _log(logger, ">> KING generation finished.")

    mapping = _read_json(route_mapping)
    king_output_dir = Path(config.get("king_output_dir", "safebench/scenario/scenario_data/king_outputs"))
    if not king_output_dir.is_absolute():
        king_output_dir = root_dir / king_output_dir

    _build_safebench_scenario_json(
        root_dir=root_dir,
        source_entries=source_entries,
        mapping=mapping,
        king_output_dir=king_output_dir,
        output_json=output_json,
        logger=logger,
    )
