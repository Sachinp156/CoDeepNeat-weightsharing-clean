# example_scripts/run_covid_cxr.py
import os
import sys
import time
import logging
import argparse
from collections import Counter
import csv

# Make TF behave on small GPUs
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

try:
    from keras.preprocessing.image import ImageDataGenerator
except Exception:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from base.shared_layers import REGISTRY
from base.population import Population
from base.structures import Datasets
from base.config import LOG_DETAIL  # custom logging level 21

# ---------------------------------------------------------------------------

def _enable_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

def _pick_val_metric_name():
    return "val_accuracy"

def _build_callbacks(log_csv, ckpt_path):
    monitor = _pick_val_metric_name()
    return [
        EarlyStopping(monitor=monitor, mode="auto", verbose=1, patience=8),
        CSVLogger(log_csv),
        ModelCheckpoint(ckpt_path, monitor=monitor, mode="auto", verbose=1, save_best_only=True),
    ]

def _ensure_dirs():
    for p in (os.path.join(ROOT, "models/"), os.path.join(ROOT, "images/")):
        os.makedirs(p, exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# ---------------------------------------------------------------------------

def _load_split_as_arrays(split_dir, class_names, img_size, max_per_class=None):
    X, y = [], []
    for label, cname in enumerate(class_names):
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            continue
        files = [f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXTS)]
        files.sort()
        if max_per_class is not None:
            files = files[:max_per_class]
        for fn in files:
            p = os.path.join(cdir, fn)
            im = Image.open(p).convert("RGB").resize((img_size, img_size))
            X.append(np.asarray(im, dtype="float32") / 255.0)
            y.append(label)
    if len(X) == 0:
        raise RuntimeError(f"No images loaded from {split_dir}. Check paths and class folders.")
    X = np.stack(X, axis=0)
    y = keras.utils.to_categorical(np.array(y), num_classes=len(class_names))
    return X, y

# ---------------------------------------------------------------------------

def _prepare_data_and_generators(img_size, fast_debug, gen_batch_size):
    DATA_DIR = os.path.join(ROOT, "data", "COVID_Radiography")
    IMG_SIZE = int(img_size)
    BATCH_SIZE = int(gen_batch_size)
    CLASSES = ["COVID", "Normal", "Viral_Pneumonia", "Lung_Opacity"]
    num_classes = len(CLASSES)

    # sanity check
    for sub in ["train", "val", "test"]:
        for cname in CLASSES:
            path = os.path.join(DATA_DIR, sub, cname)
            if not os.path.isdir(path):
                raise RuntimeError(
                    f"Expected folder not found: {path}\n"
                    f"Run your split/prep script to create train/val/test folders first."
                )

    # EA arrays (arrays path; separate from generator used for final retrain)
    max_per_class_train = 1200 if not fast_debug else 300
    max_per_class_test = 240 if not fast_debug else 100
    x_train_full, y_train_full = _load_split_as_arrays(
        os.path.join(DATA_DIR, "train"), CLASSES, IMG_SIZE, max_per_class=max_per_class_train
    )
    x_test, y_test = _load_split_as_arrays(
        os.path.join(DATA_DIR, "test"), CLASSES, IMG_SIZE, max_per_class=max_per_class_test
    )

    # stratified EA split
    from sklearn.model_selection import train_test_split
    y_train_arg = np.argmax(y_train_full, axis=1)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.15, stratify=y_train_arg, random_state=1337
    )

    # Generators for final retrain
    datagen_train = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=5, zoom_range=0.10)
    datagen_eval  = ImageDataGenerator(rescale=1./255)
    train_gen = datagen_train.flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", classes=CLASSES, shuffle=True
    )
    val_gen = datagen_eval.flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", classes=CLASSES, shuffle=False
    )

    # imbalance weights (final retrain only)
    counts = Counter(train_gen.classes); total = sum(counts.values())
    class_weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}

    return (DATA_DIR, IMG_SIZE, BATCH_SIZE, CLASSES, num_classes,
            x_tr, y_tr, x_val, y_val, x_test, y_test, train_gen, val_gen, class_weights)

# ---------------------------------------------------------------------------

def _build_search_spaces(num_classes):
    # slightly larger modules to increase reuse surface
    global_configs = {"module_range": ([2, 4], "int"), "component_range": ([2, 4], "int")}
    input_configs  = {"module_range": ([1, 1], "int"), "component_range": ([1, 1], "int")}
    output_configs = {"module_range": ([1, 1], "int"), "component_range": ([1, 1], "int")}

    possible_components = {
        "conv2d": (
            keras.layers.Conv2D,
            {
                "filters": ([24, 96], "int"),
                "kernel_size": ([1, 3], "list"),
                "strides": ([1], "list"),
                "data_format": (["channels_last"], "list"),
                "padding": (["same"], "list"),
                "activation": (["relu"], "list"),
            },
        ),
        "maxpool": (
            keras.layers.MaxPooling2D,
            {"pool_size": ([2], "list"), "strides": ([2], "list"), "padding": (["valid"], "list")},
        ),
        "bn": (keras.layers.BatchNormalization, {}),
    }

    possible_inputs = {
        "conv2d": (keras.layers.Conv2D, {"filters": ([24, 64], "int"), "kernel_size": ([1], "list"),
                                         "activation": (["relu"], "list")}),
    }

    possible_outputs = {
        "dense": (keras.layers.Dense, {"units": ([32, 128], "int"), "activation": (["relu"], "list")}),
    }

    possible_complementary_components = {"dropout": (keras.layers.Dropout, {"rate": ([0.0, 0.4], "float")})}
    possible_complementary_inputs  = None
    # lock head units to num_classes
    possible_complementary_outputs = {
        "dense": (keras.layers.Dense, {"units": ([num_classes, num_classes], "int"),
                                       "activation": (["softmax"], "list")}),
    }

    return (global_configs, input_configs, output_configs,
            possible_components, possible_complementary_components,
            possible_inputs, possible_outputs, possible_complementary_outputs)

# ---------------------------------------------------------------------------

def _make_population(*, x_tr, y_tr, x_val, y_val, img_size, population_size,
                     compiler, ea_batch_size, share_mode, parallel_sync, ema_beta,
                     fast_debug, max_steps_per_epoch, disable_graph_plots,
                     magic_t, magic_t_sync_every, magic_t_elastic_tau, magic_t_migrate_p,
                     magic_t_noise_std, magic_t_log_keys, conflict_report_every,
                     search_spaces, ab_label=None):
    (global_configs, input_configs, output_configs,
     possible_components, possible_complementary_components,
     possible_inputs, possible_outputs, possible_complementary_outputs) = search_spaces

    pop = Population(
        Datasets(training=[x_tr, y_tr], test=[x_val, y_val]),
        input_shape=(img_size, img_size, 3),
        population_size=population_size,
        compiler=compiler
    )

    pop.ea_batch_size = int(ea_batch_size)
    pop.datasets.SAMPLE_SIZE = min(15000, len(x_tr))
    pop.datasets.TEST_SAMPLE_SIZE = min(1500, len(x_val))
    pop.datasets.custom_fit_args = None

    if hasattr(pop, "BATCH_SIZE"): pop.BATCH_SIZE = int(ea_batch_size)
    if hasattr(pop, "batch_size"): pop.batch_size = int(ea_batch_size)

    pop.module_share_mode = share_mode
    pop.create_module_population(population_size*3, global_configs,
                                 possible_components, possible_complementary_components)
    pop.create_module_species(max(3, population_size // 2))
    pop.create_blueprint_population(
        max(5, population_size - 1),
        global_configs, possible_components, possible_complementary_components,
        input_configs, possible_inputs, None,
        output_configs, possible_outputs, possible_complementary_outputs,
    )
    pop.create_blueprint_species(max(3, population_size // 2))

    pop.parallel_sync = parallel_sync
    pop.ema_beta = ema_beta
    pop.fast_debug = fast_debug
    pop.disable_graph_plots = disable_graph_plots

    if hasattr(pop, "magic_t"): pop.magic_t = bool(magic_t)
    if hasattr(pop, "magic_t_sync_every"): pop.magic_t_sync_every = int(magic_t_sync_every)
    if hasattr(pop, "magic_t_elastic_tau"): pop.magic_t_elastic_tau = float(magic_t_elastic_tau)
    if hasattr(pop, "magic_t_migrate_p"): pop.magic_t_migrate_p = float(magic_t_migrate_p)
    if hasattr(pop, "magic_t_noise_std"): pop.magic_t_noise_std = float(magic_t_noise_std)
    if hasattr(pop, "magic_t_log_keys"): pop.magic_t_log_keys = bool(magic_t_log_keys)
    if hasattr(pop, "conflict_report_every"): pop.conflict_report_every = int(conflict_report_every)

    pop.max_steps_per_epoch = int(max_steps_per_epoch)

    # ✅ Tag the population with the arm label ("A", "B", or "single")
    setattr(pop, "ab_label", ab_label or "single")

    return pop

# ---------------------- Single experiment (original path) ----------------------

def run_covid_cxr_full(
    generations=2,
    training_epochs=10,
    num_populations=1,
    population_size=6,
    blueprint_population_size=5,
    module_population_size=20,
    n_blueprint_species=3,
    n_module_species=3,
    final_model_training_epochs=2,
    share_mode="module",
    parallel_sync=True,
    ema_beta=0.5,
    fast_debug=True,
    max_steps_per_epoch=15,
    disable_graph_plots=True,
    magic_t=False,
    magic_t_sync_every=1,
    magic_t_elastic_tau=0.0,
    magic_t_migrate_p=0.10,
    magic_t_noise_std=0.0,
    magic_t_log_keys=False,
    conflict_report_every=0,
    clear_registry_before_final=True,
    ea_batch_size=16,
    img_size=96,
    gen_batch_size=8,
    warm_start_final=False,
    pcgrad_every=2,              # accepted for API compatibility (used by your Population if present)
    pcgrad_sample_frac=0.5,      # accepted for API compatibility
    _ab_label=None,              # internal: label for A/B prints
):
    _enable_gpu_memory_growth()

    # --- logging ---
    logging.basicConfig(
        filename=os.path.join(ROOT, "execution.log"),
        filemode="w+",
        level=LOG_DETAIL,
        format="%(levelname)s - %(asctime)s: %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_DETAIL)
    console.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s: %(message)s"))
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler):
            root_logger.removeHandler(h)
    root_logger.addHandler(console)

    label = f"[RUN {_ab_label}]" if _ab_label else "[RUN]"
    logging.info("%s share_mode=%s magic_t=%s generations=%d training_epochs=%d ea_batch=%d steps_per_epoch=%d",
                 label, str(share_mode), str(magic_t), int(generations), int(training_epochs),
                 int(ea_batch_size), int(max_steps_per_epoch))

    # --- data / spaces ---
    (DATA_DIR, IMG_SIZE, BATCH_SIZE, CLASSES, num_classes,
     x_tr, y_tr, x_val, y_val, _x_test, _y_test, train_gen, val_gen, class_weights) = \
        _prepare_data_and_generators(img_size, fast_debug, gen_batch_size)

    search_spaces = _build_search_spaces(num_classes)

    compiler = {
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy", keras.metrics.AUC(name="auc", multi_label=False)]
    }

    # just one population (your runner already looped; keep for compatibility)
    pop = _make_population(
        x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val,
        img_size=IMG_SIZE, population_size=population_size,
        compiler=compiler, ea_batch_size=ea_batch_size,
        share_mode=share_mode, parallel_sync=parallel_sync, ema_beta=ema_beta,
        fast_debug=fast_debug, max_steps_per_epoch=max_steps_per_epoch,
        disable_graph_plots=disable_graph_plots,
        magic_t=magic_t, magic_t_sync_every=magic_t_sync_every,
        magic_t_elastic_tau=magic_t_elastic_tau, magic_t_migrate_p=magic_t_migrate_p,
        magic_t_noise_std=magic_t_noise_std, magic_t_log_keys=magic_t_log_keys,
        conflict_report_every=conflict_report_every,
        search_spaces=search_spaces, ab_label=_ab_label
    )

    if hasattr(pop, "log_shared_state"):
        pop.log_shared_state("init")

    ea_t0 = time.time()
    iteration = pop.iterate_generations(
        generations=generations,
        training_epochs=training_epochs,
        validation_split=0.15,
        mutation_rate=0.5,
        crossover_rate=0.2,
        elitism_rate=0.1,
        possible_components=search_spaces[3],
        possible_complementary_components=search_spaces[4],
    )
    ea_t1 = time.time()
    ea_wall = ea_t1 - ea_t0
    logging.info("%s [EA][WALL] total_seconds=%.3f", label, ea_wall)

    if hasattr(pop, "log_shared_state"):
        pop.log_shared_state("post_iter")

    # read best EA accuracy from pop
    best_ind = pop.return_best_individual()
    best_acc = float(best_ind.scores[1]) if (best_ind and best_ind.scores) else float("nan")
    best_loss = float(best_ind.scores[0]) if (best_ind and best_ind.scores) else float("nan")

    # Final retraining (kept; not used in A/B timing table)
    # Build dummy dataset holder for generator-based fit
    improved_ds = Datasets(training=None, test=None)
    improved_ds.training = [np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"),
                            keras.utils.to_categorical([0], num_classes=num_classes)]
    improved_ds.test     = [np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"),
                            keras.utils.to_categorical([0], num_classes=num_classes)]
    improved_ds.custom_fit_args = {
        "generator": train_gen,
        "steps_per_epoch": max(1, train_gen.samples // BATCH_SIZE),
        "epochs": int(0),  # default 0 here; single-run path below sets real epochs from args
        "verbose": 1,
        "validation_data": val_gen,
        "validation_steps": max(1, val_gen.samples // BATCH_SIZE),
        "class_weight": class_weights,
        "callbacks": _build_callbacks(
            os.path.join(ROOT, "training.csv"),
            os.path.join(ROOT, "best_model_checkpoint.h5"),
        ),
    }
    pop.datasets = improved_ds

    # ---------------- Convergence CSV dump (per-generation and per-epoch) ----------------
    converge_csv = os.path.join(ROOT, f"convergence_{_ab_label or 'single'}.csv")
    epoch_csv = os.path.join(ROOT, f"convergence_epochs_{_ab_label or 'single'}.csv")
    try:
        if getattr(pop, "convergence_log", None) and len(pop.convergence_log) > 0:
            with open(converge_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(pop.convergence_log[0].keys()))
                w.writeheader()
                w.writerows(pop.convergence_log)
            logging.info("[EA] Wrote convergence log: %s", converge_csv)
        else:
            converge_csv = None

        if getattr(pop, "convergence_epoch_log", None) and len(pop.convergence_epoch_log) > 0:
            with open(epoch_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(pop.convergence_epoch_log[0].keys()))
                w.writeheader()
                w.writerows(pop.convergence_epoch_log)
            logging.info("[EA] Wrote per-epoch convergence log: %s", epoch_csv)
        else:
            epoch_csv = None
    except Exception as e:
        logging.warning("Could not write convergence CSV: %s", e)
        converge_csv = None
        epoch_csv = None

    return {
        "ea_wall": ea_wall,
        "ea_best_acc": best_acc,
        "ea_best_loss": best_loss,
        "population": pop,
        "img_size": IMG_SIZE,
        "num_classes": num_classes,
        "train_gen": train_gen,
        "val_gen": val_gen,
        "class_weights": class_weights,
        "convergence_csv": converge_csv,
        "convergence_epochs_csv": epoch_csv,
    }

# ---------------------- A/B runner ----------------------

def _fmt_sec(s):
    try:
        return f"{float(s):.2f}"
    except Exception:
        return "nan"

def run_ab_compare(args):
    print("\n===== A/B runner: starting baseline (A) =====")
    REGISTRY.clear()

    a_steps = args.ab_a_steps_per_epoch if args.ab_a_steps_per_epoch is not None else args.max_steps_per_epoch
    a_batch = args.ab_a_ea_batch_size  if args.ab_a_ea_batch_size  is not None else args.ea_batch_size

    resA = run_covid_cxr_full(
        generations=args.generations,
        training_epochs=args.training_epochs,
        num_populations=args.num_populations,
        population_size=args.population_size,
        blueprint_population_size=args.blueprint_population_size,
        module_population_size=args.module_population_size,
        n_blueprint_species=args.n_blueprint_species,
        n_module_species=args.n_module_species,
        final_model_training_epochs=0,     # EA timing only
        share_mode="off",                  # baseline
        parallel_sync=args.parallel_sync,
        ema_beta=args.ema_beta,
        fast_debug=args.fast_debug,
        max_steps_per_epoch=int(a_steps),
        disable_graph_plots=args.disable_graph_plots,
        magic_t=False,                     # baseline
        conflict_report_every=0,
        clear_registry_before_final=True,
        ea_batch_size=int(a_batch),
        img_size=args.img_size,
        gen_batch_size=args.gen_batch_size,
        warm_start_final=False,
        _ab_label="A",
    )

    print("\n===== A/B runner: starting sharing+PCGrad (B) =====")
    if args.clear_registry_before_final:
        REGISTRY.clear()

    b_steps = args.ab_b_steps_per_epoch if args.ab_b_steps_per_epoch is not None else args.max_steps_per_epoch
    b_batch = args.ab_b_ea_batch_size  if args.ab_b_ea_batch_size  is not None else args.ea_batch_size

    resB = run_covid_cxr_full(
        generations=args.generations,
        training_epochs=args.training_epochs,
        num_populations=args.num_populations,
        population_size=args.population_size,
        blueprint_population_size=args.blueprint_population_size,
        module_population_size=args.module_population_size,
        n_blueprint_species=args.n_blueprint_species,
        n_module_species=args.n_module_species,
        final_model_training_epochs=0,
        share_mode=args.share_mode,        # e.g., "module"
        parallel_sync=args.parallel_sync,
        ema_beta=args.ema_beta,
        fast_debug=args.fast_debug,
        max_steps_per_epoch=int(b_steps),
        disable_graph_plots=args.disable_graph_plots,
        magic_t=args.magic_t,              
        magic_t_sync_every=getattr(args, "magic_t_sync_every", 1),
        magic_t_elastic_tau=getattr(args, "magic_t_elastic_tau", 0.0),
        magic_t_migrate_p=getattr(args, "magic_t_migrate_p", 0.10),
        magic_t_noise_std=getattr(args, "magic_t_noise_std", 0.0),
        magic_t_log_keys=getattr(args, "magic_t_log_keys", False),
        conflict_report_every=args.conflict_report_every,
        clear_registry_before_final=args.clear_registry_before_final,
        ea_batch_size=int(b_batch),
        img_size=args.img_size,
        gen_batch_size=args.gen_batch_size,
        warm_start_final=args.warm_start_final,
        pcgrad_every=args.pcgrad_every,
        pcgrad_sample_frac=args.pcgrad_sample_frac,
        _ab_label="B",
    )

    # ---------------- table ----------------
    print("\n================== A/B Comparison ==================\n")
    header = "Run        |  EA wall (s) |   EA Val Acc |  EA Val Loss |  Share |  PCGrad"
    print(header)
    print("-"*len(header))
    rowA = f"A: base    | { _fmt_sec(resA['ea_wall']).rjust(12) } | {resA['ea_best_acc']:.3f}       | {resA['ea_best_loss']:.2f}      |  off   |  off"
    print(rowA)
    shareB = args.share_mode
    pcgB   = "on" if args.magic_t else "off"
    extra  = f"(every={args.pcgrad_every}, frac={args.pcgrad_sample_frac:.2f})" if args.magic_t else ""
    rowB = f"B: share   | { _fmt_sec(resB['ea_wall']).rjust(12) } | {resB['ea_best_acc']:.3f}       | {resB['ea_best_loss']:.2f}      |  {shareB:<6}|  {pcgB} {extra}"
    print(rowB)
    print("\n====================================================\n")

    # Print convergence CSV paths
    print(f"Convergence logs:\n  A: {resA.get('convergence_csv')}\n  B: {resB.get('convergence_csv')}\n")
    print(f"Per-epoch logs:\n  A: {resA.get('convergence_epochs_csv')}\n  B: {resB.get('convergence_epochs_csv')}\n")

# ---------------------- CLI ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="CoDeepNEAT on COVID-19 Radiography (with weight sharing, PCGrad, and A/B runner)")

    # A/B switch
    ap.add_argument("--ab-compare", action="store_true", default=False,
                    help="Run baseline (no sharing, no PCGrad) and sharing+PCGrad back-to-back and print a timing/accuracy table.")

    # A/B per-arm overrides
    ap.add_argument("--ab-a-steps-per-epoch", type=int, default=None,
                    help="Override max steps/epoch for A (baseline).")
    ap.add_argument("--ab-b-steps-per-epoch", type=int, default=None,
                    help="Override max steps/epoch for B (sharing/PCGrad).")
    ap.add_argument("--ab-a-ea-batch-size", type=int, default=None,
                    help="Override EA batch size for A (baseline).")
    ap.add_argument("--ab-b-ea-batch-size", type=int, default=None,
                    help="Override EA batch size for B (sharing/PCGrad).")

    # EA/search scale
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--training-epochs", type=int, default=1)
    ap.add_argument("--num-populations", type=int, default=1)
    ap.add_argument("--population-size", type=int, default=6)
    ap.add_argument("--blueprint-population-size", type=int, default=5)
    ap.add_argument("--module-population-size", type=int, default=20)
    ap.add_argument("--n-blueprint-species", type=int, default=3)
    ap.add_argument("--n-module-species", type=int, default=3)

    # retrain
    ap.add_argument("--final-model-training-epochs", type=int, default=1)

    # sharing/PCGrad knobs
    ap.add_argument("--share-mode", choices=["module", "layer", "global", "off"], default="module")
    ap.add_argument("--magic-t", action="store_true", default=False, help="Enable PCGrad in the EA loop (if supported by your fork).")
    ap.add_argument("--pcgrad-every", type=int, default=2,
                    help="(Optional) If supported, run PCGrad every N steps (>=1). Larger = faster EA, weaker mitigation.")
    ap.add_argument("--pcgrad-sample-frac", type=float, default=0.5,
                    help="(Optional) Fraction (0..1] of shared variables to project per PCGrad pass.")

    # runtime knobs
    ap.add_argument("--parallel-sync", action="store_true", default=True)
    ap.add_argument("--no-parallel-sync", dest="parallel_sync", action="store_false")
    ap.add_argument("--ema-beta", type=float, default=0.5)
    ap.add_argument("--fast-debug", action="store_true", default=True)
    ap.add_argument("--no-fast-debug", dest="fast_debug", action="store_false")
    ap.add_argument("--max-steps-per-epoch", type=int, default=10)
    ap.add_argument("--disable-graph-plots", action="store_true", default=True)
    ap.add_argument("--enable-graph-plots", dest="disable_graph_plots", action="store_false")
    ap.add_argument("--conflict-report-every", type=int, default=0)

    # warm start controls
    ap.add_argument("--clear-registry-before-final", action="store_true", default=True)
    ap.add_argument("--no-clear-registry-before-final", dest="clear_registry_before_final", action="store_false")
    ap.add_argument("--warm-start-final", action="store_true", default=False,
                    help="Warm start final retrain from live-shared weights (use with --no-clear-registry-before-final).")

    # data/batch
    ap.add_argument("--ea-batch-size", type=int, default=16, help="Batch size used in EA/search stage (arrays).")
    ap.add_argument("--img-size", type=int, default=96, help="Image size (IMG_SIZE).")
    ap.add_argument("--gen-batch-size", type=int, default=8, help="Batch size for generator retraining.")
    return ap.parse_args()

# ---------------------- entry ----------------------

if __name__ == "__main__":
    _ensure_dirs()
    args = parse_args()

    if args.ab_compare:
        # Run the A/B harness and exit
        run_ab_compare(args)
        sys.exit(0)

    # Single-run path (original behaviour)
    res = run_covid_cxr_full(
        generations=args.generations,
        training_epochs=args.training_epochs,
        num_populations=args.num_populations,
        population_size=args.population_size,
        blueprint_population_size=args.blueprint_population_size,
        module_population_size=args.module_population_size,
        n_blueprint_species=args.n_blueprint_species,
        n_module_species=args.n_module_species,
        final_model_training_epochs=args.final_model_training_epochs,
        share_mode=args.share_mode,
        parallel_sync=args.parallel_sync,
        ema_beta=args.ema_beta,
        fast_debug=args.fast_debug,
        max_steps_per_epoch=args.max_steps_per_epoch,
        disable_graph_plots=args.disable_graph_plots,
        magic_t=args.magic_t,
        magic_t_sync_every=getattr(args, "magic_t_sync_every", 1),
        magic_t_elastic_tau=getattr(args, "magic_t_elastic_tau", 0.0),
        magic_t_migrate_p=getattr(args, "magic_t_migrate_p", 0.10),
        magic_t_noise_std=getattr(args, "magic_t_noise_std", 0.0),
        magic_t_log_keys=getattr(args, "magic_t_log_keys", False),
        conflict_report_every=args.conflict_report_every,
        clear_registry_before_final=args.clear_registry_before_final,
        ea_batch_size=args.ea_batch_size,
        img_size=args.img_size,
        gen_batch_size=args.gen_batch_size,
        warm_start_final=args.warm_start_final,
        pcgrad_every=args.pcgrad_every,
        pcgrad_sample_frac=args.pcgrad_sample_frac,
        _ab_label="single",
    )

    print(f"\nConvergence log saved to: {res.get('convergence_csv')}")
    print(f"Per-epoch log saved to: {res.get('convergence_epochs_csv')}")

    # Optional: final retrain in single-run mode
    if res and args.final_model_training_epochs > 0:
        pop = res["population"]
        train_gen = res["train_gen"]; val_gen = res["val_gen"]
        class_weights = res["class_weights"]; IMG_SIZE = res["img_size"]
        num_classes = res["num_classes"]

        improved_ds = Datasets(training=None, test=None)
        improved_ds.training = [np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"),
                                keras.utils.to_categorical([0], num_classes=num_classes)]
        improved_ds.test     = [np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"),
                                keras.utils.to_categorical([0], num_classes=num_classes)]
        improved_ds.custom_fit_args = {
            "generator": train_gen,
            "steps_per_epoch": max(1, train_gen.samples // args.gen_batch_size),
            "epochs": args.final_model_training_epochs,
            "verbose": 1,
            "validation_data": val_gen,
            "validation_steps": max(1, val_gen.samples // args.gen_batch_size),
            "class_weight": class_weights,
            "callbacks": _build_callbacks(
                os.path.join(ROOT, "training.csv"),
                os.path.join(ROOT, "best_model_checkpoint.h5"),
            ),
        }
        pop.datasets = improved_ds

        if args.clear_registry_before_final:
            REGISTRY.clear()
            print("[REGISTRY] Cleared before final retraining.")
        else:
            print("[REGISTRY] Kept for warm start (no clear).")

        best_model = pop.return_best_individual()
        print(f"Best fitting model chosen for retraining: {best_model.name}")
        pop.train_full_model(
            best_model,
            args.final_model_training_epochs,
            validation_split=0.15,
            custom_fit_args=improved_ds.custom_fit_args,
            warm_start=bool(args.warm_start_final),
        )
