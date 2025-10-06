# example_scripts/run_covid_cxr.py
import os
import sys
import logging
import argparse
from collections import Counter

import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# Image generators (Keras 3 vs tf.keras fallback)
try:
    from keras.preprocessing.image import ImageDataGenerator
except Exception:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

from PIL import Image

# --- Make project root importable so we can `import base.*` when this file lives in example_scripts/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# shared-layer registry (for live weight sharing + reset before final retrain)
from base.shared_layers import REGISTRY
from base.population import Population
from base.structures import Datasets
from base.config import LOG_DETAIL  # custom logging level 21


# ---------------------------
# helpers
# ---------------------------
def _pick_val_metric_name():
    """Choose the correct val_* metric name."""
    return "val_accuracy"


def _build_callbacks(log_csv, ckpt_path):
    monitor = _pick_val_metric_name()
    return [
        EarlyStopping(monitor=monitor, mode="auto", verbose=1, patience=15),
        CSVLogger(log_csv),
        ModelCheckpoint(ckpt_path, monitor=monitor, mode="auto",
                        verbose=1, save_best_only=True),
    ]


def _ensure_dirs():
    def create_dir(path):
        d = os.path.dirname(path.rstrip("/")) if path.endswith("/") else path
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    create_dir(os.path.join(ROOT, "models/"))
    create_dir(os.path.join(ROOT, "images/"))


# --- tiny loader for EA stage (numpy arrays) ---
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _load_split_as_arrays(split_dir, class_names, img_size, max_per_class=None):
    """
    Loads a manageable subset of images per class from directory -> numpy arrays.
    Useful for the EA/evaluation stage which expects (x, y) arrays.
    """
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


# ---------------------------
# main runner
# ---------------------------
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
    magic_t=True,
    magic_t_sync_every=1,
    magic_t_elastic_tau=0.0,
    magic_t_migrate_p=0.10,
    magic_t_noise_std=0.0,
    magic_t_log_keys=True,
    clear_registry_before_final=True,
):
    # --- search spaces / configs (kept lean for speed) ---
    global_configs = {"module_range": ([1, 3], "int"), "component_range": ([1, 3], "int")}
    input_configs  = {"module_range": ([1, 1], "int"), "component_range": ([1, 1], "int")}
    output_configs = {"module_range": ([1, 1], "int"), "component_range": ([1, 1], "int")}

    possible_components = {
        "conv2d": (
            keras.layers.Conv2D,
            {
                "filters": ([16, 96], "int"),
                "kernel_size": ([1, 3, 5], "list"),
                "strides": ([1], "list"),
                "data_format": (["channels_last"], "list"),
                "padding": (["same"], "list"),
                "activation": (["relu"], "list"),
            },
        ),
        "maxpool": (
            keras.layers.MaxPooling2D,
            {
                "pool_size": ([2, 3], "list"),
                "strides": ([2], "list"),
                "padding": (["valid"], "list"),
            },
        ),
        "bn": (keras.layers.BatchNormalization, {}),
    }

    possible_inputs = {
        "conv2d": (
            keras.layers.Conv2D,
            {"filters": ([16, 64], "int"), "kernel_size": ([1], "list"),
             "activation": (["relu"], "list")},
        ),
    }

    possible_outputs = {
        "dense": (
            keras.layers.Dense,
            {"units": ([32, 256], "int"), "activation": (["relu"], "list")},
        ),
    }

    possible_complementary_components = {
        "dropout": (keras.layers.Dropout, {"rate": ([0.0, 0.5], "float")}),
    }
    possible_complementary_inputs  = None

    # (temporary; will be overwritten after loading classes)
    possible_complementary_outputs = {
        "dense": (
            keras.layers.Dense,
            {"units": ([10, 10], "int"), "activation": (["softmax"], "list")},
        ),
    }

    # --- dataset (COVID-19 Radiography) ---
    DATA_DIR = os.path.join(ROOT, "data", "COVID_Radiography")  # created by your split script
    IMG_SIZE = 128
    BATCH_SIZE = 32
    CLASSES = ["COVID", "Normal", "Viral_Pneumonia", "Lung_Opacity"]
    num_classes = len(CLASSES)

    # Check expected directories
    for sub in ["train", "val", "test"]:
        for cname in CLASSES:
            path = os.path.join(DATA_DIR, sub, cname)
            if not os.path.isdir(path):
                raise RuntimeError(
                    f"Expected folder not found: {path}\n"
                    f"Run your split/prep script to create train/val/test folders first."
                )

    # EA stage expects arrays -> load a subset per class to keep memory reasonable
    max_per_class_train = 1500 if not fast_debug else 400
    max_per_class_test  = 300 if not fast_debug else 120

    x_train, y_train = _load_split_as_arrays(
        os.path.join(DATA_DIR, "train"),
        CLASSES, IMG_SIZE, max_per_class=max_per_class_train
    )
    x_test, y_test = _load_split_as_arrays(
        os.path.join(DATA_DIR, "test"),
        CLASSES, IMG_SIZE, max_per_class=max_per_class_test
    )

    # Generators for the *final retrain* (use full train/val)
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=5,
        zoom_range=0.10
    )
    datagen_eval = ImageDataGenerator(rescale=1./255)

    train_gen = datagen_train.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=[c for c in CLASSES],
        shuffle=True
    )
    val_gen = datagen_eval.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=[c for c in CLASSES],
        shuffle=False
    )

    # Class weights for imbalance (used only in final retrain)
    counts = Counter(train_gen.classes)
    total = sum(counts.values())
    class_weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}

    validation_split = 0.15   # used when fitting on the EA arrays

    # lock output head units to num_classes
    possible_complementary_outputs = {
        "dense": (
            keras.layers.Dense,
            {"units": ([num_classes, num_classes], "int"), "activation": (["softmax"], "list")},
        ),
    }

    # --- logging ---
    logging.basicConfig(
        filename=os.path.join(ROOT, "execution.log"),
        filemode="w+",
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s: %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s: %(message)s"))
    logging.getLogger().addHandler(console)

    logging.log(LOG_DETAIL, "runner started")
    logging.info("Hi, this is a COVID CXR run.")

    compiler = {
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy", keras.metrics.AUC(name="auc", multi_label=False)]
    }

    # --- build populations ---
    populations = []
    for _ in range(num_populations):
        pop = Population(
            Datasets(training=[x_train, y_train], test=[x_test, y_test]),
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            population_size=population_size,
            compiler=compiler
        )

        # caps to keep EA fast; comment out to use all loaded arrays
        pop.datasets.SAMPLE_SIZE = min(20000, len(x_train))
        pop.datasets.TEST_SAMPLE_SIZE = min(2000, len(x_test))
        pop.datasets.custom_fit_args = None

        # build search spaces
        pop.module_share_mode = share_mode
        pop.create_module_population(
            module_population_size, global_configs,
            possible_components, possible_complementary_components
        )
        pop.create_module_species(n_module_species)
        pop.create_blueprint_population(
            blueprint_population_size,
            global_configs, possible_components, possible_complementary_components,
            input_configs, possible_inputs, None,
            output_configs, possible_outputs, possible_complementary_outputs,
        )
        pop.create_blueprint_species(n_blueprint_species)

        # evolution + sync settings
        pop.parallel_sync = parallel_sync
        pop.ema_beta = ema_beta
        pop.fast_debug = fast_debug
        pop.max_steps_per_epoch = max_steps_per_epoch
        pop.disable_graph_plots = disable_graph_plots

        # MAGIC-T toggles (if present in your fork)
        if hasattr(pop, "magic_t"):
            pop.magic_t = bool(magic_t)
        if hasattr(pop, "magic_t_sync_every"):
            pop.magic_t_sync_every = int(magic_t_sync_every)
        if hasattr(pop, "magic_t_elastic_tau"):
            pop.magic_t_elastic_tau = float(magic_t_elastic_tau)
        if hasattr(pop, "magic_t_migrate_p"):
            pop.magic_t_migrate_p = float(magic_t_migrate_p)
        if hasattr(pop, "magic_t_noise_std"):
            pop.magic_t_noise_std = float(magic_t_noise_std)
        if hasattr(pop, "magic_t_log_keys"):
            pop.magic_t_log_keys = bool(magic_t_log_keys)

        # Optional cadence tweaks if your Population supports them
        if hasattr(pop, "conflict_report_every"):
            pop.conflict_report_every = 3

        populations.append(pop)

    # --- evolve + retrain ---
    for i, population in enumerate(populations):
        if hasattr(population, "log_shared_state"):
            population.log_shared_state(f"init_pop{i}")

        iteration = population.iterate_generations(
            generations=generations,
            training_epochs=training_epochs,
            validation_split=validation_split,
            mutation_rate=0.5,
            crossover_rate=0.2,
            elitism_rate=0.1,
            possible_components=possible_components,
            possible_complementary_components=possible_complementary_components,
        )

        if hasattr(population, "log_shared_state"):
            population.log_shared_state(f"post_iter_pop{i}")
        print("Best fitting:", iteration)

        # switch to generator-based full training using ALL train/val data
        improved_ds = Datasets(training=None, test=None)
        improved_ds.training = [x_train[:1], y_train[:1]]
        improved_ds.test     = [x_test[:1],  y_test[:1]]
        improved_ds.custom_fit_args = {
            "generator": train_gen,
            "steps_per_epoch": max(1, train_gen.samples // BATCH_SIZE),
            "epochs": final_model_training_epochs,
            "verbose": 1,
            "validation_data": val_gen,
            "validation_steps": max(1, val_gen.samples // BATCH_SIZE),
            "class_weight": class_weights,
            "callbacks": _build_callbacks(
                os.path.join(ROOT, "training.csv"),
                os.path.join(ROOT, "best_model_checkpoint.h5"),
            ),
        }
        population.datasets = improved_ds
        print("Using directory generators for final training.")

        if clear_registry_before_final:
            REGISTRY.clear()
            print("[REGISTRY] Cleared before final retraining.")

        best_model = population.return_best_individual()
        try:
            print(f"Best fitting model chosen for retraining: {best_model.name}")
            population.train_full_model(
                best_model,
                final_model_training_epochs,
                validation_split,
                custom_fit_args=improved_ds.custom_fit_args,
                warm_start=False,
            )
        except Exception as e:
            logging.warning(f"Final retrain failed on chosen model ({best_model.name}): {e}. Retrying with next best.")
            # Fall back to "best again" (Population will pick from last iteration summary)
            best_model = population.return_best_individual()
            print(f"Best fitting model chosen for retraining: {best_model.name}")
            population.train_full_model(
                best_model,
                final_model_training_epochs,
                validation_split,
                custom_fit_args=improved_ds.custom_fit_args,
                warm_start=False,
            )


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="CoDeepNEAT on COVID-19 Radiography (with weight sharing & MAGIC-T toggles)")
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--training-epochs", type=int, default=1)
    ap.add_argument("--final-model-training-epochs", type=int, default=1)
    ap.add_argument("--num-populations", type=int, default=1)
    ap.add_argument("--population-size", type=int, default=6)
    ap.add_argument("--blueprint-population-size", type=int, default=5)
    ap.add_argument("--module-population-size", type=int, default=20)
    ap.add_argument("--n-blueprint-species", type=int, default=3)
    ap.add_argument("--n-module-species", type=int, default=3)

    ap.add_argument("--share-mode", choices=["module", "layer", "global", "off"], default="module")
    ap.add_argument("--parallel-sync", action="store_true", default=True)
    ap.add_argument("--no-parallel-sync", dest="parallel_sync", action="store_false")
    ap.add_argument("--ema-beta", type=float, default=0.5)
    ap.add_argument("--fast-debug", action="store_true", default=True)
    ap.add_argument("--no-fast-debug", dest="fast_debug", action="store_false")
    ap.add_argument("--max-steps-per-epoch", type=int, default=10)
    ap.add_argument("--disable-graph-plots", action="store_true", default=True)
    ap.add_argument("--enable-graph-plots", dest="disable_graph_plots", action="store_false")

    ap.add_argument("--magic-t", action="store_true", default=False)
    ap.add_argument("--magic-t-sync-every", type=int, default=1)
    ap.add_argument("--magic-t-elastic-tau", type=float, default=0.0)
    ap.add_argument("--magic-t-migrate-p", type=float, default=0.10)
    ap.add_argument("--magic-t-noise-std", type=float, default=0.0)
    ap.add_argument("--magic-t-log-keys", action="store_true", default=True)
    ap.add_argument("--no-magic-t-log-keys", dest="magic_t_log_keys", action="store_false")

    ap.add_argument("--clear-registry-before-final", action="store_true", default=True)
    ap.add_argument("--no-clear-registry-before-final", dest="clear_registry_before_final", action="store_false")
    return ap.parse_args()


if __name__ == "__main__":
    _ensure_dirs()
    args = parse_args()
    run_covid_cxr_full(
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
        magic_t_sync_every=args.magic_t_sync_every,
        magic_t_elastic_tau=args.magic_t_elastic_tau,
        magic_t_migrate_p=args.magic_t_migrate_p,
        magic_t_noise_std=args.magic_t_noise_std,
        magic_t_log_keys=args.magic_t_log_keys,
        clear_registry_before_final=args.clear_registry_before_final,
    )
