import os
from os.path import join

import nnunetv2
from nnunetv2.paths import nnUNet_extTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def recursive_find_trainer_class_by_name(trainer_name: str):
    # Import here is necessary to avoid circular import
    # this function is used in the training and inference scripts
    # but the inference script needs to import the trainer class
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
        nnunetv2.__path__[0],
    )

    if nnunet_trainer is None:
        if nnUNet_extTrainer.is_set():
            ext_paths = nnUNet_extTrainer.get().split(os.pathsep)
            print(
                f"Trainer '{trainer_name}' not found in nnunetv2.training.nnUNetTrainer.\n"
                f"Searching in external trainer paths from environment variable 'nnUNet_extTrainer'..."
            )
            for path in ext_paths:
                if path.strip() and os.path.exists(path):
                    print(f"Searching in: {path}")
                    package_trainers = join(path, "trainers")
                    package_init = join(path, "__init__.py")
                    if os.path.isdir(package_trainers) and os.path.isfile(package_init):
                        package_name = os.path.basename(os.path.abspath(path))
                        search_locations = [
                            (package_trainers, f"{package_name}.trainers", os.path.dirname(os.path.abspath(path)))
                        ]
                    else:
                        search_locations = [(path, None, path)]

                    for search_folder, current_module, base_folder in search_locations:
                        candidate = recursive_find_python_class(
                            search_folder,
                            trainer_name,
                            current_module,
                            base_folder=base_folder,
                            verbose=True,
                            cleanup_imports_from_base_folder=True,
                        )
                        if candidate is not None:
                            # The cleanup pass is required while trying multiple external
                            # roots with overlapping package names. Once a class is found,
                            # however, its defining modules must remain in sys.modules so
                            # multiprocessing workers can pickle Trainer-owned datasets.
                            # Repeat the successful lookup without cleanup and return that
                            # persistent class object.
                            nnunet_trainer = recursive_find_python_class(
                                search_folder,
                                trainer_name,
                                current_module,
                                base_folder=base_folder,
                                verbose=False,
                                cleanup_imports_from_base_folder=False,
                            )
                            print(f"Using trainer '{trainer_name}' from: {path}")
                            break
                    if nnunet_trainer is not None:
                        break
        if nnunet_trainer is None:
            raise RuntimeError(
                f"Could not find requested nnunet trainer {trainer_name} in "
                f"nnunetv2.training.nnUNetTrainer ("
                f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). '
                f"If the trainer is located elsewhere, please move it there or specify the external path via the "
                f"`nnUNet_extTrainer` environment variable. "
                f"nnUNet_extTrainer={os.environ.get('nnUNet_extTrainer', '')}"
            )
    assert issubclass(nnunet_trainer, nnUNetTrainer), (
        "The requested nnunet trainer class must inherit from 'nnUNetTrainer'"
    )
    return nnunet_trainer
