"""Stub for the data package.

The inference path never instantiates a dataset (``opt.dataset_mode`` is
``None``), but ``base_options.py`` still imports this module at parse time.
"""


def get_option_setter(dataset_name):  # pragma: no cover - never reached
    raise RuntimeError(
        "Dataset loaders are not bundled in the inference release; "
        "leave --dataset_mode unset."
    )
