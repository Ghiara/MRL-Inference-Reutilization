# Experiments
This directory contains experiment files which are sorted by environment.

Most of the files are structured as follows: They first define a set of 
configuration dictionaries (with the help of functions defined in directory
*config_modules*) and then have a call to train agents with these configurations.

You can directly use the configurations by importing them or call the script
to execute them all. Here is an example:

In *toy1d/buffer_types.py*, we define multiple configurations (including ``on_policy``) which differ in
their inference exploration policy. We can ...

1. ... execute
    ```batch
    python runner.py \
        configs.environment_factory.toy1d \
        experiments.experiments.toy1d.on_off_policy.on_policy
    ```
    to run the configuration ``on_policy`` with the runner-script (see *runner.py*),
2. ... execute
    ```batch
    python experiments/experiments/toy1d/on_off_policy.py
    ```
    to run all configurations consecutively (minor adaptations in the script may
    be required, e.g. changing paths), or
3. ... import the selected configuration into a custom script:
    ```python
    from experiments.experiments.toy1d.on_off_policy import on_policy
    ```
    and use it to set up an experiment manually.
