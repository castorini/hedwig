# Instructions for DSG Hedwig Contributors

Please follow these instructions if you are a graduate student or undergrad research assistant working with the group
in the Data Systems Lab and want to run Hedwig on the lab desktop GPU machine (dragon).

If you have trouble / questions with instructions on this page, ping @tuzhucheng on Slack.

## PyTorch Environment

We already have a multi-user Conda environment with PyTorch and all other dependencies installed, so you do not need to
install anything yourself. However, you can create [Conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html)
if you need to experiment with different library versions etc.

The multi-user Conda environment is located at `/anaconda3/`.
To use this multi-user environment, just add the following to your `.bashrc` or configuration file for your favourite shell.

```bash
export PATH="/anaconda3/bin:$PATH"
export LIBRARY_PATH="/usr/lib/nvidia-375"
```

Please also ensure `/usr/local/cuda-8.0/lib64` is in the `LD_LIBRARY_PATH` environment variable **if it is not already**.
If not, you should add it in the `.bashrc` similar to above.

Please re-login or re-source your shell configuration after `.bashrc` is updated for the updated environment variables
to take effect.

## Data and Pre-Trained Models

We use shared cloned versions of the Hedwig-data and Hedwig-models repositories.
Instead of making your own cloned copies, you can just create symbolic links to the shared version instead
in your own working directory to save disk space. Assuming you want to put `Hedwig`, `Hedwig-data`, and `Hedwig-models`
under a directory called `Castorini` and you are currently in the `Castorini` directory, you can enter these commands:

```bash
ln -s /Hedwig-data Hedwig-data
ln -s /Hedwig-models Hedwig-models
```

So after you clone Hedwig, you have a directory structure under `Castorini` that looks like this:

```
.
├── Hedwig
├── Hedwig-data
└── Hedwig-models
```

where `Hedwig-data` and `Hedwig-models` are actually symbolic links to `/Hedwig-data` and `/Hedwig-models`.
