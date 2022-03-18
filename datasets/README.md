# External datasets
For datasets not available in `hedwig-data`, run the script `download_datasets.py` from the
repository root. It requires a different version of pytorch, so do this from a different virtual
environment, with the following dependencies:

```
dill==0.3.4
torchdata==0.3.0
torchtext==0.12.0
```

Some of the torchtext datasets cannot be downloaded from torchtext directly due to
an issue with torchtext and Google Drive. For these, the script will show the download
link and you should download and extract it manually.

Once all datasets are downloaded, run `process_datasets.py`. This will perform some
pre-processing on the csv files.
