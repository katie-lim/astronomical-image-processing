# Instructions

Create a new Anaconda environment:

```console
conda create -y --name astro-image-proc python=3.10
```

Install the required modules:

```console
conda install --force-reinstall -y --name astro-image-proc -c conda-forge --file requirements.txt
```

Activate the new environment:

```console
conda activate astro-image-proc
```

Run `analyse.py`.

```console
python analyse.py
```