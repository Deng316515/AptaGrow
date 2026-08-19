# Reference Linux environment

This guide defines a reproducible setup for `config/reference_linux.yaml`.
Commands assume Ubuntu, an NVIDIA GPU, and the
reference `/root` installation layout. Use equivalent non-root locations on a
shared cluster and update the YAML paths accordingly.

## 1. Create the Python environment

```bash
conda env create -f environment.yml
conda activate aptagrow
pip install -e .

# Confirm the versions recorded in the manuscript/configuration.
python -c "import RNA; print(RNA.__version__)"
pymol -c -d "print(cmd.get_version())"
```

The reference metadata records ViennaRNA 2.4.18 and PyMOL 2.5.4. If the
unconstrained conda-forge PyMOL package resolves to another version, install the
matching build or update both the manuscript and `software_versions` metadata.

## 2. Install MGLTools 1.5.7

Download `mgltools_x86_64Linux2_1.5.7.tar.gz` from the
[official MGLTools downloads page](https://ccsb.scripps.edu/mgltools/downloads/),
then run:

```bash
tar -xzf mgltools_x86_64Linux2_1.5.7.tar.gz
cd mgltools_x86_64Linux2_1.5.7
bash install.sh
cd ..
```

The pipeline expects:

```text
/root/mgltools_x86_64Linux2_1.5.7/bin/pythonsh
/root/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py
```

## 3. Configure OpenCL for Vina-GPU 2.1

```bash
sudo apt update
sudo apt install -y ocl-icd-opencl-dev opencl-headers clinfo nvidia-cuda-toolkit
find /usr -name 'libnvidia-opencl.so.*' 2>/dev/null
sudo mkdir -p /etc/OpenCL/vendors
```

Write the **actual** library returned by `find` to the NVIDIA ICD file. The
driver-specific filename below is only an example:

```bash
echo "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.<driver-version>" \
  | sudo tee /etc/OpenCL/vendors/nvidia.icd
sudo ldconfig
clinfo
```

`clinfo` must report at least one platform. Set the stack size used in the
reported environment:

```bash
ulimit -s 8192
```

## 4. Obtain Vina-GPU 2.1 and build Boost 1.84.0

```bash
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git /root/Vina-GPU-2.1
cd /root/Vina-GPU-2.1
wget https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
tar -xzf boost_1_84_0.tar.gz
cd boost_1_84_0
./bootstrap.sh --with-libraries=system,filesystem,program_options,thread
./b2
```

## 5. Build Vina-GPU 2.1

```bash
cd /root/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1
```

Use these reference Makefile values, adjusting the OpenCL path to the installed
CUDA toolkit when necessary:

```makefile
WORK_DIR=/root/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1
BOOST_LIB_PATH=../boost_1_84_0
OPENCL_LIB_PATH=/usr/local/cuda-12.4
OPENCL_VERSION=-DOPENCL_3_0
GPU_PLATFORM=-DNVIDIA_PLATFORM
DOCKING_BOX_SIZE=-DSMALL_BOX
```

Compile the source-kernel build:

```bash
make clean
make source
test -x ./AutoDock-Vina-GPU-2-1
```

## 6. Install SimRNA 3.20

```bash
wget https://genesilico.pl/software/simrna/version_3.20/SimRNA_64bitIntel_Linux.tgz \
  -O /root/SimRNA_64bitIntel_Linux.tgz
tar -xzf /root/SimRNA_64bitIntel_Linux.tgz -C /root
chmod +x /root/SimRNA_64bitIntel_Linux/SimRNA
chmod +x /root/SimRNA_64bitIntel_Linux/SimRNA_trafl2pdbs
```

The AptaGrow implementation reads TRAFL files directly and does not call
`trafl_extract_lowestE_frame.py`; consequently, the legacy script's Python 2
`print` syntax does not need to be patched with `autopep8` or `2to3`.

## 7. Validate the complete setup

Place the PFOA ligand and standard nucleotide templates as described in
`data/README.md`, then run:

```bash
aptagrow --config config/reference_linux.yaml doctor
aptagrow --config config/reference_linux.yaml build-library --max-sequences 256
```

The first command should report every tool path as present and all required
Python modules as available. The second command is only a smoke test; remove
`--max-sequences` for the manuscript-scale run.
