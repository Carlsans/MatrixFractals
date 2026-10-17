# MatrixFractals

Genetic search for 3D fractals defined by a matrix of functions, rendered on the GPU and exported as stereo 360 VR video.

A cell of the matrix is a complex number, a function of `C`, `D` or `Z`, or a bare variable. Rows are multiplied, row
results are summed, and the sum is iterated. Candidates are scored by how poorly their rendering compresses.

- How it works, what worked and what did not: [docs/PROJECT.md](docs/PROJECT.md)
- Features that were tried and then dropped: [docs/UNUSED_FEATURES.md](docs/UNUSED_FEATURES.md)

## GUI

```
git clone --recurse-submodules https://github.com/Carlsans/MatrixFractals
cd MatrixFractals
pip install -r requirements-gui.txt
python -m gui.app
```

Tabs: **Explore** (edit a matrix cell by cell, render a volume, scrub the layers, save/open `.frac`),
**Genetic algorithm** (seed, step and run the pool, send a result to Explore), **VR export** (preview a frame,
export the stereo video) and **Legacy / experimental** (random search, hill climb, volume projections).

Needs `ffmpeg` on the path. A CUDA GPU is needed for the VR tab and recommended for everything else.
`.frac` files are Python pickles: open only files you created.

## Original scripts

`main.py` contains the original entry points (choose by editing the `__main__` block). Its modules import `src.X`, so run
it with the parent directory on `PYTHONPATH`. The GUI handles this itself.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Personal, research and hobby use are allowed; commercial use is not.

## Launch the GUI

Linux: `./run.sh`  
Windows: `run_win.bat`

The first run creates a `.venv` and installs `requirements-gui.txt`; later runs start immediately.
