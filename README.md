# Dependencies
```
sudo apt install ocl-icd-opencl-dev
```

## X11 Dependencies
```
sudo apt install xorg-dev libxkbcommon-dev

```

## Wayland Dependencies
```
sudo apt install libwayland-dev
```

# Submodules
```
git submodule update --init --recursive
```

# Build and Run
```
mkdir build
cd build
cmake ..
make
./GameOfLife # GUI
./GameOfLifeTest # Test simulation (NO GUI)
```

## Examples runs
### Run CPU Test
```
 ./GameOfLifeTest --cpu --seconds 10 --workgroup-x 32 --workgroup-y 32 -n 10 -m 10
```
### Run OpenCL Test with Local memory
```
 ./GameOfLifeTest --opencl --local --seconds 10 --workgroup-x 32 --workgroup-y 32 -n 100 -m 100
```
### Run CUDA Test
```
 ./GameOfLifeTest --cuda --seconds 10 --workgroup-x 32 --workgroup-y 32 -n 100 -m 100
```
