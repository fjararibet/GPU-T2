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
