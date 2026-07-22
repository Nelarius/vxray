```
▄▄ ▄▄ ▄▄ ▄▄ ▄▄▄▄   ▄▄▄  ▄▄ ▄▄ 
██▄██ ▀█▄█▀ ██▄█▄ ██▀██ ▀███▀ 
 ▀█▀  ██ ██ ██ ██ ██▀██   █   
```

Configuring and building the project requires [CMake](https://cmake.org) and [Metal Developer Tools](https://developer.apple.com/metal/tools/).

## Configure and Build

**Basics**

Use the basic configuration command to get a debug build:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

**Xcode**

Work around a SDL_shadercross incompatibility issue with Xcode's new build system by reverting to the old one:

```sh
cmake -B build-xcode -S . -G Xcode -T buildsystem=1
```

**Ccache and clangd**

[ccache](https://ccache.dev/) and [clangd](https://clangd.llvm.org/) are recommended tools in this project. This configuration command generates a build supporting both:

```sh
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   \
               -DCMAKE_C_COMPILER_LAUNCHER=ccache   \
               -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
               -DCMAKE_BUILD_TYPE=Debug
```

## Run

### vxray

Run vxray with a Magicavoxel file:

```sh
./build/vxray <file.vox> [camera.vx]
```

Without a camera preset, vxray uses the default view for the scene. Press F2 to print the current
camera preset to standard output, then save the three printed lines as a `.vx` file and pass it as
the optional second argument.

```text
position = 1 2 3
yaw = 0
pitch = 0
```

## References

- _Ray Axis-Aligned Bounding Box Intersection_, [Ray Tracing Gems II](https://developer.nvidia.com/ray-tracing-gems-ii)
