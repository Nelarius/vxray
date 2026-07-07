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
./build/vxray <file.vox>
```

Print the current camera coordinates as a `vx_camera` initializer by pressing F2.

## References

- _Ray Axis-Aligned Bounding Box Intersection_, [Ray Tracing Gems II](https://developer.nvidia.com/ray-tracing-gems-ii)
