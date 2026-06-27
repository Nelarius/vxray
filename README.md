```
▄▄ ▄▄ ▄▄ ▄▄ ▄▄▄▄   ▄▄▄  ▄▄ ▄▄ 
██▄██ ▀█▄█▀ ██▄█▄ ██▀██ ▀███▀ 
 ▀█▀  ██ ██ ██ ██ ██▀██   █   
```

Configuring and building the project requires [CMake](https://cmake.org) and [Metal Developer Tools](https://developer.apple.com/metal/tools/).

## Configure and Build

Use the basic configuration command to get a debug build:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

[ccache](https://ccache.dev/) and [clangd](https://clangd.llvm.org/) are recommended tools in this project. This configuration command generates a build supporting both:

```sh
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER= ccache -DCMAKE_BUILD_TYPE=Debug
```
