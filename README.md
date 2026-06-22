```
▄▄ ▄▄ ▄▄ ▄▄ ▄▄▄▄   ▄▄▄  ▄▄ ▄▄ 
██▄██ ▀█▄█▀ ██▄█▄ ██▀██ ▀███▀ 
 ▀█▀  ██ ██ ██ ██ ██▀██   █   
```

## Configure and Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug
# Build with clangd support
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug
```
