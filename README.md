# Clone and Run

```shell
git clone --recurse-submodules https://github.com/username/repository.git
```

Install CMake on Ubuntu:

```shell
sudo snap install cmake --classic
```

# Intellisense

In a environment managed dynamically, such as spack, use `cluster` configuration for VS Code. You may need to generate `./build/compile_commands.json` first to make sure cmake finds the correct paths to compilers. This can also mitigate memory pressure on less performant machines.

```shell
cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
