# Clone and Run

```shell
git clone --recurse-submodules https://github.com/username/repository.git
```

Install CMake on Ubuntu:

```shell
sudo snap install cmake --classic
```

# Intellisense

In a environment managed dynamically, such as spack, use `cluster` configuration for VS Code. You may need to generate `./build/compile_commands.json` first to make sure cmake finds the correct paths to compilers.
