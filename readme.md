# vkfft-rs

`vkfft-rs` allows high-performance execution of 1, 2, or 3D FFTs on the GPU using Vulkan in Rust, with built-in support for convolutions.

`vkfft-rs` is a binding for [VkFFT](https://github.com/DTolm/VkFFT) that assumes usage with [vulkano](https://vulkano.rs/). While VkFFT, despite the name, supports multiple backends, this wrapper requires usage with Vulkan.

While `vkfft-rs` attempts to maintain a safe API, it's very likely there are some safe functions in this codebase that can still cause unsafe behavior. VkFFT's API and associated data structures are unsafe and stateful, which presents difficulties in ensuring Rust's safety guarantees. Until its safety properties can be properly verified it is recommend to proceed with caution. PRs welcome!

## Building

```.sh
# Clone VkFFT
git clone https://github.com/DTolm/VkFFT.git

# Navigate into the folder
cd VkFFT

# Create a build directory (this currently must be named "build"!)
mkdir build && cd build

# Configure build
cmake ..

# Build
make

# Build vkfft-rs
cd vkfft-rs

# VKFFT_ROOT must be set to the root directory of VkFFT!
export VKFFT_ROOT=/path/to/VkFFT

# Build
cargo build --examples

# Run convolution example
cargo run --example convolution
```

### IMPORTANT

If your system already has `libSPIRV.a` in the library search path and are encountering strange segmentation faults
in SPIRV at runtime, it's possible Rust has linked against the system `libSPIRV.a` rather than the one in VkFFT's `build`
directory. These different libraries might be ABI incompatible.

This is unfortunately a limitation of cargo/rustc's ability for a crate to specify absolute paths for static libraries. It is recommended to, unfortunately, remove the other `libSPIRV.a` from the system library path.

For example, on Ubuntu:
```.sh
sudo mv /usr/lib/x86_64-linux-gnu/libSPIRV.a /usr/lib/x86_64-linux-gnu/libSPIRV.a.backup 
```
