# OpenMP Offload C++ Tutorials

These samples are collection of Jupyter Notebooks that demonstrate OpenMP Offload.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to offload computation to GPUs using OpenMP with the Intel® oneAPI DPC++/C++ Compiler
| Time to complete      | 2 Hours
| Category              | Tutorial

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Linux*
| Hardware              | GEN9 or newer
| Software              | Intel® oneAPI DPC++/C++ Compiler

### Running the Notebooks

1. Open the Notebook `OpenMP Welcome.ipynb`.

2. Open the modules you want to review.

3. Follow the instructions in each notebook and execute cells when instructed.

## Summary of the Jupyter Notebooks

[OpenMP Welcome](OpenMP&nbsp;Welcome.ipynb)
- Introduce Developer Training Modules
- Describe oneAPI Tool Modules

[Introduction to OpenMP Offload](intro)
- oneAPI Software Model Overview and Workflow
- HPC Single-Node Workflow with oneAPI
- Simple OpenMP Code Example
- Target Directive Explanation
- *Lab Exercise*: Vector Increment with Target Directive

[Managing Data Transfers](datatransfer)
- Offloading Data
- Target Data Region
- Mapping Global Variable to Device
- *Lab Exercise*: Target Data Region

[Utilizing GPU Parallelism](parallelism)
- Device Parallelism
- OpenMP Constructs and Teams
- Host Device Concurrency
- *Lab Exercise*: OpenMP Device Parallelism

[Unified Shared Memory](USM)
- Allocating Unified Shared Memory
- USM Explicit Data Movement
- *Lab Exercise*: Unified Shared Memory

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](third-party-programs.txt).
