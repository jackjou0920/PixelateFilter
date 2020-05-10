# PixelateFilter
It was one of assignment of optional courses *- Parallel Computing with GPUs* in my second term in Sheffield University. 
The project was to implement a simple pixelate filter on a photografic image input with any size and then produced its 
photo mosaic by averaging RGB colour values of the image cells through C language. How to understand and analyse the efficency of the programme with different design ways and executive modes, such as **OpenMP** on CPU and **CUDA** on GPU, were the main goal.

## Requirements
* A windows computer with NVDIA GeForce GPU
* Development enviroment on Visual Studio 2017 with CUDA Toolkit

## Features
* It works for all images sizes as input and transfers to mosaic images with different cell sizes freely
* Speeded up by OpenMP and parallel design on CUDA
* Analysed the utilisation of memory through Visual Studio profiler in order to progressively improve the code

## Results
