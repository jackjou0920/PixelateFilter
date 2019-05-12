#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acp18dj"		//my user name

void print_help();
int process_command_line(int argc, char *argv[], int *c, char *input_file, char *output_file, char *ppm_format);
int is_two_n(int num);
int image_input(char *input_file, int c, int *width, int *height, char *header, char *output_file, char *ppm_format);
int read_header(FILE *fp, int c, int *width, int *height, char *header, char *output_file, char *ppm_format, char *format);
void cul_average_cpu(int c, int width, int height, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b);
void cul_average_openmp(int c, int width, int height, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b);
void launch_cuda_1D(int c, int width, int height, unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b);
//void launch_cuda_2D(int c, int width, int height, unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b);
void transform_2D_to_1D(unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, int width, int height);
void transform_1D_to_2D(unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, int width, int height);
int image_output(int width, int height, char *ppm_format, char *header, char *output_file);
void checkCUDAError(const char *msg);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

MODE execution_mode = CPU;
unsigned char **image_r, **image_g, **image_b;

//texture<unsigned char, cudaTextureType2D> texData_r;
//texture<unsigned char, cudaTextureType2D> texData_g;
//texture<unsigned char, cudaTextureType2D> texData_b;

__device__ unsigned long long int average_r, average_g, average_b;


__global__ void avgKernel_1D(uchar3 *image, const int width, const int height, const int c) {
	// declare share memory for calculating average across the threads
	
	__shared__ unsigned long long int sdata_r, sdata_g, sdata_b;
	//__shared__ float3 sdata;

	// the local variables for summerising values for certain thread
	uchar3 pixel;
	unsigned long long int sum_r = 0, sum_g = 0, sum_b = 0;

	// how many row should be pass for one thread
	int colPerThread = (c > 1024) ? c / 1024 : 1;

	// the imcompleted block index and its width and height 
	int block_x = -1, block_y = -1, rest_x = 0, rest_y = 0;
	if (width % c != 0) {
		block_x = width / c;
		rest_x = width % c;
	}
	if (height % c != 0) {
		block_y = height / c;
		rest_y = height % c;
	}

	// over the width boundary
	if (blockIdx.x != block_x || threadIdx.x < rest_x) {
		for (unsigned int j = 0; j < colPerThread; j++) {
			for (unsigned int i = 0; i < c; i++) {
				// over the height boundary
				if (blockIdx.y == block_y && i >= rest_y) continue;
				if ((blockIdx.x * c + threadIdx.x + j * 1024) >= width) continue;

				unsigned int offset = (blockIdx.y * width * c) + (blockIdx.x * c + threadIdx.x + j * 1024) + (width * i);
				pixel = image[offset];
				/*atomicAdd(&sdata_r, gpu_r[offset]);
				atomicAdd(&sdata_g, gpu_g[offset]);
				atomicAdd(&sdata_b, gpu_b[offset]);*/

				// summerise a the thread
				sum_r += pixel.x;
				sum_g += pixel.y;
				sum_b += pixel.z;
			}
		}
		// summerise within a block 
		/*atomicAdd(&sdata.x, sum_r);
		atomicAdd(&sdata.y, sum_g);
		atomicAdd(&sdata.z, sum_b);*/
		atomicAdd(&sdata_r, sum_r);
		atomicAdd(&sdata_g, sum_g);
		atomicAdd(&sdata_b, sum_b);
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		// summerise the values of all blocks
		/*atomicAdd(&average_r, sdata.x);
		atomicAdd(&average_g, sdata.y);
		atomicAdd(&average_b, sdata.z);*/
		atomicAdd(&average_r, sdata_r);
		atomicAdd(&average_g, sdata_g);
		atomicAdd(&average_b, sdata_b);

		// calculate the average with different size
		if (blockIdx.x == block_x && blockIdx.y == block_y) {
			/*sdata.x /= rest_x * rest_y;
			sdata.y /= rest_x * rest_y;
			sdata.z /= rest_x * rest_y;*/
			sdata_r /= rest_x * rest_y;
			sdata_g /= rest_x * rest_y;
			sdata_b /= rest_x * rest_y;
		}
		else if (blockIdx.x == block_x && blockIdx.y != block_y) {
			/*sdata.x /= rest_x * c;
			sdata.y /= rest_x * c;
			sdata.z /= rest_x * c;*/
			sdata_r /= rest_x * c;
			sdata_g /= rest_x * c;
			sdata_b /= rest_x * c;
		}
		else if (blockIdx.x != block_x && blockIdx.y == block_y) {
			sdata_r /= c * rest_y;
			sdata_g /= c * rest_y;
			sdata_b /= c * rest_y;
		}
		else {
			sdata_r /= c * c;
			sdata_g /= c * c;
			sdata_b /= c * c;
		}
	}

	__syncthreads();
	// fill in the avreage values
	if (blockIdx.x != block_x || threadIdx.x < rest_x) {
		for (unsigned int j = 0; j < colPerThread; j++) {
			for (unsigned int i = 0; i < c; i++) {
				if (blockIdx.y == block_y && i >= rest_y) continue;
				if ((blockIdx.x * c + threadIdx.x + j * 1024) >= width) continue;
				unsigned int offset = (blockIdx.y * width * c) + (blockIdx.x * c + threadIdx.x + j * 1024) + (width * i);

				image[offset].x = sdata_r;
				image[offset].y = sdata_g;
				image[offset].z = sdata_b;
				/*gpu_r[offset] = sdata_r;
				gpu_g[offset] = sdata_g;
				gpu_b[offset] = sdata_b;*/
			}
		}
	}

}


/*
__global__ void avgKernel_2D(unsigned long long int* GPUred, unsigned long long int* GPUgreen, unsigned long long int* GPUblue, unsigned char* GPUavg_r, unsigned char* GPUavg_g, unsigned char* GPUavg_b, const int width, const int height, const size_t pitch, const int c) {
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("blockIdx.x=%d, blockIdx.y=%d, blockDim.x=%d, blockDim.y=%d, threadIdx.x=%d, threadIdx.y=%d, xIndex=%d, yIndex=%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, xIndex, yIndex);

	float output_r = 0.0f;
	float output_g = 0.0f;
	float output_b = 0.0f;

	//Make sure the current thread is inside the image bounds
	if (xIndex < width && yIndex < height) {
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			//Sum the window pixels
			for (int i = 0; i < c; i++) {
				for (int j = 0; j < c; j++) {
				//The tex2D automatically handles Out-Of-Range access.
				output_r += tex2D(texData_r, xIndex + i, yIndex + j);
				output_g += tex2D(texData_g, xIndex + i, yIndex + j);
				output_b += tex2D(texData_b, xIndex + i, yIndex + j);
			}
		}
		atomicAdd(GPUred, output_r);
		atomicAdd(GPUgreen, output_g);
		atomicAdd(GPUblue, output_b);

		output_r /= (c * c);
		output_g /= (c * c);
		output_b /= (c * c);

		//Write the averaged value to the output.
		//Transform 2D index to 1D index, because image is actually in linear memory
		//pitch = blockIdx.x * blockDim.x
		int index = yIndex * pitch + xIndex;

		//printf("blockIdx.x=%d, blockIdx.y=%d, threadIdx.x=%d, threadIdx.y=%d, xIndex=%d, yIndex=%d, pitch=%d, index=%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, xIndex, yIndex, pitch, index);
		GPUavg_r[index] = static_cast<unsigned char>(output_r);
		GPUavg_g[index] = static_cast<unsigned char>(output_g);
		GPUavg_b[index] = static_cast<unsigned char>(output_b);
		}
	}
}
*/


/*
__global__ void fillKernel_2D(unsigned char* GPUoutput_r, unsigned char* GPUoutput_g, unsigned char* GPUoutput_b, unsigned char* GPUavg_r, unsigned char* GPUavg_g, unsigned char* GPUavg_b, const int width, const int height, const size_t pitch) {
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if (xIndex < width && yIndex < height) {
		int index = yIndex * pitch + xIndex;
		int base = (blockIdx.y * blockDim.y + 0) * pitch + (blockIdx.x * blockDim.x + 0);
		GPUoutput_r[index] = GPUavg_r[base];
		GPUoutput_g[index] = GPUavg_g[base];
		GPUoutput_b[index] = GPUavg_b[base];
	}
}
*/


int main(int argc, char *argv[]) {
	int c = 0;
	char *input_file = (char*)malloc(100);
	char *output_file = (char*)malloc(100);
	char *ppm_format = (char*)malloc(15);
	int width = 0, height = 0;
	char *header = (char*)malloc(1024);
	unsigned long long int ave_r = 0, ave_g = 0, ave_b = 0;

	if (process_command_line(argc, argv, &c, input_file, output_file, ppm_format) == FAILURE) {
		return 1;
	}
	printf("[c size] %d\n", c);
	printf("[input filename] %s\n", input_file);
	printf("[output filename] %s\n", output_file);
	printf("[formate] %s\n", ppm_format);

	// read input image file (either binary or plain text PPM)
	if (image_input(input_file, c, &width, &height, header, output_file, ppm_format) == FAILURE) {
		return 1;
	}
	printf("[image width] %d\n", width);
	printf("[image height] %d\n", height);
	printf("\n");

	// allocate cpu memory for storing the results from cuda
	unsigned char* cpu_r = (unsigned char *)malloc(sizeof(unsigned char)*(width)*(height));
	unsigned char* cpu_g = (unsigned char *)malloc(sizeof(unsigned char)*(width)*(height));
	unsigned char* cpu_b = (unsigned char *)malloc(sizeof(unsigned char)*(width)*(height));

	// execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		// CPU mode
		printf("======================================= CPU MODE ========================================\n");
		// calculate the average colour value
		
		cul_average_cpu(c, width, height, &ave_r, &ave_g, &ave_b);
		// Output the average colour value for the image
		printf("CPU Average image colour red = %llu, green = %llu, blue = %llu \n", ave_r, ave_g, ave_b);

		break;
	}
	case (OPENMP): {
		// OPENMP mode
		printf("====================================== OPENMP MODE ======================================\n");
		// calculate the average colour value
		cul_average_openmp(c, width, height, &ave_r, &ave_g, &ave_b);
		// Output the average colour value for the image
		printf("OPENMP Average image colour red = %llu, green = %llu, blue = %llu \n", ave_r, ave_g, ave_b);

		break;
	}
	case (CUDA): {
		// CUDA mode
		printf("======================================= CUDA MODE =======================================\n");
		transform_2D_to_1D(cpu_r, cpu_g, cpu_b, width, height);

		// calculate the average colour value
		launch_cuda_1D(c, width, height, cpu_r, cpu_g, cpu_b, &ave_r, &ave_g, &ave_b);
		//launch_cuda_2D(c, width, height, cpu_r, cpu_g, cpu_b, &ave_r, &ave_g, &ave_b);

		// Output the average colour value for the image
		printf("CUDA Average image colour red = %llu, green = %llu, blue = %llu \n", ave_r, ave_g, ave_b);
		transform_1D_to_2D(cpu_r, cpu_g, cpu_b, width, height);

		break;
	}
	case (ALL): {
		// CPU mode
		printf("======================================= CPU MODE ========================================\n");
		// CPU: calculate the average colour value
		cul_average_cpu(c, width, height, &ave_r, &ave_g, &ave_b);
		// CPU: output the average colour value for the image
		printf("CPU Average image colour red = %llu, green = %llu, blue = %llu \n\n", ave_r, ave_g, ave_b);

		// OPENMP mode
		printf("====================================== OPENMP MODE ======================================\n");
		// OPENMP: calculate the average colour value
		cul_average_openmp(c, width, height, &ave_r, &ave_g, &ave_b);
		// OPENMP: output the average colour value for the image
		printf("OPENMP Average image colour red = %llu, green = %llu, blue = %llu \n\n", ave_r, ave_g, ave_b);

		// CUDA mode
		printf("======================================= CUDA MODE =======================================\n");
		transform_2D_to_1D(cpu_r, cpu_g, cpu_b, width, height);
		// CUDA: calculate the average colour value
		launch_cuda_1D(c, width, height, cpu_r, cpu_g, cpu_b, &ave_r, &ave_g, &ave_b);
		// CUDA: output the average colour value for the image
		printf("CUDA Average image colour red = %llu, green = %llu, blue = %llu \n\n", ave_r, ave_g, ave_b);
		transform_1D_to_2D(cpu_r, cpu_g, cpu_b, width, height);

		break;
	}
	}

	//save the output image file (from last executed mode)
	if (image_output(width, height, ppm_format, header, output_file) == FAILURE) {
		return 1;
	}

	//free memory
	int k;
	for (k = 0; k < height; k++) {
		free(image_r[k]);
		free(image_g[k]);
		free(image_b[k]);
	}

	//Free CPU memory
	free(cpu_r);
	free(cpu_g);
	free(cpu_b);
	free(image_r);
	free(image_g);
	free(image_b);
	free(input_file);
	free(output_file);
	free(ppm_format);
	free(header);

	return 0;
}


void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}


int process_command_line(int argc, char *argv[], int *c, char *input_file, char *output_file, char *ppm_format) {
	// limit the number of argument between 7 and 9
	if (argc != 7 && argc != 9) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	
	// read in the non optional command line arguments
	*c = atoi(argv[1]);
	if (*c < 1) {
		fprintf(stderr, "Error: input c cannot be less than 1...\n");
		return FAILURE;
	}
	// ensure the c is power of 2
	if (is_two_n(*c) != 1) {
		fprintf(stderr, "Error: input c is not a power of 2 number...\n");
		return FAILURE;
	}

	// read in the mode
	if (strcmp("CPU", argv[2]) == 0) execution_mode = CPU;
	else if (strcmp("OPENMP", argv[2]) == 0) execution_mode = OPENMP;
	else if (strcmp("CUDA", argv[2]) == 0) execution_mode = CUDA;
	else if (strcmp("ALL", argv[2]) == 0) execution_mode = ALL;

	// read in the input image name
	if (strcmp("-i", argv[3]) == 0) {
		if ((strstr(argv[4], ".ppm") == NULL) && (strstr(argv[4], ".PPM") == NULL)) {
			fprintf(stderr, "Error: input file shoud be a ppm image...\n");
			return FAILURE;
		}
		strcpy(input_file, argv[4]);
	}
	else {
		fprintf(stderr, "Error: Wrong program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	// read in the output image name
	if (strcmp("-o", argv[5]) == 0) {
		if ((strstr(argv[6], ".ppm") == NULL) && (strstr(argv[6], ".PPM") == NULL)) {
			fprintf(stderr, "Error: output file shoud be a ppm image...\n");
			return FAILURE;
		}
		strcpy(output_file, argv[6]);
	}
	else {
		fprintf(stderr, "Error: Wrong program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	// read in any optional part 3 arguments
	// the defult output format is PPM_BINARY
	if (argc == 9) {
		if (strcmp("-f", argv[7]) == 0) {
			strcpy(ppm_format, argv[8]);
		}
		else {
			fprintf(stderr, "Error: Wrong program arguments. Correct usage is...\n");
			print_help();
			return FAILURE;
		}
	}
	else {
		strcpy(ppm_format, "PPM_BINARY");
	}

	return SUCCESS;
}


int is_two_n(int num) {
	if ((num&(num - 1))) {
		return -1;
	}
	return 1;
}


int image_input(char *input_file, int c, int *width, int *height, char *header, char *output_file, char *ppm_format) {
	// open file
	FILE *fp = fopen(input_file, "rb");
	// the input file does not exist
	if (fp == NULL) {
		fprintf(stderr, "Error: Can't find the input file...\n");
		return FAILURE;
	}

	char *format = (char*)malloc(3);
	if (read_header(fp, c, width, height, header, output_file, ppm_format, format) == FAILURE) {
		fprintf(stderr, "Error: Can't read header...\n");
		return FAILURE;
	}

	// initialise two dimensions dynamic int array
	// allocate memory space to first dimensions
	image_r = (unsigned char **)malloc(sizeof(unsigned char *)*(*height));
	image_g = (unsigned char **)malloc(sizeof(unsigned char *)*(*height));
	image_b = (unsigned char **)malloc(sizeof(unsigned char *)*(*height));
	int k;
	// allocate memory space to second dimensions of each first dimension
	for (k = 0; k < *height; k++) {
		*(image_r + k) = (unsigned char *)malloc(sizeof(unsigned char)*(*width));
		*(image_g + k) = (unsigned char *)malloc(sizeof(unsigned char)*(*width));
		*(image_b + k) = (unsigned char *)malloc(sizeof(unsigned char)*(*width));
	}

	// allocate momory 
	unsigned char *all_input = (unsigned char *)malloc(sizeof(unsigned char)*(*width)*(*height) * 3);
	// read content infomation
	// PPM_PLAIN_TEXT format
	if (strcmp(format, "P3") == 0) {
		unsigned char buf;
		int count = 0;
		while (fscanf(fp, "%hhu", &buf) == 1) {
			all_input[count] = buf;
			count++;
		}
	}
	// PPM_BINARY format
	else if (strcmp(format, "P6") == 0) {
		// read all the binary content
		fread(all_input, sizeof(unsigned char), 3 * (*width) * (*height), fp);
	}

	int i = 0, h = -1, w = 0;;
	while (i < (*width)*(*height) * 3) {
		if (i % (*width * 3) == 0) {
			h++;
			w = 0;
		}
		// red
		if (i % 3 == 0) {
			image_r[h][w] = all_input[i];
			//printf("%d ", image_r[h][w]);
		}
		// green
		else if (i % 3 == 1) {
			image_g[h][w] = all_input[i];
			//printf("%d ", image_g[h][w]);
		}
		// blue
		else {
			image_b[h][w] = all_input[i];
			//printf("%d ", image_b[h][w]);
			w++;
		}
		i++;
	}
	// close file
	fclose(fp);
	free(all_input);
	free(format);

	return SUCCESS;
}


int read_header(FILE *fp, int c, int *width, int *height, char *header, char *output_file, char *ppm_format, char *format) {
	char input[1024] = "";

	if (strcmp(ppm_format, "PPM_PLAIN_TEXT") == 0) {
		strcpy(header, "P3\n");
	}
	else if (strcmp(ppm_format, "PPM_BINARY") == 0) {
		strcpy(header, "P6\n");
	}
	strcat(header, "# COM6521 Assignment2 - ");
	strcat(header, output_file);
	strcat(header, "\n");

	// read header infomation
	while (1) {
		// exit if reading to the end of file
		if (fgets(input, sizeof(input), fp) == NULL) {
			return FAILURE;
		}
		// exit if reading to the end line of header
		if (strncmp(input, "255", 3) == 0) {
			strcat(header, input);
			break;
		}
		// file format (either P3 or P6)
		if (strncmp(input, "P3", 2) == 0) {
			strcpy(format, "P3");
		}
		else if (strncmp(input, "P6", 2) == 0) {
			strcpy(format, "P6");
		}
		// skip if reading to command line
		else if (strncmp(input, "#", 1) == 0) {
			continue;
		}
		// first number is file width and sencond one is height
		else {
			strcat(header, input);
			char * ptr = strchr(input, ' ');
			if (ptr != NULL) {
				*height = atoi(ptr);
			}
			// width is not assigned
			if (*width == 0) {
				*width = atoi(input);
			}
			else {
				*height = atoi(input);
			}
		}
	}

	// limit c should be less than width and height 
	if (c > *width || c > *height) {
		fprintf(stderr, "Error: input c is greater than width or height...\n");
		return FAILURE;
	}
	return SUCCESS;
}


void cul_average_cpu(int c, int width, int height, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b) {
	clock_t begin, end;
	float mseconds;

	// starting timing here
	begin = clock();

	// initialise the results
	unsigned long long int red = 0, green = 0, blue = 0;
	*ave_r = 0, *ave_g = 0, *ave_b = 0;
	int i = 0, j = 0, k = 0, l = 0;
	for (i = 0; i < height; i += c) {
		for (j = 0; j < width; j += c) {
			unsigned long long int sum_r = 0, sum_g = 0, sum_b = 0;
			unsigned long long int count = 0;
			// sum the values in a cell
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					count++;
					sum_r += image_r[k][l];
					sum_g += image_g[k][l];
					sum_b += image_b[k][l];
				}
			}
			// replace the origin values by the cell average
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					image_r[k][l] = (unsigned char)(sum_r / count);
					image_g[k][l] = (unsigned char)(sum_g / count);
					image_b[k][l] = (unsigned char)(sum_b / count);
				}
			}
			red += sum_r;
			green += sum_g;
			blue += sum_b;
			
		}
	}
	*ave_r = red / (height * width);
	*ave_g = green / (height * width);
	*ave_b = blue / (height * width);

	// end timing here
	end = clock();
	mseconds = (end - begin) * 1000 / (float)CLOCKS_PER_SEC;
	printf("CPU mode execution time took %d s and %d ms\n", (int)mseconds / 1000, (int)mseconds % 1000);
}


void cul_average_openmp(int c, int width, int height, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b) {
	clock_t begin, end;
	float mseconds;

	// starting timing here
	begin = clock();

	// initialise the results
	unsigned long long int red = 0, green = 0, blue = 0;
	int i;
#pragma omp parallel for
	for (i = 0; i < height; i += c) {
		int j = 0;
		//#pragma omp parallel for
#pragma omp parallel for reduction(+: red, green, blue)
		for (j = 0; j < width; j += c) {
			int k, l;
			unsigned long long int sum_r = 0, sum_g = 0, sum_b = 0, count = 0;
			// sum the values in a cell
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					count++;
					sum_r += image_r[k][l];
					sum_g += image_g[k][l];
					sum_b += image_b[k][l];
				}
			}
			// replace the origin values by the cell average
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					image_r[k][l] = (unsigned char)(sum_r / count);
					image_g[k][l] = (unsigned char)(sum_g / count);
					image_b[k][l] = (unsigned char)(sum_b / count);
				}
			}
			//#pragma omp critical
			//{
			red += sum_r;
			green += sum_g;
			blue += sum_b;
			//}
		}
	}
	*ave_r = red / (height * width);
	*ave_g = green / (height * width);
	*ave_b = blue / (height * width);

	// end timing here
	end = clock();
	mseconds = (end - begin) * 1000 / (float)CLOCKS_PER_SEC;
	printf("OPENMP mode execution time took %d s and %d ms\n", (int)mseconds / 1000, (int)mseconds % 1000);
}


void launch_cuda_1D(int c, int width, int height, unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b) {
	cudaEvent_t start, stop;
	cudaEvent_t k_start, k_stop;
	float ms, mseconds;

	uchar3 *d_image;
	uchar3 *h_image;
	h_image = (uchar3*)malloc(sizeof(uchar3)*(width)*(height));
	for (int i = 0; i < width*height; i++) {
		h_image[i].x = cpu_r[i];
		h_image[i].y = cpu_g[i];
		h_image[i].z = cpu_b[i];
	}

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&k_start);
	cudaEventCreate(&k_stop);

	// starting timing here
	cudaEventRecord(start, 0);

	// initalise the values of rgb average
	*ave_r = 0, *ave_g = 0, *ave_b = 0;

	//Declare GPU pointer
	//unsigned char *gpu_r, *gpu_g, *gpu_b;

	// allocate memory on the GPU
	cudaMalloc((void**)&d_image, sizeof(uchar3)*(width)*(height));
	/*cudaMalloc((void**)&gpu_r, sizeof(unsigned char)*(width)*(height));
	cudaMalloc((void**)&gpu_g, sizeof(unsigned char)*(width)*(height));
	cudaMalloc((void**)&gpu_b, sizeof(unsigned char)*(width)*(height));*/
	checkCUDAError("CUDA malloc");

	// For device variables
	cudaMemcpyToSymbol(average_r, ave_r, sizeof(unsigned long long int));
	cudaMemcpyToSymbol(average_g, ave_g, sizeof(unsigned long long int));
	cudaMemcpyToSymbol(average_b, ave_b, sizeof(unsigned long long int));

	// transfer memory from the host to device
	cudaMemcpy(d_image, h_image, sizeof(uchar3)*(width)*(height), cudaMemcpyHostToDevice);
	/*cudaMemcpy(gpu_r, cpu_r, sizeof(unsigned char)*(width)*(height), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_g, cpu_g, sizeof(unsigned char)*(width)*(height), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, sizeof(unsigned char)*(width)*(height), cudaMemcpyHostToDevice);*/
	checkCUDAError("CUDA memcpy to device");

	int block_x = (width % c == 0) ? width / c : width / c + 1;
	int block_y = (height % c == 0) ? height / c : height / c + 1;
	int thread = (c > 1024) ? 1024 : c;
	//cuda layout and execution
	dim3 threadsPerBlock(thread, 1, 1);
	dim3 blocksPerGrid(block_x, block_y, 1);

	cudaEventRecord(k_start, 0);
	// lauch kernel
	//avgKernel_1D << <blocksPerGrid, threadsPerBlock >> > (gpu_r, gpu_g, gpu_b, width, height, c);
	avgKernel_1D << <blocksPerGrid, threadsPerBlock >> > (d_image, width, height, c);
	
	cudaEventRecord(k_stop, 0);
	cudaEventSynchronize(k_stop);
	cudaEventElapsedTime(&ms, k_start, k_stop);
	printf("CUDA mode execution time only for kernel took %f ms\n", ms);
	

	cudaMemcpyFromSymbol(ave_r, average_r, sizeof(unsigned long long int));
	cudaMemcpyFromSymbol(ave_g, average_g, sizeof(unsigned long long int));
	cudaMemcpyFromSymbol(ave_b, average_b, sizeof(unsigned long long int));

	// transfer memory from the device to device
	cudaMemcpy(h_image, d_image, sizeof(uchar3)*(width)*(height), cudaMemcpyDeviceToHost);
	/*cudaMemcpy(cpu_r, gpu_r, sizeof(unsigned char)*(width)*(height), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_g, gpu_g, sizeof(unsigned char)*(width)*(height), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_b, gpu_b, sizeof(unsigned char)*(width)*(height), cudaMemcpyDeviceToHost);*/
	checkCUDAError("CUDA memcpy from device");

	*ave_r /= width * height;
	*ave_g /= width * height;
	*ave_b /= width * height;

	// end timing here
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mseconds, start, stop);
	checkCUDAError("timmer");
	printf("CUDA mode execution time took %d s and %f ms\n", (int)mseconds / 1000, mseconds);

	for (int i = 0; i < width*height; i++) {
		cpu_r[i] = h_image[i].x;
		cpu_g[i] = h_image[i].y;
		cpu_b[i] = h_image[i].z;
	}

	// release GPU memory
	cudaFree(d_image);
	/*cudaFree(gpu_r);
	cudaFree(gpu_g);
	cudaFree(gpu_b);*/

	free(h_image);

	// cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
}


/*
void launch_cuda_2D(int c, int width, int height, unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, unsigned long long int *ave_r, unsigned long long int *ave_g, unsigned long long int *ave_b) {
	cudaEvent_t start, stop;
	float mseconds;

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// starting timing here
	cudaEventRecord(start, 0);

	//Declare GPU pointer
	unsigned char *GPUinput_r, *GPUinput_g, *GPUinput_b, *GPUavg_r, *GPUavg_g, *GPUavg_b, *GPUoutput_r, *GPUoutput_g, *GPUoutput_b;
	unsigned long long int CPUred = 0, CPUgreen = 0, CPUblue = 0, *GPUred, *GPUgreen, *GPUblue;


	//Allocate 2D memory on GPU. Also known as Pitch Linear Memory
	size_t gpu_image_pitch = 0;
	cudaMalloc((void**)&GPUred, sizeof(int));
	cudaMalloc((void**)&GPUgreen, sizeof(int));
	cudaMalloc((void**)&GPUblue, sizeof(int));
	cudaMallocPitch<unsigned char>(&GPUinput_r, &gpu_image_pitch, width * sizeof(unsigned char), height);
	cudaMallocPitch<unsigned char>(&GPUinput_g, &gpu_image_pitch, width * sizeof(unsigned char), height);
	cudaMallocPitch<unsigned char>(&GPUinput_b, &gpu_image_pitch, width * sizeof(unsigned char), height);

	cudaMallocPitch<unsigned char>(&GPUavg_r, &gpu_image_pitch, width * sizeof(unsigned long long int), height);
	cudaMallocPitch<unsigned char>(&GPUavg_g, &gpu_image_pitch, width * sizeof(unsigned long long int), height);
	cudaMallocPitch<unsigned char>(&GPUavg_b, &gpu_image_pitch, width * sizeof(unsigned long long int), height);

	cudaMallocPitch<unsigned char>(&GPUoutput_r, &gpu_image_pitch, width * sizeof(unsigned char), height);
	cudaMallocPitch<unsigned char>(&GPUoutput_g, &gpu_image_pitch, width * sizeof(unsigned char), height);
	cudaMallocPitch<unsigned char>(&GPUoutput_b, &gpu_image_pitch, width * sizeof(unsigned char), height);

	//Copy data from host to device.
	cudaMemcpy(GPUred, &CPUred, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUgreen, &CPUgreen, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUblue, &CPUblue, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaMemcpy2D(GPUinput_r, gpu_image_pitch, cpu_r, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(GPUinput_g, gpu_image_pitch, cpu_g, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(GPUinput_b, gpu_image_pitch, cpu_b, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

	//Bind the image to the texture. Now the kernel will read the input image through the texture cache.
	//Use tex2D function to read the image
	cudaBindTexture2D(NULL, texData_r, GPUinput_r, width * sizeof(unsigned char), height, gpu_image_pitch);
	cudaBindTexture2D(NULL, texData_g, GPUinput_g, width * sizeof(unsigned char), height, gpu_image_pitch);
	cudaBindTexture2D(NULL, texData_b, GPUinput_b, width * sizeof(unsigned char), height, gpu_image_pitch);

	// Set the behavior of tex2D for out-of-range image reads.
	texData_r.addressMode[0] = texData_r.addressMode[1] = cudaAddressModeBorder;
	texData_g.addressMode[0] = texData_g.addressMode[1] = cudaAddressModeBorder;
	texData_b.addressMode[0] = texData_b.addressMode[1] = cudaAddressModeBorder;

	dim3 threadsPerBlock(c, c, 1);
	dim3 blocksPerGrid;
	blocksPerGrid.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x;  //< Greater than or equal to image width
	blocksPerGrid.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y; //< Greater than or equal to image height

	//Launch the kernel
	avgKernel_2D << <blocksPerGrid, threadsPerBlock >> >(GPUred, GPUgreen, GPUblue, GPUavg_r, GPUavg_g, GPUavg_b, width, height, gpu_image_pitch, c);
	fillKernel_2D << <blocksPerGrid, threadsPerBlock >> >(GPUoutput_r, GPUoutput_g, GPUoutput_b, GPUavg_r, GPUavg_g, GPUavg_b, width, height, gpu_image_pitch);

	//Copy the results back to CPU
	cudaMemcpy(&CPUred, GPUred, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&CPUgreen, GPUgreen, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&CPUblue, GPUblue, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy2D(cpu_r, width, GPUoutput_r, gpu_image_pitch, width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(cpu_g, width, GPUoutput_g, gpu_image_pitch, width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(cpu_b, width, GPUoutput_b, gpu_image_pitch, width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

	*ave_r = CPUred / (width* height);
	*ave_g = CPUgreen / (width* height);
	*ave_b = CPUblue / (width* height);

	//Release the texture
	cudaUnbindTexture(texData_r);
	cudaUnbindTexture(texData_g);
	cudaUnbindTexture(texData_b);

	//Free GPU memory
	cudaFree(GPUinput_r);
	cudaFree(GPUinput_g);
	cudaFree(GPUinput_b);
	cudaFree(GPUoutput_r);
	cudaFree(GPUoutput_g);
	cudaFree(GPUoutput_b);
	cudaFree(GPUavg_r);
	cudaFree(GPUavg_g);
	cudaFree(GPUavg_b);
	cudaFree(GPUred);
	cudaFree(GPUgreen);
	cudaFree(GPUblue);

	// end timing here
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mseconds, start, stop);
	checkCUDAError("timmer");
	printf("CUDA mode execution time took 0 s and %f ms\n", mseconds);
}*/



void transform_2D_to_1D(unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, int width, int height) {
	int count = 0;
	// transfer 2D array to 1D
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cpu_r[count] = image_r[i][j];
			cpu_g[count] = image_g[i][j];
			cpu_b[count] = image_b[i][j];
			count++;
		}
	}
}


void transform_1D_to_2D(unsigned char* cpu_r, unsigned char* cpu_g, unsigned char* cpu_b, int width, int height) {
	long long count = 0;
	// transfer 1D array to 2D
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			image_r[i][j] = cpu_r[count];
			image_g[i][j] = cpu_g[count];
			image_b[i][j] = cpu_b[count];
			count++;
		}
	}
}


int image_output(int width, int height, char *ppm_format, char *header, char *output_file) {
	if (strcmp(ppm_format, "PPM_BINARY") == 0) {
		// open file
		FILE *fp = fopen(output_file, "wb");
		// the output file does not exist
		if (fp == NULL) {
			fprintf(stderr, "Error: Can't find the output file...\n");
			return FAILURE;
		}
		// write header information
		fprintf(fp, "%s", header);
		unsigned char *all_output = (unsigned char *)malloc(sizeof(unsigned char)*width*height * 3);
		int i, j, k = 0;
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				// red
				all_output[k] = image_r[i][j];
				k++;
				// green
				all_output[k] = image_g[i][j];
				k++;
				// blue
				all_output[k] = image_b[i][j];
				k++;
			}
		}
		// write all information
		fwrite(all_output, sizeof(unsigned char), 3 * width*height, fp);
		// close file
		fclose(fp);
		free(all_output);
	}
	else if (strcmp(ppm_format, "PPM_PLAIN_TEXT") == 0) {
		// open file
		FILE *fp = fopen(output_file, "w");
		// the output file does not exist
		if (fp == NULL) {
			fprintf(stderr, "Error: Can't find the output file...\n");
			return FAILURE;
		}
		// write header information
		fputs(header, fp);
		int i, j;
		char out_string[4];
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				// red
				sprintf(out_string, "%u", image_r[i][j]);
				fputs(out_string, fp);
				fputc(' ', fp);
				// green
				sprintf(out_string, "%u", image_g[i][j]);
				fputs(out_string, fp);
				fputc(' ', fp);
				// blue
				sprintf(out_string, "%u", image_b[i][j]);
				fputs(out_string, fp);

				if (j == (width - 1)) continue;
				fputc('\t', fp);
			}
			// move to new line
			if (i == (height - 1)) continue;
			fputc('\n', fp);
		}
		// close file
		fclose(fp);
	}

	return SUCCESS;
}


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}