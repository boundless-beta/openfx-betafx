// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif
#include "CLFuncs.h"
#include "BetaFXCommon.h"

static std::map<int, std::string> kernelStrings;
bool buffersCreated = false;
int bufferSize = 0;
cl_mem buffer0;
cl_mem buffer1;
cl_mem buffer2;
cl_mem buffer3;
cl_mem tBuffersCL;

class Locker
{
public:
    Locker()
    {
#ifdef _WIN64
        InitializeCriticalSection(&mutex);
#else
        pthread_mutex_init(&mutex, NULL);
#endif
    }

    ~Locker()
    {
#ifdef _WIN64
        DeleteCriticalSection(&mutex);
#else
        pthread_mutex_destroy(&mutex);
#endif
    }

    void Lock()
    {
#ifdef _WIN64
        EnterCriticalSection(&mutex);
#else
        pthread_mutex_lock(&mutex);
#endif
    }

    void Unlock()
    {
#ifdef _WIN64
        LeaveCriticalSection(&mutex);
#else
        pthread_mutex_unlock(&mutex);
#endif
    }

private:
#ifdef _WIN64
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t mutex;
#endif
};

const char* fallbackKernel = "kReadIndex = (float2)(x, y); ";
cl_context clContext = NULL;



static std::string kReadBuffer(int index) {
    std::string buff = "";
    switch (index) {
    case 0:
        buff = "kBuffer0";
        break;
    case 1:
        buff = "kBuffer1";
        break;
    case 2:
        buff = "kBuffer2";
        break;
    case 3:
        buff = "kBuffer3";
        break;
    default:
        buff = "p_Input";
        break;
    }
    // if (index > -1) {
    //     return "kOutput = read_imagef(" + buff + ", imageSampler, kReadIndex.yx + (float2)0.5); \n";
    // }
    // else {
        return "\n" \
            "rIndex = ((int)kReadIndex.x + (int)kReadIndex.y * p_Width) * 4;         \n" \
            "if(rIndex >= 0 && rIndex < p_Width * p_Height * 4) {         \n" \
            "kOutput.x = " + buff + "[rIndex + 0]; \n" \
            "kOutput.y = " + buff + "[rIndex + 1];\n" \
            "kOutput.z = " + buff + "[rIndex + 2];\n" \
            "kOutput.w = " + buff + "[rIndex + 3];\n" \
            "} else {\n" \
            "kOutput = (float4)0; \n" \
                "}\n";
    // }
}

static std::string kWriteBuffer(int index) {
    std::string buff = "";
    switch (index) {
    case 0:
        buff = "kBuffer0";
        break;
    case 1:
        buff = "kBuffer1";
        break;
    case 2:
        buff = "kBuffer2";
        break;
    case 3:
        buff = "kBuffer3";
        break;
    default:
        buff = "p_Output";
        break;
    }
    return "\n" \
        + buff + "[index + 0] = kOutput.x; \n" \
        + buff + "[index + 1] = kOutput.y;\n" \
        + buff + "[index + 2] = kOutput.z;\n" \
        + buff + "[index + 3] = kOutput.w;\n";
    // return "write_imagef(" + buff + ", (int2)(x, y), kOutput); \n";
}
// H005666
// H005666
static inline cl_mem bufferQuery(cl_context clContext, cl_command_queue cmdQ, size_t bufferSize, cl_mem_flags flags, int index)
{
    static std::map<int, cl_mem> bufferIO;
    cl_mem theBuffer;
    std::map<int, cl_mem>::iterator iter = bufferIO.find(index);
    if (iter == bufferIO.end())
    {
        // create new buffer
        theBuffer = clCreateBuffer(clContext, flags, bufferSize, NULL, NULL);
        float zero = 0.;
        clEnqueueFillBuffer(cmdQ, theBuffer, &zero, sizeof(float), 0, bufferSize, 0, NULL, NULL);
        bufferIO[index] = theBuffer;
    }
    else { //buffer of differing size exists
        size_t currentSize;
        clGetMemObjectInfo(iter->second, CL_MEM_SIZE, sizeof(size_t), &currentSize, NULL);
        if (currentSize != bufferSize) {
            // update existing buffer
            theBuffer = clCreateBuffer(clContext, flags, bufferSize, NULL, NULL);
            bufferIO[index] = theBuffer;
        }
        else
        {
            // find existing buffer
            theBuffer = iter->second;
        }
    }
    return theBuffer;
}

template<class PIX>
void RunOpenCLKernelBuffers(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const PIX* p_Input, PIX* p_Output)
{
    cl_int error;

    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    // store device id and kernel per command queue (required for multi-GPU systems)
    static std::map<cl_command_queue, cl_device_id> deviceIdMap;
    static std::map<int, cl_kernel> kernelMap;

    static Locker locker; // simple lock to control access to the above maps from multiple threads

    locker.Lock();

    // find the device id corresponding to the command queue
    cl_device_id deviceId = NULL;
    if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
    {
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
        CheckError(error, "Unable to get the device");

        deviceIdMap[cmdQ] = deviceId;
    }
    else
    {
        deviceId = deviceIdMap[cmdQ];
    }

    // find the program kernel corresponding to the command queue
    cl_kernel clKernel = NULL;
    if (kernelStrings[instance] != kernel)
    {
        if (kernelMap.find(instance) != kernelMap.end())
        {
            error = clReleaseKernel(kernelMap[instance]);
        }
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
        CheckError(error, "Unable to get the context");

        std::string bitStr = bits == 8 ? "uchar" : "float";

        std::string kernelStart = "__kernel void customKernel(int p_Width, int p_Height, float kTime, float kFloat0, float kFloat1, float kFloat2, float kFloat3, float kFloat4, float kFloat5, float kFloat6, float kFloat7, float kFloat8, float kFloat9, float kFloat10, float kFloat11, float kFloat12, float kFloat13, float kFloat14, float kFloat15, __global " + bitStr + "* p_Input, __global " + bitStr + "* p_Output, __global float* kBuffer0, __global float* kBuffer1, __global float* kBuffer2, __global float* kBuffer3, __global float* kTransform, int kThread, int bits)\n" \
            "{                                                                      \n" \
            "   const int x = get_global_id(0);                                     \n" \
            "   const int y = get_global_id(1);                                     \n" \
            "   const int index = ((y * p_Width) + x) * 4;         \n" \
            "   int rIndex; \n" \
            "   float4 kOutput(0,0,0,0);                                           \n" \
            "   float2 kReadIndex(0,0);                                           \n" \
            "   if ((x < p_Width) && (y < p_Height))                                \n" \
            "   {                                                                   \n";
        std::string kernelEnd ="                                                                       \n" \
            "				if (bits == 8) {                                                                                                     \n"\
            "					kOutput.x = fmin(kOutput.x, 255.0);                                                                                 \n"\
            "					kOutput.y = fmin(kOutput.y, 255.0);                                                                                 \n"\
            "					kOutput.z = fmin(kOutput.z, 255.0);                                                                                 \n"\
            "					kOutput.w = fmin(kOutput.w, 255.0);                                                                                 \n"\
            "				}                                                                                 \n"\
            "       p_Output[index + 0] = kOutput.x;             \n" \
            "       p_Output[index + 1] = kOutput.y;             \n" \
            "       p_Output[index + 2] = kOutput.z;             \n" \
            "       p_Output[index + 3] = kOutput.w;             \n" \
            "   }                                                                   \n" \
            "}                                                                      \n";


        std::string::size_type kReadPos = kernel.find("kRead(");
        std::string::size_type kWritePos = kernel.find("kWrite(");
        std::string kernThing = kernel;
        while (kReadPos != std::string::npos) {
            const std::string kBx = kernThing.substr(kReadPos + 6, 1);
            std::string kReadB = "";
            if (kBx == "1" || kBx == "2" || kBx == "3" || kBx == "0") {
                kReadB = kReadBuffer(std::stoi(kBx));
                kernThing.replace(kReadPos, 8, kReadB);
            }
            else if (kBx == ")") {
                kReadB = kReadBuffer(-1);
                kernThing.replace(kReadPos, 8, kReadB);
            }
            kReadPos = kernThing.find("kRead(", kReadPos + 1);
        }
        while (kWritePos != std::string::npos) {
            const std::string kBx = kernel.substr(kWritePos + 7, 1);
            if (kBx == "1" || kBx == "2" || kBx == "3" || kBx == "0") {
                std::string kWriteB = kWriteBuffer(std::stoi(kBx));
                kernThing.replace(kWritePos, 9, kWriteB);
            }
            kWritePos = kernel.find("kWrite(", kWritePos + 1);
        }
        kernThing = kernelStart + kernThing + kernelEnd;

        char* kernStr = new char[kernThing.length() + 1];
        strcpy(kernStr, kernThing.c_str());
        cl_program program = clCreateProgramWithSource(clContext, 1, (const char**)&kernStr, NULL, &error);
        CheckError(error, "Unable to create program");
        if (error != CL_SUCCESS) {
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        char errorInfo[65536];
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
            kernThing = kernelStart + fallbackKernel + kReadBuffer(-1) + kernelEnd;

            char* kernFallback = new char[kernThing.length() + 1];
            strcpy(kernFallback, kernThing.c_str());
            program = clCreateProgramWithSource(clContext, 1, (const char**)&kernFallback, NULL, &error);
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        }
        CheckError(error, "Unable to build program");

        clKernel = clCreateKernel(program, "customKernel", &error);
        CheckError(error, "Unable to create kernel");

        kernelMap[instance] = clKernel;
        clReleaseProgram(program);
    }
    else if (kernelMap.find(instance) != kernelMap.end())
    {
        clKernel = kernelMap[instance];
    }
    kernelStrings[instance] = kernel;
    locker.Unlock();
    if (!buffersCreated) {
        tBuffersCL = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 64 * 25 * 16 * sizeof(float), &buffers, NULL);
    }
    else {
        clEnqueueWriteBuffer(cmdQ, tBuffersCL, CL_TRUE, 0, 64 * 25 * 16 * sizeof(float), &buffers, 0, NULL, NULL);
    }
    if (bufferSize != p_Width * p_Height * 4 * sizeof(float) || !buffersCreated) {
        bufferSize = p_Width * p_Height * 4 * sizeof(float);
        buffer0 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 0);
        buffer1 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 1);
        buffer2 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 2);
        buffer3 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 3);
        buffersCreated = true;
    }
    int count = 0;
    error = clSetKernelArg(clKernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(clKernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &timeIn);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[0]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[1]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[2]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[3]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[4]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[5]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[6]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[7]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[8]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[9]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[10]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[11]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[12]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[13]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[14]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[15]);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &p_Output);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer0);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer1);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer2);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer3);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &tBuffersCL);
    error |= clSetKernelArg(clKernel, count++, sizeof(int), &transformIndex);
    error |= clSetKernelArg(clKernel, count++, sizeof(int), &bits);

    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(clKernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
template<class PIX>
void RunOpenCLKernelImages(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const PIX* p_Input, PIX* p_Output)
{
    cl_int error;

    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    // store device id and kernel per command queue (required for multi-GPU systems)
    static std::map<cl_command_queue, cl_device_id> deviceIdMap;
    static std::map<int, cl_kernel> kernelMap;

    static Locker locker; // simple lock to control access to the above maps from multiple threads

    locker.Lock();

    // find the device id corresponding to the command queue
    cl_device_id deviceId = NULL;
    if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
    {
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
        CheckError(error, "Unable to get the device");

        deviceIdMap[cmdQ] = deviceId;
    }
    else
    {
        deviceId = deviceIdMap[cmdQ];
    }

    char deviceInfo[65536];
    error = clGetDeviceInfo(deviceId, CL_DEVICE_OPENCL_C_VERSION, sizeof(char) * 65536, &deviceInfo, NULL);

    // find the program kernel corresponding to the command queue
    cl_kernel clKernel = NULL;
    if (kernelStrings[instance] != kernel)
    {
        if (kernelMap.find(instance) != kernelMap.end())
        {
            error = clReleaseKernel(kernelMap[instance]);
        }
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
        CheckError(error, "Unable to get the context");

        std::string bitStr = bits == 8 ? "uchar" : "float";

        std::string kernelStart = "__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;  \n" \
            "__kernel void customKernel(int p_Width, int p_Height, float kTime, float kFloat0, float kFloat1, float kFloat2, float kFloat3, float kFloat4, float kFloat5, float kFloat6, float kFloat7, float kFloat8, float kFloat9, float kFloat10, float kFloat11, float kFloat12, float kFloat13, float kFloat14, float kFloat15, __read_only image2d_t p_Input, __write_only image2d_t p_Output, __global float* kBuffer0, __global float* kBuffer1, __global float* kBuffer2, __global float* kBuffer3, __global float* kTransform, int kThread, int bits)\n" \
            "{                                                                      \n" \
            "   const int x = get_global_id(0);                                     \n" \
            "   const int y = get_global_id(1);                                     \n" \
            "   const int index = ((y * p_Width) + x) * 4;         \n" \
            "   int rIndex; \n" \
            "   float4 kOutput = (float4)(0,0,0,0);                                           \n" \
            "   float2 kReadIndex = (float2)(0,0);                                           \n" \
            "   if ((x < p_Width) && (y < p_Height))                                \n" \
            "   {                                                                   \n";
        std::string kernelEnd ="write_imagef(p_Output, (int2)(x, y), kOutput);                      \n"\
            "}                                                                   \n" \
            "}                                                                      \n";
        std::string::size_type kReadPos = kernel.find("kRead(");
        std::string::size_type kWritePos = kernel.find("kWrite(");
        std::string kernThing = kernel;


        while (kReadPos != std::string::npos) {
            const std::string kBx = kernThing.substr(kReadPos + 6, 1);
            std::string kReadB = "";
            if (kBx == "1" || kBx == "2" || kBx == "3" || kBx == "0") {
                kReadB = kReadBuffer(std::stoi(kBx));
                kernThing.replace(kReadPos, 8, kReadB);
            }
            else if (kBx == ")") {
                kReadB = "kOutput = read_imagef(p_Input, imageSampler, kReadIndex + (float2)0.5);\n";
                kernThing.replace(kReadPos, 8, kReadB);
            }
            kReadPos = kernThing.find("kRead(", kReadPos + 1);
        }


        while (kWritePos != std::string::npos) {
            const std::string kBx = kernThing.substr(kWritePos + 7, 1);
            if (kBx == "1" || kBx == "2" || kBx == "3" || kBx == "0") {
                std::string kWriteB = kWriteBuffer(std::stoi(kBx));
                kernThing.replace(kWritePos, 9, kWriteB);
            }
            kWritePos = kernThing.find("kWrite(", kWritePos + 1);
        }


        kernThing = kernelStart + kernThing + kernelEnd;

        char* kernStr = new char[kernThing.length() + 1];
        strcpy(kernStr, kernThing.c_str());
        cl_program program = clCreateProgramWithSource(clContext, 1, (const char**)&kernStr, NULL, &error);
        CheckError(error, "Unable to create program");
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        char errorInfo[65536];
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
        if (error != CL_SUCCESS) {
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        char errorInfo[65536];
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
            kernThing = kernelStart + fallbackKernel + kReadBuffer(-1) + kernelEnd;

            char* kernFallback = new char[kernThing.length() + 1];
            strcpy(kernFallback, kernThing.c_str());
            program = clCreateProgramWithSource(clContext, 1, (const char**)&kernFallback, NULL, &error);
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        }

        CheckError(error, "Unable to build program");

        clKernel = clCreateKernel(program, "customKernel", &error);
        CheckError(error, "Unable to create kernel");

        kernelMap[instance] = clKernel;
        clReleaseProgram(program);
        kernelStrings[instance] = kernel;
    }
    else if (kernelMap.find(instance) != kernelMap.end())
    {
        clKernel = kernelMap[instance];
    }
    locker.Unlock();
    if (!buffersCreated) {
        tBuffersCL = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 64 * 25 * 16 * sizeof(float), &buffers, NULL);
    }
    else {
        clEnqueueWriteBuffer(cmdQ, tBuffersCL, CL_TRUE, 0, 64 * 25 * 16 * sizeof(float), &buffers, 0, NULL, NULL);
    }
    cl_image_format form;
    form.image_channel_order = CL_RGBA;
    form.image_channel_data_type = CL_FLOAT;
    if (bufferSize != p_Width * p_Height * 4 * sizeof(float) || !buffersCreated) {
        if (buffersCreated) {
            clReleaseMemObject(buffer0);
            clReleaseMemObject(buffer1);
            clReleaseMemObject(buffer2);
            clReleaseMemObject(buffer3);
        }
        bufferSize = p_Width * p_Height * 4 * sizeof(float);
        buffer0 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 0);
        buffer1 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 1);
        buffer2 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 2);
        buffer3 = bufferQuery(clContext, cmdQ, bufferSize, CL_MEM_READ_WRITE, 3);
        buffersCreated = true;
    }

    int count = 0;
    error = clSetKernelArg(clKernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(clKernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &timeIn);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[0]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[1]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[2]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[3]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[4]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[5]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[6]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[7]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[8]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[9]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[10]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[11]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[12]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[13]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[14]);
    error |= clSetKernelArg(clKernel, count++, sizeof(float), &floats[15]);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &p_Output);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer0);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer1);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer2);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &buffer3);
    error |= clSetKernelArg(clKernel, count++, sizeof(cl_mem), &tBuffersCL);
    error |= clSetKernelArg(clKernel, count++, sizeof(int), &transformIndex);
    error |= clSetKernelArg(clKernel, count++, sizeof(int), &bits);

    //float mb, float xScale, float yScale, float zScale, float xRot, float yRot, float zRot, int pIndex, int pSend, int now, int off, int forward, __read_only image2d_t p_Input, __write_only image2d_t p_Output, __global 
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(clKernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}

template void RunOpenCLKernelBuffers<unsigned char>(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const unsigned char* p_Input, unsigned char* p_Output);
template void RunOpenCLKernelBuffers<float>(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const float* p_Input, float* p_Output);

template void RunOpenCLKernelImages<unsigned char>(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const unsigned char* p_Input, unsigned char* p_Output);
template void RunOpenCLKernelImages<float>(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const float* p_Input, float* p_Output);