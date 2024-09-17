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

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif
#include "CLFuncs.h"

static std::map<int, std::string> kernelStrings;

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
        cl_context clContext = NULL;
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
        CheckError(error, "Unable to get the context");

        std::string bitStr = bits == 8 ? "unsigned char" : "float";

        std::string kernelStart = "__kernel void customKernel(int p_Width, int p_Height, float kTime, float kFloat0, float kFloat1, float kFloat2, float kFloat3, float kFloat4, float kFloat5, float kFloat6, float kFloat7, float kFloat8, float kFloat9, float kFloat10, float kFloat11, float kFloat12, float kFloat13, float kFloat14, float kFloat15, __global " + bitStr + "* p_Input, __global " + bitStr + "* p_Output, int bits)\n" \
            "{                                                                      \n" \
            "   const int x = get_global_id(0);                                     \n" \
            "   const int y = get_global_id(1);                                     \n" \
            "   float4 kOutput(0,0,0,0);                                           \n" \
            "   float2 kReadIndex(0,0);                                           \n" \
            "   if ((x < p_Width) && (y < p_Height))                                \n" \
            "   {                                                                   \n";
        std::string kernelEnd ="const int index = ((y * p_Width) + x) * 4;         \n" \
            "                                                                       \n" \
            "				if (bits == 8) {                                                                                                     \n"\
            "					kOutput.r = fmin(kOutput.r, 255.0);                                                                                 \n"\
            "					kOutput.g = fmin(kOutput.g, 255.0);                                                                                 \n"\
            "					kOutput.b = fmin(kOutput.b, 255.0);                                                                                 \n"\
            "					kOutput.a = fmin(kOutput.a, 255.0);                                                                                 \n"\
            "				}                                                                                 \n"\
            "       p_Output[index + 0] = kOutput.r;             \n" \
            "       p_Output[index + 1] = kOutput.g;             \n" \
            "       p_Output[index + 2] = kOutput.b;             \n" \
            "       p_Output[index + 3] = kOutput.a;             \n" \
            "   }                                                                   \n" \
            "}                                                                      \n";

        std::string kReadBuffer = "kOutput.r = p_Input[kReadIndex.x + kReadIndex.y * p_Width + 0];\n" \
            "kOutput.g = p_Input[kReadIndex.x + kReadIndex.y * p_Width + 1];\n" \
            "kOutput.b = p_Input[kReadIndex.x + kReadIndex.y * p_Width + 2];\n" \
            "kOutput.a = p_Input[kReadIndex.x + kReadIndex.y * p_Width + 3];\n";
        std::string::size_type kReadPos = kernel.find("kRead()");
        std::vector<std::string::size_type> kReads;
        std::string kernThing = kernel;
        while (kReadPos != std::string::npos) {
            kReads.push_back(kReadPos);
            kReadPos = kernel.find("kRead()", kReadPos + 1);
        }
        for (int i = 0; i < kReads.size(); i++) {
            kernThing.replace(kReads[kReads.size() - i - 1], 7, kReadBuffer);
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
            kernThing = kernelStart + fallbackKernel + kReadBuffer + kernelEnd;

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

    // find the program kernel corresponding to the command queue
    cl_kernel clKernel = NULL;
    cl_context clContext = NULL;
    if (kernelStrings[instance] != kernel)
    {
        if (kernelMap.find(instance) != kernelMap.end())
        {
            error = clReleaseKernel(kernelMap[instance]);
        }
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
        CheckError(error, "Unable to get the context");

        std::string bitStr = bits == 8 ? "unsigned char" : "float";

        std::string kernelStart = "__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;  \n" \
            "__kernel void customKernel(int p_Width, int p_Height, float kTime, float kFloat0, float kFloat1, float kFloat2, float kFloat3, float kFloat4, float kFloat5, float kFloat6, float kFloat7, float kFloat8, float kFloat9, float kFloat10, float kFloat11, float kFloat12, float kFloat13, float kFloat14, float kFloat15, __read_only image2d_t p_Input, __write_only image2d_t p_Output, int bits)\n" \
            "{                                                                      \n" \
            "   const int x = get_global_id(0);                                     \n" \
            "   const int y = get_global_id(1);                                     \n" \
            "   float4 kOutput = (float4)(0,0,0,0);                                           \n" \
            "   float2 kReadIndex = (float2)(0,0);                                           \n" \
            "   if ((x < p_Width) && (y < p_Height))                                \n" \
            "   {                                                                   \n";
        std::string kernelEnd ="	write_imagef(p_Output, (int2)(x, y), kOutput);                      \n"\
            "}                                                                   \n" \
            "}                                                                      \n";
        std::string kReadBuffer = "kOutput = read_imagef(p_Input, imageSampler, kReadIndex+(float2)0.5);";
        std::string::size_type kReadPos = kernel.find("kRead()");
        std::vector<std::string::size_type> kReads;
        std::string kernThing = kernel;
        while (kReadPos != std::string::npos) {
            kReads.push_back(kReadPos);
            kReadPos = kernel.find("kRead()", kReadPos+1);
        }
        for (int i = 0; i < kReads.size(); i++) {
            kernThing.replace(kReads[kReads.size() - i - 1], 7, kReadBuffer);
        }
        kernThing = kernelStart + kernThing + kernelEnd;
        char* kernStr = new char[kernThing.length() + 1];
        strcpy(kernStr, kernThing.c_str());
        cl_program program = clCreateProgramWithSource(clContext, 1, (const char**)&kernStr, NULL, &error);
        CheckError(error, "Unable to create program");

        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (error != CL_SUCCESS) {
        char errorInfo[65536];
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
            kernThing = kernelStart + fallbackKernel + kReadBuffer + kernelEnd;

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