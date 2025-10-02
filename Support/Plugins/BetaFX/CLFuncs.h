// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifndef CLFUNCS_H
#define CLFUNCS_H

static cl_command_queue fallbackQ = NULL;

inline void CheckError(cl_int p_Error, const char* p_Msg)
{
    if (p_Error != CL_SUCCESS)
    {
        fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
    }
}


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


static cl_device_id GetDevices() {
    static cl_device_id p_DeviceId = NULL;
    static cl_platform_id platforms = NULL;
    cl_int error = clGetPlatformIDs(1, &platforms, NULL);
    error = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &p_DeviceId, NULL);
    return p_DeviceId;
}
static cl_context GetContext(cl_device_id& p_DeviceId)
{
    static cl_context clContext = NULL;
    cl_int error = CL_SUCCESS;
    if (clContext == NULL)
    {
        p_DeviceId = GetDevices();
        clContext = clCreateContext(NULL, 1, &p_DeviceId, NULL, NULL, NULL);
    }
    else
    {
        clGetContextInfo(clContext, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &p_DeviceId, NULL);
    }
    return clContext;
}

static bool postCLCheck() {
    cl_context clContext = NULL;
    cl_device_id deviceId = NULL;
    clContext = GetContext(deviceId);
    cl_int error = CL_SUCCESS;
    cl_device_type deviceType = NULL;
    error = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
    return (error == CL_SUCCESS && deviceType == CL_DEVICE_TYPE_GPU ? true : false);
}

#endif