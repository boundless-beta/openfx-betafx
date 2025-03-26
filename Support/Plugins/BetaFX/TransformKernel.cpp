// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
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

cl_mem buffersCL;

const char* KernelSourceBuffers = "\n" \
"__kernel void transformBuffers(int p_Width, int p_Height, float m0x, float m0y, float m0z, float m00, float m01, float m02, float m03, float m04, float m05, float m06, float m07, float m08, float m1x, float m1y, float m1z, float m10, float m11, float m12, float m13, float m14, float m15, float m16, float m17, float m18, int p_wX, int p_wY, int p_wZ, float mb, float xScale, float yScale, float zScale, float xRot, float yRot, float zRot, int pIndex, int pSend, int now, int off, int forward, __global void* p_Input, __global void* p_Output, __global float* buffers, int bits)\n"\
"{\n"\
"	int index = get_global_id(0) + get_global_id(1) * p_Width;                                                                       \n"\
"			int x = index % p_Width;                                                                                                 \n"\
"			int y = index / p_Width;                                                                                                 \n"\
"	if (pSend != 0) {                                                                                                                \n"\
"	int ss = (pSend - 1)*25;                                                                                                          \n"\
"	    buffers[ss+0] = m0x;                                                                                                          \n"\
"	    buffers[ss+1] = m0y;                                                                                                          \n"\
"	    buffers[ss+2] = m0z;                                                                                                          \n"\
"	    buffers[ss+3] = m00;                                                                                                                   \n"\
"	    buffers[ss+4] = m01;                                                                                                                   \n"\
"	    buffers[ss+5] = m02;                                                                                                                   \n"\
"	    buffers[ss+6] = m03;                                                                                                                   \n"\
"	    buffers[ss+7] = m04;                                                                                                                   \n"\
"	    buffers[ss+8] = m05;                                                                                                                   \n"\
"	    buffers[ss+9] = m06;                                                                                                                   \n"\
"	    buffers[ss+10] = m07;                                                                                                                  \n"\
"	    buffers[ss+11] = m08;                                                                                                                  \n"\
"	    buffers[ss+12] = m1x;                                                                                                         \n"\
"	    buffers[ss+13] = m1y;                                                                                                         \n"\
"	    buffers[ss+14] = m1z;                                                                                                         \n"\
"	    buffers[ss+15] = m10;                                                                                                                  \n"\
"	    buffers[ss+16] = m11;                                                                                                                  \n"\
"	    buffers[ss+17] = m12;                                                                                                                  \n"\
"	    buffers[ss+18] = m13;                                                                                                                  \n"\
"	    buffers[ss+19] = m14;                                                                                                                  \n"\
"	    buffers[ss+20] = m15;                                                                                                                  \n"\
"	    buffers[ss+21] = m16;                                                                                                                  \n"\
"	    buffers[ss+22] = m17;                                                                                                                  \n"\
"	    buffers[ss+23] = m18;                                                                                                                  \n"\
"	    buffers[ss+24] = mb;                                                                                                                  \n"\
"	}                                                                                                                                \n"\
"	float xWin = p_Width;                                                                                                            \n"\
"	float yWin = p_Height;                                                                                                           \n"\
"	float ratio = xWin / yWin;                                                                                                       \n"\
"	float pi = 3.14159265358979323846;                                                                                              \n"\
"	if (index < p_Width * p_Height) {                                                                                                \n"\
"		if (off == 1) {                                                                                                              \n"\
"			float4 value = (float4)(0,0,0,0);                                                                                           \n"\
"			int bMax = ceil(mb * 32.);                                                                                         \n"\
"			bMax = bMax < 1 ? 1 : bMax;                                                                                            \n"\
"			float2 uv;                                                                                                               \n"\
"				float2 plot = (float2)(0., 0.);                                                                                                 \n"\
"				float2 plot1 = (float2)(0., 0.);                                                                                                 \n"\
"				float2 plot2 = (float2)(0., 0.);                                                                                                 \n"\
"				float d = 0., fac = 0.;                                                                                              \n"\
"				bool horizon, fwd;                                                                                              \n"\
"				int rTimes = mb > 0 ? 2 : 1;                                                                                              \n"\
"					for(int r = 0; r < rTimes; r++) {                                                                                                \n"\
"					uv.x = x / xWin;                                                                                                \n"\
"					uv.y = y / yWin;                                                                                                \n"\
"					uv.x -= 0.5;                                                                                                     \n"\
"					uv.y -= 0.5;                                                                                                     \n"\
"					uv.x *= ratio;                                                                                                   \n"\
"					float3 zA = (float3)(0., 0., 1.);                                                                                           \n"\
"					float3 xA = (float3)(1., 0., 0.);                                                                                           \n"\
"					float3 yA = (float3)(0., 1., 0.);                                                                                           \n"\
"					float3 mm1 = (float3)(1.,0.,0.);                                                                    \n"\
"					float3 mm2 = (float3)(0.,1.,0.);                                                                    \n"\
"					float3 mm3 = (float3)(0.,0.,1.);                                                                    \n"\
"					float3 p = (float3)(0.,0.,1.);                                                                                      \n"\
"						mm1 = (float3)(m00,m01,m02) * (float3)(1. - r * mb) + (float3)(m10,m11,m12) * (float3)(r * mb);                                              \n"\
"						mm2 = (float3)(m03,m04,m05) * (float3)(1. - r * mb) + (float3)(m13,m14,m15) * (float3)(r * mb);                                              \n"\
"						mm3 = (float3)(m06,m07,m08) * (float3)(1. - r * mb) + (float3)(m16,m17,m18) * (float3)(r * mb);                                              \n"\
"						p = (float3)(m0x,m0y,m0z) * (float3)(1. - r * mb) + (float3)(m1x,m1y,m1z) * (float3)(r * mb); //part 2 :O                                      \n"\
"					float3 l1 = (float3)(p.xy, (p.z + 1.));                                                                      \n"\
"					float3 l2 = (float3)(uv + p.xy, p.z);		// = -0.5                                                \n"\
"					float3 l1t = l1;                                                                                                 \n"\
"					float3 l2t = l2;                                                                                                 \n"\
"					l1.x = dot(l1t, mm1);                                                               \n"\
"					l2.x = dot(l2t, mm1);                                                               \n"\
"					l1.y = dot(l1t, mm2);                                                               \n"\
"					l2.y = dot(l2t, mm2);                                                               \n"\
"					l1.z = dot(l1t, mm3);                                                               \n"\
"					l2.z = dot(l2t, mm3);                                                               \n"\
"					d = dot(zA, l2 - l1);                                                                                            \n"\
"					fac = ((d > 0. ? (p_wZ == 1. ? 1. : 0.) : 0.) - dot(zA, l1)) / d;                                                \n"\
"					float2 uhh = (float2)(0., 0.);                                                                                              \n"\
"				    fwd = forward == 0 && d > 0.;                                                                                   \n"\
"					horizon = fac < 0. || fwd;                                                                          \n"\
"					if (r == 0) {                                                                          \n"\
"					plot1.x = dot((l1 + ((l2 - l1) * fabs(fac))), xA);                                                                 \n"\
"					plot1.y = dot((l1 + ((l2 - l1) * fabs(fac))), yA);                                                                 \n"\
"					plot1.x /= ratio;                                                                 \n"\
"					plot1 += (float2)0.5;                                                                 \n"\
"					} else {                                                                 \n"\
"					plot2.x = dot((l1 + ((l2 - l1) * fabs(fac))), xA);                                                                 \n"\
"					plot2.y = dot((l1 + ((l2 - l1) * fabs(fac))), yA);                                                                 \n"\
"					plot2.x /= ratio;                                                                 \n"\
"					plot2 += (float2)0.5;                                                                 \n"\
"					}                                                                 \n"\
"					}                                                                 \n"\
"			for (int b = 1; b <= bMax; b++) {                                 \n"\
"			    if(mb > 0) {                                                                         \n"\
"					float rng = sin((index)*112.9898*b + 179.233) * 43758.5453; \n"\
"					rng -= floor(rng); \n"\
"					rng += b - 1; \n"\
"					rng /= bMax; \n"\
"						plot.x = plot1.x * (1. - rng) + plot2.x * rng;                                                               \n"\
"						plot.y = plot1.y * (1. - rng) + plot2.y * rng;                                                               \n"\
"					} else {                                                                                                         \n"\
"						plot = plot1;                                                                                            \n"\
"					}                                                                                                         \n"\
"							float tx = plot.x * xWin;                                                                               \n"\
"							float ty = plot.y * yWin;                                                                               \n"\
"                                                                                                                                    \n"\
"							switch ((int)(p_wX)) {                                                                                   \n"\
"							case 2:                                                                                                  \n"\
"								tx -= xWin * floor(tx / xWin);                                                                   \n"\
"								break;                                                                                               \n"\
"							case 3:                                                                                                  \n"\
"								tx = xWin - fabs((tx - 2. * xWin * floor(tx / xWin / 2.)) - xWin);                                    \n"\
"							}                                                                                                        \n"\
"							switch ((int)(p_wY)) {                                                                                   \n"\
"							case 2:                                                                                                  \n"\
"								ty -= yWin * floor(ty / yWin);                                                                   \n"\
"								break;                                                                                               \n"\
"							case 3:                                                                                                  \n"\
"								ty = yWin - fabs((ty - 2. * yWin * floor(ty / yWin / 2.)) - yWin);                                    \n"\
"							}                                                                                                        \n"\
"							if(p_wX != 0) tx = tx < 0. ? 0. : (tx > xWin - 1. ? xWin - 1. : tx);                                    \n"\
"							if(p_wY != 0) ty = ty < 0. ? 0. : (ty > yWin - 1. ? yWin - 1. : ty);                                    \n"\
"							int ix = (int)floor(tx);                                                                                 \n"\
"							int iy = (int)floor(ty);                                                                                 \n"\
"                                                                                                                                    \n"\
"							if (!horizon && (tx >= 0. && tx < xWin) && (ty >= 0. && ty < yWin))                                              \n"\
"							{                                                                                                        \n"\
"							    tx += 0.5;                                                                                                        \n"\
"							    ty += 0.5;                                                                                                    \n"\
"							    tx = tx < 0. ? 0. : (tx > xWin - 1. ? xWin - 1. : tx);                                    \n"\
"							    ty = ty < 0. ? 0. : (ty > yWin - 1. ? yWin - 1. : ty);                                    \n"\
"								const int index2 = (iy * xWin + ix) * 4;                                                \n"\
"								if(bits == 8) {                                                                                      \n"\
"								value.x += ((__global unsigned char*)p_Input)[index2 + 0] / bMax;              \n"\
"								value.y += ((__global unsigned char*)p_Input)[index2 + 1] / bMax;              \n"\
"								value.z += ((__global unsigned char*)p_Input)[index2 + 2] / bMax;              \n"\
"								value.w += ((__global unsigned char*)p_Input)[index2 + 3] / bMax;              \n"\
"							} else {                                                                                                 \n"\
"								value.x += ((__global float*)p_Input)[index2 + 0] / bMax;                      \n"\
"								value.y += ((__global float*)p_Input)[index2 + 1] / bMax;                      \n"\
"								value.z += ((__global float*)p_Input)[index2 + 2] / bMax;                      \n"\
"								value.w += ((__global float*)p_Input)[index2 + 3] / bMax;                      \n"\
"							}                                                                                                        \n"\
"					    }                                                                                                              \n"\
"			}                                                                                                                        \n"\
"				if (bits == 8) {                                                                                                     \n"\
"					value.x = fmin(value.x, 255.f);                                                                                 \n"\
"					value.y = fmin(value.y, 255.f);                                                                                 \n"\
"					value.z = fmin(value.z, 255.f);                                                                                 \n"\
"					value.w = fmin(value.w, 255.f);                                                                                 \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 0] = (unsigned char)(value.x);                                       \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 1] = (unsigned char)(value.y);                                       \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 2] = (unsigned char)(value.z);                                       \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 3] = (unsigned char)(value.w);                                       \n"\
"				} else {                                                                                                             \n"\
"				    ((__global float*)p_Output)[index*4 + 0] = (value.x);                                                              \n"\
"				    ((__global float*)p_Output)[index*4 + 1] = (value.y);                                                              \n"\
"				    ((__global float*)p_Output)[index*4 + 2] = (value.z);                                                              \n"\
"				    ((__global float*)p_Output)[index*4 + 3] = (value.w);                                                              \n"\
"			    }                                                                                                                    \n"\
"				} else {                                                                                                             \n"\
"				if (bits == 8) {                                                                                                     \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 0] = ((__global unsigned char*)p_Input)[index*4 + 0];                              \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 1] = ((__global unsigned char*)p_Input)[index*4 + 1];                              \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 2] = ((__global unsigned char*)p_Input)[index*4 + 2];                              \n"\
"				    ((__global unsigned char*)p_Output)[index*4 + 3] = ((__global unsigned char*)p_Input)[index*4 + 3];                              \n"\
"				} else {                                                                                                             \n"\
"				    ((__global float*)p_Output)[index*4 + 0] = ((__global float*)p_Input)[index*4 + 0];                              \n"\
"				    ((__global float*)p_Output)[index*4 + 1] = ((__global float*)p_Input)[index*4 + 1];                              \n"\
"				    ((__global float*)p_Output)[index*4 + 2] = ((__global float*)p_Input)[index*4 + 2];                              \n"\
"				    ((__global float*)p_Output)[index*4 + 3] = ((__global float*)p_Input)[index*4 + 3];                              \n"\
"}                                                                                                                                   \n"\
"		}                                                                                                                            \n"\
"}                                                                                                                                   \n"\
"}                                                                                                                                   \n"\
"\n";


const char* KernelSourceImages = "\n" \
"__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;  \n" \
"__kernel void transformImages(int p_Width, int p_Height, float m0x, float m0y, float m0z, float m00, float m01, float m02, float m03, float m04, float m05, float m06, float m07, float m08, float m1x, float m1y, float m1z, float m10, float m11, float m12, float m13, float m14, float m15, float m16, float m17, float m18, int p_wX, int p_wY, int p_wZ, float mb, float xScale, float yScale, float zScale, float xRot, float yRot, float zRot, int pIndex, int pSend, int now, int off, int forward, __read_only image2d_t p_Input, __write_only image2d_t p_Output, __global float* buffers, int bits)\n"\
"{                                                                                                                                   \n"\
"	int index = get_global_id(0) + get_global_id(1) * p_Width;                                                                       \n"\
"			int x = index % p_Width;                                                                                                 \n"\
"			int y = index / p_Width;                                                                                                 \n"\
"                                                                                                                                    \n"\
"	if (pSend != 0) {                                                                                                                \n"\
"	int ss = (pSend - 1)*25;                                                                                                          \n"\
"	    buffers[ss+0] = m0x;                                                                                                          \n"\
"	    buffers[ss+1] = m0y;                                                                                                          \n"\
"	    buffers[ss+2] = m0z;                                                                                                          \n"\
"	    buffers[ss+3] = m00;                                                                                                                   \n"\
"	    buffers[ss+4] = m01;                                                                                                                   \n"\
"	    buffers[ss+5] = m02;                                                                                                                   \n"\
"	    buffers[ss+6] = m03;                                                                                                                   \n"\
"	    buffers[ss+7] = m04;                                                                                                                   \n"\
"	    buffers[ss+8] = m05;                                                                                                                   \n"\
"	    buffers[ss+9] = m06;                                                                                                                   \n"\
"	    buffers[ss+10] = m07;                                                                                                                  \n"\
"	    buffers[ss+11] = m08;                                                                                                                  \n"\
"	    buffers[ss+12] = m1x;                                                                                                         \n"\
"	    buffers[ss+13] = m1y;                                                                                                         \n"\
"	    buffers[ss+14] = m1z;                                                                                                         \n"\
"	    buffers[ss+15] = m10;                                                                                                                  \n"\
"	    buffers[ss+16] = m11;                                                                                                                  \n"\
"	    buffers[ss+17] = m12;                                                                                                                  \n"\
"	    buffers[ss+18] = m13;                                                                                                                  \n"\
"	    buffers[ss+19] = m14;                                                                                                                  \n"\
"	    buffers[ss+20] = m15;                                                                                                                  \n"\
"	    buffers[ss+21] = m16;                                                                                                                  \n"\
"	    buffers[ss+22] = m17;                                                                                                                  \n"\
"	    buffers[ss+23] = m18;                                                                                                                  \n"\
"	    buffers[ss+24] = mb;                                                                                                                  \n"\
"	}                                                                                                                                \n"\
"	float xWin = p_Width;                                                                                                            \n"\
"	float yWin = p_Height;                                                                                                           \n"\
"	float ratio = xWin / yWin;                                                                                                       \n"\
"	float pi = 3.14159265358979323846;                                                                                              \n"\
"	if (index < p_Width * p_Height) {                                                                                                \n"\
"		if (off == 1) {                                                                                                              \n"\
"			uint4 valueui = (uint4)(0,0,0,0);                                                                                           \n"\
"			float4 valuef = (float4)(0,0,0,0);                                                                                           \n"\
"			int bMax = (int)ceil(mb * 32.);                                                                                         \n"\
"			bMax = bMax < 1 ? 1 : bMax;                                                                                            \n"\
"			float2 uv;                                                                                                               \n"\
"				float2 plot = (float2)(0., 0.);                                                                                                 \n"\
"				float2 plot1 = (float2)(0., 0.);                                                                                                 \n"\
"				float2 plot2 = (float2)(0., 0.);                                                                                                 \n"\
"				float d = 0., fac = 0.;                                                                                              \n"\
"				bool horizon, fwd;                                                                                              \n"\
"				int rTimes = mb > 0 ? 2 : 1;                                                                                              \n"\
"					for(int r = 0; r < rTimes; r++) {                                                                                                \n"\
"					uv.x = x / xWin;                                                                                                \n"\
"					uv.y = y / yWin;                                                                                                \n"\
"					uv.x -= 0.5;                                                                                                     \n"\
"					uv.y -= 0.5;                                                                                                     \n"\
"					uv.x *= ratio;                                                                                                   \n"\
"					float3 zA = (float3)(0., 0., 1.);                                                                                           \n"\
"					float3 xA = (float3)(1., 0., 0.);                                                                                           \n"\
"					float3 yA = (float3)(0., 1., 0.);                                                                                           \n"\
"					float3 mm1 = (float3)(1.,0.,0.);                                                                    \n"\
"					float3 mm2 = (float3)(0.,1.,0.);                                                                    \n"\
"					float3 mm3 = (float3)(0.,0.,1.);                                                                    \n"\
"					float3 p = (float3)(0.,0.,1.);                                                                                      \n"\
"						mm1 = (float3)(m00,m01,m02) * (float3)(1. - r * mb) + (float3)(m10,m11,m12) * (float3)(r * mb);                                              \n"\
"						mm2 = (float3)(m03,m04,m05) * (float3)(1. - r * mb) + (float3)(m13,m14,m15) * (float3)(r * mb);                                              \n"\
"						mm3 = (float3)(m06,m07,m08) * (float3)(1. - r * mb) + (float3)(m16,m17,m18) * (float3)(r * mb);                                              \n"\
"						p = (float3)(m0x,m0y,m0z) * (float3)(1. - r * mb) + (float3)(m1x,m1y,m1z) * (float3)(r * mb); //part 2 :O                                      \n"\
"					float3 l1 = (float3)(p.xy, (p.z + 1.));                                                                      \n"\
"					float3 l2 = (float3)(uv + p.xy, p.z);		// = -0.5                                                \n"\
"					float3 l1t = l1;                                                                                                 \n"\
"					float3 l2t = l2;                                                                                                 \n"\
"					l1.x = dot(l1t, mm1);                                                               \n"\
"					l2.x = dot(l2t, mm1);                                                               \n"\
"					l1.y = dot(l1t, mm2);                                                               \n"\
"					l2.y = dot(l2t, mm2);                                                               \n"\
"					l1.z = dot(l1t, mm3);                                                               \n"\
"					l2.z = dot(l2t, mm3);                                                               \n"\
"					d = dot(zA, l2 - l1);                                                                                            \n"\
"					fac = ((d > 0. ? (p_wZ == 1. ? 1. : 0.) : 0.) - dot(zA, l1)) / d;                                                \n"\
"					float2 uhh = (float2)(0., 0.);                                                                                              \n"\
"				    fwd = forward == 0 && d > 0.;                                                                                   \n"\
"					horizon = fac < 0. || fwd;                                                                          \n"\
"					if (r == 0) {                                                                          \n"\
"					plot1.x = dot((l1 + ((l2 - l1) * fabs(fac))), xA);                                                                 \n"\
"					plot1.y = dot((l1 + ((l2 - l1) * fabs(fac))), yA);                                                                 \n"\
"					plot1.x /= ratio;                                                                 \n"\
"					plot1 += (float2)0.5;                                                                 \n"\
"					} else {                                                                 \n"\
"					plot2.x = dot((l1 + ((l2 - l1) * fabs(fac))), xA);                                                                 \n"\
"					plot2.y = dot((l1 + ((l2 - l1) * fabs(fac))), yA);                                                                 \n"\
"					plot2.x /= ratio;                                                                 \n"\
"					plot2 += (float2)0.5;                                                                 \n"\
"					}                                                                 \n"\
"					}                                                                 \n"\
"					float4 valueCheck;                 \n"\
"			for (int b = 1; b <= bMax; b++) {                                                                         \n"\
"			    if(mb > 0) {                                                                         \n"\
"					float rng = sin((index)*112.9898*b + 179.233) * 43758.5453; \n"\
"					rng -= floor(rng); \n"\
"					rng += b - 1; \n"\
"					rng /= bMax; \n"\
"						plot.x = plot1.x * (1. - rng) + plot2.x * rng;                                                               \n"\
"						plot.y = plot1.y * (1. - rng) + plot2.y * rng;                                                               \n"\
"					} else {                                                                                                         \n"\
"						plot = plot1;                                                                                            \n"\
"					}                                                                                                         \n"\
"							float tx = plot.x * xWin;                                                                               \n"\
"							float ty = plot.y * yWin;                                                                               \n"\
"                                                                                                                                    \n"\
"							switch ((int)(p_wX)) {                                                                                   \n"\
"							case 2:                                                                                                  \n"\
"								tx -= xWin * floor(tx / xWin);                                                                   \n"\
"								break;                                                                                               \n"\
"							case 3:                                                                                                  \n"\
"								tx = xWin - fabs((tx - 2. * xWin * floor(tx / xWin / 2.)) - xWin);                                    \n"\
"							}                                                                                                        \n"\
"							switch ((int)(p_wY)) {                                                                                   \n"\
"							case 2:                                                                                                  \n"\
"								ty -= yWin * floor(ty / yWin);                                                                   \n"\
"								break;                                                                                               \n"\
"							case 3:                                                                                                  \n"\
"								ty = yWin - fabs((ty - 2. * yWin * floor(ty / yWin / 2.)) - yWin);                                    \n"\
"							}                                                                                                        \n"\
"							if(p_wX != 0) tx = tx < 0. ? 0. : (tx > xWin - 1. ? xWin - 1. : tx);                                    \n"\
"							if(p_wY != 0) ty = ty < 0. ? 0. : (ty > yWin - 1. ? yWin - 1. : ty);                                    \n"\
"                                                                                                                                    \n"\
"							if (!horizon && (tx >= 0. && tx < xWin) && (ty >= 0. && ty < yWin))                                              \n"\
"							{                                                                                                        \n"\
"							    tx += 0.5;                                                                                          \n"\
"							    ty += 0.5;                                                                                          \n"\
"							    tx = tx < 0. ? 0. : (tx > xWin - 1. ? xWin - 1. : tx);                                    \n"\
"							    ty = ty < 0. ? 0. : (ty > yWin - 1. ? yWin - 1. : ty);                                    \n"\
"					            valueCheck = read_imagef(p_Input, imageSampler, (float2)(tx, ty));                 \n"\
"								valuef += valueCheck / bMax;              \n"\
"					    }                                                                                                              \n"\
"			}                                                                                                                        \n"\
"					            write_imagef(p_Output, (int2)(x, y), (float4)valuef);              \n"\
"		} else {                                                                                                                            \n"\
"		    float4 col = read_imagef(p_Input, imageSampler, (float2)(x+0.5, y+0.5));                 \n"\
"		    write_imagef(p_Output, (int2)(x, y), col);                                                                  \n"\
"		}                                                                                                                            \n"\
"	}                                                                                                                                \n"\
"}                                                                                                                                   \n"\
"\n";
/*
const char *KernelSourceImages = "\n" \
"__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;  \n" \
"                                                                                                           \n" \
"__kernel void GainAdjustKernelImages(                                                                      \n" \
"   int p_Width,                                                                                            \n" \
"   int p_Height,                                                                                           \n" \
"   float p_GainR,                                                                                          \n" \
"   float p_GainG,                                                                                          \n" \
"   float p_GainB,                                                                                          \n" \
"   float p_GainA,                                                                                          \n" \
"   __read_only  image2d_t p_Input,                                                                         \n" \
"   __write_only image2d_t p_Output)                                                                        \n" \
"{                                                                                                          \n" \
"   const int x = get_global_id(0);                                                                         \n" \
"   const int y = get_global_id(1);                                                                         \n" \
"                                                                                                           \n" \
"   if ((x < p_Width) && (y < p_Height))                                                                    \n" \
"   {                                                                                                       \n" \
"       int2 coord = (int2)(x, y);                                                                          \n" \
"       float4 out = read_imagef(p_Input, imageSampler, coord);                                             \n" \
"       out *= (float4)(p_GainR, p_GainG, p_GainB, p_GainA);                                                \n" \
"       write_imagef(p_Output, coord, out);                                                                 \n" \
"   }                                                                                                       \n" \
"}                                                                                                          \n" \
"\n";
*/


template<class PIX>
void RunOpenCLKernelBuffers(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const PIX* p_Input, PIX* p_Output)
{
    cl_int error;

    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    // store device id and kernel per command queue (required for multi-GPU systems)
    static std::map<cl_command_queue, cl_device_id> deviceIdMap;
    static std::map<cl_command_queue, cl_kernel> kernelMap;

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
    cl_kernel kernel = NULL, mipKernel = NULL;
    cl_context clContext = NULL;
    error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    CheckError(error, "Unable to get the context");

    if (kernelMap.find(cmdQ) == kernelMap.end())
    {
        cl_program program = clCreateProgramWithSource(clContext, 1, (const char**)&KernelSourceBuffers, NULL, &error);
        CheckError(error, "Unable to create program");

        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        char errorInfo[65536];
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char) * 65536, &errorInfo, NULL);
        CheckError(error, "Unable to build program");

        kernel = clCreateKernel(program, "transformBuffers", &error);
        CheckError(error, "Unable to create kernel");

        kernelMap[cmdQ] = kernel;
    }
    else
    {
        kernel = kernelMap[cmdQ];
    }

    if (clGetMemObjectInfo(buffersCL, CL_MEM_SIZE, sizeof(int), NULL, NULL) != 0) {
        buffersCL = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 64 * 25 * 16 * sizeof(float), &buffers, NULL);
    }
    locker.Unlock();

    int count = 0;
    error = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[4]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[5]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[6]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[7]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[8]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[9]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[10]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[11]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[4]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[5]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[6]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[7]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[8]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[9]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[10]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[11]);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_wX);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_wY);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_wZ);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &blur);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &scales[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &scales[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &scales[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &angles[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &angles[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &angles[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &index);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &send);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &now);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &toggle);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &front);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &buffersCL);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &bits);

    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (send != 0) {
        locker.Lock();
        clEnqueueReadBuffer(cmdQ, buffersCL, CL_TRUE, sizeof(float) * ((send - 1) * 25), sizeof(float) * 25, &(buffers[0][send - 1][0]), 0, NULL, NULL);
        locker.Unlock();
    }
    // clReleaseMemObject(buffersCL);
}
template<class PIX>
void RunOpenCLKernelImages(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const PIX* p_Input, PIX* p_Output)
{
    cl_int error;

    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    // store device id and kernel per command queue (required for multi-GPU systems)
    static std::map<cl_command_queue, cl_device_id> deviceIdMap;
    static std::map<cl_command_queue, cl_kernel> kernelMap;

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
    cl_kernel kernel = NULL;
    cl_context clContext = NULL;
    error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    CheckError(error, "Unable to get the context");
    if (kernelMap.find(cmdQ) == kernelMap.end())
    {
        cl_program program = clCreateProgramWithSource(clContext, 1, (const char**)&KernelSourceImages, NULL, &error);
        CheckError(error, "Unable to create program");

        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        char errorInfo[65536];
        error = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(char)*65536, &errorInfo, NULL);
        CheckError(error, "Unable to build program");

        kernel = clCreateKernel(program, "transformImages", &error);
        CheckError(error, "Unable to create kernel");

        kernelMap[cmdQ] = kernel;
    }
    else
    {
        kernel = kernelMap[cmdQ];
    }

    locker.Unlock();

    if (clGetMemObjectInfo(buffersCL, CL_MEM_SIZE, sizeof(int), NULL, NULL) != 0) {
        buffersCL = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 64 * 25 * 16 * sizeof(float), &buffers, NULL);
    }

    int count = 0;
    error = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[4]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[5]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[6]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[7]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[8]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[9]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[10]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m0[11]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[4]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[5]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[6]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[7]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[8]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[9]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[10]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &m1[11]);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_wX);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_wY);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_wZ);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &blur);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &scales[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &scales[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &scales[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &angles[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &angles[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &angles[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &index);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &send);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &now);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &toggle);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &front);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &buffersCL);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &bits);

    //float mb, float xScale, float yScale, float zScale, float xRot, float yRot, float zRot, int pIndex, int pSend, int now, int off, int forward, __read_only image2d_t p_Input, __write_only image2d_t p_Output, __global 
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (send != 0) {
        locker.Lock();
        clEnqueueReadBuffer(cmdQ, buffersCL, CL_TRUE, sizeof(float) * ((send - 1) * 25), sizeof(float) * 25, &(buffers[0][send-1][0]), 0, NULL, NULL);
        locker.Unlock();
    }
    // clReleaseMemObject(buffersCL);
}

template void RunOpenCLKernelBuffers<unsigned char>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const unsigned char* p_Input, unsigned char* p_Output);
template void RunOpenCLKernelBuffers<float>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const float* p_Input, float* p_Output);

template void RunOpenCLKernelImages<unsigned char>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const unsigned char* p_Input, unsigned char* p_Output);
template void RunOpenCLKernelImages<float>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const float* p_Input, float* p_Output);