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
"void crossMatrix(float ma[], float mb[], float mo[]) {                                                                                 \n"\
"    mo[0] = ma[0] * mb[0] + ma[1] * mb[3] + ma[2] * mb[6];                                                                                     \n"\
"    mo[1] = ma[0] * mb[1] + ma[1] * mb[4] + ma[2] * mb[7];                                                                                     \n"\
"    mo[2] = ma[0] * mb[2] + ma[1] * mb[5] + ma[2] * mb[8];                                                                                     \n"\
"    mo[3] = ma[3] * mb[0] + ma[4] * mb[3] + ma[5] * mb[6];                                                                                     \n"\
"    mo[4] = ma[3] * mb[1] + ma[4] * mb[4] + ma[5] * mb[7];                                                                                     \n"\
"    mo[5] = ma[3] * mb[2] + ma[4] * mb[5] + ma[5] * mb[8];                                                                                     \n"\
"    mo[6] = ma[6] * mb[0] + ma[7] * mb[3] + ma[8] * mb[6];                                                                                     \n"\
"    mo[7] = ma[6] * mb[1] + ma[7] * mb[4] + ma[8] * mb[7];                                                                                     \n"\
"    mo[8] = ma[6] * mb[2] + ma[7] * mb[5] + ma[8] * mb[8];                                                                                     \n"\
"}                                                                                                                                              \n"\
"void getPos(float p[], float m[], float o[]) {                                                                                         \n"\
"    o[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2];                                                                                            \n"\
"    o[1] = p[0] * m[3] + p[1] * m[4] + p[2] * m[5];                                                                                            \n"\
"    o[2] = p[0] * m[6] + p[1] * m[7] + p[2] * m[8];                                                                                            \n"\
"}                                                                                                                                              \n"\
"void inverseMatrix(float m[], float n[]) {                                                                                             \n"\
"    float t[9];                                                                                                                                \n"\
"    n[0] = m[4] * m[8] - m[5] * m[7];        // 0 1 2  0 1 2                                                                                   \n"\
"    n[1] = m[5] * m[6] - m[3] * m[8];        // 3 4 5  3 4 5                                                                                   \n"\
"    n[2] = m[3] * m[7] - m[4] * m[6];        // 6 7 8  6 7 8                                                                                   \n"\
"    n[3] = m[7] * m[2] - m[8] * m[1];                                                                                                          \n"\
"    n[4] = m[8] * m[0] - m[6] * m[2];        // 0 1 2  0 1 2                                                                                   \n"\
"    n[5] = m[6] * m[1] - m[7] * m[0];        // 3 4 5  3 4 5                                                                                   \n"\
"    n[6] = m[1] * m[5] - m[2] * m[4];        // 6 7 8  6 7 8                                                                                   \n"\
"    n[7] = m[2] * m[3] - m[0] * m[5];                                                                                                          \n"\
"    n[8] = m[0] * m[4] - m[1] * m[3];                                                                                                          \n"\
"    for(int i = 0; i < 9; i++) {                                                                                                          \n"\
"        t[i] = n[i];                                                                                                          \n"\
"    }                                                                                                          \n"\
"    n[1] = t[3];                                                                                                                               \n"\
"    n[3] = t[1];                                                                                                                               \n"\
"    n[2] = t[6];                                                                                                                               \n"\
"    n[6] = t[2];                                                                                                                               \n"\
"    n[5] = t[7];                                                                                                                               \n"\
"    n[7] = t[5];                                                                                                                               \n"\
"    for (int i = 0; i < 9; i++) {                                                                                                              \n"\
"        n[i] /= m[0] * t[0] + m[1] * t[1] + m[2] * t[2];                                                                                       \n"\
"    }                                                                                                                                          \n"\
"}                                                                                                                                              \n"\
"__kernel void transformBuffers(int p_Width, int p_Height, float m0x, float m0y, float m0z, float m00, float m01, float m02, float m03, float m04, float m05, float m06, float m07, float m08, float m1x, float m1y, float m1z, float m10, float m11, float m12, float m13, float m14, float m15, float m16, float m17, float m18, int p_wX, int p_wY, int p_wZ, float mb, float xScale, float yScale, float zScale, float xRot, float yRot, float zRot, int pIndex, int pSend, int now, int off, int forward, __global void* p_Input, __global void* p_Output, __global float* buffers, int bits)\n"\
"{                                                                                                                                   \n"\
"	float mt0[9] = { m00,m01,m02,m03,m04,m05,m06,m07,m08 };                                                                          \n"\
"	float mt1[9] = { m10,m11,m12,m13,m14,m15,m16,m17,m18 };                                                                          \n"\
"	float pt0[3] = { m0x, m0y, m0z };                                                                                                \n"\
"	float pt1[3] = { m1x, m1y, m1z };                                                                                                \n"\
"	float m0[9] = { m00,m01,m02,m03,m04,m05,m06,m07,m08 };                                                                           \n"\
"	float m1[9] = { m10,m11,m12,m13,m14,m15,m16,m17,m18 };                                                                           \n"\
"	float p0[3] = { m0x, m0y, m0z };                                                                                                 \n"\
"	float p1[3] = { m1x, m1y, m1z };                                                                                                 \n"\
"	float mp0[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };                                                                                   \n"\
"	float mp1[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };                                                                                   \n"\
"	float mpt0[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };                                                                                  \n"\
"	float mpt1[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };                                                                                  \n"\
"	float pp0[3] = { 0.,0.,0. };                                                                                                     \n"\
"	float pp1[3] = { 0.,0.,0. };                                                                                                     \n"\
"	float bP[25];                                                                                                                    \n"\
"	float mb0 = mb;                                                                                                                  \n"\
"	if (pIndex != 0) {                                                                                                               \n"\
"		float pParams[25];                                                                                                           \n"\
"    for(int i = 0; i < 25; i++) {                                                                                                          \n"\
"        pParams[i] = buffers[now*16*25+(pIndex - 1)*25+i];                                                                                                          \n"\
"    }                                                                                                          \n"\
"		pp0[0] = pParams[0];                                                                                                         \n"\
"		pp0[1] = pParams[1];                                                                                                         \n"\
"		pp0[2] = pParams[2];                                                                                                         \n"\
"		mpt0[0] = pParams[3];                                                                                                        \n"\
"		mpt0[1] = pParams[4];                                                                                                        \n"\
"		mpt0[2] = pParams[5];                                                                                                        \n"\
"		mpt0[3] = pParams[6];                                                                                                        \n"\
"		mpt0[4] = pParams[7];                                                                                                        \n"\
"		mpt0[5] = pParams[8];                                                                                                        \n"\
"		mpt0[6] = pParams[9];                                                                                                        \n"\
"		mpt0[7] = pParams[10];                                                                                                       \n"\
"		mpt0[8] = pParams[11];                                                                                                       \n"\
"		pp1[0] = pParams[12];                                                                                                        \n"\
"		pp1[1] = pParams[13];                                                                                                        \n"\
"		pp1[2] = pParams[14];                                                                                                        \n"\
"		mpt1[0] = pParams[15];                                                                                                       \n"\
"		mpt1[1] = pParams[16];                                                                                                       \n"\
"		mpt1[2] = pParams[17];                                                                                                       \n"\
"		mpt1[3] = pParams[18];                                                                                                       \n"\
"		mpt1[4] = pParams[19];                                                                                                       \n"\
"		mpt1[5] = pParams[20];                                                                                                       \n"\
"		mpt1[6] = pParams[21];                                                                                                       \n"\
"		mpt1[7] = pParams[22];                                                                                                       \n"\
"		mpt1[8] = pParams[23];                                                                                                       \n"\
"		mb0 += pParams[24];                                                                                                          \n"\
"		crossMatrix(mt0, mpt0, m0);                                                                                                  \n"\
"		crossMatrix(mt1, mpt1, m1);                                                                                                  \n"\
"		inverseMatrix(mpt0, mp0);                                                                                                    \n"\
"		inverseMatrix(mpt1, mp1);                                                                                                    \n"\
"		getPos(pt0, mp0, p0);                                                                                                        \n"\
"		getPos(pt1, mp1, p1);                                                                                                        \n"\
"	}                                                                                                                                \n"\
"                                                                                                                                    \n"\
"	bP[0] = p0[0] + pp0[0];                                                                                                          \n"\
"	bP[1] = p0[1] + pp0[1];                                                                                                          \n"\
"	bP[2] = p0[2] + pp0[2];                                                                                                          \n"\
"	bP[3] = m0[0];                                                                                                                   \n"\
"	bP[4] = m0[1];                                                                                                                   \n"\
"	bP[5] = m0[2];                                                                                                                   \n"\
"	bP[6] = m0[3];                                                                                                                   \n"\
"	bP[7] = m0[4];                                                                                                                   \n"\
"	bP[8] = m0[5];                                                                                                                   \n"\
"	bP[9] = m0[6];                                                                                                                   \n"\
"	bP[10] = m0[7];                                                                                                                  \n"\
"	bP[11] = m0[8];                                                                                                                  \n"\
"	bP[12] = p1[0] + pp1[0];                                                                                                         \n"\
"	bP[13] = p1[1] + pp1[1];                                                                                                         \n"\
"	bP[14] = p1[2] + pp1[2];                                                                                                         \n"\
"	bP[15] = m1[0];                                                                                                                  \n"\
"	bP[16] = m1[1];                                                                                                                  \n"\
"	bP[17] = m1[2];                                                                                                                  \n"\
"	bP[18] = m1[3];                                                                                                                  \n"\
"	bP[19] = m1[4];                                                                                                                  \n"\
"	bP[20] = m1[5];                                                                                                                  \n"\
"	bP[21] = m1[6];                                                                                                                  \n"\
"	bP[22] = m1[7];                                                                                                                  \n"\
"	bP[23] = m1[8];                                                                                                                  \n"\
"	bP[24] = mb0;                                                                                                                    \n"\
"                                                                                                                                    \n"\
"	if (pSend != 0) {                                                                                                                \n"\
"    for(int i = 0; i < 25; i++) {                                                                                                          \n"\
"        buffers[now*16*25+(pSend - 1)*25+i] = bP[i];                                                                                                          \n"\
"    }                                                                                                          \n"\
"	}                                                                                                                                \n"\
"	float xWin = p_Width;                                                                                                            \n"\
"	float yWin = p_Height;                                                                                                           \n"\
"	float ratio = xWin / yWin;                                                                                                       \n"\
"	float pi = 3.14159265358979323846;                                                                                              \n"\
"	int index = get_global_id(0) + get_global_id(1) * p_Width;                                                                       \n"\
"			int x = index % p_Width;                                                                                                 \n"\
"			int y = index / p_Width;                                                                                                 \n"\
"	if (index < p_Width * p_Height) {                                                                                                \n"\
"		if (off == 1) {                                                                                                              \n"\
"			float4 value(0,0,0,0);                                                                                           \n"\
"			int bMax = ceil(bP[24] * 32.);                                                                                         \n"\
"			bMax = bMax < 1 ? 1 : bMax;                                                                                            \n"\
"			float2 uv;                                                                                                               \n"\
"				float2 plot = (float2)(0., 0.);                                                                                                 \n"\
"				float2 plot1 = (float2)(0., 0.);                                                                                                 \n"\
"				float2 plot2 = (float2)(0., 0.);                                                                                                 \n"\
"				float d = 0., fac = 0.;                                                                                              \n"\
"				bool horizon, fwd;                                                                                              \n"\
"					for(int r = 0; r < 2; r++) {                                                                                                \n"\
"					uv.x = x / xWin;                                                                                                \n"\
"					uv.y = y / yWin;                                                                                                \n"\
"					uv.x -= 0.5;                                                                                                     \n"\
"					uv.y -= 0.5;                                                                                                     \n"\
"					uv.x *= ratio;                                                                                                   \n"\
"					float3 zA = (float3)(0., 0., 1.);                                                                                           \n"\
"					float3 xA = (float3)(1., 0., 0.);                                                                                           \n"\
"					float3 yA = (float3)(0., 1., 0.);                                                                                           \n"\
"					float m[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };                                                                    \n"\
"					float p[3] = { 0.,0.,1. };                                                                                      \n"\
"					for (int i = 0; i < 9; i++) {                                                                                    \n"\
"						m[i] = bP[3 + i] * (1. - r * bP[24]) + bP[15 + i] * r * bP[24];                                              \n"\
"					}                                                                                                                \n"\
"					for (int i = 0; i < 3; i++) {                                                                                    \n"\
"						p[i] = bP[i] * (1. - r * bP[24]) + bP[12 + i] * r * bP[24]; //part 2 :O                                      \n"\
"					}                                                                                                                \n"\
"					float3 l1 = (float3)(p[0], p[1], (p[2] + 1.));                                                                      \n"\
"					float3 l2 = (float3)(uv.x + p[0], uv.y + p[1], p[2]);		// = -0.5                                                \n"\
"					float3 l1t = l1;                                                                                                 \n"\
"					float3 l2t = l2;                                                                                                 \n"\
"					l1.x = l1t.x * m[0] + l1t.y * m[1] + l1t.z * m[2];                                                               \n"\
"					l2.x = l2t.x * m[0] + l2t.y * m[1] + l2t.z * m[2];                                                               \n"\
"					l1.y = l1t.x * m[3] + l1t.y * m[4] + l1t.z * m[5];                                                               \n"\
"					l2.y = l2t.x * m[3] + l2t.y * m[4] + l2t.z * m[5];                                                               \n"\
"					l1.z = l1t.x * m[6] + l1t.y * m[7] + l1t.z * m[8];                                                               \n"\
"					l2.z = l2t.x * m[6] + l2t.y * m[7] + l2t.z * m[8];                                                               \n"\
"					d = dot(zA, l2 - l1);                                                                                            \n"\
"					fac = ((d > 0. ? (p_wZ == 1. ? 1. : 0.) : 0.) - dot(zA, l1)) / d;                                                \n"\
"					float2 uhh = (float2)(0., 0.);                                                                                              \n"\
"				    fwd = forward == 0 && d > 0.;                                                                                   \n"\
"					horizon = fac < 0. || fwd;                                                                          \n"\
"					if (r == 0) {                                                                          \n"\
"					plot1.x = dot((l1 + ((l2 - l1) * fabs(fac))), xA);                                                                 \n"\
"					plot1.y = dot((l1 + ((l2 - l1) * fabs(fac))), yA);                                                                 \n"\
"					} else {                                                                 \n"\
"					plot2.x = dot((l1 + ((l2 - l1) * fabs(fac))), xA);                                                                 \n"\
"					plot2.y = dot((l1 + ((l2 - l1) * fabs(fac))), yA);                                                                 \n"\
"					}                                                                 \n"\
"					}                                                                 \n"\
"			for (int b = 1; b <= bMax; b++) {                                                                         \n"\
"					float rng = fract(sin((index)*112.9898f*b + 179.233f) * 43758.5453f, &rng); \n"\
"					rng += b - 1; \n"\
"					rng /= bMax; \n"\
"						plot.x = plot1.x * (1. - rng) + plot2.x * rng;                                                               \n"\
"						plot.y = plot1.y * (1. - rng) + plot2.y * rng;                                                               \n"\
"				plot.x /= ratio;                                                                                                     \n"\
"				plot.x += 0.5;                                                                                                       \n"\
"				plot.y += 0.5;                                                                                                       \n"\
"                                                                                                                                    \n"\
"							float tx = plot.x * xWin;                                                                               \n"\
"							float ty = plot.y * yWin;                                                                               \n"\
"							switch ((int)(p_wX)) {                                                                                   \n"\
"							case 2:                                                                                                  \n"\
"								tx = tx - xWin * floor(tx / xWin);                                                                   \n"\
"								break;                                                                                               \n"\
"							case 3:                                                                                                  \n"\
"								tx = xWin - fabs((tx - 2. * xWin * floor(tx / xWin / 2.)) - xWin);                                    \n"\
"							}                                                                                                        \n"\
"							switch ((int)(p_wY)) {                                                                                   \n"\
"							case 2:                                                                                                  \n"\
"								ty = ty - yWin * floor(ty / yWin);                                                                   \n"\
"								break;                                                                                               \n"\
"							case 3:                                                                                                  \n"\
"								ty = yWin - fabs((ty - 2. * yWin * floor(ty / yWin / 2.)) - yWin);                                    \n"\
"							}                                                                                                        \n"\
"							if(p_wX != 0) tx = tx < 0. ? 0. : (tx > xWin - 1. ? xWin - 1. : tx);                                    \n"\
"							if(p_wY != 0) ty = ty < 0. ? 0. : (ty > yWin - 1. ? yWin - 1. : ty);                                    \n"\
"							int ix = (int)floor(tx);                                                                                 \n"\
"							int iy = (int)floor(ty);                                                                                 \n"\
"                                                                                                                                    \n"\
"							bool horizon = fac < 0. || fwd;                                                                          \n"\
"							if (!horizon && (tx >= 0. && tx < xWin) && (ty >= 0. && ty < yWin))                                              \n"\
"							{                                                                                                        \n"\
"							    tx += 0.5;                                                                                                        \n"\
"							    ty += 0.5;                                                                                                    \n"\
"							    tx = tx < 0. ? 0. : (tx > xWin - 1. ? xWin - 1. : tx);                                    \n"\
"							    ty = ty < 0. ? 0. : (ty > yWin - 1. ? yWin - 1. : ty);                                    \n"\
"								const int index2 = (iy * xWin + ix) * 4;                                                \n"\
"								if(bits == 8) {                                                                                      \n"\
"								value.x += (__global unsigned char*)(p_Input[index2 + 0]) / bMax;              \n"\
"								value.y += (__global unsigned char*)(p_Input[index2 + 1]) / bMax;              \n"\
"								value.z += (__global unsigned char*)(p_Input[index2 + 2]) / bMax;              \n"\
"								value.w += (__global unsigned char*)(p_Input[index2 + 3]) / bMax;              \n"\
"							} else {                                                                                                 \n"\
"								value.x += (__global float*)(p_Input[index2 + 0]) / bMax;                      \n"\
"								value.y += (__global float*)(p_Input[index2 + 1]) / bMax;                      \n"\
"								value.z += (__global float*)(p_Input[index2 + 2]) / bMax;                      \n"\
"								value.w += (__global float*)(p_Input[index2 + 3]) / bMax;                      \n"\
"							}                                                                                                        \n"\
"					    }                                                                                                              \n"\
"			}                                                                                                                        \n"\
"				if (bits == 8) {                                                                                                     \n"\
"					value[0] = fmin(value[0], 255.0);                                                                                 \n"\
"					value[1] = fmin(value[1], 255.0);                                                                                 \n"\
"					value[2] = fmin(value[2], 255.0);                                                                                 \n"\
"					value[3] = fmin(value[3], 255.0);                                                                                 \n"\
"				    (__global unsigned char*)p_Output[index*4 + 0] = (unsigned char)(value.x);                                       \n"\
"				    (__global unsigned char*)p_Output[index*4 + 1] = (unsigned char)(value.y);                                       \n"\
"				    (__global unsigned char*)p_Output[index*4 + 2] = (unsigned char)(value.z);                                       \n"\
"				    (__global unsigned char*)p_Output[index*4 + 3] = (unsigned char)(value.w);                                       \n"\
"				} else {                                                                                                             \n"\
"				    (__global float*)p_Output[index*4 + 0] = (value.x);                                                              \n"\
"				    (__global float*)p_Output[index*4 + 1] = (value.y);                                                              \n"\
"				    (__global float*)p_Output[index*4 + 2] = (value.z);                                                              \n"\
"				    (__global float*)p_Output[index*4 + 3] = (value.w);                                                              \n"\
"			    }                                                                                                                    \n"\
"		}                                                                                                                            \n"\
"	}                                                                                                                                \n"\
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
"	int ss = now*16*25+(pSend - 1)*25;                                                                                                          \n"\
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
        CheckError(error, "Unable to build program");

        kernel = clCreateKernel(program, "transformBuffers", &error);
        CheckError(error, "Unable to create kernel");

        mipKernel = clCreateKernel(program, "createMippy", &error);
        CheckError(error, "Unable to create mip kernel");

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

    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (send != 0) clEnqueueReadBuffer(cmdQ, buffersCL, CL_TRUE, sizeof(float) * (index * 25 * 16 + (send - 1) * 25), sizeof(float) * 25, buffers, 0, NULL, NULL);
    clReleaseMemObject(buffersCL);
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
    if (send != 0) clEnqueueReadBuffer(cmdQ, buffersCL, CL_TRUE, sizeof(float) * (index * 25 * 16 + (send - 1) * 25), sizeof(float) * 25, buffers, 0, NULL, NULL);
    // clReleaseMemObject(buffersCL);
}

template void RunOpenCLKernelBuffers<unsigned char>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const unsigned char* p_Input, unsigned char* p_Output);
template void RunOpenCLKernelBuffers<float>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const float* p_Input, float* p_Output);

template void RunOpenCLKernelImages<unsigned char>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const unsigned char* p_Input, unsigned char* p_Output);
template void RunOpenCLKernelImages<float>(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const float* p_Input, float* p_Output);