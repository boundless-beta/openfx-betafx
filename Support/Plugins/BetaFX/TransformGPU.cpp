// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#pragma comment(lib, "opencl.lib")
#include "TransformGPU.h"

#include <stdio.h>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <thread>
#include <math.h>
#include <cmath>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#define kPluginName "BetaFX Dynamic Transform"
#define kPluginGrouping "BetaFX"
#define kPluginDescription "Transform an image using various static and animated parameters"
#define kPluginIdentifier "betafx:DynamicTransform"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define INSTANCE_COUNT 1024
std::vector<double> lastTime(INSTANCE_COUNT, 0);
std::vector<double> lastVel(18 * INSTANCE_COUNT, 0);
std::vector<double> lastVel2(18 * INSTANCE_COUNT, 0);
std::vector<double> lastValue(18 * INSTANCE_COUNT, 0);
std::vector<double> lastValue2(18 * INSTANCE_COUNT, 0);
std::vector<double> lastRate(18 * INSTANCE_COUNT, 0);
std::vector<double> lastRate2(18 * INSTANCE_COUNT, 0);
std::vector<double> lastRateW(18 * INSTANCE_COUNT, 0);
std::vector<double> lastRateW2(18 * INSTANCE_COUNT, 0);
int instanceStarted[INSTANCE_COUNT];

float buffersCPU[64][16][25];
static void initBufferCPU() {
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 16; j++) {
            for (int k = 0; k < 25; k++) {
                buffersCPU[i][j][k] = k % 4 == 3 ? 1. : 0.;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

class DynamicTransform : public OFX::ImageProcessor
{
public:
    explicit DynamicTransform(OFX::ImageEffect& p_Instance);

    virtual void processImagesCuda();
    virtual void processImagesOpenCL();
    virtual void processImagesMetal();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(double m0x, double m0y, double m0z, double m00, double m01, double m02, double m03, double m04, double m05, double m06, double m07, double m08, double m1x, double m1y, double m1z, double m10, double m11, double m12, double m13, double m14, double m15, double m16, double m17, double m18, int p_wX, int p_wY, int p_wZ, double mb, double xScale, double yScale, double zScale, double xRot, double yRot, double zRot, int pIndex, int pSend, int thisThread, int toggle, int forward, int bitDepth);

private:
    OFX::Image* _srcImg;
    float m0[12];
    float m1[12];
    int wX;
    int wY;
    int wZ;
    float blur;
    float scales[3];
    float angles[3];
    int index;
    int send;
    int now;
    int off;
    int front;
    int bits;
};

DynamicTransform::DynamicTransform(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

#ifdef OFX_SUPPORTS_CUDARENDER
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
#endif

void DynamicTransform::processImagesCuda()
{
#ifdef OFX_SUPPORTS_CUDARENDER
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(_pCudaStream, width, height, _scales, input, output);
#endif
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
#endif

void DynamicTransform::processImagesMetal()
{
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunMetalKernel(_pMetalCmdQ, width, height, _scales, input, output);
#endif
}

template<class PIX>
extern void RunOpenCLKernelBuffers(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int now, int send, int toggle, int front, int bits, const PIX* p_Input, PIX* p_Output);

template<class PIX>
extern void RunOpenCLKernelImages(void* p_CmdQ, int p_Width, int p_Height, float* m0, float* m1, int p_wX, int p_wY, int p_wZ, float blur, float* scales, float* angles, int index, int send, int now, int toggle, int front, int bits, const PIX* p_Input, PIX* p_Output);
// extern void RunOpenCLKernelImages(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
/* kernel parameter list
int p_Width
int p_Height
float m0x
float m0y
float m0z
float m00
float m01
float m02
float m03
float m04
float m05
float m06
float m07,
float m08,
float m1x,
float m1y,
float m1z,
float m10,
float m11,
float m12,
float m13,
float m14,
float m15,
float m16,
float m17,
float m18,
int p_wX,
int p_wY,
int p_wZ,
float mb,
float xScale,
float yScale,
float zScale,
float xRot,
double yRot,
double zRot,
int pIndex,
int pSend,
int now,
int off,
int forward,
PIX* p_Input,
PIX* p_Output,
int bits
*/

void DynamicTransform::processImagesOpenCL()
{
#ifdef OFX_SUPPORTS_OPENCLRENDER
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    int bitDepth = _srcImg->getPixelDepth() == OFX::eBitDepthUByte ? 8 : 32;
    float *inputF, *outputF;
    unsigned char *inputUI, *outputUI;
    if (bitDepth == 8) {
        inputUI = static_cast<unsigned char*>(_srcImg->getOpenCLImage());
        outputUI = static_cast<unsigned char*>(_dstImg->getOpenCLImage());
    } else {
        inputF = static_cast<float*>(_srcImg->getOpenCLImage());
        outputF = static_cast<float*>(_dstImg->getOpenCLImage());
    }

    // if a plugin supports both OpenCL Buffers and Images, the host decides which is used and
    // the plugin must determine which based on whether kOfxImageEffectPropOpenCLImage or kOfxImagePropData is set
    
    if (bitDepth == 8 && (inputUI || outputUI))
    {
            RunOpenCLKernelImages<unsigned char>(_pOpenCLCmdQ, width, height, m0, m1, wX, wY, wZ, blur, scales, angles, index, send, now, off, front, bitDepth, inputUI, outputUI);
    } else if (inputF || outputF) {
            RunOpenCLKernelImages<float>(_pOpenCLCmdQ, width, height, m0, m1, wX, wY, wZ, blur, scales, angles, index, send, now, off, front, bitDepth, inputF, outputF);
        }
    else if(bitDepth == 8)
    {
        inputUI = static_cast<unsigned char*>(_srcImg->getPixelData());
        outputUI = static_cast<unsigned char*>(_dstImg->getPixelData());

        RunOpenCLKernelBuffers<unsigned char>(_pOpenCLCmdQ, width, height, m0, m1, wX, wY, wZ, blur, scales, angles, index, send, now, off, front, bitDepth, inputUI, outputUI);
}
    else
    {
        inputF = static_cast<float*>(_srcImg->getPixelData());
        outputF = static_cast<float*>(_dstImg->getPixelData());

        RunOpenCLKernelBuffers<float>(_pOpenCLCmdQ, width, height, m0, m1, wX, wY, wZ, blur, scales, angles, index, send, now, off, front, bitDepth, inputF, outputF);
}
#endif
}
static void crossMatrix(float ma[], float mb[], float mo[]) {
    mo[0] = ma[0] * mb[0] + ma[1] * mb[3] + ma[2] * mb[6];
    mo[1] = ma[0] * mb[1] + ma[1] * mb[4] + ma[2] * mb[7];
    mo[2] = ma[0] * mb[2] + ma[1] * mb[5] + ma[2] * mb[8];
    mo[3] = ma[3] * mb[0] + ma[4] * mb[3] + ma[5] * mb[6];
    mo[4] = ma[3] * mb[1] + ma[4] * mb[4] + ma[5] * mb[7];
    mo[5] = ma[3] * mb[2] + ma[4] * mb[5] + ma[5] * mb[8];
    mo[6] = ma[6] * mb[0] + ma[7] * mb[3] + ma[8] * mb[6];
    mo[7] = ma[6] * mb[1] + ma[7] * mb[4] + ma[8] * mb[7];
    mo[8] = ma[6] * mb[2] + ma[7] * mb[5] + ma[8] * mb[8];
}
static void getPos(float p[], float m[], float o[]) {
    o[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2];
    o[1] = p[0] * m[3] + p[1] * m[4] + p[2] * m[5];
    o[2] = p[0] * m[6] + p[1] * m[7] + p[2] * m[8];
}
static void inverseMatrix(float m[], float n[]) {
    float t[9];
    n[0] = m[4] * m[8] - m[5] * m[7];// 0 1 20 1 2 
    n[1] = m[5] * m[6] - m[3] * m[8];// 3 4 53 4 5 
    n[2] = m[3] * m[7] - m[4] * m[6];// 6 7 86 7 8 
    n[3] = m[7] * m[2] - m[8] * m[1];
    n[4] = m[8] * m[0] - m[6] * m[2];// 0 1 20 1 2 
    n[5] = m[6] * m[1] - m[7] * m[0];// 3 4 53 4 5 
    n[6] = m[1] * m[5] - m[2] * m[4];// 6 7 86 7 8 
    n[7] = m[2] * m[3] - m[0] * m[5];
    n[8] = m[0] * m[4] - m[1] * m[3];
    for (int i = 0; i < 9; i++) {
        t[i] = n[i];
    }
    n[1] = t[3];
    n[3] = t[1];
    n[2] = t[6];
    n[6] = t[2];
    n[5] = t[7];
    n[7] = t[5];
    for (int i = 0; i < 9; i++) {
        n[i] /= m[0] * t[0] + m[1] * t[1] + m[2] * t[2];
    }
}

void DynamicTransform::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    if(!_isEnabledOpenCLRender) {
    float mt0[9] = { m0[3],m0[4],m0[5],m0[6],m0[7],m0[8],m0[9],m0[10],m0[11] };
    float mt1[9] = { m1[3],m1[4],m1[5],m1[6],m1[7],m1[8],m1[9],m1[10],m1[11] };
    float pt0[3] = { m0[0], m0[1], m0[2] };
    float pt1[3] = { m1[0], m1[1], m1[2] };
    float mm0[9] = { m0[3],m0[4],m0[5],m0[6],m0[7],m0[8],m0[9],m0[10],m0[11] };
    float mm1[9] = { m1[3],m1[4],m1[5],m1[6],m1[7],m1[8],m1[9],m1[10],m1[11] };
    float p0[3] = { m0[0], m0[1], m0[2] };
    float p1[3] = { m1[0], m1[1], m1[2] };
    float mp0[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
    float mp1[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
    float mpt0[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
    float mpt1[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
    float pp0[3] = { 0.,0.,0. };
    float pp1[3] = { 0.,0.,0. };
    float bP[25] = {};
    float mb0 = blur;
    if (index != 0) {
        float pParams[25];
        for (int i = 0; i < 25; i++) {
            pParams[i] = buffersCPU[now][index - 1][i];
        }
        pp0[0] = pParams[0];
        pp0[1] = pParams[1];
        pp0[2] = pParams[2];
        mpt0[0] = pParams[3];
        mpt0[1] = pParams[4];
        mpt0[2] = pParams[5];
        mpt0[3] = pParams[6];
        mpt0[4] = pParams[7];
        mpt0[5] = pParams[8];
        mpt0[6] = pParams[9];
        mpt0[7] = pParams[10];
        mpt0[8] = pParams[11];
        pp1[0] = pParams[12];
        pp1[1] = pParams[13];
        pp1[2] = pParams[14];
        mpt1[0] = pParams[15];
        mpt1[1] = pParams[16];
        mpt1[2] = pParams[17];
        mpt1[3] = pParams[18];
        mpt1[4] = pParams[19];
        mpt1[5] = pParams[20];
        mpt1[6] = pParams[21];
        mpt1[7] = pParams[22];
        mpt1[8] = pParams[23];
        mb0 += pParams[24];
        crossMatrix(mt0, mpt0, mm0);
        crossMatrix(mt1, mpt1, mm1);
        inverseMatrix(mpt0, mp0);
        inverseMatrix(mpt1, mp1);
        getPos(pt0, mp0, p0);
        getPos(pt1, mp1, p1);
    }

    bP[0] = p0[0] + pp0[0];
    bP[1] = p0[1] + pp0[1];
    bP[2] = p0[2] + pp0[2];
    bP[3] = mm0[0];
    bP[4] = mm0[1];
    bP[5] = mm0[2];
    bP[6] = mm0[3];
    bP[7] = mm0[4];
    bP[8] = mm0[5];
    bP[9] = mm0[6];
    bP[10] = mm0[7];
    bP[11] = mm0[8];
    bP[12] = p1[0] + pp1[0];
    bP[13] = p1[1] + pp1[1];
    bP[14] = p1[2] + pp1[2];
    bP[15] = mm1[0];
    bP[16] = mm1[1];
    bP[17] = mm1[2];
    bP[18] = mm1[3];
    bP[19] = mm1[4];
    bP[20] = mm1[5];
    bP[21] = mm1[6];
    bP[22] = mm1[7];
    bP[23] = mm1[8];
    bP[24] = mb0;

    if (send != 0) {
        for (int i = 0; i < 25; i++) {
            buffersCPU[now][send - 1][i] = bP[i];
        }
    }
    float xWin = p_ProcWindow.x2 - p_ProcWindow.x1;
    float yWin = p_ProcWindow.y2 - p_ProcWindow.y1;
    float ratio = xWin / yWin;
    double pi = 3.14159265358979323846;

    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        unsigned char* dstPix8 = static_cast<unsigned char*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
        float* dstPix32 = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {

            if (off == 1) {
                float value[4] = { 0.,0.,0.,0. };
                float bMax = ceil(bP[24] * 20.);
                bMax = bMax < 1. ? 1. : bMax;
                float uv[2] = { 0,0 };
                for (float bl = 0.; bl < bMax; bl += 1.) {
                    float plot[2] = { 0., 0. };
                    float size[2] = { 0., 0. };
                    float d = 0., fac = 0.;
                    for (int s = 0; s < 3; s++) {
                        int sx = x + (s % 2);
                        int sy = y + (s / 2);
                        double b = bl / bMax;
                        uv[0] = sx / xWin;
                        uv[1] = sy / yWin;
                        uv[0] -= 0.5;
                        uv[1] -= 0.5;
                        uv[0] *= ratio;
                        double zA[3] = { 0., 0., 1. };
                        double xA[3] = { 1., 0., 0. };
                        double yA[3] = { 0., 1., 0. };
                        double m[9] = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
                        double p[3] = { 0.,0.,1. };
                        for (int i = 0; i < 9; i++) {
                            m[i] = bP[3 + i] * (1. - b * bP[24]) + bP[15 + i] * b * bP[24];
                        }
                        for (int i = 0; i < 3; i++) {
                            p[i] = bP[i] * (1. - b * bP[24]) + bP[12 + i] * b * bP[24]; //part 2 :O
                        }
                        double l1[3] = { p[0], p[1], (p[2] + 1.) };
                        double l2[3] = { uv[0] + p[0], uv[1] + p[1], p[2] };		// = -0.5
                        double l1t[3] = { l1[0],l1[1],l1[2] };
                        double l2t[3] = { l2[0],l2[1],l2[2] };
                        l1[0] = l1t[0] * m[0] + l1t[1] * m[1] + l1t[2] * m[2];
                        l2[0] = l2t[0] * m[0] + l2t[1] * m[1] + l2t[2] * m[2];
                        l1[1] = l1t[0] * m[3] + l1t[1] * m[4] + l1t[2] * m[5];
                        l2[1] = l2t[0] * m[3] + l2t[1] * m[4] + l2t[2] * m[5];
                        l1[2] = l1t[0] * m[6] + l1t[1] * m[7] + l1t[2] * m[8];
                        l2[2] = l2t[0] * m[6] + l2t[1] * m[7] + l2t[2] * m[8];
                        for (int i = 0; i < 3; i++) {
                            d += zA[i] * (l2[i] - l1[i]);
                        }
                        for (int i = 0; i < 3; i++) {
                            fac += zA[i] * l1[i];
                        }
                        fac = ((d > 0. ? (wZ == 1. ? 1. : 0.) : 0.) - fac) / d;
                        double uhh[2] = { 0., 0. };
                        uhh[0] += (l1[0] + ((l2[0] - l1[0]) * fabs(fac))) * xA[0];
                        uhh[1] += (l1[0] + ((l2[0] - l1[0]) * fabs(fac))) * yA[0];
                        uhh[0] += (l1[1] + ((l2[1] - l1[1]) * fabs(fac))) * xA[1];
                        uhh[1] += (l1[1] + ((l2[1] - l1[1]) * fabs(fac))) * yA[1];
                        switch (s) {
                        case 0:
                            plot[0] = uhh[0];
                            plot[1] = uhh[1];
                            break;
                        case 1:
                            size[0] = fabs(uhh[0] - plot[0]) * xWin;
                            break;
                        case 2:
                            size[1] = fabs(uhh[1] - plot[1]) * yWin;
                        }
                    }
                    plot[0] /= ratio;
                    plot[0] += 0.5;
                    plot[1] += 0.5;
                    bool fwd = front == 0 && d > 0.;

                    double blxFloat = log2(fmax(size[0], size[1]));

                    blxFloat = blxFloat < 1. ? 1. : blxFloat;
                    // large value = smaller scale 
                    int blMax = floor(blxFloat);
                    blMax = fmin(log2(fmin(xWin, yWin) - 1), fmax(0, blMax));
                    int xmipOffset = 0;
                    int xmip = xWin, ymip = yWin;

                    double tx = plot[0] * xmip;
                    double ty = plot[1] * ymip;
                    double mipRatio = (double)xmip / ymip - xWin / yWin;
                    // tx *= 1. + mipRatio;
                    switch ((int)(wX)) {
                    case 2:
                        tx = tx - xWin * floor(tx / xWin);
                        break;
                    case 3:
                        tx = xWin - fabs((tx - 2. * xWin * floor(tx / xWin / 2.)) - xWin);
                    }
                    switch ((int)(wY)) {
                    case 2:
                        ty = ty - yWin * floor(ty / yWin);
                        break;
                    case 3:
                        ty = yWin - fabs((ty - 2. * yWin * floor(ty / yWin / 2.)) - yWin);
                    }
                    if (wX != 0) tx = fmin(fmax(tx, 0.), xWin - 1.);
                    if (wY != 0) ty = fmin(fmax(ty, 0.), yWin - 1.);
                    int ix = (int)floor(tx);
                    int iy = (int)floor(ty);

                    bool horizon = fac < 0. || fwd;
                    //mipped coords
                    if (!horizon && (tx >= 0. && tx < xWin) && (ty >= 0. && ty < yWin))
                    {
                        tx = fmin(fmax(tx, 0.5), xWin - .5);
                        ty = fmin(fmax(ty, 0.5), yWin - .5);
                        const int index2 = (iy * xWin + ix) * 4;

                        unsigned char* srcPix8 = static_cast<unsigned char*>(_srcImg ? _srcImg->getPixelAddress(ix, iy) : 0);
                        float* srcPix32 = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

                        // do we have a source image to scale up
                        if (bits == 8 && srcPix8) {
                            value[0] += srcPix8[0] / bMax;
                            value[1] += srcPix8[1] / bMax;
                            value[2] += srcPix8[2] / bMax;
                            value[3] += srcPix8[3] / bMax;
                        }
                        else if (bits == 32 && srcPix32) {
                            value[0] += srcPix32[0] / bMax;
                            value[1] += srcPix32[1] / bMax;
                            value[2] += srcPix32[2] / bMax;
                            value[3] += srcPix32[3] / bMax;
                        }
                    }
                }
                if (bits == 8) {
                    value[0] = fmin(value[0], 255.0);
                    value[1] = fmin(value[1], 255.0);
                    value[2] = fmin(value[2], 255.0);
                    value[3] = fmin(value[3], 255.0);
                    dstPix8[0] = (unsigned char)(value[0]);
                    dstPix8[1] = (unsigned char)(value[1]);
                    dstPix8[2] = (unsigned char)(value[2]);
                    dstPix8[3] = (unsigned char)(value[3]);
                }
                else {
                    dstPix32[0] = (value[0]);
                    dstPix32[1] = (value[1]);
                    dstPix32[2] = (value[2]);
                    dstPix32[3] = (value[3]);
                }
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    if (bits == 8) {
                        dstPix8[0] = 0;
                        dstPix8[1] = 0;
                        dstPix8[2] = 0;
                        dstPix8[3] = 0;
                    }
                    else {
                        dstPix32[0] = 0.;
                        dstPix32[1] = 0.;
                        dstPix32[2] = 0.;
                        dstPix32[3] = 0.;
                    }
                }
            }

            // increment the dst pixel
            dstPix8 += 4;
            dstPix32 += 4;
        }
        }
        }
}

void DynamicTransform::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void DynamicTransform::setScales(double m0x, double m0y, double m0z, double m00, double m01, double m02, double m03, double m04, double m05, double m06, double m07, double m08, double m1x, double m1y, double m1z, double m10, double m11, double m12, double m13, double m14, double m15, double m16, double m17, double m18, int p_wX, int p_wY, int p_wZ, double mb, double xScale, double yScale, double zScale, double xRot, double yRot, double zRot, int pIndex, int pSend, int thisThread, int toggle, int forward, int bitDepth)
{
    m0[0] = m0x;
    m0[1] = m0y;
    m0[2] = m0z;
    m0[3] = m00;
    m0[4] = m01;
    m0[5] = m02;
    m0[6] = m03;
    m0[7] = m04;
    m0[8] = m05;
    m0[9] = m06;
    m0[10] = m07;
    m0[11] = m08;
    m1[0] = m1x;
    m1[1] = m1y;
    m1[2] = m1z;
    m1[3] = m10;
    m1[4] = m11;
    m1[5] = m12;
    m1[6] = m13;
    m1[7] = m14;
    m1[8] = m15;
    m1[9] = m16;
    m1[10] = m17;
    m1[11] = m18;
    wX = p_wX;
    wY = p_wY;
    wZ = p_wZ;
    blur = mb;
    scales[0] = xScale;
    scales[1] = yScale;
    scales[2] = zScale;
    angles[0] = xRot;
    angles[1] = yRot;
    angles[2] = zRot;
    index = pIndex;
    send = pSend;
    now = thisThread;
    off = toggle;
    front = forward;
    bits = bitDepth;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class TransformGPU : public OFX::ImageEffect
{
public:
    explicit TransformGPU(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Set up and run a processor */
    void setupAndProcess(DynamicTransform &p_DynamicTransform, const OFX::RenderArguments& p_Args);

    virtual void getClipPreferences(OFX::ClipPreferencesSetter& clipPreferences);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::BooleanParam* m_Render;
    OFX::BooleanParam* m_Backface;
    OFX::IntParam* m_IsParent;
    OFX::IntParam* m_HasParent;
    OFX::DoubleParam* m_Px;
    OFX::DoubleParam* m_Py;
    OFX::DoubleParam* m_Pz;
    OFX::DoubleParam* m_Rx;
    OFX::DoubleParam* m_Ry;
    OFX::DoubleParam* m_Rz;
    OFX::DoubleParam* m_Sx;
    OFX::DoubleParam* m_Sy;
    OFX::DoubleParam* m_Sz;
    OFX::DoubleParam* m_WPx;
    OFX::DoubleParam* m_WPy;
    OFX::DoubleParam* m_WPz;
    OFX::DoubleParam* m_WRx;
    OFX::DoubleParam* m_WRy;
    OFX::DoubleParam* m_WRz;
    OFX::DoubleParam* m_WSx;
    OFX::DoubleParam* m_WSy;
    OFX::DoubleParam* m_WSz;
    OFX::DoubleParam* m_WrPx;
    OFX::DoubleParam* m_WrPy;
    OFX::DoubleParam* m_WrPz;
    OFX::DoubleParam* m_WrRx;
    OFX::DoubleParam* m_WrRy;
    OFX::DoubleParam* m_WrRz;
    OFX::DoubleParam* m_WrSx;
    OFX::DoubleParam* m_WrSy;
    OFX::DoubleParam* m_WrSz;
    OFX::DoubleParam* m_WrGl;
    OFX::DoubleParam* m_WGl;
    OFX::IntParam* m_WOct;
    OFX::DoubleParam* m_WSeed;
    OFX::DoubleParam* m_OPx;
    OFX::DoubleParam* m_OPy;
    OFX::DoubleParam* m_OPz;
    OFX::DoubleParam* m_ORx;
    OFX::DoubleParam* m_ORy;
    OFX::DoubleParam* m_ORz;
    OFX::DoubleParam* m_OSx;
    OFX::DoubleParam* m_OSy;
    OFX::DoubleParam* m_OSz;
    OFX::DoubleParam* m_OrPx;
    OFX::DoubleParam* m_OrPy;
    OFX::DoubleParam* m_OrPz;
    OFX::DoubleParam* m_OrRx;
    OFX::DoubleParam* m_OrRy;
    OFX::DoubleParam* m_OrRz;
    OFX::DoubleParam* m_OrSx;
    OFX::DoubleParam* m_OrSy;
    OFX::DoubleParam* m_OrSz;
    OFX::DoubleParam* m_OrGl;
    OFX::DoubleParam* m_OpPx;
    OFX::DoubleParam* m_OpPy;
    OFX::DoubleParam* m_OpPz;
    OFX::DoubleParam* m_OpRx;
    OFX::DoubleParam* m_OpRy;
    OFX::DoubleParam* m_OpRz;
    OFX::DoubleParam* m_OpSx;
    OFX::DoubleParam* m_OpSy;
    OFX::DoubleParam* m_OpSz;
    OFX::DoubleParam* m_OGl;
    OFX::DoubleParam* m_Temp;
    OFX::DoubleParam* m_Elast;
    OFX::BooleanParam* m_Bounce;
    OFX::ChoiceParam* m_WrapX;
    OFX::ChoiceParam* m_WrapY;
    OFX::BooleanParam* m_WrapZ;
    OFX::DoubleParam* m_Blur;
};

static double rateScalar(int instance, int offset, OFX::DoubleParam* rate, OFX::DoubleParam* rateG, double timeIn, double timeOut, double fr, bool isWiggle) {
    int index = instance * 18 + offset;
    double d = 0.;
    double p, inTime;
    if ((offset <= 8 && lastTime[instance] >= timeOut) || (offset > 8 && lastTime[instance] + 1. >= timeOut)) {
        p = 0.;
        inTime = timeIn;
    }
    else {
        if (isWiggle) {
            p = lastRateW[index];
            inTime = lastTime[instance];
            if (offset > 8) {
                inTime += 1.;
                p = lastRateW2[index];
            }
        }
        else {
            p = lastRate[index];
            inTime = lastTime[instance];
            if (offset > 8) {
                inTime += 1.;
                p = lastRate2[index];
            }
        }
    }
    double g = 0.;
    for (double i = inTime; i < timeOut; i += 1.) {
        d = rate->getValueAtTime(i);
        if (rateG != NULL) {
            g = rateG->getValueAtTime(i);
        }
        else {
            g = 1.;
        }
        p += (g * d);
    }
    if (isWiggle) {
        if (offset > 8) {
            lastRateW2[index] = p;
        }
        else {
            lastRateW[index] = p;
        }
    } else {
        if (offset > 8) {
            lastRate2[index] = p;
        }
        else {
            lastRate[index] = p;
        }
    }
    return p;
}

static double processToTime(int instance, int offset, OFX::DoubleParam* param, double timeIn, double timeOut, double fr, OFX::DoubleParam* a, OFX::DoubleParam* b, OFX::BooleanParam* bounce)
{
    int index = instance * 18 + offset;
    double result, vel, inTime;
    if ((offset <= 8 && lastTime[instance] >= timeOut) || (offset > 8 && lastTime[instance]+1. >= timeOut)) {
        // first set            p_Args.time               second set              p_Args.time + 1.
        vel = 0.;
        inTime = timeIn;
        result = param->getValueAtTime(timeIn);
    }
    else {
        result = lastValue[index];
        vel = lastVel[index];
        inTime = lastTime[instance];
        if (offset > 8) {
            inTime += 1.;
            result = lastValue2[index];
            vel = lastVel2[index];
        }
    }
    double phase = 0.;
    double value = 0.;
    double value0 = 0.;
    double value1 = 0.;
    double diff = 0.;
    double tempo = 1.;
    double elast = 0.;
    bool bounc = false;
    for (double i = inTime; i < timeOut; i += 1.) { // += fr / 240.
        //           0?       p_Args.time(+1)
        value = param->getValueAtTime(i);
        tempo = a->getValueAtTime(i);
        elast = b->getValueAtTime(i);
        value0 = value1 != value ? value1 : value0;
        bounc = bounce->getValueAtTime(i);
        tempo = pow(tempo, log2(1.+fr/15.)); // 64. (retain functionality)
        // vel = bounc ? fabs(vel) : vel;
        diff = value - result;
        vel += diff * tempo;
        result = result * (1. - tempo) + value * tempo;
        result += vel * elast * tempo;
        double polarity = ceil(value - value0) * 2. - 1.;
        polarity = polarity > 0. ? 1. : -1.;
        if (bounc && diff * polarity < 0.) {
            result = value;
            vel *= -1.;
        }
        value1 = value;
    }
    if (offset > 8) {
        lastValue2[index] = result;
        lastVel2[index] = vel;
    }
    else {
        lastValue[index] = result;
        lastVel[index] = vel;
    }
    return result;
}

static double oscillate(double t, double p) {
    return sin(((t / 60.) + p) * 3.14159265 * 2.);
}
static inline double perlin(double x, double s) {
    double h0 = 2. * fmod((34902.134 + s * 1000.) * sin((54.4329 + s) * 930. * floor(x)), 1.) - 1.;
    double h1 = 2. * fmod((34902.134 + s * 1000.) * sin((54.4329 + s) * 930. * floor(x + 1.)), 1.) - 1.;
    double f = pow(fmod(x, 1.), 2.) * (3. - 2. * fmod(x, 1.));
    return h0 * (-f + 1.) + h1 * f;
}
static double wiggle(double seed, double offset, double t, int oct)
{
    double n = 0.;
    for (int i = 0; i < oct; i++) {
        n += ((perlin((t / 30. + offset) * pow(2., double(i)) + 3. * double(i), seed*oct)) + 1.) / pow(2., double(i));
    }
    return n / 2.;
}

static inline std::vector<double> getMatrix(std::vector<double> m, double rx, double ry, double rz, double sx, double sy, double sz) {
    std::vector<double> n(9, 0.);
    std::vector<double> o(9, 0.);
    // scale
    o = m;
    o[0] /= sx;
    o[1] /= sy;
    o[2] /= sz;
    o[3] /= sx;
    o[4] /= sy;
    o[5] /= sz;
    o[6] /= sx;
    o[7] /= sy;
    o[8] /= sz;
    // rotate Z
    n = o;
    o[0] = n[0] * cos(rz) + n[1] * sin(rz);
    o[1] = n[0] * -sin(rz) + n[1] * cos(rz);
    o[2] = n[2];
    o[3] = n[3] * cos(rz) + n[4] * sin(rz);
    o[4] = n[3] * -sin(rz) + n[4] * cos(rz);
    o[5] = n[5];
    o[6] = n[6] * cos(rz) + n[7] * sin(rz);
    o[7] = n[6] * -sin(rz) + n[7] * cos(rz);
    o[8] = n[8];
    // rotate Y
    n = o;
    o[0] = n[0] * cos(ry) + n[2] * -sin(ry);
    o[1] = n[1];
    o[2] = n[0] * sin(ry) + n[2] * cos(ry);
    o[3] = n[3] * cos(ry) + n[5] * -sin(ry);
    o[4] = n[4];
    o[5] = n[3] * sin(ry) + n[5] * cos(ry);
    o[6] = n[6] * cos(ry) + n[8] * -sin(ry);
    o[7] = n[7];
    o[8] = n[6] * sin(ry) + n[8] * cos(ry);
    // rotate X
    n = o;
    o[0] = n[0];
    o[1] = n[1] * cos(rx) + n[2] * sin(rx);
    o[2] = n[1] * -sin(rx) + n[2] * cos(rx);
    o[3] = n[3];
    o[4] = n[4] * cos(rx) + n[5] * sin(rx);
    o[5] = n[4] * -sin(rx) + n[5] * cos(rx);
    o[6] = n[6];
    o[7] = n[7] * cos(rx) + n[8] * sin(rx);
    o[8] = n[7] * -sin(rx) + n[8] * cos(rx);
    return o;
}

static inline int threadQuery(std::thread::id threadId)
{
    static std::map<std::thread::id, int> threadIO;
    int theThread = 0;
    std::map<std::thread::id, int>::iterator iter = threadIO.find(threadId);
    if (iter == threadIO.end())
    {
        theThread = threadIO.size();
        threadIO[threadId] = theThread;
    }
    else {
        theThread = std::distance(threadIO.begin(), iter);
    }
    return theThread;
}

static inline int imageQuery(OFX::Clip* image)
{
    static std::map<OFX::Clip*, int> threadIO;
    int theImage = 0;
    std::map<OFX::Clip*, int>::iterator iter = threadIO.find(image);
    if (iter == threadIO.end())
    {
        theImage = threadIO.size();
        threadIO[image] = theImage;
    }
    else {
        theImage = std::distance(threadIO.begin(), iter);
    }
    return theImage;
}


TransformGPU::TransformGPU(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{ 
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_Px = fetchDoubleParam("posX");
    m_Py = fetchDoubleParam("posY");
    m_Pz = fetchDoubleParam("posZ");
    m_Rx = fetchDoubleParam("rotX");
    m_Ry = fetchDoubleParam("rotY");
    m_Rz = fetchDoubleParam("rotZ");
    m_Sx = fetchDoubleParam("sclX");
    m_Sy = fetchDoubleParam("sclY");
    m_Sz = fetchDoubleParam("sclZ");
    m_WPx = fetchDoubleParam("wPosX");
    m_WPy = fetchDoubleParam("wPosY");
    m_WPz = fetchDoubleParam("wPosZ");
    m_WRx = fetchDoubleParam("wRotX");
    m_WRy = fetchDoubleParam("wRotY");
    m_WRz = fetchDoubleParam("wRotZ");
    m_WSx = fetchDoubleParam("wSclX");
    m_WSy = fetchDoubleParam("wSclY");
    m_WSz = fetchDoubleParam("wSclZ");
    m_WGl = fetchDoubleParam("wGlobal");
    m_WrPx = fetchDoubleParam("wrPosX");
    m_WrPy = fetchDoubleParam("wrPosY");
    m_WrPz = fetchDoubleParam("wrPosZ");
    m_WrRx = fetchDoubleParam("wrRotX");
    m_WrRy = fetchDoubleParam("wrRotY");
    m_WrRz = fetchDoubleParam("wrRotZ");
    m_WrSx = fetchDoubleParam("wrSclX");
    m_WrSy = fetchDoubleParam("wrSclY");
    m_WrSz = fetchDoubleParam("wrSclZ");
    m_WrGl = fetchDoubleParam("wrGlobal");
    m_WOct = fetchIntParam("wOctaves");
    m_WSeed = fetchDoubleParam("wSeed");
    m_OPx = fetchDoubleParam("oPosX");
    m_OPy = fetchDoubleParam("oPosY");
    m_OPz = fetchDoubleParam("oPosZ");
    m_ORx = fetchDoubleParam("oRotX");
    m_ORy = fetchDoubleParam("oRotY");
    m_ORz = fetchDoubleParam("oRotZ");
    m_OSx = fetchDoubleParam("oSclX");
    m_OSy = fetchDoubleParam("oSclY");
    m_OSz = fetchDoubleParam("oSclZ");
    m_OGl = fetchDoubleParam("oGlobal");
    m_OrGl = fetchDoubleParam("orGlobal");
    m_OrPx = fetchDoubleParam("orPosX");
    m_OrPy = fetchDoubleParam("orPosY");
    m_OrPz = fetchDoubleParam("orPosZ");
    m_OrRx = fetchDoubleParam("orRotX");
    m_OrRy = fetchDoubleParam("orRotY");
    m_OrRz = fetchDoubleParam("orRotZ");
    m_OrSx = fetchDoubleParam("orSclX");
    m_OrSy = fetchDoubleParam("orSclY");
    m_OrSz = fetchDoubleParam("orSclZ");
    m_OpPx = fetchDoubleParam("opPosX");
    m_OpPy = fetchDoubleParam("opPosY");
    m_OpPz = fetchDoubleParam("opPosZ");
    m_OpRx = fetchDoubleParam("opRotX");
    m_OpRy = fetchDoubleParam("opRotY");
    m_OpRz = fetchDoubleParam("opRotZ");
    m_OpSx = fetchDoubleParam("opSclX");
    m_OpSy = fetchDoubleParam("opSclY");
    m_OpSz = fetchDoubleParam("opSclZ");
    m_Blur = fetchDoubleParam("mBlur");
    m_Bounce = fetchBooleanParam("bounce");
    m_Render = fetchBooleanParam("renderToggle");
    m_Backface = fetchBooleanParam("backToggle");
    m_WrapX = fetchChoiceParam("wrapX");
    m_WrapY = fetchChoiceParam("wrapY");
    m_WrapZ = fetchBooleanParam("wrapZ");
    m_IsParent = fetchIntParam("isParent");
    m_HasParent = fetchIntParam("hasParent");
    m_Temp = fetchDoubleParam("tempo");
    m_Elast = fetchDoubleParam("elast");
}

void TransformGPU::render(const OFX::RenderArguments& p_Args)
{
    bool bitCheck = m_DstClip->getPixelDepth() == OFX::eBitDepthUByte || m_DstClip->getPixelDepth() == OFX::eBitDepthFloat;
    if (bitCheck && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        DynamicTransform imageScaler(*this);
        setupAndProcess(imageScaler, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

void TransformGPU::setupAndProcess(DynamicTransform& p_DynamicTransform, const OFX::RenderArguments& p_Args)
{
    double PI = 3.14159265358979323846;
    // Get the dst image
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();
    

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }
    for (int i = 0; i < INSTANCE_COUNT; i++) {
        instanceStarted[i] = 0;
    }
    int thisThread = threadQuery(std::this_thread::get_id());
    int instanceIndex = imageQuery(m_SrcClip) % INSTANCE_COUNT;

    instanceStarted[instanceIndex] = instanceStarted[instanceIndex] == 0 ? 1 : 2;

    double t1 = m_SrcClip->getFrameRange().min;
    double frameRate = m_SrcClip->getFrameRate();

    // this is where all of the parameters start
    bool off = m_Render->getValueAtTime(p_Args.time);
    bool fwd = m_Backface->getValueAtTime(p_Args.time);
    int wx = 0, wy = 0;
    m_WrapX->getValueAtTime(p_Args.time, wx);
    m_WrapY->getValueAtTime(p_Args.time, wy);
    bool wz = m_WrapZ->getValueAtTime(p_Args.time);
    int pIndex = m_HasParent->getValueAtTime(p_Args.time);
    int pSend = m_IsParent->getValueAtTime(p_Args.time);
    double bParams[25];
    int wzInt = wz ? 1 : 0;

    double mb = m_Blur->getValueAtTime(p_Args.time); 
    double pxws = m_WPx->getValueAtTime(p_Args.time);
    double pyws = m_WPy->getValueAtTime(p_Args.time);
    double pzws = m_WPz->getValueAtTime(p_Args.time);
    double rxws = m_WRx->getValueAtTime(p_Args.time) / 180.;
    double ryws = m_WRy->getValueAtTime(p_Args.time) / 180.;
    double rzws = m_WRz->getValueAtTime(p_Args.time) / 180.;
    double sxws = m_WSx->getValueAtTime(p_Args.time);
    double syws = m_WSy->getValueAtTime(p_Args.time);
    double szws = m_WSz->getValueAtTime(p_Args.time);
    double pxos = m_OPx->getValueAtTime(p_Args.time);
    double pyos = m_OPy->getValueAtTime(p_Args.time);
    double pzos = m_OPz->getValueAtTime(p_Args.time);
    double rxos = m_ORx->getValueAtTime(p_Args.time) / 180.;
    double ryos = m_ORy->getValueAtTime(p_Args.time) / 180.;
    double rzos = m_ORz->getValueAtTime(p_Args.time) / 180.;
    double sxos = m_OSx->getValueAtTime(p_Args.time);
    double syos = m_OSy->getValueAtTime(p_Args.time);
    double szos = m_OSz->getValueAtTime(p_Args.time);
    double wstr = m_WGl->getValueAtTime(p_Args.time);
    double ostr = m_OGl->getValueAtTime(p_Args.time);

    pxos *= ostr;
    pyos *= ostr;
    pzos *= ostr;
    rxos *= ostr;
    ryos *= ostr;
    rzos *= ostr;
    sxos *= ostr;
    syos *= ostr;
    szos *= ostr;
    pxws *= wstr;
    pyws *= wstr;
    pzws *= wstr;
    rxws *= wstr;
    ryws *= wstr;
    rzws *= wstr;
    sxws *= wstr;
    syws *= wstr;
    szws *= wstr;

    double pxop = m_OpPx->getValueAtTime(p_Args.time);
    double pyop = m_OpPy->getValueAtTime(p_Args.time);
    double pzop = m_OpPz->getValueAtTime(p_Args.time);
    double rxop = m_OpRx->getValueAtTime(p_Args.time);
    double ryop = m_OpRy->getValueAtTime(p_Args.time);
    double rzop = m_OpRz->getValueAtTime(p_Args.time);
    double sxop = m_OpSx->getValueAtTime(p_Args.time);
    double syop = m_OpSy->getValueAtTime(p_Args.time);
    double szop = m_OpSz->getValueAtTime(p_Args.time);

    double seed = m_WSeed->getValueAtTime(p_Args.time);
    int octs = m_WOct->getValueAtTime(p_Args.time);
    double seedFactor = 10000 / 83; // arbitrarily odd value

    double px0 = processToTime(instanceIndex, 0, m_Px, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double py0 = processToTime(instanceIndex, 1, m_Py, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double pz0 = processToTime(instanceIndex, 2, m_Pz, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double rx0 = processToTime(instanceIndex, 3, m_Rx, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double ry0 = processToTime(instanceIndex, 4, m_Ry, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double rz0 = processToTime(instanceIndex, 5, m_Rz, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double sx0 = processToTime(instanceIndex, 6, m_Sx, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double sy0 = processToTime(instanceIndex, 7, m_Sy, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);
    double sz0 = processToTime(instanceIndex, 8, m_Sz, t1, p_Args.time, frameRate, m_Temp, m_Elast, m_Bounce);

    px0 += wiggle(seed, 1. * seedFactor, rateScalar(instanceIndex, 0, m_WrPx, m_WrGl, t1, p_Args.time, frameRate, true), octs) * pxws;
    py0 += wiggle(seed, 2. * seedFactor, rateScalar(instanceIndex, 1, m_WrPy, m_WrGl, t1, p_Args.time, frameRate, true), octs) * pyws;
    pz0 += wiggle(seed, 3. * seedFactor, rateScalar(instanceIndex, 2, m_WrPz, m_WrGl, t1, p_Args.time, frameRate, true), octs) * pzws;
    rx0 += wiggle(seed, 4. * seedFactor, rateScalar(instanceIndex, 3, m_WrRx, m_WrGl, t1, p_Args.time, frameRate, true), octs) * rxws * 180.;
    ry0 += wiggle(seed, 5. * seedFactor, rateScalar(instanceIndex, 4, m_WrRy, m_WrGl, t1, p_Args.time, frameRate, true), octs) * ryws * 180.;
    rz0 += wiggle(seed, 6. * seedFactor, rateScalar(instanceIndex, 5, m_WrRz, m_WrGl, t1, p_Args.time, frameRate, true), octs) * rzws * 180.;
    sx0 += wiggle(seed, 7. * seedFactor, rateScalar(instanceIndex, 6, m_WrSx, m_WrGl, t1, p_Args.time, frameRate, true), octs) * sxws;
    sy0 += wiggle(seed, 8. * seedFactor, rateScalar(instanceIndex, 7, m_WrSy, m_WrGl, t1, p_Args.time, frameRate, true), octs) * syws;
    sz0 += wiggle(seed, 9. * seedFactor, rateScalar(instanceIndex, 8, m_WrSz, m_WrGl, t1, p_Args.time, frameRate, true), octs) * szws;

    px0 += oscillate(rateScalar(instanceIndex, 0, m_OrPx, m_OrGl, t1, p_Args.time, frameRate, false), pxop) * pxos;
    py0 += oscillate(rateScalar(instanceIndex, 1, m_OrPy, m_OrGl, t1, p_Args.time, frameRate, false), pyop) * pyos;
    pz0 += oscillate(rateScalar(instanceIndex, 2, m_OrPz, m_OrGl, t1, p_Args.time, frameRate, false), pzop) * pzos;
    rx0 += oscillate(rateScalar(instanceIndex, 3, m_OrRx, m_OrGl, t1, p_Args.time, frameRate, false), rxop) * rxos * 180.;
    ry0 += oscillate(rateScalar(instanceIndex, 4, m_OrRy, m_OrGl, t1, p_Args.time, frameRate, false), ryop) * ryos * 180.;
    rz0 += oscillate(rateScalar(instanceIndex, 5, m_OrRz, m_OrGl, t1, p_Args.time, frameRate, false), rzop) * rzos * 180.;
    sx0 += oscillate(rateScalar(instanceIndex, 6, m_OrSx, m_OrGl, t1, p_Args.time, frameRate, false), sxop) * sxos;
    sy0 += oscillate(rateScalar(instanceIndex, 7, m_OrSy, m_OrGl, t1, p_Args.time, frameRate, false), syop) * syos;
    sz0 += oscillate(rateScalar(instanceIndex, 8, m_OrSz, m_OrGl, t1, p_Args.time, frameRate, false), szop) * szos;

    double px1 = processToTime(instanceIndex,  9, m_Px, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double py1 = processToTime(instanceIndex, 10, m_Py, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double pz1 = processToTime(instanceIndex, 11, m_Pz, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double rx1 = processToTime(instanceIndex, 12, m_Rx, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double ry1 = processToTime(instanceIndex, 13, m_Ry, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double rz1 = processToTime(instanceIndex, 14, m_Rz, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double sx1 = processToTime(instanceIndex, 15, m_Sx, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double sy1 = processToTime(instanceIndex, 16, m_Sy, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);
    double sz1 = processToTime(instanceIndex, 17, m_Sz, t1, p_Args.time + 1., frameRate, m_Temp, m_Elast, m_Bounce);

    px1 += wiggle(seed, 1. * seedFactor, rateScalar(instanceIndex,  9, m_WrPx, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * pxws;
    py1 += wiggle(seed, 2. * seedFactor, rateScalar(instanceIndex, 10, m_WrPy, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * pyws;
    pz1 += wiggle(seed, 3. * seedFactor, rateScalar(instanceIndex, 11, m_WrPz, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * pzws;
    rx1 += wiggle(seed, 4. * seedFactor, rateScalar(instanceIndex, 12, m_WrRx, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * rxws * 180.;
    ry1 += wiggle(seed, 5. * seedFactor, rateScalar(instanceIndex, 13, m_WrRy, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * ryws * 180.;
    rz1 += wiggle(seed, 6. * seedFactor, rateScalar(instanceIndex, 14, m_WrRz, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * rzws * 180.;
    sx1 += wiggle(seed, 7. * seedFactor, rateScalar(instanceIndex, 15, m_WrSx, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * sxws;
    sy1 += wiggle(seed, 8. * seedFactor, rateScalar(instanceIndex, 16, m_WrSy, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * syws;
    sz1 += wiggle(seed, 9. * seedFactor, rateScalar(instanceIndex, 17, m_WrSz, m_WrGl, t1, p_Args.time + 1., frameRate, true), octs) * szws;

    px1 += oscillate(rateScalar(instanceIndex,  9, m_OrPx, m_OrGl, t1, p_Args.time + 1., frameRate, false), pxop) * pxos;
    py1 += oscillate(rateScalar(instanceIndex, 10, m_OrPy, m_OrGl, t1, p_Args.time + 1., frameRate, false), pyop) * pyos;
    pz1 += oscillate(rateScalar(instanceIndex, 11, m_OrPz, m_OrGl, t1, p_Args.time + 1., frameRate, false), pzop) * pzos;
    rx1 += oscillate(rateScalar(instanceIndex, 12, m_OrRx, m_OrGl, t1, p_Args.time + 1., frameRate, false), rxop) * rxos * 180.;
    ry1 += oscillate(rateScalar(instanceIndex, 13, m_OrRy, m_OrGl, t1, p_Args.time + 1., frameRate, false), ryop) * ryos * 180.;
    rz1 += oscillate(rateScalar(instanceIndex, 14, m_OrRz, m_OrGl, t1, p_Args.time + 1., frameRate, false), rzop) * rzos * 180.;
    sx1 += oscillate(rateScalar(instanceIndex, 15, m_OrSx, m_OrGl, t1, p_Args.time + 1., frameRate, false), sxop) * sxos;
    sy1 += oscillate(rateScalar(instanceIndex, 16, m_OrSy, m_OrGl, t1, p_Args.time + 1., frameRate, false), syop) * syos;
    sz1 += oscillate(rateScalar(instanceIndex, 17, m_OrSz, m_OrGl, t1, p_Args.time + 1., frameRate, false), szop) * szos;
    
    lastTime[instanceIndex] = p_Args.time;
    // parameters end here

    // matrix math
    std::vector<double> mat0 = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
    std::vector<double> mat1 = { 1.,0.,0.,0.,1.,0.,0.,0.,1. };
    mat0 = getMatrix(mat0, rx0 / 180. * PI, ry0 / 180. * PI, rz0 / 180. * PI, sx0, sy0, sz0);
    mat1 = getMatrix(mat1, rx1 / 180. * PI, ry1 / 180. * PI, rz1 / 180. * PI, sx1, sy1, sz1);
    bParams[0] = px0;
    bParams[1] = py0;
    bParams[2] = pz0;
    bParams[3] = mat0[0];
    bParams[4] = mat0[1];
    bParams[5] = mat0[2];
    bParams[6] = mat0[3];
    bParams[7] = mat0[4];
    bParams[8] = mat0[5];
    bParams[9] = mat0[6];
    bParams[10] = mat0[7];
    bParams[11] = mat0[8];
    bParams[12] = px1;
    bParams[13] = py1;
    bParams[14] = pz1;
    bParams[15] = mat1[0];
    bParams[16] = mat1[1];
    bParams[17] = mat1[2];
    bParams[18] = mat1[3];
    bParams[19] = mat1[4];
    bParams[20] = mat1[5];
    bParams[21] = mat1[6];
    bParams[22] = mat1[7];
    bParams[23] = mat1[8];
    bParams[24] = mb;

    int bitDepth = srcBitDepth == OFX::eBitDepthUByte ? 8 : 32;

    p_DynamicTransform.setScales(bParams[0], bParams[1], bParams[2], bParams[3], bParams[4], bParams[5], bParams[6], bParams[7], bParams[8], bParams[9], bParams[10], bParams[11], bParams[12], bParams[13], bParams[14], bParams[15], bParams[16], bParams[17], bParams[18], bParams[19], bParams[20], bParams[21], bParams[22], bParams[23], wx, wy, wz, bParams[24], sx0, sy0, sz0, rx0, ry0, sz0, pIndex, pSend, thisThread, off, fwd, bitDepth);

    // Set the images
    p_DynamicTransform.setDstImg(dst.get());
    p_DynamicTransform.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_DynamicTransform.setGPURenderArgs(p_Args);

    // Set the render window
    p_DynamicTransform.setRenderWindow(p_Args.renderWindow);

    // Call the base class process member, this will call the derived templated process code
    p_DynamicTransform.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

TransformGPUFactory::TransformGPUFactory()
    : OFX::PluginFactoryHelper<TransformGPUFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void TransformGPU::getClipPreferences(ClipPreferencesSetter& clipPreferences) {
    clipPreferences.setOutputFrameVarying(true);
}

void TransformGPUFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthUByte);
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setRenderThreadSafety(eRenderFullySafe);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup OpenCL render capability flags
    p_Desc.setSupportsOpenCLBuffersRender(true);
    p_Desc.setSupportsOpenCLImagesRender(true);

}

static DoubleParamDescriptor* newDoubleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent, double p_RMin, double p_VMin, double p_Default, double p_VMax, double p_RMax)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(p_Default);
    param->setRange(p_RMin, p_RMax);
    param->setDisplayRange(p_VMin, p_VMax);
    param->setDoubleType(eDoubleTypePlain);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}
static IntParamDescriptor* newIntParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent, int p_RMin, int p_VMin, int p_Default, int p_VMax, int p_RMax)
{
    IntParamDescriptor* param = p_Desc.defineIntParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(p_Default);
    param->setRange(p_RMin, p_RMax);
    param->setDisplayRange(p_VMin, p_VMax);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}
static BooleanParamDescriptor* newBoolParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent, bool p_Default)
{
    BooleanParamDescriptor* param = p_Desc.defineBooleanParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(p_Default);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;

}static ChoiceParamDescriptor* newChoiceParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    ChoiceParamDescriptor* param = p_Desc.defineChoiceParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(0);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void TransformGPUFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);



    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Group param to group the scales
    GroupParamDescriptor* globalGroup = p_Desc.defineGroupParam("globalControls");
    globalGroup->setHint("Basic transform controls");
    globalGroup->setLabels("Global Controls", "Global Controls", "Global Controls");
    GroupParamDescriptor* wiggleGroup = p_Desc.defineGroupParam("wiggleControls");
    wiggleGroup->setHint("Wiggle transform controls");
    wiggleGroup->setLabels("Wiggle Controls", "Wiggle Controls", "Wiggle Controls");
    GroupParamDescriptor* oscillateGroup = p_Desc.defineGroupParam("oscillateControls");
    oscillateGroup->setHint("Oscillate transform controls");
    oscillateGroup->setLabels("Oscillate Controls", "Wiggle Controls", "Wiggle Controls");
    GroupParamDescriptor* temporalGroup = p_Desc.defineGroupParam("temporalControls");
    temporalGroup->setHint("Temporal controls");
    temporalGroup->setLabels("Temporal Controls", "Temporal Controls", "Temporal Controls");
    GroupParamDescriptor* miscGroup = p_Desc.defineGroupParam("miscControls");
    miscGroup->setHint("Miscellaneous controls");
    miscGroup->setLabels("Misc. Controls", "Misc. Controls", "Misc. Controls");

    // Make overall scale params
    DoubleParamDescriptor* param = newDoubleParam(p_Desc, "posX", "Translate X", "Translation along x-axis", globalGroup, -1000., -10., 0., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "posY", "Translate Y", "Translation along y-axis", globalGroup, -1000., -10., 0., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "posZ", "Translate Z", "Translation along z-axis", globalGroup, -1000., -10., 0., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "rotX", "Rotate X", "Rotation about x-axis", globalGroup, -360000., -360., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "rotY", "Rotate Y", "Rotation about y-axis", globalGroup, -360000., -360., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "rotZ", "Rotate Z", "Rotation about z-axis", globalGroup, -360000., -360., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "sclX", "Scale X", "Scale along x-axis", globalGroup, -1000., -10., 1., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "sclY", "Scale Y", "Scale along y-axis", globalGroup, -1000., -10., 1., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "sclZ", "Scale Z", "Scale along z-axis", globalGroup, -1000., -10., 1., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wPosX", "Wiggle Translate X", "Wiggle position along x-axis", wiggleGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrPosX", "Wiggle Translate X Rate", "Rate to wiggle position along x-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wPosY", "Wiggle Translate Y", "Wiggle position along y-axis", wiggleGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrPosY", "Wiggle Translate Y Rate", "Rate to wiggle position along y-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wPosZ", "Wiggle Translate Z", "Wiggle position along z-axis", wiggleGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrPosZ", "Wiggle Translate Z Rate", "Rate to wiggle position along z-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wRotX", "Wiggle Rotate X", "Wiggle Rotation about x-axis", wiggleGroup, -360000., 0., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrRotX", "Wiggle Rotate X Rate", "Rate to wiggle Rotation about x-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wRotY", "Wiggle Rotate Y", "Wiggle Rotation about y-axis", wiggleGroup, -360000., 0., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrRotY", "Wiggle Rotate Y Rate", "Rate to wiggle Rotation about y-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wRotZ", "Wiggle Rotate Z", "Wiggle Rotation about z-axis", wiggleGroup, -360000., 0., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrRotZ", "Wiggle Rotate Z Rate", "Rate to wiggle Rotation about z-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wSclX", "Wiggle Scale X", "Wiggle Scale along x-axis", wiggleGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrSclX", "Wiggle Scale X Rate", "Rate to wiggle Scale along x-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wSclY", "Wiggle Scale Y", "Wiggle Scale along y-axis", wiggleGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrSclY", "Wiggle Scale Y Rate", "Rate to wiggle Scale along y-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wSclZ", "Wiggle Scale Z", "Wiggle Scale along z-axis", wiggleGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrSclZ", "Wiggle Scale Z Rate", "Rate to wiggle Scale along z-axis", wiggleGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wGlobal", "Wiggle Global Strength", "Global wiggle factor", wiggleGroup, -100., 0., 1., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "wrGlobal", "Wiggle Global Rate", "Global wiggle rate", wiggleGroup, 0., 0., 1., 10., 1000.);
    page->addChild(*param);
    IntParamDescriptor *intParam = newIntParam(p_Desc, "wOctaves", "Wiggle Octaves", "Wiggle complexity", wiggleGroup, 1, 1, 1, 10, 10);
    page->addChild(*intParam);
    param = newDoubleParam(p_Desc, "wSeed", "Wiggle Seed", "Seed used to randomize wiggle", wiggleGroup, 0., 0., 0., 65536., 65536.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oPosX", "Oscillate Translate X", "Oscillate position along x-axis", oscillateGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orPosX", "Oscillate Translate X Rate", "Rate to Oscillate position along x-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opPosX", "Oscillate Translate X Phase", "Phase offset for Oscillate position along x-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oPosY", "Oscillate Translate Y", "Oscillate position along y-axis", oscillateGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orPosY", "Oscillate Translate Y Rate", "Rate to Oscillate position along y-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opPosY", "Oscillate Translate Y Phase", "Phase offset for Oscillate position along y-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oPosZ", "Oscillate Translate Z", "Oscillate position along z-axis", oscillateGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orPosZ", "Oscillate Translate Z Rate", "Rate to Oscillate position along z-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opPosZ", "Oscillate Translate Z Phase", "Phase offset for Oscillate position along z-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oRotX", "Oscillate Rotate X", "Oscillate Rotation about x-axis", oscillateGroup, -360000., 0., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orRotX", "Oscillate Rotate X Rate", "Rate to Oscillate Rotation about x-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opRotX", "Oscillate Rotate X Phase", "Phase offset for Oscillate Rotation about x-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oRotY", "Oscillate Rotate Y", "Oscillate Rotation about y-axis", oscillateGroup, -360000., 0., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orRotY", "Oscillate Rotate Y Rate", "Rate to Oscillate Rotation about y-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opRotY", "Oscillate Rotate Y Phase", "Phase offset for Oscillate Rotation about y-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oRotZ", "Oscillate Rotate Z", "Oscillate Rotation about z-axis", oscillateGroup, -360000., 0., 0., 360., 360000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orRotZ", "Oscillate Rotate Z Rate", "Rate to Oscillate Rotation about z-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opRotZ", "Oscillate Rotate Z Phase", "Phase offset for Oscillate Rotation about z-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oSclX", "Oscillate Scale X", "Oscillate Scale along x-axis", oscillateGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orSclX", "Oscillate Scale X Rate", "Rate to Oscillate Scale along x-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opSclX", "Oscillate Scale X Phase", "Phase offset for Oscillate Scale along x-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oSclY", "Oscillate Scale Y", "Oscillate Scale along y-axis", oscillateGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orSclY", "Oscillate Scale Y Rate", "Rate to Oscillate Scale along y-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opSclY", "Oscillate Scale Y Phase", "Phase offset for Oscillate Scale along y-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oSclZ", "Oscillate Scale Z", "Oscillate Scale along z-axis", oscillateGroup, -100., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orSclZ", "Oscillate Scale Z Rate", "Rate to Oscillate Scale along z-axis", oscillateGroup, 0., 0., 0., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "opSclZ", "Oscillate Scale Z Phase", "Phase offset for Oscillate Scale along z-axis", oscillateGroup, -100., -5., 0., 5., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "oGlobal", "Oscillate Global Strength", "Global Oscillate factor", oscillateGroup, -100., 0., 1., 1., 100.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "orGlobal", "Oscillate Global Rate", "Global Oscillate rate", oscillateGroup, 0., 0., 1., 10., 1000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "mBlur", "Motion Blur", "Motion blur", temporalGroup, 0., 0., 0., 1., 5.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "tempo", "Temporal Factor", "Temporal resistance factor", temporalGroup, 0., 0., 1., 1., 1.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "elast", "Elasticity", "Elasticity factor", temporalGroup, 0., 0., 0., 5., 50.);
    page->addChild(*param);
    BooleanParamDescriptor *boolParam = newBoolParam(p_Desc, "bounce", "Toggle Bouncing", "Toggle bouncing", temporalGroup, false);
    page->addChild(*boolParam);
    intParam = newIntParam(p_Desc, "hasParent", "Parent Index", "Index of parent transforms instance", miscGroup, 0, 0, 0, 16, 16);
    page->addChild(*intParam);
    intParam = newIntParam(p_Desc, "isParent", "Send Index", "Index of transforms instance to send to", miscGroup, 0, 0, 0, 16, 16);
    page->addChild(*intParam);
    ChoiceParamDescriptor* choiceParam = newChoiceParam(p_Desc, "wrapX", "Wrap X", "Method to wrap image along x-axis", miscGroup);
    choiceParam->appendOption("None");
    choiceParam->appendOption("Clamp");
    choiceParam->appendOption("Repeat");
    choiceParam->appendOption("Reflect");
    page->addChild(*choiceParam);
    choiceParam = newChoiceParam(p_Desc, "wrapY", "Wrap Y", "Method to wrap image along y-axis", miscGroup);
    choiceParam->appendOption("None");
    choiceParam->appendOption("Clamp");
    choiceParam->appendOption("Repeat");
    choiceParam->appendOption("Reflect");
    page->addChild(*choiceParam);
    boolParam = newBoolParam(p_Desc, "wrapZ", "Wrap Above Horizon", "Wrap image above horizon", miscGroup, false);
    page->addChild(*boolParam);
    boolParam = newBoolParam(p_Desc, "renderToggle", "Toggle Render", "Output rendered image", miscGroup, true);
    page->addChild(*boolParam);
    boolParam = newBoolParam(p_Desc, "backToggle", "Render Backface", "Render backface", miscGroup, true);
    page->addChild(*boolParam);
}

ImageEffect* TransformGPUFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new TransformGPU(p_Handle);
}

