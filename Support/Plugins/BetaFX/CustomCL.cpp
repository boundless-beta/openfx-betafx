// Copyright OpenFX and contributors to the OpenFX project.
// SPDX-License-Identifier: BSD-3-Clause

#pragma comment(lib, "opencl.lib")
#include "CustomCL.h"

#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <memory>
#include <thread>
#include <math.h>
#include <cmath>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#define kPluginName "BetaFX Custom OpenCL Kernel"
#define kPluginGrouping "BetaFX"
#define kPluginDescription "Use a custom OpenCL kernel as an effect"
#define kPluginIdentifier "betafx:CustomOpenCLKernel"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class CustomCL : public OFX::ImageProcessor
{
public:
    explicit CustomCL(OFX::ImageEffect& p_Instance);

    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(std::string k, double* floats, int instanceCount, float timeIn);

private:
    OFX::Image* _srcImg;
    std::string kernel;
    float kFloats[16];
    int instance;
    float time;
};

CustomCL::CustomCL(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

template<class PIX>
extern void RunOpenCLKernelBuffers(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const PIX* p_Input, PIX* p_Output);

template<class PIX>
extern void RunOpenCLKernelImages(void* p_CmdQ, int p_Width, int p_Height, std::string kernel, float* floats, int instance, float timeIn, int bits, const PIX* p_Input, PIX* p_Output);
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

void CustomCL::processImagesOpenCL()
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
            RunOpenCLKernelImages<unsigned char>(_pOpenCLCmdQ, width, height, kernel, kFloats, instance, time, bitDepth, inputUI, outputUI);
    } else if (inputF || outputF) {
            RunOpenCLKernelImages<float>(_pOpenCLCmdQ, width, height, kernel, kFloats, instance, time, bitDepth, inputF, outputF);
        }
    else if(bitDepth == 8)
    {
        inputUI = static_cast<unsigned char*>(_srcImg->getPixelData());
        outputUI = static_cast<unsigned char*>(_dstImg->getPixelData());

        RunOpenCLKernelBuffers<unsigned char>(_pOpenCLCmdQ, width, height, kernel, kFloats, instance, time, bitDepth, inputUI, outputUI);
}
    else
    {
        inputF = static_cast<float*>(_srcImg->getPixelData());
        outputF = static_cast<float*>(_dstImg->getPixelData());

        RunOpenCLKernelBuffers<float>(_pOpenCLCmdQ, width, height, kernel, kFloats, instance, time, bitDepth, inputF, outputF);
}
#endif
}

void CustomCL::multiThreadProcessImages(OfxRectI p_ProcWindow)
{ 
    int me = 0;
    /*
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
                for(int c = 0; c < 4; ++c)
                {
                    dstPix[c] = srcPix[c];
                }
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
    */
}

void CustomCL::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void CustomCL::setScales(std::string k, double* floats, int instanceCount, float timeIn)
{
    kernel = k;
    for (int i = 0; i < 16; i++) {
        kFloats[i] = floats[i];
    }
    instance = instanceCount;
    time = timeIn;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class CustomCLEffect : public OFX::ImageEffect
{
public:
    explicit CustomCLEffect(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Set up and run a processor */
    void setupAndProcess(CustomCL &p_CustomCL, const OFX::RenderArguments& p_Args);

    virtual void getClipPreferences(OFX::ClipPreferencesSetter& clipPreferences);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::StringParam* kernel;
    OFX::DoubleParam* float00;
    OFX::DoubleParam* float01;
    OFX::DoubleParam* float02;
    OFX::DoubleParam* float03;
    OFX::DoubleParam* float04;
    OFX::DoubleParam* float05;
    OFX::DoubleParam* float06;
    OFX::DoubleParam* float07;
    OFX::DoubleParam* float08;
    OFX::DoubleParam* float09;
    OFX::DoubleParam* float10;
    OFX::DoubleParam* float11;
    OFX::DoubleParam* float12;
    OFX::DoubleParam* float13;
    OFX::DoubleParam* float14;
    OFX::DoubleParam* float15;
    int instanceHandle;
};

static inline int imageQuery(OfxImageEffectHandle image)
{
    static std::map<OfxImageEffectHandle, int> threadIO;
    int theImage = 0;
    std::map<OfxImageEffectHandle, int>::iterator iter = threadIO.find(image);
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

CustomCLEffect::CustomCLEffect(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{ 
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    kernel = fetchStringParam("kInput");
    float00 = fetchDoubleParam("kFloat00");
    float01 = fetchDoubleParam("kFloat01");
    float02 = fetchDoubleParam("kFloat02");
    float03 = fetchDoubleParam("kFloat03");
    float04 = fetchDoubleParam("kFloat04");
    float05 = fetchDoubleParam("kFloat05");
    float06 = fetchDoubleParam("kFloat06");
    float07 = fetchDoubleParam("kFloat07");
    float08 = fetchDoubleParam("kFloat08");
    float09 = fetchDoubleParam("kFloat09");
    float10 = fetchDoubleParam("kFloat10");
    float11 = fetchDoubleParam("kFloat11");
    float12 = fetchDoubleParam("kFloat12");
    float13 = fetchDoubleParam("kFloat13");
    float14 = fetchDoubleParam("kFloat14");
    float15 = fetchDoubleParam("kFloat15");
    instanceHandle = imageQuery(p_Handle);
}

void CustomCLEffect::render(const OFX::RenderArguments& p_Args)
{
    bool bitCheck = m_DstClip->getPixelDepth() == OFX::eBitDepthUByte || m_DstClip->getPixelDepth() == OFX::eBitDepthFloat;
    if (bitCheck && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        CustomCL imageScaler(*this);
        setupAndProcess(imageScaler, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}


void CustomCLEffect::setupAndProcess(CustomCL& p_CustomCL, const OFX::RenderArguments& p_Args)
{
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

    double frameRate = m_SrcClip->getFrameRate();

    // this is where all of the parameters start
    double pFloats[16];
    std::string pKernel;
     kernel->getValueAtTime(p_Args.time, pKernel);
     pFloats[0] = float00->getValueAtTime(p_Args.time);
     pFloats[1] = float01->getValueAtTime(p_Args.time);
     pFloats[2] = float02->getValueAtTime(p_Args.time);
     pFloats[3] = float03->getValueAtTime(p_Args.time);
     pFloats[4] = float04->getValueAtTime(p_Args.time);
     pFloats[5] = float05->getValueAtTime(p_Args.time);
     pFloats[6] = float06->getValueAtTime(p_Args.time);
     pFloats[7] = float07->getValueAtTime(p_Args.time);
     pFloats[8] = float08->getValueAtTime(p_Args.time);
     pFloats[9] = float09->getValueAtTime(p_Args.time);
    pFloats[10] = float10->getValueAtTime(p_Args.time);
    pFloats[11] = float11->getValueAtTime(p_Args.time);
    pFloats[12] = float12->getValueAtTime(p_Args.time);
    pFloats[13] = float13->getValueAtTime(p_Args.time);
    pFloats[14] = float14->getValueAtTime(p_Args.time);
    pFloats[15] = float15->getValueAtTime(p_Args.time);
    // parameters end here

    p_CustomCL.setScales(pKernel, pFloats, instanceHandle, (p_Args.time / frameRate));

    // Set the images
    p_CustomCL.setDstImg(dst.get());
    p_CustomCL.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_CustomCL.setGPURenderArgs(p_Args);

    // Set the render window
    p_CustomCL.setRenderWindow(p_Args.renderWindow);

    // Call the base class process member, this will call the derived templated process code
    p_CustomCL.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

CustomCLEffectFactory::CustomCLEffectFactory()
    : OFX::PluginFactoryHelper<CustomCLEffectFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void CustomCLEffect::getClipPreferences(ClipPreferencesSetter& clipPreferences) {
    clipPreferences.setOutputFrameVarying(true);
}

void CustomCLEffectFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    p_Desc.setRenderThreadSafety(eRenderUnsafe);
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


void CustomCLEffectFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

    // Make overall scale params
    StringParamDescriptor* stringParam = p_Desc.defineStringParam("kInput");
    stringParam->setLabels("Script Input", "Script Input", "Script Input");
    stringParam->setScriptName("kInput");
    stringParam->setHint("Script input, use kReadIndex for reading from input image via kRead(). use kOutput as the color value stored at kReadIndex. use p_Width and p_Height for image dimensions.");
    stringParam->setDefault("kReadIndex = (float2)(x,y); kRead()");
    stringParam->setStringType(eStringTypeMultiLine);

    DoubleParamDescriptor* param = newDoubleParam(p_Desc, "kFloat00", "KFLOAT0", "Float 0", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat01", "KFLOAT1", "Float 1", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat02", "KFLOAT2", "Float 2", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat03", "KFLOAT3", "Float 3", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat04", "KFLOAT4", "Float 4", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat05", "KFLOAT5", "Float 5", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat06", "KFLOAT6", "Float 6", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat07", "KFLOAT7", "Float 7", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat08", "KFLOAT8", "Float 8", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat09", "KFLOAT9", "Float 9", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat10", "KFLOAT10", "Float 10", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat11", "KFLOAT11", "Float 11", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat12", "KFLOAT12", "Float 12", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat13", "KFLOAT13", "Float 13", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat14", "KFLOAT14", "Float 14", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
    param = newDoubleParam(p_Desc, "kFloat15", "KFLOAT15", "Float 15", 0, -100000., -1., 0., 1., 100000.);
    page->addChild(*param);
}

ImageEffect* CustomCLEffectFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new CustomCLEffect(p_Handle);
}
