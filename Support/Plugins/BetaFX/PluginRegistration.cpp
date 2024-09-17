#include <stdio.h>
#include "ofxsImageEffect.h"
#include "TransformGPU.h"
#include "CustomCL.h"

namespace OFX 
{
  namespace Plugin 
  {
    void getPluginIDs(OFX::PluginFactoryArray &ids)
    {
      static TransformGPUFactory p1;
      ids.push_back(&p1);
      static CustomCLEffectFactory p2;
      ids.push_back(&p2);
    }
  }
}
