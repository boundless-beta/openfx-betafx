# BetaFX Custom OpenCL Kernel Documentation

Below is a list of custom variables and their use in BetaFX Custom OpenCL Kernel.

*Updated as of 0.0.3*



**int p_Width**: image width in pixels

**int p_Height**: image height in pixels

**int x**: readonly, image x coordinate in pixels

**int y**: readonly, image y coordinate in pixels

**int index**: readonly, image pixel coordinates as a single number, equates to **((y \* p_Width) + x) \* 4**; for use in **kBuffer** arrays

- \+ 0 = red channel
- \+ 1 = green channel
- \+ 2 = blue channel
- \+ 3 = alpha channel

**float\* kBuffer{0-3}**: floating-point image buffers; can be read/written with **kRead({0-3})** or as an array (i.e. kBuffer0[index])

**float kTime**: time elapsed in seconds

**float kFloat{0-15}**: corresponds to the sliders below the code input box

**float2 kReadIndex**: image read coordinates used in **kRead()**

**float4 kOutput**: color value that is output when the kernel finishes

**kRead({0-3})**: keyword used to read the corresponding buffer using **kReadIndex** as coordinates and outputs to **kOutput**; using **kRead()** without a number reads source image

**kWrite({0-3})**: writes **kOutput** to the corresponding buffer; number required

**float\* kTransform** *(experimental)*: buffer containing transform matrix values sent from **BetaFX Dynamic Transform**, not the parameters used (not very useful by itself)

- size: 64 * 16 * 25 (thread, channel, parameter)
- of the 25 parameters:
   - 0 - 2: position XYZ
   - 3 - 11: 3x3 transform matrix
   - 12-23: same as 0-11 but one frame ahead (used for motion blur)
   - 24: motion blur amount

**int kThread** *(experimental)*: used in **BetaFX Dynamic Transform** to avoid flickering on multithread setups; required for properly reading **kTransform**

