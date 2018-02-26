#include "precomp.hpp"
#include "opencv2/core/opencl/runtime/opencl_clamdfft.hpp"
#include "opencv2/core/opencl/runtime/opencl_core.hpp"
#include "opencl_kernels_core.hpp"
#include "dxt.hpp"

/****************************************************************************************\
                               Discrete Cosine Transform
\****************************************************************************************/

namespace cv
{


static void
DCTInit( int n, int elem_size, void* _wave, int inv )
{
    static const double DctScale[] =
    {
    0.707106781186547570, 0.500000000000000000, 0.353553390593273790,
    0.250000000000000000, 0.176776695296636890, 0.125000000000000000,
    0.088388347648318447, 0.062500000000000000, 0.044194173824159223,
    0.031250000000000000, 0.022097086912079612, 0.015625000000000000,
    0.011048543456039806, 0.007812500000000000, 0.005524271728019903,
    0.003906250000000000, 0.002762135864009952, 0.001953125000000000,
    0.001381067932004976, 0.000976562500000000, 0.000690533966002488,
    0.000488281250000000, 0.000345266983001244, 0.000244140625000000,
    0.000172633491500622, 0.000122070312500000, 0.000086316745750311,
    0.000061035156250000, 0.000043158372875155, 0.000030517578125000
    };

    int i;
    Complex<double> w, w1;
    double t, scale;

    if( n == 1 )
        return;

    assert( (n&1) == 0 );

    if( (n & (n - 1)) == 0 )
    {
        int m;
        for( m = 0; (unsigned)(1 << m) < (unsigned)n; m++ )
            ;
        scale = (!inv ? 2 : 1)*DctScale[m];
        w1 = getDFTTab(m+2);
    }
    else
    {
        t = 1./(2*n);
        scale = (!inv ? 2 : 1)*std::sqrt(t);
        w1.im = sin(-CV_PI*t);
        w1.re = std::sqrt(1. - w1.im*w1.im);
    }
    n >>= 1;

    if( elem_size == sizeof(Complex<double>) )
    {
        Complex<double>* wave = (Complex<double>*)_wave;

        w.re = scale;
        w.im = 0.;

        for( i = 0; i <= n; i++ )
        {
            wave[i] = w;
            t = w.re*w1.re - w.im*w1.im;
            w.im = w.re*w1.im + w.im*w1.re;
            w.re = t;
        }
    }
    else
    {
        Complex<float>* wave = (Complex<float>*)_wave;
        assert( elem_size == sizeof(Complex<float>) );

        w.re = (float)scale;
        w.im = 0.f;

        for( i = 0; i <= n; i++ )
        {
            wave[i].re = (float)w.re;
            wave[i].im = (float)w.im;
            t = w.re*w1.re - w.im*w1.im;
            w.im = w.re*w1.im + w.im*w1.re;
            w.re = t;
        }
    }
}


typedef void (*DCTFunc)(const OcvDftOptions & c, const void* src, size_t src_step, void* dft_src,
                        void* dft_dst, void* dst, size_t dst_step, const void* dct_wave);

static void DCT_32f(const OcvDftOptions & c, const float* src, size_t src_step, float* dft_src, float* dft_dst,
                    float* dst, size_t dst_step, const Complexf* dct_wave)
{
    DCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

static void IDCT_32f(const OcvDftOptions & c, const float* src, size_t src_step, float* dft_src, float* dft_dst,
                    float* dst, size_t dst_step, const Complexf* dct_wave)
{
    IDCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

static void DCT_64f(const OcvDftOptions & c, const double* src, size_t src_step, double* dft_src, double* dft_dst,
                    double* dst, size_t dst_step, const Complexd* dct_wave)
{
    DCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

static void IDCT_64f(const OcvDftOptions & c, const double* src, size_t src_step, double* dft_src, double* dft_dst,
                     double* dst, size_t dst_step, const Complexd* dct_wave)
{
    IDCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

}

#ifdef HAVE_IPP
namespace cv
{

#if IPP_VERSION_X100 >= 900
typedef IppStatus (CV_STDCALL * ippiDCTFunc)(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, const void* pDCTSpec, Ipp8u* pBuffer);
typedef IppStatus (CV_STDCALL * ippiDCTInit)(void* pDCTSpec, IppiSize roiSize, Ipp8u* pMemInit );
typedef IppStatus (CV_STDCALL * ippiDCTGetSize)(IppiSize roiSize, int* pSizeSpec, int* pSizeInit, int* pSizeBuf);
#elif IPP_VERSION_X100 >= 700
typedef IppStatus (CV_STDCALL * ippiDCTFunc)(const Ipp32f*, int, Ipp32f*, int, const void*, Ipp8u*);
typedef IppStatus (CV_STDCALL * ippiDCTInitAlloc)(void**, IppiSize, IppHintAlgorithm);
typedef IppStatus (CV_STDCALL * ippiDCTFree)(void* pDCTSpec);
typedef IppStatus (CV_STDCALL * ippiDCTGetBufSize)(const void*, int*);
#endif

class DctIPPLoop_Invoker : public ParallelLoopBody
{
public:
    DctIPPLoop_Invoker(const uchar * _src, size_t _src_step, uchar * _dst, size_t _dst_step, int _width, bool _inv, bool *_ok) :
        ParallelLoopBody(), src(_src), src_step(_src_step), dst(_dst), dst_step(_dst_step), width(_width), inv(_inv), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const
    {
        if(*ok == false)
            return;

#if IPP_VERSION_X100 >= 900
        IppiSize srcRoiSize = {width, 1};

        int specSize    = 0;
        int initSize    = 0;
        int bufferSize  = 0;

        Ipp8u* pDCTSpec = NULL;
        Ipp8u* pBuffer  = NULL;
        Ipp8u* pInitBuf = NULL;

        #define IPP_RETURN              \
            if(pDCTSpec)                \
                ippFree(pDCTSpec);      \
            if(pBuffer)                 \
                ippFree(pBuffer);       \
            if(pInitBuf)                \
                ippFree(pInitBuf);      \
            return;

        ippiDCTFunc     ippiDCT_32f_C1R   = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R         : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInit     ippDctInit     = inv ? (ippiDCTInit)ippiDCTInvInit_32f         : (ippiDCTInit)ippiDCTFwdInit_32f;
        ippiDCTGetSize  ippDctGetSize  = inv ? (ippiDCTGetSize)ippiDCTInvGetSize_32f   : (ippiDCTGetSize)ippiDCTFwdGetSize_32f;

        if(ippDctGetSize(srcRoiSize, &specSize, &initSize, &bufferSize) < 0)
        {
            *ok = false;
            return;
        }

        pDCTSpec = (Ipp8u*)CV_IPP_MALLOC(specSize);
        if(!pDCTSpec && specSize)
        {
            *ok = false;
            return;
        }

        pBuffer  = (Ipp8u*)CV_IPP_MALLOC(bufferSize);
        if(!pBuffer && bufferSize)
        {
            *ok = false;
            IPP_RETURN
        }
        pInitBuf = (Ipp8u*)CV_IPP_MALLOC(initSize);
        if(!pInitBuf && initSize)
        {
            *ok = false;
            IPP_RETURN
        }

        if(ippDctInit(pDCTSpec, srcRoiSize, pInitBuf) < 0)
        {
            *ok = false;
            IPP_RETURN
        }

        for(int i = range.start; i < range.end; ++i)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippiDCT_32f_C1R, (float*)(src + src_step * i), static_cast<int>(src_step), (float*)(dst + dst_step * i), static_cast<int>(dst_step), pDCTSpec, pBuffer) < 0)
            {
                *ok = false;
                IPP_RETURN
            }
        }
        IPP_RETURN
#undef IPP_RETURN
#elif IPP_VERSION_X100 >= 700
        void* pDCTSpec;
        AutoBuffer<uchar> buf;
        uchar* pBuffer = 0;
        int bufSize=0;

        IppiSize srcRoiSize = {width, 1};

        CV_SUPPRESS_DEPRECATED_START

        ippiDCTFunc ippDctFun           = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R             : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInitAlloc ippInitAlloc   = inv ? (ippiDCTInitAlloc)ippiDCTInvInitAlloc_32f   : (ippiDCTInitAlloc)ippiDCTFwdInitAlloc_32f;
        ippiDCTFree ippFree             = inv ? (ippiDCTFree)ippiDCTInvFree_32f             : (ippiDCTFree)ippiDCTFwdFree_32f;
        ippiDCTGetBufSize ippGetBufSize = inv ? (ippiDCTGetBufSize)ippiDCTInvGetBufSize_32f : (ippiDCTGetBufSize)ippiDCTFwdGetBufSize_32f;

        if (ippInitAlloc(&pDCTSpec, srcRoiSize, ippAlgHintNone)>=0 && ippGetBufSize(pDCTSpec, &bufSize)>=0)
        {
            buf.allocate( bufSize );
            pBuffer = (uchar*)buf;

            for( int i = range.start; i < range.end; ++i)
            {
                if(ippDctFun((float*)(src + src_step * i), static_cast<int>(src_step), (float*)(dst + dst_step * i), static_cast<int>(dst_step), pDCTSpec, (Ipp8u*)pBuffer) < 0)
                {
                    *ok = false;
                    break;
                }
            }
        }
        else
            *ok = false;

        if (pDCTSpec)
            ippFree(pDCTSpec);

        CV_SUPPRESS_DEPRECATED_END
#else
        CV_UNUSED(range);
        *ok = false;
#endif
    }

private:
    const uchar * src;
    size_t src_step;
    uchar * dst;
    size_t dst_step;
    int width;
    bool inv;
    bool *ok;
};

static bool DctIPPLoop(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, bool inv)
{
    bool ok;
    parallel_for_(Range(0, height), DctIPPLoop_Invoker(src, src_step, dst, dst_step, width, inv, &ok), height/(double)(1<<4) );
    return ok;
}

static bool ippi_DCT_32f(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, bool inv, bool row)
{
    CV_INSTRUMENT_REGION_IPP()

    if(row)
        return DctIPPLoop(src, src_step, dst, dst_step, width, height, inv);
    else
    {
#if IPP_VERSION_X100 >= 900
        IppiSize srcRoiSize = {width, height};

        int specSize    = 0;
        int initSize    = 0;
        int bufferSize  = 0;

        Ipp8u* pDCTSpec = NULL;
        Ipp8u* pBuffer  = NULL;
        Ipp8u* pInitBuf = NULL;

        #define IPP_RELEASE             \
            if(pDCTSpec)                \
                ippFree(pDCTSpec);      \
            if(pBuffer)                 \
                ippFree(pBuffer);       \
            if(pInitBuf)                \
                ippFree(pInitBuf);      \

        ippiDCTFunc     ippiDCT_32f_C1R      = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R         : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInit     ippDctInit     = inv ? (ippiDCTInit)ippiDCTInvInit_32f         : (ippiDCTInit)ippiDCTFwdInit_32f;
        ippiDCTGetSize  ippDctGetSize  = inv ? (ippiDCTGetSize)ippiDCTInvGetSize_32f   : (ippiDCTGetSize)ippiDCTFwdGetSize_32f;

        if(ippDctGetSize(srcRoiSize, &specSize, &initSize, &bufferSize) < 0)
            return false;

        pDCTSpec = (Ipp8u*)CV_IPP_MALLOC(specSize);
        if(!pDCTSpec && specSize)
            return false;

        pBuffer  = (Ipp8u*)CV_IPP_MALLOC(bufferSize);
        if(!pBuffer && bufferSize)
        {
            IPP_RELEASE
            return false;
        }
        pInitBuf = (Ipp8u*)CV_IPP_MALLOC(initSize);
        if(!pInitBuf && initSize)
        {
            IPP_RELEASE
            return false;
        }

        if(ippDctInit(pDCTSpec, srcRoiSize, pInitBuf) < 0)
        {
            IPP_RELEASE
            return false;
        }

        if(CV_INSTRUMENT_FUN_IPP(ippiDCT_32f_C1R, (float*)src, static_cast<int>(src_step), (float*)dst, static_cast<int>(dst_step), pDCTSpec, pBuffer) < 0)
        {
            IPP_RELEASE
            return false;
        }

        IPP_RELEASE
        return true;
#undef IPP_RELEASE
#elif IPP_VERSION_X100 >= 700
        IppStatus status;
        void* pDCTSpec;
        AutoBuffer<uchar> buf;
        uchar* pBuffer = 0;
        int bufSize=0;

        IppiSize srcRoiSize = {width, height};

        CV_SUPPRESS_DEPRECATED_START

        ippiDCTFunc ippDctFun           = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R             : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInitAlloc ippInitAlloc   = inv ? (ippiDCTInitAlloc)ippiDCTInvInitAlloc_32f   : (ippiDCTInitAlloc)ippiDCTFwdInitAlloc_32f;
        ippiDCTFree ippFree             = inv ? (ippiDCTFree)ippiDCTInvFree_32f             : (ippiDCTFree)ippiDCTFwdFree_32f;
        ippiDCTGetBufSize ippGetBufSize = inv ? (ippiDCTGetBufSize)ippiDCTInvGetBufSize_32f : (ippiDCTGetBufSize)ippiDCTFwdGetBufSize_32f;

        status = ippStsErr;

        if (ippInitAlloc(&pDCTSpec, srcRoiSize, ippAlgHintNone)>=0 && ippGetBufSize(pDCTSpec, &bufSize)>=0)
        {
            buf.allocate( bufSize );
            pBuffer = (uchar*)buf;

            status = ippDctFun((float*)src, static_cast<int>(src_step), (float*)dst, static_cast<int>(dst_step), pDCTSpec, (Ipp8u*)pBuffer);
        }

        if (pDCTSpec)
            ippFree(pDCTSpec);

        CV_SUPPRESS_DEPRECATED_END

        return status >= 0;
#else
        CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(inv); CV_UNUSED(row);
        return false;
#endif
    }
}
}
#endif

namespace cv {

class OcvDctImpl : public hal::DCT2D
{
public:
    OcvDftOptions opt;

    int _factors[34];
    AutoBuffer<uint> wave_buf;
    AutoBuffer<int> itab_buf;

    DCTFunc dct_func;
    bool isRowTransform;
    bool isInverse;
    bool isContinuous;
    int start_stage;
    int end_stage;
    int width;
    int height;
    int depth;

    void init(int _width, int _height, int _depth, int flags)
    {
        width = _width;
        height = _height;
        depth = _depth;
        isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
        isRowTransform = (flags & CV_HAL_DFT_ROWS) != 0;
        isContinuous = (flags & CV_HAL_DFT_IS_CONTINUOUS) != 0;
        static DCTFunc dct_tbl[4] =
        {
            (DCTFunc)DCT_32f,
            (DCTFunc)IDCT_32f,
            (DCTFunc)DCT_64f,
            (DCTFunc)IDCT_64f
        };
        dct_func = dct_tbl[(int)isInverse + (depth == CV_64F)*2];
        opt.nf = 0;
        opt.isComplex = false;
        opt.isInverse = false;
        opt.noPermute = false;
        opt.scale = 1.;
        opt.factors = _factors;

        if (isRowTransform || height == 1 || (width == 1 && isContinuous))
        {
            start_stage = end_stage = 0;
        }
        else
        {
            start_stage = (width == 1);
            end_stage = 1;
        }
    }
    void apply(const uchar *src, size_t src_step, uchar *dst, size_t dst_step)
    {
        CV_IPP_RUN(IPP_VERSION_X100 >= 700 && depth == CV_32F, ippi_DCT_32f(src, src_step, dst, dst_step, width, height, isInverse, isRowTransform))

        AutoBuffer<uchar> dct_wave;
        AutoBuffer<uchar> src_buf, dst_buf;
        uchar *src_dft_buf = 0, *dst_dft_buf = 0;
        int prev_len = 0;
        int elem_size = (depth == CV_32F) ? sizeof(float) : sizeof(double);
        int complex_elem_size = elem_size*2;

        for(int stage = start_stage ; stage <= end_stage; stage++ )
        {
            const uchar* sptr = src;
            uchar* dptr = dst;
            size_t sstep0, sstep1, dstep0, dstep1;
            int len, count;

            if( stage == 0 )
            {
                len = width;
                count = height;
                if( len == 1 && !isRowTransform )
                {
                    len = height;
                    count = 1;
                }
                sstep0 = src_step;
                dstep0 = dst_step;
                sstep1 = dstep1 = elem_size;
            }
            else
            {
                len = height;
                count = width;
                sstep1 = src_step;
                dstep1 = dst_step;
                sstep0 = dstep0 = elem_size;
            }

            opt.n = len;
            opt.tab_size = len;

            if( len != prev_len )
            {
                if( len > 1 && (len & 1) )
                    CV_Error( CV_StsNotImplemented, "Odd-size DCT\'s are not implemented" );

                opt.nf = DFTFactorize( len, opt.factors );
                bool inplace_transform = opt.factors[0] == opt.factors[opt.nf-1];

                wave_buf.allocate(len*complex_elem_size);
                opt.wave = wave_buf;
                itab_buf.allocate(len);
                opt.itab = itab_buf;
                DFTInit( len, opt.nf, opt.factors, opt.itab, complex_elem_size, opt.wave, isInverse );

                dct_wave.allocate((len/2 + 1)*complex_elem_size);
                src_buf.allocate(len*elem_size);
                src_dft_buf = src_buf;
                if(!inplace_transform)
                {
                    dst_buf.allocate(len*elem_size);
                    dst_dft_buf = dst_buf;
                }
                else
                {
                    dst_dft_buf = src_buf;
                }
                DCTInit( len, complex_elem_size, dct_wave, isInverse);
                prev_len = len;
            }
            // otherwise reuse the tables calculated on the previous stage
            for(unsigned i = 0; i < static_cast<unsigned>(count); i++ )
            {
                dct_func( opt, sptr + i*sstep0, sstep1, src_dft_buf, dst_dft_buf,
                          dptr + i*dstep0, dstep1, dct_wave);
            }
            src = dst;
            src_step = dst_step;
        }
    }
};

struct ReplacementDCT2D : public hal::DCT2D
{
    cvhalDFT *context;
    bool isInitialized;

    ReplacementDCT2D() : context(0), isInitialized(false) {}
    bool init(int width, int height, int depth, int flags)
    {
        int res = hal_ni_dctInit2D(&context, width, height, depth, flags);
        isInitialized = (res == CV_HAL_ERROR_OK);
        return isInitialized;
    }
    void apply(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step)
    {
        if (isInitialized)
        {
            CALL_HAL(dct2D, cv_hal_dct2D, context, src_data, src_step, dst_data, dst_step);
        }
    }
    ~ReplacementDCT2D()
    {
        if (isInitialized)
        {
            CALL_HAL(dctFree2D, cv_hal_dctFree2D, context);
        }
    }
};

namespace hal {

Ptr<DCT2D> DCT2D::create(int width, int height, int depth, int flags)
{
    {
        ReplacementDCT2D *impl = new ReplacementDCT2D();
        if (impl->init(width, height, depth, flags))
        {
            return Ptr<DCT2D>(impl);
        }
        delete impl;
    }
    {
        OcvDctImpl *impl = new OcvDctImpl();
        impl->init(width, height, depth, flags);
        return Ptr<DCT2D>(impl);
    }
}

} // cv::hal::
} // cv::

void cv::dct( InputArray _src0, OutputArray _dst, int flags )
{
    CV_INSTRUMENT_REGION()

    Mat src0 = _src0.getMat(), src = src0;
    int type = src.type(), depth = src.depth();

    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );
    _dst.create( src.rows, src.cols, type );
    Mat dst = _dst.getMat();

    int f = 0;
    if ((flags & DFT_ROWS) != 0)
        f |= CV_HAL_DFT_ROWS;
    if ((flags & DCT_INVERSE) != 0)
        f |= CV_HAL_DFT_INVERSE;
    if (src.isContinuous() && dst.isContinuous())
        f |= CV_HAL_DFT_IS_CONTINUOUS;

    Ptr<hal::DCT2D> c = hal::DCT2D::create(src.cols, src.rows, depth, f);
    c->apply(src.data, src.step, dst.data, dst.step);
}


void cv::idct( InputArray src, OutputArray dst, int flags )
{
    CV_INSTRUMENT_REGION()

    dct( src, dst, flags | DCT_INVERSE );
}
