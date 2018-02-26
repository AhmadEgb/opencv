#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_mulSpectrums( InputArray _srcA, InputArray _srcB,
                              OutputArray _dst, int flags, bool conjB )
{
    int atype = _srcA.type(), btype = _srcB.type(),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    Size asize = _srcA.size(), bsize = _srcB.size();
    CV_Assert(asize == bsize);

    if ( !(atype == CV_32FC2 && btype == CV_32FC2) || flags != 0 )
        return false;

    UMat A = _srcA.getUMat(), B = _srcB.getUMat();
    CV_Assert(A.size() == B.size());

    _dst.create(A.size(), atype);
    UMat dst = _dst.getUMat();

    ocl::Kernel k("mulAndScaleSpectrums",
                  ocl::core::mulspectrums_oclsrc,
                  format("%s", conjB ? "-D CONJ" : ""));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(A), ocl::KernelArg::ReadOnlyNoSize(B),
           ocl::KernelArg::WriteOnly(dst), rowsPerWI);

    size_t globalsize[2] = { (size_t)asize.width, ((size_t)asize.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

namespace {

#define VAL(buf, elem) (((T*)((char*)data ## buf + (step ## buf * (elem))))[0])
#define MUL_SPECTRUMS_COL(A, B, C) \
    VAL(C, 0) = VAL(A, 0) * VAL(B, 0); \
    for (size_t j = 1; j <= rows - 2; j += 2) \
    { \
        double a_re = VAL(A, j), a_im = VAL(A, j + 1); \
        double b_re = VAL(B, j), b_im = VAL(B, j + 1); \
        if (conjB) b_im = -b_im; \
        double c_re = a_re * b_re - a_im * b_im; \
        double c_im = a_re * b_im + a_im * b_re; \
        VAL(C, j) = (T)c_re; VAL(C, j + 1) = (T)c_im; \
    } \
    if ((rows & 1) == 0) \
        VAL(C, rows-1) = VAL(A, rows-1) * VAL(B, rows-1)

template <typename T, bool conjB> static inline
void mulSpectrums_processCol_noinplace(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows)
{
    MUL_SPECTRUMS_COL(A, B, C);
}

template <typename T, bool conjB> static inline
void mulSpectrums_processCol_inplaceA(const T* dataB, T* dataAC, size_t stepB, size_t stepAC, size_t rows)
{
    MUL_SPECTRUMS_COL(AC, B, AC);
}
template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processCol(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows)
{
    if (inplaceA)
        mulSpectrums_processCol_inplaceA<T, conjB>(dataB, dataC, stepB, stepC, rows);
    else
        mulSpectrums_processCol_noinplace<T, conjB>(dataA, dataB, dataC, stepA, stepB, stepC, rows);
}
#undef MUL_SPECTRUMS_COL
#undef VAL

template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processCols(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols)
{
    mulSpectrums_processCol<T, conjB, inplaceA>(dataA, dataB, dataC, stepA, stepB, stepC, rows);
    if ((cols & 1) == 0)
    {
        mulSpectrums_processCol<T, conjB, inplaceA>(dataA + cols - 1, dataB + cols - 1, dataC + cols - 1, stepA, stepB, stepC, rows);
    }
}

#define VAL(buf, elem) (data ## buf[(elem)])
#define MUL_SPECTRUMS_ROW(A, B, C) \
    for (size_t j = j0; j < j1; j += 2) \
    { \
        double a_re = VAL(A, j), a_im = VAL(A, j + 1); \
        double b_re = VAL(B, j), b_im = VAL(B, j + 1); \
        if (conjB) b_im = -b_im; \
        double c_re = a_re * b_re - a_im * b_im; \
        double c_im = a_re * b_im + a_im * b_re; \
        VAL(C, j) = (T)c_re; VAL(C, j + 1) = (T)c_im; \
    }
template <typename T, bool conjB> static inline
void mulSpectrums_processRow_noinplace(const T* dataA, const T* dataB, T* dataC, size_t j0, size_t j1)
{
    MUL_SPECTRUMS_ROW(A, B, C);
}
template <typename T, bool conjB> static inline
void mulSpectrums_processRow_inplaceA(const T* dataB, T* dataAC, size_t j0, size_t j1)
{
    MUL_SPECTRUMS_ROW(AC, B, AC);
}
template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processRow(const T* dataA, const T* dataB, T* dataC, size_t j0, size_t j1)
{
    if (inplaceA)
        mulSpectrums_processRow_inplaceA<T, conjB>(dataB, dataC, j0, j1);
    else
        mulSpectrums_processRow_noinplace<T, conjB>(dataA, dataB, dataC, j0, j1);
}
#undef MUL_SPECTRUMS_ROW
#undef VAL

template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processRows(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols, size_t j0, size_t j1, bool is_1d_CN1)
{
    while (rows-- > 0)
    {
        if (is_1d_CN1)
            dataC[0] = dataA[0]*dataB[0];
        mulSpectrums_processRow<T, conjB, inplaceA>(dataA, dataB, dataC, j0, j1);
        if (is_1d_CN1 && (cols & 1) == 0)
            dataC[j1] = dataA[j1]*dataB[j1];

        dataA = (const T*)(((char*)dataA) + stepA);
        dataB = (const T*)(((char*)dataB) + stepB);
        dataC =       (T*)(((char*)dataC) + stepC);
    }
}


template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_Impl_(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols, size_t j0, size_t j1, bool is_1d, bool isCN1)
{
    if (!is_1d && isCN1)
    {
        mulSpectrums_processCols<T, conjB, inplaceA>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols);
    }
    mulSpectrums_processRows<T, conjB, inplaceA>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols, j0, j1, is_1d && isCN1);
}
template <typename T, bool conjB> static inline
void mulSpectrums_Impl(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols, size_t j0, size_t j1, bool is_1d, bool isCN1)
{
    if (dataA == dataC)
        mulSpectrums_Impl_<T, conjB, true>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols, j0, j1, is_1d, isCN1);
    else
        mulSpectrums_Impl_<T, conjB, false>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols, j0, j1, is_1d, isCN1);
}

} // namespace

void cv::mulSpectrums( InputArray _srcA, InputArray _srcB,
                       OutputArray _dst, int flags, bool conjB )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_dst.isUMat() && _srcA.dims() <= 2 && _srcB.dims() <= 2,
            ocl_mulSpectrums(_srcA, _srcB, _dst, flags, conjB))

    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    size_t rows = srcA.rows, cols = srcA.cols;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    Mat dst = _dst.getMat();

    // correct inplace support
    // Case 'dst.data == srcA.data' is handled by implementation,
    // because it is used frequently (filter2D, matchTemplate)
    if (dst.data == srcB.data)
        srcB = srcB.clone(); // workaround for B only

    bool is_1d = (flags & DFT_ROWS)
        || (rows == 1)
        || (cols == 1 && srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous());

    if( is_1d && !(flags & DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    bool isCN1 = cn == 1;
    size_t j0 = isCN1 ? 1 : 0;
    size_t j1 = cols*cn - (((cols & 1) == 0 && cn == 1) ? 1 : 0);

    if (depth == CV_32F)
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        if (!conjB)
            mulSpectrums_Impl<float, false>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
        else
            mulSpectrums_Impl<float, true>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        if (!conjB)
            mulSpectrums_Impl<double, false>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
        else
            mulSpectrums_Impl<double, true>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
    }
}
