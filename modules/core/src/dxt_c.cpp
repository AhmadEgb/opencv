#include "precomp.hpp"

CV_IMPL void
cvDFT( const CvArr* srcarr, CvArr* dstarr, int flags, int nonzero_rows )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
    int _flags = ((flags & CV_DXT_INVERSE) ? cv::DFT_INVERSE : 0) |
        ((flags & CV_DXT_SCALE) ? cv::DFT_SCALE : 0) |
        ((flags & CV_DXT_ROWS) ? cv::DFT_ROWS : 0);

    CV_Assert( src.size == dst.size );

    if( src.type() != dst.type() )
    {
        if( dst.channels() == 2 )
            _flags |= cv::DFT_COMPLEX_OUTPUT;
        else
            _flags |= cv::DFT_REAL_OUTPUT;
    }

    cv::dft( src, dst, _flags, nonzero_rows );
    CV_Assert( dst.data == dst0.data ); // otherwise it means that the destination size or type was incorrect
}


CV_IMPL void
cvMulSpectrums( const CvArr* srcAarr, const CvArr* srcBarr,
                CvArr* dstarr, int flags )
{
    cv::Mat srcA = cv::cvarrToMat(srcAarr),
        srcB = cv::cvarrToMat(srcBarr),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( srcA.size == dst.size && srcA.type() == dst.type() );

    cv::mulSpectrums(srcA, srcB, dst,
        (flags & CV_DXT_ROWS) ? cv::DFT_ROWS : 0,
        (flags & CV_DXT_MUL_CONJ) != 0 );
}


CV_IMPL void
cvDCT( const CvArr* srcarr, CvArr* dstarr, int flags )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    int _flags = ((flags & CV_DXT_INVERSE) ? cv::DCT_INVERSE : 0) |
            ((flags & CV_DXT_ROWS) ? cv::DCT_ROWS : 0);
    cv::dct( src, dst, _flags );
}


CV_IMPL int
cvGetOptimalDFTSize( int size0 )
{
    return cv::getOptimalDFTSize(size0);
}
