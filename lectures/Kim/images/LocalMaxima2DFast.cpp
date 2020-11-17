#include "mex.h"
#include <stdio.h>
#include <math.h>
#include "matrix.h"

/*
 * This is a MEX-file for MATLAB.
 * 
 */



// Find maxima in 2D using 8 neighbors
void Maxima(double *I, double *M,  const int *dims)
{
  int idx;
  int idim   = dims[0];
  int jdim   = dims[1];
  
  for (int i=1; i<idim-1; i++) 
      for (int j=1; j<jdim-1; j++) {
          idx = i + j*idim;
	
          // Finish this code to do a 8 neighbor comparison
          if ((*(I + idx + idim) <= *(I+idx)) && (*(I + idx - idim) <= *(I+idx))         // +/- column
              && (*(I + idx + 1) <= *(I+idx)) && (*(I + idx - 1) <= *(I+idx))               // +/- row
              && (*(I + idx - 1 - idim) <= *(I+idx)) && (*(I + idx + 1 - idim) <= *(I+idx)) // +/- row -column
              && (*(I + idx - 1 + idim) <= *(I+idx)) && (*(I + idx + 1 + idim) <= *(I+idx)) // +/- row +column
              )
              *(M+idx) = *(I+idx);
          else
              *(M+idx) = 0;
      }
}

// The gateway routine 
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  double *L, *M;
  double  x;
  int     status, ndim;
  const int *dims;
  /*  Check for proper number of arguments. */
  /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.) 
  */
  if(nrhs != 1) 
    mexErrMsgTxt("One input required.");
  if(nlhs != 1) 
    mexErrMsgTxt("One output required.");
    
  // Create a pointer to the input matrix.
  L = mxGetPr(prhs[0]);
  
  // Get the dimensions of the matrix input.
  ndim = mxGetNumberOfDimensions(prhs[0]);
  if (ndim != 2)
    mexErrMsgTxt("LocalMaxima2DFast only works for 2-dimensional matrices.");

  dims = mxGetDimensions(prhs[0]);
	   
  // Set the output pointer to the output matrix. Array initialized to zero. 
  plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
  
  // Create a C pointer to a copy of the output matrix.
  M = mxGetPr(plhs[0]);
  
  // Call the C subroutine.
  Maxima(L, M, dims);
}
