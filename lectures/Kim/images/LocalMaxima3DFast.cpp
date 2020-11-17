/****************************************************************************
 *                                                                           *
 *  LocalMaxima3DFast version 1.0                                            *
 *  Copyright (C) 2011 Kim Steenstrup Pedersen (kimstp@diku.dk)              *
 *  Department of Computer Science, University of Copenhagen, Denmark        *
 *                                                                           *
 *  This file is part of the multi-scale Harris detector package version 1.0 *
 *                                                                           *
 *  The multi-scale Harris detector package is free software: you can        *
 *  redistribute it and/or modify it under the terms of the GNU Lesser       *
 *  General Public License as published by the Free Software Foundation,     * 
 *  either version 3 of the License, or (at your option) any later version.  *
 *                                                                           *
 *  The multi-scale Harris detector package is distributed in the hope that  *
 *  it will be useful, but WITHOUT ANY WARRANTY; without even the implied    *
 *  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the *
 *  GNU Lesser General Public License for more details.                      *
 *                                                                           *
 *  You should have received a copy of the GNU Lesser General Public License *
 *  along with the multi-scale Harris detector package. If not, see          *
 *  <http://www.gnu.org/licenses/>.                                          *
 *                                                                           *
 ****************************************************************************/

#include "mex.h"
#include <stdio.h>
#include <math.h>
#include "matrix.h"

/*
 * This is a MEX-file for MATLAB.
 * Compile using the build-in mex compiler.
 * 
 */



// Find maxima in 3D using 26 neighbors
void Maxima(double *I, double *M,  const int *dims)
{
  int idx;
  int idim   = dims[0];
  int jdim   = dims[1];
  int kdim   = dims[2];
  int offset = idim*jdim;
  
  for (int i=1; i<idim-1; i++) 
    for (int j=1; j<jdim-1; j++) 
      for (int k=0; k<kdim; k++) { // Allow maxima at the first scale level
          idx = i + j*idim + k*offset;
	
          if (k==0) // Handle first scale level
              if ((*(I + idx + offset) <= *(I+idx))                                             // + scale
                  && (*(I + idx + idim) <= *(I+idx)) && (*(I + idx - idim) <= *(I+idx))         // +/- column
                  && (*(I + idx + 1) <= *(I+idx)) && (*(I + idx - 1) <= *(I+idx))               // +/- row
                  && (*(I + idx - 1 - idim) <= *(I+idx)) && (*(I + idx + 1 - idim) <= *(I+idx)) // +/- row -column
                  && (*(I + idx - 1 + idim) <= *(I+idx)) && (*(I + idx + 1 + idim) <= *(I+idx)) // +/- row +column
                  
                  && (*(I + idx + idim + offset) <= *(I+idx)) && (*(I + idx - idim + offset) <= *(I+idx))         // +/- column + scale
                  && (*(I + idx + 1 + offset) <= *(I+idx)) && (*(I + idx - 1 + offset) <= *(I+idx))               // +/- row + scale
                  && (*(I + idx - 1 - idim + offset) <= *(I+idx)) && (*(I + idx + 1 - idim + offset) <= *(I+idx)) // +/- row -column + scale
                  && (*(I + idx - 1 + idim + offset) <= *(I+idx)) && (*(I + idx + 1 + idim + offset) <= *(I+idx)) // +/- row +column + scale
                  )
                  
                  *(M+idx) = *(I+idx);
              else
                  *(M+idx) = 0;
          else
              if (k==(kdim-1)) // Handle the last scale level
                  if ((*(I + idx - offset) <= *(I+idx))        // - scale
                      && (*(I + idx + idim) <= *(I+idx)) && (*(I + idx - idim) <= *(I+idx))         // +/- column
                      && (*(I + idx + 1) <= *(I+idx)) && (*(I + idx - 1) <= *(I+idx))               // +/- row
                      && (*(I + idx - 1 - idim) <= *(I+idx)) && (*(I + idx + 1 - idim) <= *(I+idx)) // +/- row -column
                      && (*(I + idx - 1 + idim) <= *(I+idx)) && (*(I + idx + 1 + idim) <= *(I+idx)) // +/- row +column
                      
                      && (*(I + idx + idim - offset) <= *(I+idx)) && (*(I + idx - idim - offset) <= *(I+idx))         // +/- column - scale
                      && (*(I + idx + 1 - offset) <= *(I+idx)) && (*(I + idx - 1 - offset) <= *(I+idx))               // +/- row - scale
                      && (*(I + idx - 1 - idim - offset) <= *(I+idx)) && (*(I + idx + 1 - idim - offset) <= *(I+idx)) // +/- row -column - scale
                      && (*(I + idx - 1 + idim - offset) <= *(I+idx)) && (*(I + idx + 1 + idim - offset) <= *(I+idx)) // +/- row +column - scale              
                      )
                      
                      *(M+idx) = *(I+idx);
                  else
                      *(M+idx) = 0;
              else // Handle all other scale levels
                  if ((*(I + idx + offset) <= *(I+idx)) && (*(I + idx - offset) <= *(I+idx))        // +/- scale
                      && (*(I + idx + idim) <= *(I+idx)) && (*(I + idx - idim) <= *(I+idx))         // +/- column
                      && (*(I + idx + 1) <= *(I+idx)) && (*(I + idx - 1) <= *(I+idx))               // +/- row
                      && (*(I + idx - 1 - idim) <= *(I+idx)) && (*(I + idx + 1 - idim) <= *(I+idx)) // +/- row -column
                      && (*(I + idx - 1 + idim) <= *(I+idx)) && (*(I + idx + 1 + idim) <= *(I+idx)) // +/- row +column
              
                      && (*(I + idx + idim + offset) <= *(I+idx)) && (*(I + idx - idim + offset) <= *(I+idx))         // +/- column + scale
                      && (*(I + idx + 1 + offset) <= *(I+idx)) && (*(I + idx - 1 + offset) <= *(I+idx))               // +/- row + scale
                      && (*(I + idx - 1 - idim + offset) <= *(I+idx)) && (*(I + idx + 1 - idim + offset) <= *(I+idx)) // +/- row -column + scale
                      && (*(I + idx - 1 + idim + offset) <= *(I+idx)) && (*(I + idx + 1 + idim + offset) <= *(I+idx)) // +/- row +column + scale

                      && (*(I + idx + idim - offset) <= *(I+idx)) && (*(I + idx - idim - offset) <= *(I+idx))         // +/- column - scale
                      && (*(I + idx + 1 - offset) <= *(I+idx)) && (*(I + idx - 1 - offset) <= *(I+idx))               // +/- row - scale
                      && (*(I + idx - 1 - idim - offset) <= *(I+idx)) && (*(I + idx + 1 - idim - offset) <= *(I+idx)) // +/- row -column - scale
                      && (*(I + idx - 1 + idim - offset) <= *(I+idx)) && (*(I + idx + 1 + idim - offset) <= *(I+idx)) // +/- row +column - scale              
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
  if (ndim != 3)
    mexErrMsgTxt("LocalMaxima3DFast only works for 3-dimensional matrices.");

  dims = mxGetDimensions(prhs[0]);
    
  if (dims[0] < 3 || dims[1] < 3 || dims[2] < 3)
      mexErrMsgTxt("LocalMaxima3DFast only works for 3-dimensional matrices with size 3 or larger.");
	   
  // Set the output pointer to the output matrix. Array initialized to zero. 
  plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
  
  // Create a C pointer to a copy of the output matrix.
  M = mxGetPr(plhs[0]);
  
  // Call the C subroutine.
  Maxima(L, M, dims);
}
