#ifndef _CROSSBAR_H
#define _CROSSBAR_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <random>
#include "config.h"

#include "../../intel/compilers_and_libraries_2019.3.199/linux/mkl/include/mkl.h"
#include "../../intel/compilers_and_libraries_2019.3.199/linux/mkl/include/mkl_vsl.h"
#include "../../intel/compilers_and_libraries_2019.3.199/linux/mkl/examples/vslc/source/errcheck.inc"


using namespace std;

#define LAYER 19

typedef struct Crossbar
{
    float *std_d;
/*    int CB_l;
    int CB_w;
    int CB_n;
  */  float *CB_cell[LAYER+1];		// # of layers, start from 1
    float *input  [LAYER+1];
    float *output [LAYER+1];
    float *CB_std [LAYER+1];
    float *CB_comp[LAYER+1];
    int sizes     [LAYER+1][3];

    float **a_array  ;
    float **b_array  ;
    float **c_array  ;

    VSLStreamStatePtr stream;
    VSLStreamStatePtr streamS[56];	// # of max cpu cores	
    int S;	// # of omp parallel cpu cores
    float * noise;

//    Crossbar(){}

    Crossbar(/* int n, int l, int w */ ){
/*        CB_n = n;
        CB_l = l;
        CB_w = w;
*/	int sizes_tmp[LAYER+1][3] = {
0,0,0,
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 	// 5
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W,	//10 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W, 	//15
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W_512, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L, CROSSBAR_W_4096, 
(AD_WIDTH/DA_WIDTH), CROSSBAR_L_4608, CROSSBAR_W_2048,
(AD_WIDTH/DA_WIDTH), CROSSBAR_L_2304, CROSSBAR_W, 
};
	memcpy( & sizes[0][0],   &sizes_tmp[0][0], sizeof( sizes) );
	CB_cell[0] = CB_std[0] = input[0] = output[0] = CB_comp[0] = NULL;
	int max_noise = 0;
	for(int i=1; i<=LAYER; i++)
	{
		CB_cell[i] = new float[ sizes[i][1] * sizes[i][2] ];
		CB_std [i] = new float[ sizes[i][1] * sizes[i][2] ];
		CB_comp[i] = new float[ sizes[i][1] * sizes[i][2] ];
		input  [i] = new float[ sizes[i][0] * sizes[i][1] ];
		output [i] = new float[ sizes[i][0] * sizes[i][2] ];
		max_noise += sizes[i][1] * sizes[i][2];
	}
	a_array=new float*[ LAYER ];       // A
	b_array=new float*[ LAYER ];       // B
	c_array=new float*[ LAYER ];       // Ca


    /***** Initialize noise *****/

#define SEED    10007
#define BRNG    VSL_BRNG_MCG31
#define METHOD  VSL_RNG_METHOD_GAUSSIAN_ICDF
    	int errcode = vslNewStream (&stream, BRNG, SEED);
    	CheckVslError (errcode);
	noise = new float[ max_noise  ] ; 

	S = omp_get_max_threads() ;
    for (int i = 0; i < S; i++)
      {
        errcode = vslCopyStream (&streamS[i], stream);
        CheckVslError (errcode);
//        errcode = vslSkipAheadStream (streamS[i], (long long) (i * NS));
//        CheckVslError (errcode);
// need to exec later
      }

    }



    ~Crossbar(){
	for (int i=1;i<=LAYER;i++)
	{
         delete []CB_cell[i];
         delete []CB_std [i];
         delete []CB_comp[i];
         delete []input  [i];
         delete []output [i];
	}
	delete []a_array;
	delete []b_array;
	delete []c_array;
    	int errcode = vslDeleteStream (&stream);
    	CheckVslError (errcode);
	
    }

    void init(){
        get_std();
    }

    void get_std(){

        float max_conductance = 40;  // 25k ohm
	float soso = 1/ max_conductance;
	for (int i=1 ; i<=LAYER ;i++)
		for(int j=0; j< sizes[i][1] * sizes[i][2] ; j++) {
			float tmp = fabsf( CB_cell[i][j] );
			float tmp2= (tmp * max_conductance + 4);
			CB_std[i][j] = (-0.0006034 * (tmp2) * (tmp2) + 0.06184 * (tmp2) + 0.7240) * soso ;


// 	(-0.0006034 * (tmp * max_conductance + 4) * (tmp * max_conductance + 4) + 0.06184 * (tmp * max_conductance + 4) + 0.7240) / max_conductance;

		}
/*
        for (int i = 0; i < CB_w; ++i) {
            for (int j = 0; j < CB_l; ++j) {
                float tmp = fabsf(CB_cell[i*CB_l+j]);
                CB_std[i*CB_l+j] = (-0.0006034 * (tmp * max_conductance + 4) * (tmp * max_conductance + 4)
                                    + 0.06184 * (tmp * max_conductance + 4) + 0.7240) / max_conductance;
            }
        }
*/
    }

    void run(){
extern int use_noise;
extern int layer_comp_who[LAYER + 1];  // 15 conv and 4 linear crossbar total, but define all
if ( use_noise) {
	/******** generate noise ************/
	{
	        int num_noise = 0;
	        for(int i=1; i<=LAYER; i++)
		  if (layer_comp_who[i] )
	            {
	                num_noise += sizes[i][1] * sizes[i][2];
	            }
		int NS = num_noise / S;
		for(int i=0;i <S; i++)
		{
		        int errcode = vslSkipAheadStream (streamS[i], (long long) (i * NS));
		        CheckVslError (errcode);
		}

	      #pragma omp parallel for
	        for (int i = 0; i < S; i++)     // 并行生成随机数
	          {
	            int errcode =
	              vsRngGaussian (METHOD, streamS[i], NS, &noise[i * NS], 0, 1);
	            CheckVslError (errcode);
          }
	
	}
	
	/*********** add noise ****************/
	{
		int k=0;
	        for(int i=1; i<=LAYER; i++)
	          if (layer_comp_who[i] )
		    for(int j=0; j< sizes[i][1] * sizes[i][2]; j++)
		    {
			CB_comp[i][j] = CB_cell[i][j] + CB_std[i][j] * noise[k];
			k++;
	            }
	}
}

extern int batch_num;
if ( batch_num ) {
	int tot_conv=0;
	for(int i=1;i<=15;i++)	// total 15 conv layers
	  if (layer_comp_who[i] )
	    {
		a_array[ tot_conv ] = input[i];
		b_array[ tot_conv ] = use_noise ? CB_comp[i] : CB_cell[i];
		c_array[ tot_conv ] = output[i];
		tot_conv ++;
	    }
	
	{ // batch gemm
	      CBLAS_LAYOUT    layout = CblasRowMajor;
	      CBLAS_TRANSPOSE trans_tmp =CblasNoTrans, *transA=&trans_tmp ,  *transB=&trans_tmp;
	      float alpha_tmp=1,  *alpha=&alpha_tmp;
	      float beta_tmp=0 ,  *beta = &beta_tmp;
	      int * sizea = & sizes[ 1 ][0];	// 1-15 have the same sizes, so choose 1 as example
	      int * sizeb = & sizes[ 1 ][1];
	      int * sizec = & sizes[ 1 ][2];
	
	      int grp_sizes[1];
	      grp_sizes[0] = tot_conv ; 
	      cblas_sgemm_batch(layout, transA, transB, sizea, sizec, sizeb, alpha,
	            (const float **) a_array, sizeb, (const float **) b_array, sizec,
	            beta, c_array, sizec, 1, grp_sizes);
	} // batch gemm
	
	
	
	for(int i=16;i<=LAYER;i++)	// total 15 conv layers
	if ( layer_comp_who[i] )
	{
	      CBLAS_LAYOUT layout = CblasRowMajor;
	      CBLAS_TRANSPOSE transA = CblasNoTrans, transB = CblasNoTrans;
	      float alpha = 1, beta = 0;
	      int sizea = sizes[i][0];
	      int sizeb = sizes[i][1];
	      int sizec = sizes[i][2];
	        cblas_sgemm (layout, transA, transB, sizea, sizec, sizeb,
	                 alpha, input[i], sizeb,
	                 CB_cell[i], sizec, beta,
	                 output[i], sizec);
	}
}
else	// BATCH_MATMUL
{
	for(int i=1;i<=LAYER;i++)
	if ( layer_comp_who[i] )
	{
	      CBLAS_LAYOUT layout = CblasRowMajor;
	      CBLAS_TRANSPOSE transA = CblasNoTrans, transB = CblasNoTrans;
	      float alpha = 1, beta = 0;
	      int sizea = sizes[i][0];
	      int sizeb = sizes[i][1];
	      int sizec = sizes[i][2];
	        cblas_sgemm (layout, transA, transB, sizea, sizec, sizeb,
	                 alpha, input[i], sizeb,
	                 CB_cell[i], sizec, beta,
	                 output[i], sizec);
	}
}	// BATCH_MATMUL

    }

}CROSSBAR;

CROSSBAR entire_cb /* ( 1, ENTIRE_L, ENTIRE_W ) */ ;

#endif // !_CROSSBAR_H
