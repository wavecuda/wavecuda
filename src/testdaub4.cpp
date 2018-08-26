#include "daub4.h"
#include "utils.h"
#include "thresh.h"

int main(void){
  real *x, *x1, *x2;
  real *xm, *xm1, *xm2;
  uint len, modlen;
  int res;
  int i, power;
  double t,ops,gflops;
  uint nlevels;
  
  printf("\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("\nDaubechies 4 Transform in C++ - serial & OpenMP\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("%-20s","Compiled:");
  printf("%s, ",__TIME__);
  printf("%s\n",__DATE__);
  printf("%-20s","Written by:");
  printf("%-20s","jw1408@ic.ac.uk\n");
  printf("\n");
  //  printf("FLT_MAX = %g",FLT_MAX);
  for(len=0;len<50;len++) printf("#");

  // ###########################################
  // A few variables to control the timing runs
  //
  int p0 = 3, p = 15; // first & last size in loop
  int reps = 1; // repetitions
  int modwt = 1; // running DWT or MODWT?
  int inverse = 1; // inverse transform too?
  uint levels = 0; // levels of transform
  //
  // ###########################################
  

  // ###################################################

  for(power = p0; power<=p; power++){
    len= 1 << power;
    printf("\nLength = %i, i.e. 2^%i\n",len,power);
    x=(real *)malloc(len*sizeof(real));
    x1=(real *)malloc(len*sizeof(real));
    x2=(real *)malloc(len*sizeof(real));
    initrandvec(x,len);
    // x[0]=0.; x[1]=0.; x[2]=5.; x[3]=4.;
    // x[4]=8.; x[5]=6.; x[6]=7.; x[7]=3.;
    copyvec(x,x1,len);
    copyvec(x,x2,len);
    timer(-1);

    nlevels = check_len_levels(len,levels,4);
    if(modwt) modlen = len*2*nlevels;

    if(modwt){
	xm=(real *)malloc(modlen*sizeof(real));
	xm1=(real *)malloc(modlen*sizeof(real));
	xm2=(real *)malloc(modlen*sizeof(real));
      }
    
    // printf("\n### Serial transform ### \n");
    // for(i=0;i<reps;i++){
    //   // #########_transform_one_#############################
    //   res=Daub4(x1,len,FWD,nlevels);
    //   printvec(x1,len);
    //   if(res!=0) printf("\n## Error in transform\n");
    //   // res=threshold(x1,len,univ_thresh(x1,len),HARD);
    //   // if(res!=0) printf("\n## Error in threshold\n");
    //   res=Daub4(x1,len,0,0);
    //   // #####################################################
    //   if(res!=0) printf("\n## Error in transform\n");
    //   t=timer(1);
    // }
    // ops=(double) (len -2) * (double) reps * 2. * 14.;
    // gflops=ops/(1e9*t);
    // printf("Time to do forward & backward transform: %gs\n",t/reps);
    // printf("Gigaflops: %g\n",gflops);
    // printvec(x1,len);



    // printf("\n### Serial transform - LIFTED ### \n");
    // timer(-1);
    // for(i=0;i<reps;i++){
    //   // #########_transform_two_#############################
    //   res=lDaub4(x2,len,1,0);
    //   if(res!=0) printf("\n## Error in transform\n");
    //   // res=threshold(x2,len,univ_thresh(x1,len),HARD);
    //   // if(res!=0) printf("\n## Error in threshold\n");
    //   printvec(x2,len);
    //   res=lDaub4(x2,len,0,0);
    //   // #####################################################
    //   if(res!=0) printf("\n## Error in transform\n");
    // }
    // t=timer(1);
    // ops=(double) len * (double) reps * 2. * 9. * 2.;
    // gflops=ops/(1e9*t);
    // printf("Time to do forward & backward transform: %gs\n",t/reps);
    // // printf("Gigaflops: %g\n",reps*2*9*(2*len-4)/(1e9*t));
    // printf("Gigaflops: %g\n",gflops);
    // printvec(x2,len);


    
    // printf("\n### Transform - LIFTED V2 ### \n");
    // timer(-1);
    // for(i=0;i<reps;i++){
    //   // #########_transform_three_###########################
    //   res=l2Daub4(x,len,1,0);
    //   printvec(x,len);
    //   if(res!=0) printf("\n## Error in transform\n");
    //   // res=threshold(x2,len,univ_thresh(x1,len),HARD);
    //   // if(res!=0) printf("\n## Error in threshold\n");
    //   res=l2Daub4(x,len,0,0);
    //   // #####################################################
    //   if(res!=0) printf("\n## Error in transform\n");
    // }
    // t=timer(1);
    // ops=(double) len * (double) reps * 2. * 10. * 2.;
    // gflops=ops/(1e9*t);
    // printf("Time to do forward & backward transform: %gs\n",t/reps);
    // //    printf("Gigaflops: %g\n",reps*2*9*(2*len-4)/(1e9*t));
    // printf("Gigaflops: %g\n",gflops);
    // printvec(x,len);

    // res = lDaub4(x,len,FWD,levels);

    // printf("\n### Transform - MODWT PO ### \n");
    printf("\n### Transform - MODWT PO (lifted!) ### \n");
    mptimer(-1);
    for(i=0;i<reps;i++){
      // res = Daub4MODWTpo(x,xm,len,FWD,levels);
      res = lDaub4MODWTpo(x,xm,len,FWD,levels);
      if(res!=0) printf("\n## Error in transform\n");
      if(inverse){
	// res=Daub4MODWTpo(x,xm,len,BWD,levels);
	res=lDaub4MODWTpo(x,xm,len,BWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
      } //inverse
    }
    t=mptimer(1);
    printf("%g,",t/(double)reps);

    // printf("\n### Transform - MODWT TO ### \n");
    printf("\n### Transform - MODWT TO (lifted!) ### \n");
    mptimer(-1);
    for(i=0;i<reps;i++){
      // res = Daub4MODWTto(x1,xm1,len,FWD,levels);
      res = lDaub4MODWTto(x1,xm1,len,FWD,levels);
      if(res!=0) printf("\n## Error in transform\n");
      if(inverse){
	// res=Daub4MODWTto(x1,xm1,len,BWD,levels);
	res=lDaub4MODWTto(x1,xm1,len,BWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
      } //inverse
    }
    t=mptimer(1);
    printf("%g,",t/(double)reps);

    // print_modwt_vec_po(xm,len,nlevels);

    // print_modwt_vec_to(xm1,len,nlevels);
    
    wst* w = create_wvtstruct(MODWT_PO,DAUB4,4,levels,len);
    // wst* w = create_wvtstruct(MODWT_TO,DAUB4,4,levels,len);
    wst* w1 = create_wvtstruct(MODWT_TO,DAUB4,4,levels,len);
    
    w->transformed = 1;
    w1->transformed = 1;

    copyvec(x,w->x,len);
    copyvec(x1,w1->x,len);
    

    copyvec(xm,w->xmod,modlen);
    copyvec(xm1,w1->xmod,modlen);
    
    // res = cmpmodwt(w,w1,-1,10);
    
    // printvec(xm,modlen);
    // printvec(xm1,modlen);
    
    kill_wvtstruct(w);
    kill_wvtstruct(w1);


    // printvec(x,len);
    // printvec(x1,len);
    
    
    // res=cmpvec(x,x2,len);
    // res=cmpvec(x,x1,len);
    
    res=cmpvec(x,x1,len);
    
    free(x);
    free(x1);
    free(x2);
    
    if(modwt){
      free(xm);
      free(xm1);
      free(xm2);
    }

  }
  return(0);
}
