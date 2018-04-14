#include "transformcuda.cuh"


int transform(cuwst *w,short sense){
  // wrapper to "master" version taking a stream argument
  return(transform(w,sense,NULL));
}

int transform(cuwst *w,short sense, cudaStream_t stream){
  int ret;
  
  if(sense != w->transformed ? BWD : FWD){
    printf("\nSense doesn't make sense. We can't FWD/BWD something that is/not transformed!\n");
    return(1);
  }

  if(w->ttype == DWT){

    switch(w->filt){
    case HAAR:
      ret = HaarCUDAMLv2(w->x_h,w->x_d,w->len,sense,w->levels,stream);
      break;
    case DAUB4:
      //ret = Daub4(w->x,w->len,sense,w->levels);
      printf("\nNeed to edit Daub4 CUDA function appropriately like Haar\n");
      break;
    case HAARNOHOST:
      ret = HaarCUDAMLv3(w->x_d,w->len,sense,w->levels,stream);
      break;
    default:
      printf("\nUnrecognised filter\n");
      return(1);
      break;
    }
  }

  if(w->ttype == MODWT_PO){
    switch(w->filt){
    case HAAR:
      ret = HaarCUDAMODWTv4(w->x_h,w->xmod_h,w->x_d,w->xmod_d,w->len,sense,w->levels);
      // this PO version uses streams itself to run the different packets
      // however, it is far slower than the TO version!
      break;
    case HAARNOHOST:     
      ret = HaarCUDAMODWT(w->x_d, w->xmod_d,w->len,sense,w->levels);
      // this is a version without host memory
      // it doesn't use streams. It's just slow!
      break;
    default:
      printf("\nUnrecognised filter\n");
      return(1);
      break;
    }  
  }

  if(w->ttype == MODWT_TO){
    switch(w->filt){
    case HAAR:
      ret = HaarCUDAMODWTv6(w->x_h,w->xmod_h,w->x_d,w->xmod_d,w->len,sense,w->levels,stream);
      break;
    case HAARNOHOST:
      ret = HaarCUDAMODWTv6d(w->x_d,w->xmod_d,w->len,sense,w->levels,stream);
      // this is a version just using device memory
      break;
    default:
      printf("\nUnrecognised filter\n");
      return(1);
      break;
    }  
  }
  
  // we switch the 'transformed' boolean
  w->transformed = !(w->transformed);
  return(ret);
  
}