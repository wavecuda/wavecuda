#include "transform.h"

int transform(wst *w,short sense){
  int ret;
  
  if(sense != w->transformed ? BWD : FWD){
    printf("\nSense doesn't make sense. We can't FWD/BWD something that is/not transformed!\n");
    return(1);
  }
  
  
  if(w->ttype == DWT){

    switch(w->filt){
    case HAAR:
      ret = Haar(w->x,w->len,sense,w->levels);
      break;
    case HAARMP:
      ret = Haarmp(w->x,w->len,sense,w->levels);
      break;
    case DAUB4:
      ret = Daub4(w->x,w->len,sense,w->levels);
      break;
    case DAUB4MP:
      ret = lompDaub4(w->x,w->len,sense,w->levels);
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
      ret = HaarMODWT(w->x,w->xmod,w->len,sense,w->levels);
      break;
    case HAARMP:
      ret = HaarMODWTomp2(w->x,w->xmod,w->len,sense,w->levels);
      break;
    case DAUB4:
      ret = Daub4MODWTpo(w->x,w->xmod,w->len,sense,w->levels);
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
      ret = HaarMODWTto(w->x,w->xmod,w->len,sense,w->levels);
      break;
    case HAARMP:
      ret = HaarMODWTtomp(w->x,w->xmod,w->len,sense,w->levels);
      break;
    case DAUB4:
      ret = Daub4MODWTto(w->x,w->xmod,w->len,sense,w->levels);
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
