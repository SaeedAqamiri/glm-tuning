#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
static ggml_backend_buffer_type_t amx_type(){auto reg=ggml_backend_cpu_reg();auto f=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts");auto bs=f(ggml_backend_reg_dev_get(reg,0));for(int i=0;bs[i];i++)if(!strcmp(ggml_backend_buft_name(bs[i]),"AMX"))return bs[i];return nullptr;}
static std::vector<float> run(ggml_type type,int K,int N,int M,int E,int B,int re,int rb,bool amx,bool strided,const std::vector<char>&qw,const std::vector<float>&xv){
 ggml_init_params ip={16*1024*1024,nullptr,true};auto wc=ggml_init(ip),c=ggml_init(ip);
 auto w=ggml_new_tensor_4d(wc,type,K,N,E,B);ggml_set_name(w,"batched-weight");
 auto xs=ggml_new_tensor_4d(c,GGML_TYPE_F32,K,strided?E*re:M,strided?M:E*re,B*rb);
 auto x=strided?ggml_permute(c,xs,0,2,1,3):xs;
 auto y=ggml_mul_mat(c,w,x);auto gf=ggml_new_graph(c);ggml_build_forward_expand(gf,y);
 auto wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,amx?amx_type():ggml_backend_cpu_buffer_type());auto cb=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());if(!wb||!cb)abort();
 ggml_backend_tensor_set(w,qw.data(),0,qw.size());ggml_backend_tensor_set(xs,xv.data(),0,xv.size()*4);
 auto b=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(b,8);
 if(!ggml_backend_dev_supports_op(ggml_backend_get_device(b),y)){puts("unsupported");abort();}
 if(ggml_backend_graph_compute(b,gf)!=GGML_STATUS_SUCCESS)abort();std::vector<float>out(ggml_nelements(y));ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
 ggml_backend_free(b);ggml_backend_buffer_free(cb);ggml_backend_buffer_free(wb);ggml_free(c);ggml_free(wc);return out;
}
int main(){ggml_time_init();if(!amx_type())return 2;int failed=0;
 for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0})for(int M:{1,2,4,7,33})for(int repeat:{1,2})for(bool strided:{false,true}){
 if(type==GGML_TYPE_F16&&strided)continue;
 int K=512,N=128,E=4,B=2;std::vector<float>w(K*N*E*B),x(K*M*E*B*repeat*repeat);for(size_t i=0;i<w.size();i++)w[i]=sin(i*.713)*.1+cos(i*.031)*.02;for(size_t i=0;i<x.size();i++)x[i]=sin(i*.143)+cos(i*.117)*.1;
 std::vector<char>q(ggml_row_size(type,K)*N*E*B);ggml_quantize_chunk(type,w.data(),q.data(),0,N*E*B,K,nullptr);
 auto ref=run(type,K,N,M,E,B,repeat,repeat,false,strided,q,x),got=run(type,K,N,M,E,B,repeat,repeat,true,strided,q,x);
 double err=0,norm=0,maxerr=0;for(size_t i=0;i<ref.size();i++){double d=got[i]-ref[i];if(!std::isfinite(d)){err=INFINITY;break;}err+=d*d;norm+=ref[i]*ref[i];maxerr=std::max(maxerr,std::abs(d));}double rel=sqrt(err/std::max(norm,1e-30));bool ok=rel<(type==GGML_TYPE_F16?5e-4:1e-5);printf("%s M=%d repeat=%d strided=%d rel=%.3g max=%.3g %s\n",ggml_type_name(type),M,repeat,int(strided),rel,maxerr,ok?"PASS":"FAIL");fflush(stdout);failed+=!ok;
 }return failed?1:0;
}
