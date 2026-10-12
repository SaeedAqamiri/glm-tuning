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
#include <chrono>
static ggml_backend_buffer_type_t amx_type() {
    auto reg=ggml_backend_cpu_reg();
    auto f=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts");
    auto bufts=f(ggml_backend_reg_dev_get(reg,0));
    for(int i=0;bufts[i];++i)if(!strcmp(ggml_backend_buft_name(bufts[i]),"AMX"))return bufts[i];
    return nullptr;
}
static std::vector<float> run(ggml_type type,int K,int N,int E,int S,int T,int channels,bool amx,const std::vector<char>&qw,const std::vector<float>&xv,const std::vector<int32_t>&iv){
    if(amx)setenv("GGML_PREFILL_SCALED_K","1",1);else unsetenv("GGML_PREFILL_SCALED_K");
    if(amx)setenv("GGML_VNNI_ROWS2","1",1);else unsetenv("GGML_VNNI_ROWS2");
    ggml_init_params ip={16*1024*1024,nullptr,true};
    auto wc=ggml_init(ip),c=ggml_init(ip);
    auto w=ggml_new_tensor_3d(wc,type,K,N,E);auto w2=ggml_new_tensor_3d(wc,type,K,N,E);ggml_set_name(w,"experts");
    auto x=ggml_new_tensor_3d(c,GGML_TYPE_F32,K,channels,T);
    auto ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,S,T);
    auto y1=ggml_mul_mat_id(c,w,x,ids);auto y2=ggml_mul_mat_id(c,w2,x,ids);auto y=ggml_add(c,y1,y2);auto gf=ggml_new_graph(c);ggml_build_forward_expand(gf,y);
    auto wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,amx_type());
    auto cb=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());
    if(!wb||!cb)std::abort();
    ggml_backend_tensor_set(w,qw.data(),0,qw.size());auto q2=qw;std::rotate(q2.begin(),q2.begin()+ggml_type_size(type),q2.end());ggml_backend_tensor_set(w2,q2.data(),0,q2.size());ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);ggml_backend_tensor_set(ids,iv.data(),0,iv.size()*4);
    auto b=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(b,atoi(getenv("TEST_THREADS")?getenv("TEST_THREADS"):"8"));if(amx){setenv("GGML_FUSE_MOE_PAIR","1",1);setenv("GGML_AMX_SHARED_DECODE","1",1);}else{unsetenv("GGML_FUSE_MOE_PAIR");unsetenv("GGML_AMX_SHARED_DECODE");}
    if(!ggml_backend_dev_supports_op(ggml_backend_get_device(b),y)){puts("op unsupported");std::abort();}
    auto t=std::chrono::steady_clock::now();
    if(ggml_backend_graph_compute(b,gf)!=GGML_STATUS_SUCCESS)std::abort();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    size_t n=ggml_nelements(y);std::vector<float> out(n*3);ggml_backend_tensor_get(y,out.data(),0,n*4);ggml_backend_tensor_get(y1,out.data()+n,0,n*4);ggml_backend_tensor_get(y2,out.data()+n*2,0,n*4);
    printf(" %s=%.2fms",amx?"packed":"plain",ms);
    ggml_backend_free(b);ggml_backend_buffer_free(cb);ggml_backend_buffer_free(wb);ggml_free(c);ggml_free(wc);return out;
}
int main(){
    setenv("GGML_AMX_MOE","1",1);ggml_time_init();
    if(!amx_type()){puts("AMX missing");return 2;}
    int failed=0;
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0})for(int T:{1,2,3,7,16,32,33,65,128})for(int ch:{1,2}){
        int K=getenv("LARGE")?4096:256,N=getenv("LARGE")?2048:64,E=4,S=2;
        std::vector<float>w(K*N*E),x(K*ch*T);
        for(size_t i=0;i<w.size();++i)w[i]=std::sin(i*0.713)*0.1+std::cos(i*0.031)*0.02;
        for(size_t i=0;i<x.size();++i)x[i]=std::sin(i*0.143)+std::cos(i*0.117)*0.1;
        std::vector<char>q(ggml_row_size(type,K)*N*E);
        ggml_quantize_chunk(type,w.data(),q.data(),0,N*E,K,nullptr);
        std::vector<int32_t>ids(S*T);for(int t=0;t<T;++t){ids[t*S]=(t*3+1)%E;ids[t*S+1]=(t*3+3)%E;}
        printf("%s T=%d channels=%d",ggml_type_name(type),T,ch);
        auto ref=run(type,K,N,E,S,T,ch,false,q,x,ids),got=run(type,K,N,E,S,T,ch,true,q,x,ids);
        double err=0,norm=0,maxerr=0;
        for(size_t i=0;i<ref.size();++i){double d=got[i]-ref[i];if(!std::isfinite(d)){err=INFINITY;break;}err+=d*d;norm+=ref[i]*ref[i];maxerr=std::max(maxerr,std::abs(d));}
        double rel=std::sqrt(err/std::max(norm,1e-30));bool ok=ref.size()==got.size() && !memcmp(ref.data(),got.data(),ref.size()*sizeof(float));
        printf(" rel=%.3g max=%.3g %s\n",rel,maxerr,ok?"PASS":"FAIL");fflush(stdout);failed+=!ok;
    }
    return failed?1:0;
}
