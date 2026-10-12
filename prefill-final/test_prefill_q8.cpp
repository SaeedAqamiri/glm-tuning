#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static ggml_backend_buffer_type_t amx_type(){auto reg=ggml_backend_cpu_reg();auto f=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts");auto bs=f(ggml_backend_reg_dev_get(reg,0));for(int i=0;bs[i];i++)if(!strcmp(ggml_backend_buft_name(bs[i]),"AMX"))return bs[i];return nullptr;}
static std::vector<float> run(int K,int N,int M,int threads,bool enabled,bool intermediate){
 if(enabled)setenv("GGML_Q8_PREFILL_SIGNED","1",1);else unsetenv("GGML_Q8_PREFILL_SIGNED");
 if(enabled)setenv("GGML_VNNI_ROWS2","1",1);else unsetenv("GGML_VNNI_ROWS2");
 if(enabled)setenv("GGML_FUSE_QKV","1",1);else unsetenv("GGML_FUSE_QKV");
 auto wc=ggml_init({16*1024*1024,nullptr,true}),c=ggml_init({16*1024*1024,nullptr,true});ggml_tensor *w[3];for(int i=0;i<3;i++)w[i]=ggml_new_tensor_2d(wc,GGML_TYPE_Q8_0,K,N);
 auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,K,M);auto q=ggml_mul_mat(c,w[0],x),k=ggml_mul_mat(c,w[1],x),v=ggml_mul_mat(c,w[2],x);auto out=ggml_concat(c,ggml_concat(c,q,k,0),v,0);if(intermediate)ggml_set_output(q);
 auto g=ggml_new_graph(c);ggml_build_forward_expand(g,out);auto wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,amx_type()),cb=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());if(!wb||!cb)abort();
 std::vector<float>wf(K*N),xf(K*M);std::vector<char>qw(ggml_row_size(GGML_TYPE_Q8_0,K)*N);
 for(int j=0;j<3;j++){for(size_t i=0;i<wf.size();i++)wf[i]=sin(i*.31+j)*.2+cos(i*.009)*.03;ggml_quantize_chunk(GGML_TYPE_Q8_0,wf.data(),qw.data(),0,N,K,nullptr);for(size_t block=0;block<qw.size()/34;++block)for(int q=0;q<32;++q){const int v[5]={-128,127,0,-1,1};qw[block*34+2+q]=(char)v[(block+q+j)%5];}ggml_backend_tensor_set(w[j],qw.data(),0,qw.size());}
 for(size_t i=0;i<xf.size();i++)xf[i]=sin(i*.713)+cos(i*.111)*.1;ggml_backend_tensor_set(x,xf.data(),0,xf.size()*4);
 auto b=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(b,threads);if(ggml_backend_graph_compute(b,g)!=GGML_STATUS_SUCCESS)abort();std::vector<float>result(3*N*M+(intermediate?N*M:0));ggml_backend_tensor_get(out,result.data(),0,3*N*M*4);if(intermediate)ggml_backend_tensor_get(q,result.data()+3*N*M,0,N*M*4);
 if(const char *path=getenv("GGML_TEST_DUMP")){FILE*f=fopen(path,"ab");if(!f)abort();fwrite(result.data(),sizeof(float),result.size(),f);fclose(f);}
 ggml_backend_free(b);ggml_backend_buffer_free(wb);ggml_backend_buffer_free(cb);ggml_free(c);ggml_free(wc);return result;
}
int main(){int count=0;setenv("GGML_AMX_NUMA_DENSE","1",1);for(int K:{512,4096})for(int N:{96,128,256})for(int M:{1,2,7,16,32,33,65,512})for(int threads:{1,3,8,48})for(bool intermediate:{false,true}){
 auto a=run(K,N,M,threads,false,intermediate),b=run(K,N,M,threads,true,intermediate);if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*4)){double max=0;for(size_t i=0;i<a.size();i++)max=fmax(max,fabs(a[i]-b[i]));printf("FAIL K=%d N=%d M=%d t=%d intermediate=%d max=%g\n",K,N,M,threads,intermediate,max);return 1;}count++;}
 printf("PASS %d QKV cases, bit-exact packed reference including fallback and preserved intermediates\n",count);
}
