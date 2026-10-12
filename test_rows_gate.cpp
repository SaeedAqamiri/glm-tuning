#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static std::vector<float> gate(int width,int rows,int threads,bool enabled,bool perm,float bias,bool intermediate){
 if(enabled)setenv("GGML_FUSE_GATE","1",1);else unsetenv("GGML_FUSE_GATE");
 auto c=ggml_init({16*1024*1024,nullptr,true});auto base=ggml_new_tensor_2d(c,GGML_TYPE_F32,perm?rows:width,perm?width:rows);auto x=perm?ggml_cont(c,ggml_transpose(c,base)):base;
 auto first=ggml_scale_bias(c,x,-1.25f,bias);auto middle=ggml_sigmoid(c,first);auto last=ggml_scale(c,middle,-5.0f);if(intermediate)ggml_set_output(first);
 auto g=ggml_new_graph(c);ggml_build_forward_expand(g,last);auto buf=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());
 std::vector<float> in(width*rows);for(size_t i=0;i<in.size();i++)in[i]=sin(i*.37)*17;ggml_backend_tensor_set(base,in.data(),0,in.size()*4);
 auto cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,threads);if(ggml_backend_graph_compute(cpu,g)!=GGML_STATUS_SUCCESS)abort();std::vector<float> out(width*rows);ggml_backend_tensor_get(last,out.data(),0,out.size()*4);
 ggml_backend_free(cpu);ggml_backend_buffer_free(buf);ggml_free(c);return out;
}
int main(){int count=0;setenv("GGML_ROWS_CHUNKS","1",1);
 for(auto type:{GGML_TYPE_F32,GGML_TYPE_I32})for(int width:{63,4096,8193,1048576})for(int count_rows:{1,7})for(int threads:{1,3,8,48})for(bool perm:{false,true}){
 int nr=5,groups=2,batches=2;auto c=ggml_init({16*1024*1024,nullptr,true});auto base=ggml_new_tensor_4d(c,type,width,perm?groups:nr,perm?nr:groups,batches);auto x=perm?ggml_permute(c,base,0,2,1,3):base;auto idx=ggml_new_tensor_3d(c,GGML_TYPE_I32,count_rows,groups,batches);auto y=ggml_get_rows(c,x,idx);auto g=ggml_new_graph(c);ggml_build_forward_expand(g,y);auto buf=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());
 for(int z=0;z<batches;z++)for(int k=0;k<groups;k++)for(int j=0;j<nr;j++)for(int i=0;i<width;i++){int v=i+11003*j+700001*k+13000001*z;memcpy((char*)x->data+i*x->nb[0]+j*x->nb[1]+k*x->nb[2]+z*x->nb[3],&v,4);}
 std::vector<int> ids(count_rows*groups*batches);for(size_t i=0;i<ids.size();i++)ids[i]=(i*3)%nr;ggml_backend_tensor_set(idx,ids.data(),0,ids.size()*4);auto cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,threads);if(ggml_backend_graph_compute(cpu,g)!=GGML_STATUS_SUCCESS)abort();
 for(int z=0;z<batches;z++)for(int k=0;k<groups;k++)for(int j=0;j<count_rows;j++)for(int i=0;i<width;i++){int expected=i+11003*ids[j+count_rows*(k+groups*z)]+700001*k+13000001*z,v;memcpy(&v,(char*)y->data+i*y->nb[0]+j*y->nb[1]+k*y->nb[2]+z*y->nb[3],4);if(v!=expected){printf("GET_ROWS FAIL\n");return 1;}}
 count++;ggml_backend_free(cpu);ggml_backend_buffer_free(buf);ggml_free(c);}
 printf("PASS %d GET_ROWS cases, independent bit-exact reference\n",count);count=0;
 for(int width:{64,8192})for(int rows:{1,7,128})for(int threads:{1,3,8,48})for(bool perm:{false,true})for(float bias:{0.0f,0.7f})for(bool intermediate:{false,true}){
 auto a=gate(width,rows,threads,false,perm,bias,intermediate),b=gate(width,rows,threads,true,perm,bias,intermediate);if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*4)){printf("GATE FAIL width=%d rows=%d threads=%d perm=%d bias=%g intermediate=%d\n",width,rows,threads,perm,bias,intermediate);return 2;}count++;}
 printf("PASS %d gate cases, bit-exact versus separate kernels\n",count);
}
