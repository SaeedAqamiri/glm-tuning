#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <vector>
#include <cstdio>
#include <cstring>
int main(){int count=0;for(int dim=0;dim<4;dim++)for(int threads:{1,3,8,48})for(bool perm:{false,true}){
 auto c=ggml_init({16*1024*1024,nullptr,true});int64_t n[4]={7,9,2,3},m[4]={7,9,2,3};m[dim]+=2;
 auto make=[&](int64_t *s){auto t=ggml_new_tensor_4d(c,GGML_TYPE_F32,perm?s[1]:s[0],perm?s[0]:s[1],s[2],s[3]);return perm?ggml_permute(c,t,1,0,2,3):t;};
 auto a=make(n),b=make(m);auto y=ggml_concat(c,a,b,dim);auto g=ggml_new_graph(c);ggml_build_forward_expand(g,y);auto buf=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());
 for(auto t:{a,b})for(int64_t z=0;z<t->ne[3];z++)for(int64_t k=0;k<t->ne[2];k++)for(int64_t j=0;j<t->ne[1];j++)for(int64_t i=0;i<t->ne[0];i++){float v=(t==a?0:10000)+i+17*j+113*k+997*z;memcpy((char*)t->data+i*t->nb[0]+j*t->nb[1]+k*t->nb[2]+z*t->nb[3],&v,4);}
 auto cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,threads);if(ggml_backend_graph_compute(cpu,g)!=GGML_STATUS_SUCCESS)return 2;
 for(int64_t z=0;z<y->ne[3];z++)for(int64_t k=0;k<y->ne[2];k++)for(int64_t j=0;j<y->ne[1];j++)for(int64_t i=0;i<y->ne[0];i++){int64_t q[4]={i,j,k,z};bool second=q[dim]>=n[dim];if(second)q[dim]-=n[dim];float expected=(second?10000:0)+q[0]+17*q[1]+113*q[2]+997*q[3],actual;memcpy(&actual,(char*)y->data+i*y->nb[0]+j*y->nb[1]+k*y->nb[2]+z*y->nb[3],4);if(actual!=expected){printf("FAIL dim=%d threads=%d perm=%d\n",dim,threads,perm);return 1;}}
 count++;ggml_backend_free(cpu);ggml_backend_buffer_free(buf);ggml_free(c);
 }printf("PASS %d concat cases\n",count);}
