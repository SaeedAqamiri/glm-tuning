#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static std::vector<float> run(int S,int H,int T,int K,int nt,bool kda,bool perm,int split) {
    if(split) { char b[16]; snprintf(b,sizeof b,"%d",split);setenv("GGML_PREFILL_GDN_SPLIT",b,1); } else unsetenv("GGML_PREFILL_GDN_SPLIT");
    auto c=ggml_init({16*1024*1024,nullptr,true});
    auto make=[&](int h){auto t=ggml_new_tensor_4d(c,GGML_TYPE_F32,S,perm?T:h,perm?h:T,2);return perm?ggml_permute(c,t,0,2,1,3):t;};
    auto q=make(H),k=make(H),v=make(2*H);
    auto g=ggml_new_tensor_4d(c,GGML_TYPE_F32,kda?S:1,2*H,T,2);
    auto beta=ggml_new_tensor_4d(c,GGML_TYPE_F32,1,2*H,T,2);
    auto state=ggml_new_tensor_4d(c,GGML_TYPE_F32,S,S,2*H,2);
    auto y=ggml_gated_delta_net(c,q,k,v,g,beta,state,K);
    auto graph=ggml_new_graph(c);ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors_from_buft(c,ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_clear(buffer,0); // Initialize unused snapshot slots deterministically.
    for(auto t=ggml_get_first_tensor(c);t;t=ggml_get_next_tensor(c,t))if(!t->view_src&&t!=y){
        std::vector<float> data(ggml_nelements(t));
        for(size_t i=0;i<data.size();++i)data[i]=t==g?-.1f-std::abs(std::sin(i*.17f))*.1f:t==beta?.3f:(std::sin(i*.17f)*.05f);
        ggml_backend_tensor_set(t,data.data(),0,data.size()*4);
    }
    auto backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,nt);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)abort();
    std::vector<float> out(ggml_nelements(y));ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
    ggml_backend_free(backend);ggml_backend_buffer_free(buffer);ggml_free(c);return out;
}
int main(){int count=0;for(int S:{16,128})for(int T:{2,17,128})for(int K:{1,3})for(bool kda:{false,true})for(bool perm:{false,true})for(int nt:{3,112}){
    auto ref=run(S,2,T,K,nt,kda,perm,0);
    for(int split:{2,4,8}){auto got=run(S,2,T,K,nt,kda,perm,split);if(ref.size()!=got.size()||memcmp(ref.data(),got.data(),ref.size()*4)){printf("FAIL S%d T%d K%d kda%d perm%d nt%d split%d\n",S,T,K,kda,perm,nt,split);return 1;}++count;}
}printf("PASS %d GDN cases bit-exact\n",count);}
