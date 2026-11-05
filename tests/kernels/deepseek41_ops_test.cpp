#include "basalt/deepseek41/ops.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/deepseek41/execution_ops.hpp"
#include "basalt/deepseek41/weights.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <map>

namespace ds = basalt::deepseek41;
static int failures = 0;
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::printf("CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
template<class T> static T* device(size_t n) { T* p=nullptr; ck(cudaMalloc(&p,n*sizeof(T))); return p; }
template<class T> static void upload(T* p, const T* x, size_t n) { ck(cudaMemcpy(p,x,n*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> static std::vector<T> read(T* p,size_t n) { std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
static void compare(const char* name,const std::vector<float>& got,const std::vector<double>& ref) {
    double max=0, norm=0, diff=0, magnitude=0;
    for(size_t i=0;i<got.size();++i) {
        failures += !std::isfinite(got[i]);
        max=std::max(max,std::abs(got[i]-ref[i])); norm=std::max(norm,std::abs(ref[i]));
        diff+=std::abs(got[i]-ref[i]); magnitude+=std::abs(ref[i]);
    }
    const double relative=diff/(magnitude+1e-30);
    failures += max>3e-5*std::max(1.,norm) || relative>3e-5;
    std::printf("%s: max abs %.3g L1 rel %.3g\n",name,max,relative);
}
static void same(const std::vector<float>& a,const float* b,const char* name) {
    if(std::memcmp(a.data(),b,a.size()*4)) { ++failures; std::printf("FAIL: %s window/position bits\n",name); }
}
static void reference_split(const double* m,const float* s,const float* b,double* pre,double* post,double* c) {
    for(int i=0;i<4;++i) {
        pre[i]=1./(1.+std::exp(-(m[i]*s[0]+b[i])))+1e-6;
        post[i]=2./(1.+std::exp(-(m[4+i]*s[1]+b[4+i])));
    }
    for(int j=0;j<4;++j) {
        double max=-1e300,sum=0;
        for(int k=0;k<4;++k) { c[j*4+k]=m[8+j*4+k]*s[2]+b[8+j*4+k]; max=std::max(max,c[j*4+k]); }
        for(int k=0;k<4;++k) { c[j*4+k]=std::exp(c[j*4+k]-max); sum+=c[j*4+k]; }
        for(int k=0;k<4;++k) c[j*4+k]=c[j*4+k]/sum+1e-6;
    }
    for(int it=0;it<20;++it) {
        if(it) for(int j=0;j<4;++j) {
            double sum=1e-6; for(int k=0;k<4;++k) sum+=c[j*4+k];
            for(int k=0;k<4;++k) c[j*4+k]/=sum;
        }
        for(int k=0;k<4;++k) {
            double sum=1e-6; for(int j=0;j<4;++j) sum+=c[j*4+k];
            for(int j=0;j<4;++j) c[j*4+k]/=sum;
        }
    }
}
static void hc(int h,const float* fn,const float* scale,const float* base,const float* norm,
               std::mt19937& rng,bool zero=false,bool bench=false) {
    constexpr int T=5;
    std::normal_distribution<float> nd;
    std::vector<float> x(T*4*h),branch(T*h);
    for(auto& v:x) v=zero?0.f:nd(rng);
    for(auto& v:branch) v=nd(rng);
    std::vector<double> mixes(T*24),pre(T*4),post(T*4),comb(T*16),raw(T*h),normalized(T*h),out(T*4*h);
    for(int t=0;t<T;++t) {
        double squares=0; for(int i=0;i<4*h;++i) squares+=double(x[t*4*h+i])*x[t*4*h+i];
        const double inv=1./std::sqrt(squares/(4*h)+1e-20);
        for(int m=0;m<24;++m) {
            double dot=0; for(int i=0;i<4*h;++i) dot+=double(x[t*4*h+i])*fn[m*4*h+i];
            mixes[t*24+m]=dot*inv;
        }
        reference_split(mixes.data()+t*24,scale,base,pre.data()+t*4,post.data()+t*4,comb.data()+t*16);
        squares=0;
        for(int i=0;i<h;++i) {
            for(int s=0;s<4;++s) raw[t*h+i]+=pre[t*4+s]*x[(t*4+s)*h+i];
            squares+=raw[t*h+i]*raw[t*h+i];
        }
        const double rstd=1./std::sqrt(squares/h+1e-20);
        for(int i=0;i<h;++i) normalized[t*h+i]=raw[t*h+i]*rstd*norm[i];
        for(int k=0;k<4;++k) for(int i=0;i<h;++i) {
            double a=post[t*4+k]*branch[t*h+i];
            for(int j=0;j<4;++j) a+=comb[t*16+j*4+k]*x[(t*4+j)*h+i];
            out[(t*4+k)*h+i]=a;
        }
    }
    auto* dx=device<float>(x.size()); auto* db=device<float>(branch.size()); auto* df=device<float>(24*4*h);
    auto* dscl=device<float>(3); auto* dbase=device<float>(24); auto* dn=device<float>(h);
    auto* dm=device<float>(T*24); auto* dp=device<float>(T*4); auto* dq=device<float>(T*4); auto* dc=device<float>(T*16);
    auto* dy=device<float>(T*h); auto* dout=device<float>(T*4*h);
    upload(dx,x.data(),x.size()); upload(db,branch.data(),branch.size()); upload(df,fn,size_t(24*4*h));
    upload(dscl,scale,3); upload(dbase,base,24); upload(dn,norm,size_t(h));
    auto run=[&](int tokens,const float* input,const float* b,float* m,float* p,float* q,float* c,float* y,float* z) {
        ds::hc_project(input,df,m,h,tokens,1e-20f,nullptr);
        ds::hc_split(m,dscl,dbase,p,q,c,tokens,20,1e-6f,nullptr);
        ds::hc_reduce(input,p,dn,y,h,tokens,1e-20f,nullptr);
        ds::hc_expand(b,input,q,c,z,h,tokens,nullptr);
    };
    run(T,dx,db,dm,dp,dq,dc,dy,dout);
    const auto gm=read(dm,T*24),gp=read(dp,T*4),gq=read(dq,T*4),gc=read(dc,T*16),gy=read(dy,T*h),go=read(dout,T*4*h);
    compare("HC projection",gm,mixes); compare("HC pre",gp,pre); compare("HC post",gq,post); compare("HC Sinkhorn",gc,comb);
    compare("HC normalized reduction",gy,normalized); compare("HC residual expansion",go,out);
    for(int t=0;t<T;++t) {
        run(1,dx+t*4*h,db+t*h,dm,dp,dq,dc,dy,dout);
        same(read(dm,24),gm.data()+t*24,"projection"); same(read(dp,4),gp.data()+t*4,"pre");
        same(read(dq,4),gq.data()+t*4,"post"); same(read(dc,16),gc.data()+t*16,"Sinkhorn");
        same(read(dy,h),gy.data()+t*h,"normalized reduction"); same(read(dout,4*h),go.data()+t*4*h,"expansion");
    }
    run(T,dx,db,dm,dp,dq,dc,dy,dout);
    ds::hc_reduce(dx,dp,nullptr,dy,h,T,1e-20f,nullptr); compare("HC unnormalized reduction",read(dy,T*h),raw);
    ds::hc_expand(db,dx,dq,dc,dx,h,T,nullptr); same(read(dx,T*4*h),go.data(),"in-place expansion");
    if(bench) {
        upload(dx,x.data(),x.size());
        for(int i=0;i<10;++i) run(1,dx,db,dm,dp,dq,dc,dy,dout);
        cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b)); ck(cudaEventRecord(a));
        for(int i=0;i<100;++i) run(1,dx,db,dm,dp,dq,dc,dy,dout);
        ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b)); float ms=0; ck(cudaEventElapsedTime(&ms,a,b));
        std::printf("HC four-kernel chain hidden=%d T=1: %.2f us/call\n",h,ms*10);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    for(auto* p:{dx,db,df,dscl,dbase,dn,dm,dp,dq,dc,dy,dout}) cudaFree(p);
}
static void router(int n,int k) {
    constexpr int T=5;
    std::mt19937 rng(417);
    std::vector<float> logits(T*n),bias(n);
    // Raw integer samples keep frozen GPU fixtures identical across standard libraries.
    for(auto& v:logits) v=float(int(rng()>>16)-32768)*0x1p-10f;
    for(auto& v:bias) v=float(int(rng()>>16)-32768)*0x1p-17f;
    logits[0]=-12.f; bias[0]=100.f;
    logits[n]=-1000.f; logits[n+1]=1000.f; logits[n+2]=20.f; logits[n+3]=20.001f;
    auto* dl=device<float>(logits.size()); auto* db=device<float>(n); auto* dw=device<float>(T*k); auto* di=device<int32_t>(T*k);
    upload(dl,logits.data(),logits.size()); upload(db,bias.data(),bias.size());
    uint64_t hash=14695981039346656037ull;
    const auto hash_bytes=[&](const void* data,size_t bytes) {
        const auto* p=static_cast<const uint8_t*>(data);
        for(size_t i=0;i<bytes;++i)hash=(hash^p[i])*1099511628211ull;
    };
    for(bool biased:{false,true}) {
        ds::route(dl,biased?db:nullptr,di,dw,n,k,T,1.5f,nullptr);
        const auto ids=read(di,T*k); const auto weights=read(dw,T*k);
        hash_bytes(ids.data(),ids.size()*sizeof(int32_t));hash_bytes(weights.data(),weights.size()*sizeof(float));
        std::vector<double> ref(T*k);
        for(int t=0;t<T;++t) {
            std::vector<double> score(n),rank(n); std::vector<int> order(n); std::iota(order.begin(),order.end(),0);
            for(int i=0;i<n;++i) { double l=logits[t*n+i]; score[i]=std::sqrt(l>20?l:std::log1p(std::exp(l))); rank[i]=score[i]+(biased?bias[i]:0.f); }
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return rank[a]>rank[b];});
            double sum=1e-20; for(int r=0;r<k;++r) sum+=score[order[r]];
            for(int r=0;r<k;++r) { failures+=ids[t*k+r]!=order[r]; ref[t*k+r]=score[order[r]]/sum*1.5; }
        }
        compare(biased?"biased router weights":"router weights",weights,ref);
        for(int t=0;t<T;++t) {
            ds::route(dl+t*n,biased?db:nullptr,di,dw,n,k,1,1.5f,nullptr);
            const auto one=read(di,k);
            failures+=std::memcmp(one.data(),ids.data()+t*k,k*4)!=0;
            same(read(dw,k),weights.data()+t*k,"router");
        }
    }
    std::fill(logits.begin(),logits.end(),0.f); upload(dl,logits.data(),logits.size());
    ds::route(dl,nullptr,di,dw,n,k,T,1.5f,nullptr);
    const auto ties=read(di,T*k);
    for(int t=0;t<T;++t) for(int r=0;r<k;++r) failures+=ties[t*k+r]!=r;
    std::printf("Router experts=%d top-%d: ties and window/position bits checked\n",n,k);
    std::printf("Router experts=%d frozen output hash %016llx\n",n,(unsigned long long)hash);
    failures+=hash!=(n==384?0x61814bb88b397e68ull:0x44a1177b4fbd223eull);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<100;++i)ds::route(dl,db,di,dw,n,k,1,1.5f,stream);
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    for(int i=0;i<10;++i)ck(cudaGraphLaunch(exec,stream));
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a,stream));
    for(int i=0;i<20;++i)ck(cudaGraphLaunch(exec,stream));
    ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
    const double us=ms*.5;
    std::printf("micro Router experts=%d top-%d T=1: %.3f us, %.3f logical GB/s\n",n,k,us,(n*8.+k*8.)/us/1000.);
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    cudaFree(dl); cudaFree(db); cudaFree(dw); cudaFree(di);
}
static void primary_plan(std::mt19937& rng) {
    constexpr int experts=384,capacity=36;
    auto* lookup=device<unsigned long long>(experts);auto* ids=device<int32_t>(capacity);
    auto* ptr=device<unsigned long long>(capacity+2);auto* start=device<int32_t>(capacity+3);auto* count=device<int32_t>(3);
    auto* tok=device<int32_t>(capacity+2);auto* dst=device<int32_t>(capacity+2);
    std::vector<unsigned long long> table(experts);
    for(int trial=0;trial<1152;++trial) {
        const int top=1+trial%6,nt=1+(trial/6)%6,entries=nt*top;
        const int residency=(trial/36)%3;
        for(auto& value:table)value=residency==0?0:residency==1 || rng()%3?0x100000000ull+rng():0;
        std::vector<int32_t> route(capacity);
        for(auto& e:route)e=int(rng()%experts);
        if(trial%4==0)route[1]=route[0];
        if(trial%7==0) {route[2]=-1;route[3]=experts;}
        std::vector<unsigned long long> pointers(capacity+2,0xdeadbeefull);
        std::vector<int32_t> starts(capacity+3,-77),groups(3,-77),tokens(capacity+2,-77),destinations(capacity+2,-77);
        upload(lookup,table.data(),table.size());upload(ids,route.data(),route.size());
        upload(ptr,pointers.data(),pointers.size());upload(start,starts.data(),starts.size());
        upload(count,groups.data(),groups.size());upload(tok,tokens.data(),tokens.size());upload(dst,destinations.data(),destinations.size());
        std::map<int,std::vector<int>> reference;
        for(int j=0;j<entries;++j)if(route[j]>=0 && route[j]<experts && table[route[j]])reference[route[j]].push_back(j);
        int ng=0,rows=0;
        for(const auto& [expert,members]:reference) {
            pointers[1+ng]=table[expert];starts[1+ng++]=rows;
            for(int j:members) {tokens[1+rows]=j/top;destinations[1+rows++]=j;}
        }
        starts[1+ng]=rows;groups[1]=ng;
        ds::plan_primary_experts(lookup,ids,experts,top,ptr+1,start+1,count+1,tok+1,dst+1,nullptr,nt);
        failures+=read(ptr,capacity+2)!=pointers;failures+=read(start,capacity+3)!=starts;failures+=read(count,3)!=groups;
        failures+=read(tok,capacity+2)!=tokens;failures+=read(dst,capacity+2)!=destinations;
    }
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    for(int nt:{1,6}) {
        for(int i=0;i<20;++i)ds::plan_primary_experts(lookup,ids,experts,6,ptr+1,start+1,count+1,tok+1,dst+1,nullptr,nt);
        ck(cudaEventRecord(a));
        for(int i=0;i<1000;++i)ds::plan_primary_experts(lookup,ids,experts,6,ptr+1,start+1,count+1,tok+1,dst+1,nullptr,nt);
        ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
        std::printf("device expert plan: tokens=%d top=6 micro %.3f us\n",nt,ms);
    }
    std::printf("device expert plan: 1152 CPU-exact cases, widths 1..6, canaries, duplicates, invalid IDs, empty/full residency\n");
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
    cudaFree(lookup);cudaFree(ids);cudaFree(ptr);cudaFree(start);cudaFree(count);cudaFree(tok);cudaFree(dst);
}
static int hc_token_tile(std::mt19937& rng) {
    constexpr int h=5120,width=4*h,max_tokens=2048;
    std::vector<float> inputs(size_t(width)*max_tokens),weights(24*width);
    for(auto* v:{&inputs,&weights})for(auto& f:*v)f=float(int(rng()>>16)-32768)*0x1p-16f;
    auto* x=device<float>(inputs.size());auto* w=device<float>(weights.size());auto* y=device<float>(max_tokens*24);
    upload(x,inputs.data(),inputs.size());upload(w,weights.data(),weights.size());
    auto& flag=basalt::core::diagnostics().exp["ds-prefill-hc-token-tile"];
    const auto launch=[&](int n){ds::hc_project_grouped(x,w,y,h,n,1e-20f,nullptr);};
    for(int n:{1,3,4,5,31,max_tokens}) {
        flag=0;launch(n);const auto ref=read(y,n*24);
        for(int tile:{4,8}) {flag=tile;launch(n);same(read(y,n*24),ref.data(),"tiled HC projection");}
    }
    cudaEvent_t start{},end{};ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    for(int enabled:{0,4,8,8,4,0}) {
        flag=enabled;launch(max_tokens);
        ck(cudaEventRecord(start));
        for(int i=0;i<30;++i)launch(max_tokens);
        ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms=0;
        ck(cudaEventElapsedTime(&ms,start,end));
        std::printf("HC tile=%d: %.3f us per 2048 rows\n",enabled,ms*1000.f/30);
    }
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
    ck(cudaFree(x));ck(cudaFree(w));ck(cudaFree(y));flag=0;
    std::printf("HC token tile: %d failures\n",failures);
    return failures?1:0;
}
static int router_token_tile(std::mt19937& rng) {
    constexpr int in=5120,out=384,tokens=2048;
    std::vector<float> inputs(size_t(in)*tokens),weights(size_t(in)*out);
    for(auto* v:{&inputs,&weights})for(auto& f:*v)f=float(int(rng()>>16)-32768)*0x1p-16f;
    auto* x=device<float>(inputs.size());auto* w=device<float>(weights.size());auto* y=device<float>(tokens*out);
    upload(x,inputs.data(),inputs.size());upload(w,weights.data(),weights.size());
    const auto launch=[&](int n,int tile){ds::router_linear_tiled(w,x,y,n,nullptr,8,tile);};
    for(int n:{1,7,17,33,tokens-1,tokens}) {
        launch(n,1);const auto ref=read(y,n*out);
        for(int tile:{2,4}) {launch(n,tile);same(read(y,n*out),ref.data(),"wider router projection");}
    }
    cudaEvent_t start{},end{};ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    for(int tile:{1,2,4,4,2,1}) {
        launch(tokens,tile);ck(cudaEventRecord(start));
        for(int i=0;i<30;++i)launch(tokens,tile);
        ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms=0;
        ck(cudaEventElapsedTime(&ms,start,end));
        std::printf("Router experts per warp=%d: %.3f us per 2048 rows\n",tile,ms*1000.f/30);
    }
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
    ck(cudaFree(x));ck(cudaFree(w));ck(cudaFree(y));
    std::printf("Router token tile: %d failures\n",failures);
    return failures?1:0;
}
static int run(int argc,char** argv) {
    int dev=0; std::string target,draft;bool tile_only=false,router_only=false;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--device" && i+1<argc) dev=std::stoi(argv[++i]);
        else if(a=="--model" && i+1<argc) target=argv[++i];
        else if(a=="--dspark" && i+1<argc) draft=argv[++i];
        else if(a=="--hc-token-tile")tile_only=true;
        else if(a=="--router-token-tile")router_only=true;
        else return 2;
    }
    ck(cudaSetDevice(dev)); std::mt19937 rng(41);
    if(tile_only)return hc_token_tile(rng);
    if(router_only)return router_token_tile(rng);
    for(int dim:{7,5120}) {
        std::normal_distribution<float> nd;
        std::vector<float> fn(24*4*dim),base(24),norm(dim); const float scale[]={.3f,.4f,.2f};
        for(auto& v:fn) v=nd(rng)*.01f; for(auto& v:base) v=nd(rng)*.1f; for(auto& v:norm) v=1.f+nd(rng)*.1f;
        hc(dim,fn.data(),scale,base.data(),norm.data(),rng,false,dim==5120);
        if(dim==7) hc(dim,fn.data(),scale,base.data(),norm.data(),rng,true);
    }
    router(384,6); router(128,3);
    {
        constexpr int in=5120, out=384, tokens=17;
        std::vector<float> weights(in*out), input(in*tokens);
        for(auto* v:{&weights,&input})for(float& f:*v)f=float(int(rng()>>16)-32768)*0x1p-14f;
        auto* w=device<float>(weights.size());auto* x=device<float>(input.size());auto* y=device<float>(tokens*out);
        upload(w,weights.data(),weights.size());upload(x,input.data(),input.size());
        ds::dense_linear(0,w,x,y,in,out,tokens,in,out,nullptr);
        const auto reference=read(y,tokens*out);
        for(int rows:{4,8}) {
            ds::router_linear_tiled(w,x,y,tokens,nullptr,rows);
            same(read(y,tokens*out),reference.data(),"tiled router FP32 projection including tail");
        }
        ck(cudaFree(y));ck(cudaFree(x));ck(cudaFree(w));
    }
    {
        constexpr int in=256, vocab=129280;
        std::vector<uint16_t> matrix(size_t(in)*vocab);
        for(size_t i=0;i<matrix.size();++i)
            matrix[i]=uint16_t(0x3c00u+(i*29u+i/307u)%768u);
        std::vector<float> input(in),logits(vocab);
        for(int i=0;i<vocab;++i)logits[i]=float(i%300-150)*0.003f;
        auto* m=device<uint16_t>(matrix.size());auto* x=device<float>(in);
        auto* y=device<float>(vocab);auto* bias=device<float>(vocab);
        auto* cache=device<float>(size_t(ds::kMarkovCacheSlots)*vocab);
        auto* keys=device<int32_t>(ds::kMarkovCacheSlots);auto* previous=device<int32_t>(1);
        upload(m,matrix.data(),matrix.size());ck(cudaMemset(keys,0xff,ds::kMarkovCacheSlots*sizeof(int32_t)));
        for(int32_t id:{9,9,73,73,9}) {
            for(int i=0;i<in;++i)input[i]=float((i*id)%197-100)*0.007f;
            upload(x,input.data(),in);upload(previous,&id,1);upload(y,logits.data(),vocab);
            ds::dense_linear(30,m,x,bias,in,vocab,1,in,vocab,nullptr);ds::add_rows(y,bias,vocab,nullptr);
            const auto expected=read(y,vocab);
            upload(y,logits.data(),vocab);ds::cached_markov_add(m,x,previous,keys,cache,y,nullptr);
            same(read(y,vocab),expected.data(),"cached Markov bias miss/hit/replacement");
        }
        for(void* p:{static_cast<void*>(m),static_cast<void*>(x),static_cast<void*>(y),static_cast<void*>(bias),
                     static_cast<void*>(cache),static_cast<void*>(keys),static_cast<void*>(previous)})ck(cudaFree(p));
    }
    {
        constexpr int h=37,k=6,tokens=6;
        constexpr uint64_t mask=(uint64_t{1}<<0)|(uint64_t{1}<<11)|(uint64_t{1}<<35);
        std::vector<float> gpu(h*k*tokens),host(h*k*tokens),reference_host(h*k*tokens),weights(k*tokens),shared(h*tokens);
        for(auto* v:{&gpu,&weights,&shared})for(auto& f:*v)f=float(int(rng()>>16)-32768)*0x1p-16f;
        for(int row=0;row<k*tokens;++row)for(int i=0;i<h;++i) {
            const int index=row*h+i;
            reference_host[index]=((mask>>row)&1)?float(index%199-100)*0.01f:0.f;
            host[index]=((mask>>row)&1)?reference_host[index]:std::nanf("");
        }
        auto* g=device<float>(gpu.size());auto* c=device<float>(host.size());auto* w=device<float>(weights.size());
        auto* s=device<float>(shared.size());auto* y=device<float>(shared.size());
        upload(g,gpu.data(),gpu.size());upload(c,reference_host.data(),host.size());
        upload(w,weights.data(),weights.size());upload(s,shared.data(),shared.size());
        ds::combine_experts(g,c,w,s,y,h,k,tokens,nullptr);
        const auto expected=read(y,shared.size());
        upload(c,host.data(),host.size());
        ds::combine_experts_masked(g,c,w,s,y,h,k,tokens,mask,nullptr);
        same(read(y,shared.size()),expected.data(),"masked expert combine ignores stale host rows");
        for(float* ptr:{g,c,w,s,y})ck(cudaFree(ptr));
    }
    primary_plan(rng);
    if(!target.empty() || !draft.empty()) {
        ds::Weights weights(target,draft);
        for(bool is_draft:{false,true}) for(int l:is_draft?std::vector<int>{0,2}:std::vector<int>{0,14,39}) {
            auto tensor=[&](const std::string& suffix,const std::vector<uint64_t>& shape) {
                const auto name="blk."+std::to_string(l)+"."+suffix;
                const auto v=is_draft?weights.draft_tensor(name):weights.target_tensor(name);
                ds::expect(v.tensor->type==0 && v.tensor->shape==shape,"invalid HC tensor "+name);
                return reinterpret_cast<const float*>(v.data());
            };
            std::printf("Real %s HC attention weights layer %d\n",is_draft?"DSpark":"target",l);
            hc(5120,tensor("hc_attn_fn.weight",{20480,24}),tensor("hc_attn_scale.weight",{3}),
               tensor("hc_attn_base.weight",{24}),tensor("attn_norm.weight",{5120}),rng);
        }
    }
    std::printf("deepseek41_ops_test: %d failures\n",failures);
    return failures?1:0;
}
int main(int argc,char** argv) {
    try{return run(argc,argv);} catch(const std::exception& e){std::printf("deepseek41_ops_test: %s\n",e.what());return 1;}
}
