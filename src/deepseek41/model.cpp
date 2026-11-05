#include "model_state.hpp"
#include "basalt/platform/system.hpp"
#include "expert_tiers.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>

namespace basalt::deepseek41 {
namespace {
constexpr int H=Shape::hidden;
std::vector<float> frequencies(bool compressed) {
    std::vector<float> f(32);
    const double base=compressed?160000.:10000.;
    const double low=std::max(0.,std::floor(64*std::log(65536./(32*2*3.141592653589793))/(2*std::log(base))));
    const double high=std::min(63.,std::ceil(64*std::log(65536./(2*3.141592653589793))/(2*std::log(base))));
    for(int i=0;i<32;++i) {
        const double raw=std::pow(base,-double(2*i)/64);
        const double ramp=std::clamp((i-low)/std::max(high-low,.001),0.,1.);
        f[i]=float(compressed?raw*(1-ramp)+raw/16*ramp:raw);
    }
    return f;
}
}
HostTransferStorage::HostTransferStorage(size_t bytes,bool pin,bool map) {
    if(pin) {
        const auto status=cudaHostAlloc(reinterpret_cast<void**>(&pinned),bytes,cudaHostAllocPortable|(map?cudaHostAllocMapped:0));
        if(status==cudaSuccess) {
            if(map) {
                const auto mapped_status=cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),pinned,0);
                if(mapped_status!=cudaSuccess) {
                    mapped=nullptr;cudaGetLastError();
                    basalt::log::warn("basalt deepseek41","transfer buffer mapping: %s",cudaGetErrorString(mapped_status));
                }
            }
            return;
        }
        pinned=nullptr;cudaGetLastError();
        basalt::log::warn("basalt deepseek41","transfer buffer pinning: %s",cudaGetErrorString(status));
    }
    pageable.resize(bytes);
}
HostTransferStorage::~HostTransferStorage(){if(pinned)cudaFreeHost(pinned);}
RouteTransport::RouteTransport() {
    const core::OnDevice on(0);
    cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const auto status=cudaEventCreateWithFlags(&ready,cudaEventDisableTiming);
    if(status!=cudaSuccess) {cudaStreamDestroy(stream);stream=nullptr;cuda_check(status);}
}
RouteTransport::~RouteTransport() {
    const core::OnDevice on(0);
    if(stream) {cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}
    if(ready)cudaEventDestroy(ready);
}
Work::Work(DeviceModel& d,int context,bool draft,int capacity,bool tensor) {
    const int T=capacity;
    auto f=[&](size_t n){return d.alloc<float>(n);};
    auto i=[&](size_t n){return d.alloc<int32_t>(n);};
    h=f(T*4*H);x=f(T*H);qr=f(T*1280);q=f(T*64*512);kv=f(T*512);attn=f(T*64*512);out_a=f(T*8192);
    branch=f(T*H);mix=f(T*24);pre=f(T*4);a_pre=f(T*4);f_pre=f(T*4);post=f(T*4);comb=f(T*16);
    gate=f(T*2304);up=f(T*2304);activated=f(T*2304);shared=f(T*H);router=f(T*384);weights=f(T*6);
    gpu_parts=f(T*6*H);cpu_parts=f(T*6*H);head_x=f(T*H);
    logit_capacity=std::min(T,kAttentionWindow);logits=f(logit_capacity*129280);
    logit_check=d.allocate(finite_argmax_workspace_bytes(Shape::vocab),MemoryKind::scratch);
    ids=i(T*6);quant=d.allocate(T*H/32*36,MemoryKind::scratch);
    expert_scratch=d.allocate(kernels::native_expert_scratch_bytes(T*6,2304),MemoryKind::scratch);
    group_pointer=d.alloc<unsigned long long>(T*6);group_start=i(T*6+1);group_count=i(1);
    entry_token=i(T*6);entry_destination=i(T*6);selected=i(T*512);
    index_q=f(draft?1:T*32*128);index_weights=f(draft?1:T*32);
    score_stride=draft?1:context;candidate_stride=draft?1:(context+7)/8;
    score_rows=draft?T:std::min(T,tensor?kAttentionWindow:8);
    scores=f(score_rows*score_stride);sorted_scores=f(score_stride);block_scores=f(candidate_stride);
    sort_ids=i(score_stride);sorted_ids=i(score_stride);candidate_ids=i(2048);
    sort_bytes=topk_workspace_bytes(score_stride);
    sort_workspace=d.allocate(sort_bytes,MemoryKind::scratch);
    candidates=d.allocate(size_t(T)*candidate_stride,MemoryKind::scratch);
    engram_raw=d.allocate(draft?1:T*24*144,MemoryKind::scratch);
    engram_rows=f(draft?1:T*6144);engram_projected=f(draft?1:T*25600);anchors=f(T*15360);
    freq_window=f(32);freq_compressed=f(32);
    const auto a=frequencies(false),b=frequencies(true);
    d.upload(freq_window,a.data(),128);d.upload(freq_compressed,b.data(),128);d.synchronize();
    const size_t nx=size_t(T)*H,np=size_t(T)*6*H,ni=size_t(T)*6;
    const bool direct=!draft && core::exp_flag("ds-direct-host")!=0;
    host_transfer=std::make_unique<HostTransferStorage>((nx+np)*sizeof(float)+ni*sizeof(int32_t),
                    direct || core::exp_flag("ds-pinned-transfers")!=0 || (!draft && core::exp_flag("ds-device-route")!=0),direct);
    auto* host=host_transfer->data();
    host_x={reinterpret_cast<float*>(host),nx};host_parts={reinterpret_cast<float*>(host+nx*sizeof(float)),np};
    host_ids={reinterpret_cast<int32_t*>(host+(nx+np)*sizeof(float)),ni};host_kind.resize(T*6);
    if(host_transfer->mapped)mapped_parts=reinterpret_cast<float*>(host_transfer->mapped+nx*sizeof(float));
    host_pointer.reserve(T*6);host_start.reserve(T*6+1);host_entry_token.reserve(T*6);host_entry_destination.reserve(T*6);
}

Model::Impl::Impl(const std::string& path,RunOptions opts)
    : packed(path,opts.draft_path),options(opts),cpu(opts.workers),hasher(packed.metadata_file(),packed.token_map()) {
    expect(core::exp_flag("ds-attention-check")==0 || core::exp_flag("ds-graph")==0,"attention check requires uncaptured execution");
    expect(options.context>=16 && options.context<=packed.contract().context,"invalid context capacity");
    expect(options.slots>=1 && options.slots<=8,"slots must be in 1..8");
    expect(options.reserve_mib>=200 && options.helper_reserve_mib>=200,"reserve must be at least 200 MiB");
    expect(options.tensor_prefill_tokens>=16 && options.tensor_prefill_tokens<=kPromptWindow,"invalid tensor prefill capacity");
    expect(options.engram_inflight>=1 && options.engram_inflight<=4096 && options.engram_cache_rows<=hasher.table_rows(),"invalid Engram reader capacity");
    const auto& meta=packed.metadata_file().metadata();
    for(const auto& [key,value]:std::vector<std::pair<std::string,int>>{
        {"attention.head_count",64},{"attention.head_count_kv",1},{"attention.key_length",512},{"attention.value_length",512},
        {"attention.q_lora_rank",1280},{"attention.sliding_window",128},{"attention.output_group_count",8},{"attention.output_lora_rank",1024},
        {"attention.indexer.head_count",32},{"attention.indexer.key_length",128},{"attention.indexer.top_k",512},
        {"attention.indexer.candidate_source_layer",20},{"attention.indexer.candidate_top_k_blocks",2048},{"attention.indexer.candidate_block_size",8},
        {"rope.dimension_count",64},{"rope.scaling.original_context_length",65536},{"expert_shared_count",1}})
        expect(integer(meta,"deepseek41."+key)==value,"unsupported target geometry: "+key);
    expect(integers(meta,"deepseek41.attention.kv_source_layers")==std::vector<int>({2,8,14,20}) &&
           integers(meta,"deepseek41.attention.indexer.source_layers")==std::vector<int>({2,8,14,20,24,28,32,36}) &&
           integers(meta,"deepseek41.ple.layers")==std::vector<int>({1,14}),"unsupported attention/Engram source schedule");
    for(int layer=0;layer<40;++layer)expect(packed.contract().compress_ratios[layer]==(layer<2?0:layer<20?2:1),"unsupported compression schedule");
    for(const auto& [key,value]:std::vector<std::pair<std::string,double>>{
        {"rope.freq_base",10000.},{"attention.compress_rope_freq_base",160000.},{"rope.scaling.factor",16.},
        {"rope.scaling.yarn_beta_fast",32.},{"rope.scaling.yarn_beta_slow",1.},
        {"attention.layer_norm_rms_epsilon",1e-20},{"hyper_connection.epsilon",1e-6}}) {
        const auto& v=required(meta,"deepseek41."+key);
        expect(v.is_num() && std::abs(v.num()-value)<=value*1e-6,"unsupported target parameter: "+key);
    }
    const auto& draft_meta=packed.draft_metadata_file().metadata();
    expect(integer(draft_meta,"dflash.hyper_connection.single_pass")==1,"unsupported DSpark hyper-connection graph");
    history=hasher.initial_history();
    int devices=0;cuda_check(cudaGetDeviceCount(&devices));
    if(options.draft_device<0)options.draft_device=devices>1?1:0;
    if(options.helper_device==-1)options.helper_device=devices>1?1:-2;
    expect(options.helper_device>=-2 && options.helper_device!=0 && options.helper_device<devices,"invalid expert helper device");
    expect(options.draft_device>=0 && options.draft_device<devices,"invalid DSpark device");
    basalt::log::info("basalt deepseek41","loading target, context %d, CPU workers %d",options.context,cpu.workers());
    const int tile=options.slots>1 && options.tensor_prefill_tokens>kPrefillWindow?kConcurrentPrefill:kPrefillWindow;
    window_capacity=options.tensor_prefill?std::min(options.tensor_prefill_tokens,tile):kAttentionWindow;
    primary=std::make_unique<DeviceModel>(packed,false,0,window_capacity);
    main_work=std::make_unique<Work>(*primary,options.context,false,window_capacity,options.tensor_prefill);
    if(options.tensor_prefill && options.tensor_prefill_tokens>window_capacity)
        long_work=std::make_unique<LongWork>(*primary,options.context,options.tensor_prefill_tokens);
    for(int l=0;l<40;++l)layers.push_back(make_layer(*primary,l,false));
    initialize_kv(*this);
    if(options.dspark) {
        basalt::log::info("basalt deepseek41","loading DSpark on device %d%s",options.draft_device,
                          options.draft_path.empty()?"":" (external draft)");
        drafter=std::make_unique<DeviceModel>(packed,true,options.draft_device,Shape::draft_window);
        if(core::exp_flag("ds-draft-shared-attention")!=0)
            draft_attention_scratch=drafter->alloc<float>(kDraftAttentionScratch);
        if(core::exp_flag("ds-draft-gpu-stages")!=0)
            draft_gpu_stages=std::make_unique<DraftGpuStages>(drafter->device());
        draft_work=std::make_unique<Work>(*drafter,options.context,true,Shape::draft_window);
        for(int l=0;l<3;++l)draft_layers.push_back(make_layer(*drafter,l,true));
        draft_experts=drafter->allocate(packed.draft_expert_layout().total,MemoryKind::draft);
        drafter->load_draft_experts(draft_experts,packed);
        draft_lookup=drafter->alloc<unsigned long long>(3*128);
        std::vector<unsigned long long> draft_table(3*128);
        for(int l=0;l<3;++l)for(int e=0;e<128;++e)
            draft_table[l*128+e]=reinterpret_cast<unsigned long long>(draft_experts+packed.draft_expert_layout().blob_offset(l,e));
        drafter->upload(draft_lookup,draft_table.data(),draft_table.size()*sizeof(draft_table[0]));
        std::vector<int32_t> draft_selection(5*512,0);
        for(int t=0;t<5;++t)for(int j=0;j<5;++j)draft_selection[t*512+j]=j;
        drafter->upload(draft_work->selected,draft_selection.data(),draft_selection.size()*sizeof(int32_t));
        draft_main_x=drafter->alloc<float>(Shape::draft_window*H);draft_logits=drafter->alloc<float>(5*Shape::vocab);
        markov_x=drafter->alloc<float>(256);markov_bias=drafter->alloc<float>(Shape::vocab);
        if(core::exp_flag("ds-draft-markov-cache")!=0 && drafter->tensor("markov_w2.weight").type==30) {
            markov_cache_keys=drafter->alloc<int32_t>(kMarkovCacheSlots);
            markov_cache=drafter->alloc<float>(size_t(kMarkovCacheSlots)*Shape::vocab);
            const core::OnDevice on(drafter->device());
            cuda_check(cudaMemsetAsync(markov_cache_keys,0xff,kMarkovCacheSlots*sizeof(int32_t),
                                       static_cast<cudaStream_t>(drafter->stream())));
        }
        confidence_input=drafter->alloc<float>(5*(H+256));confidence_output=drafter->alloc<float>(5);
        draft_picks=drafter->alloc<int32_t>(7);
        draft_pick_workspace=drafter->allocate(finite_argmax_workspace_bytes(Shape::vocab),MemoryKind::scratch);
        if(options.draft_device!=0) {
            std::string error;
            expect(to_draft.open(0,options.draft_device,5*Shape::vocab*sizeof(float),2,error),error);
            expect(to_primary.open(options.draft_device,0,Shape::draft_window*H*sizeof(float),2,error),error);
        }
        drafter->synchronize();
    }
    const auto table=packed.target_tensor("per_layer_token_embd.weight");
    std::string error;
    expect(engram_reader.open(table.file->path(),table.file->data_start()+table.tensor->offset,
                             hasher.table_rows(),uint32_t(options.engram_inflight),options.engram_cache_rows,error,144),error);
    engram_reader.set_keepalive(100.,30.);
    host_engram.resize(window_capacity*48*144);host_embedding.resize(window_capacity*H);row_ids.resize(window_capacity*48);
    primary->zero(main_work->x,window_capacity*H*sizeof(float));
    primary->linear("output.weight",main_work->x,main_work->logits,1);
    primary->synchronize();
    initialize_slots();
    initialize_cache();
    if(long_work)basalt::log::info("basalt deepseek41","prompt chunk %d, physical tile %d, helper weight cache",
                                  options.tensor_prefill_tokens,window_capacity);
    usage.resize(40*384);primary_residency.resize(40*384,-1);
    for(int l=0;l<40;++l)for(int e=0;e<384;++e)primary_residency[l*384+e]=cache.slot_of(l,e);
    refresh_primary_lookup();
    if(profile.front_enabled)for(const auto& l:layers)
        basalt::log::info("basalt deepseek41","projection layer=%d qa=%u qb=%u kv=%u oa=%u ob=%u",l.id,l.qa->type,l.qb->type,l.kv->type,l.oa->type,l.ob->type);
    basalt::log::info("basalt deepseek41","ready, DSpark %s",drafter?"enabled":"disabled");
}
Model::Impl::~Impl() {
    if(batch_stage)basalt::log::info("basalt deepseek41","decode staging copies=%llu rows=%llu",
        (unsigned long long)batch_staged_experts,(unsigned long long)batch_staged_rows);
    if(core::exp_flag("ds-engram-metrics")!=0 && engram_reader.is_open()) {
        const auto e=engram_reader.snapshot();
        basalt::log::info("basalt deepseek41","engram rows=%llu hits=%llu reads=%llu wait_ms=%.3f submit_ms=%.3f p50_us=%.1f p95_us=%.1f p99_us=%.1f",
            (unsigned long long)e.requests,(unsigned long long)e.cache_hits,(unsigned long long)e.reads,
            e.wait_us/1000.,e.submit_us/1000.,e.percentile(.5),e.percentile(.95),e.percentile(.99));
    }
    profile.report();
    if(draft_gpu_stages)draft_gpu_stages->report();
    uint64_t updates=0,rebuilds=0,cached=0;
    const auto count_graphs=[&](const std::vector<Layer>& ls) {
        for(const auto& l:ls) {
            for(const auto* graph:{l.front_graph.get(),l.expert_graph.get()})if(graph) {
                updates+=graph->updates;rebuilds+=graph->rebuilds;cached+=graph->cached_launches;
            }
            for(const auto* group:{&l.verify_graphs,&l.verify_front_graphs})
                for(const auto& graph:*group)if(graph) {
                    updates+=graph->updates;rebuilds+=graph->rebuilds;cached+=graph->cached_launches;
                }
        }
    };
    count_graphs(layers);for(const auto& sequence:sequences)count_graphs(sequence.layers);
    if(updates || rebuilds)basalt::log::info("basalt deepseek41","decode graphs updates=%llu rebuilds=%llu",
                                           (unsigned long long)updates,(unsigned long long)rebuilds);
    if(cached)basalt::log::info("basalt deepseek41","static expert graph launches=%llu",(unsigned long long)cached);
    if(helper.single_token_graph_launches())basalt::log::info("basalt deepseek41","helper graphs launches=%llu shapes=%zu",
        (unsigned long long)helper.single_token_graph_launches(),helper.single_token_graph_shapes());
    if(core::exp_flag("ds-cache-audit")!=0) {
        if(helper.valid())helper.apply_pending(true);
        struct Bin {std::vector<float> heat;double primary=0,cpu=0;int slots=0;};
        std::map<uint64_t,Bin> bins;
        for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
            const bool resident=primary_slot(l,e)>=0;
            if(!resident && ((helper.valid() && helper.has(l,e)) || !source.has_resident(l,e)))continue;
            auto& bin=bins[packed.expert_layout().blob_bytes(l)];
            const float heat=usage[size_t(l)*384+e];bin.heat.push_back(heat);
            if(resident) {++bin.slots;bin.primary+=heat;}else bin.cpu+=heat;
        }
        for(auto& [bytes,bin]:bins) {
            std::sort(bin.heat.begin(),bin.heat.end(),std::greater<float>());
            double best=0;for(int i=0;i<bin.slots;++i)best+=bin.heat[size_t(i)];
            basalt::log::info("deepseek41","cache audit bytes=%llu slots=%d primary_heat=%.0f cpu_heat=%.0f cross_layer_gain=%.0f",
                (unsigned long long)bytes,bin.slots,bin.primary,bin.cpu,best-bin.primary);
        }
    }
    // The helper can still own refills whose source or device storage belongs to this model.
    helper.close();
    prompt_experts.reset();
    prompt_index.reset();
    if(primary) { const core::OnDevice on(0);primary->synchronize();cache.close(); }
    if(drafter)drafter->synchronize();
    engram_reader.close();
    // Release graph handles and pinned sequence buffers while their device owners are alive.
    sequences.clear();kv.reset();layers.clear();draft_layers.clear();draft_pick_graph.reset();draft_body_graph.reset();
    main_work.reset();draft_work.reset();
}
Layer Model::Impl::make_layer(DeviceModel& d,int id,bool draft) {
    const int T=draft?Shape::draft_window:window_capacity;
    Layer l;l.id=id;l.prefix="blk."+std::to_string(id)+".";const auto& p=l.prefix;
    l.experts=draft?128:384;l.top=draft?3:6;l.ratio=draft?0:packed.contract().compress_ratios[id];
    const auto& meta=packed.metadata_file().metadata();
    if(!draft) {
        const auto sources=integers(meta,"deepseek41.attention.kv_source_layers");
        const auto indices=integers(meta,"deepseek41.attention.indexer.source_layers");
        l.kv_source=std::find(sources.begin(),sources.end(),id)!=sources.end();
        l.index_source=std::find(indices.begin(),indices.end(),id)!=indices.end();
        for(int source:sources)if(source<=id)l.source=source;
        l.engram=id==1 || id==14;
    }
    l.qa=&d.tensor(p+"attn_q_a.weight",{H,1280});l.qb=&d.tensor(p+"attn_q_b.weight",{1280,32768});
    l.kv=&d.tensor(p+"attn_kv.weight",{H,512});l.oa=&d.tensor(p+"attn_output_a.weight",{4096,8192});
    l.ob=&d.tensor(p+"attn_output_b.weight",{8192,H});l.router=&d.tensor(p+"ffn_gate_inp.weight",{H,uint64_t(l.experts)});
    l.shared_gate=&d.tensor(p+"ffn_gate_shexp.weight",{H,2304});l.shared_up=&d.tensor(p+"ffn_up_shexp.weight",{H,2304});
    l.shared_down=&d.tensor(p+"ffn_down_shexp.weight",{2304,H});
    l.attn_norm=d.f32(p+"attn_norm.weight",{H});l.ffn_norm=d.f32(p+"ffn_norm.weight",{H});
    l.q_norm=d.f32(p+"attn_q_a_norm.weight",{1280});l.kv_norm=d.f32(p+"attn_kv_a_norm.weight",{512});
    l.sinks=d.f32(p+"attn_sinks.weight",{64});l.bias=d.f32(p+"exp_probs_b.bias",{uint64_t(l.experts)});
    if(!draft && core::exp_flag("ds-vision")!=0)l.image_bias=d.f32(p+"exp_probs_b_vl.bias",{uint64_t(l.experts)});
    l.a_fn=d.f32(p+"hc_attn_fn.weight",{20480,24});l.a_base=d.f32(p+"hc_attn_base.weight",{24});l.a_scale=d.f32(p+"hc_attn_scale.weight",{3});
    l.f_fn=d.f32(p+"hc_ffn_fn.weight",{20480,24});l.f_base=d.f32(p+"hc_ffn_base.weight",{24});l.f_scale=d.f32(p+"hc_ffn_scale.weight",{3});
    auto state=[&](size_t bytes){return d.allocate(bytes,MemoryKind::state);};
    l.ring=state(128*528);l.extended=state((128+(draft?Shape::draft_window:T))*528);d.zero(l.ring,128*528);
    if(l.kv_source) {
        l.compress_kv=&d.tensor(p+"attn_compressor_kv.weight",{H,512});l.compress_norm=d.f32(p+"attn_compressor_norm.weight",{512});
        if(l.ratio==2)l.compress_gate=&d.tensor(p+"attn_compressor_gate.weight",{H,512});
        if(!options.elastic_kv) {
            l.compressed=state(size_t(options.context/l.ratio)*288);
            l.index_keys=state(size_t(options.context/l.ratio)*68);
        }
        l.raw_kv=d.alloc<float>(T*512);l.raw_gate=d.alloc<float>(T*512);l.pooled=d.alloc<float>(T*512);l.index_latent=d.alloc<float>(T*128);
        l.kv_history=d.alloc<float>(1024,MemoryKind::state);l.score_history=d.alloc<float>(1024,MemoryKind::state);
        l.saved_kv=d.alloc<float>(1024);l.saved_score=d.alloc<float>(1024);
        d.zero(l.kv_history,1024*sizeof(float));d.zero(l.score_history,1024*sizeof(float));
        l.index_k=&d.tensor(p+"indexer.attn_k.weight",{512,128});l.index_norm=d.f32(p+"indexer.k_norm.weight",{128});
    }
    if(l.index_source) {l.index_q=&d.tensor(p+"indexer.attn_q_b.weight",{1280,4096});l.index_weights=&d.tensor(p+"indexer.proj.weight",{H,32});}
    if(l.engram) {l.engram_kv=&d.tensor(p+"engram_kv.weight",{6144,25600});l.engram_gate=d.f32(p+"engram_gate.weight",{H,4});}
    return l;
}
void Model::Impl::embeddings(const std::vector<int32_t>& tokens,DeviceModel& d,Work& w) {
    const auto emb=packed.target_tensor("token_embd.weight");const auto* traits=ggml_get_type_traits(ggml_type(emb.tensor->type));
    expect(traits && traits->to_float,"missing embedding decoder");
    const size_t bytes=size_t(emb.bytes/Shape::vocab);
    for(size_t t=0;t<tokens.size();++t) {
        expect(tokens[t]>=0 && tokens[t]<Shape::vocab,"token outside vocabulary");
        if(d.device()==0 && &d==primary.get() && !image_rows.empty()) {
            for(int c=0;c<H;++c)host_embedding[t*H+c]=std::bit_cast<float>(uint32_t(image_rows[t*H+c])<<16);
        } else traits->to_float(emb.data()+size_t(tokens[t])*bytes,host_embedding.data()+t*H,H);
    }
    const size_t transfer=tokens.size()==1?profile.begin_async("embedding",d.stream()):size_t(-1);
    d.upload(w.x,host_embedding.data(),tokens.size()*H*sizeof(float));
    expand_embedding(w.x,w.h,w.pre,H,int(tokens.size()),d.stream());
    profile.end_async(transfer,d.stream());
    d.synchronize();
}
void Model::Impl::block_prefix(DeviceModel& d,Work& w,Layer& l,int p0,int count,bool draft) {
    const core::OnDevice on(d.device());auto stream=d.stream();
    const bool detailed=profile.front_enabled && !prefill_active && !draft && count==1;
    const bool long_metrics=profile.prefill_enabled && !draft && long_work && prefill_active;
    if(draft && draft_gpu_stages)draft_gpu_stages->record(l.id,0,stream);
    const size_t before_attention=(detailed || long_metrics)?profile.begin_async(long_metrics?"long_hc_before_attention":"front_hc_before_attention",stream):size_t(-1);
    const auto project=count>11 && prefill_active &&
        core::exp_flag("ds-prefill-hc-grouped",long_work && count>=128?1:0)!=0?hc_project_grouped:hc_project;
    project(w.h,l.a_fn,w.mix,H,count,1e-20f,stream);
    hc_split(w.mix,l.a_scale,l.a_base,w.a_pre,w.post,w.comb,count,20,1e-6f,stream);
    hc_reduce(w.h,w.pre,l.attn_norm,w.x,H,count,1e-20f,stream);
    profile.end_async(before_attention,stream);
    if(draft && draft_gpu_stages)draft_gpu_stages->record(l.id,1,stream);
    attention(d,w,l,p0,count,draft);
    if(draft && draft_gpu_stages)draft_gpu_stages->record(l.id,2,stream);
    const size_t after_attention=(detailed || long_metrics)?profile.begin_async(long_metrics?"long_hc_after_attention":"front_hc_after_attention",stream):size_t(-1);
    hc_expand(w.branch,w.h,w.post,w.comb,w.h,H,count,stream);
    project(w.h,l.f_fn,w.mix,H,count,1e-20f,stream);
    hc_split(w.mix,l.f_scale,l.f_base,w.f_pre,w.post,w.comb,count,20,1e-6f,stream);
    hc_reduce(w.h,w.a_pre,l.ffn_norm,w.x,H,count,1e-20f,stream);
    profile.end_async(after_attention,stream);
    const size_t router=(detailed || long_metrics)?profile.begin_async(long_metrics?"long_router":"front_router",stream):size_t(-1);
    {
        AsyncGpuScope projection(long_metrics?&profile:nullptr,"long_router_projection",stream);
        if(count>11 && !draft && prefill_active && l.router->type==0 && core::exp_flag("ds-prefill-tile-router-linear")!=0)
            router_linear_tiled(reinterpret_cast<const float*>(l.router->data),w.x,w.router,count,stream,
                                core::exp_flag("ds-prefill-router-tile8")!=0?8:4,
                                int(core::exp_flag("ds-prefill-router-expert-tile",long_work && count>=128?4:1)));
        else d.linear(*l.router,w.x,w.router,count);
    }
    {
        AsyncGpuScope selection(long_metrics?&profile:nullptr,"long_router_selection",stream);
        route(w.router,!draft && !image_rows.empty()?l.image_bias:l.bias,w.ids,w.weights,l.experts,l.top,count,1.5f,stream);
    }
    profile.end_async(router,stream);
}
void Model::Impl::block(DeviceModel& d,Work& w,Layer& l,int p0,int count,bool draft) {
    const core::OnDevice on(d.device());
    enqueue_prefix(d,w,l,p0,count,draft);
    if(draft && draft_gpu_stages)draft_gpu_stages->record(l.id,3,d.stream());
    expert_layer(d,w,l,count,draft);
    {AsyncGpuScope timing(count==1 && !draft?&profile:nullptr,"layer_tail",d.stream());
    hc_expand(w.branch,w.h,w.post,w.comb,w.h,H,count,d.stream());}
    std::swap(w.pre,w.f_pre);
}
void Model::Impl::enqueue_prefix(DeviceModel& d,Work& w,Layer& l,int p0,int count,bool draft) {
    const core::OnDevice on(d.device());auto stream=d.stream();
    const auto front=[&]{block_prefix(d,w,l,p0,count,draft);};
    if(!draft && image_rows.empty() && ((count==1 && !drafter) || (count<=6 && core::exp_flag("ds-verify-front",1)!=0)) &&
       !profile.gpu_enabled && core::exp_flag("ds-graph")!=0) {
        // Verification widths can have different graph topologies. Keep a
        // bounded executable per width instead of discarding it on each switch.
        auto& graph=count>=2 && core::exp_flag("ds-verify-front-width-cache")!=0?
            l.verify_front_graphs[size_t(count-2)]:l.front_graph;
        if(!graph)graph=std::make_unique<UpdateGraph>();
        graph->run(stream,front,&profile,"front_graph");
    } else front();
}
void Model::Impl::engram(Layer& l,int count,int first_token) {
    std::string error;
    if(engram_pending) {
        ProfileScope waited(profile,"engram_collect");
        expect(engram_reader.collect(engram_ticket,error),error);engram_pending=false;
    }
    auto& d=*primary;auto& w=*main_work;const core::OnDevice on(0);
    AsyncGpuScope timing(count==1?&profile:nullptr,"engram",d.stream());
    cuda_check(cudaMemcpy2DAsync(w.engram_raw,24*144,host_engram.data()+size_t(first_token)*48*144+(l.id==1?0:24*144),48*144,
                                24*144,count,cudaMemcpyHostToDevice,static_cast<cudaStream_t>(d.stream())));
    kernels::iq_dequant_f32(20,w.engram_raw,count*24*256,w.engram_rows,d.stream());
    d.linear(*l.engram_kv,w.engram_rows,w.engram_projected,count);
    engram_add(w.h,w.engram_projected,l.engram_gate,H,count,1e-20f,d.stream());
}
ForwardResult Model::Impl::forward(const std::vector<int32_t>& tokens,LogitsMode mode,bool copy_logits) {
    profile.select_position(pos,!prefill_active);
    ProfileScope timer(profile,"target_forward");
    try {
        const core::OnDevice on(0);
        std::string error;
        expect(fit_kv(pos+int(tokens.size()),error),error);
        begin_forward(tokens);
        auto& d=*primary;auto& w=*main_work;const int count=int(tokens.size());
        for(auto& l:layers) {
            if(l.engram)engram(l,count);
            if(l.id>=37)stream_mean(w.h,w.anchors,H,count,15360,size_t(l.id-37)*H,d.stream());
            block(d,w,l,pos,count,false);
        }
        return finish_forward(mode,copy_logits);
    } catch(...) {broken=true;throw;}
}
void Model::Impl::begin_forward(const std::vector<int32_t>& tokens) {
    expect(!broken && !pending,"unfinished forward window or failed model");
    const int count=int(tokens.size());
    expect(count>0 && count<=window_capacity && pos<=options.context-count,"invalid forward window");
    for(int t:tokens)expect(t>=0 && t<Shape::vocab,"token outside vocabulary");
    finish_deferred_adapt();
    if(helper.valid()) {ProfileScope waited(profile,"helper_pending");helper.apply_pending(true);}
    const core::OnDevice on(0);auto& d=*primary;auto& w=*main_work;
    base_pos=pos;saved_history=history;pending_tokens=tokens;
    if(!image_rows.empty()) {
        history=hasher.initial_history();
        std::fill_n(row_ids.begin(),size_t(count)*48,UINT32_MAX);
    } else for(int t=0;t<count;++t) {
        const auto rows=hasher.advance(tokens[t],history);
        std::copy(rows.begin(),rows.end(),row_ids.begin()+t*48);
    }
    {ProfileScope submitted(profile,"engram_issue");
    engram_ticket=engram_reader.issue(row_ids.data(),count*48,host_engram.data());engram_pending=true;}
    embeddings(tokens,d,w);
}
ForwardResult Model::Impl::finish_forward(LogitsMode mode,bool copy_logits) {
    const core::OnDevice on(0);auto& d=*primary;auto& w=*main_work;
    const int count=int(pending_tokens.size());
    const size_t head=count==1?profile.begin_async("head",d.stream()):size_t(-1);
    hc_reduce(w.h,w.pre,d.f32("output_norm.weight",{H}),w.head_x,H,count,1e-20f,d.stream());
    ForwardResult result;
    if(mode!=LogitsMode::none) {
        primary->set_prefill_tensor(false);
        result.rows=mode==LogitsMode::all?count:1;
        expect(copy_logits || result.rows<=w.logit_capacity,"device logits exceed work capacity");
        if(copy_logits)result.logits.resize(size_t(result.rows)*Shape::vocab);
        for(int first=0;first<result.rows;first+=w.logit_capacity) {
            const int nr=std::min(w.logit_capacity,result.rows-first);
            d.linear("output.weight",w.head_x+(mode==LogitsMode::all?first:(count-1))*H,w.logits,nr);
            if(first==0)profile.end_async(head,d.stream());
            if(copy_logits)d.read(result.logits.data()+size_t(first)*Shape::vocab,w.logits,size_t(nr)*Shape::vocab*sizeof(float));
        }
        if(copy_logits)expect(std::all_of(result.logits.begin(),result.logits.end(),[](float x){return std::isfinite(x);}),"nonfinite target logits");
        logits_checked=copy_logits;
        primary->set_prefill_tensor(prefill_tensor);
    } else {profile.end_async(head,d.stream());d.synchronize();}
    if(!copy_logits && profile.async_enabled)d.synchronize();
    profile.collect_async();
    logit_rows=result.rows<=w.logit_capacity?result.rows:0;pos+=count;pending=true;return result;
}
void Model::Impl::adapt_after_commit(int primary_limit) {
    if(cooperative_prefill)return;
    const core::OnDevice on(0);
    std::string error;
    if(core::exp_flag("ds-primary-adapt",1)!=0) {
        ProfileScope timer(profile,"primary_adapt");
        if(helper.valid())helper.apply_pending(true);
        const auto before=execution.primary_swaps;
        expect(adapt_primary(cache,source,helper,usage.data(),40,384,primary_limit,primary_residency.data(),
                             static_cast<cudaStream_t>(primary->stream()),execution.primary_swaps,execution.peer_promotions,error),error);
        if(execution.primary_swaps!=before)refresh_primary_lookup();
    }
    if(helper.valid()) {
        ProfileScope timer(profile,"helper_adapt");
        expect(helper.adapt(usage.data(),primary_residency.data(),8,error),error);
    }
}
void Model::Impl::finish_deferred_adapt() {
    if(!deferred_adapt_limit)return;
    const int limit=deferred_adapt_limit;
    deferred_adapt_limit=0;
    try {adapt_after_commit(limit);}
    catch(...) {broken=true;throw;}
}
void Model::Impl::commit(int accepted,bool overlap_next_draft) {
    expect(!broken && pending && accepted>=0 && accepted<=int(pending_tokens.size()),"invalid forward commit");
    try {
        const core::OnDevice on(0);auto& d=*primary;const auto stream=static_cast<cudaStream_t>(d.stream());
        const int proposed=int(pending_tokens.size());
        if(accepted!=proposed && proposed<=128 && core::exp_flag("ds-rollback-window")!=0) {
            ProfileScope rollback(profile,"rollback_window_enqueue");
            WindowRollback window{};
            for(const auto& l:layers) {
                auto& row=window.layers[l.id];
                row.ring=l.ring;row.extended=l.extended;
                if(l.kv_source && l.ratio==2) {
                    row.kv_history=l.kv_history;row.score_history=l.score_history;
                    row.saved_kv=l.saved_kv;row.saved_score=l.saved_score;
                    row.raw_kv=l.raw_kv;row.raw_gate=l.raw_gate;
                }
            }
            // Compressed rows outside the committed position are invisible. Only
            // the rolling history is needed to construct the next compressed row.
            rollback_window(window,base_pos,proposed,accepted,stream);
        } else if(accepted!=proposed)for(auto& l:layers) {
            const int start=base_pos%128;
            cuda_check(cudaMemcpyAsync(l.ring+start*528,l.extended,(128-start)*528,cudaMemcpyDeviceToDevice,stream));
            if(start)cuda_check(cudaMemcpyAsync(l.ring,l.extended+(128-start)*528,start*528,cudaMemcpyDeviceToDevice,stream));
            update_window_ring(l.ring,l.extended,base_pos,accepted,528,d.stream());
            if(l.kv_source && l.ratio==2) {
                d.copy(l.kv_history,l.saved_kv,1024*sizeof(float));d.copy(l.score_history,l.saved_score,1024*sizeof(float));
                for(int t=0;t<accepted;++t)compress_step(l.raw_kv+t*512,l.raw_gate+t*512,l.kv_history,l.score_history,l.compress_norm,l.pooled,
                                                      512,2,base_pos+t,1e-20f,d.stream());
            }
        }
        {ProfileScope sync(profile,"target_commit_sync");d.synchronize();}
        history=saved_history;
        if(accepted && !image_rows.empty())history=hasher.initial_history();
        else for(int t=0;t<accepted;++t)hasher.advance(pending_tokens[t],history);
        pos=base_pos+accepted;
        if(drafter && accepted)seed_draft(base_pos,accepted);
        if(!options.freeze_placement && accepted) {
            const int limit=prefill_active?256:8*accepted;
            if(overlap_next_draft)deferred_adapt_limit=limit;
            else adapt_after_commit(limit);
        }
        pending=false;pending_tokens.clear();
    } catch(...) {broken=true;throw;}
}
void Model::Impl::reset(bool release_kv) {
    expect(!broken,"reload model after an execution failure");
    if(pending)commit(0);
    finish_deferred_adapt();
    if(helper.valid())helper.apply_pending(true);
    pos=base_pos=logit_rows=0;history=hasher.initial_history();draft_window={};logits_checked=true;
    profile.select_position(0,false);
    if(release_kv)trim_kv(0);
}
Model::Model(const std::string& path,RunOptions options):impl_(std::make_unique<Impl>(path,options)){}
Model::~Model()=default;
void Model::reset_metrics(){
    impl_->profile.reset_metrics();
    if(impl_->draft_gpu_stages)impl_->draft_gpu_stages->reset();
    if(core::exp_flag("ds-engram-metrics")!=0)impl_->engram_reader.reset_stats();
}
void Model::reset(){impl_->reset();}
bool Model::prepare_kv(int end_position,std::string& error,bool during_prefill) {
    return impl_->fit_kv(end_position,error,during_prefill);
}
uint64_t Model::trim_kv(bool replenish){return impl_->trim_kv(impl_->pos,replenish);}
uint64_t Model::release_idle_kv(){impl_->reset(false);return impl_->trim_kv(0,false);}
void Model::replenish_expert_cache(){impl_->regrow_experts(0);if(impl_->memory_changed)impl_->memory_changed();}
void Model::resume_turn() {
    expect(!impl_->broken && !impl_->pending,"cannot resume an unfinished prefix");
    impl_->finish_deferred_adapt();
    impl_->draft_window={};
}
namespace {
template<class Visit>
void prefix_regions(const std::vector<Layer>& layers,int position,Visit visit) {
    for(const auto& l:layers) {
        visit(l.ring,size_t(std::min(position,Shape::window))*528);
        if(!l.kv_source)continue;
        const size_t rows=position/l.ratio;
        visit(l.compressed,rows*288);
        visit(l.index_keys,rows*68);
        if(l.ratio==2) {
            visit(l.kv_history,1024*sizeof(float));
            visit(l.score_history,1024*sizeof(float));
        }
    }
}
size_t prefix_bytes(const SequenceState& state,int position) {
    size_t bytes=0;
    const auto add=[&](void*,size_t n){bytes+=n;};
    prefix_regions(state.layers,position,add);
    prefix_regions(state.draft_layers,position,add);
    return bytes;
}
void transfer_prefix(DeviceModel& device,const std::vector<Layer>& layers,int position,
                     uint8_t*& cursor,bool restore) {
    const core::OnDevice on(device.device());
    const auto stream=static_cast<cudaStream_t>(device.stream());
    try {
        prefix_regions(layers,position,[&](void* ptr,size_t bytes) {
            if(!bytes)return;
            cuda_check(restore?cudaMemcpyAsync(ptr,cursor,bytes,cudaMemcpyHostToDevice,stream):
                               cudaMemcpyAsync(cursor,ptr,bytes,cudaMemcpyDeviceToHost,stream));
            cursor+=bytes;
        });
        device.synchronize();
    } catch(...) {
        cudaStreamSynchronize(stream);
        throw;
    }
}
}
size_t Model::prefix_snapshot_bytes() const {
    const auto& s=*impl_;
    if(s.broken || s.pending || s.engram_pending || !s.image_rows.empty() || s.pos<=0)return 0;
    return prefix_bytes(s,s.pos);
}
PrefixSnapshot Model::save_prefix() {
    auto& s=*impl_;
    PrefixSnapshot snapshot;
    snapshot.bytes_=prefix_snapshot_bytes();
    expect(snapshot.bytes_>0,"cannot save an uncommitted prefix");
    snapshot.data_=std::make_unique_for_overwrite<uint8_t[]>(snapshot.bytes_);
    snapshot.owner_=this;snapshot.position_=s.pos;snapshot.history_=s.history.previous;
    auto* cursor=snapshot.data_.get();
    transfer_prefix(*s.primary,s.layers,s.pos,cursor,false);
    if(s.drafter)transfer_prefix(*s.drafter,s.draft_layers,s.pos,cursor,false);
    return snapshot;
}
void Model::restore_prefix(const PrefixSnapshot& snapshot) {
    auto& s=*impl_;
    expect(!s.broken && !s.pending && !s.engram_pending && s.image_rows.empty() &&
           snapshot.owner_==this && snapshot.data_ && snapshot.position_>0 &&
           snapshot.position_<=context() && snapshot.bytes_==prefix_bytes(s,snapshot.position_),
           "cannot restore this prefix");
    std::string error;
    expect(s.fit_kv(snapshot.position_,error),error);
    s.reset(false);
    try {
        auto* cursor=snapshot.data_.get();
        transfer_prefix(*s.primary,s.layers,snapshot.position_,cursor,true);
        if(s.drafter)transfer_prefix(*s.drafter,s.draft_layers,snapshot.position_,cursor,true);
    } catch(...) {s.broken=true;throw;}
    s.pos=s.base_pos=snapshot.position_;
    s.history.previous=snapshot.history_;
    s.saved_history=s.history;
    s.trim_kv(s.pos);
}
bool Model::accepts_images() const {return !impl_->layers.empty() && impl_->layers.front().image_bias;}
ForwardResult Model::prefill_image(std::span<const uint16_t> rows) {
    expect(accepts_images() && !rows.empty() && rows.size()%Shape::hidden==0 &&
           rows.size()/Shape::hidden<=size_t(impl_->window_capacity),"invalid DeepSeek image prefill");
    impl_->image_rows=rows;
    struct Clear {Impl& m;~Clear(){m.image_rows={};}} clear{*impl_};
    const std::vector<int32_t> tokens(rows.size()/Shape::hidden,129264);
    return prefill(tokens,int(tokens.size()));
}
void Model::Impl::pause_prefill(const std::function<void()>& decode) {
    primary->synchronize();
    const int slot=active_slot;
    const bool tensor=prefill_tensor,streaming=prefill_streaming;
    prefill_active=prefill_tensor=prefill_streaming=false;
    primary->set_prefill_tensor(false);
    decode();
    select_slot(slot);
    prefill_active=true;prefill_tensor=tensor;prefill_streaming=streaming;
    primary->set_prefill_tensor(tensor);
    profile.select_position(pos,false);
}
ForwardResult Model::prefill(const std::vector<int32_t>& tokens,int chunk,LogitsMode mode,
                             const std::function<void()>& between_steps,const std::function<void()>& check_stop) {
    const int logical=impl_->long_work?impl_->options.tensor_prefill_tokens:impl_->window_capacity;
    if(!chunk)chunk=logical;
    expect(!impl_->broken && !impl_->pending && !tokens.empty() && chunk>0 && chunk<=logical &&
           tokens.size()<=size_t(context()-position()) &&
           std::all_of(tokens.begin(),tokens.end(),[](int32_t t){return t>=0 && t<Shape::vocab;}),"invalid prefill request");
    expect(mode!=LogitsMode::all || std::min(size_t(chunk),tokens.size())<=size_t(impl_->window_capacity),
           "all-logits long prompts exceed the bounded readback path");
    std::string error;
    expect(impl_->fit_kv(position()+int(tokens.size()),error),error);
    ForwardResult result,all;
    if(mode==LogitsMode::all && tokens.size()>size_t(chunk))all.logits.reserve(tokens.size()*Shape::vocab);
    impl_->prefill_active=true;
    impl_->profile.select_position(position(),false);
    impl_->prefill_tensor=impl_->options.tensor_prefill && core::exp_flag("ds-prefill-tensor",1)!=0;
    impl_->prefill_streaming=impl_->prefill_tensor || (impl_->options.stream_prefill_experts && core::exp_flag("ds-prefill-stream",1)!=0);
    impl_->primary->set_prefill_tensor(impl_->prefill_tensor);
    const auto pool_before=impl_->prompt_experts?impl_->prompt_experts->pool_bytes():0;
    struct Restore {Impl& state;~Restore(){state.prefill_active=state.prefill_streaming=state.prefill_tensor=false;state.primary->set_prefill_tensor(false);}} restore{*impl_};
    for(size_t first=0;first<tokens.size();) {
        if(check_stop)check_stop();
        size_t end=std::min(first+size_t(chunk),tokens.size());
        for(;;) {
            const auto logits=mode==LogitsMode::all?LogitsMode::all:end==tokens.size()?mode:LogitsMode::none;
            std::vector<int32_t> rows(tokens.begin()+first,tokens.begin()+end);
            const size_t tile=impl_->window_capacity;
            if(rows.size()>tile || (impl_->long_work && core::exp_flag("ds-force-long")!=0)) {
                if(auto long_result=impl_->prefill_long(rows,logits,between_steps,check_stop)) {
                    result=std::move(*long_result);
                    break;
                }
                if(rows.size()>tile) {
                    end=first+std::max(tile,(rows.size()*3/4/tile)*tile);
                    basalt::log::info("basalt deepseek41","prefill retry: %zu -> %zu tokens",rows.size(),end-first);
                    if(between_steps)chunk=std::min(chunk,int(end-first));
                    continue;
                }
            }
            result=forward(rows,logits);
            commit(int(rows.size()));
            break;
        }
        if(first/256!=end/256 || end==tokens.size())
            basalt::log::info("basalt deepseek41","prefill %zu/%zu",end,tokens.size());
        if(mode==LogitsMode::all && tokens.size()>size_t(chunk)) {
            all.rows+=result.rows;all.logits.insert(all.logits.end(),result.logits.begin(),result.logits.end());
        }
        first=end;
        if(check_stop)check_stop();
        if(between_steps && first<tokens.size())impl_->pause_prefill(between_steps);
    }
    if(core::exp_flag("ds-check-primary-map")!=0 && impl_->primary_lookup) {
        std::vector<unsigned long long> actual(40*384);
        impl_->primary->read(actual.data(),impl_->primary_lookup,actual.size()*sizeof(actual[0]));
        for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
            const int slot=impl_->primary_slot(l,e);
            const auto expected=slot<0?0ull:reinterpret_cast<unsigned long long>(impl_->cache.device_slot(slot));
            expect(actual[l*384+e]==expected,"stale primary expert plan after prefill");
        }
        basalt::log::info("basalt deepseek41 plan check","verified after prefill, primary_swaps=%llu",
                          (unsigned long long)impl_->execution.primary_swaps);
    }
    if(impl_->memory_changed && impl_->prompt_experts && impl_->prompt_experts->pool_bytes()!=pool_before)
        impl_->memory_changed();
    if(all.rows)return all;
    return result;
}
int Model::position()const{return impl_->pos;}
int Model::context()const{return impl_->options.context;}
int Model::slots()const{return impl_->options.slots;}
int Model::current_slot()const{return impl_->active_slot;}
void Model::select_slot(int slot){impl_->select_slot(slot);}
ForwardResult Model::forward(const std::vector<int32_t>& tokens,LogitsMode mode){return impl_->forward(tokens,mode);}
void Model::commit(int accepted){impl_->commit(accepted);}
DraftResult Model::draft(int32_t anchor){return impl_->draft(anchor);}
DecodeResult Model::decode_greedy(int32_t anchor,int max_tokens,int32_t stop_token) {
    kernels::SamplerParams parameters;parameters.greedy=true;
    return decode(anchor,parameters,max_tokens,stop_token);
}
std::vector<int32_t> Model::sample(const kernels::SamplerParams& parameters,int rows,
                                   std::span<const int32_t> tail,std::span<const int32_t> window) {
    ProfileScope timer(impl_->profile,"target_sample");
    expect(!impl_->broken && rows>0 && rows<=impl_->logit_rows,"no target logits for sampling");
    expect(std::isfinite(parameters.temperature) && parameters.temperature>=0 && parameters.top_k>=0 &&
           parameters.top_p>0 && parameters.top_p<=1 && parameters.min_p>=0 && parameters.min_p<=1 &&
           parameters.penalty_last_n>=0 && parameters.penalty_last_n<=4096 && !parameters.logit_bias &&
           std::isfinite(parameters.penalty_repeat) && parameters.penalty_repeat>0 &&
           std::isfinite(parameters.penalty_freq) && std::isfinite(parameters.penalty_present),"unsupported sampling parameters");
    auto& d=*impl_->primary;const core::OnDevice on(0);
    auto& w=*impl_->main_work;
    const bool check=!impl_->logits_checked;
    if(check) {
        // Decode needs token IDs; retain nonfinite validation on-device instead of copying the vocabulary.
        d.zero(w.ids+impl_->logit_rows,sizeof(int32_t));
        for(int row=0;row<impl_->logit_rows;++row)
            finite_argmax(w.logits+size_t(row)*Shape::vocab,Shape::vocab,w.ids+row,w.ids+impl_->logit_rows,
                          w.logit_check,finite_argmax_workspace_bytes(Shape::vocab),d.stream());
    }
    if(!parameters.greedy && parameters.temperature>0 && kernels::sampler_list_len(parameters.top_k,Shape::vocab)>kernels::kSamplerNarrowMax)
        expect(kernels::sampler_wide_reserve(Shape::vocab,d.stream()),"sampler workspace allocation failed");
    const int h=parameters.penalty_last_n;
    if(h) {
        expect(window.size()==size_t(rows) || (window.empty() && rows==1),"missing sampling history");
        const size_t count=size_t(rows)*h;
        if(count>impl_->penalty_rows_capacity) {
            // Verification has at most six rows; reserve the bounded maximum on first use.
            impl_->penalty_rows_capacity=std::max(count,size_t(6)*4096);
            impl_->penalty_rows_device=d.alloc<int32_t>(impl_->penalty_rows_capacity,MemoryKind::scratch);
        }
        auto& history=impl_->penalty_rows_host;
        history.assign(count,-1);
        if(!window.empty())kernels::penalty_rows(tail.data(),int64_t(tail.size()),window.data(),rows,h,history.data());
        else if(!tail.empty()) {
            const auto take=std::min(tail.size(),size_t(h));
            std::copy(tail.end()-take,tail.end(),history.end()-take);
        }
        d.upload(impl_->penalty_rows_device,history.data(),count*sizeof(int32_t));
    }
    kernels::sample_tokens(w.logits,rows,Shape::vocab,h?impl_->penalty_rows_device:nullptr,h,parameters,w.ids,d.stream());
    std::vector<int32_t> result(size_t(check?impl_->logit_rows+1:rows),0);
    d.read(result.data(),impl_->main_work->ids,result.size()*sizeof(int32_t));
    if(check) {
        if(result.back()) {impl_->broken=true;throw std::runtime_error("nonfinite target logits");}
        impl_->logits_checked=true;result.resize(size_t(rows));
    }
    return result;
}
int32_t Model::resample(const kernels::SamplerParams& parameters,int row,std::span<const float> bias) {
    expect(!impl_->broken && row>=0 && row<impl_->logit_rows && bias.size()==Shape::vocab,"invalid grammar draw");
    const core::OnDevice on(0);
    auto& d=*impl_->primary;auto& w=*impl_->main_work;
    if(!impl_->grammar_bias_device)impl_->grammar_bias_device=d.alloc<float>(Shape::vocab,MemoryKind::scratch);
    d.upload(impl_->grammar_bias_device,bias.data(),bias.size_bytes());
    auto sp=parameters;sp.counter+=uint64_t(row);sp.logit_bias=impl_->grammar_bias_device;
    const int h=sp.penalty_last_n;
    const auto* hist=h?impl_->penalty_rows_device+size_t(row)*h:nullptr;
    kernels::sample_tokens(w.logits+size_t(row)*Shape::vocab,1,Shape::vocab,hist,h,sp,w.ids,d.stream());
    int32_t token=-1;d.read(&token,w.ids,sizeof token);return token;
}
DecodeResult Model::verify_picks(std::span<const int32_t> input,std::span<const int32_t> picks,
                                 int32_t stop_token,const TokenFilter& filter) {
    DecodeResult result;result.proposed=int(input.size())-1;
    // The anchor was already accepted by the caller. Do not advance grammar on uncommitted draft rows.
    int keep=1;
    for(;;) {
        if(filter && input[keep-1]==stop_token)break;
        const int row=keep-1;
        result.next=filter?filter(row,picks[row]):picks[row];
        if(keep==int(input.size()) || input[row]==stop_token || input[keep]!=result.next)break;
        ++keep;
    }
    result.tokens.assign(input.begin(),input.begin()+keep);result.accepted=keep-1;
    return result;
}
DecodeResult Model::decode(int32_t anchor,const kernels::SamplerParams& parameters,int max_tokens,int32_t stop_token,
                           std::span<const int32_t> proposals,const TokenFilter& filter,
                           std::span<const int32_t> penalty_tail) {
    impl_->profile.select_position(position(),true);
    expect(max_tokens>0 && !impl_->pending && !impl_->broken,"invalid decode request");
    max_tokens=std::min({max_tokens,6,context()-position()});
    expect(max_tokens>0,"context capacity reached");
    std::vector<int32_t> input={anchor};
    std::vector<float> confidence;
    const bool adaptive=impl_->drafter && core::exp_flag("ds-adaptive-window",1)!=0;
    if(!proposals.empty() && anchor!=stop_token) {
        expect(!impl_->drafter,"external proposals require DSpark disabled");
        input.insert(input.end(),proposals.begin(),proposals.begin()+std::min(proposals.size(),size_t(max_tokens-1)));
    } else if(impl_->drafter && max_tokens>1 && position()<=context()-5 && anchor!=stop_token) {
        const auto draft_result=draft(anchor);
        const int count=adaptive?impl_->draft_window.choose(draft_result.confidence,max_tokens-1):max_tokens-1;
        input.insert(input.end(),draft_result.tokens.begin(),draft_result.tokens.begin()+count);
        if(adaptive)confidence=draft_result.confidence;
    }
    impl_->forward(input,LogitsMode::all,false);
    const auto picks=sample(parameters,int(input.size()),penalty_tail,input);
    auto result=verify_picks(input,picks,stop_token,filter);
    const int keep=int(result.tokens.size());
    // Only delay when another step is still possible, so normal EOS and
    // length completion still report every cache update before DONE.
    const bool overlap=impl_->options.slots==1 && impl_->drafter &&
        impl_->drafter->device()!=0 && keep<max_tokens && result.tokens.back()!=stop_token &&
        core::exp_flag("ds-draft-overlap-adapt")!=0;
    impl_->commit(keep,overlap);
    if(adaptive)impl_->draft_window.observe(confidence,result.proposed,result.accepted);
    return result;
}
ExecutionStats Model::stats()const {auto result=impl_->execution;result.helper_entries=uint64_t(impl_->helper.entries());result.helper_swaps=uint64_t(impl_->helper.swaps());result.expert_file_reads=uint64_t(impl_->source.file_reads());return result;}
std::array<uint64_t,3> Model::decode_routes(int slot) const {
    expect(slot>=0 && slot<slots(),"invalid route statistics slot");
    return (slot==impl_->active_slot?static_cast<const SequenceState&>(*impl_):impl_->sequences[size_t(slot)]).decode_expert_routes;
}
std::vector<MemoryUsage> Model::memory()const {
    std::vector<MemoryUsage> result;
    for(const auto* d:{impl_->primary.get(),impl_->drafter.get()})if(d)for(auto [kind,bytes]:d->memory())
        result.push_back({d->device(),kind==MemoryKind::weights?"weights":kind==MemoryKind::state?"kv_state":
                          kind==MemoryKind::draft?"dspark":kind==MemoryKind::long_prefill?"long_prefill":
                          kind==MemoryKind::mmq?"mmq_buffers":kind==MemoryKind::dense_prefill?"dense_prefill":"scratch",bytes});
    if(impl_->prompt_experts)result.push_back({impl_->primary->device(),"mmq_buffers",impl_->prompt_experts->pool_bytes()});
    uint64_t kv_bytes=impl_->kv?impl_->kv->mapped_bytes():0;
    for(const auto& s:impl_->sequences)if(s.kv)kv_bytes+=s.kv->mapped_bytes();
    if(kv_bytes)result.push_back({0,"kv_state",kv_bytes});
    result.push_back({0,"experts",uint64_t(impl_->cache.mapped_bytes())-impl_->borrowed_prefill_bytes});
    if(impl_->long_work && impl_->long_work->borrowed)
        result.push_back({0,"long_prefill",impl_->borrowed_prefill_bytes});
    if(impl_->helper.valid()) {
        result.push_back({impl_->helper.device(),"expert_helper",uint64_t(impl_->helper.cache().mapped_bytes())});
        result.push_back({impl_->helper.device(),"scratch",core::PeerExperts::device_overhead_bytes()});
    }
    return result;
}
std::vector<MemoryUsage> Model::host_memory(uint64_t checkpoint_bytes) const {
    const auto& src=impl_->source;
    return {{-1,"resident_experts",src.resident_bytes()},
            {-1,"pinned_experts",src.pinned_bytes()},
            {-1,"locked_experts",src.locked_bytes()},
            {-1,"expert_homes",src.homes_bytes()},
            {-1,"checkpoints",checkpoint_bytes},
            {-1,"engram_workspace",impl_->host_engram.capacity()+impl_->row_ids.capacity()*sizeof(impl_->row_ids[0])}};
}
void Model::watch_memory(std::function<void()> changed){impl_->memory_changed=std::move(changed);}
void Model::log_host_memory(const char* phase,uint64_t rid,uint64_t checkpoint_bytes)const {
    const auto p=platform::process_memory();
    basalt::log::info("basalt deepseek41","host RAM: %s rid=%llu slot=%d avail=%lluMiB rss=%lluMiB peak=%lluMiB private_commit=%lluMiB resident_experts=%.1fMiB expert_homes=%.1fMiB checkpoints=%.1fMiB engram_workspace=%.1fMiB",
                      phase,(unsigned long long)rid,impl_->active_slot,p.avail_mib,p.rss_mib,p.peak_rss_mib,p.private_commit_mib,
                      impl_->source.resident_bytes()/1048576.,impl_->source.homes_bytes()/1048576.,checkpoint_bytes/1048576.,
                      (impl_->host_engram.capacity()+impl_->row_ids.capacity()*sizeof(impl_->row_ids[0]))/1048576.);
}
} // namespace basalt::deepseek41
