#include "model_state.hpp"
#include <algorithm>
#include <cmath>

namespace basalt::deepseek41 {
void Model::Impl::group_output(DeviceModel& d,Work& w,Layer& l,int count) {
    const bool detailed=profile.front_enabled && !prefill_active && &w==main_work.get() && count==1;
    const uint64_t row=l.oa->bytes/8192;
    if((count==1 || (count<=6 && core::exp_flag("ds-grouped-window",1)!=0)) && (l.oa->type==13 || l.oa->type==14)) {
        {AsyncGpuScope timing(detailed?&profile:nullptr,"front_output_a",d.stream());d.linear_groups(*l.oa,w.attn,w.out_a,8,count);}
        {AsyncGpuScope timing(detailed?&profile:nullptr,"front_output_b",d.stream());d.linear(*l.ob,w.out_a,w.branch,count);}
        return;
    }
    for(int group=0;group<8;++group) {
        const DeviceTensor slice{l.oa->data+group*1024*row,l.oa->type,{4096,1024},1024*row};
        d.linear(slice,w.attn+group*4096,w.out_a+group*1024,count,32768,8192);
    }
    d.linear(*l.ob,w.out_a,w.branch,count);
}
void Model::Impl::attention(DeviceModel& d,Work& w,Layer& l,int p0,int count,bool draft) {
    ProfileScope timer(profile,draft?"draft_attention_enqueue":"target_attention_enqueue");
    const core::OnDevice on(d.device());
    const auto stream=static_cast<cudaStream_t>(d.stream());
    auto* pp=profile.prefill_enabled && prefill_active && long_work?&profile:nullptr;
    const bool detailed=profile.front_enabled && !prefill_active && !draft && count==1;
    const size_t qkv_stamp=detailed?profile.begin_async("front_qkv",stream):size_t(-1);
    GpuProfileScope gpu_time(profile,draft?"draft_attention":prefill_active?"prefill_attention":"decode_attention",d.stream());
    AsyncGpuScope prefill_attention(pp,"long_attention",stream);
    const size_t pp_qkv=pp?pp->begin_async("long_qkv",stream):size_t(-1);
    const int first=p0%128;
    const bool overlay=!draft && !prefill_active && count>=2 && count<=6 && attention_scratch &&
        core::exp_flag("ds-attention-predecode",1)!=0 && core::exp_flag("ds-rollback-window")!=0 &&
        core::exp_flag("ds-verify-ring-overlay")!=0;
    if(!overlay) {
        d.copy(l.extended,l.ring+first*528,(128-first)*528);
        if(first)d.copy(l.extended+(128-first)*528,l.ring,first*528);
    }
    const size_t qa_stamp=detailed?profile.begin_async("front_query_a",stream):size_t(-1);
    d.linear(*l.qa,w.x,w.qr,count);
    RowFinish f;f.norm=l.q_norm;
    finish_rows(w.qr,count,1280,1280,f,nullptr,0,stream);
    profile.end_async(qa_stamp,stream);
    const size_t qb_stamp=detailed?profile.begin_async("front_query_b",stream):size_t(-1);
    d.linear(*l.qb,w.qr,w.q,count);
    f={};f.frequencies=l.ratio?w.freq_compressed:w.freq_window;f.rotary=64;f.first_position=p0;f.vectors_per_token=64;
    const float* rotary=nullptr;
    if(!draft && count>11 && long_work && long_work->rotary_count &&
       p0>=long_work->rotary_first && p0+count<=long_work->rotary_first+long_work->rotary_count)
        rotary=long_work->rotary+(size_t(l.ratio!=0)*2*long_work->capacity+p0-long_work->rotary_first)*64;
    f.rotary_pairs=rotary;
    finish_rows(w.q,count*64,512,512,f,nullptr,0,stream);
    profile.end_async(qb_stamp,stream);
    const size_t kv_stamp=detailed?profile.begin_async("front_kv",stream):size_t(-1);
    d.linear(*l.kv,w.x,w.kv,count);
    f.norm=l.kv_norm;f.vectors_per_token=1;f.quant=CacheFormat::window_fp8;
    finish_rows(w.kv,count,512,512,f,l.extended+128*528,528,stream);
    // Draft rows are noncausal block-local KV; only committed target anchors may change its persistent ring.
    if(!draft && !overlay)update_window_ring(l.ring,l.extended,p0,count,528,d.stream());
    profile.end_async(kv_stamp,stream);
    profile.end_async(qkv_stamp,stream);
    if(pp)pp->end_async(pp_qkv,stream);
    PackedRows window{l.extended,528,CacheFormat::window_fp8},compressed;
    const uint16_t* decoded_compressed=nullptr;
    if(draft) {
        // Every draft query sees the same accepted prefix and the entire provisional block.
        if(core::exp_flag("ds-draft-device-body",1)!=0)expect(count==5,"invalid draft block width");
        else {
            std::vector<int32_t> selection(size_t(count)*512);
            for(int t=0;t<count;++t)for(int j=0;j<count;++j)selection[t*512+j]=j;
            d.upload(w.selected,selection.data(),selection.size()*sizeof(int32_t));d.synchronize();
        }
        compressed={l.extended+128*528,528,CacheFormat::window_fp8};
    } else if(l.ratio) {
        const size_t compress_stamp=detailed?profile.begin_async("front_compressor",stream):size_t(-1);
        auto& source_layer=layers[size_t(l.source)];
        const int keys=(p0+count)/l.ratio;
        if(l.kv_source) {
            AsyncGpuScope prefill_compress(pp,"long_compressor",stream);
            d.copy(l.saved_kv,l.kv_history,1024*sizeof(float));d.copy(l.saved_score,l.score_history,1024*sizeof(float));
            {AsyncGpuScope project(pp,"long_compressor_projection",stream);
            d.linear(*l.compress_kv,w.x,l.raw_kv,count);
            if(l.ratio==2)d.linear(*l.compress_gate,w.x,l.raw_gate,count);}
            const int pooled=[&]{
                AsyncGpuScope pool(pp,"long_compressor_pool",stream);
                return compress_rows(l.raw_kv,l.ratio==2?l.raw_gate:nullptr,l.kv_history,l.score_history,
                                            l.compress_norm,l.pooled,512,l.ratio,p0,count,1e-20f,stream);
            }();
            if(pooled) {
                AsyncGpuScope key_encode(pp,"long_compressor_keys",stream);
                const int group_start=p0/l.ratio;
                d.linear(*l.index_k,l.pooled,l.index_latent,pooled);
                RowFinish key;key.norm=l.index_norm;key.frequencies=w.freq_compressed;key.rotary=64;
                key.first_position=group_start*l.ratio;key.position_step=l.ratio;key.quant=CacheFormat::index_fp4;
                finish_rows(l.index_latent,pooled,128,128,key,l.index_keys+size_t(group_start)*68,68,stream);
                key.norm=nullptr;key.quant=CacheFormat::compressed_fp4;
                finish_rows(l.pooled,pooled,512,512,key,l.compressed+size_t(group_start)*288,288,stream);
            }
        }
        profile.end_async(compress_stamp,stream);
        if(l.index_source && keys) {
            AsyncGpuScope prefill_index(pp,"long_index",stream);
            AsyncGpuScope index_detail(detailed?&profile:nullptr,"front_index",stream);
            GpuProfileScope index_time(profile,prefill_active?"prefill_index":"decode_index",d.stream());
            { AsyncGpuScope projection(pp,"long_index_projection",stream);
            d.linear(*l.index_q,w.qr,w.index_q,count);d.linear(*l.index_weights,w.x,w.index_weights,count);
            RowFinish index;index.frequencies=w.freq_compressed;index.rotary=64;index.first_position=p0;
            index.vectors_per_token=32;index.quant=CacheFormat::index_fp4;
            finish_rows(w.index_q,count*32,128,128,index,nullptr,0,stream); }
            if(prefill_tensor && prompt_index)
                prompt_index->prepare_keys({source_layer.index_keys,68,CacheFormat::index_fp4},keys);
            const int tile=prefill_tensor?w.score_rows:std::min(8,w.score_rows);
            for(int first=0;first<count;first+=tile) {
            const int nr=std::min(tile,count-first);IndexRows visible;
            for(int t=0;t<nr;++t)visible.visible[t]=(p0+first+t+1)/l.ratio;
            { AsyncGpuScope scores(pp,"long_index_scores",stream);
            if(prefill_tensor && prompt_index)prompt_index->run(w.index_q+size_t(first)*32*128,w.index_weights+first*32,
                         {source_layer.index_keys,68,CacheFormat::index_fp4},l.id>20?w.candidates+size_t(first)*w.candidate_stride:nullptr,
                         w.candidate_stride,w.scores,w.score_stride,nr,(p0+first+nr)/l.ratio,visible);
            else index_scores(w.index_q+size_t(first)*32*128,w.index_weights+first*32,{source_layer.index_keys,68,CacheFormat::index_fp4},
                         l.id>20?w.candidates+size_t(first)*w.candidate_stride:nullptr,w.candidate_stride,8,w.scores,w.score_stride,
                         32,128,nr,(p0+first+nr)/l.ratio,visible,stream,1.f/64); }
            if(prefill_tensor) {
                { AsyncGpuScope select(pp,"long_index_topk",stream);
                select_topk_rows(w.scores,w.score_stride,visible,nr,w.selected+first*512,512,stream); }
                if(l.id==20) {
                    AsyncGpuScope select(pp,"long_index_candidates",stream);
                    candidate_topk_rows(w.scores,w.score_stride,visible,nr,
                                               w.candidates+size_t(first)*w.candidate_stride,w.candidate_stride,stream);
                }
                continue;
            }
            for(int row=0;row<nr;++row) {
                const int t=first+row,n=visible.visible[row];if(!n)continue;
                select_topk(w.scores+size_t(row)*w.score_stride,n,std::min(512,n),w.sort_ids,w.sorted_scores,w.sorted_ids,
                            w.selected+t*512,w.sort_workspace,w.sort_bytes,stream);
                if(l.id==20) {
                    const int blocks=(n+7)/8;
                    candidate_block_scores(w.scores+size_t(row)*w.score_stride,n,8,w.block_scores,stream);
                    select_topk(w.block_scores,blocks,std::min(2048,blocks),w.sort_ids,w.sorted_scores,w.sorted_ids,
                                w.candidate_ids,w.sort_workspace,w.sort_bytes,stream);
                    candidate_mask(w.candidate_ids,std::min(2048,blocks),w.candidates+size_t(t)*w.candidate_stride,blocks,stream);
                }
            }
            }
        }
        compressed={source_layer.compressed,288,CacheFormat::compressed_fp4};
        if(prefill_tensor && count>11 && long_work && long_work->compressed_active &&
           keys>0 && keys<=long_work->capacity && core::exp_flag("ds-attention-mma",1)!=0) {
            auto& scratch=*long_work;
            if(scratch.compressed_source!=compressed.data) {
                scratch.compressed_source=compressed.data;
                scratch.compressed_count=0;
            }
            if(keys>scratch.compressed_count) {
                decode_attention_bf16(compressed,scratch.compressed_count,keys-scratch.compressed_count,
                                       scratch.compressed_bf16,stream);
                scratch.compressed_count=keys;
            }
            decoded_compressed=scratch.compressed_bf16;
        }
    }
    const uint16_t* decoded_window=nullptr;
    if(!draft && prefill_tensor && count>11 && long_work && long_work->compressed_active &&
       long_work->window_bf16 && core::exp_flag("ds-prefill-window-bf16")!=0 && core::exp_flag("ds-attention-mma",1)!=0) {
        decode_attention_bf16(window,0,128+count,long_work->window_bf16,stream);
        decoded_window=long_work->window_bf16;
    }
    for(int first=0;first<count;first+=kAttentionWindow) {
        const int nr=std::min(kAttentionWindow,count-first);AttentionRows rows;
        for(int row=0;row<nr;++row) {
            const int t=first+row;
            rows.window_start[row]=std::max(128-p0,draft?0:t+1);
            rows.window_count[row]=(draft?128:128+t+1)-rows.window_start[row];
            rows.selected_count[row]=draft?count:l.ratio?std::min(512,(p0+t+1)/l.ratio):0;
        }
        {
            AsyncGpuScope prefill_sparse(pp,"long_sparse_attention",stream);
            AsyncGpuScope sparse_detail(detailed?&profile:nullptr,"front_sparse_attention",stream);
            GpuProfileScope sparse_time(profile,draft?"draft_sparse_attention":prefill_active?"prefill_sparse_attention":"decode_sparse_attention",d.stream());
            if(draft && draft_attention_scratch && nr==5) {
                const int first_row=rows.window_start[0];
                sparse_attention_draft_shared(w.q,window,first_row,rows.window_count[0]+5,l.sinks,
                                              1.f/std::sqrt(512.f),w.attn,draft_attention_scratch,stream);
            } else if(count==1 && !draft && core::exp_flag("ds-decode-matrix")!=0)
                d.decode_attention(w.q,window,compressed,w.selected,rows,l.sinks,1.f/std::sqrt(512.f),w.attn,attention_scratch);
            else if(count==1 && !draft && rows.window_count[0]+rows.selected_count[0]>=128 && core::exp_flag("ds-decode-split",1)!=0)
                sparse_attention_split(w.q,window,compressed,w.selected,512,rows,l.sinks,
                                         1.f/std::sqrt(512.f),w.attn,attention_scratch,stream);
            else if(prefill_tensor && !draft && count>11 && core::exp_flag("ds-attention-mma",long_work?1:0)!=0)
                sparse_attention_tensor(w.q+size_t(first)*64*512,window,compressed,w.selected+first*512,512,rows,l.sinks,64,512,nr,
                                        1.f/std::sqrt(512.f),w.attn+size_t(first)*64*512,stream,decoded_compressed,decoded_window);
            else if(attention_scratch && !draft && core::exp_flag("ds-attention-predecode",1)!=0)
                sparse_attention_predecoded(w.q+size_t(first)*64*512,window,compressed,w.selected+first*512,512,rows,l.sinks,64,512,nr,
                                             1.f/std::sqrt(512.f),w.attn+size_t(first)*64*512,attention_scratch,stream,
                                             !prefill_active && count>=2 && count<=6 && core::exp_flag("ds-verify-fixed-attention")!=0,
                                             overlay?l.ring:nullptr,p0%128);
            else sparse_attention(w.q+size_t(first)*64*512,window,compressed,w.selected+first*512,512,rows,l.sinks,64,512,nr,
                                  1.f/std::sqrt(512.f),w.attn+size_t(first)*64*512,stream);
        }
    }
    if(overlay)append_window_snapshot(l.ring,l.extended,p0,count,d.stream());
    const size_t output_stamp=detailed?profile.begin_async("front_attention_output",stream):size_t(-1);
    AsyncGpuScope prefill_output(pp,"long_attention_output",stream);
    double check_raw_max=0;
    const bool checking=count==1 && !draft && attention_check_output;
    if(checking) {
        AttentionRows rows;rows.window_start[0]=std::max(128-p0,1);rows.window_count[0]=129-rows.window_start[0];
        rows.selected_count[0]=l.ratio?std::min(512,(p0+1)/l.ratio):0;
        sparse_attention_predecoded(w.q,window,compressed,w.selected,512,rows,l.sinks,64,512,1,
                                     1.f/std::sqrt(512.f),attention_check_output,attention_scratch,stream);
        const auto candidate=d.read(w.attn,64*512),reference=d.read(attention_check_output,64*512);
        for(size_t i=0;i<reference.size();++i)check_raw_max=std::max(check_raw_max,std::abs(double(candidate[i])-reference[i]));
    }
    f={};f.frequencies=l.ratio?w.freq_compressed:w.freq_window;f.rotary=64;f.first_position=p0;
    f.vectors_per_token=64;f.inverse=true;f.quant=CacheFormat::window_fp8;
    if(rotary)f.rotary_pairs=rotary+size_t(long_work->capacity)*64;
    finish_rows(w.attn,count*64,512,512,f,nullptr,0,stream);
    if(checking) {
        finish_rows(attention_check_output,64,512,512,f,nullptr,0,stream);
        const auto candidate=d.read(w.attn,64*512),reference=d.read(attention_check_output,64*512);
        double maximum=0;size_t changed=0;
        for(size_t i=0;i<reference.size();++i) {maximum=std::max(maximum,std::abs(double(candidate[i])-reference[i]));changed+=candidate[i]!=reference[i];}
        basalt::log::info("basalt deepseek41","attention check pos=%d layer=%d raw_max=%.8g packed_max=%.8g changed=%zu",p0,l.id,check_raw_max,maximum,changed);
    }
    group_output(d,w,l,count);
    profile.end_async(output_stamp,stream);
}
} // namespace basalt::deepseek41
