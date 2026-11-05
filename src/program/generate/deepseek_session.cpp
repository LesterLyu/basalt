#include "deepseek_session.hpp"
#include "../cli/tokens.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/mapped_file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/log.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/conversation_store.hpp"
#include "basalt/spec/suffix_drafter.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <deque>
#include <list>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_set>

namespace basalt::program::generate {
namespace {
using Clock=std::chrono::steady_clock;
struct ImageSpan {
    size_t start=0,rows=0;
    std::array<uint64_t,2> fingerprint{};
    bool operator==(const ImageSpan&)const=default;
};
std::span<const ImageSpan> images_before(std::span<const ImageSpan> images,size_t end) {
    return images.first(size_t(std::lower_bound(images.begin(),images.end(),end,
        [](const ImageSpan& image,size_t pos){return image.start<pos;})-images.begin()));
}
bool same_images(std::span<const ImageSpan> a,std::span<const ImageSpan> b,size_t end) {
    a=images_before(a,end);b=images_before(b,end);
    return (a.empty() || a.back().start+a.back().rows<=end) && std::ranges::equal(a,b);
}
struct Request {
    uint64_t rid=0;
    std::vector<int64_t> tokens;
    std::string image_file,grammar_json,grammar_error;
    std::unique_ptr<grammar::Guard> guard;
    platform::MappedFile image_data;
    std::vector<size_t> image_offsets;
    std::vector<ImageSpan> images;
    std::vector<Tokenizer::Span> plain;
    size_t tail_bytes=0;
    bool clip=false;
    int max_new=0;
    kernels::SamplerParams sampler;
};
uint64_t route_id(const std::string& line) {
    std::istringstream input(line);std::string word;
    while(input>>word)if(word.rfind("rid=",0)==0) {
        uint64_t rid=0;const auto n=std::from_chars(word.data()+4,word.data()+word.size(),rid);
        return n.ec==std::errc{} && n.ptr==word.data()+word.size()?rid:0;
    }
    return 0;
}
bool parse(const std::string& line,Request& r,std::string& error,bool text) {
    r.rid=route_id(line);std::istringstream input(line);std::string tag,limit,word;
    if(!(input>>tag>>limit) || tag!="GEN") {error="expected GEN <max_new> [sampling keys] <ids>";return false;}
    const auto n=std::from_chars(limit.data(),limit.data()+limit.size(),r.max_new);
    if(n.ec!=std::errc{} || n.ptr!=limit.data()+limit.size() || r.max_new<1) {error="invalid max_new";return false;}
    r.sampler.greedy=true;r.sampler.temperature=0;r.sampler.top_k=0;r.sampler.top_p=1;
    r.sampler.seed=uint64_t(Clock::now().time_since_epoch().count());bool tagged=false,penalty_window_given=false;
    while(input>>word) {
        const auto eq=word.find('=');
        if(eq==std::string::npos) {
            std::string tail;std::getline(input,tail);word+=tail;
            if(!cli::parse_i64_list(word.c_str(),r.tokens,error))return false;
            break;
        }
        const auto key=word.substr(0,eq),value=word.substr(eq+1);
        if(text && key=="clip") {
            if(value!="1") {error="invalid clip option";return false;}
            r.clip=true;
        } else if(text && key=="tail") {
            const auto n=std::from_chars(value.data(),value.data()+value.size(),r.tail_bytes);
            if(n.ec!=std::errc{} || n.ptr!=value.data()+value.size()) {error="invalid tail length";return false;}
        } else if(text && key=="plain") {
            size_t i=0;
            while(i<value.size()) {
                const size_t colon=value.find(':',i),comma=value.find(',',i);
                const size_t end=comma==std::string::npos?value.size():comma;
                if(colon==std::string::npos || colon>=end || r.plain.size()>=65536) {
                    error="invalid plain spans";return false;
                }
                Tokenizer::Span span{};
                const auto a=std::from_chars(value.data()+i,value.data()+colon,span.begin);
                const auto b=std::from_chars(value.data()+colon+1,value.data()+end,span.end);
                if(a.ec!=std::errc{} || a.ptr!=value.data()+colon || b.ec!=std::errc{} || b.ptr!=value.data()+end ||
                   span.begin>=span.end || (!r.plain.empty() && r.plain.back().end>span.begin)) {
                    error="invalid plain spans";return false;
                }
                r.plain.push_back(span);
                i=end+(comma!=std::string::npos);
            }
        } else if(key=="grammar") {
            if(value.empty() || value.size()>(1u<<20) || !r.grammar_json.empty()) {
                error="grammar: invalid request size or duplicate key";return false;
            }
            r.grammar_json=value;
        } else if(key=="image_file") {
            if(value.empty() || value.size()%2 || value.size()>8192 || !r.image_file.empty()) {
                error="invalid image file key";return false;
            }
            for(size_t i=0;i<value.size();i+=2) {
                unsigned byte=0;
                const auto result=std::from_chars(value.data()+i,value.data()+i+2,byte,16);
                if(result.ec!=std::errc{} || result.ptr!=value.data()+i+2 || byte==0) {
                    error="invalid image file key";return false;
                }
                r.image_file+=char(byte);
            }
        } else if(key=="rid" || key=="seed") {
            uint64_t v=0;const auto result=std::from_chars(value.data(),value.data()+value.size(),v);
            if(result.ec!=std::errc{} || result.ptr!=value.data()+value.size() || (key=="rid" && (!v || tagged))) {
                error="invalid request id or seed";return false;
            }
            if(key=="rid") {r.rid=v;tagged=true;}else r.sampler.seed=v;
        } else if(key=="top_k" || key=="penalty_last_n") {
            auto& target=key=="top_k"?r.sampler.top_k:r.sampler.penalty_last_n;
            if(key=="penalty_last_n")penalty_window_given=true;
            const auto result=std::from_chars(value.data(),value.data()+value.size(),target);
            if(result.ec!=std::errc{} || result.ptr!=value.data()+value.size()) {error="invalid "+key;return false;}
        } else if(key=="temperature" || key=="top_p" || key=="min_p" ||
                  key=="penalty_repeat" || key=="penalty_freq" || key=="penalty_present") {
            float v=0;const auto result=std::from_chars(value.data(),value.data()+value.size(),v);
            if(result.ec!=std::errc{} || result.ptr!=value.data()+value.size() || !std::isfinite(v)) {error="invalid sampling value";return false;}
            if(key=="temperature")r.sampler.temperature=v;
            else if(key=="top_p")r.sampler.top_p=v;
            else if(key=="min_p")r.sampler.min_p=v;
            else if(key=="penalty_repeat")r.sampler.penalty_repeat=v;
            else if(key=="penalty_freq")r.sampler.penalty_freq=v;
            else r.sampler.penalty_present=v;
        } else {error="unsupported request key: "+key;return false;}
    }
    auto& p=r.sampler;p.greedy=p.temperature==0;
    if(p.top_k==-1)p.top_k=0;
    if(p.penalty_repeat==1 && !p.penalty_freq && !p.penalty_present && p.penalty_last_n>=0 && p.penalty_last_n<=4096)
        p.penalty_last_n=0;
    else if(!penalty_window_given)p.penalty_last_n=64;

    if((text?!r.tokens.empty():r.tokens.empty()) || p.temperature<0 || p.top_k<0 || p.penalty_last_n<0 || p.penalty_last_n>4096 || p.penalty_repeat<=0 || !(p.top_p>0 && p.top_p<=1) || !(p.min_p>=0 && p.min_p<=1)) {
        error="invalid tokens or sampling range";return false;
    }
    return true;
}
constexpr int image_token=129264,image_width=5120;
bool read_images(Request& r,std::string& error) {
    const auto count=size_t(std::count(r.tokens.begin(),r.tokens.end(),image_token));
    if(!count || r.image_file.empty()) {error="invalid DeepSeek image span";return false;}
    auto file=platform::MappedFile::open_ro(r.image_file,&error);
    if(!file)return false;
    static const auto seeds=[] {
        std::random_device entropy;
        const auto next=[&]{return (uint64_t(entropy())<<32)|uint64_t(entropy());};
        return std::array<uint64_t,2>{next(),next()};
    }();
    auto cursor=r.tokens.begin();
    size_t offset=0,seen=0;
    while(seen<count) {
        uint32_t header[3]{};
        if(sizeof(header)>file->size()-offset) {error="short DeepSeek image embedding header";return false;}
        std::memcpy(header,file->data()+offset,sizeof(header));
        offset+=sizeof(header);
        if(header[0]!=0x31495644 || header[1]<1 || header[1]>8192 ||
           header[1]>uint32_t(count-seen) || header[2]!=image_width) {
            error="invalid DeepSeek image embedding header";return false;
        }
        const size_t values=size_t(header[1])*image_width,bytes=values*sizeof(uint16_t);
        if(bytes>file->size()-offset) {
            error="short DeepSeek image embedding file";return false;
        }
        const auto* data=reinterpret_cast<const uint16_t*>(file->data()+offset);
        if(std::any_of(data,data+values,[](uint16_t v){return (v&0x7f80)==0x7f80;})) {
            error="nonfinite DeepSeek image embedding";return false;
        }
        const auto start=std::find(cursor,r.tokens.end(),image_token);
        if(r.tokens.end()-start<header[1] ||
           !std::all_of(start,start+header[1],[](int64_t id){return id==image_token;})) {
            error="DeepSeek image rows do not match their token span";return false;
        }
        r.images.push_back({size_t(start-r.tokens.begin()),header[1],
                            {core::conversation_store_hash(data,bytes,seeds[0]),core::conversation_store_hash(data,bytes,seeds[1])}});
        r.image_offsets.push_back(offset);
        offset+=bytes;
        cursor=start+header[1];
        seen+=header[1];
    }
    if(offset!=file->size()) {error="extra DeepSeek image data";return false;}
    r.image_data=std::move(*file);
    return true;
}
std::string tag(const char* name,uint64_t rid) {return std::string(name)+(rid?" "+std::to_string(rid):"");}
void error_line(uint64_t rid,const std::string& text) {std::printf("%s %s\n",tag("ERR",rid).c_str(),text.c_str());std::fflush(stdout);}
struct InputLine {
    std::string header;
    std::optional<std::string> text;
};
struct Inbox {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<InputLine> lines;
    size_t queued_bytes=0;
    bool eof=false,overflow=false;
};
std::shared_ptr<Inbox> input_lines(size_t max_line,size_t max_text) {
    auto box=std::make_shared<Inbox>();
    // Read the descriptor without holding stdio's input lock while the engine shuts down.
    std::thread([box,max_line,max_text] {
        char bytes[4096];std::string buffer,header;bool quit=false;
        size_t remaining=0;
        const auto enqueue=[&](InputLine input) {
            std::lock_guard lock(box->mutex);
            const size_t size=input.header.size()+(input.text?input.text->size():0);
            if(box->lines.size()>=64 || size>max_line+2*max_text-box->queued_bytes) {
                box->overflow=true;quit=true;return;
            }
            box->queued_bytes+=size;
            quit=input.header=="QUIT";
            box->lines.push_back(std::move(input));box->changed.notify_one();
        };
        const auto overflow=[&] {std::lock_guard lock(box->mutex);box->overflow=true;quit=true;};
        while(!quit) {
            const auto n=platform::pipe_read(0,bytes,sizeof bytes);
            if(n<0 && errno==EINTR)continue;
            if(n<=0)break;
            buffer.append(bytes,size_t(n));
            while(!quit) {
                if(remaining) {
                    if(buffer.size()<remaining)break;
                    enqueue({std::move(header),buffer.substr(0,remaining)});
                    buffer.erase(0,remaining);remaining=0;
                    continue;
                }
                const size_t end=buffer.find('\n');
                if(end==std::string::npos) {
                    if(buffer.size()>max_line)overflow();
                    break;
                }
                if(end>max_line) {overflow();break;}
                std::string line=buffer.substr(0,end);buffer.erase(0,end+1);
                if(!line.empty() && line.back()=='\r')line.pop_back();
                if(line.rfind("GENT ",0)==0) {
                    const size_t sep=line.find(' ',5);
                    if(sep==std::string::npos) {overflow();break;}
                    const auto count=std::from_chars(line.data()+5,line.data()+sep,remaining);
                    if(count.ec!=std::errc{} || count.ptr!=line.data()+sep || !remaining || remaining>max_text) {
                        overflow();break;
                    }
                    header="GEN "+line.substr(sep+1);
                } else enqueue({std::move(line),std::nullopt});
            }
        }
        if(remaining && !quit)overflow();
        if(!buffer.empty() && !quit)enqueue({std::move(buffer),std::nullopt});
        std::lock_guard lock(box->mutex);
        box->eof=true;box->changed.notify_one();
    }).detach();
    return box;
}
struct Active {
    Request request;
    std::optional<spec::SuffixDrafter> suffix;
    std::array<int32_t,5> proposals{};
    size_t read=0,image_index=0,reused=0;
    int generated=0,accepted=0,offered=0;
    int32_t next=-1;
    double prefill_ms=0;
    Clock::time_point decoding;
    Clock::time_point route_report;
    std::array<uint64_t,3> routes_before{};
    std::vector<int32_t> captured,penalty_tail;
    deepseek41::ExecutionStats before,segment_stats;
    int segment_tokens=0;
    Clock::time_point segment_time;
};
struct IdlePrefix {
    std::vector<int64_t> tokens;
    std::vector<ImageSpan> images;
    uint64_t age=0;
};
struct ParkedPrefix {
    IdlePrefix prefix;
    deepseek41::PrefixSnapshot snapshot;
    uint64_t bytes=0;
    bool before_decode=false;
};
}
int deepseek_session(deepseek41::Model& model,const Options& options,int window,int tuned_prefill,
                     const std::optional<grammar::PackVocab>& vocab,Tokenizer* tokenizer) {
    const int suffix_width=window==1?std::clamp(int(core::exp_flag("ds-suffix-width")),0,5):0;
    const int suffix_bigram=std::clamp(int(core::exp_flag("ds-suffix-bigram")),0,5);
    const bool suffix_trim=core::exp_flag("ds-suffix-trim-minority")!=0;
    const bool suffix_skip=core::exp_flag("ds-suffix-skip-pairs")!=0;
    std::string buses;
    for(int device:{0,options.peer_device})if(device>=0) {
        char bus[32]{};
        if(cudaDeviceGetPCIBusId(bus,sizeof bus,device)==cudaSuccess)buses+=(buses.empty()?"":",")+std::string(bus);
    }
    platform::stop_memory_watchdog();
    std::printf("INFO architecture=deepseek41 quant=UD-IQ1_S logit_bias=129280 context=%d slots=%d spec=%d kv=fp8_fp4 text_prompt=%d memory=1 deepseek_grammar=%d deepseek_penalties=1 elastic_kv=%d gpus=%s\n",
                model.context(),model.slots(),window,int(tokenizer!=nullptr),int(vocab.has_value()),int(options.vram_elastic),buses.c_str());
    std::printf("READY %d stop multi=%d\n",model.context(),model.slots());std::fflush(stdout);
    auto box=input_lines(size_t(model.context())*8+(1u<<20)+8192,size_t(model.context())*32+(1u<<20));
    std::vector<std::optional<Active>> active(size_t(model.slots()));
    std::vector<IdlePrefix> idle(active.size());
    std::vector<bool> trim_pending(active.size(),false);
    const bool cooperative=tuned_prefill>0 && core::exp_flag("ds-prefill-borrow")!=0 &&
                           core::exp_flag("ds-prefill-cooperative")!=0;
    const bool reuse_prefix=options.prompt_cache>0;
    const uint64_t cache_budget=reuse_prefix?uint64_t(options.conversation_cache_mib)<<20:0;
    const uint64_t cache_floor=uint64_t(options.conversation_cache_min_free_mib)<<20;
    const auto cache_slots=size_t(options.conversation_cache_slots);
    std::list<ParkedPrefix> parked;
    uint64_t parked_bytes=0;
    uint64_t prefix_age=0;
    bool memory_events=false;
    const auto publish_ram=[&] {
        if(!memory_events)return;
        std::printf("RAM");
        for(const auto& m:model.host_memory(parked_bytes))
            std::printf(" %s:%llu",m.component.c_str(),(unsigned long long)m.bytes);
        std::printf("\n");std::fflush(stdout);
    };
    if(cache_budget && cache_slots)
        basalt::log::info("basalt deepseek41","conversation RAM cache %lld MiB, %zu snapshots",
                          (long long)options.conversation_cache_mib,cache_slots);
    const auto discard=[&](auto it) {
        parked_bytes-=it->bytes;
        parked.erase(it);
        publish_ram();
    };
    const auto save_prefix=[&](std::span<const int64_t> tokens,std::span<const ImageSpan> images,uint64_t age,
                               auto preserve,IdlePrefix* idle_prefix) -> bool {
        if(!cache_budget || !cache_slots || tokens.empty() ||
           tokens.size()<size_t(options.conversation_cache_min_tokens) ||
           tokens.size()!=size_t(model.position()))return false;
        const bool before_decode=idle_prefix==nullptr;
        for(auto& cached:parked)if(std::ranges::equal(cached.prefix.tokens,tokens) && std::ranges::equal(cached.prefix.images,images)) {
            cached.prefix.age=std::max(cached.prefix.age,age);
            cached.before_decode|=before_decode;
            return true;
        }
        const size_t payload=model.prefix_snapshot_bytes();
        if(!payload)return false;
        const auto token_capacity=idle_prefix?idle_prefix->tokens.capacity():tokens.size();
        const auto image_capacity=idle_prefix?idle_prefix->images.capacity():images.size();
        const uint64_t bytes=payload+token_capacity*sizeof(int64_t)+image_capacity*sizeof(ImageSpan)+sizeof(ParkedPrefix);
        if(bytes>cache_budget)return false;
        while(parked_bytes>cache_budget-bytes || parked.size()>=cache_slots ||
              !core::conversation_memory_admit(core::conversation_available_memory(),bytes,cache_floor)) {
            auto victim=parked.end();
            for(auto it=parked.begin();it!=parked.end();++it)
                if(it!=preserve && (victim==parked.end() || it->prefix.age<victim->prefix.age))victim=it;
            if(victim==parked.end())return false;
            discard(victim);
        }
        const auto started=Clock::now();
        try {
            auto snapshot=model.save_prefix();
            std::vector<int64_t> copied;
            std::vector<ImageSpan> copied_images;
            if(before_decode) {
                copied.assign(tokens.begin(),tokens.end());
                copied_images.assign(images.begin(),images.end());
            }
            auto& entry=parked.emplace_back();
            entry.prefix.tokens=before_decode?std::move(copied):std::move(idle_prefix->tokens);
            entry.prefix.images=before_decode?std::move(copied_images):std::move(idle_prefix->images);
            entry.prefix.age=age;
            entry.snapshot=std::move(snapshot);
            entry.bytes=bytes;
            entry.before_decode=before_decode;
        } catch(const std::bad_alloc&) {
            basalt::log::warn("basalt deepseek41","conversation RAM cache: allocation failed");
            return false;
        }
        parked_bytes+=bytes;
        publish_ram();
        basalt::log::info("basalt deepseek41","conversation %s tokens=%zu bytes=%llu cache_bytes=%llu entries=%zu ms=%.2f",
            before_decode?"checkpoint":"parked",tokens.size(),(unsigned long long)bytes,(unsigned long long)parked_bytes,parked.size(),
            std::chrono::duration<double,std::milli>(Clock::now()-started).count());
        return true;
    };
    const auto park=[&](IdlePrefix& prefix,auto preserve) {
        return save_prefix(prefix.tokens,prefix.images,prefix.age,preserve,&prefix);
    };
    const bool log_tokens=core::exp_flag("ds-token-log")!=0;
    model.log_host_memory("ready",0,parked_bytes);
    auto last_ram_log=Clock::now();
    std::deque<Request> waiting;std::unordered_set<uint64_t> ids;bool quit=false;size_t prefill_cursor=0;
    const auto routes_for=[&](int slot,const Active& a) {
        auto counts=model.decode_routes(slot);
        for(size_t i=0;i<counts.size();++i)counts[i]-=a.routes_before[i];
        return counts;
    };
    const auto finish=[&](int slot,const char* reason) {
        auto& a=*active[size_t(slot)];
        const double tg=a.next<0 && !a.generated?0:std::chrono::duration<double,std::milli>(Clock::now()-a.decoding).count();
        const auto [ram,primary,helper]=routes_for(slot,a);
        a.request.image_data.reset();
        if(std::string_view(reason)!="error")std::printf("%s %d %zu %.3f %.3f %s %d %d %zu experts=%llu,%llu,%llu\n",
                    tag("DONE",a.request.rid).c_str(),a.generated,a.request.tokens.size(),a.prefill_ms,tg,reason,a.accepted,a.offered,a.reused,
                    (unsigned long long)primary,(unsigned long long)helper,(unsigned long long)ram);
        std::fflush(stdout);
        basalt::log::info("basalt deepseek41","done prompt=%zu pp_ms=%.3f generated=%d decode_ms=%.3f rid=%llu slot=%d batch_rounds=%llu end=%s",
            a.request.tokens.size(),a.prefill_ms,a.generated,tg,(unsigned long long)a.request.rid,slot,
            (unsigned long long)model.stats().batch_rounds,reason);
        model.log_host_memory("done",a.request.rid,parked_bytes);
        const auto after=model.stats();
        basalt::log::info("basalt deepseek41","expert entries primary=%llu helper=%llu cpu=%llu helper_swaps=%llu primary_swaps=%llu file_reads=%llu peer_promotions=%llu shared=%d",
            (unsigned long long)(after.primary_entries-a.before.primary_entries),(unsigned long long)(after.helper_entries-a.before.helper_entries),
            (unsigned long long)(after.cpu_entries-a.before.cpu_entries),(unsigned long long)(after.helper_swaps-a.before.helper_swaps),
            (unsigned long long)(after.primary_swaps-a.before.primary_swaps),(unsigned long long)(after.expert_file_reads-a.before.expert_file_reads),
            (unsigned long long)(after.peer_promotions-a.before.peer_promotions),model.slots()>1?1:0);
        if(log_tokens) {
            std::string tokens;
            for(int32_t id:a.captured) {if(!tokens.empty())tokens+=',';tokens+=std::to_string(id);}
            basalt::log::info("basalt deepseek41 tokens","ids=%s rid=%llu",tokens.c_str(),(unsigned long long)a.request.rid);
        }
        if(reuse_prefix && a.generated>0 && (std::string_view(reason)=="stop" || std::string_view(reason)=="length")) {
            auto& prefix=idle[size_t(slot)];
            prefix.tokens=std::move(a.request.tokens);
            prefix.tokens.insert(prefix.tokens.end(),a.captured.begin(),a.captured.end());
            prefix.images=std::move(a.request.images);
            prefix.age=++prefix_age;
        }
        ids.erase(a.request.rid);active[size_t(slot)].reset();
        if(options.vram_elastic)trim_pending[size_t(slot)]=true;
    };
    const auto stop_requests=[&](const std::string& line,int reader_slot) {
        uint64_t rid=0;
        if(line!="STOP") {
            const auto n=std::from_chars(line.data()+5,line.data()+line.size(),rid);
            if(n.ec!=std::errc{} || n.ptr!=line.data()+line.size() || !rid) {error_line(0,"invalid STOP id");return false;}
        }
        bool stop_reader=false;
        for(int slot=0;slot<model.slots();++slot)if(active[size_t(slot)] && (!rid || active[size_t(slot)]->request.rid==rid)) {
            if(slot==reader_slot)stop_reader=true;
            else finish(slot,"cancelled");
        }
        for(auto it=waiting.begin();it!=waiting.end();) {
            if(!rid || it->rid==rid) {
                it->image_data.reset();
                std::printf("%s 0 %zu 0 0 cancelled 0 0 0\n",tag("DONE",it->rid).c_str(),it->tokens.size());
                ids.erase(it->rid);it=waiting.erase(it);
            } else ++it;
        }
        std::fflush(stdout);return stop_reader;
    };
    const auto poll_prefill=[&](int slot) {
        std::vector<std::string> controls;
        bool eof=false,stop_all=false;
        {
            std::lock_guard lock(box->mutex);
            for(auto it=box->lines.begin();it!=box->lines.end();) {
                const auto& line=it->header;
                if(!it->text && line=="STOP") {
                    // Leave the broadcast in FIFO order so it also sees earlier queued GEN commands.
                    stop_all=true;
                    break;
                }
                bool current=false;
                if(!it->text && line.rfind("STOP ",0)==0) {
                    uint64_t rid=0;
                    const auto n=std::from_chars(line.data()+5,line.data()+line.size(),rid);
                    current=n.ec==std::errc{} && n.ptr==line.data()+line.size() && ids.contains(rid);
                }
                if(!it->text && (current || line=="QUIT")) {
                    box->queued_bytes-=line.size();
                    controls.push_back(std::move(it->header));
                    it=box->lines.erase(it);
                } else ++it;
            }
            eof=box->eof;
        }
        bool cancelled=stop_all;
        for(const auto& line:controls) {
            if(line=="QUIT")quit=true;
            else if(stop_requests(line,slot))cancelled=true;
        }
        if(eof)quit=true;
        if(quit || cancelled)throw deepseek41::PrefillCancelled{};
    };
    std::vector<float> grammar_bias;
    const auto guarded=[&](Request& r,int row,int32_t token) -> int32_t {
        if(!r.guard || r.guard->scan(&token,1)<0)return token;
        r.guard->mask_bias({},grammar_bias);
        if(std::none_of(grammar_bias.begin(),grammar_bias.end(),[](float x){return std::isfinite(x);})) {
            r.grammar_error="grammar: no vocabulary token matches the current constraint";
            return -1;
        }
        token=model.resample(r.sampler,row,grammar_bias);
        if(token<0 || size_t(token)>=grammar_bias.size() || !std::isfinite(grammar_bias[token])) {
            r.grammar_error="grammar: sampler returned a masked token ("+std::to_string(token)+")";
            return -1;
        }
        if(!r.guard->take(token)) {
            r.grammar_error="grammar: resampled token did not match the constraint ("+std::to_string(token)+")";
            return -1;
        }
        return token;
    };
    uint64_t released_kv=0;
    const auto fit_for=[&](int slot,int end,std::string& error,bool beside=false) {
        if(!options.vram_elastic)return true;
        model.select_slot(slot);
        if(model.prepare_kv(end,error,beside))return true;
        if(beside)return false;
        for(size_t i=0;i<idle.size();++i)if(int(i)!=slot && !active[i] && !idle[i].tokens.empty()) {
            model.select_slot(int(i));
            if(!park(idle[i],parked.end()))continue;
            idle[i]={};
            trim_pending[i]=false;
            const auto released=model.release_idle_kv();
            released_kv+=released;
            model.select_slot(slot);
            if(!released)continue;
            error.clear();
            if(model.prepare_kv(end,error)) {
                basalt::log::info("basalt deepseek41","elastic KV: slot %d served before replenishing experts (%.1f MiB released)",
                                  slot,released/1048576.);
                return true;
            }
        }
        model.select_slot(slot);
        return false;
    };
    const auto decode_ready=[&](bool beside=false) {
        std::vector<deepseek41::DecodeRequest> requests;
        for(int slot=0;slot<model.slots();++slot)if(active[size_t(slot)] && active[size_t(slot)]->next>=0) {
            auto& a=*active[size_t(slot)];
            if(options.vram_elastic) {
                model.select_slot(slot);
                const int need=std::min(model.context(),model.position()+std::min(6,a.request.max_new-a.generated));
                std::string error;
                if(!fit_for(slot,need,error,beside)) {
                    if(!beside) {
                        basalt::log::warn("basalt deepseek41","elastic KV: rid=%llu slot=%d position=%d: %s",
                                          (unsigned long long)a.request.rid,slot,need,error.c_str());
                        error_line(a.request.rid,"elastic KV: "+error);
                        finish(slot,"error");
                    }
                    continue;
                }
            }
            int n=0;
            if(a.suffix && a.next!=1) {
                a.suffix->append(a.next);
                n=a.suffix->propose(std::min(suffix_width,a.request.max_new-a.generated-1),a.proposals.data());
            }
            deepseek41::TokenFilter filter;
            if(a.request.guard)filter=[&,request=&a.request](int row,int32_t token){return guarded(*request,row,token);};
            requests.push_back({slot,a.next,a.request.sampler,std::min(window+n,a.request.max_new-a.generated),1,
                                {a.proposals.data(),size_t(n)},std::move(filter),a.penalty_tail});
        }
        const auto output=model.decode_batch(requests);
        if(!requests.empty() && Clock::now()-last_ram_log>=std::chrono::seconds(5)) {
            const auto slot=model.current_slot();
            model.log_host_memory("decode",active[size_t(slot)]?active[size_t(slot)]->request.rid:0,parked_bytes);
            last_ram_log=Clock::now();
        }
        for(size_t i=0;i<requests.size();++i) {
            const int slot=requests[i].slot;auto& a=*active[size_t(slot)];const auto& row=output[i];
            if(a.suffix && row.tokens.size()>1)a.suffix->append(row.tokens.data()+1,row.tokens.size()-1);
            if(log_tokens || reuse_prefix)
                a.captured.insert(a.captured.end(),row.tokens.begin(),row.tokens.end());
            if(const int h=a.request.sampler.penalty_last_n) {
                a.penalty_tail.insert(a.penalty_tail.end(),row.tokens.begin(),row.tokens.end());
                if(a.penalty_tail.size()>size_t(h))
                    a.penalty_tail.erase(a.penalty_tail.begin(),a.penalty_tail.end()-h);
            }
            a.accepted+=row.accepted;a.offered+=row.proposed;
            const auto now=Clock::now();
            if(now-a.route_report>=std::chrono::seconds(1)) {
                const auto [ram,primary,helper]=routes_for(slot,a);
                std::printf("%s %d %d %llu %llu %llu\n",tag("STAT",a.request.rid).c_str(),a.accepted,a.offered,
                    (unsigned long long)primary,(unsigned long long)helper,(unsigned long long)ram);
                a.route_report=now;
            }
            for(int32_t token:row.tokens)std::printf("%s %d\n",tag("T",a.request.rid).c_str(),token);
            std::fflush(stdout);a.generated+=int(row.tokens.size());a.request.sampler.counter+=row.tokens.size();a.next=row.next;
            if(core::exp_flag("ds-segment-metrics")!=0 && (a.generated%64<int(row.tokens.size()) || row.tokens.back()==1 || a.generated==a.request.max_new)) {
                const auto now=Clock::now();const auto s=model.stats();
                basalt::log::info("basalt deepseek41 segment","from=%d to=%d ms=%.3f primary=%llu helper=%llu cpu=%llu primary_swaps=%llu helper_swaps=%llu file_reads=%llu slot=%d shared=%d",
                    a.segment_tokens,a.generated,std::chrono::duration<double,std::milli>(now-a.segment_time).count(),
                    (unsigned long long)(s.primary_entries-a.segment_stats.primary_entries),(unsigned long long)(s.helper_entries-a.segment_stats.helper_entries),
                    (unsigned long long)(s.cpu_entries-a.segment_stats.cpu_entries),(unsigned long long)(s.primary_swaps-a.segment_stats.primary_swaps),
                    (unsigned long long)(s.helper_swaps-a.segment_stats.helper_swaps),(unsigned long long)(s.expert_file_reads-a.segment_stats.expert_file_reads),slot,model.slots()>1?1:0);
                a.segment_time=now;a.segment_tokens=a.generated;a.segment_stats=s;
            }
            if(row.tokens.back()==1 || a.generated==a.request.max_new)finish(slot,row.tokens.back()==1?"stop":"length");
            else if(row.next<0) {
                error_line(a.request.rid,a.request.grammar_error);
                finish(slot,"error");
            }
        }
    };
    try {
        for(;;) {
            for(size_t i=0;i<trim_pending.size();++i)if(trim_pending[i]) {
                model.select_slot(int(i));
                if(idle[i].tokens.empty())released_kv+=model.release_idle_kv();
                else released_kv+=model.trim_kv(false);
                trim_pending[i]=false;
            }
            std::deque<InputLine> lines;bool eof=false;
            {
                std::unique_lock lock(box->mutex);
                const bool busy=std::any_of(active.begin(),active.end(),[](const auto& a){return a.has_value();});
                if(!busy && waiting.empty() && !released_kv)box->changed.wait(lock,[&]{return !box->lines.empty() || box->eof;});
                lines.swap(box->lines);box->queued_bytes=0;eof=box->eof;
                if(box->overflow)throw std::runtime_error("input queue or line exceeds capacity");
            }
            for(const auto& input:lines) {
                const auto& line=input.header;
                if(line=="QUIT") {quit=true;break;}
                if(line=="MEMORY") {
                    memory_events=true;
                    const auto publish=[&model,publish_ram] {
                        std::printf("MEM");
                        for(const auto& m:model.memory())
                            std::printf(" %d:%s:%llu",m.device,m.component.c_str(),(unsigned long long)m.bytes);
                        std::printf("\n");std::fflush(stdout);
                        publish_ram();
                    };
                    model.watch_memory(publish);
                    publish();
                    continue;
                }
                if(line=="STOP" || line.rfind("STOP ",0)==0) {
                    stop_requests(line,-1);
                    continue;
                }
                Request r;std::string error;
                if(!parse(line,r,error,input.text.has_value())) {error_line(r.rid,error);continue;}
                if(input.text) {
                    const std::string_view body=*input.text;
                    if(!tokenizer || !r.image_file.empty() || r.tail_bytes>body.size() ||
                       (!r.plain.empty() && r.plain.back().end>body.size()-r.tail_bytes)) {
                        error_line(r.rid,"bad request: invalid native text prompt");continue;
                    }
                    std::vector<int32_t> tids,tail;
                    if(!tokenizer->encode(body.substr(0,body.size()-r.tail_bytes),true,r.plain,tids,error) ||
                       (r.tail_bytes && !tokenizer->encode(body.substr(body.size()-r.tail_bytes),true,{},tail,error))) {
                        error_line(r.rid,"bad request: tokenizer: "+error);continue;
                    }
                    r.tokens.assign(tids.begin(),tids.end());r.tokens.insert(r.tokens.end(),tail.begin(),tail.end());
                    const auto n=r.tokens.size();
                    const int64_t room=int64_t(model.context())-int64_t(n)-8;
                    const std::string detail="prompt ("+std::to_string(n)+" tokens)";
                    if(room<1) {
                        error_line(r.rid,detail+" leaves no room to answer in the context ("+std::to_string(model.context())+")");continue;
                    }
                    if(r.max_new>room) {
                        if(r.clip)r.max_new=int(room);
                        else {
                            error_line(r.rid,detail+" + max_new ("+std::to_string(r.max_new)+") exceeds the context ("+
                                       std::to_string(model.context())+")");continue;
                        }
                    }
                }
                if(r.max_new>model.context() || r.tokens.size()>size_t(model.context()-r.max_new) ||
                   std::any_of(r.tokens.begin(),r.tokens.end(),[](int64_t t){return t<0 || t>=129280;})) {
                    error_line(r.rid,"tokens exceed vocabulary or context capacity");continue;
                }
                if(!r.image_file.empty() && (!model.accepts_images() || !read_images(r,error))) {
                    error_line(r.rid,error.empty()?"DeepSeek vision is disabled":error);continue;
                }
                if(model.accepts_images() && r.image_file.empty() &&
                   std::find(r.tokens.begin(),r.tokens.end(),image_token)!=r.tokens.end()) {
                    error_line(r.rid,"image placeholder without embeddings");continue;
                }
                if(!r.grammar_json.empty()) {
                    if(!vocab) {error_line(r.rid,"grammar: DeepSeek grammar is disabled");continue;}
                    r.guard=std::make_unique<grammar::Guard>(*vocab);
                    try {
                        if(!r.guard->begin(r.grammar_json,r.tokens.data(),r.tokens.size(),error,true) && error.empty())
                            error="could not initialize request grammar";
                    } catch(const std::exception& e) {error=e.what();}
                    if(!error.empty()) {error_line(r.rid,"grammar: "+error);continue;}
                    if(!r.guard->on())r.guard.reset();
                    r.grammar_json.clear();
                }
                if(ids.contains(r.rid)) {error_line(0,"duplicate request id");continue;}
                if(waiting.size()>=size_t(model.slots())*2) {error_line(r.rid,"request queue full");continue;}
                if(input.text) {
                    std::printf("%s %zu %d\n",tag("PT",r.rid).c_str(),r.tokens.size(),r.max_new);
                    std::fflush(stdout);
                }
                ids.insert(r.rid);waiting.push_back(std::move(r));
            }
            if(quit || eof) {
                for(int slot=0;slot<model.slots();++slot)if(active[size_t(slot)])finish(slot,"cancelled");
                break;
            }
            while(!waiting.empty()) {
                const auto& request=waiting.front();
                int slot=-1,retry=-1;size_t reused=0;
                for(size_t i=0;i<active.size();++i)if(!active[i]) {
                    const auto& prefix=idle[i].tokens;
                    if(!reused && (slot<0 || idle[i].age<idle[size_t(slot)].age))slot=int(i);
                    if(!reuse_prefix || prefix.empty())continue;
                    const size_t common=std::min(prefix.size(),request.tokens.size());
                    if(!same_images(idle[i].images,request.images,common) ||
                       !std::equal(prefix.begin(),prefix.begin()+common,request.tokens.begin()))continue;
                    // A retry can reset its own slot without evicting another conversation.
                    if(prefix.size()>=request.tokens.size())retry=int(i);
                    else if(prefix.size()>reused) {slot=int(i);reused=prefix.size();}
                }
                if(!reused && retry>=0)slot=retry;
                if(slot<0)break;
                const size_t live_reused=reused;
                auto cached=parked.end();
                for(auto it=parked.begin();it!=parked.end();++it) {
                    const auto& prefix=it->prefix.tokens;
                    if(prefix.size()>reused && prefix.size()<request.tokens.size() &&
                       same_images(it->prefix.images,request.images,prefix.size()) &&
                       std::equal(prefix.begin(),prefix.end(),request.tokens.begin())) {
                        cached=it;
                        reused=prefix.size();
                    }
                }
                model.select_slot(slot);
                if(cached!=parked.end()) {
                    park(idle[size_t(slot)],cached);
                    const auto started=Clock::now();
                    std::string error;
                    if(model.prepare_kv(int(reused),error)) {
                        model.restore_prefix(cached->snapshot);
                        const bool before_decode=cached->before_decode;
                        if(before_decode)cached->prefix.age=++prefix_age;
                        else discard(cached);
                        basalt::log::info("basalt deepseek41","conversation restored tokens=%zu slot=%d cache_bytes=%llu ms=%.2f source=%s",
                            reused,slot,(unsigned long long)parked_bytes,
                            std::chrono::duration<double,std::milli>(Clock::now()-started).count(),before_decode?"checkpoint":"parked");
                    } else if(live_reused && size_t(model.position())==live_reused) {
                        reused=live_reused;
                        model.resume_turn();
                    } else {
                        released_kv+=model.release_idle_kv();
                        reused=0;
                    }
                } else if(reused && size_t(model.position())==reused)model.resume_turn();
                else {
                    park(idle[size_t(slot)],parked.end());
                    released_kv+=model.release_idle_kv();reused=0;
                }
                idle[size_t(slot)]={};
                auto& a=active[size_t(slot)].emplace();
                a.request=std::move(waiting.front());waiting.pop_front();
                a.read=a.reused=reused;
                a.image_index=images_before(a.request.images,reused).size();
                a.before=model.stats();
                a.routes_before=model.decode_routes(slot);
                basalt::log::info("basalt deepseek41","admit rid=%llu slot=%d reused=%zu",
                                  (unsigned long long)a.request.rid,slot,reused);
                model.log_host_memory("admit",a.request.rid,parked_bytes);
            }
            // Bound prompt work so an arriving long prompt cannot occupy every decode turn.
            const bool had_decoder=std::any_of(active.begin(),active.end(),[](const auto& a){return a && a->next>=0;});
            const bool alone=std::count_if(active.begin(),active.end(),[](const auto& a){return a.has_value();})==1;
            const size_t prefill_budget=(had_decoder || (tuned_prefill>0 && alone))?1:active.size();
            for(size_t turn=0;turn<prefill_budget;++turn) {
            bool read_prompt=false;
            for(size_t n=0;n<active.size();++n) {
                const size_t slot=(prefill_cursor+n)%active.size();
                if(!active[slot] || active[slot]->next>=0)continue;
                auto& a=*active[slot];model.select_slot(int(slot));const auto started=Clock::now();
                const bool visual=!a.request.images.empty() && a.request.tokens[a.read]==image_token;
                const bool beside=cooperative && had_decoder && !visual &&
                                  a.request.tokens.size()-a.read>size_t(deepseek41::kConcurrentPrefill);
                const size_t slice=tuned_prefill>0 && (alone || beside) && !visual?size_t(tuned_prefill):
                    options.prefill_chunk>0?size_t(std::min<int64_t>(deepseek41::kConcurrentPrefill,options.prefill_chunk)):
                    size_t(deepseek41::kConcurrentPrefill);
                size_t end=std::min(a.read+slice,a.request.tokens.size());
                if(!a.request.images.empty()) {
                    if(visual) {
                        const auto& image=a.request.images[a.image_index];
                        end=std::min({end,a.read+128,image.start+image.rows});
                    }
                    size_t next=a.read+1;
                    while(next<end && (a.request.tokens[next]==image_token)==visual)++next;
                    end=next;
                }
                std::string kv_error;
                bool fits=fit_for(int(slot),int(end),kv_error);
                while(!fits && !visual && end-a.read>size_t(deepseek41::kConcurrentPrefill)) {
                    end=a.read+std::max(size_t(deepseek41::kConcurrentPrefill),(end-a.read)*3/4);
                    kv_error.clear();
                    fits=fit_for(int(slot),int(end),kv_error);
                }
                if(!fits) {
                    basalt::log::warn("basalt deepseek41","elastic KV: rid=%llu slot=%zu position=%zu: %s",
                                      (unsigned long long)a.request.rid,slot,end,kv_error.c_str());
                    error_line(a.request.rid,"elastic KV: "+kv_error);
                    finish(int(slot),"error");
                    prefill_cursor=(slot+1)%active.size();read_prompt=true;break;
                }
                if(visual) {
                    const auto& image=a.request.images[a.image_index];
                    const auto* data=reinterpret_cast<const uint16_t*>(
                        a.request.image_data.data()+a.request.image_offsets[a.image_index]);
                    model.prefill_image(std::span(data+(a.read-image.start)*image_width,(end-a.read)*image_width));
                    if(end==image.start+image.rows && ++a.image_index==a.request.images.size())
                        a.request.image_data.reset();
                } else {
                    const std::function<void()> step=beside?std::function<void()>([&]{decode_ready(true);}):std::function<void()>{};
                    try {
                        model.prefill(std::vector<int32_t>(a.request.tokens.begin()+a.read,a.request.tokens.begin()+end),
                                      0,deepseek41::LogitsMode::last,step,[&]{poll_prefill(int(slot));});
                    } catch(const deepseek41::PrefillCancelled&) {
                        a.prefill_ms+=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
                        finish(int(slot),"cancelled");
                        // The partially written prompt is never reused, even with static KV.
                        model.select_slot(int(slot));
                        released_kv+=model.release_idle_kv();
                        trim_pending[slot]=false;
                        prefill_cursor=(slot+1)%active.size();read_prompt=true;break;
                    }
                }
                a.read=end;
                if(Clock::now()-last_ram_log>=std::chrono::seconds(5)) {
                    model.log_host_memory("prefill",a.request.rid,parked_bytes);
                    last_ram_log=Clock::now();
                }
                const bool prompt_end=end==a.request.tokens.size();
                const bool before_image=!visual && !a.request.images.empty() && !prompt_end && a.request.tokens[end]==image_token;
                if(cache_budget && (prompt_end || before_image))
                    save_prefix(std::span(a.request.tokens).first(end),images_before(a.request.images,end),++prefix_age,parked.end(),nullptr);
                if(prompt_end) {
                    a.request.image_data.reset();
                    if(suffix_width) {
                        a.suffix.emplace(3,32,a.request.tokens.size()+size_t(a.request.max_new),suffix_bigram,suffix_trim,suffix_skip);
                        for(int64_t token:a.request.tokens)a.suffix->append(int32_t(token));
                    }
                    if(const int h=a.request.sampler.penalty_last_n) {
                        const auto& tokens=a.request.tokens;
                        const auto take=std::min(tokens.size(),size_t(h));
                        a.penalty_tail.assign(tokens.end()-take,tokens.end());
                    }
                    a.next=guarded(a.request,0,model.sample(a.request.sampler,1,a.penalty_tail)[0]);
                    if(a.next<0) {
                        error_line(a.request.rid,a.request.grammar_error);
                        finish(slot,"error");
                        prefill_cursor=(slot+1)%active.size();read_prompt=true;break;
                    }
                    a.request.sampler.counter=1;a.decoding=Clock::now();
                    a.segment_time=a.decoding;a.segment_stats=model.stats();
                    if(model.slots()==1 && (core::exp_flag("ds-metrics")!=0 || core::exp_flag("ds-front-metrics")!=0 || core::exp_flag("ds-cpu-timing")!=0))model.reset_metrics();
                }
                a.prefill_ms+=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
                std::printf("%s %zu %zu %.3f 0 0\n",tag("PP",a.request.rid).c_str(),a.read,a.request.tokens.size(),a.prefill_ms);
                std::fflush(stdout);prefill_cursor=(slot+1)%active.size();read_prompt=true;break;
            }
            if(quit || !read_prompt)break;
            }
            if(quit) {
                for(int slot=0;slot<model.slots();++slot)if(active[size_t(slot)])finish(slot,"cancelled");
                break;
            }
            decode_ready();
            // Let every runnable slot map its KV before the expert cache takes the rest.
            if(released_kv) {
                model.replenish_expert_cache();
                released_kv=0;
            }
        }
    } catch(const std::exception& e) {
        basalt::log::error("basalt deepseek41","%s",e.what());
        for(const auto& a:active)if(a)error_line(a->request.rid,e.what());
        if(ids.empty())error_line(0,e.what());return 1;
    }
    return 0;
}
} // namespace basalt::program::generate
