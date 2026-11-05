#include "basalt/deepseek41/engram.hpp"
#include <limits>

namespace basalt::deepseek41 {
namespace {
std::vector<uint64_t> unsigned_array(const GgufFile& file,const std::string& name,uint64_t max_items) {
    const auto a=file.full_array(name,max_items);
    expect(a.elem!=MetaType::STRING && a.elem!=MetaType::F32 && a.elem!=MetaType::F64 && a.elem!=MetaType::ARRAY,
           "integer array required: "+name);
    std::vector<uint64_t> out;
    for (const auto& v:a.items) {
        expect(v.u<=uint64_t((std::numeric_limits<int64_t>::max)()),"negative or excessive array value: "+name);
        out.push_back(v.u);
    }
    return out;
}
std::vector<uint32_t> read_token_map(const GgufFile& file) {
    const auto values = unsigned_array(file, "deepseek41.engram.token_map", 129280);
    std::vector<uint32_t> map;
    for (auto v : values) {
        expect(v <= UINT32_MAX, "Engram token map value exceeds uint32");
        map.push_back(uint32_t(v));
    }
    return map;
}
}

EngramHasher::EngramHasher(const GgufFile& file) : EngramHasher(file, read_token_map(file)) {}

EngramHasher::EngramHasher(const GgufFile& file, std::vector<uint32_t> map) {
    const auto& m=file.metadata();
    expect(required(m,"general.architecture").s=="deepseek41","Engram requires deepseek41 metadata");
    expect(integers(m,"deepseek41.ple.layers")==std::vector<int>({1,14}) &&
           integer(m,"deepseek41.ple.ngram_size")==4 && integer(m,"deepseek41.ple.heads_per_ngram")==8 &&
           integer(m,"deepseek41.embedding_length_per_layer_input")==256,"unsupported Engram geometry");
    const int compressed=integer(m,"deepseek41.engram.compressed_vocab_size");
    const int pad=integer(m,"deepseek41.engram.pad_token_id");
    expect(compressed>0 && !map.empty() && pad>=0 && size_t(pad)<map.size(),"invalid Engram token map");
    for (auto token:map) expect(token<uint64_t(compressed),"Engram token map value exceeds vocabulary");
    token_map_ = std::move(map);
    pad_=token_map_[size_t(pad)];
    const auto mult=unsigned_array(file,"deepseek41.ple.layer_multipliers",8);
    const auto offset=unsigned_array(file,"deepseek41.ple.head_offsets",48);
    const auto mod=unsigned_array(file,"deepseek41.ple.head_vocab_sizes",48);
    expect(mult.size()==8 && offset.size()==48 && mod.size()==48,"incomplete Engram hash parameters");
    for (size_t i=0;i<8;++i) {
        expect(mult[i] && mult[i]<=uint64_t((std::numeric_limits<int64_t>::max)())/uint64_t(compressed),
               "Engram hash product exceeds signed arithmetic range");
        multipliers_[i]=mult[i];
    }
    for (size_t i=0;i<48;++i) {
        expect(offset[i]==rows_ && mod[i]>0 && mod[i]<=UINT32_MAX && rows_<=UINT32_MAX-mod[i],"invalid Engram bucket range");
        offsets_[i]=offset[i]; moduli_[i]=uint32_t(mod[i]); rows_+=mod[i];
    }
}

EngramHasher::Rows EngramHasher::advance(int32_t token,History& history) const {
    expect(token>=0 && size_t(token)<token_map_.size(),"Engram token is outside the vocabulary");
    const uint32_t mapped=token_map_[size_t(token)];
    Rows out{};
    for (size_t layer=0;layer<2;++layer) {
        uint64_t rolling=uint64_t(mapped)*multipliers_[layer*4];
        for (size_t n=1;n<4;++n) {
            rolling^=uint64_t(history.previous[n-1])*multipliers_[layer*4+n];
            for (size_t h=0;h<8;++h) {
                const size_t column=layer*24+(n-1)*8+h;
                out[column]=uint32_t(offsets_[column]+rolling%moduli_[column]);
            }
        }
    }
    history.previous[2]=history.previous[1]; history.previous[1]=history.previous[0]; history.previous[0]=mapped;
    return out;
}
}  // namespace basalt::deepseek41
