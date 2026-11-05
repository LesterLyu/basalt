#pragma once
#include "basalt/deepseek41/model.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/grammar/guard.hpp"
#include <optional>
#include "basalt/program/tokenizer.hpp"
namespace basalt::program::generate {
int deepseek_session(deepseek41::Model& model,const Options& options,int window,int tuned_prefill,
                     const std::optional<grammar::PackVocab>& vocab,Tokenizer* tokenizer);
}
