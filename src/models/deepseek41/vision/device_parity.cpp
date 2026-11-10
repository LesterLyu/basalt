// Compare owned and borrowed image storage on each selected device, including caller-device restoration.
#include "basalt/models/deepseek41/vision.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace basalt::models::deepseek41;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("usage: %s MODEL.basalt IMAGE\n", argv[0]);
        return 2;
    }
    std::string err;
    basalt::core::ModelFile pack;
    basalt::vision::Mmproj mp;
    if (!pack.open(argv[1],err) || !mp.open(pack.range("vision/mmproj"),err,"deepseek41v")) {
        std::printf("FAIL %s\n",err.c_str()); return 1;
    }
    std::ifstream input(argv[2],std::ios::binary);
    const std::vector<uint8_t> image((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
    basalt::vision::Image decoded;
    if (!basalt::vision::decode_image(image.data(),image.size(),decoded,err)) {
        std::printf("FAIL %s\n",err.c_str()); return 1;
    }
    const VisionPlan plan = plan_image(decoded.w,decoded.h,mp.hp().merge,mp.hp().min_pixels);
    std::vector<float> rows, first;
    image_patches(decoded,plan,rows);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::printf("FAIL two devices required\n"); return 1;
    }
    cudaSetDevice(0);
    for (int device = 0; device < 2; ++device) {
        const basalt::core::OnDevice on(device);
        cudaDeviceSynchronize();
        size_t before = 0, warm = 0, after = 0, total = 0;
        cudaMemGetInfo(&before,&total);
        {
            VisionBlas probe;
            if (!probe.open(err)) { std::printf("FAIL blas probe %s\n",err.c_str()); return 1; }
            cudaDeviceSynchronize();
            cudaMemGetInfo(&warm,&total);
        }
        cudaDeviceSynchronize();
        cudaMemGetInfo(&after,&total);
        std::printf("BLAS device %d: warm %.3f MiB, returned %.3f MiB, retained %.3f MiB\n",device,
                    ((double)before-warm)/1048576.,((double)after-warm)/1048576.,((double)before-after)/1048576.);
    }
    for (int device = 0; device < 2; ++device) {
        VisionBlas blas;
        std::vector<float> reference, actual;
        {
            VisionEncoder owned;
            {
                const basalt::core::OnDevice on(device);
                if (!owned.init(mp,plan.n_patches(),blas,err)) {
                    std::printf("FAIL init %s\n",err.c_str()); return 1;
                }
            }
            if (!owned.encode(rows,plan,reference,err)) {
                std::printf("FAIL encode %s\n",err.c_str()); return 1;
            }
            std::vector<float> again;
            if (!owned.encode(rows,plan,again,err) || again.size() != reference.size() ||
                std::memcmp(reference.data(),again.data(),reference.size()*sizeof(float)) != 0) {
                std::printf("FAIL device %d owned same-device repeat\n",device); return 1;
            }
            std::printf("PASS device %d owned same-device repeat: bitwise identical\n",device);
        }
        const uint64_t needed = VisionEncoder::required_bytes(mp,plan);
        basalt::core::DeviceArena arena(needed+512,device);
        cudaSetDevice(0);
        {
            const basalt::core::OnDevice on(device);
            cudaMemset(arena.base(),0xee,(size_t)arena.capacity());
        }
        {
            VisionEncoder borrowed;
            {
                const basalt::core::OnDevice on(device);
                if (!borrowed.init_borrowed(mp,blas,static_cast<uint8_t*>(arena.base())+256,needed,err)) {
                    std::printf("FAIL borrow %s\n",err.c_str()); return 1;
                }
            }
            if (!borrowed.fits(plan) || !borrowed.encode(rows,plan,actual,err)) {
                std::printf("FAIL borrowed encode %s\n",err.c_str()); return 1;
            }
            std::vector<float> again;
            if (!borrowed.encode(rows,plan,again,err) || again.size() != actual.size() ||
                std::memcmp(actual.data(),again.data(),actual.size()*sizeof(float)) != 0) {
                std::printf("FAIL device %d borrowed same-device repeat\n",device); return 1;
            }
            std::printf("PASS device %d borrowed same-device repeat: bitwise identical\n",device);
        }
        int current = -1;
        cudaGetDevice(&current);
        uint8_t guard[512] = {};
        {
            const basalt::core::OnDevice on(device);
            if (cudaMemcpy(guard,arena.base(),256,cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(guard+256,static_cast<uint8_t*>(arena.base())+256+needed,256,cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::printf("FAIL borrowed storage was freed\n"); return 1;
            }
        }
        const bool equal = reference.size() == actual.size() &&
            std::memcmp(reference.data(),actual.data(),reference.size()*sizeof(float)) == 0;
        const bool guards = std::all_of(std::begin(guard),std::end(guard),[](uint8_t v) { return v == 0xee; });
        if (current != 0 || !equal || !guards) {
            std::printf("FAIL device %d: caller=%d bitwise=%d guards=%d\n",device,current,(int)equal,(int)guards); return 1;
        }
        double max_abs = 0, square = 0, norm = 0;
        if (first.empty()) first = actual;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double delta = (double)actual[i]-first[i];
            max_abs = std::max(max_abs,std::abs(delta));
            square += delta*delta;
            norm += (double)first[i]*first[i];
        }
        std::printf("PASS device %d: borrowed bitwise equal; %llu MiB; cross-device max_abs %.9g rel_rms %.9g\n",
                    device,(unsigned long long)(needed>>20),max_abs,std::sqrt(square/(norm+1e-30)));
    }
    return 0;
}
