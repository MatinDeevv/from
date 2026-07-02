#include "model/sequence_model.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>

int main() {
    const char* path = "checkpoint_smoke.from";
    from::SequenceModel model;
    from::SequenceModelIO::save(model, path);

    from::SequenceModelIO::CheckpointInfo info;
    std::string error;
    const bool valid = from::SequenceModelIO::inspect(path, &info, &error);
    std::remove(path);
    if (!valid || info.parameter_count != from::SequenceModelIO::kParameterCount) {
        std::cerr << "checkpoint smoke failed: " << error << "\n";
        return 1;
    }
    {
        std::ofstream legacy("legacy_checkpoint_smoke.from", std::ios::binary);
        legacy.write("FSQ3", 4);
    }
    const bool legacy_accepted = from::SequenceModelIO::inspect("legacy_checkpoint_smoke.from", nullptr, &error);
    std::remove("legacy_checkpoint_smoke.from");
    if (legacy_accepted) {
        std::cerr << "checkpoint smoke failed: legacy checkpoint was accepted\n";
        return 1;
    }
    std::cout << "checkpoint smoke passed: " << info.parameter_count << " parameters\n";
    return 0;
}
