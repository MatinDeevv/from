#include "commands.hpp"

#include "model/sequence_model.hpp"

#include <iomanip>
#include <iostream>

namespace from {

int run_verify_checkpoint(const CliArgs& args) {
    if (args.has("--help") || !args.has("--checkpoint")) {
        std::cout << "Usage: from verify-checkpoint --checkpoint <path>\n"
                  << "Validates a SequenceModel FROM v1 checkpoint without training or inference.\n";
        return args.has("--help") ? 0 : 2;
    }

    SequenceModelIO::CheckpointInfo info;
    std::string error;
    const std::string path = args.get("--checkpoint");
    if (!SequenceModelIO::inspect(path, &info, &error)) {
        std::cerr << "CHECKPOINT INVALID: " << path << " - " << error << "\n";
        return 1;
    }

    std::cout << "CHECKPOINT VALID\n"
              << "  path: " << path << "\n"
              << "  magic: FROM\n"
              << "  version: " << info.version << "\n"
              << "  architecture_hash: 0x" << std::hex << info.architecture_hash << std::dec << "\n"
              << "  parameter_count: " << info.parameter_count << "\n"
              << "  normalization_stats: " << (info.normalization_stats_present ? "present" : "absent") << "\n";
    return 0;
}

}  // namespace from
