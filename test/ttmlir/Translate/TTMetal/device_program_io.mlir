// RUN: ttmlir-opt --ttcore-register-device %s -o %t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir -o %t.ttm

// A program may take and return device buffers, not only host tensors: a load
// program writes a host tensor into a tiled DRAM buffer and returns it, and a
// run program takes that buffer as an input (resident parameters). The
// flatbuffer describes such a tensor by its device shape with tiles expanded
// to elements and the tile's data type.

#dram = #ttcore.memory_space<dram>
module {
  func.func @load(%arg0: memref<64x64xbf16>) -> memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #dram> attributes {tt.function_type = "forward_device"} {
    %0 = "ttmetal.create_buffer"() <{address = 2968640 : i64}> : () -> memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #dram>
    %1 = "ttmetal.create_buffer"() <{address = 2976832 : i64}> : () -> memref<1x1x64x64xbf16, #ttcore.shard<128x2, 1>, #dram>
    "ttmetal.enqueue_write_buffer"(%arg0, %1) : (memref<64x64xbf16>, memref<1x1x64x64xbf16, #ttcore.shard<128x2, 1>, #dram>) -> ()
    "ttmetal.deallocate_buffer"(%1) : (memref<1x1x64x64xbf16, #ttcore.shard<128x2, 1>, #dram>) -> ()
    return %0 : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #dram>
  }
  func.func @run(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #dram>) -> memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #dram> attributes {tt.function_type = "forward_device"} {
    "ttmetal.finish"() : () -> ()
    return %arg0 : memref<1x1x2x2x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x2048, 1>, #dram>
  }
}
