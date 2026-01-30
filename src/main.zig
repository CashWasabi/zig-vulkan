const texture_mapping_main = @import("vulkan_tutorial/04_texture_mapping/texture_mapping.zig").main;
pub fn main() !void {
    try texture_mapping_main();
}
