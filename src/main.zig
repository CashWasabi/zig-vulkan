// const vertex_buffer_main = @import("vulkan_tutorial/vertex_buffers/01_basic.zig").main;
// const vertex_buffer_main = @import("vulkan_tutorial/vertex_buffers/02_staging_buffer.zig").main;
const vertex_buffer_main = @import("vulkan_tutorial/vertex_buffers/03_index_buffer.zig").main;
pub fn main() !void {
    try vertex_buffer_main();
}
