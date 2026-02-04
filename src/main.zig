const std = @import("std");

// const run = @import("vulkan_tutorial/07_generating_mipmaps/generating_mipmaps.zig").main;
const run = @import("vulkan_tutorial/08_multisampling/multisampling.zig").main;

pub fn main() !void {
    try run();
}
