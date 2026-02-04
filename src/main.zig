const std = @import("std");

const run = @import("vulkan_tutorial/07_generating_mipmaps/generating_mipmaps.zig").main;

pub fn main() !void {
    try run();
}
