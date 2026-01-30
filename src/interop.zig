const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("signal.h");
    @cInclude("GLFW/glfw3.h");
    // @cInclude("vulkan/vulkan.h");
});
const C = std.builtin.CallingConvention.c;
const std = @import("std");

// ==========================================
// PACKED TYPES BECAUSE THEIR LAYOUT MATTERS
// ==========================================
const Vec2 = extern struct { x: f32, y: f32 };
const Vec3 = extern struct { x: f32, y: f32, z: f32 };
const Vec4 = extern struct { x: f32, y: f32, z: f32, w: f32 };

const VertexIndex = enum(u16) { _ };
// const VertexIndex = enum(u32) { _ };

const Vertex = extern struct {
    pos: Vec2,
    color: Vec3,

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            // NOTE: this also exists! VK_VERTEX_INPUT_RATE_INSTANCE
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    // TODO: this doesn't work we need to allocate memory or have a static array somewhere
    pub fn getAttributeDescriptions() [2]c.VkVertexInputAttributeDescription {
        var attribute_descriptions: [2]c.VkVertexInputAttributeDescription = undefined;
        attribute_descriptions[0] = .{
            .binding = 0,
            .location = 0,
            .format = c.VK_FORMAT_R32G32_SFLOAT,
            .offset = @offsetOf(Vertex, "pos"),
        };
        attribute_descriptions[1] = .{
            .binding = 0,
            .location = 1,
            .format = c.VK_FORMAT_R32G32B32_SFLOAT,
            .offset = @offsetOf(Vertex, "color"),
        };

        return attribute_descriptions;
    }
};

// Column-major 4x4 matrix, compatible with GLSL mat4
// Each column is a Vec4, stored contiguously
const Mat4 = extern struct {
    cols: [4]Vec4,

    pub const identity: Mat4 = .{
        .cols = .{
            .{ .x = 1, .y = 0, .z = 0, .w = 0 },
            .{ .x = 0, .y = 1, .z = 0, .w = 0 },
            .{ .x = 0, .y = 0, .z = 1, .w = 0 },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        },
    };

    // Access element at (row, col)
    pub fn at(self: Mat4, row: usize, col: usize) f32 {
        const col_vec = self.cols[col];
        return switch (row) {
            0 => col_vec.x,
            1 => col_vec.y,
            2 => col_vec.z,
            3 => col_vec.w,
            else => unreachable,
        };
    }

    pub fn set(self: *Mat4, row: usize, col: usize, val: f32) void {
        switch (row) {
            0 => self.cols[col].x = val,
            1 => self.cols[col].y = val,
            2 => self.cols[col].z = val,
            3 => self.cols[col].w = val,
            else => unreachable,
        }
    }
};

pub fn mat4Translation(tx: f32, ty: f32, tz: f32) Mat4 {
    return .{
        .cols = .{
            .{ .x = 1, .y = 0, .z = 0, .w = 0 },
            .{ .x = 0, .y = 1, .z = 0, .w = 0 },
            .{ .x = 0, .y = 0, .z = 1, .w = 0 },
            .{ .x = tx, .y = ty, .z = tz, .w = 1 },
        },
    };
}

pub fn mat4Scale(sx: f32, sy: f32, sz: f32) Mat4 {
    return .{
        .cols = .{
            .{ .x = sx, .y = 0, .z = 0, .w = 0 },
            .{ .x = 0, .y = sy, .z = 0, .w = 0 },
            .{ .x = 0, .y = 0, .z = sz, .w = 0 },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        },
    };
}

pub fn mat4RotationZ(angle: f32) Mat4 {
    const cos = @cos(angle);
    const sin = @sin(angle);
    return .{
        .cols = .{
            .{ .x = cos, .y = sin, .z = 0, .w = 0 },
            .{ .x = -sin, .y = cos, .z = 0, .w = 0 },
            .{ .x = 0, .y = 0, .z = 1, .w = 0 },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        },
    };
}

pub fn mat4Perspective(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4 {
    const tan_half_fov = @tan(fov_y / 2.0);
    return .{
        .cols = .{
            .{ .x = 1.0 / (aspect * tan_half_fov), .y = 0, .z = 0, .w = 0 },
            .{ .x = 0, .y = 1.0 / tan_half_fov, .z = 0, .w = 0 },
            .{ .x = 0, .y = 0, .z = -(far + near) / (far - near), .w = -1 },
            .{ .x = 0, .y = 0, .z = -(2.0 * far * near) / (far - near), .w = 0 },
        },
    };
}

pub fn mat4Mul(a: Mat4, b: Mat4) Mat4 {
    var result: Mat4 = undefined;
    inline for (0..4) |col| {
        inline for (0..4) |row| {
            var sum: f32 = 0;
            inline for (0..4) |k| {
                sum += a.at(row, k) * b.at(k, col);
            }
            result.set(row, col, sum);
        }
    }
    return result;
}

const UniformBufferObject = extern struct {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
};

// NOTE: make sure we have the correct alignment
comptime {
    std.debug.assert(@sizeOf(Vec4) == 16);
    std.debug.assert(@alignOf(Vec4) == 4);
    std.debug.assert(@sizeOf(Mat4) == 64);
}
