const builtin = @import("builtin");
// const build_options = @import("build_options");

const c = @cImport({
    @cDefine("STB_IMAGE_IMPLEMENTATION", "");
    // Disable SIMD to fix alignment issues with Zig's cImport
    @cDefine("STBI_NO_SIMD", "");
    // some implementations have issues so we only want to enable necessary formats
    @cDefine("STBI_ONLY_JPEG", "");
    @cInclude("stb_image.h");
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("signal.h");
    @cInclude("GLFW/glfw3.h");
    // @cInclude("vulkan/vulkan.h");
});

const C = std.builtin.CallingConvention.c;
const std = @import("std");

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const MAX_FRAMES_IN_FLIGHT = 2;

// enable validation layers for debug mode
const ENABLE_VALIDATION_LAYERS: bool = builtin.mode == .Debug;

const VALIDATION_LAYERS = [_][*c]const u8{
    "VK_LAYER_KHRONOS_validation",
};

var DEVICE_EXTENSIONS = [_][*c]const u8{
    c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

const ENABLE_DYNAMIC_STATE: bool = true;
var DYNAMIC_STATES = [_]c_uint{
    c.VK_DYNAMIC_STATE_VIEWPORT,
    c.VK_DYNAMIC_STATE_SCISSOR,
};

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(indices: QueueFamilyIndices) bool {
        return indices.graphics_family != null and
            indices.present_family != null;
    }
};

const SwapChainSupportDetails = struct {
    capabilities: c.VkSurfaceCapabilitiesKHR = .{},
    formats: std.ArrayList(c.VkSurfaceFormatKHR) = .empty,
    present_modes: std.ArrayList(c.VkPresentModeKHR) = .empty,

    pub fn deinit(self: *SwapChainSupportDetails, allocator: std.mem.Allocator) void {
        self.formats.deinit(allocator);
        self.present_modes.deinit(allocator);
    }
};

// ==========================================
// PACKED TYPES BECAUSE THEIR LAYOUT MATTERS
// ==========================================
const Vec2 = extern struct { x: f32, y: f32 };
const Vec3 = extern struct { x: f32, y: f32, z: f32 };
const Vec4 = extern struct { x: f32, y: f32, z: f32, w: f32 };

const VertexIndex = u16; // can also be u32!
const Vertex = extern struct {
    pos: Vec3,
    color: Vec3,
    texCoord: Vec2,

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            // NOTE: this also exists! VK_VERTEX_INPUT_RATE_INSTANCE
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    pub fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
        return [_]c.VkVertexInputAttributeDescription{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32B32_SFLOAT,
                .offset = @offsetOf(Vertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R32G32B32_SFLOAT,
                .offset = @offsetOf(Vertex, "color"),
            },
            .{
                .binding = 0,
                .location = 2,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(Vertex, "texCoord"),
            },
        };
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

fn sub3(a: Vec3, b: Vec3) Vec3 {
    return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
}

fn dot3(a: Vec3, b: Vec3) f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        .x = a.y * b.z - a.z * b.y,
        .y = a.z * b.x - a.x * b.z,
        .z = a.x * b.y - a.y * b.x,
    };
}

fn normalize3(v: Vec3) Vec3 {
    const len = @sqrt(dot3(v, v));
    return .{ .x = v.x / len, .y = v.y / len, .z = v.z / len };
}
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

pub fn mat4LookAt(eye: Vec3, target: Vec3, world_up: Vec3) Mat4 {
    // Forward vector (from target to eye, because we look down -Z)
    const f = normalize3(sub3(eye, target));
    // Right vector
    const r = normalize3(cross(world_up, f));
    // Up vector
    const u = cross(f, r);

    return .{
        .cols = .{
            .{ .x = r.x, .y = u.x, .z = f.x, .w = 0 },
            .{ .x = r.y, .y = u.y, .z = f.y, .w = 0 },
            .{ .x = r.z, .y = u.z, .z = f.z, .w = 0 },
            .{ .x = -dot3(r, eye), .y = -dot3(u, eye), .z = -dot3(f, eye), .w = 1 },
        },
    };
}

/// Rotation around an arbitrary axis (angle in radians)
pub fn mat4Rotation(angle: f32, axis: Vec3) Mat4 {
    const a = normalize3(axis);
    const cos = @cos(angle);
    const sin = @sin(angle);
    const omc = 1.0 - cos; // one minus cosine

    return .{
        .cols = .{
            .{
                .x = cos + a.x * a.x * omc,
                .y = a.y * a.x * omc + a.z * sin,
                .z = a.z * a.x * omc - a.y * sin,
                .w = 0,
            },
            .{
                .x = a.x * a.y * omc - a.z * sin,
                .y = cos + a.y * a.y * omc,
                .z = a.z * a.y * omc + a.x * sin,
                .w = 0,
            },
            .{
                .x = a.x * a.z * omc + a.y * sin,
                .y = a.y * a.z * omc - a.x * sin,
                .z = cos + a.z * a.z * omc,
                .w = 0,
            },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        },
    };
}

const UniformBufferObject = extern struct {
    pub const default: UniformBufferObject = .{ .model = .identity, .view = .identity, .proj = .identity };
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

const Context = struct {
    dbga: std.heap.DebugAllocator(.{}) = .init,

    debug_messenger: c.VkDebugUtilsMessengerEXT = null,

    queue_familiy_indices: QueueFamilyIndices = undefined,
    swap_chain_image_format: c.VkFormat = undefined,
    swap_chain_extent: c.VkExtent2D = undefined,
    swap_chain_images: std.ArrayList(c.VkImage) = .empty,
    swap_chain_image_views: std.ArrayList(c.VkImageView) = .empty,
    swap_chain_framebuffers: std.ArrayList(c.VkFramebuffer) = .empty,
    descriptor_set_layout: c.VkDescriptorSetLayout = null,

    window: ?*c.GLFWwindow = null,
    instance: c.VkInstance = null,
    surface: c.VkSurfaceKHR = null,

    physical_device: c.VkPhysicalDevice = null,
    logical_device: c.VkDevice = null,

    graphics_queue: c.VkQueue = null,
    present_queue: c.VkQueue = null,

    pipeline_layout: c.VkPipelineLayout = null,
    graphics_pipeline: c.VkPipeline = null,
    swap_chain: c.VkSwapchainKHR = null,
    render_pass: c.VkRenderPass = null,

    command_pool: c.VkCommandPool = null,
    descriptor_pool: c.VkDescriptorPool = null,

    vert_shader_module: c.VkShaderModule = null,
    frag_shader_module: c.VkShaderModule = null,

    index_buffer: c.VkBuffer = null,
    index_buffer_memory: c.VkDeviceMemory = null,
    vertex_buffer: c.VkBuffer = null,
    vertex_buffer_memory: c.VkDeviceMemory = null,

    texture_image: c.VkImage = null,
    texture_image_memory: c.VkDeviceMemory = null,
    texture_image_view: c.VkImageView = null,
    texture_sampler: c.VkSampler = null,

    depth_image: c.VkImage = null,
    depth_image_memory: c.VkDeviceMemory = null,
    depth_image_view: c.VkImageView = null,

    vertices: []const Vertex = &[_]Vertex{
        // ==============
        // UPPER TEXTURE
        // ==============

        // {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
        .{
            .pos = .{ .x = -0.5, .y = -0.5, .z = 0 },
            .color = .{ .x = 1, .y = 0, .z = 0 },
            .texCoord = .{ .x = 0, .y = 0 },
        },
        // {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        .{
            .pos = .{ .x = 0.5, .y = -0.5, .z = 0 },
            .color = .{ .x = 0, .y = 1, .z = 0 },
            .texCoord = .{ .x = 1, .y = 0 },
        },
        // {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        .{
            .pos = .{ .x = 0.5, .y = 0.5, .z = 0 },
            .color = .{ .x = 0, .y = 0, .z = 1 },
            .texCoord = .{ .x = 1, .y = 1 },
        },
        // {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
        .{
            .pos = .{ .x = -0.5, .y = 0.5, .z = 0 },
            .color = .{ .x = 1, .y = 1, .z = 1 },
            .texCoord = .{ .x = 0, .y = 1 },
        },

        // ==============
        // LOWER TEXTURE
        // ==============

        // {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
        .{
            .pos = .{ .x = -0.5, .y = -0.5, .z = -0.5 },
            .color = .{ .x = 1, .y = 0, .z = 0 },
            .texCoord = .{ .x = 0, .y = 0 },
        },
        // {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        .{
            .pos = .{ .x = 0.5, .y = -0.5, .z = -0.5 },
            .color = .{ .x = 0, .y = 1, .z = 0 },
            .texCoord = .{ .x = 1, .y = 0 },
        },
        // {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        .{
            .pos = .{ .x = 0.5, .y = 0.5, .z = -0.5 },
            .color = .{ .x = 0, .y = 0, .z = 1 },
            .texCoord = .{ .x = 1, .y = 1 },
        },
        // {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
        .{
            .pos = .{ .x = -0.5, .y = 0.5, .z = -0.5 },
            .color = .{ .x = 1, .y = 1, .z = 1 },
            .texCoord = .{ .x = 0, .y = 1 },
        },
    },

    indices: []const VertexIndex = &[_]VertexIndex{
        // ==============
        // LOWER TEXTURE
        // ==============

        0, 1, 2, 2, 3, 0,

        // ==============
        // LOWER TEXTURE
        // ==============

        4, 5, 6, 6, 7, 4,
    },

    ubo: UniformBufferObject = .default,

    // ======================
    // per frame definitions
    // ======================

    descriptor_sets: [MAX_FRAMES_IN_FLIGHT]c.VkDescriptorSet = undefined,
    command_buffers: [MAX_FRAMES_IN_FLIGHT]c.VkCommandBuffer = undefined,

    // sync objects
    image_available_semaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore = undefined,
    render_finished_semaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore = undefined,
    in_flight_fences: [MAX_FRAMES_IN_FLIGHT]c.VkFence = undefined,
    uniform_buffers: [MAX_FRAMES_IN_FLIGHT]c.VkBuffer = undefined,
    uniform_buffers_memory: [MAX_FRAMES_IN_FLIGHT]c.VkDeviceMemory = undefined,
    uniform_buffers_mapped: [MAX_FRAMES_IN_FLIGHT]?*anyopaque = undefined,

    const init: Context = .{};

    pub fn run(self: *Context) void {
        self.initWindow();
        self.initVulkan();
        self.mainLoop();
        self.cleanup();
    }

    fn initWindow(self: *Context) void {
        if (c.glfwInit() == c.GLFW_FALSE) @panic("glwfInit failed!");

        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);

        self.window = c.glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", null, null);
    }

    fn initVulkan(self: *Context) void {
        self.createInstance() catch unreachable;
        self.setupDebugMessenger();
        self.createSurface();
        self.pickPhysicalDevice() catch unreachable;
        self.createLogicalDevice() catch unreachable;
        self.createSwapChain() catch unreachable;
        self.createImageViews();
        self.createRenderPass();
        self.createDescriptorSetLayout();
        self.createGraphicsPipeline() catch unreachable;
        self.createCommandPool();
        self.createDepthResources();
        self.createFramebuffers();
        self.createTextureImage();
        self.createTextureImageView();
        self.createTextureSampler();
        self.createVertexBuffer();
        self.createIndexBuffer();
        self.createUniformBuffers();
        self.createDescriptorPool();
        self.createDescriptorSets();
        self.createCommandBuffers();
        self.createSyncObjects();
    }

    fn cleanup(self: *Context) void {
        const allocator = self.dbga.allocator();

        self.cleanupSwapChain();

        //clean up sync objects
        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            c.vkDestroySemaphore(self.logical_device, self.image_available_semaphores[i], null);
            c.vkDestroySemaphore(self.logical_device, self.render_finished_semaphores[i], null);
            c.vkDestroyFence(self.logical_device, self.in_flight_fences[i], null);
        }

        c.vkFreeMemory(self.logical_device, self.texture_image_memory, null);

        c.vkDestroyShaderModule(self.logical_device, self.vert_shader_module, null);
        c.vkDestroyShaderModule(self.logical_device, self.frag_shader_module, null);

        // cleanup bufffers
        c.vkDestroyBuffer(self.logical_device, self.index_buffer, null);
        c.vkFreeMemory(self.logical_device, self.index_buffer_memory, null);
        c.vkDestroyBuffer(self.logical_device, self.vertex_buffer, null);
        c.vkFreeMemory(self.logical_device, self.vertex_buffer_memory, null);

        c.vkDestroyDescriptorSetLayout(self.logical_device, self.descriptor_set_layout, null);

        // cleanup pools
        c.vkDestroyDescriptorPool(self.logical_device, self.descriptor_pool, null);
        c.vkDestroyCommandPool(self.logical_device, self.command_pool, null);

        c.vkDestroyPipeline(self.logical_device, self.graphics_pipeline, null);
        c.vkDestroyPipelineLayout(self.logical_device, self.pipeline_layout, null);

        c.vkDestroySampler(self.logical_device, self.texture_sampler, null);

        self.swap_chain_images.deinit(allocator);

        c.vkDestroyRenderPass(self.logical_device, self.render_pass, null);

        c.vkDestroySwapchainKHR(self.logical_device, self.swap_chain, null);

        // destroy logical_device
        c.vkDestroyDevice(self.logical_device, null);

        // TODO: physical device?

        destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);

        // destroy surface (gpu window?)
        c.vkDestroySurfaceKHR(self.instance, self.surface, null);

        // destroy vulkan instance
        c.vkDestroyInstance(self.instance, null);

        // destroy glfw
        c.glfwDestroyWindow(self.window);
        c.glfwTerminate();

        if (self.dbga.deinit() == .leak) @panic("Debug allocator has leaked meory!");
    }

    fn mainLoop(self: *Context) void {
        const start_time_ms = std.time.milliTimestamp();
        var current_frame: usize = 0;

        // while (c.glfwWindowShouldClose(self.window) == c.GLFW_FALSE or c.glfwGetKey(window, c.GLFW_KEY_ESCAPE) != c.GLFW_PRESS) {}
        while (c.glfwWindowShouldClose(self.window) == c.GLFW_FALSE) {
            // getting dt
            const current_time_ms = std.time.milliTimestamp();
            const time_s: f32 = @as(f32, @floatFromInt(current_time_ms - start_time_ms)) / 1_000;

            c.glfwPollEvents();
            self.drawFrame(time_s, &current_frame);
        }

        if (c.vkDeviceWaitIdle(self.logical_device) != c.VK_SUCCESS) {
            @panic("Failed while trying to gracefully shutdown!");
        }
    }

    fn drawFrame(self: *Context, dt: f32, current_frame: *usize) void {
        const frame = current_frame.*;
        defer current_frame.* = (frame + 1) % MAX_FRAMES_IN_FLIGHT;

        if (c.vkWaitForFences(
            self.logical_device,
            1,
            &self.in_flight_fences[frame],
            c.VK_TRUE,
            c.UINT64_MAX,
        ) != c.VK_SUCCESS) {
            @panic("failed while trying to wait for fences in main loop.");
        }

        var image_index: u32 = 0;
        const result = c.vkAcquireNextImageKHR(
            self.logical_device,
            self.swap_chain,
            c.UINT64_MAX,
            self.image_available_semaphores[frame],
            null,
            &image_index,
        );
        if (result == c.VK_ERROR_OUT_OF_DATE_KHR) {
            self.recreateSwapChain() catch unreachable;
            return;
        } else if (result != c.VK_SUCCESS and result != c.VK_SUBOPTIMAL_KHR) {
            @panic("failed while trying to aquire next image_khr in main loop.");
        }

        // UPDATE UNIFORM BUFFER
        {
            // update UBO transform
            self.ubo.model = mat4Rotation(
                dt * std.math.degreesToRadians(90),
                .{ .x = 0.0, .y = 0.0, .z = 1.0 },
            );
            self.ubo.view = mat4LookAt(
                .{ .x = 2.0, .y = 2.0, .z = 2.0 },
                .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                .{ .x = 0.0, .y = 0.0, .z = 1.0 },
            );
            self.ubo.proj = mat4Perspective(
                std.math.degreesToRadians(45),
                @as(f32, @floatFromInt(self.swap_chain_extent.width)) / @as(f32, @floatFromInt(self.swap_chain_extent.height)),
                0.1,
                10.0,
            );

            // NOTE: ubo y flip with rasterizer
            self.ubo.proj.set(1, 1, self.ubo.proj.at(1, 1) * -1);

            // update UBO buffer
            const dst: [*]u8 = @ptrCast(self.uniform_buffers_mapped[frame]);
            const src: [*]const u8 = @ptrCast(&self.ubo);
            @memcpy(dst[0..@sizeOf(UniformBufferObject)], src[0..@sizeOf(UniformBufferObject)]);
        }

        if (c.vkResetFences(self.logical_device, 1, &self.in_flight_fences[frame]) != c.VK_SUCCESS) {
            @panic("failed while trying to reset fences in main loop.");
        }
        if (c.vkResetCommandBuffer(self.command_buffers[frame], 0) != c.VK_SUCCESS) {
            @panic("failed while trying to reset command buffer in main loop.");
        }

        self.recordCommandBuffer(
            self.command_buffers[frame],
            image_index,
            self.vertex_buffer,
            self.index_buffer,
            frame,
        );

        var wait_semaphores = [_]c.VkSemaphore{self.image_available_semaphores[frame]};
        var wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        var signal_semaphores = [_]c.VkSemaphore{self.render_finished_semaphores[frame]};

        var submit_info: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffers[frame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphores,
        };
        if (c.vkQueueSubmit(
            self.graphics_queue,
            1,
            &submit_info,
            self.in_flight_fences[frame],
        ) != c.VK_SUCCESS) {
            @panic("failed to submit draw command buffer!");
        }

        // TODO: I don't understand what we're doing here

        // // subpass dependencies
        // var dependency: c.VkSubpassDependency = .{
        //     .srcSubpass = c.VK_SUBPASS_EXTERNAL,
        //     .dstSubpass = 0,
        //     .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        //     .srcAccessMask = 0,
        //     .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        //     .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        // };
        //
        // render_pass_create_info.dependencyCount = 1;
        // render_pass_create_info.pDependencies = &dependency;

        // presentation
        const swap_chains = [_]c.VkSwapchainKHR{self.swap_chain};
        var present_info_khr: c.VkPresentInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signal_semaphores[0..].ptr,
            .swapchainCount = 1,
            .pSwapchains = swap_chains[0..].ptr,
            .pImageIndices = &image_index,
            .pResults = null,
        };
        if (c.vkQueuePresentKHR(self.present_queue, &present_info_khr) != c.VK_SUCCESS) {
            @panic("Oh man! Failed while trying to queue present_khr. This is the fun part with colors and bling!");
        }
    }

    fn createInstance(self: *Context) !void {
        const allocator = self.dbga.allocator();

        if (ENABLE_VALIDATION_LAYERS and try checkValidationLayerSupport(allocator) == false) {
            @panic("Validation Layers not available!");
        }

        var app_info: c.VkApplicationInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hello Depth Buffer",
            .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = c.VK_API_VERSION_1_0,
        };

        const extensions = try getRequiredExtensions(allocator);
        defer allocator.free(extensions);

        var create_info: c.VkInstanceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = @intCast(VALIDATION_LAYERS.len),
            .enabledExtensionCount = @intCast(extensions.len),
            .ppEnabledExtensionNames = extensions.ptr,
        };

        // SETUP VALIDATION LAYERS (OPTIONAL)
        var debug_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
        if (ENABLE_VALIDATION_LAYERS) {
            create_info.ppEnabledLayerNames = VALIDATION_LAYERS[0..].ptr;

            populateDebugMessengerCreateInfo(&debug_create_info);
            create_info.pNext = &debug_create_info;
        } else {
            create_info.enabledLayerCount = 0;
            create_info.pNext = null;
        }

        if (c.vkCreateInstance(&create_info, null, &self.instance) != c.VK_SUCCESS) {
            @panic("vkCreateInstance failed!");
        }
    }

    fn setupDebugMessenger(self: *Context) void {
        var extension_count: u32 = 0;
        if (c.vkEnumerateInstanceExtensionProperties(null, &extension_count, null) != c.VK_SUCCESS) {
            @panic("vkEnumerateInstanceExtensionProperties failed!");
        }

        var debug_messenger_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
        if (ENABLE_VALIDATION_LAYERS) {
            populateDebugMessengerCreateInfo(&debug_messenger_create_info);

            if (createDebugUtilsMessengerEXT(
                self.instance,
                &debug_messenger_create_info,
                null,
                &self.debug_messenger,
            ) != c.VK_SUCCESS) {
                @panic("failed to set up debug messenger!");
            }
        }
    }

    fn createSurface(self: *Context) void {
        if (c.glfwCreateWindowSurface(self.instance, self.window, null, &self.surface) != c.VK_SUCCESS) {
            @panic("failed to create window surface!");
        }
    }

    fn pickPhysicalDevice(self: *Context) !void {
        const allocator = self.dbga.allocator();

        var device_count: u32 = 0;
        if (c.vkEnumeratePhysicalDevices(self.instance, &device_count, null) != c.VK_SUCCESS) {
            @panic("failed calling vkEnumeratePhysicalDevices while looking for devices count!");
        }
        if (device_count == 0) @panic("failed to find GPUs with Vulkan support!");

        const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
        defer allocator.free(devices);

        if (c.vkEnumeratePhysicalDevices(self.instance, &device_count, devices.ptr) != c.VK_SUCCESS) {
            @panic("failed calling vkEnumeratePhysicalDevices while getting devices!");
        }
        for (devices) |device| {
            if (try isDeviceSuitable(allocator, device, self.surface)) {
                self.physical_device = device;
                break;
            }
        }
        if (self.physical_device == null) {
            @panic("failed to find a suitable GPU!");
        }
    }

    fn createLogicalDevice(self: *Context) !void {
        const allocator = self.dbga.allocator();

        self.queue_familiy_indices = try findQueueFamilies(
            allocator,
            self.physical_device,
            self.surface,
        );

        const graphics_queue_priority: f32 = 1.0;
        const graphics_queue_create_info: c.VkDeviceQueueCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = self.queue_familiy_indices.graphics_family.?,
            .queueCount = 1,
            .pQueuePriorities = &graphics_queue_priority,
        };

        const present_queue_priority: f32 = 1.0;
        const present_queue_create_info: c.VkDeviceQueueCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = self.queue_familiy_indices.present_family.?,
            .queueCount = 1,
            .pQueuePriorities = &present_queue_priority,
        };

        // TODO: how does this even work?
        // if both have the same family index we can skip one
        var queue_slice: []c.VkDeviceQueueCreateInfo = undefined;
        defer allocator.free(queue_slice);

        if (graphics_queue_create_info.queueFamilyIndex == present_queue_create_info.queueFamilyIndex) {
            queue_slice = try allocator.alloc(c.VkDeviceQueueCreateInfo, 1);
            queue_slice[0] = graphics_queue_create_info;
        } else {
            queue_slice = try allocator.alloc(c.VkDeviceQueueCreateInfo, 2);
            queue_slice[0] = graphics_queue_create_info;
            queue_slice[1] = present_queue_create_info;
        }

        // ======================
        // CREATE LOGICAL DEVICE
        // ======================
        const device_features: c.VkPhysicalDeviceFeatures = .{
            .samplerAnisotropy = c.VK_TRUE,
        };
        var logical_device_create_info: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pEnabledFeatures = &device_features,
            // register our queues BEFORE we create our logical device
            .queueCreateInfoCount = @intCast(queue_slice.len), // graphics queue and present queue
            .pQueueCreateInfos = queue_slice.ptr,
            // enable extensions
            .enabledExtensionCount = @intCast(DEVICE_EXTENSIONS.len),
            .ppEnabledExtensionNames = DEVICE_EXTENSIONS[0..].ptr,
        };

        // enable validation layers (optional)
        if (ENABLE_VALIDATION_LAYERS) {
            logical_device_create_info.enabledLayerCount = @intCast(VALIDATION_LAYERS.len);
            logical_device_create_info.ppEnabledLayerNames = VALIDATION_LAYERS[0..].ptr;
        } else {
            logical_device_create_info.enabledLayerCount = 0;
        }

        if (c.vkCreateDevice(
            self.physical_device,
            &logical_device_create_info,
            null,
            &self.logical_device,
        ) != c.VK_SUCCESS) {
            @panic("failed to create logical device!");
        }

        // TODO: MOVE THIS INTO ITS OWN THING ?
        // CREATE GRAPHICS QUEUE
        // CREATE PRESENT QUEUE

        // ======================
        // CREATE GRAPHICS QUEUE
        // ======================
        c.vkGetDeviceQueue(
            self.logical_device,
            self.queue_familiy_indices.graphics_family.?,
            0,
            &self.graphics_queue,
        );

        // =====================
        // CREATE PRESENT QUEUE
        // =====================
        c.vkGetDeviceQueue(
            self.logical_device,
            self.queue_familiy_indices.present_family.?,
            0,
            &self.present_queue,
        );
    }

    fn createSwapChain(self: *Context) !void {
        const allocator = self.dbga.allocator();

        var swap_chain_support: SwapChainSupportDetails = try querySwapChainSupport(
            allocator,
            self.physical_device,
            self.surface,
        );
        defer swap_chain_support.deinit(allocator);

        const surface_format: c.VkSurfaceFormatKHR = chooseSwapSurfaceFormat(swap_chain_support.formats.items);
        const present_mode: c.VkPresentModeKHR = chooseSwapPresentMode(swap_chain_support.present_modes.items);
        const extent: c.VkExtent2D = chooseSwapExtent(self.window.?, swap_chain_support.capabilities);

        var image_count: u32 = swap_chain_support.capabilities.minImageCount + 1;
        if (swap_chain_support.capabilities.maxImageCount > 0 and image_count > swap_chain_support.capabilities.maxImageCount) {
            image_count = swap_chain_support.capabilities.maxImageCount;
        }
        var swap_chain_create_info: c.VkSwapchainCreateInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = self.surface,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = swap_chain_support.capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
        };

        var queue_family_indices: [2]u32 = undefined;
        queue_family_indices[0] = self.queue_familiy_indices.graphics_family.?;
        queue_family_indices[1] = self.queue_familiy_indices.present_family.?;

        if (self.queue_familiy_indices.graphics_family.? != self.queue_familiy_indices.present_family.?) {
            swap_chain_create_info.imageSharingMode = c.VK_SHARING_MODE_CONCURRENT;
            swap_chain_create_info.queueFamilyIndexCount = 2;
            swap_chain_create_info.pQueueFamilyIndices = &queue_family_indices;
        } else {
            swap_chain_create_info.imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
            swap_chain_create_info.queueFamilyIndexCount = 0; // Optional
            swap_chain_create_info.pQueueFamilyIndices = null; // Optional
        }

        if (c.vkCreateSwapchainKHR(
            self.logical_device,
            &swap_chain_create_info,
            null,
            &self.swap_chain,
        ) != c.VK_SUCCESS) {
            @panic("failed to create swap chain!");
        }

        var swap_chain_image_count: u32 = 0;
        if (c.vkGetSwapchainImagesKHR(
            self.logical_device,
            self.swap_chain,
            &swap_chain_image_count,
            null,
        ) != c.VK_SUCCESS) {
            @panic("failed getting swap chain image count from vkGetSwapchainImagesKHR function call");
        }
        try self.swap_chain_images.resize(allocator, swap_chain_image_count);
        if (c.vkGetSwapchainImagesKHR(
            self.logical_device,
            self.swap_chain,
            &swap_chain_image_count,
            self.swap_chain_images.items.ptr,
        ) != c.VK_SUCCESS) {
            @panic("failed getting swap chain images from vkGetSwapchainImagesKHR function call");
        }

        self.swap_chain_image_format = surface_format.format;
        self.swap_chain_extent = extent;
    }

    fn recreateSwapChain(self: *Context) !void {
        var width: i32 = 0;
        var height: i32 = 0;
        while (width == 0 or height == 0) {
            c.glfwGetFramebufferSize(self.window, &width, &height);
            c.glfwWaitEvents();
        }

        if (c.vkDeviceWaitIdle(self.logical_device) != c.VK_SUCCESS) {
            @panic("failed waiting on device.");
        }

        self.cleanupSwapChain();

        try self.createSwapChain();
        self.createImageViews();
        self.createDepthResources();
        self.createFramebuffers();
    }

    fn cleanupSwapChain(self: *Context) void {
        const allocator = self.dbga.allocator();

        c.vkDestroyImageView(self.logical_device, self.depth_image_view, null);
        c.vkDestroyImage(self.logical_device, self.depth_image, null);
        c.vkFreeMemory(self.logical_device, self.depth_image_memory, null);

        for (self.swap_chain_framebuffers.items) |framebuffer| {
            c.vkDestroyFramebuffer(self.logical_device, framebuffer, null);
        }
        self.swap_chain_framebuffers.deinit(allocator);

        for (self.swap_chain_image_views.items) |image_view| {
            c.vkDestroyImageView(self.logical_device, image_view, null);
        }
        self.swap_chain_image_views.deinit(allocator);

        c.vkDestroySwapchainKHR(self.logical_device, self.swap_chain, null);
    }

    fn createRenderPass(self: *Context) void {
        const color_attachment: c.VkAttachmentDescription = .{
            .format = self.swap_chain_image_format,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        const depth_attachment: c.VkAttachmentDescription = .{
            .format = self.findDepthFormat(),
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        var color_attachment_ref: c.VkAttachmentReference = .{
            .attachment = 0,
            .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        var depth_attachment_ref: c.VkAttachmentReference = .{
            .attachment = 1,
            .layout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        var subpass: c.VkSubpassDescription = .{
            .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_ref,
            .pDepthStencilAttachment = &depth_attachment_ref,
        };

        var dependency: c.VkSubpassDependency = .{
            .srcSubpass = c.VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | c.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | c.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        };

        var attachments = [_]c.VkAttachmentDescription{ color_attachment, depth_attachment };
        var render_pass_create_info: c.VkRenderPassCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = attachments.len,
            .pAttachments = &attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        if (c.vkCreateRenderPass(
            self.logical_device,
            &render_pass_create_info,
            null,
            &self.render_pass,
        ) != c.VK_SUCCESS) {
            @panic("failed to create render pass!");
        }
    }

    fn createDescriptorSetLayout(self: *Context) void {
        const ubo_layout_binding: c.VkDescriptorSetLayoutBinding = .{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = null, // Optional
        };

        const sampler_layout_binding: c.VkDescriptorSetLayoutBinding = .{
            .binding = 1,
            .descriptorCount = 1,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImmutableSamplers = null,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };

        const bindings = [_]c.VkDescriptorSetLayoutBinding{
            ubo_layout_binding,
            sampler_layout_binding,
        };

        const layout_info: c.VkDescriptorSetLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = bindings.len,
            .pBindings = &bindings,
        };
        if (c.vkCreateDescriptorSetLayout(
            self.logical_device,
            &layout_info,
            null,
            &self.descriptor_set_layout,
        ) != c.VK_SUCCESS) {
            @panic("failed to create descriptor set layout!");
        }
    }

    fn createDescriptorPool(self: *Context) void {
        const descriptor_pool_sizes = [_]c.VkDescriptorPoolSize{
            .{
                .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = @intCast(MAX_FRAMES_IN_FLIGHT),
            },
            .{
                .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = @intCast(MAX_FRAMES_IN_FLIGHT),
            },
        };

        const descriptor_pool_info: c.VkDescriptorPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .poolSizeCount = descriptor_pool_sizes.len,
            .pPoolSizes = &descriptor_pool_sizes,
            .maxSets = @intCast(MAX_FRAMES_IN_FLIGHT),
        };

        if (c.vkCreateDescriptorPool(
            self.logical_device,
            &descriptor_pool_info,
            null,
            &self.descriptor_pool,
        ) != c.VK_SUCCESS) {
            @panic("failed to create descriptor pool!");
        }
    }

    fn createDescriptorSets(self: *Context) void {
        // TODO: is this a correct translation?
        // std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        const descriptor_set_layouts = [MAX_FRAMES_IN_FLIGHT]c.VkDescriptorSetLayout{
            self.descriptor_set_layout,
            self.descriptor_set_layout,
        };

        const descriptor_set_alloc_info: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = @intCast(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = &descriptor_set_layouts,
        };

        if (c.vkAllocateDescriptorSets(
            self.logical_device,
            &descriptor_set_alloc_info,
            &self.descriptor_sets,
        ) != c.VK_SUCCESS) {
            @panic("failed to allocate descriptor sets!");
        }

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            const descriptor_buffer_info: c.VkDescriptorBufferInfo = .{
                .buffer = self.uniform_buffers[i],
                .offset = 0,
                .range = @sizeOf(UniformBufferObject),
            };

            const descriptor_image_info: c.VkDescriptorImageInfo = .{
                .imageLayout = c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .imageView = self.texture_image_view,
                .sampler = self.texture_sampler,
            };

            const descriptor_writes = [_]c.VkWriteDescriptorSet{
                .{
                    .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = self.descriptor_sets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .pBufferInfo = &descriptor_buffer_info,
                },
                .{
                    .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = self.descriptor_sets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .pImageInfo = &descriptor_image_info,
                },
            };
            c.vkUpdateDescriptorSets(
                self.logical_device,
                descriptor_writes.len,
                &descriptor_writes,
                0,
                null,
            );
        }
    }

    fn createCommandBuffers(self: *Context) void {
        var command_buffer_alloc_info: c.VkCommandBufferAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = self.command_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = @intCast(self.command_buffers.len),
        };

        if (c.vkAllocateCommandBuffers(
            self.logical_device,
            &command_buffer_alloc_info,
            &self.command_buffers,
        ) != c.VK_SUCCESS) {
            @panic("failed to allocate command buffers!");
        }
    }

    fn createSyncObjects(self: *Context) void {
        var semaphore_create_info: c.VkSemaphoreCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };

        var fence_create_info: c.VkFenceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            // NOTE: workaround preventing block on first frame
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            if (c.vkCreateSemaphore(
                self.logical_device,
                &semaphore_create_info,
                null,
                &self.image_available_semaphores[i],
            ) != c.VK_SUCCESS or
                c.vkCreateSemaphore(
                    self.logical_device,
                    &semaphore_create_info,
                    null,
                    &self.render_finished_semaphores[i],
                ) != c.VK_SUCCESS or
                c.vkCreateFence(
                    self.logical_device,
                    &fence_create_info,
                    null,
                    &self.in_flight_fences[i],
                ) != c.VK_SUCCESS)
            {
                @panic("failed to create semaphores!");
            }
        }
    }

    fn createGraphicsPipeline(self: *Context) !void {
        const allocator = self.dbga.allocator();

        // VERTEX
        const vert_shader_code = @embedFile("spv/vert.spv");
        const vert_shader_code_aligned = try loadSpirV(allocator, vert_shader_code);
        defer allocator.free(vert_shader_code_aligned);
        self.vert_shader_module = createShaderModule(self.logical_device, vert_shader_code_aligned);

        const vert_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
            .module = self.vert_shader_module,
            .pName = "main",
        };

        // FRAGMENT
        const frag_shader_code = @embedFile("spv/frag.spv");
        const frag_shader_code_aligned = try loadSpirV(allocator, frag_shader_code);
        defer allocator.free(frag_shader_code_aligned);
        self.frag_shader_module = createShaderModule(self.logical_device, frag_shader_code_aligned);

        const frag_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = self.frag_shader_module,
            .pName = "main",
        };

        const shader_stages = try allocator.alloc(c.VkPipelineShaderStageCreateInfo, 2);
        defer allocator.free(shader_stages);
        shader_stages[0] = vert_shader_stage_info;
        shader_stages[1] = frag_shader_stage_info;

        // FIXED FUNCTIONS
        var dynamic_state_create_info: c.VkPipelineDynamicStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = @intCast(DYNAMIC_STATES.len),
            .pDynamicStates = DYNAMIC_STATES[0..],
        };

        const binding_description = Vertex.getBindingDescription();
        const attribute_descriptions = Vertex.getAttributeDescriptions();
        var vertex_input_info: c.VkPipelineVertexInputStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_description, // Optional
            .vertexAttributeDescriptionCount = @intCast(attribute_descriptions.len),
            .pVertexAttributeDescriptions = &attribute_descriptions, // Optional
        };

        var input_assembly: c.VkPipelineInputAssemblyStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
        };

        var viewport: c.VkViewport = .{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(self.swap_chain_extent.width),
            .height = @floatFromInt(self.swap_chain_extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };

        var scissor: c.VkRect2D = .{
            .offset = .{
                .x = 0,
                .y = 0,
            },
            .extent = self.swap_chain_extent,
        };

        var viewport_state_create_info: c.VkPipelineViewportStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };
        if (ENABLE_DYNAMIC_STATE == false) {
            viewport_state_create_info.pViewports = &viewport;
            viewport_state_create_info.pScissors = &scissor;
        }

        var rasterizer_create_info: c.VkPipelineRasterizationStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            // .cullMode = c.VK_CULL_MODE_BACK_BIT,
            // .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
            // NOTE: this and the y flip in ubo belong together
            .cullMode = c.VK_CULL_MODE_BACK_BIT,
            .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
            .depthBiasConstantFactor = 0.0, // Optional
            .depthBiasClamp = 0.0, // Optional
            .depthBiasSlopeFactor = 0.0, // Optional
        };

        var multisampling_create_info: c.VkPipelineMultisampleStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
            .minSampleShading = 1.0, // Optional
            .pSampleMask = null, // Optional
            .alphaToCoverageEnable = c.VK_FALSE, // Optional
            .alphaToOneEnable = c.VK_FALSE, // Optional
        };

        var color_blend_attachment: c.VkPipelineColorBlendAttachmentState = .{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_FALSE,
            .srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE, // Optional
            .dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO, // Optional
            .colorBlendOp = c.VK_BLEND_OP_ADD, // Optional
            .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE, // Optional
            .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO, // Optional
            .alphaBlendOp = c.VK_BLEND_OP_ADD, // Optional
            // NOTE: this are settings for alpha channel blending
            // .blendEnable = c.VK_TRUE;
            // .srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA;
            // .dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            // .colorBlendOp = c.VK_BLEND_OP_ADD;
            // .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE;
            // .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO;
            // .alphaBlendOp = c.VK_BLEND_OP_ADD;
        };

        var color_blending: c.VkPipelineColorBlendStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = c.VK_FALSE,
            .logicOp = c.VK_LOGIC_OP_COPY, // Optional
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };
        color_blending.blendConstants[0] = 0.0; // Optional
        color_blending.blendConstants[1] = 0.0; // Optional
        color_blending.blendConstants[2] = 0.0; // Optional
        color_blending.blendConstants[3] = 0.0; // Optional

        var depth_stencil: c.VkPipelineDepthStencilStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = c.VK_TRUE,
            .depthWriteEnable = c.VK_TRUE,
            .depthCompareOp = c.VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = c.VK_FALSE,
            .minDepthBounds = 0, // Optional
            .maxDepthBounds = 1, // Optional
            .stencilTestEnable = c.VK_FALSE,
            .front = .{}, // Optional
            .back = .{}, // Optional
        };

        var pipeline_layout_create_info: c.VkPipelineLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &self.descriptor_set_layout,
            .pushConstantRangeCount = 0, // Optional
            .pPushConstantRanges = null, // Optional
        };

        if (c.vkCreatePipelineLayout(
            self.logical_device,
            &pipeline_layout_create_info,
            null,
            &self.pipeline_layout,
        ) != c.VK_SUCCESS) {
            @panic("failed to create pipeline layout!");
        }

        var pipeline_create_info: c.VkGraphicsPipelineCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            // TODO: would this work with static [_]shader_stages => shader_stages[0..].ptr ???
            // then we could use it on the other "slices", too
            .pStages = shader_stages.ptr,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state_create_info,
            .pRasterizationState = &rasterizer_create_info,
            .pMultisampleState = &multisampling_create_info,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_state_create_info,
            .layout = self.pipeline_layout,
            .renderPass = self.render_pass,
            .subpass = 0,
            .basePipelineHandle = null, // Optional
            .basePipelineIndex = -1, // Optional
        };

        if (c.vkCreateGraphicsPipelines(
            self.logical_device,
            null,
            1,
            &pipeline_create_info,
            null,
            &self.graphics_pipeline,
        ) != c.VK_SUCCESS) {
            @panic("failed to create graphics pipeline!");
        }
    }

    fn createFramebuffers(self: *Context) void {
        const allocator = self.dbga.allocator();

        self.swap_chain_framebuffers.resize(allocator, self.swap_chain_image_views.items.len) catch unreachable;
        for (0..self.swap_chain_image_views.items.len) |i| {
            var attachments = [_]c.VkImageView{
                self.swap_chain_image_views.items[i],
                self.depth_image_view,
            };

            var framebuffer_create_info: c.VkFramebufferCreateInfo = .{
                .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = self.render_pass,
                .attachmentCount = attachments.len,
                .pAttachments = &attachments,
                .width = self.swap_chain_extent.width,
                .height = self.swap_chain_extent.height,
                .layers = 1,
            };

            if (c.vkCreateFramebuffer(
                self.logical_device,
                &framebuffer_create_info,
                null,
                &self.swap_chain_framebuffers.items[i],
            ) != c.VK_SUCCESS) {
                @panic("failed to create framebuffer!");
            }
        }
    }

    fn createCommandPool(self: *Context) void {
        var command_pool_create_info: c.VkCommandPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.queue_familiy_indices.graphics_family.?,
        };
        if (c.vkCreateCommandPool(
            self.logical_device,
            &command_pool_create_info,
            null,
            &self.command_pool,
        ) != c.VK_SUCCESS) {
            @panic("failed to create command pool!");
        }
    }

    fn createDepthResources(self: *Context) void {
        const depth_format: c.VkFormat = self.findDepthFormat();

        self.createImage(
            self.swap_chain_extent.width,
            self.swap_chain_extent.height,
            depth_format,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &self.depth_image,
            &self.depth_image_memory,
        );

        self.transitionImageLayout(
            self.depth_image,
            depth_format,
            c.VK_IMAGE_LAYOUT_UNDEFINED,
            c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        self.depth_image_view = self.createImageView(
            self.depth_image,
            depth_format,
            c.VK_IMAGE_ASPECT_DEPTH_BIT,
        );
    }

    fn copyBuffer(self: *Context, src_buffer: c.VkBuffer, dst_buffer: c.VkBuffer, size: c.VkDeviceSize) void {
        const command_buffer: c.VkCommandBuffer = self.beginSingleTimeCommands();
        defer self.endSingleTimeCommands(command_buffer);

        var copy_region: c.VkBufferCopy = .{ .size = size };
        c.vkCmdCopyBuffer(
            command_buffer,
            src_buffer,
            dst_buffer,
            1,
            &copy_region,
        );
    }

    fn beginSingleTimeCommands(self: *Context) c.VkCommandBuffer {
        const alloc_info: c.VkCommandBufferAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandPool = self.command_pool,
            .commandBufferCount = 1,
        };

        var command_buffer: c.VkCommandBuffer = null;
        if (c.vkAllocateCommandBuffers(
            self.logical_device,
            &alloc_info,
            &command_buffer,
        ) != c.VK_SUCCESS) {
            @panic("Failed trying to allocate command buffer!");
        }

        const begin_info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        if (c.vkBeginCommandBuffer(command_buffer, &begin_info) != c.VK_SUCCESS) {
            @panic("Failed trying to begin command buffer context");
        }

        return command_buffer;
    }

    fn endSingleTimeCommands(self: *Context, command_buffer: c.VkCommandBuffer) void {
        if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
            @panic("Failed trying to end command buffer!");
        }

        const submit_info: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
        };
        if (c.vkQueueSubmit(
            self.graphics_queue,
            1,
            &submit_info,
            null,
        ) != c.VK_SUCCESS) {
            @panic("Failed sumbitting to graphics queue!");
        }

        if (c.vkQueueWaitIdle(self.graphics_queue) != c.VK_SUCCESS) {
            @panic("Failed while waiting on idle queue.");
        }

        c.vkFreeCommandBuffers(
            self.logical_device,
            self.command_pool,
            1,
            &command_buffer,
        );
    }

    fn createUniformBuffers(self: *Context) void {
        const uniform_buffer_size: c.VkDeviceSize = @sizeOf(UniformBufferObject);

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            self.createBuffer(
                uniform_buffer_size,
                c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &self.uniform_buffers[i],
                &self.uniform_buffers_memory[i],
            );
            if (c.vkMapMemory(
                self.logical_device,
                self.uniform_buffers_memory[i],
                0,
                uniform_buffer_size,
                0,
                &self.uniform_buffers_mapped[i],
            ) != c.VK_SUCCESS) {
                @panic("Failed mapping memory for uniform buffers!");
            }
        }
    }

    fn createIndexBuffer(self: *Context) void {
        const buffer_size: c.VkDeviceSize = @sizeOf(@TypeOf(self.indices[0])) * self.indices.len;

        var staging_buffer: c.VkBuffer = null;
        defer c.vkDestroyBuffer(self.logical_device, staging_buffer, null);
        var staging_buffer_memory: c.VkDeviceMemory = null;
        defer c.vkFreeMemory(self.logical_device, staging_buffer_memory, null);

        self.createBuffer(
            buffer_size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer,
            &staging_buffer_memory,
        );

        var data: ?*anyopaque = null;
        if (c.vkMapMemory(
            self.logical_device,
            staging_buffer_memory,
            0,
            buffer_size,
            0,
            &data,
        ) != c.VK_SUCCESS) {
            @panic("Failed to map staging memory!");
        }
        defer c.vkUnmapMemory(self.logical_device, staging_buffer_memory);

        // Cast the void pointer to a byte slice destination
        const dst: [*]u8 = @ptrCast(data.?);
        // Get the vertices as a byte slice source
        const src: [*]const u8 = @ptrCast(self.indices.ptr);
        @memcpy(dst[0..buffer_size], src[0..buffer_size]);

        self.createBuffer(
            buffer_size,
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &self.index_buffer,
            &self.index_buffer_memory,
        );

        self.copyBuffer(staging_buffer.?, self.index_buffer, buffer_size);
    }

    fn createVertexBuffer(self: *Context) void {
        const buffer_size: c.VkDeviceSize = @sizeOf(@TypeOf(self.vertices[0])) * self.vertices.len;

        var staging_buffer: c.VkBuffer = null;
        defer c.vkDestroyBuffer(self.logical_device, staging_buffer, null);
        var staging_buffer_memory: c.VkDeviceMemory = null;
        defer c.vkFreeMemory(self.logical_device, staging_buffer_memory, null);
        self.createBuffer(
            buffer_size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer,
            &staging_buffer_memory,
        );

        var data: ?*anyopaque = null;
        if (c.vkMapMemory(
            self.logical_device,
            staging_buffer_memory,
            0,
            buffer_size,
            0,
            &data,
        ) != c.VK_SUCCESS) {
            @panic("Failed to map staging memory!");
        }
        defer c.vkUnmapMemory(self.logical_device, staging_buffer_memory);

        // Cast the void pointer to a byte slice destination
        const dst: [*]u8 = @ptrCast(data.?);
        // Get the vertices as a byte slice source
        const src: [*]const u8 = @ptrCast(self.vertices.ptr);
        @memcpy(dst[0..buffer_size], src[0..buffer_size]);

        self.createBuffer(
            buffer_size,
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &self.vertex_buffer,
            &self.vertex_buffer_memory,
        );
        self.copyBuffer(staging_buffer, self.vertex_buffer, buffer_size);
    }

    fn createBuffer(
        self: *Context,
        size: c.VkDeviceSize,
        usage: c.VkBufferUsageFlags,
        properties: c.VkMemoryPropertyFlags,
        buffer: *c.VkBuffer,
        buffer_memory: *c.VkDeviceMemory,
    ) void {
        const buffer_info: c.VkBufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        if (c.vkCreateBuffer(self.logical_device, &buffer_info, null, buffer) != c.VK_SUCCESS) {
            @panic("failed to create buffer!");
        }

        var mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.logical_device, buffer.*, &mem_requirements);

        var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(self.physical_device, &mem_properties);

        const alloc_info: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, mem_properties, properties),
        };

        if (c.vkAllocateMemory(self.logical_device, &alloc_info, null, buffer_memory) != c.VK_SUCCESS) {
            @panic("failed to allocate buffer memory!");
        }

        _ = c.vkBindBufferMemory(self.logical_device, buffer.*, buffer_memory.*, 0);
    }

    fn createTextureSampler(self: *Context) void {
        var properties: c.VkPhysicalDeviceProperties = .{};
        c.vkGetPhysicalDeviceProperties(self.physical_device, &properties);

        const sampler_info: c.VkSamplerCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = c.VK_FILTER_LINEAR,
            .minFilter = c.VK_FILTER_LINEAR,
            .addressModeU = c.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = c.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = c.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            // NOTE: if we don't want anisotropy
            // .anisotropyEnable = c.VK_FALSE,
            // .maxAnisotropy = 1.0,
            // NOTE: anisotropy has to be enabled in the logical device, too
            .anisotropyEnable = c.VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .borderColor = c.VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = c.VK_FALSE,
            .compareEnable = c.VK_FALSE,
            .compareOp = c.VK_COMPARE_OP_ALWAYS,
            .mipmapMode = c.VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .mipLodBias = 0.0,
            .minLod = 0.0,
            .maxLod = 0.0,
        };

        if (c.vkCreateSampler(
            self.logical_device,
            &sampler_info,
            null,
            &self.texture_sampler,
        ) != c.VK_SUCCESS) {
            @panic("failed to create texture sampler!");
        }
    }

    fn createTextureImageView(self: *Context) void {
        self.texture_image_view = self.createImageView(
            self.texture_image,
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_ASPECT_COLOR_BIT,
        );
    }

    fn createImageViews(self: *Context) void {
        const allocator = self.dbga.allocator();

        self.swap_chain_image_views.resize(allocator, self.swap_chain_images.items.len) catch unreachable;
        for (0..self.swap_chain_images.items.len) |i| {
            self.swap_chain_image_views.items[i] = self.createImageView(
                self.swap_chain_images.items[i],
                self.swap_chain_image_format,
                c.VK_IMAGE_ASPECT_COLOR_BIT,
            );
        }
    }

    fn createImageView(self: *Context, image: c.VkImage, format: c.VkFormat, aspect_flags: c.VkImageAspectFlags) c.VkImageView {
        var view_create_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = .{
                .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = .{
                .aspectMask = aspect_flags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        var image_view: c.VkImageView = null;
        if (c.vkCreateImageView(
            self.logical_device,
            &view_create_info,
            null,
            &image_view,
        ) != c.VK_SUCCESS) {
            @panic("failed to create texture image view!");
        }

        return image_view;
    }

    fn createTextureImage(self: *Context) void {
        var texture_width: i32 = 0;
        var texture_height: i32 = 0;
        var texture_channels: i32 = 0;
        const pixels: ?*c.stbi_uc = c.stbi_load(
            "assets/texture.jpg",
            &texture_width,
            &texture_height,
            &texture_channels,
            c.STBI_rgb_alpha,
        );
        if (pixels == null) @panic("failed to load texture image!");
        const image_size: c.VkDeviceSize = @intCast(texture_width * texture_height * 4);

        var staging_buffer: c.VkBuffer = null;
        defer c.vkDestroyBuffer(self.logical_device, staging_buffer, null);
        var staging_buffer_memory: c.VkDeviceMemory = null;
        defer c.vkFreeMemory(self.logical_device, staging_buffer_memory, null);

        self.createBuffer(
            image_size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer,
            &staging_buffer_memory,
        );

        var data: ?*anyopaque = null;
        if (c.vkMapMemory(
            self.logical_device,
            staging_buffer_memory,
            0,
            image_size,
            0,
            &data,
        ) != c.VK_SUCCESS) {
            @panic("Failed to map staging memory!");
        }
        defer c.vkUnmapMemory(self.logical_device, staging_buffer_memory);

        // Cast the void pointer to a byte slice destination
        const dst: [*]u8 = @ptrCast(data.?);
        // Get the vertices as a byte slice source
        const src: [*]const u8 = @ptrCast(pixels.?);
        @memcpy(dst[0..image_size], src[0..image_size]);

        c.stbi_image_free(pixels);

        self.createImage(
            @intCast(texture_width),
            @intCast(texture_height),
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | c.VK_IMAGE_USAGE_SAMPLED_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &self.texture_image,
            &self.texture_image_memory,
        );

        self.transitionImageLayout(
            self.texture_image,
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_LAYOUT_UNDEFINED,
            c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        );

        self.copyBufferToImage(
            staging_buffer,
            self.texture_image,
            @intCast(texture_width),
            @intCast(texture_height),
        );

        self.transitionImageLayout(
            self.texture_image,
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        );
    }

    fn transitionImageLayout(
        self: *Context,
        image: c.VkImage,
        format: c.VkFormat,
        old_layout: c.VkImageLayout,
        new_layout: c.VkImageLayout,
    ) void {
        const command_buffer = self.beginSingleTimeCommands();

        var barrier: c.VkImageMemoryBarrier = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        var source_stage: c.VkPipelineStageFlags = 0;
        var destination_stage: c.VkPipelineStageFlags = 0;

        if (new_layout == c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = c.VK_IMAGE_ASPECT_DEPTH_BIT;

            if (self.hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= c.VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        } else {
            barrier.subresourceRange.aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT;
        }

        if (old_layout == c.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;

            source_stage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (old_layout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and new_layout == c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT;

            source_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
            destination_stage = c.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (old_layout == c.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            source_stage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = c.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else {
            @panic("unsupported layout transition!");
        }

        c.vkCmdPipelineBarrier(
            command_buffer,
            source_stage,
            destination_stage,
            0,
            0,
            null,
            0,
            null,
            1,
            &barrier,
        );

        self.endSingleTimeCommands(command_buffer);
    }

    fn copyBufferToImage(self: *Context, buffer: c.VkBuffer, image: c.VkImage, width: u32, height: u32) void {
        const command_buffer: c.VkCommandBuffer = self.beginSingleTimeCommands();

        var region: c.VkBufferImageCopy = .{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,

            .imageSubresource = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },

            .imageOffset = .{ .x = 0, .y = 0, .z = 0 },
            .imageExtent = .{
                .width = width,
                .height = height,
                .depth = 1,
            },
        };

        c.vkCmdCopyBufferToImage(
            command_buffer,
            buffer,
            image,
            c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region,
        );

        self.endSingleTimeCommands(command_buffer);
    }

    fn createImage(
        self: *Context,
        width: u32,
        height: u32,
        format: c.VkFormat,
        tiling: c.VkImageTiling,
        usage: c.VkImageUsageFlags,
        properties: c.VkMemoryPropertyFlags,
        image: *c.VkImage,
        image_memory: *c.VkDeviceMemory,
    ) void {
        const image_info: c.VkImageCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = c.VK_IMAGE_TYPE_2D,
            .extent = .{
                .width = @intCast(width),
                .height = @intCast(height),
                .depth = 1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .format = format,
            .tiling = tiling,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .flags = 0, // Optional
        };
        if (c.vkCreateImage(
            self.logical_device,
            &image_info,
            null,
            image,
        ) != c.VK_SUCCESS) {
            @panic("failed to create image!");
        }

        var image_mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetImageMemoryRequirements(self.logical_device, image.*, &image_mem_requirements);

        var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(self.physical_device, &mem_properties);

        const image_alloc_info: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = image_mem_requirements.size,
            .memoryTypeIndex = findMemoryType(image_mem_requirements.memoryTypeBits, mem_properties, properties),
        };

        if (c.vkAllocateMemory(self.logical_device, &image_alloc_info, null, image_memory) != c.VK_SUCCESS) {
            @panic("failed to allocate image memory!");
        }

        if (c.vkBindImageMemory(self.logical_device, image.*, image_memory.*, 0) != c.VK_SUCCESS) {
            @panic("Binding image failed!");
        }
    }

    pub fn recordCommandBuffer(
        self: *Context,
        command_buffer: c.VkCommandBuffer,
        image_index: u32,
        vertex_buffer: c.VkBuffer,
        index_buffer: c.VkBuffer,
        current_frame: usize,
    ) void {

        // begin render pass
        var command_buffer_begin_info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0, // Optional
            .pInheritanceInfo = null, // Optional
        };
        if (c.vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info) != c.VK_SUCCESS) {
            @panic("failed to begin recording command buffer!");
        }
        defer {
            // end command buffer
            if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
                @panic("failed to record command buffer!");
            }
        }

        // clear values
        var clear_values = [_]c.VkClearValue{
            .{ .color = .{ .float32 = [4]f32{ 0, 0, 0, 1 } } },
            .{ .depthStencil = .{ .depth = 1, .stencil = 0 } },
        };

        var render_pass_begin_info: c.VkRenderPassBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = self.render_pass,
            .framebuffer = self.swap_chain_framebuffers.items[image_index],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swap_chain_extent,
            },
            .clearValueCount = clear_values.len,
            .pClearValues = &clear_values,
        };
        c.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, c.VK_SUBPASS_CONTENTS_INLINE);
        {
            // end render pass
            defer c.vkCmdEndRenderPass(command_buffer);

            // draw commands
            c.vkCmdBindPipeline(
                command_buffer,
                c.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.graphics_pipeline,
            );

            const vertex_buffers = [_]c.VkBuffer{vertex_buffer};
            const offsets = [_]c.VkDeviceSize{0};
            c.vkCmdBindVertexBuffers(
                command_buffer,
                0,
                1,
                vertex_buffers[0..].ptr,
                offsets[0..].ptr,
            );
            c.vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, c.VK_INDEX_TYPE_UINT16);

            var viewport: c.VkViewport = .{
                .x = 0.0,
                .y = 0.0,
                .width = @floatFromInt(self.swap_chain_extent.width),
                .height = @floatFromInt(self.swap_chain_extent.height),
                .minDepth = 0.0,
                .maxDepth = 1.0,
            };
            c.vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            var scissor: c.VkRect2D = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swap_chain_extent,
            };
            c.vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            c.vkCmdBindDescriptorSets(
                command_buffer,
                c.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipeline_layout,
                0,
                1,
                &self.descriptor_sets[current_frame],
                0,
                null,
            );
            c.vkCmdDrawIndexed(
                command_buffer,
                @intCast(self.indices.len),
                1,
                0,
                0,
                0,
            );
        }
    }

    fn findDepthFormat(self: *Context) c.VkFormat {
        return self.findSupportedFormat(
            &[_]c.VkFormat{
                c.VK_FORMAT_D32_SFLOAT,
                c.VK_FORMAT_D32_SFLOAT_S8_UINT,
                c.VK_FORMAT_D24_UNORM_S8_UINT,
            },
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        );
    }

    fn findSupportedFormat(self: *Context, candidates: []const c.VkFormat, tiling: c.VkImageTiling, features: c.VkFormatFeatureFlags) c.VkFormat {
        for (candidates) |format| {
            var props: c.VkFormatProperties = .{};
            c.vkGetPhysicalDeviceFormatProperties(self.physical_device, format, &props);

            if (tiling == c.VK_IMAGE_TILING_LINEAR and (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == c.VK_IMAGE_TILING_OPTIMAL and (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        @panic("failed to find supported format!");
    }

    fn hasStencilComponent(self: *Context, format: c.VkFormat) bool {
        _ = self;
        return format == c.VK_FORMAT_D32_SFLOAT_S8_UINT or format == c.VK_FORMAT_D24_UNORM_S8_UINT;
    }
};

pub fn main() !void {
    var ctx: Context = .init;
    ctx.run();
}

pub fn checkValidationLayerSupport(allocator: std.mem.Allocator) !bool {
    var layer_count: u32 = 0;
    if (c.vkEnumerateInstanceLayerProperties(&layer_count, null) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceLayerProperties in check_validation_layer_support failed!");
    }

    const available_layers = try allocator.alloc(c.VkLayerProperties, layer_count);
    defer allocator.free(available_layers);

    if (c.vkEnumerateInstanceLayerProperties(&layer_count, available_layers.ptr) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceLayerProperties in check_validation_layer_support failed!");
    }

    // NOTE: refactor with slices
    for (VALIDATION_LAYERS) |layer_name| {
        var layer_found: bool = false;
        for (available_layers) |layer_properties| {
            var slices_eql: bool = true;
            const layer_name_slice = std.mem.span(layer_name);
            for (0..layer_name_slice.len) |i| {
                if (layer_name[i] != layer_properties.layerName[i]) {
                    slices_eql = false;
                    break;
                }
            }

            if (slices_eql) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) return false;
    }

    return true;
}

// TODO: do I need to do callconc(C) here?
pub fn getRequiredExtensions(allocator: std.mem.Allocator) ![][*c]const u8 {
    // NOTE: since vulkan is platform agnostic we use a glfw helper here.
    var glfw_extension_count: u32 = 0;
    const glfw_extensions = c.glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    if (ENABLE_VALIDATION_LAYERS) {
        const extensions = try allocator.alloc([*c]const u8, glfw_extension_count + 1);
        for (0..glfw_extension_count) |i| extensions[i] = glfw_extensions[i];
        extensions[extensions.len - 1] = c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME.ptr;
        return extensions;
    } else {
        const extensions = try allocator.alloc([*c]const u8, glfw_extension_count);
        for (0..glfw_extension_count) |i| extensions[i] = glfw_extensions[i];
        return extensions;
    }
}

pub fn populateDebugMessengerCreateInfo(create_info: *c.VkDebugUtilsMessengerCreateInfoEXT) void {
    create_info.sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debugCallback;
}

pub fn debugCallback(
    // VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    messageSeverity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    // VkDebugUtilsMessageTypeFlagsEXT messageType,
    messageType: c.VkDebugUtilsMessageTypeFlagsEXT,
    // const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    pCallbackData: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
    // void* pUserData
    pUserData: ?*anyopaque,
    //) {
) callconv(C) u32 {
    _ = messageSeverity;
    _ = messageType;
    _ = pUserData;

    // std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    std.debug.print("validation layer: {s}\n", .{pCallbackData.*.pMessage});

    return c.VK_FALSE;
}

// NOTE: some functions need to be loaded first and then be casted into the right type!
//       this is (as in opengl) a thing for extensions. They need to be loaded at runtime.
pub fn createDebugUtilsMessengerEXT(
    instance: c.VkInstance,
    p_create_info: *const c.VkDebugUtilsMessengerCreateInfoEXT,
    p_allocator: ?*const c.VkAllocationCallbacks,
    p_debug_messenger: *c.VkDebugUtilsMessengerEXT,
) c.VkResult {
    const opt_func = c.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (opt_func) |func| {
        const vkCreateDebugUtilsMessengerEXT_fn: c.PFN_vkCreateDebugUtilsMessengerEXT = @ptrCast(func);
        return vkCreateDebugUtilsMessengerEXT_fn.?(instance, p_create_info, p_allocator, p_debug_messenger);
    } else {
        return c.VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

pub fn destroyDebugUtilsMessengerEXT(
    instance: c.VkInstance,
    debug_messenger: c.VkDebugUtilsMessengerEXT,
    p_allocator: ?*const c.VkAllocationCallbacks,
) void {
    const opt_func = c.vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (opt_func) |func| {
        const vkDestroyDebugUtilsMessengerEXT_fn: c.PFN_vkDestroyDebugUtilsMessengerEXT = @ptrCast(func);
        vkDestroyDebugUtilsMessengerEXT_fn.?(instance, debug_messenger, p_allocator);
    }
}

pub fn isDeviceSuitable(allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !bool {
    const query_indices: QueueFamilyIndices = try findQueueFamilies(allocator, physical_device, surface);
    if (!query_indices.isComplete()) return false;

    var supported_features: c.VkPhysicalDeviceFeatures = .{};
    c.vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
    if (supported_features.samplerAnisotropy == c.VK_FALSE) return false;

    const extensions_supported = try checkDeviceExtensionSupport(allocator, physical_device);
    if (extensions_supported) {
        var swap_chain_adequate: bool = false;
        var details = try querySwapChainSupport(allocator, physical_device, surface);
        defer details.deinit(allocator);
        swap_chain_adequate = details.formats.items.len > 0 and details.present_modes.items.len > 0;

        return swap_chain_adequate;
    } else {
        return false;
    }
}

pub fn findQueueFamilies(allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !QueueFamilyIndices {
    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

    var indices: QueueFamilyIndices = .{};

    // look for queue support
    for (queue_families, 0..) |queue_family, i| {
        if ((queue_family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0) {
            indices.graphics_family = @intCast(i);
            break;
        }
    }

    // look for window support
    for (0..queue_families.len) |i| {
        var present_support: c.VkBool32 = 0;

        if (c.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(i), surface, &present_support) != c.VK_SUCCESS) {
            @panic("failed while calling vkGetPhysicalDeviceSurfaceSupportKHR in findQueueFamilies!");
        }

        if (present_support == c.VK_TRUE) {
            // if (present_support == 1) {
            indices.present_family = @intCast(i);
            break;
        }
    }

    return indices;
}

pub fn checkDeviceExtensionSupport(allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice) !bool {
    var extension_count: u32 = 0;
    if (c.vkEnumerateDeviceExtensionProperties(physical_device, null, &extension_count, null) != c.VK_SUCCESS) {
        @panic("failed while calling vkEnumerateDeviceExtensionProperties in checkDeviceExtensionSupport for extension_count lookup!");
    }

    const available_extensions = try allocator.alloc(c.VkExtensionProperties, extension_count);
    defer allocator.free(available_extensions);
    if (c.vkEnumerateDeviceExtensionProperties(physical_device, null, &extension_count, available_extensions.ptr) != c.VK_SUCCESS) {
        @panic("failed while calling vkEnumerateDeviceExtensionProperties in checkDeviceExtensionSupport for available_extensions lookup!");
    }

    var required_extensions: std.ArrayList([*c]const u8) = .empty;
    defer required_extensions.deinit(allocator);
    for (DEVICE_EXTENSIONS) |ext| try required_extensions.append(allocator, ext);

    for (available_extensions) |ext| {
        var i = required_extensions.items.len;
        while (i > 0) {
            i -= 1;

            const required_span = std.mem.span(required_extensions.items[i]);
            const available_span = std.mem.sliceTo(&ext.extensionName, 0);

            if (std.mem.eql(u8, required_span, available_span)) {
                _ = required_extensions.swapRemove(i);
            }
        }
    }

    return required_extensions.items.len == 0;
}

pub fn querySwapChainSupport(allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !SwapChainSupportDetails {
    var details: SwapChainSupportDetails = .{};
    if (c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &details.capabilities) != c.VK_SUCCESS) {
        @panic("failed while calling vkGetPhysicalDeviceSurfaceCapabilitiesKHR in querySwapChainSupport!");
    }

    var format_count: u32 = 0;
    if (c.vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, null) != c.VK_SUCCESS) {
        @panic("failed while calling vkGetPhysicalDeviceSurfaceFormatsKHR for count in querySwapChainSupport!");
    }
    if (format_count != 0) {
        try details.formats.resize(allocator, format_count);
        if (c.vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, details.formats.items.ptr) != c.VK_SUCCESS) {
            @panic("failed while calling vkGetPhysicalDeviceSurfaceFormatsKHR for details in querySwapChainSupport!");
        }
    }

    var present_mode_count: u32 = 0;
    if (c.vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, null) != c.VK_SUCCESS) {
        @panic("failed while calling vkGetPhysicalDeviceSurfacePresentModesKHR for count in querySwapChainSupport!");
    }
    if (present_mode_count != 0) {
        try details.present_modes.resize(allocator, present_mode_count);
        if (c.vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, details.present_modes.items.ptr) != c.VK_SUCCESS) {
            @panic("failed while calling vkGetPhysicalDeviceSurfacePresentModesKHR for details in querySwapChainSupport!");
        }
    }

    return details;
}

pub fn chooseSwapSurfaceFormat(available_formats: []c.VkSurfaceFormatKHR) c.VkSurfaceFormatKHR {
    for (available_formats) |available_format| {
        if (available_format.format == c.VK_FORMAT_B8G8R8A8_SRGB and
            available_format.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return available_format;
        }
    }

    return available_formats[0];
}

pub fn chooseSwapPresentMode(available_present_modes: []c.VkPresentModeKHR) c.VkPresentModeKHR {
    for (available_present_modes) |available_present_mode| {
        if (available_present_mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
            return available_present_mode;
        }
    }

    return c.VK_PRESENT_MODE_FIFO_KHR;
}

pub fn chooseSwapExtent(window: *c.GLFWwindow, capabilities: c.VkSurfaceCapabilitiesKHR) c.VkExtent2D {
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        var width: i32 = 0;
        var height: i32 = 0;
        c.glfwGetFramebufferSize(window, &width, &height);

        var actual_extent: c.VkExtent2D = .{
            .width = @intCast(width),
            .height = @intCast(height),
        };

        actual_extent.width = clamp(actual_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actual_extent.height = clamp(actual_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actual_extent;
    }
}

pub fn clamp(val: u32, min: u32, max: u32) u32 {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

pub fn createShaderModule(logical_device: c.VkDevice, spirv_code: []const u32) c.VkShaderModule {
    var create_info: c.VkShaderModuleCreateInfo = .{};
    create_info.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    // times 4 because []const u32 is 4 byte aligned []const u8
    create_info.codeSize = spirv_code.len * 4;
    create_info.pCode = spirv_code.ptr;

    var shader_module: c.VkShaderModule = null;
    if (c.vkCreateShaderModule(logical_device, &create_info, null, &shader_module) != c.VK_SUCCESS) {
        @panic("failed to create shader module!");
    }

    return shader_module;
}

pub fn loadSpirV(allocator: std.mem.Allocator, code: []const u8) ![]align(4) const u32 {
    std.debug.assert(code.len % @sizeOf(u32) == 0);

    // Allocate a 4-byte aligned copy
    const aligned = try allocator.alignedAlloc(u8, .@"4", code.len);
    @memcpy(aligned, code);

    // Reinterpret as u32 slice
    return std.mem.bytesAsSlice(u32, aligned);
}

// TODO: is this correct?
fn findMemoryType(
    type_filter: u32,
    mem_properties: c.VkPhysicalDeviceMemoryProperties,
    properties: c.VkMemoryPropertyFlags,
) u32 {
    for (0..mem_properties.memoryTypeCount) |i| {
        const idx: u32 = @intCast(i);
        const shift: u5 = @intCast(idx);

        if ((type_filter & (@as(u32, 1) << shift)) != 0 and
            (mem_properties.memoryTypes[idx].propertyFlags & properties) == properties)
        {
            return idx;
        }
    }

    @panic("failed to find suitable memory type!");
}
