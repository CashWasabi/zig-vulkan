const builtin = @import("builtin");
// const build_options = @import("build_options");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("signal.h");
    @cInclude("GLFW/glfw3.h");
    // @cInclude("vulkan/vulkan.h");
});
const C = std.builtin.CallingConvention.c;
const std = @import("std");

var G_SHOULD_EXIT: bool = false;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// enable validation layers for debug mode
const ENABLE_VALIDATION_LAYERS: bool = builtin.mode == .Debug;

const VALIDATION_LAYERS = [_][:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const DEVICE_EXTENSIONS = [_][:0]const u8{
    c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

const ENABLE_DYNAMIC_STATE: bool = true;
const DYNAMIC_STATES = [_]c_uint{
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
const Vec2 = packed struct {
    x: f32,
    y: f32,
};
const Vec3 = packed struct {
    x: f32,
    y: f32,
    z: f32,
};
const Vec4 = packed struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
};
const Vertex = packed struct {
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
const VertexIndex = u16; // can also be u32!

fn handleSigint(sig: c_int) callconv(C) void {
    _ = sig;
    G_SHOULD_EXIT = true;
}

pub fn main() !void {
    if (c.signal(c.SIGINT, handleSigint) == c.SIG_ERR) {
        @panic("Failed to register SIGINT handler");
    }

    var dbga: std.heap.DebugAllocator(.{}) = .init;
    defer if (dbga.deinit() == .leak) @panic("Debug allocator has leaked meory!");
    const allocator = dbga.allocator();

    // We'll need those slices later!
    const validation_layers_slice = try allocator.alloc([*c]const u8, VALIDATION_LAYERS.len);
    defer allocator.free(validation_layers_slice);
    for (0..VALIDATION_LAYERS.len) |i| validation_layers_slice[i] = VALIDATION_LAYERS[i].ptr;

    const device_extensions_slice = try allocator.alloc([*c]const u8, DEVICE_EXTENSIONS.len);
    defer allocator.free(device_extensions_slice);
    for (0..DEVICE_EXTENSIONS.len) |i| device_extensions_slice[i] = DEVICE_EXTENSIONS[i].ptr;

    const dynamic_states_slice = try allocator.alloc(c_uint, DYNAMIC_STATES.len);
    defer allocator.free(dynamic_states_slice);
    for (0..DYNAMIC_STATES.len) |i| dynamic_states_slice[i] = DYNAMIC_STATES[i];

    if (c.glfwInit() == c.GLFW_FALSE) @panic("glwfInit failed!");
    defer c.glfwTerminate();

    // ==============
    // CREATE WINDOW
    // ==============
    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);
    const window: ?*c.GLFWwindow = c.glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", null, null);
    defer c.glfwDestroyWindow(window);

    // ========================================
    // CHECK IF WE HAVE ANY EXTENSION FEATURES
    // ========================================
    var extension_count: u32 = 0;
    if (c.vkEnumerateInstanceExtensionProperties(null, &extension_count, null) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceExtensionProperties failed!");
    }

    // =======================
    // CREATE VULKAN INSTANCE
    // =======================
    if (ENABLE_VALIDATION_LAYERS and try checkValidationLayerSupport(allocator) == false) {
        @panic("Validation Layers not available!");
    }

    var app_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Hello Triangle",
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
        create_info.ppEnabledLayerNames = validation_layers_slice.ptr;

        populateDebugMessengerCreateInfo(&debug_create_info);
        create_info.pNext = &debug_create_info;
    } else {
        create_info.enabledLayerCount = 0;
        create_info.pNext = null;
    }

    // ======================
    // SETUP VULKAN INSTANCE
    // ======================
    var instance: c.VkInstance = undefined;
    // NOTE: the second parameter here is a custom memory allocator callback!
    if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS) {
        @panic("vkCreateInstance failed!");
    }
    defer c.vkDestroyInstance(instance, null);

    // ======================
    // SETUP DEBUG MESSENGER
    // ======================
    var debug_messenger: c.VkDebugUtilsMessengerEXT = undefined;
    var debug_messenger_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
    if (ENABLE_VALIDATION_LAYERS) {
        populateDebugMessengerCreateInfo(&debug_messenger_create_info);

        if (createDebugUtilsMessengerEXT(instance, &debug_messenger_create_info, null, &debug_messenger) != c.VK_SUCCESS) {
            @panic("failed to set up debug messenger!");
        }
    }
    defer destroyDebugUtilsMessengerEXT(instance, debug_messenger, null);

    // =================
    // CREATE A SURFACE
    // =================
    var surface: c.VkSurfaceKHR = null;
    if (c.glfwCreateWindowSurface(instance, window, null, &surface) != c.VK_SUCCESS) {
        @panic("failed to create window surface!");
    }
    defer c.vkDestroySurfaceKHR(instance, surface, null);

    // =====================
    // PICK PHYSICAL DEVICE
    // =====================
    var physical_device: c.VkPhysicalDevice = null;
    var device_count: u32 = 0;
    if (c.vkEnumeratePhysicalDevices(instance, &device_count, null) != c.VK_SUCCESS) {
        @panic("failed calling vkEnumeratePhysicalDevices while looking for devices count!");
    }
    if (device_count == 0) {
        @panic("failed to find GPUs with Vulkan support!");
    }
    const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    if (c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr) != c.VK_SUCCESS) {
        @panic("failed calling vkEnumeratePhysicalDevices while getting devices!");
    }
    for (devices) |device| {
        if (try isDeviceSuitable(allocator, device, surface)) {
            physical_device = device;
            break;
        }
    }
    if (physical_device == null) {
        @panic("failed to find a suitable GPU!");
    }

    const indices = try findQueueFamilies(allocator, physical_device, surface);

    // ===============================================================
    // PREPARE DATA TO BE REGISTERED FOR OUR QUEUES ON LOGICAL DEVICE
    // ===============================================================

    // GRAPHICS QUEUE
    const graphics_queue_priority: f32 = 1.0;
    const graphics_queue_create_info: c.VkDeviceQueueCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = indices.graphics_family.?,
        .queueCount = 1,
        .pQueuePriorities = &graphics_queue_priority,
    };

    // PRESENT QUEUE
    const present_queue_priority: f32 = 1.0;
    const present_queue_create_info: c.VkDeviceQueueCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = indices.present_family.?,
        .queueCount = 1,
        .pQueuePriorities = &present_queue_priority,
    };

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
    var logical_device: c.VkDevice = null;

    const device_features: c.VkPhysicalDeviceFeatures = .{};
    var logical_device_create_info: c.VkDeviceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pEnabledFeatures = &device_features,
        // register our queues BEFORE we create our logical device
        .queueCreateInfoCount = @intCast(queue_slice.len), // graphics queue and present queue
        .pQueueCreateInfos = queue_slice.ptr,
        // enable extensions
        .enabledExtensionCount = @intCast(device_extensions_slice.len),
        .ppEnabledExtensionNames = device_extensions_slice.ptr,
    };

    // enable validation layers (optional)
    if (ENABLE_VALIDATION_LAYERS) {
        logical_device_create_info.enabledLayerCount = @intCast(VALIDATION_LAYERS.len);
        logical_device_create_info.ppEnabledLayerNames = validation_layers_slice.ptr;
    } else {
        logical_device_create_info.enabledLayerCount = 0;
    }

    if (c.vkCreateDevice(physical_device, &logical_device_create_info, null, &logical_device) != c.VK_SUCCESS) {
        @panic("failed to create logical device!");
    }
    defer c.vkDestroyDevice(logical_device, null);

    // =========================================================
    // GET QUEUES HANDLES AFTER LOGICAL DEVICE HAS BEEN CREATED
    // =========================================================
    var graphics_queue: c.VkQueue = null;
    c.vkGetDeviceQueue(logical_device, indices.graphics_family.?, 0, &graphics_queue);

    var present_queue: c.VkQueue = null;
    c.vkGetDeviceQueue(logical_device, indices.present_family.?, 0, &present_queue);

    // ==================
    // CREATE SWAP CHAIN
    // ==================
    var swap_chain_support: SwapChainSupportDetails = try querySwapChainSupport(allocator, physical_device, surface);
    defer swap_chain_support.deinit(allocator);

    const surface_format: c.VkSurfaceFormatKHR = chooseSwapSurfaceFormat(swap_chain_support.formats.items);
    const present_mode: c.VkPresentModeKHR = chooseSwapPresentMode(swap_chain_support.present_modes.items);
    const extent: c.VkExtent2D = chooseSwapExtent(window.?, swap_chain_support.capabilities);

    var image_count: u32 = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 and image_count > swap_chain_support.capabilities.maxImageCount) {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }
    var swap_chain_create_info: c.VkSwapchainCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    };

    // TODO: is that really necessary? :(
    const queue_family_indices_slice = try allocator.alloc(u32, 2);
    defer allocator.free(queue_family_indices_slice);
    queue_family_indices_slice[0] = indices.graphics_family.?;
    queue_family_indices_slice[1] = indices.present_family.?;

    if (indices.graphics_family.? != indices.present_family.?) {
        swap_chain_create_info.imageSharingMode = c.VK_SHARING_MODE_CONCURRENT;
        swap_chain_create_info.queueFamilyIndexCount = 2;
        swap_chain_create_info.pQueueFamilyIndices = queue_family_indices_slice.ptr;
    } else {
        swap_chain_create_info.imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
        swap_chain_create_info.queueFamilyIndexCount = 0; // Optional
        swap_chain_create_info.pQueueFamilyIndices = null; // Optional
    }
    swap_chain_create_info.preTransform = swap_chain_support.capabilities.currentTransform;
    swap_chain_create_info.compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap_chain_create_info.presentMode = present_mode;
    swap_chain_create_info.clipped = c.VK_TRUE;
    swap_chain_create_info.oldSwapchain = null;

    var swap_chain: c.VkSwapchainKHR = null;
    if (c.vkCreateSwapchainKHR(logical_device, &swap_chain_create_info, null, &swap_chain) != c.VK_SUCCESS) {
        @panic("failed to create swap chain!");
    }
    defer c.vkDestroySwapchainKHR(logical_device, swap_chain, null);

    // ========================
    // SWAP CHAIN IMAGE ACCESS
    // ========================
    var swap_chain_image_count: u32 = 0;
    var swap_chain_images: std.ArrayList(c.VkImage) = .empty;
    defer swap_chain_images.deinit(allocator);
    if (c.vkGetSwapchainImagesKHR(logical_device, swap_chain, &swap_chain_image_count, null) != c.VK_SUCCESS) {
        @panic("failed getting swap chain image count from vkGetSwapchainImagesKHR function call");
    }
    try swap_chain_images.resize(allocator, swap_chain_image_count);
    if (c.vkGetSwapchainImagesKHR(logical_device, swap_chain, &swap_chain_image_count, swap_chain_images.items.ptr) != c.VK_SUCCESS) {
        @panic("failed getting swap chain images from vkGetSwapchainImagesKHR function call");
    }

    const swap_chain_image_format = surface_format.format;
    const swap_chain_extent = extent;

    // ========================
    // SWAP CHAIN IMAGE ACCESS
    // ========================
    var swap_chain_image_views: std.ArrayList(c.VkImageView) = .empty;
    defer swap_chain_image_views.deinit(allocator);
    try swap_chain_image_views.resize(allocator, swap_chain_images.items.len);
    for (0..swap_chain_images.items.len) |i| {
        var swap_chain_image_view_create_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = swap_chain_images.items[i],
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = swap_chain_image_format,
            .components = .{
                .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        if (c.vkCreateImageView(logical_device, &swap_chain_image_view_create_info, null, &swap_chain_image_views.items[i]) != c.VK_SUCCESS) {
            @panic("failed to create image views!");
        }
    }
    defer {
        for (swap_chain_image_views.items) |swap_chain_image_view| {
            c.vkDestroyImageView(logical_device, swap_chain_image_view, null);
        }
    }
    // ===================
    // CREATE RENDER PASS
    // ===================
    var color_attachment: c.VkAttachmentDescription = .{
        .format = swap_chain_image_format,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    var color_attachment_ref: c.VkAttachmentReference = .{
        .attachment = 0,
        .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    var subpass: c.VkSubpassDescription = .{
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
    };

    var render_pass: c.VkRenderPass = null;
    defer c.vkDestroyRenderPass(logical_device, render_pass, null);

    var render_pass_create_info: c.VkRenderPassCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };
    if (c.vkCreateRenderPass(logical_device, &render_pass_create_info, null, &render_pass) != c.VK_SUCCESS) {
        @panic("failed to create render pass!");
    }

    // =========================
    // CREATE GRAPHICS PIPELINE
    // =========================

    // VERTEX
    const vert_shader_code = @embedFile("spv/vert.spv");
    const vert_shader_code_aligned = try loadSpirV(allocator, vert_shader_code);
    defer allocator.free(vert_shader_code_aligned);
    const vert_shader_module = createShaderModule(logical_device, vert_shader_code_aligned);
    defer c.vkDestroyShaderModule(logical_device, vert_shader_module, null);

    var vert_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{};
    vert_shader_stage_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_shader_stage_info.stage = c.VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_info.module = vert_shader_module;
    vert_shader_stage_info.pName = "main";

    // FRAGMENT
    const frag_shader_code = @embedFile("spv/frag.spv");
    const frag_shader_code_aligned = try loadSpirV(allocator, frag_shader_code);
    defer allocator.free(frag_shader_code_aligned);
    const frag_shader_module = createShaderModule(logical_device, frag_shader_code_aligned);
    defer c.vkDestroyShaderModule(logical_device, frag_shader_module, null);

    var frag_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{};
    frag_shader_stage_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_shader_stage_info.stage = c.VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_info.module = frag_shader_module;
    frag_shader_stage_info.pName = "main";

    const shader_stages = try allocator.alloc(c.VkPipelineShaderStageCreateInfo, 2);
    defer allocator.free(shader_stages);
    shader_stages[0] = vert_shader_stage_info;
    shader_stages[1] = frag_shader_stage_info;

    // FIXED FUNCTIONS
    var dynamic_state_create_info: c.VkPipelineDynamicStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = @intCast(DYNAMIC_STATES.len),
        .pDynamicStates = dynamic_states_slice.ptr,
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
        .width = @floatFromInt(swap_chain_extent.width),
        .height = @floatFromInt(swap_chain_extent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };

    var scissor: c.VkRect2D = .{
        .offset = .{
            .x = 0,
            .y = 0,
        },
        .extent = swap_chain_extent,
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
        .cullMode = c.VK_CULL_MODE_BACK_BIT,
        .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
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
    };
    // NOTE: this are settings for alpha channel blending
    // color_blend_attachment.blendEnable = c.VK_TRUE;
    // color_blend_attachment.srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA;
    // color_blend_attachment.dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    // color_blend_attachment.colorBlendOp = c.VK_BLEND_OP_ADD;
    // color_blend_attachment.srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE;
    // color_blend_attachment.dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO;
    // color_blend_attachment.alphaBlendOp = c.VK_BLEND_OP_ADD;

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

    var pipeline_layout: c.VkPipelineLayout = null;
    defer c.vkDestroyPipelineLayout(logical_device, pipeline_layout, null);
    var pipeline_layout_create_info: c.VkPipelineLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0, // Optional
        .pSetLayouts = null, // Optional
        .pushConstantRangeCount = 0, // Optional
        .pPushConstantRanges = null, // Optional
    };

    if (c.vkCreatePipelineLayout(logical_device, &pipeline_layout_create_info, null, &pipeline_layout) != c.VK_SUCCESS) {
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
        .pDepthStencilState = null, // Optional
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state_create_info,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = null, // Optional
        .basePipelineIndex = -1, // Optional
    };

    var graphics_pipeline: c.VkPipeline = null;
    if (c.vkCreateGraphicsPipelines(logical_device, null, 1, &pipeline_create_info, null, &graphics_pipeline) != c.VK_SUCCESS) {
        @panic("failed to create graphics pipeline!");
    }
    defer c.vkDestroyPipeline(logical_device, graphics_pipeline, null);

    // =====================
    // CREATE FRAME BUFFERS
    // =====================
    var swap_chain_framebuffers: std.ArrayList(c.VkFramebuffer) = .empty;
    defer swap_chain_framebuffers.deinit(allocator);
    try swap_chain_framebuffers.resize(allocator, swap_chain_image_views.items.len);
    for (0..swap_chain_image_views.items.len) |i| {
        // var attachments: []c.VkImageView  = {};
        var framebuffer_create_info: c.VkFramebufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            // NOTE: small hack to have a 1 item sized slice that can coerce to a [*c] bu using .ptr
            //       not sure if this actually works or just sends garbage over!
            .pAttachments = swap_chain_image_views.items[i .. i + 1].ptr,
            .width = swap_chain_extent.width,
            .height = swap_chain_extent.height,
            .layers = 1,
        };

        if (c.vkCreateFramebuffer(logical_device, &framebuffer_create_info, null, &swap_chain_framebuffers.items[i]) != c.VK_SUCCESS) {
            @panic("failed to create framebuffer!");
        }
    }
    defer {
        for (swap_chain_framebuffers.items) |swap_chain_framebuffer| {
            c.vkDestroyFramebuffer(logical_device, swap_chain_framebuffer, null);
        }
    }

    // ====================
    // CREATE COMMAND POOL
    // ====================
    var command_pool: c.VkCommandPool = null;

    var command_pool_create_info: c.VkCommandPoolCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = indices.graphics_family.?,
    };
    if (c.vkCreateCommandPool(logical_device, &command_pool_create_info, null, &command_pool) != c.VK_SUCCESS) {
        @panic("failed to create command pool!");
    }
    defer c.vkDestroyCommandPool(logical_device, command_pool, null);

    // =====================
    // CREATE VERTEX BUFFER
    // =====================
    var vertices: [4]Vertex = [_]Vertex{
        .{
            .pos = .{ .x = -0.5, .y = -0.5 },
            .color = .{ .x = 1.0, .y = 0.0, .z = 0.0 },
        },
        .{
            .pos = .{ .x = 0.5, .y = -0.5 },
            .color = .{ .x = 0.0, .y = 1.0, .z = 0.0 },
        },
        .{
            .pos = .{ .x = 0.5, .y = 0.5 },
            .color = .{ .x = 0.0, .y = 0.0, .z = 1.0 },
        },
        .{
            .pos = .{ .x = -0.5, .y = 0.5 },
            .color = .{ .x = 1.0, .y = 1.0, .z = 1.0 },
        },
    };
    var vertex_indices: [6]VertexIndex = [_]VertexIndex{ 0, 1, 2, 2, 3, 0 };

    var vertex_buffer: c.VkBuffer = null;
    defer c.vkDestroyBuffer(logical_device, vertex_buffer, null);
    var vertex_buffer_memory: c.VkDeviceMemory = null;
    defer c.vkFreeMemory(logical_device, vertex_buffer_memory, null);
    createVertexBuffer(
        &command_pool,
        &graphics_queue,
        &vertex_buffer,
        &vertex_buffer_memory,
        physical_device,
        logical_device,
        &vertices,
    );

    var index_buffer: c.VkBuffer = null;
    defer c.vkDestroyBuffer(logical_device, index_buffer, null);
    var index_buffer_memory: c.VkDeviceMemory = null;
    defer c.vkFreeMemory(logical_device, index_buffer_memory, null);
    createIndexBuffer(
        &command_pool,
        &graphics_queue,
        &index_buffer,
        &index_buffer_memory,
        physical_device,
        logical_device,
        &vertex_indices,
    );

    // ======================
    // CREATE COMMAND BUFFER
    // ======================
    var command_buffer: c.VkCommandBuffer = null;

    var command_buffer_alloc_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (c.vkAllocateCommandBuffers(logical_device, &command_buffer_alloc_info, &command_buffer) != c.VK_SUCCESS) {
        @panic("failed to allocate command buffers!");
    }

    // ====================
    // CREATE SYNC OBJECTS
    // ====================
    var image_available_semaphore: c.VkSemaphore = null;
    var render_finished_semaphore: c.VkSemaphore = null;
    var in_flight_fence: c.VkFence = null;

    var semaphore_create_info: c.VkSemaphoreCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    var fence_create_info: c.VkFenceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        // NOTE: workaround preventing block on first frame
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    if (c.vkCreateSemaphore(logical_device, &semaphore_create_info, null, &image_available_semaphore) != c.VK_SUCCESS or
        c.vkCreateSemaphore(logical_device, &semaphore_create_info, null, &render_finished_semaphore) != c.VK_SUCCESS or
        c.vkCreateFence(logical_device, &fence_create_info, null, &in_flight_fence) != c.VK_SUCCESS)
    {
        @panic("failed to create semaphores!");
    }
    defer c.vkDestroySemaphore(logical_device, image_available_semaphore, null);
    defer c.vkDestroySemaphore(logical_device, render_finished_semaphore, null);
    defer c.vkDestroyFence(logical_device, in_flight_fence, null);

    // ==============
    // RUN MAIN LOOP
    // ==============
    // gracefully finish async tasks before exiting
    defer _ = c.vkDeviceWaitIdle(logical_device);
    // NOTE: we use sigint as long as we don't have a real window! Without a window we can't close correctly
    while (!G_SHOULD_EXIT) {
        // while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE or c.glfwGetKey(window, c.GLFW_KEY_ESCAPE) != c.GLFW_PRESS) {
        c.glfwPollEvents();

        // ===========
        // DRAW FRAME
        // ===========
        if (c.vkWaitForFences(logical_device, 1, &in_flight_fence, c.VK_TRUE, c.UINT64_MAX) != c.VK_SUCCESS) {
            @panic("failed while trying to wait for fences in main loop.");
        }
        if (c.vkResetFences(logical_device, 1, &in_flight_fence) != c.VK_SUCCESS) {
            @panic("failed while trying to reset fences in main loop.");
        }

        var image_index: u32 = 0;
        if (c.vkAcquireNextImageKHR(logical_device, swap_chain, c.UINT64_MAX, image_available_semaphore, null, &image_index) != c.VK_SUCCESS) {
            @panic("failed while trying to aquire next image_khr in main loop.");
        }

        if (c.vkResetCommandBuffer(command_buffer, 0) != c.VK_SUCCESS) {
            @panic("failed while trying to reset command buffer in main loop.");
        }
        recordCommandBuffer(
            command_buffer,
            image_index,
            render_pass,
            swap_chain_framebuffers.items,
            swap_chain_extent,
            graphics_pipeline,
            vertex_buffer,
            index_buffer,
            &vertex_indices,
        );

        const wait_semaphores = [_]c.VkSemaphore{image_available_semaphore};
        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]c.VkSemaphore{render_finished_semaphore};

        var submit_info: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = wait_semaphores[0..].ptr,
            .pWaitDstStageMask = wait_stages[0..].ptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = signal_semaphores[0..].ptr,
        };
        if (c.vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fence) != c.VK_SUCCESS) {
            @panic("failed to submit draw command buffer!");
        }

        // subpass dependencies
        var dependency: c.VkSubpassDependency = .{
            .srcSubpass = c.VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        // TODO: I don't understand what we're doing here
        render_pass_create_info.dependencyCount = 1;
        render_pass_create_info.pDependencies = &dependency;

        // presentation
        const swap_chains = [_]c.VkSwapchainKHR{swap_chain};
        var present_info_khr: c.VkPresentInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signal_semaphores[0..].ptr,
            .swapchainCount = 1,
            .pSwapchains = swap_chains[0..].ptr,
            .pImageIndices = &image_index,
            .pResults = null,
        };
        if (c.vkQueuePresentKHR(present_queue, &present_info_khr) != c.VK_SUCCESS) {
            @panic("Oh man! Failed while trying to queue present_khr. This is the fun part with colors and bling!");
        }
    }
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
            for (0..layer_name.len) |i| {
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
    const indices: QueueFamilyIndices = try findQueueFamilies(allocator, physical_device, surface);
    if (!indices.isComplete()) return false;
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

    var required_extensions: std.ArrayList([:0]const u8) = .empty;
    defer required_extensions.deinit(allocator);
    for (DEVICE_EXTENSIONS) |ext| try required_extensions.append(allocator, ext);

    // NOTE: dear god refactor this!
    // we could just use slices and do std.mem.eql
    // that way we can compare the buffer from extensionName with our [:0]const u8
    for (available_extensions) |ext| {
        var i = required_extensions.items.len;
        while (i > 0) {
            i -= 1;
            var match: bool = true;
            for (0..required_extensions.items[i].len) |j| {
                if (required_extensions.items[i][j] != ext.extensionName[j]) {
                    match = false;
                    break;
                }
            }
            if (match) _ = required_extensions.swapRemove(i); // if we have duplicates for some reason
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

pub fn recordCommandBuffer(
    command_buffer: c.VkCommandBuffer,
    image_index: u32,
    render_pass: c.VkRenderPass,
    swap_chain_framebuffers: []c.VkFramebuffer,
    swap_chain_extent: c.VkExtent2D,
    graphics_pipeline: c.VkPipeline,
    vertex_buffer: c.VkBuffer,
    index_buffer: c.VkBuffer,
    vertex_indices: []VertexIndex,
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

    // begin render pass
    const clear_color: c.VkClearValue = .{
        .color = .{
            .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 },
        },
    };
    var render_pass_begin_info: c.VkRenderPassBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass,
        .framebuffer = swap_chain_framebuffers[image_index],
        .renderArea = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swap_chain_extent,
        },
        .clearValueCount = 1,
        .pClearValues = &clear_color,
    };
    c.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, c.VK_SUBPASS_CONTENTS_INLINE);

    // draw commands
    c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    const vertex_buffers: [1]c.VkBuffer = [_]c.VkBuffer{vertex_buffer};
    const offsets: [1]c.VkDeviceSize = [_]c.VkDeviceSize{0};
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
        .width = @floatFromInt(swap_chain_extent.width),
        .height = @floatFromInt(swap_chain_extent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };
    c.vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    var scissor: c.VkRect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = swap_chain_extent,
    };
    c.vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    c.vkCmdDrawIndexed(command_buffer, @intCast(vertex_indices.len), 1, 0, 0, 0);

    // end render pass
    c.vkCmdEndRenderPass(command_buffer);

    // end command buffer
    if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
        @panic("failed to record command buffer!");
    }
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

fn createIndexBuffer(
    command_pool: *c.VkCommandPool,
    graphics_queue: *c.VkQueue,
    index_buffer: *c.VkBuffer,
    index_buffer_memory: *c.VkDeviceMemory,
    physical_device: c.VkPhysicalDevice,
    logical_device: c.VkDevice,
    indices: []u16,
) void {
    const buffer_size: c.VkDeviceSize = @sizeOf(@TypeOf(indices[0])) * indices.len;

    var staging_buffer: c.VkBuffer = null;
    defer c.vkDestroyBuffer(logical_device, staging_buffer, null);
    var staging_buffer_memory: c.VkDeviceMemory = null;
    defer c.vkFreeMemory(logical_device, staging_buffer_memory, null);

    createBuffer(
        buffer_size,
        c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging_buffer,
        &staging_buffer_memory,
        physical_device,
        logical_device,
    );

    var data: ?*anyopaque = null;
    if (c.vkMapMemory(logical_device, staging_buffer_memory, 0, buffer_size, 0, &data) != c.VK_SUCCESS) {
        @panic("Failed to map staging memory!");
    }
    defer c.vkUnmapMemory(logical_device, staging_buffer_memory);

    // Cast the void pointer to a byte slice destination
    const dst: [*]u8 = @ptrCast(data.?);
    // Get the vertices as a byte slice source
    const src: [*]const u8 = @ptrCast(indices.ptr);
    @memcpy(dst[0..buffer_size], src[0..buffer_size]);

    createBuffer(
        buffer_size,
        c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        index_buffer,
        index_buffer_memory,
        physical_device,
        logical_device,
    );

    copyBuffer(
        &staging_buffer,
        index_buffer,
        buffer_size,
        logical_device,
        command_pool,
        graphics_queue,
    );
}

/// copy from CPU (staging) to device (vertex_buffer)
fn createVertexBuffer(
    command_pool: *c.VkCommandPool,
    graphics_queue: *c.VkQueue,
    vertex_buffer: *c.VkBuffer,
    vertex_buffer_memory: *c.VkDeviceMemory,
    physical_device: c.VkPhysicalDevice,
    logical_device: c.VkDevice,
    vertices: []Vertex,
) void {
    const buffer_size: c.VkDeviceSize = @sizeOf(@TypeOf(vertices[0])) * vertices.len;

    var staging_buffer: c.VkBuffer = null;
    defer c.vkDestroyBuffer(logical_device, staging_buffer, null);
    var staging_buffer_memory: c.VkDeviceMemory = null;
    defer c.vkFreeMemory(logical_device, staging_buffer_memory, null);
    createBuffer(
        buffer_size,
        c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging_buffer,
        &staging_buffer_memory,
        physical_device,
        logical_device,
    );

    var data: ?*anyopaque = null;
    if (c.vkMapMemory(logical_device, staging_buffer_memory, 0, buffer_size, 0, &data) != c.VK_SUCCESS) {
        @panic("Failed to map staging memory!");
    }
    defer c.vkUnmapMemory(logical_device, staging_buffer_memory);

    // Cast the void pointer to a byte slice destination
    const dst: [*]u8 = @ptrCast(data.?);
    // Get the vertices as a byte slice source
    const src: [*]const u8 = @ptrCast(vertices.ptr);
    @memcpy(dst[0..buffer_size], src[0..buffer_size]);

    createBuffer(
        buffer_size,
        c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vertex_buffer,
        vertex_buffer_memory,
        physical_device,
        logical_device,
    );
    copyBuffer(
        &staging_buffer,
        vertex_buffer,
        buffer_size,
        logical_device,
        command_pool,
        graphics_queue,
    );
}

fn createBuffer(
    size: c.VkDeviceSize,
    usage: c.VkBufferUsageFlags,
    properties: c.VkMemoryPropertyFlags,
    buffer: *c.VkBuffer,
    buffer_memory: *c.VkDeviceMemory,
    physical_device: c.VkPhysicalDevice,
    logical_device: c.VkDevice,
) void {
    const buffer_info: c.VkBufferCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
    };

    if (c.vkCreateBuffer(logical_device, &buffer_info, null, buffer) != c.VK_SUCCESS) {
        @panic("failed to create buffer!");
    }

    var mem_requirements: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(logical_device, buffer.*, &mem_requirements);

    var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    const alloc_info: c.VkMemoryAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, mem_properties, properties),
    };

    if (c.vkAllocateMemory(logical_device, &alloc_info, null, buffer_memory) != c.VK_SUCCESS) {
        @panic("failed to allocate buffer memory!");
    }

    _ = c.vkBindBufferMemory(logical_device, buffer.*, buffer_memory.*, 0);
}

fn copyBuffer(
    src_buffer: *c.VkBuffer,
    dst_buffer: *c.VkBuffer,
    size: c.VkDeviceSize,
    logical_device: c.VkDevice,
    command_pool: *c.VkCommandPool,
    graphics_queue: *c.VkQueue,
) void {
    const alloc_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = command_pool.*,
        .commandBufferCount = 1,
    };

    var command_buffer: c.VkCommandBuffer = null;
    if (c.vkAllocateCommandBuffers(logical_device, &alloc_info, &command_buffer) != c.VK_SUCCESS) {
        @panic("Couldn't allocate command buffer!");
    }
    defer c.vkFreeCommandBuffers(logical_device, command_pool.*, 1, &command_buffer);

    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (c.vkBeginCommandBuffer(command_buffer, &begin_info) != c.VK_SUCCESS) {
        @panic("Couldn't begin command buffer context!");
    }
    {
        defer if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
            @panic("Couldn't end command buffer context!");
        };

        const copy_region: c.VkBufferCopy = .{
            .srcOffset = 0, // Optional
            .dstOffset = 0, // Optional
            .size = size,
        };
        c.vkCmdCopyBuffer(command_buffer, src_buffer.*, dst_buffer.*, 1, &copy_region);
    }

    const submit_info: c.VkSubmitInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    if (c.vkQueueSubmit(graphics_queue.*, 1, &submit_info, null) != c.VK_SUCCESS) {
        @panic("Failed submitting to graphics queue!");
    }
    if (c.vkQueueWaitIdle(graphics_queue.*) != c.VK_SUCCESS) {
        @panic("Waiting for graphics queue failed!");
    }
}
