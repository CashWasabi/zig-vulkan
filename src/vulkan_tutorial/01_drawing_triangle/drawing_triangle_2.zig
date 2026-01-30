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

const ENABLE_VALIDATION_LAYERS: bool = builtin.mode == .Debug;

const VALIDATION_LAYERS = [_][*c]const u8{
    "VK_LAYER_KHRONOS_validation".ptr,
};

const DEVICE_EXTENSIONS = [_][*c]const u8{
    c.VK_KHR_SWAPCHAIN_EXTENSION_NAME.ptr,
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

    var instance_create_info: c.VkInstanceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = @intCast(extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,
        .enabledLayerCount = @intCast(VALIDATION_LAYERS.len),
    };

    // SETUP VALIDATION LAYERS (OPTIONAL)
    var debug_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
    if (ENABLE_VALIDATION_LAYERS) {
        instance_create_info.ppEnabledLayerNames = VALIDATION_LAYERS[0..].ptr;
        populateDebugMessengerCreateInfo(&debug_create_info);
        instance_create_info.pNext = &debug_create_info;
    } else {
        instance_create_info.enabledLayerCount = 0;
        instance_create_info.pNext = null;
    }

    var instance: c.VkInstance = undefined;
    if (c.vkCreateInstance(&instance_create_info, null, &instance) != c.VK_SUCCESS) {
        @panic("vkCreateInstance failed!");
    }
    defer c.vkDestroyInstance(instance, null);

    // =======================
    // SETUP DEBUG MESSENGER
    // =======================
    var debug_messenger_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
    var debug_messenger: c.VkDebugUtilsMessengerEXT = undefined;
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

    // ==========================
    // FIND QUEUE FAMILY INDICES
    // ==========================
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
    const physical_device_features: c.VkPhysicalDeviceFeatures = .{};
    var logical_device_create_info: c.VkDeviceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pEnabledFeatures = &physical_device_features,

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

    var logical_device: c.VkDevice = null;
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

    // NOTE: is that really necessary? :(
    const queue_family_indices_slice = try allocator.alloc(u32, 2);
    defer allocator.free(queue_family_indices_slice);
    queue_family_indices_slice[0] = indices.graphics_family.?;
    queue_family_indices_slice[1] = indices.present_family.?;

    var swap_chain_create_info: c.VkSwapchainCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
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
    if (indices.graphics_family.? != indices.present_family.?) {
        swap_chain_create_info.imageSharingMode = c.VK_SHARING_MODE_CONCURRENT;
        swap_chain_create_info.queueFamilyIndexCount = 2;
        swap_chain_create_info.pQueueFamilyIndices = queue_family_indices_slice.ptr;
    } else {
        swap_chain_create_info.imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
        swap_chain_create_info.queueFamilyIndexCount = 0; // Optional
        swap_chain_create_info.pQueueFamilyIndices = null; // Optional
    }

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
    // =========================
    // CREATE RENDER PASS
    // =========================
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

    const vert_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shader_module,
        .pName = "main",
    };

    // FRAGMENT
    const frag_shader_code = @embedFile("spv/frag.spv");
    const frag_shader_code_aligned = try loadSpirV(allocator, frag_shader_code);
    defer allocator.free(frag_shader_code_aligned);
    const frag_shader_module = createShaderModule(logical_device, frag_shader_code_aligned);
    defer c.vkDestroyShaderModule(logical_device, frag_shader_module, null);

    const frag_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shader_module,
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
        .pDynamicStates = DYNAMIC_STATES[0..].ptr,
    };

    var vertex_input_info: c.VkPipelineVertexInputStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = null, // Optional
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = null, // Optional
    };

    var input_assembly: c.VkPipelineInputAssemblyStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
    };

    var viewport_state_create_info: c.VkPipelineViewportStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
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
        .offset = .{ .x = 0, .y = 0 },
        .extent = swap_chain_extent,
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

    var pipeline_layout_create_info: c.VkPipelineLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0, // Optional
        .pSetLayouts = null, // Optional
        .pushConstantRangeCount = 0, // Optional
        .pPushConstantRanges = null, // Optional

    };

    var pipeline_layout: c.VkPipelineLayout = null;
    if (c.vkCreatePipelineLayout(logical_device, &pipeline_layout_create_info, null, &pipeline_layout) != c.VK_SUCCESS) {
        @panic("failed to create pipeline layout!");
    }
    defer c.vkDestroyPipelineLayout(logical_device, pipeline_layout, null);

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
        var framebuffer_create_info: c.VkFramebufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            // NOTE: small hack to have a 1 item sized slice that can coerce to a [*c] bu using .ptr
            //       not sure if this actually works or just sends garbage over!
            // var attachments: []c.VkImageView  = {};
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

    // =====================================
    // CREATE COMMAND POOL AND ALLOC BUFFER
    // =====================================
    var command_pool_create_info: c.VkCommandPoolCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = indices.graphics_family.?,
    };

    var command_pool: c.VkCommandPool = null;
    if (c.vkCreateCommandPool(logical_device, &command_pool_create_info, null, &command_pool) != c.VK_SUCCESS) {
        @panic("failed to create command pool!");
    }
    defer c.vkDestroyCommandPool(logical_device, command_pool, null);

    var command_buffer_alloc_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    // NOTE: will be deallocated when command pool is destroyed
    var command_buffer: c.VkCommandBuffer = null;
    if (c.vkAllocateCommandBuffers(logical_device, &command_buffer_alloc_info, &command_buffer) != c.VK_SUCCESS) {
        @panic("failed to allocate command buffers!");
    }

    // ====================
    // CREATE SYNC OBJECTS
    // ====================
    var semaphore_create_info: c.VkSemaphoreCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    var fence_create_info: c.VkFenceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        // NOTE: workaround preventing block on first frame
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    var image_available_semaphore: c.VkSemaphore = null;
    var render_finished_semaphore: c.VkSemaphore = null;
    var in_flight_fence: c.VkFence = null;
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
        recordCommandBuffer(command_buffer, image_index, render_pass, swap_chain_framebuffers.items, swap_chain_extent, graphics_pipeline);

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

        // TODO: I don't understand what we're doing here!
        //       <(~_~)> is this dynamic maigc?
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

    for (VALIDATION_LAYERS) |layer_name| {
        var layer_found: bool = false;
        for (available_layers) |layer_properties| {
            const str_len = std.mem.len(layer_name);
            if (std.mem.eql(u8, layer_name[0..str_len], layer_properties.layerName[0..str_len])) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) return false;
    }

    return true;
}

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
    messageSeverity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    messageType: c.VkDebugUtilsMessageTypeFlagsEXT,
    pCallbackData: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
    pUserData: ?*anyopaque,
) callconv(C) u32 {
    _ = messageSeverity;
    _ = messageType;
    _ = pUserData;
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
            // if ((queue_family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != c.VK_FALSE) {
            indices.graphics_family = @intCast(i);
            break;
        }
    }

    // look for window support
    for (0..queue_families.len) |i| {
        var present_support: c.VkBool32 = c.VK_FALSE;

        if (c.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(i), surface, &present_support) != c.VK_SUCCESS) {
            @panic("failed while calling vkGetPhysicalDeviceSurfaceSupportKHR in findQueueFamilies!");
        }

        if (present_support == c.VK_TRUE) {
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
        // NOTE: dont break if we have duplicates for some reason!
        var i = required_extensions.items.len;
        while (i > 0) {
            i -= 1;
            const str_len = std.mem.len(required_extensions.items[i]);
            if (std.mem.eql(u8, required_extensions.items[i][0..str_len], ext.extensionName[0..str_len])) {
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

pub fn recordCommandBuffer(
    command_buffer: c.VkCommandBuffer,
    image_index: u32,
    render_pass: c.VkRenderPass,
    swap_chain_framebuffers: []c.VkFramebuffer,
    swap_chain_extent: c.VkExtent2D,
    graphics_pipeline: c.VkPipeline,
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
        if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
            @panic("failed to record command buffer!");
        }
    }

    // begin render pass
    var render_pass_begin_info: c.VkRenderPassBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass,
        .framebuffer = swap_chain_framebuffers[image_index],
        .renderArea = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swap_chain_extent,
        },
    };
    const clear_color: c.VkClearValue = .{ .color = .{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } };
    render_pass_begin_info.clearValueCount = 1;
    render_pass_begin_info.pClearValues = &clear_color;
    c.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, c.VK_SUBPASS_CONTENTS_INLINE);
    {
        // end render pass
        defer c.vkCmdEndRenderPass(command_buffer);

        // draw commands
        c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
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

        c.vkCmdDraw(command_buffer, 3, 1, 0, 0);
    }
}
