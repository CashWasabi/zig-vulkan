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

    var app_info: c.VkApplicationInfo = .{};
    app_info.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Hello Triangle";
    app_info.applicationVersion = c.VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = c.VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = c.VK_API_VERSION_1_0;

    var create_info: c.VkInstanceCreateInfo = .{};
    create_info.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    const extensions = try getRequiredExtensions(allocator);
    defer allocator.free(extensions);

    create_info.enabledExtensionCount = @intCast(extensions.len);
    create_info.ppEnabledExtensionNames = extensions.ptr;

    // SETUP VALIDATION LAYERS (OPTIONAL)
    var debug_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
    create_info.enabledLayerCount = @intCast(VALIDATION_LAYERS.len);
    if (ENABLE_VALIDATION_LAYERS) {
        create_info.ppEnabledLayerNames = validation_layers_slice.ptr;

        populateDebugMessengerCreateInfo(&debug_create_info);
        create_info.pNext = &debug_create_info;
    } else {
        create_info.enabledLayerCount = 0;
        create_info.pNext = null;
    }

    var instance: c.VkInstance = undefined;
    // NOTE: the second parameter here is a custom memoru allocator callback!
    if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS) {
        @panic("vkCreateInstance failed!");
    }
    defer c.vkDestroyInstance(instance, null);

    // =======================
    // SETUP DEBUG MESSENGER
    // =======================
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
    var graphics_queue_create_info: c.VkDeviceQueueCreateInfo = .{};
    graphics_queue_create_info.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphics_queue_create_info.queueFamilyIndex = indices.graphics_family.?;
    graphics_queue_create_info.queueCount = 1;
    graphics_queue_create_info.pQueuePriorities = &graphics_queue_priority;

    // PRESENT QUEUE
    const present_queue_priority: f32 = 1.0;
    var present_queue_create_info: c.VkDeviceQueueCreateInfo = .{};
    present_queue_create_info.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    present_queue_create_info.queueFamilyIndex = indices.present_family.?;
    present_queue_create_info.queueCount = 1;
    present_queue_create_info.pQueuePriorities = &present_queue_priority;

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
    var logical_device_create_info: c.VkDeviceCreateInfo = .{};
    logical_device_create_info.sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    logical_device_create_info.pEnabledFeatures = &device_features;

    // register our queues BEFORE we create our logical device
    logical_device_create_info.queueCreateInfoCount = @intCast(queue_slice.len); // graphics queue and present queue
    logical_device_create_info.pQueueCreateInfos = queue_slice.ptr;

    // enable extensions
    logical_device_create_info.enabledExtensionCount = @intCast(device_extensions_slice.len);
    logical_device_create_info.ppEnabledExtensionNames = device_extensions_slice.ptr;

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
    var swap_chain_create_info: c.VkSwapchainCreateInfoKHR = .{};
    swap_chain_create_info.sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swap_chain_create_info.surface = surface;
    swap_chain_create_info.minImageCount = image_count;
    swap_chain_create_info.imageFormat = surface_format.format;
    swap_chain_create_info.imageColorSpace = surface_format.colorSpace;
    swap_chain_create_info.imageExtent = extent;
    swap_chain_create_info.imageArrayLayers = 1;
    swap_chain_create_info.imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // is that really necessary? :(
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
        var swap_chain_image_view_create_info: c.VkImageViewCreateInfo = .{};
        swap_chain_image_view_create_info.sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        swap_chain_image_view_create_info.image = swap_chain_images.items[i];
        swap_chain_image_view_create_info.viewType = c.VK_IMAGE_VIEW_TYPE_2D;
        swap_chain_image_view_create_info.format = swap_chain_image_format;
        swap_chain_image_view_create_info.components.r = c.VK_COMPONENT_SWIZZLE_IDENTITY;
        swap_chain_image_view_create_info.components.g = c.VK_COMPONENT_SWIZZLE_IDENTITY;
        swap_chain_image_view_create_info.components.b = c.VK_COMPONENT_SWIZZLE_IDENTITY;
        swap_chain_image_view_create_info.components.a = c.VK_COMPONENT_SWIZZLE_IDENTITY;
        swap_chain_image_view_create_info.subresourceRange.aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT;
        swap_chain_image_view_create_info.subresourceRange.baseMipLevel = 0;
        swap_chain_image_view_create_info.subresourceRange.levelCount = 1;
        swap_chain_image_view_create_info.subresourceRange.baseArrayLayer = 0;
        swap_chain_image_view_create_info.subresourceRange.layerCount = 1;
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
    var color_attachment: c.VkAttachmentDescription = .{};
    color_attachment.format = swap_chain_image_format;
    color_attachment.samples = c.VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = c.VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    var color_attachment_ref: c.VkAttachmentReference = .{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    var subpass: c.VkSubpassDescription = .{};
    subpass.pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    var render_pass: c.VkRenderPass = null;
    defer c.vkDestroyRenderPass(logical_device, render_pass, null);

    var render_pass_create_info: c.VkRenderPassCreateInfo = .{};
    render_pass_create_info.sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.attachmentCount = 1;
    render_pass_create_info.pAttachments = &color_attachment;
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &subpass;

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
    var dynamic_state_create_info: c.VkPipelineDynamicStateCreateInfo = .{};
    dynamic_state_create_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state_create_info.dynamicStateCount = @intCast(DYNAMIC_STATES.len);
    dynamic_state_create_info.pDynamicStates = dynamic_states_slice.ptr;

    var vertex_input_info: c.VkPipelineVertexInputStateCreateInfo = .{};
    vertex_input_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.pVertexBindingDescriptions = null; // Optional
    vertex_input_info.vertexAttributeDescriptionCount = 0;
    vertex_input_info.pVertexAttributeDescriptions = null; // Optional

    var input_assembly: c.VkPipelineInputAssemblyStateCreateInfo = .{};
    input_assembly.sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = c.VK_FALSE;

    var viewport: c.VkViewport = .{};
    viewport.x = 0.0;
    viewport.y = 0.0;
    viewport.width = @floatFromInt(swap_chain_extent.width);
    viewport.height = @floatFromInt(swap_chain_extent.height);
    viewport.minDepth = 0.0;
    viewport.maxDepth = 1.0;

    var scissor: c.VkRect2D = .{};
    scissor.offset = .{ .x = 0, .y = 0 };
    scissor.extent = swap_chain_extent;

    var viewport_state_create_info: c.VkPipelineViewportStateCreateInfo = .{};
    viewport_state_create_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state_create_info.viewportCount = 1;
    viewport_state_create_info.scissorCount = 1;
    if (ENABLE_DYNAMIC_STATE == false) {
        viewport_state_create_info.pViewports = &viewport;
        viewport_state_create_info.pScissors = &scissor;
    }

    var rasterizer_create_info: c.VkPipelineRasterizationStateCreateInfo = .{};
    rasterizer_create_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer_create_info.depthClampEnable = c.VK_FALSE;
    rasterizer_create_info.rasterizerDiscardEnable = c.VK_FALSE;
    rasterizer_create_info.polygonMode = c.VK_POLYGON_MODE_FILL;
    rasterizer_create_info.lineWidth = 1.0;
    rasterizer_create_info.cullMode = c.VK_CULL_MODE_BACK_BIT;
    rasterizer_create_info.frontFace = c.VK_FRONT_FACE_CLOCKWISE;
    rasterizer_create_info.depthBiasEnable = c.VK_FALSE;
    rasterizer_create_info.depthBiasConstantFactor = 0.0; // Optional
    rasterizer_create_info.depthBiasClamp = 0.0; // Optional
    rasterizer_create_info.depthBiasSlopeFactor = 0.0; // Optional

    var multisampling_create_info: c.VkPipelineMultisampleStateCreateInfo = .{};
    multisampling_create_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling_create_info.sampleShadingEnable = c.VK_FALSE;
    multisampling_create_info.rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT;
    multisampling_create_info.minSampleShading = 1.0; // Optional
    multisampling_create_info.pSampleMask = null; // Optional
    multisampling_create_info.alphaToCoverageEnable = c.VK_FALSE; // Optional
    multisampling_create_info.alphaToOneEnable = c.VK_FALSE; // Optional

    var color_blend_attachment: c.VkPipelineColorBlendAttachmentState = .{};
    color_blend_attachment.colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = c.VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.colorBlendOp = c.VK_BLEND_OP_ADD; // Optional
    color_blend_attachment.srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.alphaBlendOp = c.VK_BLEND_OP_ADD; // Optional
    // NOTE: this are settings for alpha channel blending
    // color_blend_attachment.blendEnable = c.VK_TRUE;
    // color_blend_attachment.srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA;
    // color_blend_attachment.dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    // color_blend_attachment.colorBlendOp = c.VK_BLEND_OP_ADD;
    // color_blend_attachment.srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE;
    // color_blend_attachment.dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO;
    // color_blend_attachment.alphaBlendOp = c.VK_BLEND_OP_ADD;

    var color_blending: c.VkPipelineColorBlendStateCreateInfo = .{};
    color_blending.sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = c.VK_FALSE;
    color_blending.logicOp = c.VK_LOGIC_OP_COPY; // Optional
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0; // Optional
    color_blending.blendConstants[1] = 0.0; // Optional
    color_blending.blendConstants[2] = 0.0; // Optional
    color_blending.blendConstants[3] = 0.0; // Optional

    var pipeline_layout: c.VkPipelineLayout = null;
    defer c.vkDestroyPipelineLayout(logical_device, pipeline_layout, null);
    var pipeline_layout_create_info: c.VkPipelineLayoutCreateInfo = .{};
    pipeline_layout_create_info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 0; // Optional
    pipeline_layout_create_info.pSetLayouts = null; // Optional
    pipeline_layout_create_info.pushConstantRangeCount = 0; // Optional
    pipeline_layout_create_info.pPushConstantRanges = null; // Optional

    if (c.vkCreatePipelineLayout(logical_device, &pipeline_layout_create_info, null, &pipeline_layout) != c.VK_SUCCESS) {
        @panic("failed to create pipeline layout!");
    }

    var pipeline_create_info: c.VkGraphicsPipelineCreateInfo = .{};
    pipeline_create_info.sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_create_info.stageCount = 2;
    // TODO: would this work with static [_]shader_stages => shader_stages[0..].ptr ???
    // then we could use it on the other "slices", too
    pipeline_create_info.pStages = shader_stages.ptr;
    pipeline_create_info.pVertexInputState = &vertex_input_info;
    pipeline_create_info.pInputAssemblyState = &input_assembly;
    pipeline_create_info.pViewportState = &viewport_state_create_info;
    pipeline_create_info.pRasterizationState = &rasterizer_create_info;
    pipeline_create_info.pMultisampleState = &multisampling_create_info;
    pipeline_create_info.pDepthStencilState = null; // Optional
    pipeline_create_info.pColorBlendState = &color_blending;
    pipeline_create_info.pDynamicState = &dynamic_state_create_info;
    pipeline_create_info.layout = pipeline_layout;
    pipeline_create_info.renderPass = render_pass;
    pipeline_create_info.subpass = 0;
    pipeline_create_info.basePipelineHandle = null; // Optional
    pipeline_create_info.basePipelineIndex = -1; // Optional

    var graphics_pipeline: c.VkPipeline = null;
    if (c.vkCreateGraphicsPipelines(logical_device, null, 1, &pipeline_create_info, null, &graphics_pipeline) != c.VK_SUCCESS) {
        @panic("failed to create graphics pipeline!");
    }
    defer c.vkDestroyPipeline(logical_device, graphics_pipeline, null);

    // ==============
    // RUN MAIN LOOP
    // ==============
    // NOTE: we use sigint as long as we don't have a real window! Without a window we can't close correctly
    while (!G_SHOULD_EXIT) {
        // while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE or c.glfwGetKey(window, c.GLFW_KEY_ESCAPE) != c.GLFW_PRESS) {
        c.glfwPollEvents();
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
